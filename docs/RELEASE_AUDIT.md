# Release Audit

Migration of the MediaPipe/SO-101 system out of the `hand-teleop` meta-workspace
into this repository. Run 2026-09-01.

Evidence is reported at three levels and never mixed: **software** (tests and
static checks), **locked inference** (offline evaluation), and **physical robot**
(observed closed-loop behaviour). This audit covers software and physical.

## Software gate

| Check | Result |
| --- | --- |
| Full suite | **409 passed**, 0 failed |
| `packages/webcam_input` | 21 |
| `packages/so101_teleop` | 91 |
| `integrations/pressurevision` | 142 |
| `training` | 33 |
| guards (parse, imports, wrappers, boundaries) | the remainder |
| Fresh clone into `/tmp`, tests run from it | **passed**, 2.4 MB, 128 tracked files, no `local/` |
| Tracked developer paths (`/home/...`) | 0 |
| Tracked IR/FLIR/Lepton/thermal runtime references | 0 (outside the boundary tests' own patterns) |
| Tracked `.mp4/.avi/.zip/.pt/.pth/.ckpt/.onnx` | 0 |
| Largest tracked file | 61.6 KB |
| `git ls-files local` | 0 |
| `git status --short` | empty |
| Remotes | none |
| Linked worktrees | none |

### Test accounting against the pre-migration baseline

The pre-migration workspace collected 347 tests (webcam-input 192, hand-pressure
155) with 3 pre-existing IR collection errors. That total is not comparable
directly, because most of it is IR work that belongs to the private project.

| Group | Source | Here |
| --- | --- | --- |
| SO-101 core (7 files) | 31 | 31 |
| `webcam_input` | 19 | 19 |
| hand-pressure tool tests (5 files) | 88 | 88 |
| PV module tests (4 files) | 49 | 46 |
| IR tests | 26 files | 0 — migrate to `ir-camera-force` |
| New boundary / contract / guard tests | — | +146 |

The 3 missing PV tests exercised `ir_pressure_proposal`, the private shadow-proposal
path; they are named in `integrations/pressurevision/tests/test_pv_pressure.py` and
migrate with that module. **No test that belongs here was dropped.**

## Physical robot gate

Run on the development machine: SO-101 on `usb-1a86_USB_Single_Serial_5B14110850`,
Creative Live! Cam workspace camera, monocular hand tracking, RTX 4060.

### 1. Teleoperation — PASS

`./scripts/run_arm_ee.sh` connected the arm, computed the down-ready pose
(EE centre `[0.218, -0.001, 0.069]`, down rotvec `[0.076, 3.141, 0.000]`, i.e. the
gripper pointing down), initialised MediaPipe, entered the control loop, and
disconnected cleanly.

### 2. Recording — PASS

`./scripts/run_record_ee.sh` recorded one episode of **150 frames at 10 fps**,
encoded video, and kept it. The dataset reloads through the LeRobot API with the
expected contract:

```
episodes 1 | frames 150 | fps 10
action                     float32 (6,)
observation.state          float32 (6,)
observation.images.front   video   (480, 640, 3)  -> decodes to (3,480,640) in [0,1]
```

**The recorded actions are the evidence that control works end to end.** Over the
150 frames:

| joint | range |
| --- | --- |
| shoulder_pan | 14.82 |
| shoulder_lift | 30.52 |
| elbow_flex | 52.97 |
| wrist_flex | 5.46 |
| wrist_roll | 14.75 |
| **gripper** | **50.31** (0.18 → 50.49) |

The arm tracked the operator's hand across all five body joints. The gripper took
63 distinct values, changed by more than 0.5 on 49 of 149 frame transitions, and
reached both the clamped end (0.18) and 50.49 — the release position
`RELEASE_POS = 50.0`. The largest single-frame change was 37.19, consistent with
the asymmetric smoothing's fast close (`GRIP_CLOSE_ALPHA = 0.7`).

This exercises the refactored gripper contract on real hardware. Combined with the
offline check that the refactor is numerically identical to the pre-refactor
smoothing over 220 samples (max deviation 3.6e-15), the MediaPipe grip path is
confirmed migrated without behaviour change.

### 3. Deployment — PASS

`./scripts/run_deploy_ee.sh --policy local/checkpoints/act_kaggle_output/act_pickplace/checkpoints/030000/pretrained_model --duration 30`
loaded the policy on CUDA, ramped to the down-ready pose, and ran autonomously:

```
[deploy] done — 299 steps (~10.0 Hz).
```

299 steps in 30 s is the configured 10 Hz cap, so the loop kept up with no
degradation. Note this is an **ACT** policy: it does not exercise the DDIM
diffusion sampler, so the ~9 Hz DDIM claim is **not** re-verified by this run.

## Defects found by the physical gate

Five failures reached a connected robot that 400+ passing tests did not catch.
Each now has a regression guard.

| # | Defect | Origin | Guard |
| --- | --- | --- | --- |
| 1 | No run wrappers; `python -m` resolved a stale pre-migration package installed in the venv, so a smoke test could silently exercise code this repo does not contain | migration | `test_wrappers_resolve_this_repo.py` |
| 2 | `WebcamEEController` had no `close()`; the recorder, reconstructed from a larger worktree controller, called it | migration | `test_programs_only_use_existing_api.py` |
| 3 | `oak_camera.py` calls the depthai **v3** `Camera.build()` API; the environment has depthai 2.32 | pre-existing | see open gates |
| 4 | `WebcamSource.start()` and `start_oak()` created different attribute sets; `latest_frame()` raised on the monocular path | pre-existing | `test_source_paths_agree.py` |
| 5 | The shared wrapper exported `CUDA_VISIBLE_DEVICES=""` for every program, hiding the GPU from deployment | migration | GPU-policy tests in `test_wrappers_resolve_this_repo.py` |

A sixth issue was an operator error rather than a defect: a policy trained on two
cameras was handed to the single-camera deployment path, and the mismatch only
surfaced as a `KeyError` **after** the arm had ramped and started moving
autonomously. `deploy_so101_ee.py` now refuses a camera-mismatched policy before
`robot.connect`, and a test asserts that ordering.

Each guard was mutation-tested: the defect was reintroduced and the guard
confirmed to fail, so none of them is vacuously green.

## Open gates

- **OAK-D path unverified.** `--oak` cannot work against depthai 2.32; it needs
  depthai v3 or a rewritten `oak_camera.py`. Monocular is the default and is what
  the physical gate above exercised.
- **DDIM deployment rate unverified here.** The ~9 Hz figure comes from diffusion
  policies; this audit ran ACT.
- **PressureVision mode unverified on hardware.** `--gripper-mode pressurevision`
  passes its software tests, but no physical run has exercised the PV sender path.
- **PressureVision comparison study unfinished.** See `CLAIMS_AND_GATES.md`.

## Publication

This audit covers the repository's readiness. Creating a GitHub repository, adding
a remote, pushing, or uploading datasets is a separate action requiring explicit
authorization, and none of it has been performed.
