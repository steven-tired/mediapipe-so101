# Release Audit

Migration of the MediaPipe/SO-101 system out of the `hand-teleop` meta-workspace
into this repository. Run 2026-09-01.

Evidence is reported at three levels and never mixed: **software** (tests and
static checks), **locked inference** (offline evaluation), and **physical robot**
(observed closed-loop behaviour).

The first two sections were run on 2026-09-01 on the development machine and
cover software and physical. The **Phase 2** section below was run the same day
but remotely, and covers software only.

## Software gate

Numbers as of the migration run. The Phase 2 section below carries the current
ones; where the two disagree, this table is the historical record, not a claim
about the repository today.

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

## Phase 2 — the PressureVision force-control runtime

Run 2026-09-01, **remotely over SSH: no robot, no cameras, no physical evidence
of any kind.** Everything below is the software level only.

### Why this phase existed

The migration above left a reproducibility gap. This repository held the
consumer (`research/train_grip_residual_head.py`), the data (`local/evidence/`)
and the documentation (`training/TRAINING_HANDOFF.md`) — but not the code that
produces the teacher labels those episodes carry. The PV grip runtime lived
only in a worktree of the pre-split checkout.

That code is now here: the adjustment lock, the range and relative mappers, the
proposal state machine, the closure limiter, the schema-v7 telemetry, the PV
recorder and the grip-supervised deploy program.

### Software gate

Numbers as of the migration run. The Phase 2 section below carries the current
ones; where the two disagree, this table is the historical record, not a claim
about the repository today.

| Check | Result |
| --- | --- |
| Full suite | **745 passed**, 0 failed |
| `packages/webcam_input` | 27 |
| `packages/so101_teleop` | 383 |
| `packages/policy_grip_aux` | 5 |
| `integrations/pressurevision` | 330 |
| Tracked files | 192 |
| Largest tracked file | 72.8 KB (`training/TRAINING_HANDOFF.md`) |
| `git ls-files local` | 0 |
| `git status --short` | empty |
| Core imports `pressurevision_integration` | only `grip/compose.py`, lazily |
| Tracked imports of `ir_force` | 0 |
| Tracked developer paths in published Python | 0 |
| Remote | `steven-tired/mediapipe-so101`, **private** (anonymous fetch 404) |

`scripts/smoke_env.sh` does carry absolute paths. It is this machine's
environment file rather than published code, and the boundary test scans
Python under `packages/`, `integrations/` and `research/`. Stated here so the
"0 developer paths" row is not read more broadly than it is meant.

### What this phase changed in behaviour — all software evidence only

Four decisions changed how the system behaves. None has been observed on
hardware. Each is listed with what would confirm or refute it.

| # | Change | Why | What would test it |
| --- | --- | --- | --- |
| 1 | Losing the hand (HOLD) or clutching disarms PV; it must earn a fresh baseline before driving again | A hand that leaves the frame may have put the object down or moved on the pad, and the sensor cannot tell. Resuming from the old zero means commanding force against a scene it can no longer see. `local/evidence/` was recorded under this behaviour, so reproducing those labels needs the same control law | Park and resume mid-grasp; the command must follow the pinch path until a baseline lands, not jump back to the previous PV target |
| 2 | A dead sender latches and holds, rather than falling back to the pinch path | The pre-split controller reverted to pinch control mid-grasp. Swapping control laws while holding an object is not something a bench test catches | Kill the sender during a grasp; the gripper must hold its position, not re-track the hand |
| 3 | The middle-pose gesture is configurable; the PV recorder uses the right-hand V-sign | PressureVision occupies the left hand — it is pressing the pad — so a left fist is not a gesture the operator can make. The default stays the fist for the non-PV path, which is what the 100-episode dataset was recorded through | Both gestures on the arm; the V-sign's 0.4 s dwell should not fire on a hand passing through the pose |
| 4 | The gripper is chosen at construction; there is no "legacy" decision record when PV is absent | A consequence of moving PV out of the controller. It removes the pre-split fallback in which a closed sender silently returned control to the pinch path | Covered by 2 |

Two migration-only changes are also unverified on hardware: the PV wrapper
hides the GPU per-invocation rather than for the whole script (so the sender
keeps the GPU it needs), and `deploy_so101_grip_ee.py` defaults to the Hub copy
of the ACT 80k final instead of a local checkpoint path.

### Tooling migrated afterwards

Auditing what was left behind turned up work the phase had missed, including a
precondition of the procedure below:

- **The camera alignment tool.** `view_camera.py`, its dp50/dp100 profiles and
  its tests lived only in the meta-workspace, so `--profile` — which the
  procedure below depends on, and which decides whether a policy sees the
  layout it was trained on — did not exist here at all.
- **A mislabelled wrapper.** `view_camera.sh` ran `teleop_viz`, which *drives
  the arm*, while its own comment said "no arm motion". They are now separate
  wrappers (`view_camera.sh` and `run_teleop_viz.sh`) and a test checks which
  module each one actually launches, ignoring what its comments mention.
- **Five PV/OAK entry points**: the pad rig (`run_pv_pad.sh`), the fixed-pose
  carton trials, the correction-recording deploy, and the OAK probe with its
  wrapper. The programs were already here; only the entry points were missing.

### What was deliberately not migrated

The IR-shadow recording path (`record_so101_ee` under `SO101_IR_PRESSURE`, and
its 21 tests) exists in neither repository. It was left behind by the phase-1
split and will not be recovered: IR recording is not planned. Recorded IR
evidence is unaffected; only the ability to record more of it is gone.

## Open gates

- **OAK-D path unverified.** `--oak` cannot work against depthai 2.32; it needs
  depthai v3 or a rewritten `oak_camera.py`. Monocular is the default and is what
  the physical gate above exercised.
- **DDIM deployment rate unverified here.** The ~9 Hz figure comes from diffusion
  policies; the physical gate above ran ACT.
- **The PressureVision force-control path has never run connected to a robot.**
  Restated after phase 2, because it is now a larger claim than it was: the
  adjustment lock, the range mapper, the proposal machine and the closure
  limiter are all here and all green in software, and phase 2 additionally
  changed four behaviours (see the table above) on software evidence alone. The
  gap between "both suites green" and "PV force control works on the arm" is
  this gate, and the physical gate above already showed what that gap can hide:
  five defects survived 400+ passing tests.
- **PressureVision comparison study unfinished.** See `CLAIMS_AND_GATES.md`.

### Closing the PressureVision gate

The procedure, so it can be run without re-deriving it. Preconditions: SO-101 on
`usb-1a86_USB_Single_Serial_5B14110850`, the Creative Live! Cam workspace
camera, `./scripts/view_camera.sh --profile dp100` for alignment, a fitted
`levels.json` (`./scripts/run_pv_pad.sh aim | capture | fit`), and
`SO101_PV_PYTHON` pointing at the PressureVision environment. **Never substitute
`python -m` for the wrappers** — defect #1 above is exactly that mistake.

1. **PV teleoperation.** `./scripts/run_record_pv_ee.sh` with `PV_LEVELS` set to
   a freshly fitted `levels.json`, gripping a paper carton (the `carton_span`
   mapping is calibrated for it: zero=32, one=20). Four state transitions to
   observe, not "it did not crash":
   - the command follows PV while in contact and stays inside 20..32;
   - contact lost for ≥1.0 s latches — `adjustment_state` goes
     `adjusting → temporary_hold → locked`;
   - a single flickering re-contact frame does **not** unlatch; ≥0.15 s does;
   - opening the right hand releases the grip **with the PV sender killed**.
     This last one is the hardware evidence for the safety contract, and the
     software test for it uses a fake sender.
2. **PV recording.** Read the episode back: schema v7 fields present
   (`grip_intervention`, the teacher label, `mapping_contract`), and the teacher
   values must **vary**. A constant column means the adjustment lock never
   reached `locked`, and the residual head would be learning noise.
3. **DP deployment.** `./scripts/run_deploy_ee.sh --duration 30` with **DP100,
   not ACT**, which also settles the DDIM gate: ~270 steps in 30 s is ~9 Hz.

Record the result here in the format above, per step, and **write FAIL and
leave the gate open if any step fails.** Partial success is not a pass.

## Publication

This audit covers the repository's readiness, not its publication.

Since the migration run, this repository has been pushed to a **private** GitHub
repository (`steven-tired/mediapipe-so101`) with the owner's authorization. It
was verified private by anonymous fetch: unauthenticated HTTP returns 404 and an
unauthenticated `git ls-remote` demands credentials. The private counterpart
`steven-tired/ir-camera-force` is likewise private.

Making either repository **public**, or uploading datasets or checkpoints,
remains a separate action requiring explicit authorization, and none of it has
been performed. The open gates above are capability gaps, not a publication
checklist: closing them does not by itself authorize publication.
