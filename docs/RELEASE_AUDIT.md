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

- ~~**OAK-D path unverified.**~~ **Closed 2026-09-02.** `oak_camera.py` now
  speaks both pipeline APIs and picks by installed version, so the pin is no
  longer the constraint. Both were run against the hardware. **Use v2 (2.32).**
  On depthai 3.7.1 the device firmware crashed on *every* pipeline start —
  `RTEMS_FATAL_SOURCE_INVALID_HEAP_FREE` in `PlgSrcMipi`, "Start Source: Invalid
  config steps" — three runs, three crash dumps, with the host silently
  reconnecting afterwards, which is why 15 s of clean frames and a crash dump
  coexist. 2.32 produced no dump. The PV recorder defaults to OAK, so this was
  not a side path: PV recording could not start at all under 2.32 before this.
- **DDIM deployment rate unverified here.** The ~9 Hz figure comes from diffusion
  policies; the physical gate above ran ACT.
- **The PressureVision force-control gate is PARTLY closed — it stays open.**
  Run on hardware 2026-09-02; see "PressureVision gate — 2026-09-02" below.
  Steps 1a, 1b and 2 pass with evidence. Step 1c has only partial evidence,
  step 1d **fails**, and step 3 was not run. Per the rule at the end of the
  procedure, partial success is not a pass.
- **PressureVision comparison study unfinished.** See `CLAIMS_AND_GATES.md`.

### Closing the PressureVision gate

The procedure, so it can be run without re-deriving it.

Preconditions: SO-101 on `usb-1a86_USB_Single_Serial_5B14110850`, the C270 pad
camera on `/dev/video2`, the Creative front camera, the Etron side camera, and
an OAK-D on **depthai 2.32** (see the OAK entry under Open gates). `source
scripts/smoke_env.sh` sets the four PV variables — `SO101_PV_PYTHON`,
`SO101_PV_REPO` (the released PressureVision checkout holding `config/` and
`data/model/`), `PV_LEVELS` and `PV_SESSION_DIR`. **Never substitute `python -m`
for the wrappers** — defect #1 of the first physical gate is exactly that
mistake, and a missing `SO101_PV_REPO` is the same family.

A full refit is usually unnecessary: `./scripts/run_pv_pad.sh rematch <session>`
puts the camera back where an existing `levels.json` was fitted, and the
sender's `--require-scene-match` refuses to stream if it did not work. Refit
(`aim | capture | fit`) only when rematch cannot converge. Copy an existing
`levels.json` with `cp -p`: the freshness gate reads file **mtime**, so a plain
copy makes a stale fit look new, and raising `PV_MAX_LEVEL_AGE_MINUTES` should
be a visible decision rather than a side effect of copying.

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

## PressureVision gate — 2026-09-02

Run from this repository with the SO-101, the C270 pad rig, the Creative front
camera, the Etron side camera and an OAK-D on depthai 2.32. The calibration was
**not** refitted: session 26's `levels.json` (2026-08-26) was copied into
`local/pv_sessions/` with `cp -p`, and `run_pv_pad.sh rematch 26` put the camera
back — measured drift afterwards 2.4 px and 2.8 points of area, against limits
of 40 px and 6.0. The freshness gate was raised deliberately for the run
(`PV_MAX_LEVEL_AGE_MINUTES`); mtime was preserved precisely so that override had
to be explicit, because the gate reads file mtime and an ordinary `cp` would
have made a week-old fit look minutes old. The scene fingerprint is the guard
doing the real work here.

### 1a. Command follows PV, inside 20..32 — PASS

`relative_closure` 0 → 11.98 against a span of 12, `range_live_target`
20.02 → 32.0, and `proposed_gripper_pos` equal to `actual_gripper_pos`
throughout, never outside the span.

### 1b. `adjusting → temporary_hold → locked` — PASS

`pv_adjustment_state`: adjusting 146, temporary_hold 28, locked 25.
`release_elapsed_s` reached 1.003 s against its 1.0 s threshold. Anchors latched
at q=29.16 and q=23.39, and in a later run at q=20.84 then q=31.43 — a deeper
grip after a resume, which is the behaviour the lock exists for.

### 1c. Flicker does not unlatch, ≥0.15 s does — PARTIAL

One `recontact_started` event and two non-null `recontact_since_s` frames were
observed. That shows the debounce ran; it does **not** separately demonstrate
both halves — that a single flickering frame is rejected *and* that ≥0.15 s is
accepted. Deliberately not recorded as a pass.

### 1d. Right hand releases the grip with the sender dead — FAIL

Killing `serve_pad_pressure.py` mid-grip and opening the right hand did **not**
release. Two distinct causes, both still true of the design:

* **The control loop stops.** `invalidate_episode()` sets `stop_recording` and
  `exit_early` together, so a latched PV fault ends the whole session on the
  next frame — `fault_latched` is True on exactly one row (tick 323) and the
  loop is gone. Nothing was left running to notice a hand opening. The recorded
  `base_gripper_pos` over the final 25 frames is 20–26, nowhere near
  `GRIP_LATCH_EXIT = 65`, because those frames precede the kill.
* **`explicit_release` has no producer.** It is hardcoded `False` at both
  `GripInput` sites in `ee_controller.py`, and nothing in the repo sets it true,
  so its three consumers are dead code. `pv_grip_adapter.py`'s comment —
  "Explicit release is MediaPipe's alone. It must work with the PV sender dead,
  so it never consults the runtime" — describes a mechanism that does not exist.
  Release today depends on the range mapper's grasp latch, which runs *inside*
  the PV runtime: exactly the dependency the contract forbids.

**What was fixed instead, and what it is not.** The session now opens the
gripper before the bus closes (`release_gripper_before_disconnect`), on every
exit path including a crashing one — confirmed on hardware. Torque-off alone
does not do this: `disable_torque_on_disconnect` is already True, but gear
friction keeps the jaw closed on the object. That is a real improvement and it
covers ESC, PV fault and exception alike. **It is not this gate.** This gate
asks for release *while running*, with PV dead, and that remains unreachable.

An earlier attempt to keep the loop alive after a fault for a 30 s "release
window" was written and then reverted: it means driving the arm on sensor state
already judged untrustworthy, and stopping — which drops torque — is the better
failure mode. Recorded here so it is not re-proposed as an obvious fix.

### 2. Recording readback — PASS

Episode `success_no_slip`, promoted, 55 frames. `observation.grip_intent_teacher`
carries 15 distinct values over 0..0.972 — **not** a constant column, so the lock
did reach `locked`. `meta/pv_mapping_contract.json` is present with the full
carton_span contract. Provenance is live: `grip_intent_sequence` 303..464 (55
distinct), `grip_intent_frame_age_s` 0.021–0.070 s, and the three timestamps
float64 spanning the episode.

**Read those three timestamps from the parquet, not through `ds[i]`.** The
tensor path converts to float32, and a unix timestamp needs more precision than
float32 carries, so they read back collapsed to a single value ~128 s wide.
`frame_age_s` is the one to use from the tensor path.

### 3. DP deployment / DDIM rate — NOT RUN

## Defects found by this gate

Ten, on top of the five the first physical gate found. All have regression
guards, and every guard was mutation-tested — the defect reintroduced and the
guard confirmed to fail.

| # | Defect | Shape |
| --- | --- | --- |
| 1 | No PressureVision checkout: the migrated default became `None` and nothing replaced it | absent default |
| 2 | `WebcamSample`, `latest_sample()`, `image_xy`, `depth_m`, `sample_landmark_depths` all missing | thin type taken from the wrong branch |
| 3 | `LatestFrameSource` filed on the private side, imported from the public side | one-way dependency violated |
| 4 | `WebcamSource.stop()` not idempotent — a clean ESC ended in a traceback | double close |
| 5 | `oak_camera.py` v3-only against a 2.32 runtime | API pin |
| 6 | **Control frame timestamp constant 0.0** | `hasattr(...) else 0.0` |
| 7 | `so101_diag ids` could not report an absent motor | `ping()` returns None, only exceptions counted |
| 8 | Five of seven PV dataset columns constant 0 | four wrong field names, `getattr(..., None)` |
| 9 | `mapping_contract` never reached the dataset | written only to the evidence manifest |
| 10 | The API guard skipped the PV tools directory and knew only the controller | guard scope |

**Six of these share one shape**: a defensive construct — `hasattr`/`getattr`
fallbacks, a test double built to whatever the code asked for, counting only
exceptions, a `callable()` check — turning something that should have raised
into a silently wrong value. #6 is the clearest: a missing attribute became a
frozen clock, which froze the PV low-pass *and* every duration the adjustment
lock measures. One dead value produced every PV symptom, and 787 passing tests
saw none of it.

#7 misled this audit in the moment: its false "missed 0/10 ok" was used to claim
a motor was answering while `robot.connect()` was failing on that same motor.

An eleventh defect was introduced *during* this session and fixed here: tightening
#8 from `getattr` to named access removed the tolerance for `reading is None`,
which is a real state before the first PV packet, and crashed a recording. The
lesson runs both ways — the point is to distinguish a legitimate absent state
from a misspelled field, not to prefer one style of access.

## Environment note

The LeRobot venv carried three editable installs pointing into the pre-split
`webcam-input/` tree. Because setuptools' `_EditableFinder` sits after
`PathFinder` on `sys.meta_path`, they were a **silent fallback**: any submodule
this repo lacked was served from the old tree instead of raising. "Imports
succeed" and "the suite passes" therefore said nothing about whether this
repository is complete. They were replaced with editable installs of this repo's
own packages on 2026-09-02, and `webcam-input/` was retired;
`test_no_module_comes_from_outside_this_repo.py` now asserts completeness
statically, which is the only form of the question that survives the old tree
going away.

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
