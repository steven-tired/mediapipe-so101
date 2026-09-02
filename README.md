# mediapipe-so101

Single-webcam hand tracking driving an [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm: live end-effector teleoperation, [LeRobot](https://github.com/huggingface/lerobot)
dataset recording, and policy deployment. PressureVision-based grip control is an
optional integration.

A commodity webcam plus MediaPipe replaces a VR headset as the input device. The
operator's right wrist pose drives the arm through inverse kinematics; pinch drives
the gripper. The same controller runtime backs preview, live teleoperation, and
recording, so a recorded action path cannot silently differ from the one the
operator tested.

## What it can do

| Capability | Entry point | Verified on hardware |
| --- | --- | --- |
| Live EE teleoperation (webcam → arm) | `./scripts/run_arm_ee.sh` | yes |
| Record a LeRobot dataset from teleoperation | `./scripts/run_record_ee.sh` | yes |
| Deploy a trained policy autonomously | `./scripts/run_deploy_ee.sh` | yes (ACT) |
| Workspace camera alignment | `./scripts/view_camera.sh --profile dp100` | yes |
| Servo fault diagnosis (registers, ids, lift, PID) | `./scripts/run_so101_diag.sh` | yes |
| Deployment telemetry harness | `./scripts/run_diagnose.sh` | yes |
| PV pad rig: aim, rematch, capture, fit, serve | `./scripts/run_pv_pad.sh` | yes |
| PV-supervised recording (grip severity from pressure) | `./scripts/run_record_pv_ee.sh` | partly — see below |
| PV grip-supervised deployment | `./scripts/run_deploy_grip_ee.sh` | no |
| OAK-D depth hand tracking (opt-in) | `--oak` / `probe_oak.sh` | yes, on depthai 2.32 |

**Read `docs/CLAIMS_AND_GATES.md` and `docs/RELEASE_AUDIT.md` before citing any
result from this repository.** They separate what has run on the arm from what has
only passed software checks, and name the gates still open. The PV force-control
gate is **partly closed and still open**: the command follows pressure inside its
mapped span and the adjustment lock latches, both with hardware evidence, but the
"release while running with the PV sender dead" contract is not satisfied.

## Layout

```
packages/webcam_input/            the input device, robot-free
  detector.py                     reuses LeFranX's SingleHandDetector
  webcam_source.py                capture loop; publishes one atomic WebcamSample
  wrist_estimator.py / depth.py   wrist pose; monocular scale or OAK metric depth
  oak_camera.py                   OAK-D; speaks both depthai v2 and v3 APIs
  gestures.py / types.py          fist clutch, middle-pose; payload dataclasses

packages/so101_teleop/            everything that touches the arm
  ee_controller.py                the control frame: pose -> IK -> joint targets
  ee_control.py                   grip ratchet, span mapping, bounded wrist roll
  grip/                           gripper contract; the MediaPipe implementation
  programs/                       teleop, record, deploy, diagnostics, probes
  paths.py                        resolves SO-ARM100 URDF and local storage

packages/policy_grip_aux/         ACT/DP wrappers with privileged grip supervision

integrations/pressurevision/      optional, runs as a SEPARATE process
  src/pressurevision_integration/ protocol, adjustment lock, range mapping,
                                  grip runtime/adapter, shadow telemetry
  tools/                          the sender, the pad rig, the PV recorder,
                                  trial analyzers

scripts/                          run wrappers; they resolve this repo and the
                                  right interpreter, so never use `python -m`
training/                         reproducible training scripts and notebooks
research/                         exploratory work, status stated per directory
docs/                             claims, gates, release audit, dataset map
local/                            git-ignored: datasets, checkpoints, evidence
```

### Data flow

```
webcam ──▶ MediaPipe Hands(2) ──▶ WebcamSample{wrist, landmarks, frame, t, id}
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
        WebcamEEController.step()                  gripper controller
        wrist delta ──▶ IK ──▶ joints              pinch ──▶ position
                     │                                     │
                     └────────────▶ joint action ◀─────────┘
                                        │
                        LeRobot SOFollower ──▶ SO-101

optional:  pad camera ──▶ PressureVision net ──▶ UDP :8090 ──▶ grip severity
```

The PV sender is a separate process on purpose: it needs its own environment
(torch + segmentation-models-pytorch), and the teleop must keep working when it
is not running.

## Setup

Python **3.12** (the packages pin `>=3.12,<3.13`). These steps use
[uv](https://docs.astral.sh/uv/); plain `pip` works with the same arguments.

### 1. LeRobot, from source

`packages/so101_teleop` pins `lerobot==0.5.2`. **PyPI does not have that
version** — it is at 0.6.x — so LeRobot must come from a source checkout.

```bash
git clone https://github.com/huggingface/lerobot
git -C lerobot checkout da92db8
```

**Why that commit, honestly:** it is upstream `main` from 2026-06-17, the day
this line of work started. It declares `0.5.2` in its own `pyproject.toml` and
matches no release tag. The pin records what was installed, not a compatibility
finding — nobody has tested 0.6.x here.

Upgrading is real work rather than a version bump, because this repository
imports well below LeRobot's public surface: `datasets.lerobot_dataset`,
`model.kinematics`, `motors.feetech`, `policies.*.processor_*`,
`policies.factory`, `common.control_utils`, and `record_loop` from
`scripts.lerobot_record`. Those move between versions — the
`so101_follower` → `so_follower` rename already happened once. The recorded
datasets are LeRobot format `v3.0` and the trained policies came out of this
version's training stack, so an upgrade needs a dataset readback and a physical
gate, not just a green test suite.

### 2. The environment

MediaPipe 0.10.21 declares `numpy<2` while LeRobot needs `numpy>=2`. They do
coexist — this repository runs on numpy 2.2.6 and mediapipe 0.10.21 — so the
declared pin is relaxed with an override rather than worked around:

```bash
uv venv --python 3.12 .venv
echo 'numpy>=2.0,<2.3' > overrides.txt

uv pip install --override overrides.txt \
  "mediapipe==0.10.21" \
  -e "./lerobot[feetech,dataset,placo-dep]" \
  -e mediapipe-so101/packages/webcam_input \
  -e mediapipe-so101/packages/so101_teleop \
  -e mediapipe-so101/packages/policy_grip_aux
```

`feetech` is the servo bus, `dataset` the LeRobot dataset stack, `placo-dep` the
IK solver. This resolves to numpy 2.2.6, mediapipe 0.10.21, placo 0.9.15,
feetech-servo-sdk 1.0.0 and torch 2.11.0+cu128 — the versions listed under
"The environment this was built on" below.

For the OAK-D path only:

```bash
uv pip install "depthai==2.32.0.0"
```

**Use 2.32, not 3.x.** `oak_camera.py` speaks both pipeline APIs, but on 3.7.1
the OAK firmware crashes on every pipeline start (`PlgSrcMipi`, "Start Source:
Invalid config steps") and the host reconnects silently, so a run looks healthy
and leaves a crash dump behind.

### 3. Two checkouts this repo reads, and does not vendor

```bash
git clone https://github.com/wengmister/vr-dex-retargeting
git clone https://github.com/TheRobotStudio/SO-ARM100
```

`vr-dex-retargeting` supplies `SingleHandDetector`, the MediaPipe → 21 landmarks
→ MANO pipeline this repo reuses rather than reimplements. `SO-ARM100` supplies
the URDF the IK solves against. Neither is copied in; both are named by
environment variable, so nothing assumes what sits next to it on disk.

### 4. Configure

The repository contains **no machine-specific paths**. They all live in one
file, `scripts/smoke_env.sh` — copy it and edit, or export these yourself:

| Variable | Points at | Needed by |
| --- | --- | --- |
| `SO101_PYTHON` | the venv's `bin/python` | everything |
| `VR_DEX_RETARGETING_DIR` | `<clone>/example/vector_retargeting` | hand tracking |
| `SO_ARM100_DIR` | the SO-ARM100 checkout | arm control (URDF) |
| `SO101_ARM_PORT` | the arm's `/dev/serial/by-id/...` | arm control |
| `SO101_WORKSPACE_CAM` | the workspace camera's `/dev/v4l/by-id/...` | recording, deploy |
| `SO101_LOCAL_DIR` etc. | where datasets and evidence go | recording |

Every one of these fails with a message naming the variable if it is wrong or
missing, so a misconfiguration surfaces at startup rather than mid-episode.

Then calibrate the arm once with LeRobot's own tooling
(`lerobot-find-port`, `lerobot-calibrate`), as for any SO-101.

### 5. Check it

```bash
python -m pytest -q                    # 987 tests, no hardware needed
./scripts/probe_oak.sh                 # only if you have an OAK-D
./scripts/view_camera.sh --profile dp100
./scripts/run_arm_ee.sh                # first motion; keep the e-stop in reach
```

**Always go through `scripts/`.** The wrappers put this checkout first on
`PYTHONPATH` and select the right interpreter; `python -m` can silently resolve
a different installed copy, which has already cost one debugging session here.

### Optional: the PressureVision path

Needs a second environment, because the network is torch + segmentation-models-pytorch
and the teleop must keep working when the sender is not running.

```bash
git clone https://github.com/facebookresearch/PressureVision
# fetch its released weights into data/model/paper_59.pt per its README
uv venv --python 3.12 .venv-pv
uv pip install --python .venv-pv/bin/python \
  torch torchvision segmentation_models_pytorch opencv-contrib-python pyyaml timm

export SO101_PV_REPO=$PWD/PressureVision      # holds config/ and data/model/
export SO101_PV_PYTHON=$PWD/.venv-pv/bin/python
```

Then `./scripts/run_pv_pad.sh aim | capture | fit` to calibrate the pad rig, and
`./scripts/run_record_pv_ee.sh` to record with PV supervision. The pad rig needs
an overhead camera on the pad and a side camera distinct from the workspace one.

### Hardware

An SO-101 on a Feetech bus, a camera for the hand, and a workspace camera. The
PV path adds an overhead pad camera and a distinct side camera. An OAK-D is
optional and replaces the hand camera with metric stereo depth.

### The environment this was built on

Everything above is the recipe that produced **this** machine's working setup,
written down after the fact. It has not been rebuilt from an empty environment,
and no other configuration has been tried. Treat it as "known to work here",
not as a support matrix — if your setup differs, the recipe is a starting point
rather than a guarantee.

| | |
| --- | --- |
| OS | Ubuntu 26.04 "Resolute Raccoon", kernel 7.0.4-76070004 (System76) |
| CPU / RAM | Intel Core i9-14900HX, 32 threads, 62 GiB |
| GPU | RTX 4060 Laptop, 8 GiB, driver 580.159.03 |
| Python | 3.12.13, via uv 0.11.21 |
| Robot | SO-101 follower, Feetech bus on a CH340 USB serial adapter |
| Cameras | Chicony built-in (hand), Creative Live! Cam (workspace), Logitech C270 (PV pad), Etron USB2.0 (side) |
| Depth | OAK-D on depthai 2.32.0.0 |

Upstream checkouts, at the commits this was run against:

| Repository | Commit |
| --- | --- |
| `huggingface/lerobot` | `da92db8` (declares 0.5.2) |
| `wengmister/vr-dex-retargeting` | `664abe2` |
| `TheRobotStudio/SO-ARM100` | `fda892c` |
| `facebookresearch/PressureVision` | `16fd342` |

Key resolved versions: numpy 2.2.6, mediapipe 0.10.21, torch 2.11.0+cu128,
placo 0.9.15, opencv-python 4.11.0.86, feetech-servo-sdk 1.0.0, torchcodec
0.11.1, av 15.1.0.

The GPU is used for policy training and deployment only. Teleoperation and
recording run CPU-only on purpose — the wrappers hide the GPU for those, because
this laptop's dGPU wedges on wake from runtime suspend.

## Workflows

### Gripper numbers, once

The gripper command is LeRobot's `RANGE_0_100`: **0 is fully closed, 100 fully
open.** Lower means tighter. `50` is the calibrated centre and is where the
MediaPipe path parks on release. Two thresholds matter downstream: below **30**
the operator has committed to a grasp, above **65** the hand is deliberately
open. The gap is wide and asymmetric on purpose — MediaPipe degrades as fingers
occlude each other while closing and recovers while opening, so the release
signal is trustworthy exactly when the grip signal is not.

### Live teleoperation

```bash
source scripts/smoke_env.sh
./scripts/view_camera.sh --profile dp100    # no arm motion
./scripts/run_arm_ee.sh
```

Hold your right hand in view; after **3 continuous seconds** the arm arms itself
and prints `ARM ENABLED`. Until then it will not move, so a half-tracked hand
cannot jerk it. Then:

- **right wrist** moves the end effector — differential, so where you start does
  not matter, only how you move;
- **right pinch** drives the gripper;
- **left fist** is the clutch: it freezes motion so you can reposition your arm,
  the way lifting a mouse re-centres it;
- **the middle pose** (`fist` by default, `right_v` optional) parks the arm at
  its middle pose and resets the grip state.

### Recording a dataset

```bash
./scripts/run_record_ee.sh
```

Same controls, plus episode keys in the preview window: **SPACE** saves the
episode, **ESC** discards it and ends the session, **R** re-records the current
attempt. Recording appends, so a session adds to whatever is already at
`SO101_DATASET_ROOT`.

### Deploying a policy

```bash
./scripts/view_camera.sh --profile dp100    # ALWAYS first
./scripts/run_deploy_ee.sh --policy <checkpoint dir> --duration 30
```

The camera profile is not cosmetic. `view_camera.sh` overlays the pixel targets
the policy was trained against — `dp100` expects the tag at `(416, 297)`,
`dp50` at `(421, 143)`, both within 30 px — and a mismatch is a visual domain
shift that the policy sees and **nothing reports**. Align first, every time.

`--duration` bounds the run. The deployment path refuses a policy whose camera
set does not match before the robot connects, rather than after the arm has
started moving.

### PressureVision grip control

PV supplies grip *severity* only, while a MediaPipe grasp is active. It can
squeeze harder or softer; it can never open the gripper or move the arm.

**1. Build the rig.** A sheet of **white paper flat on the table** is the pad —
that is the whole surface. Mount the pad camera (a C270 here) **overhead**,
looking down at it, on a fixed mount that will not be nudged; the workspace and
side cameras stay where recording needs them. Viewpoint is the variable that
decides whether the network responds at all, so aim by pressing and watching the
response rather than by measuring the frame. Lock the camera before trusting
anything you see:

```bash
v4l2-ctl -d /dev/video2 -c white_balance_automatic=0 \
    -c white_balance_temperature=4000 -c auto_exposure=1
```

Colour carries the blanching cue the network reads, so an unlocked camera drifts
away from the calibration it is scored against.

**2. Calibrate — once per rig, not per session.**

```bash
./scripts/run_pv_pad.sh aim              # frame the pad; writes the crop
./scripts/run_pv_pad.sh capture 07       # labelled press trials
./scripts/run_pv_pad.sh fit 07           # -> local/pv_sessions/pv_levels_07.json
```

`aim` is a live loop: arrow keys move the crop, `a/d/w/s` resize it, `g` snaps to
a suggested box, `space` prints the crop. Hug the pad; do not chase an aspect
ratio.

`capture` runs **8 repeats × 3 labels** (none / light / hard) — 24 trials. For
each, press and hold, then **SPACE** to record; the window counts down the hold
and tells you if you lifted early. Press at *your own* idea of light and hard:
the default `--intent-labels` mode deliberately has no kitchen scale, because a
scale under the pad raises the pressing surface and calibration on a geometry
the teleop never runs in produces anchors that do not transfer.

`fit` scores the separation and writes the boundaries. Read its confusion matrix
before trusting it — a session that cannot tell light from hard will not get
better downstream.

**If the rig has not moved, do not recalibrate.** Put the camera back instead:

```bash
./scripts/run_pv_pad.sh rematch 07
```

It overlays where the pad sat at calibration; nudge the camera until they agree.
The sender refuses to stream on a mismatch (`--require-scene-match`), so this
cannot be skipped silently. Copy an existing `levels.json` with `cp -p`: the
freshness gate reads file **mtime**, so a plain copy makes a stale fit look
minutes old, and raising `PV_MAX_LEVEL_AGE_MINUTES` should be a deliberate act.

**3. Record with PV supervision.**

```bash
PV_LEVELS=local/pv_sessions/pv_levels_07.json ./scripts/run_record_pv_ee.sh
```

This starts two processes: the PV sender (GPU, its own interpreter) and the
recorder (CPU). Grasp the object with your right hand as usual; then press the
pad harder or softer to modulate how hard the gripper squeezes.

**What the numbers mean.** The default `carton_span` mapping is calibrated for a
paper carton and spans **20..32**:

| PV reading | Command | Meaning |
| --- | --- | --- |
| no grasp | 100 | open |
| pressure 0 | **32** | touching, not squeezing |
| pressure 1 | **20** | firm squeeze |

So more pressure means a *lower* number, and during a grasp the command never
leaves 20..32. A different object needs a different span. `PV_MAPPING` selects it: the
recorder accepts `carton_span` (default), `soft_direct`, and `hard_profile`,
the last taking a fitted object profile via `PV_OBJECT_PROFILE`.

**The adjustment lock.** Pressing adjusts; letting go *keeps* what you set. Lose
pad contact for **1.0 s** and the state machine goes
`adjusting → temporary_hold → locked`, holding the grip at the position you had
reached — you can take your hand off the pad without the object dropping.
Touching the pad again for **0.15 s** resumes adjustment; a single flickering
frame does not. Each latch prints `[pv] adjustment LOCKED at q=…`.

**What to check in a run**, rather than "it did not crash": the command follows
pressure and stays inside 20..32; the lock reaches `locked` and prints an
anchor; and in the recorded episode `observation.grip_intent_teacher` **varies**
— a constant column means the lock never latched and the supervision is noise.

**If the PV sender dies**, the episode is discarded automatically and the
session ends. Ending is also how the grip is released: the gripper is commanded
open before the bus closes, because dropping torque alone does not open a jaw
held by gear friction.

### When the arm misbehaves

```bash
./scripts/run_so101_diag.sh ids      # ping every servo; finds a dropping one
./scripts/run_so101_diag.sh health   # torque/temp/current/gain registers
./scripts/run_so101_diag.sh relax    # torque off, to check binding by hand
./scripts/run_so101_diag.sh lift     # instrumented lift, run cold
```

Start with `ids`. A servo that answers 0/10 is absent from the bus even if the
chain past it still works — the two connectors on a Feetech board are wired in
parallel, so a half-seated plug loses that servo while the bus continues to the
next one. Reseat both connectors on the servo that dropped, not the cable
upstream of it.

`run_diagnose.sh` is the deployment-side harness: it logs camera, joint, load
and temperature alongside the predicted action, for separating "the policy is
wrong" from "the arm is not doing what it was told".

## The grip safety contract

```bash
--gripper-mode mediapipe      # default
--gripper-mode pressurevision # optional, requires the PV sender process
```

**MediaPipe always owns arm motion and grasp/release authority.** A
`GripperController` only decides *how far* to close, never whether to hold or
let go. That is the whole reason the PV path cannot open the gripper, and it is
enforced by the seam in `grip/contract.py` rather than by convention.

Missing, invalid, or stale PV input **holds** the current command — a silent
sender keeps the gripper closed rather than dropping whatever is in it. Only an
explicit MediaPipe release opens it. A session that ends for any reason — key,
fault, or exception — commands the gripper open before closing the bus, because
dropping torque alone does not: gear friction keeps the jaw on the object.

One gate in this contract is **not** satisfied, and `docs/RELEASE_AUDIT.md`
records it as open: releasing by opening your hand *while running* with the PV
sender dead. In practice the session ends when the sender dies, and ending
releases — but that is a different mechanism from the one the contract
describes, and it is not written down here as if it were the same thing.

The core packages import and run without PressureVision installed.

## Tests

```bash
python -m pytest -q     # 987 tests, no robot and no cameras required
```

The suite is robot-free by construction. That is also its limit: the physical
gates have repeatedly found defects that hundreds of passing tests did not, so
every guard added in response to one is mutation-tested — the defect reintroduced
and the guard confirmed to fail. `docs/RELEASE_AUDIT.md` records each.

## License

Apache-2.0. See `LICENSE` and `NOTICE` — `NOTICE` carries the required attributions
for LeRobot and LeFranX (Apache-2.0) and PressureVision and vr-dex-retargeting (MIT).
