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
version** — it is at 0.6.x — so LeRobot must come from a source checkout at a
commit that declares 0.5.2. This repository is developed against `da92db8`.

```bash
git clone https://github.com/huggingface/lerobot
git -C lerobot checkout da92db8
```

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
feetech-servo-sdk 1.0.0 and torch 2.11.0+cu128.

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

## Gripper modes

```bash
--gripper-mode mediapipe      # default
--gripper-mode pressurevision # optional, requires the PV sender process
```

MediaPipe always owns arm motion and grasp/release authority. The default mode
derives gripper position from pinch. The optional PressureVision mode uses PV only
for grip *severity* while a MediaPipe grasp is active; PV never opens the gripper.

Missing, invalid, or stale PV input holds the current gripper command — a silent
sender keeps the gripper closed rather than dropping a held object. Only an explicit
MediaPipe release opens it. A session that ends for any reason — key, fault, or
exception — opens the gripper before closing the bus, because dropping torque alone
does not: gear friction keeps the jaw on the object.

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
