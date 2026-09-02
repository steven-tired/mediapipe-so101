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

## Requirements

The repository carries no machine-specific paths — every one lives in
`scripts/smoke_env.sh`, and each missing piece fails with a message naming the
variable to set. It is independent of the pre-split `webcam-input` tree, but it
is **not self-contained**; it needs, from outside:

| What | Variable | Needed by |
| --- | --- | --- |
| A Python 3.12 env with LeRobot installed | `SO101_PYTHON` | everything |
| `vr-dex-retargeting`'s `vector_retargeting` | `VR_DEX_RETARGETING_DIR` | hand tracking |
| An SO-ARM100 checkout (URDF for IK) | `SO_ARM100_DIR` | arm control |
| The released PressureVision checkout + weights | `SO101_PV_REPO` | PV only |
| A torch env for the PV network | `SO101_PV_PYTHON` | PV only |

Hardware: an SO-101 on a Feetech bus, a hand camera, and a workspace camera. The
PV path additionally needs an overhead pad camera and a distinct side camera.

**Use depthai 2.32 for the OAK path.** On 3.7.1 the device firmware crashes on
every pipeline start; `oak_camera.py` supports both APIs so the pin is a choice,
not a constraint.

## Quickstart

```bash
source scripts/smoke_env.sh        # or your own copy of it
./scripts/view_camera.sh --profile dp100
./scripts/run_arm_ee.sh            # keep the right hand visible for 3 s to arm
./scripts/run_record_ee.sh         # SPACE saves the episode, ESC discards, R re-records
```

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
