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

## Layout

```
packages/webcam_input/      MediaPipe hand tracking, depth, gestures, camera sources
packages/so101_teleop/      EE control, IK bridge, teleop, recording, deployment
integrations/pressurevision/  optional PV grip-severity control (separate process)
training/                   reproducible training scripts and notebooks
scripts/                    run wrappers (camera view, teleop, record, deploy, diagnose)
research/                   exploratory work, status stated per directory
local/                      git-ignored: datasets, checkpoints, evidence, scratch
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
MediaPipe release opens it.

The core packages import and run without PressureVision installed.

## Status

See `docs/CLAIMS_AND_GATES.md`. It separates what has been verified on hardware
from what has only passed software checks, and names the gates that remain open.
Please read it before citing any result from this repository.

## License

Apache-2.0. See `LICENSE` and `NOTICE` — `NOTICE` carries the required attributions
for LeRobot and LeFranX (Apache-2.0) and PressureVision and vr-dex-retargeting (MIT).
