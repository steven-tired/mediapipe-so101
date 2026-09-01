# CLAUDE.md — mediapipe-so101 (public)

Single-webcam MediaPipe hand tracking → SO-101 arm teleoperation, LeRobot
recording/training/deploy, plus the optional PressureVision grip integration.
Split out of the `hand-teleop` meta-workspace; intended to be publishable.

## Two rules

**1. Nothing here may import the private IR repo.** No `ir_force`, no `flir`,
no `lepton`, no `thermal_project`. `packages/so101_teleop/tests/test_public_boundary.py`
enforces it, along with "no developer home paths anywhere published".

**2. The core package never imports `pressurevision_integration`.**
`grip/compose.py` is the single exception, and it imports lazily, so the core
keeps working when PV is not installed. The seam is
`grip/contract.py`: MediaPipe owns arm motion and grasp/release authority; a
`GripperController` only decides *how far* to close. That is what keeps the PV
path from ever being able to open the gripper.

## Layout

```
packages/webcam_input/     MediaPipe hand input device
packages/so101_teleop/     the LeRobot teleoperator plugin + programs/
packages/policy_grip_aux/  LeRobot policy plugin: ACT/Diffusion with grip-intent supervision
integrations/pressurevision/  optional PV integration: src/ library, tools/ programs
research/                  training scripts
training/, local/          checkpoints and evidence (local/ is git-ignored)
```

## Running things

```bash
env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q     # 502 tests
./scripts/run_arm_ee.sh        # live EE teleop      (wrappers resolve their own paths)
./scripts/run_record_ee.sh     # record demos
./scripts/run_deploy_ee.sh     # autonomous policy, DDIM @ 10 steps (~9 Hz)
```

`scripts/_common.sh` **prepends** this repo's src dirs to PYTHONPATH, to defeat
a stale editable install of the pre-split package. Keep it that way.

Set `LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT=1` to import the pure
helpers (`grip.proposal`, `gripper_hardware`, `hand_startup_gate`) without
registering the plugin and pulling in `lerobot.motors`.

## Before changing a program

Three guards will fail if you get these wrong, and each exists because the
mistake was made: every tracked file must AST-parse
(`test_every_python_file_compiles.py`), every program must import with no robot
configured (`test_programs_import_cleanly.py` — resolve paths lazily, never
`str(urdf_path())` at module level), and a program may only use attributes its
collaborators actually have (`test_programs_only_use_existing_api.py`).

`docs/CLAIMS_AND_GATES.md` separates what is measured from what is assumed.
`docs/RELEASE_AUDIT.md` lists the gates still open before this can be published.
