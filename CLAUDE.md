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
env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q     # 987 tests
./scripts/run_arm_ee.sh        # live EE teleop      (wrappers resolve their own paths)
./scripts/run_record_ee.sh     # record demos
./scripts/run_deploy_ee.sh     # autonomous policy, DDIM @ 10 steps (~9 Hz)

./scripts/run_record_pv_ee.sh  # PV-supervised recording: sender + recorder, two processes
./scripts/run_deploy_grip_ee.sh  # deploy with grip supervision (residual head, intervention)
```

The two PV wrappers need `integrations/pressurevision/tools` on the path, which
they add themselves. The PV sender needs its own environment (torch +
segmentation-models-pytorch) **and** the released PressureVision checkout that
holds `config/` and `data/model/`: point `SO101_PV_PYTHON` and `SO101_PV_REPO`
at them. `scripts/smoke_env.sh` sets every machine-specific path this repo
needs, and is the only file that carries one.

`scripts/_common.sh` **prepends** this repo's src dirs to PYTHONPATH. It was
written to defeat a stale editable install of the pre-split package; that
install was replaced on 2026-09-02 and `webcam-input/` retired, so the original
reason is gone. Keep it anyway — it makes the wrappers run this checkout rather
than whatever is installed, which is the property that mattered all along.

**Use depthai 2.32.** `oak_camera.py` speaks both the v2 and v3 pipeline APIs and
picks by installed version, but on 3.7.1 the device firmware crashes on every
pipeline start (`PlgSrcMipi`, "Start Source: Invalid config steps") and the host
silently reconnects, so a run looks fine and leaves a crash dump.

Set `LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT=1` to import the pure
helpers (`grip.proposal`, `gripper_hardware`, `hand_startup_gate`) without
registering the plugin and pulling in `lerobot.motors`.

## Before changing a program

Guards will fail if you get these wrong, and each exists because the mistake was
made: every tracked file must AST-parse (`test_every_python_file_compiles.py`),
every program must import with no robot configured
(`test_programs_import_cleanly.py` — resolve paths lazily, never
`str(urdf_path())` at module level), a program may only use attributes its
collaborators actually have (`test_programs_only_use_existing_api.py`, which
covers `integrations/pressurevision/tools/` too), every intra-repo import must
resolve to a file this repo owns (`test_no_module_comes_from_outside_this_repo.py`),
and the control frame's timestamp must come from a clock
(`test_control_frame_timestamp.py`).

## The failure shape to watch for

Six of the ten defects the 2026-09-02 physical gate found share one shape: a
defensive construct turning something that should have raised into a silently
wrong value. `hasattr(x, "f") else 0.0` on a field that does not exist froze the
control clock, which froze the PV filter and every duration the adjustment lock
measures — pressing the pad did nothing, while telemetry reported the reading as
active. A test double built to whatever the code asked for answered four wrong
field names as happily as the right ones, so five dataset columns were constant
zero. `bus.ping()` returns None rather than raising, and a diagnostic that
counted only exceptions reported a dead servo as "ok".

Prefer named access and let it raise. When a value really can be absent, say so
explicitly for that case only — the opposite mistake (removing tolerance for a
legitimate `None`) crashed a recording the same day. `docs/RELEASE_AUDIT.md`
records all of them.

`docs/CLAIMS_AND_GATES.md` separates what is measured from what is assumed.
`docs/RELEASE_AUDIT.md` lists the gates still open before this can be published.
