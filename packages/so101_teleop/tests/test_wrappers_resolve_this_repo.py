"""The wrappers must run THIS tree, not an older installed copy.

`.venv-lerobot` on the development machine has an older build of
`lerobot_teleoperator_so101_webcam` installed, pointing at the pre-migration
checkout. A wrapper that merely cleared PYTHONPATH would silently run that copy —
a smoke test could then "pass" against code this repository does not contain.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
WRAPPERS = sorted(p for p in SCRIPTS.glob("*.sh") if p.name not in {"_common.sh", "smoke_env.sh"})


def test_the_expected_wrappers_exist():
    assert {p.name for p in WRAPPERS} == {
        "run_arm_ee.sh", "run_record_ee.sh", "run_record_pv_ee.sh",
        "run_deploy_ee.sh", "run_deploy_grip_ee.sh", "run_deploy_pv_corrections.sh",
        "run_diagnose.sh", "run_teleop_viz.sh", "run_pv_pad.sh", "run_so101_diag.sh",
        "run_carton_fixed_grip_trials.sh", "run_gripper_deadband.sh",
        "probe_oak.sh", "view_camera.sh",
    }


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_is_executable_and_sources_common(wrapper):
    assert os.access(wrapper, os.X_OK), f"{wrapper.name} is not executable"
    assert "_common.sh" in wrapper.read_text(encoding="utf-8")


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_puts_this_repo_first_on_the_path(wrapper):
    """Resolve the module the way the wrapper would, and check where it lands."""
    probe = (
        "import lerobot_teleoperator_so101_webcam as m, pathlib, sys;"
        "print(pathlib.Path(m.__file__).resolve())"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # The interpreter running the tests: the one that actually has the deps, and
    # the one whose site-packages may hold the stale pre-migration install.
    env["SO101_PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS}/_common.sh"; exec "$PYTHON" -c {probe!r}'],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip())
    assert REPO in resolved.parents, f"wrapper resolved to {resolved}, outside {REPO}"


def test_no_wrapper_hardcodes_a_developer_path():
    for wrapper in WRAPPERS + [SCRIPTS / "_common.sh"]:
        assert "/home/" not in wrapper.read_text(encoding="utf-8"), wrapper.name


# --- GPU policy ------------------------------------------------------------
# `CUDA_VISIBLE_DEVICES=""` hides every GPU. Teleop and recording want that
# (CPU-only, and it stops torch waking a flaky dGPU on import). Deployment and
# diagnostics run a policy and must NOT inherit it -- exporting it for every
# wrapper made deploy fail with "No CUDA GPUs are available".
CPU_ONLY = {"run_arm_ee.sh", "run_record_ee.sh", "run_teleop_viz.sh",
            "view_camera.sh", "probe_oak.sh", "run_so101_diag.sh",
            "run_gripper_deadband.sh"}
NEEDS_GPU = {"run_deploy_ee.sh", "run_deploy_grip_ee.sh", "run_diagnose.sh"}
# These cannot be probed by running them: they need hardware or a fitted
# levels.json before they reach any python, or they start several processes
# with opposite GPU needs. The PV recorder is the clearest case -- the recorder
# is CPU-only, its sender runs the PressureVision network -- so it hides the
# GPU per-invocation, which test_record_so101_pv_ee checks at the source level.
HARDWARE_GATED = {
    "run_record_pv_ee.sh",
    "run_deploy_pv_corrections.sh",
    "run_pv_pad.sh",
    "run_carton_fixed_grip_trials.sh",
}


def test_the_gpu_policy_covers_every_wrapper():
    assert CPU_ONLY | NEEDS_GPU | HARDWARE_GATED == {p.name for p in WRAPPERS}


@pytest.mark.parametrize("name", sorted(HARDWARE_GATED))
def test_a_hardware_gated_wrapper_refuses_before_it_reaches_hardware(name):
    """Each of these checks its preconditions before opening a camera, a serial
    bus or a socket. Running one with nothing configured must fail loudly, not
    start something."""
    env = dict(os.environ)
    for var in ("PV_LEVELS", "GRIP_LIGHT_POS", "GRIP_HARD_POS"):
        env.pop(var, None)
    out = subprocess.run(
        ["bash", str(SCRIPTS / name)], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.returncode != 0, f"{name} started with nothing configured"


def _cuda_visible_devices(wrapper, tmp_path):
    """What CUDA_VISIBLE_DEVICES is when the program is finally exec'd.

    Runs the wrapper for real, with SO101_PYTHON pointing at a stub that reports
    the variable instead of starting Python. Sourcing the wrapper in-process does
    not work: it re-sources _common.sh and would replace any stubbed run_module.
    """
    stub = tmp_path / "fake-python"
    stub.write_text('#!/usr/bin/env bash\necho "CVD=[${CUDA_VISIBLE_DEVICES-<unset>}]"\n')
    stub.chmod(0o755)
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SO101_PYTHON"] = str(stub)
    out = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize("name", sorted(NEEDS_GPU))
def test_policy_runners_do_not_hide_the_gpu(name, tmp_path):
    assert _cuda_visible_devices(SCRIPTS / name, tmp_path) == "CVD=[<unset>]", (
        f"{name} masks the GPU; a policy would fall back to CPU or fail to load"
    )


@pytest.mark.parametrize("name", sorted(CPU_ONLY))
def test_cpu_only_runners_hide_the_gpu(name, tmp_path):
    assert _cuda_visible_devices(SCRIPTS / name, tmp_path) == "CVD=[]"


# --- which wrappers move the arm ---
# view_camera.sh used to run `teleop_viz`, which drives the arm, while its own
# comment said "no arm motion". Someone opening it for a passive preview would
# have got a moving robot. The two are separate wrappers now, and this keeps
# them separate.
PASSIVE = {"view_camera.sh"}


ARM_DRIVING_MODULES = {"teleop_viz", "record_so101_ee", "deploy_so101_ee", "diagnose_deploy"}


def _launched_module(body):
    """The module the wrapper actually execs, ignoring what its comments mention."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("run_module "):
            return stripped.split()[1]
    return None


@pytest.mark.parametrize("name", sorted(PASSIVE))
def test_a_passive_wrapper_does_not_launch_an_arm_driving_program(name):
    body = (SCRIPTS / name).read_text(encoding="utf-8")

    assert _launched_module(body) not in ARM_DRIVING_MODULES
    assert "NO ARM MOTION" in body


def test_the_split_wrappers_launch_different_programs():
    """The bug this pair exists for: one wrapper, two jobs, and the dangerous
    one hidden behind the safe one's name."""
    passive = _launched_module((SCRIPTS / "view_camera.sh").read_text(encoding="utf-8"))
    driving = _launched_module((SCRIPTS / "run_teleop_viz.sh").read_text(encoding="utf-8"))

    assert passive == "view_camera"
    assert driving == "teleop_viz"


def test_the_arm_driving_wrappers_say_so():
    for name in {"run_teleop_viz.sh", "run_arm_ee.sh", "run_record_pv_ee.sh",
                 "run_deploy_grip_ee.sh"}:
        body = (SCRIPTS / name).read_text(encoding="utf-8").lower()
        assert "e-stop" in body or "the arm moves" in body, name
