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
        "run_deploy_ee.sh", "run_deploy_grip_ee.sh", "run_diagnose.sh",
        "view_camera.sh",
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
CPU_ONLY = {"run_arm_ee.sh", "run_record_ee.sh", "view_camera.sh"}
NEEDS_GPU = {"run_deploy_ee.sh", "run_deploy_grip_ee.sh", "run_diagnose.sh"}
# A wrapper that starts two processes with opposite needs cannot answer this
# question once for the whole script: the PV recorder is CPU-only, its sender
# runs the PressureVision network. It hides the GPU per-invocation instead,
# which test_record_so101_pv_ee checks at the source level -- running it for
# real needs a fitted levels.json and two cameras.
MIXED_GPU = {"run_record_pv_ee.sh"}


def test_the_gpu_policy_covers_every_wrapper():
    assert CPU_ONLY | NEEDS_GPU | MIXED_GPU == {p.name for p in WRAPPERS}


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
