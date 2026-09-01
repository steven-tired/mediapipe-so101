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
        "run_arm_ee.sh", "run_record_ee.sh", "run_deploy_ee.sh",
        "run_diagnose.sh", "view_camera.sh",
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
