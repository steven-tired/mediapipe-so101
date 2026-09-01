"""Every program must import without a configured robot.

Regression guard: a module-level `str(urdf_path())` made importing a program
raise when SO101_URDF was unset, and no test caught it because nothing imported
the programs. Configuration errors belong at run time, not import time.
"""

import importlib
import os
import pkgutil

import pytest

import lerobot_teleoperator_so101_webcam.programs as programs

MODULES = sorted(m.name for m in pkgutil.iter_modules(programs.__path__))


def test_there_are_programs_to_check():
    assert len(MODULES) >= 10


@pytest.mark.parametrize("name", MODULES)
def test_program_imports_without_robot_configuration(name, monkeypatch):
    for var in ("SO101_URDF", "SO_ARM100_DIR", "SO101_DATASET_ROOT",
                "SO101_EVIDENCE_DIR", "SO101_LOCAL_DIR"):
        monkeypatch.delenv(var, raising=False)
    importlib.import_module(f"lerobot_teleoperator_so101_webcam.programs.{name}")


def test_no_program_evaluates_a_path_at_module_level():
    root = programs.__path__[0]
    for name in MODULES:
        source = (os.path.join(root, f"{name}.py"))
        with open(source, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if line.startswith(("URDF_PATH", "DATASET_ROOT")) and "urdf_path()" in line:
                    pytest.fail(f"{name}.py:{lineno} resolves the URDF at import time")
