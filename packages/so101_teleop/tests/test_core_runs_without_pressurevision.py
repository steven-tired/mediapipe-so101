"""The core must import and run with PressureVision absent.

Guards the spec's rule that PressureVision is optional: the public packages may
not acquire a hard dependency on `pressurevision_integration`, and no core module
may import it even lazily at module scope.
"""

import builtins
import importlib
import re
from pathlib import Path

import pytest

CORE = Path(__file__).parents[1] / "src" / "lerobot_teleoperator_so101_webcam"
PV_IMPORT = re.compile(r"^\s*(from|import)\s+pressurevision", re.MULTILINE)


#: The single composition module allowed to know PressureVision exists.
COMPOSE = CORE / "grip" / "compose.py"


def test_only_the_composition_module_names_the_pressurevision_integration():
    offenders = sorted(
        p.relative_to(CORE).as_posix()
        for p in CORE.rglob("*.py")
        if p != COMPOSE and PV_IMPORT.search(p.read_text(encoding="utf-8"))
    )
    assert not offenders, offenders


def test_the_composition_module_imports_pressurevision_lazily():
    """A module-scope import would make PressureVision a hard dependency."""
    source = COMPOSE.read_text(encoding="utf-8")
    matches = list(PV_IMPORT.finditer(source))
    assert matches, "compose.py should be the module that resolves the PV adapter"
    for m in matches:
        line = source[m.start():source.index("\n", m.start())]
        assert line.startswith((" ", "\t")), f"module-scope PV import: {line!r}"


def test_core_imports_with_pressurevision_unimportable(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("pressurevision"):
            raise ImportError(f"blocked for this test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for module in ("ee_controller", "ee_control", "grip.contract", "grip.mediapipe", "paths"):
        importlib.reload(
            importlib.import_module(f"lerobot_teleoperator_so101_webcam.{module}")
        )


def test_default_gripper_needs_no_pressurevision():
    from lerobot_teleoperator_so101_webcam.grip.contract import GripInput
    from lerobot_teleoperator_so101_webcam.grip.mediapipe import MediaPipeGripperController

    c = MediaPipeGripperController()
    command = c.step(
        GripInput(grasp_active=True, explicit_release=False, severity=0.5,
                  valid=True, observed_at_s=0.0),
        actual_pos=50.0,
    )
    assert 0.0 <= command <= 100.0
