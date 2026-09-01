"""The public SO-101 core must not carry or reference private IR/thermal code."""

import re
from pathlib import Path

IR_NAME = re.compile(r"(^|[^a-z])ir([^a-z]|$)|rep08", re.IGNORECASE)
IR_IMPORT = re.compile(r"\b(flir|lepton|thermal_project|ir_force)\b", re.IGNORECASE)

ROOT = Path(__file__).parents[1] / "src" / "lerobot_teleoperator_so101_webcam"


def test_so101_core_has_no_ir_modules():
    # A `startswith("ir_")` check would be vacuous here: no file in the source
    # tree is named that way. The IR programs are analyze_ir_*, record_ir_*, etc.
    leaked = [p.name for p in ROOT.rglob("*.py") if IR_NAME.search(p.stem)]
    assert not leaked, leaked


def test_so101_core_does_not_import_private_ir_code():
    for path in ROOT.rglob("*.py"):
        hits = set(IR_IMPORT.findall(path.read_text(encoding="utf-8")))
        assert not hits, f"{path.name} references {hits}"


def test_so101_core_has_no_developer_home_paths():
    for path in ROOT.rglob("*.py"):
        assert "/hand-teleop" not in path.read_text(encoding="utf-8"), path.name
