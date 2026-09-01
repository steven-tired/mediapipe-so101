"""The public package must not reach into the private IR/thermal project."""

from pathlib import Path

FORBIDDEN = ("ir_pressure", "flir", "lepton", "thermal_project")


def test_webcam_input_has_no_private_ir_imports():
    package_root = Path(__file__).parents[1] / "src" / "webcam_input"
    for path in package_root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        hits = [name for name in FORBIDDEN if name in text]
        assert not hits, f"{path.name} references {hits}"


def test_webcam_input_has_no_developer_home_paths():
    package_root = Path(__file__).parents[1] / "src" / "webcam_input"
    for path in package_root.glob("*.py"):
        assert "hand-teleop" not in path.read_text(encoding="utf-8"), path.name
