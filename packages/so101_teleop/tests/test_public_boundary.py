"""The public SO-101 core must not carry or reference private IR/thermal code."""

import re
from pathlib import Path

IR_NAME = re.compile(r"(^|[^a-z])ir([^a-z]|$)|rep08", re.IGNORECASE)
IR_IMPORT = re.compile(r"\b(flir|lepton|thermal_project|ir_force)\b", re.IGNORECASE)

ROOT = Path(__file__).parents[1] / "src" / "lerobot_teleoperator_so101_webcam"

REPO = Path(__file__).parents[3]
#: Every tracked Python file in the repo, not just the core package. The
#: developer-path check below used to scan only ROOT, and two absolute
#: /home/zhuokai paths walked in through `integrations/.../tools/` and
#: `research/` during the PV migration without tripping anything.
#: Assembled rather than written literally: a check for the workspace name is
#: itself a file containing the workspace name, and the first draft of this test
#: failed on its own source.
WORKSPACE_MARKER = "/" + "hand" + "-teleop"

PUBLISHED_TREES = [
    REPO / "packages",
    REPO / "integrations",
    REPO / "research",
    REPO / "scripts",
]


def published_python_files():
    for tree in PUBLISHED_TREES:
        if not tree.is_dir():
            continue
        for path in tree.rglob("*.py"):
            if "__pycache__" in path.parts or "local" in path.parts:
                continue
            yield path


def test_so101_core_has_no_ir_modules():
    # A `startswith("ir_")` check would be vacuous here: no file in the source
    # tree is named that way. The IR programs are analyze_ir_*, record_ir_*, etc.
    leaked = [p.name for p in ROOT.rglob("*.py") if IR_NAME.search(p.stem)]
    assert not leaked, leaked


def test_so101_core_does_not_import_private_ir_code():
    for path in ROOT.rglob("*.py"):
        hits = set(IR_IMPORT.findall(path.read_text(encoding="utf-8")))
        assert not hits, f"{path.name} references {hits}"


def test_nothing_published_carries_a_developer_home_path():
    offenders = [
        str(path.relative_to(REPO))
        for path in published_python_files()
        if WORKSPACE_MARKER in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_the_published_scan_actually_reaches_the_trees_it_names():
    """Guards the guard: an empty scan passes the check above vacuously."""
    scanned = {path.relative_to(REPO).parts[0] for path in published_python_files()}
    assert {"packages", "integrations", "research"} <= scanned, scanned
