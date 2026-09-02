"""Every module this repo names must be a module this repo owns.

Three stale editable installs in the LeRobot venv point at the pre-migration
`webcam-input/` tree, and setuptools' `_EditableFinder` sits *after* `PathFinder`
on `sys.meta_path`. That makes them a silent fallback: a submodule the migration
forgot is served from the old tree instead of raising. The PV recorder imported
`lerobot_teleoperator_so101_webcam.ir_capture`, which this repo does not have,
and the failure waited until a robot was connected and an episode was starting.

Stripping the finders at interpreter start does not hold -- they get reinstalled
during import -- so the check is static: resolve every intra-repo import against
the files on disk. It answers "is this repo complete", not "what did the venv
happen to supply", which is the only question that survives retiring the old
tree.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]

#: top-level package -> the directory that must contain its submodules
OWNED = {
    "lerobot_teleoperator_so101_webcam": REPO / "packages/so101_teleop/src/lerobot_teleoperator_so101_webcam",
    "webcam_input": REPO / "packages/webcam_input/src/webcam_input",
    "lerobot_policy_grip_aux": REPO / "packages/policy_grip_aux/src/lerobot_policy_grip_aux",
    "pressurevision_integration": REPO / "integrations/pressurevision/src/pressurevision_integration",
}

SOURCE_FILES = sorted(
    path
    for directory in ("packages", "integrations", "scripts")
    for path in (REPO / directory).rglob("*.py")
)


def imported_submodules(path):
    """(dotted module, lineno) for each `<owned package>.<sub>` the file imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module
        elif isinstance(node, ast.Import):
            module = node.names[0].name
        else:
            continue
        if module and module.split(".")[0] in OWNED and "." in module:
            yield module, node.lineno


def test_there_are_source_files_to_check():
    assert len(SOURCE_FILES) >= 50


def test_every_owned_package_directory_exists():
    """Guard the guard: a mistyped path here would pass everything vacuously."""
    for name, directory in OWNED.items():
        assert directory.is_dir(), f"{name} -> {directory} is not a directory"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_intra_repo_imports_resolve_inside_this_repo(path):
    missing = []
    for module, lineno in imported_submodules(path):
        top, sub = module.split(".")[0], module.split(".")[1]
        base = OWNED[top]
        if not ((base / f"{sub}.py").exists() or (base / sub).is_dir()):
            missing.append(
                f"{path.relative_to(REPO)}:{lineno} imports {module}, which this repo "
                f"does not contain ({base / sub}.py missing) -- it would be served by a "
                "stale editable install of the pre-migration tree"
            )
    assert not missing, "\n".join(missing)
