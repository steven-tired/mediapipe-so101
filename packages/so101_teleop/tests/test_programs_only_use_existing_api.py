"""Programs may only use attributes their collaborators actually have.

The recorder was reconstructed from a worktree whose `WebcamEEController` was
five times larger. Three attributes came along that the public controller does
not define — `close`, `middle_gesture_active`, `middle_gesture_seen`. Nothing
caught it: import succeeds, 377 tests pass, and the failure only appears once a
real robot is connected and `_run_recording` gets far enough to touch them.

This walks the AST for attribute access on names bound to a known class and
checks each one against the real class, so the same class of dangling reference
fails in CI instead of on the bench.
"""

import ast
from pathlib import Path

import pytest

from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController

PROGRAMS = Path(__file__).parents[1] / "src" / "lerobot_teleoperator_so101_webcam" / "programs"

#: local variable / attribute name -> the class it is bound to in these programs
BINDINGS = {"controller": WebcamEEController, "_ctl": WebcamEEController}


def attribute_uses(path):
    """(binding, attribute, lineno) for every `<binding>.<attr>` in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        target = node.value
        name = (target.id if isinstance(target, ast.Name)
                else target.attr if isinstance(target, ast.Attribute) else None)
        if name in BINDINGS:
            yield name, node.attr, node.lineno


def instance_attributes(cls):
    """Class members plus anything __init__ assigns to self."""
    names = set(dir(cls))
    tree = ast.parse(__import__("inspect").getsource(cls))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self" and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
    return names


PROGRAM_FILES = sorted(PROGRAMS.glob("*.py"))


def test_there_are_programs_to_check():
    assert len(PROGRAM_FILES) >= 10


def probed_names(path):
    """Attributes the file explicitly probes with hasattr/getattr.

    Declaring an attribute optional once makes it optional for the whole file:
    the guard and the use are usually a few lines apart, inside the same branch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in {"hasattr", "getattr"} and len(node.args) >= 2 \
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            names.add(node.args[1].value)
    return names


@pytest.mark.parametrize("path", PROGRAM_FILES, ids=lambda p: p.name)
def test_program_uses_only_real_controller_attributes(path):
    optional = probed_names(path)
    dangling = [
        f"{path.name}:{lineno} {binding}.{attr} not on {BINDINGS[binding].__name__}"
        for binding, attr, lineno in attribute_uses(path)
        if attr not in instance_attributes(BINDINGS[binding]) and attr not in optional
    ]
    assert not dangling, "\n".join(dangling)
