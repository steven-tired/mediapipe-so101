"""`start()` and `start_oak()` must leave the object in the same shape.

`latest_frame()` read `self._latest_frame`, which only `start_oak()` created. The
monocular path therefore raised AttributeError the first time a caller asked for
a preview frame — after the robot was connected and an episode was about to
start. Import checks and 390 unit tests all passed.

These tests compare the two start paths structurally, without a camera.
"""

import ast
import inspect
from pathlib import Path

import pytest

from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator
from webcam_input.depth import ScaleDepthStrategy


def assigned_self_attributes(func):
    source = inspect.getsource(func)
    tree = ast.parse("\n".join(line[4:] if line.startswith("    ") else line
                               for line in source.splitlines()))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self" and isinstance(node.ctx, ast.Store)
    }


def read_self_attributes(func):
    source = inspect.getsource(func)
    tree = ast.parse("\n".join(line[4:] if line.startswith("    ") else line
                               for line in source.splitlines()))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self" and isinstance(node.ctx, ast.Load)
    }


def test_public_readers_only_touch_attributes_init_creates():
    """Whatever latest()/latest_frame() read must exist straight after __init__."""
    created = assigned_self_attributes(WebcamSource.__init__)
    for reader in (WebcamSource.latest, WebcamSource.latest_frame):
        needed = read_self_attributes(reader)
        missing = sorted(a for a in needed if a not in created and not hasattr(WebcamSource, a))
        assert not missing, f"{reader.__name__} reads {missing}, which __init__ never creates"


def test_latest_frame_is_safe_before_any_start():
    source = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=0.3))
    assert source.latest_frame() is None


@pytest.mark.parametrize("attr", ["_latest_frame", "_mp_draw", "_mp_conns"])
def test_preview_state_exists_before_any_start(attr):
    source = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=0.3))
    assert hasattr(source, attr), f"{attr} is created by a start path, not by __init__"


def test_both_capture_loops_publish_a_preview_frame():
    """A preview that only works with an OAK makes the monocular mode unusable."""
    for loop in (WebcamSource._loop, WebcamSource._loop_oak):
        assert "_latest_frame" in assigned_self_attributes(loop), \
            f"{loop.__name__} never publishes a preview frame"
