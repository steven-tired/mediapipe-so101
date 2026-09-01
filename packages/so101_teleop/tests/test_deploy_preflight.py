"""Deployment must refuse a policy it cannot feed, before the arm moves.

The single-camera deploy path was handed a policy trained on front+side. The
mismatch surfaced as `KeyError: 'observation.images.side'` inside the first
inference — after the arm had ramped to the ready pose and started running
autonomously. A camera-set check belongs before `robot.connect`.
"""

import os
import types

import pytest

from lerobot_teleoperator_so101_webcam.programs import deploy_so101_ee as deploy
from lerobot_teleoperator_so101_webcam.programs import record_so101_ee as record


def fake_policy(*image_features):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(image_features=list(image_features))
    )


def test_matching_camera_set_is_accepted():
    deploy._check_policy_cameras(fake_policy("observation.images.front"), {"front": None})


def test_missing_camera_is_refused_with_an_actionable_message():
    with pytest.raises(SystemExit) as excinfo:
        deploy._check_policy_cameras(
            fake_policy("observation.images.front", "observation.images.side"),
            {"front": None},
        )
    message = str(excinfo.value)
    assert "observation.images.side" in message
    assert "single-camera" in message


def test_a_policy_with_no_image_features_is_accepted():
    deploy._check_policy_cameras(fake_policy(), {"front": None})


def test_the_check_runs_before_the_robot_is_connected():
    """Order matters: a late check still moves the arm before failing."""
    import inspect
    source = inspect.getsource(deploy.main)
    assert source.index("_check_policy_cameras") < source.index("robot.connect"), \
        "camera check must precede robot.connect"


def test_deploy_and_record_agree_on_the_workspace_camera(monkeypatch):
    """One repo, one way to point at the workspace camera."""
    assert "SO101_WORKSPACE_CAM" in inspect_source(deploy)
    assert "SO101_WORKSPACE_CAM" in inspect_source(record)


def inspect_source(module):
    import pathlib
    return pathlib.Path(module.__file__).read_text(encoding="utf-8")
