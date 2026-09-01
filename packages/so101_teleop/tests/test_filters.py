from lerobot_teleoperator_so101_webcam.control import (
    REST_ACTION,
    clamp_joints,
    ema,
    rate_limit,
)


def test_rest_action_keys_and_values():
    assert REST_ACTION["gripper.pos"] == 0.0
    assert all(v == 0.0 for v in REST_ACTION.values())


def test_clamp_body_and_gripper_bounds():
    raw = {
        "shoulder_pan.pos": 9999.0, "shoulder_lift.pos": -9999.0,
        "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0, "gripper.pos": 250.0,
    }
    out = clamp_joints(raw)
    assert out["shoulder_pan.pos"] == 100.0
    assert out["shoulder_lift.pos"] == -100.0
    assert out["gripper.pos"] == 100.0


def test_ema_is_between_prev_and_target():
    prev = {"shoulder_pan.pos": 0.0}
    target = {"shoulder_pan.pos": 10.0}
    out = ema(target, prev, alpha=0.3)
    assert out["shoulder_pan.pos"] == 3.0


def test_rate_limit_caps_step():
    prev = {"shoulder_pan.pos": 0.0}
    target = {"shoulder_pan.pos": 100.0}
    out = rate_limit(target, prev, max_delta=8.0)
    assert out["shoulder_pan.pos"] == 8.0
