import numpy as np

from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
from lerobot_teleoperator_so101_webcam.control import (
    MOTORS,
    REST_ACTION,
    hand_roll,
    retarget,
    retarget_delta,
)

IDENT_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _landmarks(pinch: float) -> np.ndarray:
    lm = np.zeros((21, 3))
    lm[4] = np.array([pinch, 0.0, 0.0])   # thumb tip
    lm[8] = np.array([0.0, 0.0, 0.0])     # index tip
    return lm


def test_returns_all_six_motor_keys():
    cfg = SO101WebcamConfig()
    out = retarget(np.zeros(3), IDENT_QUAT, _landmarks(0.07), cfg)
    assert set(out) == {f"{m}.pos" for m in MOTORS}


def test_hand_right_increases_shoulder_pan():
    cfg = SO101WebcamConfig()
    left = retarget(np.array([-0.1, 0, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    right = retarget(np.array([0.1, 0, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    assert right["shoulder_pan.pos"] > left["shoulder_pan.pos"]


def test_hand_up_increases_shoulder_lift():
    cfg = SO101WebcamConfig()
    down = retarget(np.array([0, -0.1, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    up = retarget(np.array([0, 0.1, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    assert up["shoulder_lift.pos"] > down["shoulder_lift.pos"]


def test_wrist_flex_couples_to_keep_gripper_down():
    # wrist_flex must cancel the pitch from shoulder_lift + elbow_flex so the gripper holds
    # its downward orientation as the arm lowers/reaches (FK: pitch ~ lift + elbow + wrist_flex).
    cfg = SO101WebcamConfig()
    out = retarget(np.array([0.0, 0.1, 0.6]), IDENT_QUAT, _landmarks(0.07), cfg)
    expected = cfg.wrist_flex_offset - cfg.wrist_flex_couple * (
        out["shoulder_lift.pos"] + out["elbow_flex.pos"])
    assert abs(out["wrist_flex.pos"] - expected) < 1e-9
    # raising the hand (more lift) drives wrist_flex the opposite way to stay pointed down
    low = retarget(np.array([0.0, -0.1, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    high = retarget(np.array([0.0, 0.1, 0.5]), IDENT_QUAT, _landmarks(0.07), cfg)
    assert high["shoulder_lift.pos"] > low["shoulder_lift.pos"]
    assert high["wrist_flex.pos"] < low["wrist_flex.pos"]


def test_delta_no_jump_at_latch():
    # At the latch instant (hand at reference) the differential target must equal the latched
    # arm pose -> no jump (this is what stops the arm diving to an absolute pose).
    cfg = SO101WebcamConfig()
    arm_ref = {"shoulder_pan.pos": 12.0, "shoulder_lift.pos": -30.0, "elbow_flex.pos": 25.0,
               "wrist_flex.pos": 5.0, "wrist_roll.pos": -8.0, "gripper.pos": 40.0}
    pos_ref = np.array([0.05, -0.10, 0.55])
    out = retarget_delta(pos_ref, IDENT_QUAT, _landmarks(0.07), pos_ref, hand_roll(IDENT_QUAT), arm_ref, cfg)
    for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        assert abs(out[f"{m}.pos"] - arm_ref[f"{m}.pos"]) < 1e-9


def test_delta_moves_relative_and_keeps_down():
    cfg = SO101WebcamConfig()
    arm_ref = dict(REST_ACTION)
    pos_ref = np.array([0.0, 0.0, 0.5])
    # lower the hand (y down) -> shoulder_lift drops, wrist_flex compensates UP to stay pointed down
    out = retarget_delta(np.array([0.0, -0.1, 0.5]), IDENT_QUAT, _landmarks(0.07), pos_ref,
                         hand_roll(IDENT_QUAT), arm_ref, cfg)
    assert out["shoulder_lift.pos"] < arm_ref["shoulder_lift.pos"]
    expected_wf = -cfg.wrist_flex_couple * (out["shoulder_lift.pos"] - arm_ref["shoulder_lift.pos"])
    assert abs(out["wrist_flex.pos"] - expected_wf) < 1e-9


def test_wider_pinch_opens_gripper():
    cfg = SO101WebcamConfig()
    closed = retarget(np.zeros(3), IDENT_QUAT, _landmarks(cfg.grip_pinch_min), cfg)
    opened = retarget(np.zeros(3), IDENT_QUAT, _landmarks(cfg.grip_pinch_max), cfg)
    assert closed["gripper.pos"] < opened["gripper.pos"]
    assert 0.0 <= closed["gripper.pos"] <= 100.0
    assert 0.0 <= opened["gripper.pos"] <= 100.0
