import numpy as np
from dataclasses import dataclass

from lerobot_teleoperator_so101_webcam.ee_control import (
    EE_ACTION_KEYS,
    ee_action_from_hand,
    gripper_pos_from_pinch,
    gripper_vel_from_pinch,
    joint_center,
)


@dataclass
class Cfg:
    ee_x_idx: int = 0; ee_x_sign: float = 1.0
    ee_y_idx: int = 1; ee_y_sign: float = 1.0
    ee_z_idx: int = 2; ee_z_sign: float = 1.0
    grip_pinch_min: float = 0.02; grip_pinch_max: float = 0.12; grip_sign: float = 1.0


def test_keys_present_and_orientation_zero():
    a = ee_action_from_hand(np.zeros(3), 0.07, True, Cfg())
    assert set(a) == set(EE_ACTION_KEYS)
    assert a["target_wx"] == 0.0 and a["target_wy"] == 0.0 and a["target_wz"] == 0.0
    assert a["enabled"] is True


def test_displacement_maps_to_targets_with_sign():
    a = ee_action_from_hand(np.array([0.1, -0.2, 0.3]), 0.07, True, Cfg())
    assert a["target_x"] == 0.1
    assert a["target_y"] == -0.2
    assert a["target_z"] == 0.3
    b = ee_action_from_hand(np.array([0.1, 0, 0]), 0.07, True, Cfg(ee_x_sign=-1.0))
    assert b["target_x"] == -0.1


def test_pinch_closes_open_opens():
    cfg = Cfg()
    assert gripper_vel_from_pinch(cfg.grip_pinch_min, cfg) > 0   # tight pinch -> close
    assert gripper_vel_from_pinch(cfg.grip_pinch_max, cfg) < 0   # open hand -> open
    assert -1.0 <= gripper_vel_from_pinch(0.5, cfg) <= 1.0       # clipped


def test_disabled_passthrough():
    a = ee_action_from_hand(np.array([0.1, 0.1, 0.1]), 0.07, False, Cfg())
    assert a["enabled"] is False
    assert a["gripper_vel"] == 0.0


def test_disabled_freezes_gripper_even_with_tight_pinch():
    # Regression for the clutch/dropout bug: a zeroed/garbage pinch reading must NOT
    # be interpreted as a tight pinch and drive the gripper closed while disabled.
    cfg = Cfg()
    a = ee_action_from_hand(np.zeros(3), cfg.grip_pinch_min, False, cfg)
    assert a["gripper_vel"] == 0.0


def test_enabled_tight_pinch_still_closes():
    cfg = Cfg()
    a = ee_action_from_hand(np.zeros(3), cfg.grip_pinch_min, True, cfg)
    assert a["gripper_vel"] > 0.0


def test_gripper_pos_absolute_maps_pinch_to_0_100():
    cfg = Cfg()
    assert gripper_pos_from_pinch(cfg.grip_pinch_min, cfg) == 0.0    # tight pinch -> closed
    assert gripper_pos_from_pinch(cfg.grip_pinch_max, cfg) == 100.0  # open hand -> open
    mid = 0.5 * (cfg.grip_pinch_min + cfg.grip_pinch_max)
    assert abs(gripper_pos_from_pinch(mid, cfg) - 50.0) < 1e-6       # midway
    # out-of-range pinch is clipped, never rails past the joint limits
    assert gripper_pos_from_pinch(-1.0, cfg) == 0.0
    assert gripper_pos_from_pinch(1.0, cfg) == 100.0


def test_config_defaults_apply_vr_to_robot_transform():
    # The shipped config defaults must implement LeFranX's VR->robot axis map so that
    # hand height (VR y) drives robot height (EE z) -- otherwise you cannot lower to pick.
    from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
    cfg = SO101WebcamEEConfig()
    vr = np.array([0.11, 0.22, 0.33])  # VR [x=right, y=up, z=depth]
    a = ee_action_from_hand(vr, 0.07, True, cfg)
    assert a["target_x"] == vr[2]    # robot forward  = VR depth
    assert a["target_y"] == -vr[0]   # robot left     = -VR right
    assert a["target_z"] == vr[1]    # robot up       = VR height  (raise hand -> raise gripper)


def test_joint_center_calibration_midpoints():
    # body joints (DEGREES) and RANGE_M100_100 centre at 0; gripper (RANGE_0_100) at 50
    assert joint_center("degrees") == 0.0
    assert joint_center("range_m100_100") == 0.0
    assert joint_center("range_0_100") == 50.0
