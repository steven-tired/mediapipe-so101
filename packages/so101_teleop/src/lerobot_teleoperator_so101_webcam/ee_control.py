"""Pure mapping: webcam wrist displacement + pinch + clutch -> SO-101 EE-delta action.

The output dict is consumed by LeRobot's EEReferenceAndDelta processor. Orientation
deltas are zero in v1 (gripper holds the orientation latched at clutch-engage).
No camera, no robot -> unit-testable.
"""

import numpy as np

EE_ACTION_KEYS = (
    "enabled", "target_x", "target_y", "target_z",
    "target_wx", "target_wy", "target_wz", "gripper_vel",
)


def gripper_vel_from_pinch(pinch: float, cfg) -> float:
    """Signed gripper velocity in [-1, 1]; >0 closes (tight pinch), <0 opens."""
    mid = 0.5 * (cfg.grip_pinch_min + cfg.grip_pinch_max)
    half_range = max(0.5 * (cfg.grip_pinch_max - cfg.grip_pinch_min), 1e-6)
    vel = cfg.grip_sign * (mid - float(pinch)) / half_range
    return float(np.clip(vel, -1.0, 1.0))


def joint_center(norm_mode_value: str) -> float:
    """Calibration-midpoint joint value, in the units send_action expects.

    For SO-101 with use_degrees=True the body joints normalize as DEGREES, whose
    mid-range maps to 0 deg (motors_bus._normalize); the gripper is RANGE_0_100 whose
    centre is 50. RANGE_M100_100 (use_degrees=False) also centres at 0. So this is the
    safe "middle position" = geometric centre of each joint's calibrated travel.
    """
    return 50.0 if str(norm_mode_value) == "range_0_100" else 0.0


def gripper_pos_from_pinch(pinch: float, cfg) -> float:
    """ABSOLUTE gripper position in [0, 100] from thumb-index pinch distance.

    Tight pinch -> 0 (closed), open hand -> 100 (open). This is the validated joint-path
    mapping (control.retarget). Use this instead of GripperVelocityToJoint's *integrated*
    velocity, which drifts to the rail under any steady pinch bias.
    """
    span = max(cfg.grip_pinch_max - cfg.grip_pinch_min, 1e-6)
    grip = (float(pinch) - cfg.grip_pinch_min) / span * 100.0
    return float(np.clip(grip, 0.0, 100.0))


def ee_action_from_hand(displacement, pinch, enabled, cfg) -> dict:
    """displacement: (3,) webcam VR meters since the latched reference (x right, y up, z depth).

    When `enabled` is False (left-fist clutch engaged or hand tracking lost), gripper velocity
    is frozen at 0.0 rather than derived from `pinch` -- otherwise a zeroed/garbage pinch reading
    would be interpreted as a tight pinch and drive the gripper closed while motion is disabled.
    """
    d = np.asarray(displacement, dtype=float)
    return {
        "enabled": bool(enabled),
        "target_x": cfg.ee_x_sign * float(d[cfg.ee_x_idx]),
        "target_y": cfg.ee_y_sign * float(d[cfg.ee_y_idx]),
        "target_z": cfg.ee_z_sign * float(d[cfg.ee_z_idx]),
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": gripper_vel_from_pinch(pinch, cfg) if enabled else 0.0,
    }
