"""Direct joint mapping: human hand (webcam VR-frame pose + MANO landmarks) -> SO-101 joints.

Pure functions only (no camera, no robot) so they are unit-testable.
"""

import numpy as np
from scipy.spatial.transform import Rotation

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY_MOTORS = MOTORS[:5]

_THUMB_TIP = 4
_INDEX_TIP = 8


def retarget(position, quaternion, landmarks, config) -> dict[str, float]:
    """Map a right-hand pose to 6 SO-101 joint targets (raw, pre-clamp except gripper).

    Args:
        position: (3,) VR-frame meters [x=+right, y=+up, z=+depth].
        quaternion: (4,) [x, y, z, w] VR-frame wrist orientation.
        landmarks: (21, 3) MANO joint_pos (for thumb-index pinch).
        config: SO101WebcamConfig with scale/offset fields.
    """
    position = np.asarray(position, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)

    roll, pitch, _yaw = Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_euler("xyz")

    pinch = float(np.linalg.norm(landmarks[_THUMB_TIP] - landmarks[_INDEX_TIP]))
    span = max(config.grip_pinch_max - config.grip_pinch_min, 1e-6)
    grip = (pinch - config.grip_pinch_min) / span * 100.0
    grip = float(np.clip(grip, 0.0, 100.0))

    shoulder_lift = config.lift_offset + config.lift_scale * position[1]
    elbow_flex = config.elbow_offset + config.elbow_scale * position[2]
    # Auto-keep the gripper pointing down: cancel the pitch contributed by lift+elbow so the
    # operator can lower onto an object without it tilting off-target. `pitch` adds optional
    # manual tilt on top (wrist_flex_scale defaults to 0 = pure auto-down).
    wrist_flex = (config.wrist_flex_offset
                  - config.wrist_flex_couple * (shoulder_lift + elbow_flex)
                  + config.wrist_flex_scale * pitch)

    return {
        "shoulder_pan.pos": config.pan_offset + config.pan_scale * position[0],
        "shoulder_lift.pos": shoulder_lift,
        "elbow_flex.pos": elbow_flex,
        "wrist_flex.pos": wrist_flex,
        "wrist_roll.pos": config.wrist_roll_offset + config.wrist_roll_scale * roll,
        "gripper.pos": grip,
    }


def hand_roll(quaternion) -> float:
    """Roll (rad) of the hand, for differential wrist_roll latching."""
    roll, _pitch, _yaw = Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_euler("xyz")
    return float(roll)


def retarget_delta(position, quaternion, landmarks, position_ref, roll_ref, arm_ref, config) -> dict[str, float]:
    """DIFFERENTIAL mapping: command joints relative to a latched arm pose + hand reference.

    Absolute mapping (retarget) ties the arm to the hand's absolute frame position, which drives
    the arm into the table for many hand poses. Here the arm starts from `arm_ref` (its pose when
    the clutch engaged) and moves only by the hand's motion since `position_ref`/`roll_ref`, so at
    the latch instant the target equals the current pose (no jump) and small hand motion -> small,
    anchored arm motion. Gripper stays ABSOLUTE (pinch). Matches franka-vr's differential control.

    Args:
        position, quaternion, landmarks: current hand pose + MANO landmarks.
        position_ref: (3,) hand position at the latch.
        roll_ref: hand roll (rad) at the latch.
        arm_ref: dict of the 6 latched joint targets (the arm's pose when engaged).
    """
    d = np.asarray(position, dtype=float) - np.asarray(position_ref, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)

    pinch = float(np.linalg.norm(landmarks[_THUMB_TIP] - landmarks[_INDEX_TIP]))
    span = max(config.grip_pinch_max - config.grip_pinch_min, 1e-6)
    grip = float(np.clip((pinch - config.grip_pinch_min) / span * 100.0, 0.0, 100.0))

    d_lift = config.lift_scale * d[1]
    d_elbow = config.elbow_scale * d[2]
    # Keep-gripper-down: wrist_flex cancels the pitch from the lift/elbow CHANGE since the latch,
    # so if the gripper was pointing down when engaged it stays down as the arm moves.
    d_wrist_flex = -config.wrist_flex_couple * (d_lift + d_elbow)

    return {
        "shoulder_pan.pos": arm_ref["shoulder_pan.pos"] + config.pan_scale * d[0],
        "shoulder_lift.pos": arm_ref["shoulder_lift.pos"] + d_lift,
        "elbow_flex.pos": arm_ref["elbow_flex.pos"] + d_elbow,
        "wrist_flex.pos": arm_ref["wrist_flex.pos"] + d_wrist_flex,
        "wrist_roll.pos": arm_ref["wrist_roll.pos"] + config.wrist_roll_scale * (hand_roll(quaternion) - roll_ref),
        "gripper.pos": grip,
    }


REST_ACTION: dict[str, float] = {f"{m}.pos": 0.0 for m in MOTORS}


def clamp_joints(action: dict[str, float]) -> dict[str, float]:
    out = {}
    for m in BODY_MOTORS:
        out[f"{m}.pos"] = float(np.clip(action[f"{m}.pos"], -100.0, 100.0))
    out["gripper.pos"] = float(np.clip(action["gripper.pos"], 0.0, 100.0))
    return out


def ema(target: dict[str, float], prev: dict[str, float], alpha: float) -> dict[str, float]:
    return {k: alpha * target[k] + (1.0 - alpha) * prev[k] for k in target}


def rate_limit(target: dict[str, float], prev: dict[str, float], max_delta: float) -> dict[str, float]:
    out = {}
    for k, v in target.items():
        lo, hi = prev[k] - max_delta, prev[k] + max_delta
        out[k] = float(np.clip(v, lo, hi))
    return out
