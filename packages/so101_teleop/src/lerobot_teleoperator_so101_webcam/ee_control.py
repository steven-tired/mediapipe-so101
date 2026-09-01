"""Pure mapping: webcam wrist displacement + pinch + clutch -> SO-101 EE-delta action.

The output dict is consumed by LeRobot's EEReferenceAndDelta processor. Orientation
deltas are zero by default. The live EE controller may supply one bounded
tool-axis roll delta while keeping the gripper pointed down.
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


# --- Pinch -> command mapping ---
# The validated map shifts toward closed by GRIP_OVERDRIVE and clips at zero, so
# a moderate pinch already commands a full clamp. The clip also flattens the
# bottom of the pinch range onto a single command: with the default 0.02-0.12
# range and an overdrive of 18, every pinch from 0.020 to 0.038 commands 0. A
# partial loosening near a firm grip is therefore not representable at all --
# before any ratchet blocks it.
#
# Span mode buys that resolution back by narrowing the pinch range instead of
# clipping it, keeping the map monotone all the way down. The resolution has to
# come from somewhere: a full clamp then needs a nearly-closed pinch rather than
# a moderate one. Which trade is right is a question for the hand, not the code.
GRIP_SPAN_PINCH_MAX = 0.10


def raw_grip_from_pinch(pinch, cfg, *, grip_map="overdrive", overdrive=0.0):
    """Thumb-index pinch distance -> gripper command in [0, 100], pre-smoothing."""
    if grip_map == "span":
        span = max(GRIP_SPAN_PINCH_MAX - cfg.grip_pinch_min, 1e-6)
        scaled = (float(pinch) - cfg.grip_pinch_min) / span * 100.0
        return float(np.clip(scaled, 0.0, 100.0))
    return max(0.0, gripper_pos_from_pinch(pinch, cfg) - float(overdrive))


# --- Grip ratchet (arm B of the grip-mode comparison) ---
# Below this command the operator has committed to a grasp; above the exit the
# hand is deliberately open. The gap between them is wide on purpose, and the
# thresholds are asymmetric for a specific reason: MediaPipe degrades as the
# hand closes and the fingers occlude each other, and recovers as it opens. So
# the release signal is reliable exactly when the grip signal is not.
GRIP_LATCH_ENTER = 30.0
GRIP_LATCH_EXIT = 65.0
# A single spurious "open" frame must not release a grasp mid-lift.
GRIP_LATCH_EXIT_FRAMES = 5
# Opening weight for the ratchet. The tracked arm uses a slow 0.15 to *resist*
# loosening; the ratchet blocks loosening outright, so keeping that value here
# is a redundant mechanism that only adds lag -- measured on foam as a release
# that badly lags the hand. Open at the closing rate instead.
GRIP_RELEASE_ALPHA = 0.7


def grip_ratchet(raw_grip, smoothed, latched, open_frames, *,
                 close_alpha, open_alpha):
    """Asymmetric grip EMA with an explicit ratchet; returns the new state.

    Returns (command, latched, open_frames). The ratchet is the same asymmetric
    EMA the tracked mode already uses, with the opening weight taken to zero
    once the operator has committed to a grasp: closing still tracks, because
    squeezing harder is deliberate and readable, while loosening is blocked
    until an explicit, debounced release.

    This exists to test whether the loosening compensations in the tracked path
    (a slow-open EMA plus a fixed overdrive) are needed at all, or whether the
    failure they paper over is simply that the command follows a signal that
    degrades during the grasp.
    """
    if smoothed is None:
        return float(raw_grip), False, 0

    if latched:
        open_frames = open_frames + 1 if raw_grip > GRIP_LATCH_EXIT else 0
        if open_frames >= GRIP_LATCH_EXIT_FRAMES:
            latched, open_frames = False, 0
        else:
            open_alpha = 0.0

    alpha = close_alpha if raw_grip < smoothed else open_alpha
    command = alpha * float(raw_grip) + (1 - alpha) * float(smoothed)

    # Latch while closing or steady, never while opening. Testing the command
    # alone re-latches on the frame after a release, because the command leaves
    # a released grasp near zero and climbs out slowly -- it is still below the
    # enter threshold while the hand is already open. The comparison is
    # non-strict so a grasp that is already steady at full clamp still latches;
    # requiring strict closing never fires once the command has settled.
    if not latched and raw_grip <= smoothed and command < GRIP_LATCH_ENTER:
        latched = True

    return float(command), bool(latched), int(open_frames)


def ee_action_from_hand(displacement, pinch, enabled, cfg, *, roll_delta=0.0) -> dict:
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
        "target_wz": float(roll_delta) if enabled else 0.0,
        "gripper_vel": gripper_vel_from_pinch(pinch, cfg) if enabled else 0.0,
    }
