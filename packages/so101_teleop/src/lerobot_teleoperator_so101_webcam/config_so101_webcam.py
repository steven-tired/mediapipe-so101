from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_webcam")
@dataclass(kw_only=True)
class SO101WebcamConfig(TeleoperatorConfig):
    # Capture / source
    camera_index: int = 0
    workspace_size_m: float = 0.4

    # Safety / smoothing
    smoothing: float = 0.3      # EMA alpha in (0, 1]; lower = smoother
    max_delta: float = 8.0      # max change per joint per get_action() call

    # Direct joint mapping: out = clamp(offset + scale * signal)
    # Body joints -> [-100, 100]; signals are VR-frame meters / radians.
    pan_scale: float = 500.0     # position.x (+right)  -> shoulder_pan
    pan_offset: float = 0.0
    lift_scale: float = 500.0    # position.y (+up)     -> shoulder_lift
    lift_offset: float = 0.0
    elbow_scale: float = 200.0   # position.z (+depth)  -> elbow_flex
    elbow_offset: float = -100.0  # nominal z~0.5 -> 0
    # Keep-gripper-down coupling (FK-derived): the SO-101 gripper pitch ~ shoulder_lift +
    # elbow_flex + wrist_flex, and at the calibration middle it already points ~down. So
    # wrist_flex = offset - couple*(shoulder_lift + elbow_flex) holds the gripper pointing
    # down as the arm lowers/reaches -> picking works WITHOUT IK or noise amplification.
    # couple ~= 1 in normalized units (lift/elbow/wrist have near-equal travel).
    wrist_flex_couple: float = 1.0
    wrist_flex_offset: float = 0.0   # rest down-pitch trim (+ tilts nose up)
    wrist_flex_scale: float = 0.0    # manual hand-pitch contribution (0 = pure auto-down)
    wrist_roll_scale: float = 64.0   # roll (rad)  -> wrist_roll
    wrist_roll_offset: float = 0.0

    # Gripper from thumb-index pinch distance (MANO units) -> [0, 100]
    grip_pinch_min: float = 0.02   # closed
    grip_pinch_max: float = 0.12   # open
