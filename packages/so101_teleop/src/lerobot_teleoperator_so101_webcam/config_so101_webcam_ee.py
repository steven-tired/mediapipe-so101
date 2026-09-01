from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_webcam_ee")
@dataclass(kw_only=True)
class SO101WebcamEEConfig(TeleoperatorConfig):
    camera_index: int = 0
    workspace_size_m: float = 0.4
    # VR(webcam)-frame displacement -> robot-base-frame EE delta, following LeFranX's
    # ArmIKProcessor transform (VR +x=right,+y=up,+z=depth -> robot +x=fwd,+y=left,+z=up):
    #   robot_x (forward) =  VR z (depth)   -> ee_x_idx=2
    #   robot_y (left)    = -VR x (right)   -> ee_y_idx=0, sign -1
    #   robot_z (up)      =  VR y (height)  -> ee_z_idx=1   <- lets the hand raise/lower the gripper
    ee_x_idx: int = 2
    ee_x_sign: float = 1.0
    ee_y_idx: int = 0
    ee_y_sign: float = -1.0
    ee_z_idx: int = 1
    ee_z_sign: float = 1.0
    # gripper from thumb-index pinch (MANO units)
    grip_pinch_min: float = 0.02
    grip_pinch_max: float = 0.12
    grip_sign: float = 1.0
