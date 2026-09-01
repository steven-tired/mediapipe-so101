"""Single source of truth for the validated webcam->SO-101 END-EFFECTOR control.

Both the live preview (`teleop_viz_ee.py`) and the LeRobot `record_loop` recorder drive the arm
through `WebcamEEController.step()`, so recording can never diverge from what was validated live.

The controller owns the EE->joints pipeline (with the fixes proven during bring-up):
  - FixedEEOrientation -> gripper points straight DOWN (top-down picking),
  - SlewLimitedEEBounds -> rate-limit big EE steps instead of dropping them (no freeze),
  - InverseKinematicsEEToJoints with a raised orientation_weight (so "down" is actually held),
  - body-joint EMA + gripper EMA (anti-shake / anti-loosening),
  - OPEN-LOOP: the IK is fed our last COMMANDED joints (no per-frame bus read during control).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)

from .ee_control import ee_action_from_hand, gripper_pos_from_pinch, joint_center

_THUMB_TIP = 4
_INDEX_TIP = 8

# --- tuning (validated during bring-up) ---
EE_FWD_SCALE = 0.5    # robot x (forward) <- noisy monocular depth; OAK overrides to 1.0
EE_LAT_SCALE = 1.6    # robot y (lateral) <- clean image x; >1 = wider shoulder_pan sweep per hand move
EE_UP_SCALE = 1.0     # robot z (height)  <- clean image y
# +/- m around the ready EE centre. Lateral (y, 2nd) widened 0.20->0.30 so the EE target isn't
# clamped early -> shoulder_pan can rotate further. Raise EE_LAT_SCALE/this for even more pan.
WORKSPACE_HALF_BOX = np.array([0.16, 0.30, 0.16])
Z_FLOOR_M = 0.0       # table height (base frame); IK target z clamped >= this
MAX_EE_STEP_M = 0.10  # per-frame EE jump cap (slew-limited, not dropped)
SMOOTHING_ALPHA = 0.3            # body-joint EMA (lower = smoother)
EE_ORI_WEIGHT = 0.05             # IK orientation weight (0.01 default ignores orientation)
MIDDLE_WRIST_DOWN_DEG = 90.0     # clutch/ready pose pitches wrist down so the gripper points down

# --- Gripper strength (0 = fully closed/clamped, 100 = open) ---
# ASYMMETRIC EMA: close fast & firm, open slow. The slow-open is what stops the claw loosening when
# pinch tracking jitters mid-lift (a brief "open" reading barely moves the command). Raise
# GRIP_CLOSE_ALPHA toward 1.0 for an even snappier/firmer clamp; lower GRIP_OPEN_ALPHA for more
# loosening resistance. GRIP_OVERDRIVE shifts the whole command toward closed so a normal (not
# fully-touching) pinch still reaches a firm grip -- increase it if the grip is still too soft.
# The tuned values and the smoothing itself now live behind the gripper contract,
# so the optional PressureVision mode can be swapped in without touching this file.
from .grip.contract import GripInput
from .grip.mediapipe import (  # noqa: F401  (re-exported for existing callers)
    GRIP_CLOSE_ALPHA,
    GRIP_OPEN_ALPHA,
    GRIP_OVERDRIVE,
    MediaPipeGripperController,
)


@dataclass
class FixedEEOrientation(RobotActionProcessorStep):
    """Force a constant target EE orientation (rotvec) every frame.

    SO-101 is 5-DOF and can't track full position + orientation, so we lock ONE orientation. The
    correct constant points the gripper straight DOWN (computed from a +90 deg wrist-pitch FK), NOT
    the all-zero pose which actually points the gripper forward.
    """

    rotvec: tuple

    def action(self, action):
        action["ee.wx"], action["ee.wy"], action["ee.wz"] = (
            float(self.rotvec[0]), float(self.rotvec[1]), float(self.rotvec[2]))
        return action

    def transform_features(self, features):
        return features


@dataclass
class SlewLimitedEEBounds(EEBoundsAndSafety):
    """EEBoundsAndSafety that SLEW-LIMITS an oversized EE step instead of dropping it.

    Upstream computes a clamped position for an over-cap jump but then raises instead of using it,
    so _last_pos never advances and every later frame is dropped (the arm freezes). Here we clamp
    the step to max_ee_step_m and keep going -- fast hand motion / depth noise is rate-limited.
    """

    def action(self, action):
        pos = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
        self._last_pos = pos
        action["ee.x"], action["ee.y"], action["ee.z"] = float(pos[0]), float(pos[1]), float(pos[2])
        return action


class WebcamEEController:
    """Stateful per-frame controller: (wrist, landmarks) -> SO-101 joint targets.

    Usage:
        ctl = WebcamEEController(robot, kin, cfg, use_oak=True)
        ctl.build(ee_centre)          # after the arm is at the ready pose (FK gives ee_centre)
        ctl.seed(start_joint_dict)    # seed the open-loop state from a settled read
        joints, state = ctl.step(wrist, landmarks)   # state in {MIDDLE, MOVING, HOLD}
    """

    def __init__(self, robot, kin, cfg, use_oak: bool, gripper=None):
        self.kin = kin
        self.cfg = cfg
        self.motors = list(robot.bus.motors.keys())
        self.body_motors = [m for m in self.motors if m != "gripper"]
        self.fwd_scale = 1.0 if use_oak else EE_FWD_SCALE

        # Down ready/clutch pose: centred joints, EXCEPT wrist pitched down so the gripper points
        # straight down here too (matches r_down) -- this re-centres the workspace box on the
        # down-reachable region and removes the orientation snap on re-engage.
        self.middle_pose = {f"{m}.pos": joint_center(robot.bus.motors[m].norm_mode.value)
                            for m in self.motors}
        self.middle_pose["wrist_flex.pos"] = MIDDLE_WRIST_DOWN_DEG

        # Fixed downward orientation = FK orientation at a +90 deg wrist pitch (gripper-Z -> base -Z).
        q_down = np.zeros(len(self.motors), dtype=float)
        q_down[self.motors.index("wrist_flex")] = MIDDLE_WRIST_DOWN_DEG
        self.r_down = Rotation.from_matrix(kin.forward_kinematics(q_down)[:3, :3]).as_rotvec()

        # Raise the IK orientation weight (default 0.01 ignores orientation) so "down" is held.
        _ik = kin.inverse_kinematics
        kin.inverse_kinematics = (
            lambda q, pose, position_weight=1.0, orientation_weight=EE_ORI_WEIGHT:
            _ik(q, pose, position_weight=position_weight, orientation_weight=orientation_weight))

        self.pipeline = None
        self.ref = None
        self.prev_enabled = False
        self.smoothed = None
        self.gripper = gripper or MediaPipeGripperController()
        self.cmd_state = dict(self.middle_pose)

    def build(self, ee_centre):
        """Build the EE->joints pipeline with a workspace box centred on the ready EE pose."""
        ee_min = (np.asarray(ee_centre, float) - WORKSPACE_HALF_BOX).tolist()
        ee_min[2] = max(ee_min[2], Z_FLOOR_M)            # never below the table
        ee_bounds = {"min": ee_min, "max": (np.asarray(ee_centre, float) + WORKSPACE_HALF_BOX).tolist()}
        self.pipeline = RobotProcessorPipeline(
            steps=[
                EEReferenceAndDelta(kinematics=self.kin,
                                    end_effector_step_sizes={"x": self.fwd_scale, "y": EE_LAT_SCALE, "z": EE_UP_SCALE},
                                    motor_names=self.motors, use_latched_reference=True),
                SlewLimitedEEBounds(end_effector_bounds=ee_bounds, max_ee_step_m=MAX_EE_STEP_M),
                FixedEEOrientation(rotvec=tuple(self.r_down)),
                GripperVelocityToJoint(speed_factor=20.0),
                InverseKinematicsEEToJoints(kinematics=self.kin, motor_names=self.motors,
                                            initial_guess_current_joints=True),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    def seed(self, joint_state: dict):
        """Seed the open-loop state from a settled joint read (keys '<motor>.pos')."""
        self.cmd_state = {f"{m}.pos": float(joint_state[f"{m}.pos"]) for m in self.motors}

    def step(self, wrist, landmarks):
        """Return (joint_targets | None, state). None = HOLD (no new command; arm holds last)."""
        if self.pipeline is None:
            raise RuntimeError("WebcamEEController.build(ee_centre) must be called before step().")
        clutch = (wrist.fist_state == "closed")
        enabled = bool(wrist.valid and not clutch)

        if enabled and (not self.prev_enabled or self.ref is None):
            self.ref = np.asarray(wrist.position, dtype=float).copy()
        displacement = (np.asarray(wrist.position, dtype=float) - self.ref) if enabled else np.zeros(3)
        self.prev_enabled = enabled

        lm = np.asarray(landmarks.landmarks, dtype=float)
        pinch = float(np.linalg.norm(lm[_THUMB_TIP] - lm[_INDEX_TIP]))
        ee_act = ee_action_from_hand(displacement, pinch, enabled, self.cfg)

        if clutch:
            self.pipeline.reset()
            self.ref = None
            self.smoothed = None
            self.gripper.reset()
            self.cmd_state = dict(self.middle_pose)
            return dict(self.middle_pose), "MIDDLE"

        if enabled:
            joint_act = self.pipeline((ee_act, self.cmd_state))
            if self.smoothed is None:
                self.smoothed = {m: joint_act[f"{m}.pos"] for m in self.body_motors}
            for m in self.body_motors:
                self.smoothed[m] = SMOOTHING_ALPHA * joint_act[f"{m}.pos"] + (1 - SMOOTHING_ALPHA) * self.smoothed[m]
                joint_act[f"{m}.pos"] = self.smoothed[m]
            # Overdrive and the asymmetric close/open smoothing live in the gripper
            # controller now. severity is the pinch mapping normalised: 1 = grip hardest.
            severity = 1.0 - gripper_pos_from_pinch(pinch, self.cfg) / 100.0
            grip_in = GripInput(
                grasp_active=True,
                explicit_release=False,
                severity=severity,
                valid=True,
                observed_at_s=wrist.observed_at_s if hasattr(wrist, "observed_at_s") else 0.0,
            )
            joint_act["gripper.pos"] = self.gripper.step(
                grip_in, actual_pos=self.cmd_state.get("gripper.pos", 50.0)
            )
            self.cmd_state = dict(joint_act)
            return joint_act, "MOVING"

        return None, "HOLD"
