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

from .ee_control import (
    ee_action_from_hand,
    gripper_pos_from_pinch,  # noqa: F401  (re-exported for existing callers)
    joint_center,
    raw_grip_from_pinch,
)

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
# SO-101 is 5-DOF: every degree of tool roll is spent out of the budget that holds the
# gripper down. Opting in past this range trades the down-pose for roll authority.
MAX_WRIST_ROLL_RANGE_DEG = 45.0

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
    GRIP_MODES,
    GRIP_OPEN_ALPHA,
    GRIP_OVERDRIVE,
    MediaPipeGripperController,
)

GRIP_MAPS = ("overdrive", "span")


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


def bounded_wrist_roll_delta(quaternion, reference_roll, range_deg, gain=1.0) -> float:
    """Relative hand roll, wrapped at pi and bounded about the latched pose.

    The wrap matters: a hand held near +-180 deg crosses the branch cut on its
    own, and an unwrapped difference would read that as a full turn.
    """
    current_roll = float(
        Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_euler("xyz")[0]
    )
    delta = float(gain) * float(np.arctan2(
        np.sin(current_roll - reference_roll),
        np.cos(current_roll - reference_roll),
    ))
    limit = float(np.deg2rad(range_deg))
    return float(np.clip(delta, -limit, limit))


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

    def __init__(
        self,
        robot,
        kin,
        cfg,
        use_oak: bool,
        gripper=None,
        *,
        grip_mode: str = "tracked",
        grip_map: str = "overdrive",
        wrist_roll_range_deg: float = 0.0,
        wrist_roll_gain: float = 1.0,
    ):
        if grip_mode not in GRIP_MODES:
            raise ValueError(f"unknown grip_mode {grip_mode!r}; expected one of {GRIP_MODES}")
        if grip_mode != "tracked" and gripper is not None:
            # grip_mode configures the default MediaPipe gripper. Silently
            # ignoring it on an injected one is exactly the class of bug this
            # constructor is supposed to catch.
            raise ValueError("grip_mode applies to the default gripper; an injected gripper owns its own mode")
        # "overdrive" is the validated map and stays the default: the recorded
        # datasets and the trained policies assume it, so changing it silently
        # would put recording and deploy on a different action distribution.
        if grip_map not in GRIP_MAPS:
            raise ValueError(f"unknown grip_map {grip_map!r}; expected one of {GRIP_MAPS}")
        self.grip_map = grip_map
        # Span mode carries no overdrive -- narrowing the range is what replaces
        # it, and applying both would reintroduce the clipped dead zone.
        self.grip_overdrive = 0.0 if grip_map == "span" else GRIP_OVERDRIVE
        self.wrist_roll_range_deg = float(wrist_roll_range_deg)
        if not 0.0 <= self.wrist_roll_range_deg <= MAX_WRIST_ROLL_RANGE_DEG:
            raise ValueError(f"wrist_roll_range_deg must be within 0..{MAX_WRIST_ROLL_RANGE_DEG:g}")
        self.wrist_roll_gain = float(wrist_roll_gain)
        if not 0.0 < self.wrist_roll_gain <= 4.0:
            raise ValueError("wrist_roll_gain must be within (0, 4]")
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
        self.roll_ref = None
        self.prev_enabled = False
        self.smoothed = None
        self.gripper = gripper or MediaPipeGripperController(
            overdrive=self.grip_overdrive, grip_mode=grip_mode
        )
        self.cmd_state = dict(self.middle_pose)

    # The grip smoothing state lives behind the gripper contract now. These
    # forward to it so the controller stays the single place callers read
    # control state from.
    @property
    def grip_mode(self) -> str:
        return getattr(self.gripper, "grip_mode", "tracked")

    @property
    def grip_smoothed(self):
        return getattr(self.gripper, "current_command", None)

    @property
    def grip_latched(self) -> bool:
        return bool(getattr(self.gripper, "latched", False))

    @property
    def grip_open_frames(self) -> int:
        return int(getattr(self.gripper, "open_frames", 0))

    def build(self, ee_centre):
        """Build the EE->joints pipeline with a workspace box centred on the ready EE pose."""
        ee_min = (np.asarray(ee_centre, float) - WORKSPACE_HALF_BOX).tolist()
        ee_min[2] = max(ee_min[2], Z_FLOOR_M)            # never below the table
        ee_bounds = {"min": ee_min, "max": (np.asarray(ee_centre, float) + WORKSPACE_HALF_BOX).tolist()}
        # Roll is opt-in and mutually exclusive with the fixed orientation:
        # FixedEEOrientation overwrites the whole rotvec every frame, so leaving
        # it in would silently discard the roll delta rather than bound it.
        steps = [
            EEReferenceAndDelta(kinematics=self.kin,
                                end_effector_step_sizes={"x": self.fwd_scale, "y": EE_LAT_SCALE, "z": EE_UP_SCALE},
                                motor_names=self.motors, use_latched_reference=True),
            SlewLimitedEEBounds(end_effector_bounds=ee_bounds, max_ee_step_m=MAX_EE_STEP_M),
        ]
        if self.wrist_roll_range_deg <= 0.0:
            steps.append(FixedEEOrientation(rotvec=tuple(self.r_down)))
        self.pipeline = RobotProcessorPipeline(
            steps=[
                *steps,
                GripperVelocityToJoint(speed_factor=20.0),
                InverseKinematicsEEToJoints(kinematics=self.kin, motor_names=self.motors,
                                            initial_guess_current_joints=True),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    def close(self) -> None:
        """Release control state. Safe to call more than once.

        The controller owns the gripper controller's latched command and the
        smoothing state; a recorder that stops mid-episode must not leave either
        behind for the next session to inherit.
        """
        self.gripper.reset()
        self.smoothed = None
        self.ref = None
        self.roll_ref = None
        self.prev_enabled = False
        if self.pipeline is not None:
            self.pipeline.reset()

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
            if self.wrist_roll_range_deg > 0.0:
                self.roll_ref = float(
                    Rotation.from_quat(
                        np.asarray(wrist.quaternion, dtype=float)
                    ).as_euler("xyz")[0]
                )
        displacement = (np.asarray(wrist.position, dtype=float) - self.ref) if enabled else np.zeros(3)
        self.prev_enabled = enabled

        lm = np.asarray(landmarks.landmarks, dtype=float)
        pinch = float(np.linalg.norm(lm[_THUMB_TIP] - lm[_INDEX_TIP]))
        roll_delta = (
            bounded_wrist_roll_delta(
                wrist.quaternion,
                self.roll_ref,
                self.wrist_roll_range_deg,
                self.wrist_roll_gain,
            )
            if enabled and self.roll_ref is not None
            else 0.0
        )
        ee_act = ee_action_from_hand(
            displacement, pinch, enabled, self.cfg, roll_delta=roll_delta
        )

        if clutch:
            self.pipeline.reset()
            self.ref = None
            self.roll_ref = None
            self.smoothed = None
            self.gripper.reset()
            self._disarm_gripper(pinch, wrist, transition="middle")
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
            severity = 1.0 - raw_grip_from_pinch(pinch, self.cfg, grip_map=self.grip_map) / 100.0
            observed_at_s = wrist.observed_at_s if hasattr(wrist, "observed_at_s") else 0.0
            # A gripper that reads its own sensor gets the frame here. This hook
            # is the whole reason the core never has to import an integration
            # package: it is satisfied by duck typing, not by an import.
            if hasattr(self.gripper, "observe_frame"):
                self.gripper.observe_frame(
                    landmarks, pinch=pinch, enabled=enabled, observed_at_s=observed_at_s
                )
            grip_in = GripInput(
                grasp_active=True,
                explicit_release=False,
                severity=severity,
                valid=True,
                observed_at_s=observed_at_s,
            )
            joint_act["gripper.pos"] = self.gripper.step(
                grip_in, actual_pos=self.cmd_state.get("gripper.pos", 50.0)
            )
            self.cmd_state = dict(joint_act)
            return joint_act, "MOVING"

        # No command this frame. A gripper that measures the world has to be
        # told: the hand left, so whatever zero it calibrated against may no
        # longer hold. Continuing from a stale baseline is how a sensor keeps
        # commanding force against a scene it can no longer see.
        self._disarm_gripper(pinch, wrist, transition="hold")
        return None, "HOLD"

    def _disarm_gripper(self, pinch, wrist, *, transition: str) -> None:
        """Tell a sensing gripper this frame produced no command. Optional hook."""
        disarm = getattr(self.gripper, "disarm", None)
        if disarm is None:
            return
        severity = 1.0 - raw_grip_from_pinch(pinch, self.cfg, grip_map=self.grip_map) / 100.0
        disarm(
            GripInput(
                grasp_active=False,
                explicit_release=False,
                severity=severity,
                valid=True,
                observed_at_s=wrist.observed_at_s if hasattr(wrist, "observed_at_s") else 0.0,
            ),
            self.cmd_state.get("gripper.pos", 50.0),
            transition=transition,
        )
