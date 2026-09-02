"""Deploy-time grip context and the opt-in low-level gripper controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import torch


GRIP_CONTEXTS = ("soft", "hard", "unknown")
GRIP_CONTEXT_FEATURES = tuple(f"grip_context.{name}" for name in GRIP_CONTEXTS)
GRIP_RESIDUAL_DIRECTIONS = ("tighten", "hold", "loosen")
GRIP_RESIDUAL_FEATURES = (
    "policy_target",
    "command_target",
    "actual_pos",
    "present_current",
    "present_load",
    "position_lag",
)
GRIP_CANDIDATE_DELTAS = (-0.2, 0.0, 0.2)
GRIP_VISUAL_FEATURES_PER_VIEW = (
    "motion_mean",
    "motion_p90",
    "flow_magnitude_p90",
    "lower_vs_upper_flow_y",
    "lower_vs_upper_flow_x",
    "horizontal_strain",
    "vertical_strain",
)
GRIP_VISUAL_FEATURES = tuple(
    f"{view}.{name}"
    for view in ("front", "side")
    for name in GRIP_VISUAL_FEATURES_PER_VIEW
)
GRIP_CANDIDATE_FEATURES = GRIP_RESIDUAL_FEATURES + GRIP_VISUAL_FEATURES


def grip_context_vector(context: str) -> np.ndarray:
    if context not in GRIP_CONTEXTS:
        raise ValueError(f"unknown grip context {context!r}")
    return np.asarray([float(context == name) for name in GRIP_CONTEXTS], dtype=np.float32)


def append_grip_context(state: np.ndarray, *, context: str, expected_dim: int) -> np.ndarray:
    """Append context for new policies while remaining compatible with legacy 6D policies."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if expected_dim == state.size:
        return state
    if expected_dim == state.size + len(GRIP_CONTEXTS):
        return np.concatenate((state, grip_context_vector(context)))
    raise ValueError(
        f"policy expects observation.state dim {expected_dim}, but robot supplies "
        f"{state.size} motor values or {state.size + len(GRIP_CONTEXTS)} values with grip_context"
    )


@dataclass(frozen=True)
class GripFeedbackConfig:
    """Calibrated light-to-hard range used after the policy requests a grasp."""

    light_pos: float
    hard_pos: float
    max_step: float = 2.0
    grasp_enter: float = 30.0
    grasp_exit: float = 65.0

    def __post_init__(self) -> None:
        values = (self.light_pos, self.hard_pos, self.max_step, self.grasp_enter, self.grasp_exit)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grip feedback values must be finite")
        if not 0.0 <= self.hard_pos < self.light_pos <= 100.0:
            raise ValueError("grip feedback requires 0 <= hard_pos < light_pos <= 100")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if not self.grasp_enter < self.grasp_exit:
            raise ValueError("grasp_enter must be smaller than grasp_exit")


class GripFeedbackController:
    """Use the legacy gripper action as an open/grasp gate, not a position target.

    While open, the original policy command passes through. Once grasp is latched, the
    auxiliary intent selects a calibrated light-to-hard target and actual position
    readback closes the loop. This controller is deliberately opt-in at deployment.
    """

    def __init__(self, config: GripFeedbackConfig):
        self.config = config
        self.grasp_latched = False

    def reset(self) -> None:
        self.grasp_latched = False

    def update(
        self,
        *,
        policy_target: float,
        grip_intent: float,
        actual_pos: float,
        force_grasp: bool = False,
    ) -> float:
        values = (policy_target, grip_intent, actual_pos)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("grip controller inputs must be finite")
        cfg = self.config
        policy_target = float(policy_target)
        if force_grasp or (not self.grasp_latched and policy_target <= cfg.grasp_enter):
            self.grasp_latched = True
        elif self.grasp_latched and policy_target >= cfg.grasp_exit:
            self.grasp_latched = False

        if not self.grasp_latched:
            return policy_target

        intent = float(np.clip(grip_intent, 0.0, 1.0))
        desired = cfg.light_pos + intent * (cfg.hard_pos - cfg.light_pos)
        command = float(actual_pos) + float(np.clip(desired - float(actual_pos), -cfg.max_step, cfg.max_step))
        return float(np.clip(command, 0.0, 100.0))


@dataclass(frozen=True)
class StallConfig:
    """When a frozen body trajectory counts as ACT withholding the lift."""

    window_s: float = 2.0
    motion_epsilon: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be positive and finite")
        if not math.isfinite(self.motion_epsilon) or self.motion_epsilon < 0.0:
            raise ValueError("motion_epsilon must be non-negative and finite")


class BodyStallDetector:
    """Detect the stall ACT enters when its grasp is not good enough to lift.

    In trial05 the `shoulder_lift` and `elbow_flex` commands were bit-identical
    at `-19.32` and `35.82` for about nine seconds, and resumed a large
    coordinated motion within one second of the grasp being tightened by hand.
    So the policy withholds the lift rather than failing open-loop, and the
    withholding is visible in the commands alone -- no operator judgement and
    no camera.

    Commands, not readback: readback carries servo noise and the standing
    command-to-readback offset, and a joint holding a load under gravity is not
    still in readback even when nothing is being asked of it.
    """

    def __init__(self, config: StallConfig | None = None):
        self.config = config or StallConfig()
        self._reference: np.ndarray | None = None
        self._still_since_s: float | None = None

    def reset(self) -> None:
        self._reference = None
        self._still_since_s = None

    def update(self, *, t: float, body_command) -> dict:
        command = np.asarray(body_command, dtype=np.float64).reshape(-1)
        if command.size == 0:
            raise ValueError("body_command must have at least one joint")
        if self._reference is not None and self._reference.size != command.size:
            raise ValueError("body_command changed width mid-run")

        # Time since the command last moved, measured against the command it
        # settled at rather than against the previous sample. A drift smaller
        # than `motion_epsilon` per step would otherwise never break the stall,
        # however far the arm travelled.
        #
        # Not a rolling window over a deque: a window whose span has to reach
        # `window_s` before it can report a stall spends every other cycle a
        # float hair under the threshold, and the ramp below reads each of
        # those as "the stall broke". That fabricated a lift boundary equal to
        # the *untightened* depth on every second cycle, and did it silently.
        spread = 0.0 if self._reference is None else float(
            np.max(np.abs(command - self._reference))
        )
        if self._reference is None or spread > self.config.motion_epsilon:
            self._reference = command
            self._still_since_s = float(t)
            spread = 0.0
        still_for = float(t) - self._still_since_s
        return {
            "stalled": still_for >= self.config.window_s,
            "still_for_s": still_for,
            "command_spread": spread,
        }


@dataclass(frozen=True)
class TightenRampConfig:
    """A tighten ramp sized by the measured deadband, with a hard floor."""

    step: float
    interval_s: float
    floor_pos: float
    release_threshold: float = 65.0

    def __post_init__(self) -> None:
        values = (self.step, self.interval_s, self.floor_pos, self.release_threshold)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("tighten ramp values must be finite")
        if self.step <= 0.0:
            raise ValueError("step must be positive; size it from the deadband calibration")
        if self.interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        if not 0.0 <= self.floor_pos <= 100.0:
            raise ValueError("floor_pos must be a gripper position")


class StallTightenRamp:
    """Break an ACT stall by tightening, then hand the gripper straight back.

    This is the no-learning baseline the 2026-09-01 gates require any grip head
    to beat. It has no model, no features, and three rules: tighten only while
    the body is stalled, never past a fixed floor, and never while the policy
    is asking for a release. The depth at which the stall breaks is recorded as
    that grasp's lift boundary, which is the paired label the collection
    protocol wants.

    The floor is fixed rather than learned because the tight bound has no
    labelled example and possibly no sensor: no run has been reviewed as
    crushed, `Present_Load` is quantized to multiples of four and near constant
    while holding, and `Present_Current` spans about sixteen counts.
    """

    def __init__(self, config: TightenRampConfig, *, stall: StallConfig | None = None):
        self.config = config
        self.detector = BodyStallDetector(stall)
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self.target: float | None = None
        self.steps_applied = 0
        self.at_floor = False
        self.lift_boundary: float | None = None
        self._last_step_at_s: float | None = None
        # What the ramp did over the whole run, which is not what it is doing
        # now. `_release()` clears the live fields the moment the stall breaks
        # -- that is, on exactly the trials where the ramp worked -- so a
        # summary read from those at the end would report every successful
        # trial as having tightened nothing.
        self.total_steps_applied = 0
        self.deepest_target: float | None = None
        self.reached_floor = False

    def update(self, *, t: float, policy_target: float, actual_pos: float, body_command):
        """Return the gripper target to command, and what was decided."""
        stall = self.detector.update(t=t, body_command=body_command)
        policy_target = float(policy_target)
        actual_pos = float(actual_pos)

        if policy_target >= self.config.release_threshold:
            # The policy is opening. Tightening into a release would fight the
            # only part of the task these checkpoints do reliably.
            self._release()
            return policy_target, self._label(stall, "release", 0.0)

        if not stall["stalled"]:
            if self.target is not None:
                # The stall broke while we were ramping. Only a ramp that
                # actually tightened produced a boundary: if the body resumed
                # before the first step landed, the grasp was adequate on its
                # own and there is nothing to label. Recording `actual_pos`
                # regardless would fill the column with the starting depth.
                if self.steps_applied:
                    self.lift_boundary = actual_pos
                    action = "resumed_after_tighten"
                else:
                    action = "resumed_untouched"
                self._release()
                return policy_target, self._label(stall, action, 0.0)
            return policy_target, self._label(stall, "following_policy", 0.0)

        if self.target is None:
            self.target = policy_target
            self._last_step_at_s = t
            return self.target, self._label(stall, "latched", 0.0)

        if t - self._last_step_at_s < self.config.interval_s:
            # Dwell. Stepping faster than the jaw settles would stack commands
            # the encoder has not yet answered, which is how a ramp reads as
            # deeper than the jaw has actually gone.
            return self.target, self._label(stall, "dwelling", 0.0)

        self._last_step_at_s = t
        stepped = self.target - self.config.step
        if stepped < self.config.floor_pos:
            self.at_floor = True
            self.reached_floor = True
            return self.target, self._label(stall, "at_floor", 0.0)
        delta = stepped - self.target
        self.target = stepped
        self.steps_applied += 1
        self.total_steps_applied += 1
        if self.deepest_target is None or stepped < self.deepest_target:
            self.deepest_target = stepped
        return self.target, self._label(stall, "tighten", delta)

    def _release(self) -> None:
        self.target = None
        self.steps_applied = 0
        self.at_floor = False
        self._last_step_at_s = None

    def _label(self, stall: dict, action: str, delta_q: float) -> dict:
        return {
            "action": action,
            "delta_q": float(delta_q),
            "ramp_target": self.target,
            "steps_applied": self.steps_applied,
            "at_floor": self.at_floor,
            "lift_boundary": self.lift_boundary,
            **stall,
        }


PAIRED_PHASES = ("following", "loosening", "done")


@dataclass(frozen=True)
class LoosenRampConfig:
    """The post-lift loosen ramp, sized by the measured loosen deadband."""

    step: float
    interval_s: float
    ceiling_pos: float = 60.0

    def __post_init__(self) -> None:
        values = (self.step, self.interval_s, self.ceiling_pos)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("loosen ramp values must be finite")
        if self.step <= 0.0:
            raise ValueError("step must be positive; size it from the deadband calibration")
        if self.interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        if not 0.0 <= self.ceiling_pos <= 100.0:
            raise ValueError("ceiling_pos must be a gripper position")


class PairedBoundaryProtocol:
    """Collect a lift boundary and a slip boundary within one grasp.

    The two-branch protocol from 2026-09-01. ACT runs normally; if it stalls,
    the tighten ramp breaks the stall and that depth is the lift boundary.
    After a lift -- from either branch, whether the ramp helped or ACT managed
    alone -- the loosen ramp opens the jaw step by step until the carton drops,
    and that depth is the slip boundary.

    Continuing into the loosen ramp on the tighten branch too is the whole
    point. Branching on outcome instead would measure the lift boundary only on
    loose grasps and the slip boundary only on tight ones, on disjoint
    populations that could never be compared. Their order is not fixed --
    trial01 lifted and then slipped, so its lift boundary was looser than its
    slip boundary, while trial05 suggests the reverse -- and the sign and size
    of that gap is what says whether ACT's own threshold is conservative or
    permissive.

    The steps are asymmetric because the hardware is: 2026-09-02 measured the
    smallest resolvable loosen step at 0.5 and the smallest resolvable tighten
    step at 2.0. A lift boundary is therefore about four times coarser than a
    slip boundary, and the tighten step is itself wider than ACT's own
    lift-versus-fail band.

    Both stopping criteria are events, not fixed windows, which is what makes
    this immune to the observation-window confound that correlated window
    length with label in the 2026-08-31 collection.
    """

    def __init__(
        self,
        *,
        tighten: TightenRampConfig,
        loosen: LoosenRampConfig,
        stall: StallConfig | None = None,
    ):
        self.tighten_ramp = StallTightenRamp(tighten, stall=stall)
        self.loosen = loosen
        self.reset()

    def reset(self) -> None:
        self.tighten_ramp.reset()
        self.phase = "following"
        self.lift_boundary: float | None = None
        self.slip_boundary: float | None = None
        self.loosen_steps = 0
        self.at_ceiling = False
        self.trace: list[dict] = []
        self._target: float | None = None
        self._last_step_at_s: float | None = None
        self._lift_requested = False
        self._drop_requested = False

    @property
    def freeze_body(self) -> bool:
        """Hold ACT's last body target while the loosen ramp runs.

        A carton the policy has already put back on the table cannot be dropped,
        so the slip boundary has to be measured while the lift is still held.
        """
        return self.phase == "loosening"

    def confirm_lift(self) -> None:
        """Operator: the carton is stably lifted. Starts the loosen ramp."""
        self._lift_requested = True

    def mark_drop(self) -> None:
        """Operator: the carton has dropped. Ends the loosen ramp."""
        self._drop_requested = True

    def update(self, *, t: float, policy_target: float, actual_pos: float, body_command):
        lift_requested, self._lift_requested = self._lift_requested, False
        drop_requested, self._drop_requested = self._drop_requested, False
        policy_target = float(policy_target)
        actual_pos = float(actual_pos)

        if self.phase == "following":
            if lift_requested:
                self.phase = "loosening"
                self._target = actual_pos
                self._last_step_at_s = t
                return self._record(t, self._target, actual_pos, "lift_confirmed", 0.0, None)
            target, label = self.tighten_ramp.update(
                t=t,
                policy_target=policy_target,
                actual_pos=actual_pos,
                body_command=body_command,
            )
            if self.tighten_ramp.lift_boundary is not None and self.lift_boundary is None:
                self.lift_boundary = self.tighten_ramp.lift_boundary
            return self._record(t, target, actual_pos, label["action"], label["delta_q"], label)

        if self.phase == "done":
            return self._record(t, policy_target, actual_pos, "done", 0.0, None)

        # Loosening.
        if drop_requested:
            # Readback, not the commanded value: the command runs ahead of the
            # jaw by the standing offset, and on this gripper only about 90% of
            # a commanded loosen becomes travel.
            self.slip_boundary = actual_pos
            self.phase = "done"
            return self._record(t, policy_target, actual_pos, "drop_marked", 0.0, None)

        if t - self._last_step_at_s < self.loosen.interval_s:
            return self._record(t, self._target, actual_pos, "dwelling", 0.0, None)

        self._last_step_at_s = t
        stepped = self._target + self.loosen.step
        if stepped > self.loosen.ceiling_pos:
            self.at_ceiling = True
            return self._record(t, self._target, actual_pos, "at_ceiling", 0.0, None)
        delta = stepped - self._target
        self._target = stepped
        self.loosen_steps += 1
        return self._record(t, self._target, actual_pos, "loosen", delta, None)

    def _record(self, t, target, actual_pos, action, delta_q, stall_label):
        label = {
            "phase": self.phase,
            "action": action,
            "delta_q": float(delta_q),
            "ramp_target": None if target is None else float(target),
            "actual_pos": actual_pos,
            "loosen_steps": self.loosen_steps,
            "at_ceiling": self.at_ceiling,
            "lift_boundary": self.lift_boundary,
            "slip_boundary": self.slip_boundary,
            "freeze_body": self.freeze_body,
            "stall": stall_label,
        }
        # The whole ramp is kept, not just the marked instant. The operator's
        # keypress lags the drop, and the videos are 10 fps locked one frame per
        # control step, so the true event has to be found offline in this trace.
        self.trace.append({"t": float(t), **label})
        return (float(target) if target is not None else 0.0), label


class GripResidualHead(torch.nn.Module):
    """Small numeric head for a signed gripper suggestion and grasp stability."""

    def __init__(self, *, history_steps: int = 4, hidden_dim: int = 32):
        super().__init__()
        if history_steps <= 0 or hidden_dim <= 0:
            raise ValueError("grip residual dimensions must be positive")
        self.history_steps = history_steps
        self.hidden = torch.nn.Linear(history_steps * len(GRIP_RESIDUAL_FEATURES), hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, len(GRIP_RESIDUAL_DIRECTIONS) + 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features.flatten(start_dim=1))))


class GripResidualShadow:
    """Run a trained residual head for logging without returning a motor command."""

    def __init__(
        self,
        *,
        head: GripResidualHead,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        self.head = head.eval()
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float()
        self.history = deque(maxlen=head.history_steps)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "GripResidualShadow":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        history_steps = int(checkpoint["history_steps"])
        hidden_dim = int(checkpoint["hidden_dim"])
        feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
        feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32)
        expected_shape = (len(GRIP_RESIDUAL_FEATURES),)
        if feature_mean.shape != expected_shape or feature_std.shape != expected_shape:
            raise ValueError(f"grip residual normalization must have shape {expected_shape}")
        if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all():
            raise ValueError("grip residual normalization must be finite")
        if torch.any(feature_std <= 0.0):
            raise ValueError("grip residual feature_std must be positive")
        head = GripResidualHead(history_steps=history_steps, hidden_dim=hidden_dim)
        head.load_state_dict(checkpoint["model_state_dict"])
        return cls(head=head, feature_mean=feature_mean, feature_std=feature_std)

    def observe(
        self,
        *,
        policy_target: float,
        command_target: float,
        actual_pos: float,
        present_current: float,
        present_load: float,
        position_lag: float,
    ) -> dict | None:
        values = np.asarray(
            [
                policy_target,
                command_target,
                actual_pos,
                present_current,
                present_load,
                position_lag,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise ValueError("grip residual shadow inputs must be finite")
        self.history.append(torch.from_numpy(values))
        if len(self.history) < self.history.maxlen:
            return None

        features = torch.stack(tuple(self.history))
        features = (features - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            prediction = self.head(features.unsqueeze(0))[0]
            direction_probabilities = torch.softmax(prediction[:3], dim=0)
            stable_probability = torch.sigmoid(prediction[3])
        direction_index = int(direction_probabilities.argmax())
        return {
            "direction": GRIP_RESIDUAL_DIRECTIONS[direction_index],
            "delta_q_sign": direction_index - 1,
            "direction_probabilities": {
                name: float(direction_probabilities[index])
                for index, name in enumerate(GRIP_RESIDUAL_DIRECTIONS)
            },
            "grasp_stable_probability": float(stable_probability),
            "prediction_for": "next_control_step",
        }


def _grip_view_motion_features(previous_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
    """Measure image motion plus coarse slip/deformation proxies in one fixed view."""
    previous_rgb = np.asarray(previous_rgb)
    current_rgb = np.asarray(current_rgb)
    if previous_rgb.shape != current_rgb.shape or previous_rgb.ndim != 3:
        raise ValueError("grip video frames must be same-shaped HWC images")

    previous = cv2.resize(cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY), (160, 120))
    current = cv2.resize(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY), (160, 120))
    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,
        2,
        15,
        3,
        5,
        1.2,
        0,
    )
    difference = np.abs(current.astype(np.float32) - previous.astype(np.float32)) / 255.0
    magnitude = np.linalg.norm(flow, axis=2)

    upper = flow[:42, 56:104]
    lower = flow[66:108, 24:144]
    left, right = lower[:, :60], lower[:, 60:]
    top, bottom = lower[:21], lower[21:]
    features = np.asarray(
        [
            difference.mean(),
            np.quantile(difference, 0.9),
            np.quantile(magnitude, 0.9),
            np.median(lower[..., 1]) - np.median(upper[..., 1]),
            np.median(lower[..., 0]) - np.median(upper[..., 0]),
            np.median(right[..., 0]) - np.median(left[..., 0]),
            np.median(bottom[..., 1]) - np.median(top[..., 1]),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError("grip video features must be finite")
    return features


def grip_visual_features(
    *,
    previous_front_rgb: np.ndarray,
    current_front_rgb: np.ndarray,
    previous_side_rgb: np.ndarray,
    current_side_rgb: np.ndarray,
) -> np.ndarray:
    """Return fixed, low-dimensional motion features for slip/deformation observation."""
    return np.concatenate(
        (
            _grip_view_motion_features(previous_front_rgb, current_front_rgb),
            _grip_view_motion_features(previous_side_rgb, current_side_rgb),
        )
    )


class GripCandidateHead(torch.nn.Module):
    """Score post-action stability for one proposed gripper delta q."""

    def __init__(self, *, history_steps: int = 4, hidden_dim: int = 32):
        super().__init__()
        if history_steps <= 0 or hidden_dim <= 0:
            raise ValueError("grip candidate dimensions must be positive")
        self.history_steps = history_steps
        input_dim = history_steps * len(GRIP_CANDIDATE_FEATURES) + 1
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, delta_q: torch.Tensor) -> torch.Tensor:
        candidate = delta_q.reshape(-1, 1) / 0.2
        inputs = torch.cat((features.flatten(start_dim=1), candidate), dim=1)
        return self.output(torch.relu(self.hidden(inputs))).squeeze(1)


def select_stability_effort_candidate(
    stability_probabilities: dict[float, float],
    predicted_loads: dict[float, float],
    *,
    present_load: float,
    minimum_probability: float,
    minimum_load_for_loosen: float,
) -> float:
    """Select hold/loosen exactly as evaluated by the post-lift offline gate."""
    expected = {0.0, 0.2}
    if set(stability_probabilities) != expected or set(predicted_loads) != expected:
        raise ValueError("post-lift candidate scores must cover hold and +0.2")
    eligible = [
        delta
        for delta in (0.0, 0.2)
        if stability_probabilities[delta] >= minimum_probability
        and (delta == 0.0 or abs(float(present_load)) > minimum_load_for_loosen)
    ]
    return min(eligible, key=predicted_loads.get) if eligible else 0.0


class GripCandidateScorer:
    """Load the post-lift stability/effort heads and score one telemetry context."""

    def __init__(
        self,
        *,
        stability_head: GripCandidateHead,
        effort_head: GripCandidateHead,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        effort_mean: float,
        effort_std: float,
        minimum_probability: float,
        minimum_load_for_loosen: float,
        visual_gap_frames: int,
    ):
        self.stability_head = stability_head.eval()
        self.effort_head = effort_head.eval()
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float()
        self.effort_mean = float(effort_mean)
        self.effort_std = float(effort_std)
        self.minimum_probability = float(minimum_probability)
        self.minimum_load_for_loosen = float(minimum_load_for_loosen)
        self.visual_gap_frames = int(visual_gap_frames)
        self.history = deque(maxlen=stability_head.history_steps)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "GripCandidateScorer":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "action_conditioned_grip_stability_effort_v1":
            raise ValueError("checkpoint is not a stability/effort grip candidate model")
        if checkpoint.get("training_view") != "post_lift_hold_loosen":
            raise ValueError("bounded trial requires the post-lift-only training view")
        if not checkpoint.get("offline_gate_pass", False):
            raise ValueError("checkpoint did not pass its offline gate")
        if tuple(float(value) for value in checkpoint["candidate_deltas"]) != (0.0, 0.2):
            raise ValueError("bounded trial supports only hold and +0.2")

        history_steps = int(checkpoint["history_steps"])
        hidden_dim = int(checkpoint["hidden_dim"])
        feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32)
        feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32)
        expected_shape = (len(GRIP_CANDIDATE_FEATURES),)
        if feature_mean.shape != expected_shape or feature_std.shape != expected_shape:
            raise ValueError(f"grip candidate normalization must have shape {expected_shape}")
        if tuple(checkpoint["feature_names"]) != GRIP_CANDIDATE_FEATURES:
            raise ValueError("grip candidate feature order does not match runtime")

        stability_head = GripCandidateHead(history_steps=history_steps, hidden_dim=hidden_dim)
        effort_head = GripCandidateHead(history_steps=history_steps, hidden_dim=hidden_dim)
        stability_head.load_state_dict(checkpoint["stability_model_state_dict"])
        effort_head.load_state_dict(checkpoint["effort_model_state_dict"])
        policy = checkpoint["selection_policy"]
        return cls(
            stability_head=stability_head,
            effort_head=effort_head,
            feature_mean=feature_mean,
            feature_std=feature_std,
            effort_mean=float(checkpoint["effort_mean"]),
            effort_std=float(checkpoint["effort_std"]),
            minimum_probability=float(policy["minimum_probability"]),
            minimum_load_for_loosen=float(policy["minimum_present_load_for_loosen"]),
            visual_gap_frames=int(checkpoint["visual_gap_frames"]),
        )

    def observe(
        self,
        *,
        policy_target: float,
        command_target: float,
        actual_pos: float,
        present_current: float,
        present_load: float,
        position_lag: float,
        previous_front_rgb: np.ndarray,
        current_front_rgb: np.ndarray,
        previous_side_rgb: np.ndarray,
        current_side_rgb: np.ndarray,
        use_load_gate: bool = True,
    ) -> dict | None:
        numeric = np.asarray(
            [
                policy_target,
                command_target,
                actual_pos,
                present_current,
                present_load,
                position_lag,
            ],
            dtype=np.float32,
        )
        visual = grip_visual_features(
            previous_front_rgb=previous_front_rgb,
            current_front_rgb=current_front_rgb,
            previous_side_rgb=previous_side_rgb,
            current_side_rgb=current_side_rgb,
        )
        self.history.append(torch.from_numpy(np.concatenate((numeric, visual))))
        if len(self.history) < self.history.maxlen:
            return None

        features = torch.stack(tuple(self.history))
        features = (features - self.feature_mean) / self.feature_std
        probabilities: dict[float, float] = {}
        predicted_loads: dict[float, float] = {}
        with torch.inference_mode():
            for delta in (0.0, 0.2):
                candidate = torch.tensor([delta], dtype=torch.float32)
                context = features.unsqueeze(0)
                probabilities[delta] = float(torch.sigmoid(self.stability_head(context, candidate)))
                predicted_loads[delta] = float(
                    self.effort_head(context, candidate) * self.effort_std + self.effort_mean
                )
        selected = select_stability_effort_candidate(
            probabilities,
            predicted_loads,
            present_load=present_load,
            minimum_probability=self.minimum_probability,
            minimum_load_for_loosen=(self.minimum_load_for_loosen if use_load_gate else -1.0),
        )
        return {
            "selected_delta_q": selected,
            "stability_probabilities": {f"{delta:+.1f}": probabilities[delta] for delta in (0.0, 0.2)},
            "predicted_loads": {f"{delta:+.1f}": predicted_loads[delta] for delta in (0.0, 0.2)},
            "present_load_abs": abs(float(present_load)),
            "load_gate_enabled": bool(use_load_gate),
            "minimum_present_load_for_loosen": self.minimum_load_for_loosen,
            "prediction_for": "next_control_step",
        }


def select_grip_candidate(
    stability_probabilities: dict[float, float],
    *,
    supported_deltas: set[float],
    stable_lift_seen: bool,
    minimum_probability: float = 0.65,
    minimum_hold_advantage: float = 0.10,
) -> float:
    """Choose a supported candidate conservatively; uncertainty falls back to hold."""
    if set(stability_probabilities) != set(GRIP_CANDIDATE_DELTAS):
        raise ValueError("stability probabilities must cover all grip candidate deltas")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in stability_probabilities.values()):
        raise ValueError("stability probabilities must be finite values in [0, 1]")

    allowed = {delta for delta in supported_deltas if delta in GRIP_CANDIDATE_DELTAS}
    allowed.add(0.0)
    if not stable_lift_seen:
        allowed.discard(0.2)
    best = max(allowed, key=lambda delta: stability_probabilities[delta])
    if best == 0.0:
        return 0.0
    if stability_probabilities[best] < minimum_probability:
        return 0.0
    if stability_probabilities[best] - stability_probabilities[0.0] < minimum_hold_advantage:
        return 0.0
    return best


def pv_teacher_label(reading) -> tuple[np.ndarray, np.ndarray]:
    """Convert one live PV reading into the dataset target and its validity mask."""
    valid = bool(
        reading is not None
        and getattr(reading, "active", False)
        and getattr(reading, "available", False)
        and getattr(reading, "fresh", True)
        and getattr(reading, "status", None)
        not in {"pv_stale", "pv_unavailable", "pv_time_skew", "pressure_error", "pressure_unavailable"}
    )
    value = float(getattr(reading, "pressure_0_1", 0.0)) if valid else 0.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        valid, value = False, 0.0
    return (
        np.asarray([value], dtype=np.float32),
        np.asarray([float(valid)], dtype=np.float32),
    )
