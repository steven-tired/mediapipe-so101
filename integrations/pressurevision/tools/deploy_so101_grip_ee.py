"""Deploy a trained LeRobot policy on the SO-101 carton task.

The recorded ACTION was post-IK joint targets, so deployment needs **no IK and no
webcam/hand-tracking**: the policy maps (front-cam image + joint state) -> 6 joint targets
directly. This mirrors `record_so101_ee.py` MINUS the teleop:

  * same robot/camera setup (SO101Follower, Creative workspace camera as observation.images.front),
  * same down-ready start pose (controller.middle_pose) the demos started from,
  * LeRobot's own predict_action() (loads the saved normalizers, so units/scaling match training).

By default this runs observation and policy inference only. It never sends goal positions
unless ``--arm-enabled`` is passed explicitly. Armed deployment starts from the current pose;
the old fixed ready-pose ramp is available only through ``--start-mode ready``.

SAFETY: ``--arm-enabled`` moves the arm autonomously. Clear the workspace, place one carton,
and keep a hand on the power switch. Ctrl-C stops cleanly and leaves torque ON.
"""

import argparse
from collections import deque
import os
from copy import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import prepare_observation_for_inference
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins

from record_so101_pv_ee import (
    HUMAN_INTERVENTION_FEATURE,
    PV_AUXILIARY_FEATURES,
    PVRecorderRobot,
    SIDE_CAMERA_FPS,
    SIDE_CAM_FOURCC,
    SIDE_CAM_PATH,
    _build_pressure_source,
    _read_positions,
    _ramp_to,
    build_training_features,
    pv_supervision_from_reading,
    validate_dataset_schema,
)
from lerobot_teleoperator_so101_webcam.ee_control import joint_center
from lerobot_teleoperator_so101_webcam.ee_controller import MIDDLE_WRIST_DOWN_DEG
from lerobot_teleoperator_so101_webcam.grip.runtime import (
    GRIP_CONTEXTS,
    GripFeedbackConfig,
    GripFeedbackController,
    GripCandidateScorer,
    GripResidualShadow,
    append_grip_context,
    pv_teacher_label,
)
from lerobot_teleoperator_so101_webcam.gripper_hardware import _read_reg

ARM_ID = os.environ.get("SO101_ARM_ID", "so101_follower_1")
WORKSPACE_CAM_PATH = os.environ.get(
    "SO101_WORKSPACE_CAM",
    "/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0",
)
WORKSPACE_CAM_FOURCC = "YUYV"
FPS = 10                             # match the dataset's record rate
# The Hub copy of the same ACT 80k final the worktree pointed at by local path
# (see training/PHASE_C_CHECKPOINTS.md). Point this at a local
# `<step>/pretrained_model` directory to deploy a different checkpoint.
DEFAULT_POLICY = os.environ.get("SO101_GRIP_POLICY", "stevenzenith/act_carton_phase_c_80k")
TASK = (
    "Gently grasp and lift the 250 g paper carton, tighten the gripper if it slips, "
    "then return it to the table and release it."
)
DEFAULT_CORRECTION_REPO = "local/hand_tracking_pv_pick_place"


def _load_policy(repo: str, device: torch.device):
    """Load the trained policy + its saved pre/post processors (normalizers) from a Hub repo/dir."""
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.pretrained_path = repo
    cfg.device = str(device)
    policy = get_policy_class(cfg.type).from_pretrained(repo, config=cfg)
    policy.to(device).eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=repo,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post


def _policy_state(obs: dict, motors: list[str], policy, grip_context: str) -> np.ndarray:
    feature = policy.config.robot_state_feature
    if feature is None:
        raise ValueError("policy does not define observation.state")
    motor_state = np.asarray([obs[f"{motor}.pos"] for motor in motors], dtype=np.float32)
    return append_grip_context(
        motor_state,
        context=grip_context,
        expected_dim=int(feature.shape[0]),
    )


def _policy_camera_inputs(policy, preprocessor) -> set[str]:
    """Return raw camera features expected before the saved policy preprocessor runs."""
    policy_image_features = set(policy.config.image_features)
    supported = {
        "observation.images.front",
        "observation.images.side",
    }
    if policy_image_features and policy_image_features <= supported:
        return policy_image_features

    if policy.config.type == "smolvla":
        rename_map = {}
        for step in preprocessor.steps:
            rename_map.update(getattr(step, "rename_map", {}))
        camera_inputs = {
            source
            for source, target in rename_map.items()
            if source in supported and target in policy_image_features
        }
        if camera_inputs:
            return camera_inputs

    raise ValueError(
        "policy image schema must use front and optionally side, either directly or through "
        f"its saved SmolVLA rename processor; got {sorted(policy_image_features)}"
    )


def _queued_policy_actions(policy) -> list[torch.Tensor]:
    if policy.config.type == "act":
        if policy.config.temporal_ensemble_coeff is not None:
            return []
        return list(policy._action_queue)
    return list(policy._queues[ACTION])


class PolicyChunkTrace:
    """Expose the normalized execution chunk without changing policy queue semantics."""

    def __init__(self, policy, postprocessor):
        self.policy = policy
        self.postprocessor = postprocessor
        self.chunk_id = -1
        self.execution_index = -1
        self.raw_normalized_chunk = None
        self.denormalized_chunk = None

    def select(self, processed_observation: dict) -> tuple[torch.Tensor, dict]:
        starts_chunk = not _queued_policy_actions(self.policy)
        raw_action = self.policy.select_action(processed_observation)
        if starts_chunk:
            self.chunk_id += 1
            self.execution_index = 0
            self.raw_normalized_chunk = torch.stack(
                [raw_action, *_queued_policy_actions(self.policy)], dim=1
            )
            self.denormalized_chunk = self.postprocessor(self.raw_normalized_chunk)
        else:
            self.execution_index += 1
        action = self.postprocessor(raw_action)
        return action, {
            "chunk_id": self.chunk_id,
            "execution_index": self.execution_index,
            "raw_normalized_action": raw_action,
            "raw_normalized_chunk": self.raw_normalized_chunk,
            "denormalized_chunk": self.denormalized_chunk,
        }


def hold_body_action(action: np.ndarray, state: np.ndarray, *, gripper_index: int) -> np.ndarray:
    """Keep every body joint at readback while preserving the selected gripper target."""
    held = state.copy()
    held[gripper_index] = action[gripper_index]
    return held


def apply_gripper_close_offset(
    action: np.ndarray,
    *,
    gripper_index: int,
    offset: float,
) -> np.ndarray:
    """Loosen predicted close actions while leaving open/release actions unchanged."""
    adjusted = action.copy()
    if adjusted[gripper_index] < 90.0:
        adjusted[gripper_index] += offset
    return adjusted


def read_present_positions(bus, *, tries: int = 8, retry_delay_s: float = 0.03) -> dict:
    """Retry transient Feetech status-packet loss during armed evidence readback."""
    for attempt in range(tries):
        try:
            return bus.sync_read("Present_Position")
        except ConnectionError:
            if attempt + 1 == tries:
                raise
            time.sleep(retry_delay_s)


def _predict_action_with_trace(
    observation: dict[str, np.ndarray],
    tracer: PolicyChunkTrace,
    device: torch.device,
    preprocessor,
    task: str,
    robot_type: str,
) -> tuple[torch.Tensor, dict]:
    prepared = copy(observation)
    with torch.inference_mode():
        prepared = prepare_observation_for_inference(prepared, device, task, robot_type)
        prepared = preprocessor(prepared)
        return tracer.select(prepared)


def _predict_action_and_grip(
    observation: dict[str, np.ndarray],
    policy,
    device: torch.device,
    preprocessor,
    postprocessor,
    task: str,
) -> tuple[torch.Tensor, float]:
    """Run one shared preprocessing pass for the 6D action and auxiliary intent."""
    if not hasattr(policy, "predict_grip_intent"):
        raise ValueError("--grip-control aux requires a grip_aux policy checkpoint")
    prepared = copy(observation)
    with torch.inference_mode():
        prepared = prepare_observation_for_inference(prepared, device, task, None)
        prepared = preprocessor(prepared)
        action = postprocessor(policy.select_action(prepared))
        intent = policy.predict_grip_intent(prepared)
    return action, float(intent.reshape(-1)[0].detach().cpu())


class CorrectionToggle:
    """Non-blocking `c` toggle for PV correction windows; Escape stops deployment."""

    def __init__(self, key: str = "c"):
        if len(key) != 1:
            raise ValueError("PV takeover key must be one character")
        self.key = key.lower()
        self.active = False
        self.stop = False
        self._pressed = False
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                self.stop = True
                return
            char = getattr(key, "char", None)
            if char is not None and char.lower() == self.key and not self._pressed:
                self._pressed = True
                self.active = not self.active
                state = "ON" if self.active else "OFF"
                print(f"[correction] PV takeover {state}")

        def on_release(key):
            char = getattr(key, "char", None)
            if char is not None and char.lower() == self.key:
                self._pressed = False

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class GripInterventionController:
    """Latch a manually adjusted policy target while ACT continues the body trajectory."""

    def __init__(self, *, step: float, key: str = "c", release_threshold: float = 65.0):
        if not 0.0 < step <= 1.0:
            raise ValueError("grip intervention step must be in (0, 1]")
        if len(key) != 1:
            raise ValueError("grip intervention key must be one character")
        self.step = float(step)
        self.key = key.lower()
        self.release_threshold = float(release_threshold)
        self.active = False
        self.paused = False
        self.stop = False
        self._target = None
        self._actual_pos = None
        self._previous_policy_target = None
        self._toggle_requested = False
        self._pending_steps = 0
        self._resume_after_cycle = False
        self.window_name = "Grip intervention: c latch | [ tighten | ] loosen | q stop"

    def request_toggle(self) -> None:
        self._toggle_requested = True

    def request_steps(self, steps: int) -> None:
        self._pending_steps += int(steps)

    def start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 720, 200)

    def poll_input(self) -> None:
        canvas = np.zeros((200, 720, 3), dtype=np.uint8)
        if self.paused:
            status = "PAUSED - adjust q, then press c to resume"
        elif self.active:
            status = "RUNNING WITH MANUAL q"
        else:
            status = "waiting for c at contact or after lift"
        target = "--" if self._target is None else f"{self._target:.2f}"
        actual = "--" if self._actual_pos is None else f"{self._actual_pos:.2f}"
        cv2.putText(
            canvas,
            f"status: {status}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.active else (0, 200, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"q target: {target}    q read: {actual}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"c latch | [ tighten {-self.step:g} | ] loosen +{self.step:g} | q stop",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(self.key):
            self.request_toggle()
            print(f"[grip intervention input] queued {self.key}")
        elif key == ord("["):
            self.request_steps(-1)
            print(f"[grip intervention input] queued tighten {-self.step:g}")
        elif key == ord("]"):
            self.request_steps(1)
            print(f"[grip intervention input] queued loosen +{self.step:g}")
        elif key in {ord("q"), 27}:
            self.stop = True

    def update(self, *, policy_target: float, actual_pos: float) -> tuple[float, dict]:
        self._actual_pos = float(actual_pos)
        previous_policy_target = self._previous_policy_target
        self._previous_policy_target = float(policy_target)
        toggle_requested = self._toggle_requested
        pending_steps = self._pending_steps
        self._toggle_requested = False
        self._pending_steps = 0

        if policy_target >= self.release_threshold and not self.paused:
            self.active = False
            self.paused = False
            self._target = None
            return float(policy_target), {
                "active": False,
                "paused": False,
                "direction": "release",
                "delta_q": 0.0,
                "label_valid": False,
                "resume_after_cycle": False,
            }

        if toggle_requested:
            if not self.active:
                self.active = True
                self.paused = True
                self._target = float(
                    policy_target if previous_policy_target is None else previous_policy_target
                )
                print("[grip intervention] PAUSED; adjust q, then press c to resume")
            elif self.paused:
                self._resume_after_cycle = True
                print("[grip intervention] resume queued after this held cycle")
            else:
                self.paused = True
                print("[grip intervention] PAUSED")

        if not self.active:
            return float(policy_target), {
                "active": False,
                "paused": False,
                "direction": "unknown",
                "delta_q": 0.0,
                "label_valid": False,
                "resume_after_cycle": False,
            }

        if self._target is None:
            self._target = float(actual_pos)
        delta_q = pending_steps * self.step
        self._target = float(np.clip(self._target + delta_q, 0.0, 100.0))
        direction = "hold"
        if pending_steps < 0:
            direction = "tighten"
        elif pending_steps > 0:
            direction = "loosen"
        if pending_steps:
            print(
                f"[grip intervention] {direction} delta_q={delta_q:+g} "
                f"target={self._target:.2f}"
            )
        return self._target, {
            "active": True,
            "paused": self.paused,
            "direction": direction,
            "delta_q": float(delta_q),
            "label_valid": True,
            "resume_after_cycle": self._resume_after_cycle,
        }

    def finish_cycle(self) -> bool:
        if not self._resume_after_cycle:
            return False
        self._resume_after_cycle = False
        self.paused = False
        print("[grip intervention] RUNNING; ACT continues the buffered chunk")
        return True

    def close(self) -> None:
        self.stop = True
        cv2.destroyWindow(self.window_name)


class GripCandidateTrialController:
    """One-shot post-lift loosen trial; activation is an operator lift confirmation."""

    def __init__(self, scorer: GripCandidateScorer, *, load_gate_enabled: bool = True):
        self.scorer = scorer
        self.load_gate_enabled = bool(load_gate_enabled)
        self.active = False
        self.stop = False
        self._activate_requested = False
        self._target = None
        self._actual_pos = None
        self._previous_policy_target = None
        self._last_prediction = None
        self._pending_tighten = False
        self._pending_loosen = False
        self._pending_rollback = False
        self._floor_tighten_count = 0
        self._loosen_count = 0
        self._rolled_back = False
        self._next_adjustment_at = 0.0
        self.window_name = "Grip candidate trial: c after stable lift | q stop"

    def start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 760, 240)

    def poll_input(self) -> None:
        canvas = np.zeros((240, 760, 3), dtype=np.uint8)
        mode = "Load floor 60" if self.load_gate_enabled else "HEAD ONLY - no Load gate"
        status = f"ACTIVE - {mode}" if self.active else "press c only after stable lift"
        target = "--" if self._target is None else f"{self._target:.2f}"
        actual = "--" if self._actual_pos is None else f"{self._actual_pos:.2f}"
        load = "--"
        decision = "waiting"
        if self._last_prediction is not None:
            load = f"{self._last_prediction['present_load_abs']:.0f}"
            decision = f"next delta q: {self._last_prediction['selected_delta_q']:+.1f}"
        lines = (
            (status, 42),
            (f"q target: {target}    q read: {actual}    abs load: {load}", 100),
            (decision, 158),
            ("c activate after lift | q or Esc stop", 216),
        )
        for line, y in lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and not self.active:
            self._activate_requested = True
            print("[grip candidate] activation queued after operator-confirmed stable lift")
        elif key in {ord("q"), 27}:
            self.stop = True

    def update(self, *, policy_target: float, actual_pos: float) -> tuple[float, dict]:
        self._actual_pos = float(actual_pos)
        previous_policy_target = self._previous_policy_target
        self._previous_policy_target = float(policy_target)

        if policy_target >= 65.0:
            self.active = False
            self._target = None
            return float(policy_target), {"active": False, "action": "release", "delta_q": 0.0}

        if self._activate_requested:
            self._activate_requested = False
            self.active = True
            self._target = float(
                policy_target if previous_policy_target is None else previous_policy_target
            )
            self._last_prediction = None
            self._pending_tighten = False
            self._pending_loosen = False
            print(f"[grip candidate] ACTIVE at q={self._target:.2f}; waiting for a fresh score")

        if not self.active:
            return float(policy_target), {"active": False, "action": "policy", "delta_q": 0.0}

        action = "hold"
        delta_q = 0.0
        if self._pending_rollback:
            self._target -= 0.2
            self._pending_rollback = False
            self._pending_loosen = False
            self._rolled_back = True
            action, delta_q = "rollback", -0.2
            self._next_adjustment_at = time.perf_counter() + 0.5
            print(f"[grip candidate] ROLLBACK -0.2 -> q={self._target:.2f}")
        elif self._pending_tighten and self._floor_tighten_count < 3:
            self._target -= 0.2
            self._pending_tighten = False
            self._floor_tighten_count += 1
            action, delta_q = "tighten_to_load_floor", -0.2
            self._next_adjustment_at = time.perf_counter() + 0.5
            print(
                f"[grip candidate] LOAD FLOOR tighten -0.2 -> q={self._target:.2f} "
                f"({self._floor_tighten_count}/3)"
            )
        elif self._pending_loosen and self._loosen_count == 0 and not self._rolled_back:
            self._target += 0.2
            self._pending_loosen = False
            self._loosen_count = 1
            action, delta_q = "loosen", 0.2
            self._next_adjustment_at = time.perf_counter() + 0.5
            print(f"[grip candidate] LOOSEN +0.2 -> q={self._target:.2f}")
        return float(np.clip(self._target, 0.0, 100.0)), {
            "active": True,
            "action": action,
            "delta_q": delta_q,
            "floor_tighten_count": self._floor_tighten_count,
            "loosen_count": self._loosen_count,
            "rolled_back": self._rolled_back,
        }

    def accept_prediction(self, prediction: dict | None) -> None:
        if prediction is None:
            return
        self._last_prediction = prediction
        if not self.active:
            return
        if time.perf_counter() < self._next_adjustment_at:
            return
        hold_probability = prediction["stability_probabilities"]["+0.0"]
        below_load_gate = (
            prediction["present_load_abs"]
            <= prediction["minimum_present_load_for_loosen"]
        )
        if self.load_gate_enabled and self._loosen_count and not self._rolled_back and (
            below_load_gate or hold_probability < self.scorer.minimum_probability
        ):
            self._pending_rollback = True
        elif self.load_gate_enabled and below_load_gate and self._floor_tighten_count < 3:
            self._pending_tighten = True
        elif (
            self._loosen_count == 0
            and not self._rolled_back
            and prediction["selected_delta_q"] == 0.2
        ):
            self._pending_loosen = True

    def close(self) -> None:
        self.stop = True
        cv2.destroyWindow(self.window_name)


class CorrectionRecorder:
    """Write only PV takeover windows using the same training schema as demonstrations."""

    def __init__(self, *, root: Path, repo_id: str, robot, task: str):
        features = build_training_features(robot)
        if root.exists():
            self.dataset = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4)
            validate_dataset_schema(self.dataset.features, features)
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=FPS,
                features=features,
                robot_type=robot.name,
                root=root,
                use_videos=True,
                image_writer_threads=4,
            )
        self.frame_features = {
            key: feature for key, feature in self.dataset.features.items() if key not in PV_AUXILIARY_FEATURES
        }
        self.task = task
        self.saved_segments = 0

    def add(self, *, observation: dict, action: dict, supervision: dict) -> None:
        observation_frame = build_dataset_frame(self.frame_features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.frame_features, action, prefix=ACTION)
        self.dataset.add_frame(
            {
                **observation_frame,
                **action_frame,
                **supervision,
                HUMAN_INTERVENTION_FEATURE: np.asarray([1.0], dtype=np.float32),
                "task": self.task,
            }
        )

    def end_segment(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.save_episode()
            self.saved_segments += 1
            print(f"[correction] saved PV takeover segment {self.saved_segments}")

    def discard_pending(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def close(self) -> None:
        self.dataset.finalize()


class DeploymentEvidence:
    """Record the exact observations, predictions, commands, and dual-view video."""

    def __init__(
        self,
        path: Path,
        *,
        policy: str,
        task: str,
        arm_enabled: bool,
        start_mode: str,
        policy_action_steps: int | None,
        motors: list[str],
        image_features: set[str],
        gripper_only: bool = False,
        gripper_close_offset: float = 0.0,
        gripper_telemetry_hz: float = 0.0,
        grip_residual_shadow_model: str | None = None,
        grip_candidate_trial_model: str | None = None,
        grip_candidate_load_gate: bool = True,
        grip_intervention_step: float = 0.0,
        action_step_repeat: int = 1,
        max_steps: int | None = None,
        start_joints: list[float] | None = None,
    ):
        self.path = Path(path)
        if self.path.exists():
            if not self.path.is_dir() or any(self.path.iterdir()):
                raise ValueError(f"refusing to overwrite non-empty evidence directory: {self.path}")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
        self.control_path = self.path / "control.jsonl"
        self._control = self.control_path.open("x", encoding="utf-8")
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._closed = False
        self.motors = list(motors)
        self.image_names = sorted(feature.rsplit(".", 1)[-1] for feature in image_features)
        self.manifest = {
            "schema_version": 8,
            "created_at_s": time.time(),
            "policy": policy,
            "task": task,
            "arm_enabled": arm_enabled,
            "start_mode": start_mode,
            "policy_action_steps": policy_action_steps,
            "gripper_only": gripper_only,
            "gripper_close_offset": gripper_close_offset,
            "gripper_telemetry_hz": gripper_telemetry_hz,
            "grip_residual_shadow_model": grip_residual_shadow_model,
            "grip_candidate_trial_model": grip_candidate_trial_model,
            "grip_candidate_load_gate": grip_candidate_load_gate,
            "grip_intervention_step": grip_intervention_step,
            "action_step_repeat": action_step_repeat,
            "max_steps": max_steps,
            "start_joints": start_joints,
            "motors": self.motors,
            "image_names": self.image_names,
            "control_log": self.control_path.name,
            "videos": {name: f"{name}.avi" for name in self.image_names},
            "status": "running",
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_video_frame(self, name: str, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        height, width = frame.shape[:2]
        writer = self._writers.get(name)
        if writer is None:
            path = self.path / f"{name}.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                float(FPS),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open deployment evidence writer: {path}")
            self._writers[name] = writer
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add(
        self,
        *,
        step: int,
        elapsed_s: float,
        inference_ms: float,
        observation: dict,
        state: np.ndarray,
        predicted_action: np.ndarray,
        planned_action: np.ndarray,
        action_trace: dict | None,
        bus_action: dict | None,
        readback_state: dict | None,
        gripper_telemetry: dict | None = None,
        grip_residual_shadow: dict | None = None,
        grip_candidate_trial: dict | None = None,
        grip_intervention: dict | None = None,
        command_sent: bool,
    ) -> None:
        for name in self.image_names:
            self._write_video_frame(name, observation[name])
        row = {
            "step": int(step),
            "elapsed_s": float(elapsed_s),
            "inference_ms": float(inference_ms),
            "state": {motor: float(state[index]) for index, motor in enumerate(self.motors)},
            "predicted_action": {
                motor: float(predicted_action[index]) for index, motor in enumerate(self.motors)
            },
            "planned_action": {
                motor: float(planned_action[index]) for index, motor in enumerate(self.motors)
            },
            "bus_target": (
                None
                if bus_action is None
                else {motor: float(bus_action[f"{motor}.pos"]) for motor in self.motors}
            ),
            "readback_state": (
                None
                if readback_state is None
                else {motor: float(readback_state[f"{motor}.pos"]) for motor in self.motors}
            ),
            "gripper_telemetry": gripper_telemetry,
            "grip_residual_shadow": grip_residual_shadow,
            "grip_candidate_trial": grip_candidate_trial,
            "grip_intervention": grip_intervention,
            "command_sent": bool(command_sent),
        }
        if action_trace is not None:
            raw_action = action_trace["raw_normalized_action"].detach().cpu().numpy().reshape(-1)
            raw_chunk = action_trace["raw_normalized_chunk"].detach().cpu().numpy()[0]
            denormalized_chunk = action_trace["denormalized_chunk"].detach().cpu().numpy()[0]
            row.update(
                {
                    "policy_chunk_id": int(action_trace["chunk_id"]),
                    "policy_execution_index": int(action_trace["execution_index"]),
                    "raw_normalized_action": {
                        motor: float(raw_action[index]) for index, motor in enumerate(self.motors)
                    },
                    "raw_normalized_chunk": [
                        {motor: float(action[index]) for index, motor in enumerate(self.motors)}
                        for action in raw_chunk
                    ],
                    "denormalized_chunk": [
                        {motor: float(action[index]) for index, motor in enumerate(self.motors)}
                        for action in denormalized_chunk
                    ],
                }
            )
        self._control.write(json.dumps(row, sort_keys=True) + "\n")
        self._control.flush()

    def close(
        self,
        *,
        status: str,
        elapsed_s: float,
        steps: int,
        commands_sent: int,
        error: str | None = None,
    ) -> None:
        if self._closed:
            return
        for writer in self._writers.values():
            writer.release()
        self._control.close()
        self.manifest.update(
            {
                "status": status,
                "ended_at_s": time.time(),
                "elapsed_s": float(elapsed_s),
                "steps": int(steps),
                "commands_sent": int(commands_sent),
                "achieved_hz": 0.0 if elapsed_s <= 0.0 else float(steps / elapsed_s),
                "error": error,
            }
        )
        self._write_manifest()
        self._closed = True


def _apply_diffusion_overrides(policy, scheduler, steps, tag="deploy"):
    """Optionally swap the diffusion sampler and/or set denoising steps. DDIM samples well in
    ~10 steps even for a DDPM-trained model (DDPM itself needs ~100 -> slow). No effect on ACT."""
    if not hasattr(policy, "diffusion"):
        return
    d = policy.diffusion
    if scheduler is not None:
        from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
        c = d.noise_scheduler.config
        d.noise_scheduler = _make_noise_scheduler(
            scheduler.upper(),
            num_train_timesteps=c.num_train_timesteps, beta_start=c.beta_start, beta_end=c.beta_end,
            beta_schedule=c.beta_schedule, clip_sample=c.clip_sample,
            clip_sample_range=c.clip_sample_range, prediction_type=c.prediction_type)
        print(f"[{tag}] diffusion scheduler -> {scheduler.upper()}")
    if steps is not None:
        d.num_inference_steps = steps
        print(f"[{tag}] diffusion denoising steps -> {steps}")


def _apply_act_action_steps(policy, steps: int | None) -> None:
    if steps is None:
        return
    if policy.config.type != "act":
        raise ValueError("--act-action-steps requires an ACT checkpoint")
    if not 1 <= steps <= policy.config.chunk_size:
        raise ValueError(
            f"--act-action-steps must be between 1 and chunk_size={policy.config.chunk_size}"
        )
    policy.config.n_action_steps = steps
    policy.reset()
    print(f"[deploy] ACT action queue -> {steps} steps per inference")


def _apply_smolvla_action_steps(policy, steps: int | None) -> None:
    if steps is None:
        return
    if policy.config.type != "smolvla":
        raise ValueError("--smolvla-action-steps requires a SmolVLA checkpoint")
    if not 1 <= steps <= policy.config.chunk_size:
        raise ValueError(
            f"--smolvla-action-steps must be between 1 and chunk_size={policy.config.chunk_size}"
        )
    policy.config.n_action_steps = steps
    policy.reset()
    print(f"[deploy] SmolVLA action queue -> {steps} steps per inference")


def _ready_pose(robot) -> dict:
    """The same down-ready pose the demos started from: centred joints, wrist pitched down 90 deg."""
    pose = {f"{m}.pos": joint_center(robot.bus.motors[m].norm_mode.value)
            for m in robot.bus.motors}
    pose["wrist_flex.pos"] = MIDDLE_WRIST_DOWN_DEG
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="HF Hub repo id or local dir")
    # Stable by-id symlink: the arm's /dev/ttyACM* index changes across replugs (was ACM1, then ACM0),
    # but this serial-number path is constant. Pass --port /dev/ttyACMx to override.
    ap.add_argument("--port",
                    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
                    help="SO-101 serial port (default: stable by-id symlink)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to run before stopping")
    ap.add_argument(
        "--max-steps",
        type=int,
        help="Stop after exactly this many control cycles; 0 performs only the start-joint ramp.",
    )
    ap.add_argument("--task", default=TASK, help="Language task supplied to text-conditioned policies.")
    ap.add_argument(
        "--arm-enabled",
        action="store_true",
        help="Allow policy goal commands. Default is inference only.",
    )
    ap.add_argument(
        "--start-mode",
        choices=("current", "ready"),
        default="current",
        help="current starts from the observed pose; ready opts into the legacy fixed-pose ramp.",
    )
    ap.add_argument(
        "--start-joints",
        help="Optional comma-separated joint targets in robot motor order; ramps before the run.",
    )
    ap.add_argument(
        "--gripper-only",
        action="store_true",
        help="Hold all body joints at each readback and execute only the policy gripper target.",
    )
    ap.add_argument(
        "--gripper-close-offset",
        type=float,
        default=0.0,
        help="Direct mode only: add this non-negative offset when the predicted gripper target "
        "is below 90; positive values are looser and open/release actions are unchanged.",
    )
    ap.add_argument(
        "--act-action-steps",
        type=int,
        help="ACT only: execute this many actions from each predicted chunk before replanning.",
    )
    ap.add_argument(
        "--action-step-repeat",
        type=int,
        default=1,
        help="Repeat each policy position target for this many 10 Hz control frames; use 2 for half-speed trials.",
    )
    ap.add_argument(
        "--smolvla-action-steps",
        type=int,
        help="SmolVLA only: execute this many actions from each predicted chunk before replanning.",
    )
    ap.add_argument(
        "--evidence-dir",
        type=Path,
        help="Fresh directory for manifest, control JSONL, and observation videos.",
    )
    ap.add_argument(
        "--gripper-telemetry-hz",
        type=float,
        default=0.0,
        help="Sample raw gripper Present_Current/Present_Load into evidence; 0 disables it.",
    )
    ap.add_argument(
        "--grip-residual-shadow-model",
        type=Path,
        help="Run a trained numeric grip head on telemetry and log its next-step suggestion; "
        "never changes a motor command.",
    )
    ap.add_argument(
        "--grip-candidate-trial-model",
        type=Path,
        help="Bounded post-lift trial: press c after stable lift; the validated candidate head "
        "may loosen once by +0.2 with its stored Present_Load gate.",
    )
    ap.add_argument(
        "--grip-candidate-no-load-gate",
        action="store_true",
        help="Bounded research ablation: one learned +0.2 maximum, with no Load veto, "
        "floor tightening, or automatic rollback.",
    )
    ap.add_argument(
        "--grip-intervention-step",
        type=float,
        default=0.0,
        help="Enable direct labeled grip intervention with this q step; use 0.2. "
        "Focus the intervention window and press the takeover key to latch current q, "
        "'[' to tighten, or ']' to loosen.",
    )
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="Diffusion only: # of denoising steps per inference. DDPM needs ~100 (slow); "
                         "with --scheduler ddim, ~10 is enough. No effect on ACT.")
    ap.add_argument("--scheduler", choices=["ddpm", "ddim"], default=None,
                    help="Diffusion only: override the sampler. 'ddim' + '--inference-steps 10' gives "
                         "~10x faster inference at ~full quality (the right speed fix). 'ddpm' is the "
                         "trained default.")
    ap.add_argument("--grip-context", choices=GRIP_CONTEXTS, default="unknown",
                    help="Deployable object context appended to observation.state for grip-aux policies.")
    ap.add_argument("--grip-control", choices=("direct", "aux"), default="direct",
                    help="direct keeps the existing 6D position action; aux delegates grasp force to "
                         "the auxiliary head and low-level position feedback.")
    ap.add_argument("--grip-light-pos", type=float,
                    help="Calibrated light-contact gripper position required by aux/PV control.")
    ap.add_argument("--grip-hard-pos", type=float,
                    help="Calibrated hard-contact gripper position required by aux/PV control.")
    ap.add_argument("--grip-max-step", type=float, default=2.0,
                    help="Maximum low-level gripper position change per control frame.")
    ap.add_argument("--correction-dataset-root", type=Path,
                    help="If set, press the takeover key to record PV-controlled recovery segments.")
    ap.add_argument("--correction-repo-id", default=DEFAULT_CORRECTION_REPO)
    ap.add_argument("--pv-port", type=int, default=8090)
    ap.add_argument("--takeover-key", default="c")
    ap.add_argument("--front-camera", default=WORKSPACE_CAM_PATH)
    ap.add_argument("--side-camera", default=SIDE_CAM_PATH)
    args = ap.parse_args()

    if args.duration <= 0.0:
        ap.error("--duration must be positive")
    if args.arm_enabled and args.evidence_dir is None:
        ap.error("--arm-enabled requires --evidence-dir")
    if args.start_joints is not None and not args.arm_enabled:
        ap.error("--start-joints requires --arm-enabled")
    if args.start_joints is not None and args.start_mode == "ready":
        ap.error("--start-joints cannot be combined with --start-mode ready")
    if args.max_steps is not None and args.max_steps < 0:
        ap.error("--max-steps must be non-negative")
    if not 1 <= args.action_step_repeat <= 4:
        ap.error("--action-step-repeat must be between 1 and 4")
    if args.gripper_close_offset < 0.0:
        ap.error("--gripper-close-offset must be non-negative")
    if args.gripper_close_offset and args.grip_control != "direct":
        ap.error("--gripper-close-offset requires --grip-control direct")
    if args.gripper_telemetry_hz < 0.0:
        ap.error("--gripper-telemetry-hz must be non-negative")
    if args.gripper_telemetry_hz and not args.arm_enabled:
        ap.error("--gripper-telemetry-hz requires --arm-enabled")
    if args.grip_residual_shadow_model is not None:
        if args.grip_control != "direct":
            ap.error("--grip-residual-shadow-model requires --grip-control direct")
        if args.evidence_dir is None or args.gripper_telemetry_hz <= 0.0:
            ap.error(
                "--grip-residual-shadow-model requires --evidence-dir and "
                "--gripper-telemetry-hz > 0"
            )
    if args.grip_candidate_trial_model is not None:
        if not args.arm_enabled or args.evidence_dir is None or args.gripper_telemetry_hz <= 0.0:
            ap.error(
                "--grip-candidate-trial-model requires --arm-enabled, --evidence-dir, and "
                "--gripper-telemetry-hz > 0"
            )
        if args.grip_control != "direct" or args.gripper_close_offset:
            ap.error("grip candidate trial requires direct control with zero close offset")
        if args.gripper_only:
            ap.error("grip candidate trial must preserve the ACT body trajectory")
        if args.grip_residual_shadow_model is not None or args.grip_intervention_step:
            ap.error("grip candidate trial cannot be combined with another grip head/intervention")
        if args.correction_dataset_root is not None:
            ap.error("grip candidate trial cannot be combined with PV correction recording")
    elif args.grip_candidate_no_load_gate:
        ap.error("--grip-candidate-no-load-gate requires --grip-candidate-trial-model")
    if args.grip_intervention_step < 0.0 or args.grip_intervention_step > 1.0:
        ap.error("--grip-intervention-step must be in [0, 1]")
    if args.grip_intervention_step:
        if not args.arm_enabled or args.evidence_dir is None or args.gripper_telemetry_hz <= 0.0:
            ap.error(
                "--grip-intervention-step requires --arm-enabled, --evidence-dir, and "
                "--gripper-telemetry-hz > 0"
            )
        if args.grip_control != "direct" or args.gripper_close_offset:
            ap.error("grip intervention requires direct control with zero close offset")
        if args.gripper_only:
            ap.error("grip intervention must execute the ACT body trajectory, not --gripper-only")
        if args.correction_dataset_root is not None:
            ap.error("grip intervention cannot be combined with PV correction recording")
    if not args.arm_enabled and args.correction_dataset_root is not None:
        ap.error("--correction-dataset-root requires --arm-enabled")
    needs_feedback = args.grip_control == "aux" or args.correction_dataset_root is not None
    if needs_feedback and (args.grip_light_pos is None or args.grip_hard_pos is None):
        ap.error("aux/PV control requires --grip-light-pos and --grip-hard-pos from calibration")
    if len(args.takeover_key) != 1:
        ap.error("--takeover-key must be one character")

    device = torch.device(args.device)
    print(f"[deploy] loading policy {args.policy} on {device} ...")
    policy, pre, post = _load_policy(args.policy, device)
    _apply_diffusion_overrides(policy, args.scheduler, args.inference_steps)
    try:
        _apply_act_action_steps(policy, args.act_action_steps)
        _apply_smolvla_action_steps(policy, args.smolvla_action_steps)
    except ValueError as exc:
        ap.error(str(exc))
    if args.action_step_repeat > 1:
        print(
            f"[deploy] each policy action repeats for {args.action_step_repeat} control frames "
            f"({1 / args.action_step_repeat:.2f}x trajectory speed)"
        )
    if args.grip_control == "aux" and not hasattr(policy, "predict_grip_intent"):
        ap.error("--grip-control aux requires a grip_aux policy checkpoint")
    action_tracer = None if args.grip_control == "aux" else PolicyChunkTrace(policy, post)

    grip_residual_shadow = None
    if args.grip_residual_shadow_model is not None:
        try:
            grip_residual_shadow = GripResidualShadow.from_checkpoint(
                args.grip_residual_shadow_model
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            ap.error(f"invalid grip residual shadow checkpoint: {exc}")
        print(f"[deploy] grip residual shadow -> {args.grip_residual_shadow_model}")

    grip_candidate_scorer = None
    if args.grip_candidate_trial_model is not None:
        try:
            grip_candidate_scorer = GripCandidateScorer.from_checkpoint(
                args.grip_candidate_trial_model
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            ap.error(f"invalid grip candidate trial checkpoint: {exc}")
        print(f"[deploy] bounded grip candidate trial -> {args.grip_candidate_trial_model}")

    grip_feedback = None
    if needs_feedback:
        grip_feedback = GripFeedbackController(
            GripFeedbackConfig(
                light_pos=args.grip_light_pos,
                hard_pos=args.grip_hard_pos,
                max_step=args.grip_max_step,
            )
        )

    try:
        camera_input_features = _policy_camera_inputs(policy, pre)
    except ValueError as exc:
        ap.error(str(exc))
    if grip_candidate_scorer is not None and not {
        "observation.images.front",
        "observation.images.side",
    }.issubset(camera_input_features):
        ap.error("grip candidate trial requires both front and side policy cameras")
    cameras = {}
    if "observation.images.front" in camera_input_features:
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=args.front_camera,
            width=640,
            height=480,
            fps=FPS,
            fourcc=WORKSPACE_CAM_FOURCC,
            warmup_s=3,
        )
    if "observation.images.side" in camera_input_features:
        cameras["side"] = OpenCVCameraConfig(
            index_or_path=args.side_camera,
            width=640,
            height=480,
            # Etron exposes 640x480 YUYV only at 30 Hz; the control loop samples at 10 Hz.
            fps=SIDE_CAMERA_FPS,
            fourcc=SIDE_CAM_FOURCC,
            warmup_s=3,
        )
    robot = PVRecorderRobot(
        SO101FollowerConfig(
            port=args.port,
            id=ARM_ID,
            use_degrees=True,
            cameras=cameras,
            disable_torque_on_disconnect=False,
        ),
        grip_context=args.grip_context,
    )
    correction_source = None
    correction_recorder = None
    correction_toggle = None
    grip_intervention = None
    grip_candidate_trial = None
    evidence = None
    connected = False
    normal_exit = False
    run_status = "failed"
    run_error = None
    run_started = None
    n = 0
    evidence_step = 0
    paused_elapsed_s = 0.0
    commands_sent = 0

    try:
        robot.connect(calibrate=False)
        connected = True
        motors = list(robot.bus.motors.keys())
        if "gripper" not in motors:
            raise ValueError(f"SO-101 motor list has no gripper: {motors}")
        gripper_index = motors.index("gripper")
        start_values = (
            None
            if args.start_joints is None
            else [float(value) for value in args.start_joints.split(",")]
        )
        if start_values is not None and len(start_values) != len(motors):
            raise ValueError(f"--start-joints has {len(start_values)} values for {len(motors)} motors")
        if args.evidence_dir is not None:
            evidence = DeploymentEvidence(
                args.evidence_dir,
                policy=args.policy,
                task=args.task,
                arm_enabled=args.arm_enabled,
                start_mode=args.start_mode,
                policy_action_steps=getattr(policy.config, "n_action_steps", None),
                motors=motors,
                image_features=camera_input_features,
                gripper_only=args.gripper_only,
                gripper_close_offset=args.gripper_close_offset,
                gripper_telemetry_hz=args.gripper_telemetry_hz,
                grip_residual_shadow_model=(
                    None
                    if args.grip_residual_shadow_model is None
                    else str(args.grip_residual_shadow_model)
                ),
                grip_candidate_trial_model=(
                    None
                    if args.grip_candidate_trial_model is None
                    else str(args.grip_candidate_trial_model)
                ),
                grip_candidate_load_gate=not args.grip_candidate_no_load_gate,
                grip_intervention_step=args.grip_intervention_step,
                action_step_repeat=args.action_step_repeat,
                max_steps=args.max_steps,
                start_joints=start_values,
            )
        if args.correction_dataset_root is not None:
            correction_recorder = CorrectionRecorder(
                root=args.correction_dataset_root,
                repo_id=args.correction_repo_id,
                robot=robot,
                task=args.task,
            )
            correction_source = _build_pressure_source(args.pv_port)
            correction_toggle = CorrectionToggle(args.takeover_key)
            correction_toggle.start()
            print(
                f"[correction] press {args.takeover_key!r} to toggle PV takeover; "
                "only takeover windows are added to the correction dataset"
            )
        if args.grip_intervention_step:
            grip_intervention = GripInterventionController(
                step=args.grip_intervention_step,
                key=args.takeover_key,
            )
            grip_intervention.start()
            print(
                f"[grip intervention] focus the control window and press "
                f"{args.takeover_key!r} at contact or after lift to pause and latch q; '[' tightens "
                f"{args.grip_intervention_step:g}, and ']' loosens "
                f"{args.grip_intervention_step:g}; press {args.takeover_key!r} again to continue "
                "the buffered ACT chunk"
            )
        if grip_candidate_scorer is not None:
            grip_candidate_trial = GripCandidateTrialController(
                grip_candidate_scorer,
                load_gate_enabled=not args.grip_candidate_no_load_gate,
            )
            grip_candidate_trial.start()
            mode = (
                "HEAD ONLY: no Load gate or automatic rollback"
                if args.grip_candidate_no_load_gate
                else "Load floor enabled"
            )
            print(
                "[grip candidate] press 'c' only after the carton is stably lifted; "
                f"one automatic +0.2 maximum; {mode}; q stops"
            )

        if args.arm_enabled:
            print("ARM ENABLED: policy goal commands will be sent.")
            if start_values is not None:
                start_target = {
                    f"{motor}.pos": value for motor, value in zip(motors, start_values, strict=True)
                }
                print(f"[deploy] ramping to explicit start joints: {start_target}")
                _ramp_to(robot, start_target)
                _read_positions(robot)
            elif args.start_mode == "ready":
                print("[deploy] ramping to down-ready pose ...")
                _ramp_to(robot, _ready_pose(robot))
                _read_positions(robot)
            else:
                positions = _read_positions(robot)
                print(f"[deploy] starting from current pose: {positions}")
            print(f"[deploy] RUNNING autonomously for {args.duration:.0f}s  (Ctrl-C to stop) ...")
        else:
            print("ARM LOCKED: inference only; no ready-pose ramp or goal commands will be sent.")
            print(f"[deploy] OBSERVING for {args.duration:.0f}s  (Ctrl-C to stop) ...")
        print(f"[deploy] grip_context={args.grip_context}; PV is not a policy observation")
        period = 1.0 / FPS
        telemetry_period = (
            None if args.gripper_telemetry_hz == 0.0 else 1.0 / args.gripper_telemetry_hz
        )
        last_telemetry_at = None
        last_predicted_action = None
        repeat_remaining = 0
        pending_action_count = False
        run_started = time.perf_counter()
        t_end = run_started + args.duration
        was_takeover = False
        candidate_frames = (
            None
            if grip_candidate_scorer is None
            else deque(maxlen=grip_candidate_scorer.visual_gap_frames + 1)
        )
        while time.perf_counter() < t_end and (args.max_steps is None or n < args.max_steps) and not (
            (correction_toggle is not None and correction_toggle.stop)
            or (grip_intervention is not None and grip_intervention.stop)
            or (grip_candidate_trial is not None and grip_candidate_trial.stop)
        ):
            t0 = time.perf_counter()
            if grip_intervention is not None:
                grip_intervention.poll_input()
            if grip_candidate_trial is not None:
                grip_candidate_trial.poll_input()
            obs = robot.get_observation()
            if candidate_frames is not None:
                candidate_frames.append((obs["front"], obs["side"]))
            state = _policy_state(obs, motors, policy, args.grip_context)
            policy_obs = {"observation.state": state}
            for feature in camera_input_features:
                policy_obs[feature] = obs[feature.rsplit(".", 1)[-1]]
            grip_intent = None
            action_trace = None
            if grip_intervention is not None and grip_intervention.paused:
                if last_predicted_action is None:
                    raise RuntimeError("grip intervention paused before the first policy action")
                a = last_predicted_action.copy()
                inference_ms = 0.0
            elif repeat_remaining:
                a = last_predicted_action.copy()
                repeat_remaining -= 1
                inference_ms = 0.0
            else:
                inference_started = time.perf_counter()
                if args.grip_control == "aux":
                    action, grip_intent = _predict_action_and_grip(
                        policy_obs, policy, device, pre, post, args.task
                    )
                else:
                    action, action_trace = _predict_action_with_trace(
                        policy_obs, action_tracer, device, pre, args.task, robot.name
                    )
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                a = action.cpu().numpy().reshape(-1)
                last_predicted_action = a.copy()
                repeat_remaining = args.action_step_repeat - 1
                pending_action_count = True
            if a.size != len(motors):
                raise ValueError(f"policy returned {a.size} actions for {len(motors)} motors")
            predicted_action = a.copy()
            a = apply_gripper_close_offset(
                a,
                gripper_index=gripper_index,
                offset=args.gripper_close_offset,
            )

            takeover = bool(correction_toggle is not None and correction_toggle.active)
            if was_takeover and not takeover:
                correction_recorder.end_segment()
                correction_source.reset()
                grip_feedback.reset()
            elif takeover and not was_takeover:
                correction_source.reset()
            was_takeover = takeover

            pv_target = np.asarray([0.0], dtype=np.float32)
            pv_valid = np.asarray([0.0], dtype=np.float32)
            pv_supervision = None
            if correction_source is not None:
                reading = correction_source.update(None, pinch=0.0, enabled=takeover)
                pv_target, pv_valid = pv_teacher_label(reading)
                pv_supervision = pv_supervision_from_reading(reading)

            actual_gripper = float(obs["gripper.pos"])
            grip_intervention_label = None
            grip_candidate_control = None
            if takeover and bool(pv_valid[0]):
                a[gripper_index] = grip_feedback.update(
                    policy_target=float(a[gripper_index]),
                    grip_intent=float(pv_target[0]),
                    actual_pos=actual_gripper,
                    force_grasp=True,
                )
            elif args.grip_control == "aux":
                a[gripper_index] = grip_feedback.update(
                    policy_target=float(a[gripper_index]),
                    grip_intent=grip_intent,
                    actual_pos=actual_gripper,
                )
            elif grip_intervention is not None:
                a[gripper_index], grip_intervention_label = grip_intervention.update(
                    policy_target=float(a[gripper_index]),
                    actual_pos=actual_gripper,
                )
            elif grip_candidate_trial is not None:
                a[gripper_index], grip_candidate_control = grip_candidate_trial.update(
                    policy_target=float(a[gripper_index]),
                    actual_pos=actual_gripper,
                )
            paused_cycle = grip_intervention is not None and grip_intervention.paused
            if args.gripper_only:
                a = hold_body_action(a, state, gripper_index=gripper_index)
            # A paused intervention already reuses last_predicted_action. Keep that
            # fixed body target instead of replacing it with lagging readback.

            planned_action = {f"{motor}.pos": float(a[i]) for i, motor in enumerate(motors)}
            command_sent = False
            bus_action = None
            readback_state = None
            gripper_telemetry = None
            grip_residual_prediction = None
            grip_candidate_prediction = None
            if args.arm_enabled:
                bus_action = robot.send_action(planned_action)
                command_sent = True
                commands_sent += 1
                if evidence is not None:
                    readback_state = {
                        f"{motor}.pos": float(value)
                        for motor, value in read_present_positions(robot.bus).items()
                    }
                    now = time.perf_counter()
                    if telemetry_period is not None and (
                        last_telemetry_at is None or now - last_telemetry_at >= telemetry_period
                    ):
                        last_telemetry_at = now
                        q_cmd = float(bus_action["gripper.pos"])
                        q_read = float(readback_state["gripper.pos"])
                        gripper_telemetry = {
                            "sample_elapsed_s": now - run_started,
                            "present_current": _read_reg(robot, "Present_Current", "gripper"),
                            "present_load": _read_reg(robot, "Present_Load", "gripper"),
                            "position_lag": q_read - q_cmd,
                            "absolute_position_lag": abs(q_read - q_cmd),
                        }
                        if grip_residual_shadow is not None:
                            shadow_started = time.perf_counter()
                            grip_residual_prediction = grip_residual_shadow.observe(
                                policy_target=float(predicted_action[gripper_index]),
                                command_target=q_cmd,
                                actual_pos=q_read,
                                present_current=float(gripper_telemetry["present_current"]),
                                present_load=float(gripper_telemetry["present_load"]),
                                position_lag=float(gripper_telemetry["position_lag"]),
                            )
                            if grip_residual_prediction is not None:
                                grip_residual_prediction["inference_ms"] = (
                                    time.perf_counter() - shadow_started
                                ) * 1000.0
                        if (
                            grip_candidate_scorer is not None
                            and candidate_frames is not None
                            and len(candidate_frames) == candidate_frames.maxlen
                        ):
                            candidate_started = time.perf_counter()
                            previous_front, previous_side = candidate_frames[0]
                            current_front, current_side = candidate_frames[-1]
                            grip_candidate_prediction = grip_candidate_scorer.observe(
                                policy_target=float(predicted_action[gripper_index]),
                                command_target=q_cmd,
                                actual_pos=q_read,
                                present_current=float(gripper_telemetry["present_current"]),
                                present_load=float(gripper_telemetry["present_load"]),
                                position_lag=float(gripper_telemetry["position_lag"]),
                                previous_front_rgb=previous_front,
                                current_front_rgb=current_front,
                                previous_side_rgb=previous_side,
                                current_side_rgb=current_side,
                                use_load_gate=grip_candidate_trial.load_gate_enabled,
                            )
                            if grip_candidate_prediction is not None:
                                grip_candidate_prediction["inference_ms"] = (
                                    time.perf_counter() - candidate_started
                                ) * 1000.0
                            grip_candidate_trial.accept_prediction(grip_candidate_prediction)
            if takeover:
                correction_recorder.add(
                    observation=obs,
                    action=planned_action,
                    supervision=pv_supervision,
                )
            if evidence is not None:
                evidence.add(
                    step=evidence_step,
                    elapsed_s=time.perf_counter() - run_started,
                    inference_ms=inference_ms,
                    observation=obs,
                    state=state,
                    predicted_action=predicted_action,
                    planned_action=a,
                    action_trace=action_trace,
                    bus_action=bus_action,
                    readback_state=readback_state,
                    gripper_telemetry=gripper_telemetry,
                    grip_residual_shadow=grip_residual_prediction,
                    grip_candidate_trial=(
                        None
                        if grip_candidate_trial is None
                        else {
                            "control": grip_candidate_control,
                            "prediction": grip_candidate_prediction,
                        }
                    ),
                    grip_intervention=grip_intervention_label,
                    command_sent=command_sent,
                )
            if grip_intervention is not None:
                grip_intervention.finish_cycle()
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
            cycle_elapsed_s = time.perf_counter() - t0
            evidence_step += 1
            if paused_cycle:
                paused_elapsed_s += cycle_elapsed_s
                t_end += cycle_elapsed_s
            elif pending_action_count:
                n += 1
                pending_action_count = False
        elapsed = time.perf_counter() - run_started
        active_elapsed = elapsed - paused_elapsed_s
        print(f"[deploy] done — {n} steps ({n / active_elapsed:.1f} Hz).")
        normal_exit = True
        run_status = "complete"
    except KeyboardInterrupt:
        print("\n[deploy] stopped by user.")
        normal_exit = True
        run_status = "stopped"
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if correction_toggle is not None:
            correction_toggle.close()
        if grip_intervention is not None:
            grip_intervention.close()
        if grip_candidate_trial is not None:
            grip_candidate_trial.close()
        if correction_source is not None:
            correction_source.close()
        if correction_recorder is not None:
            if normal_exit:
                correction_recorder.end_segment()
            else:
                correction_recorder.discard_pending()
            correction_recorder.close()
        elapsed = 0.0 if run_started is None else time.perf_counter() - run_started
        if evidence is not None:
            evidence.close(
                status=run_status,
                elapsed_s=elapsed,
                steps=n,
                commands_sent=commands_sent,
                error=run_error,
            )
            print(f"[deploy] evidence: {evidence.path}")
        if connected:
            # disconnect with torque held (disable_torque_on_disconnect=False) so the arm keeps its pose.
            robot.disconnect()


if __name__ == "__main__":
    main()
