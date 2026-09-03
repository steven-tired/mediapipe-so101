"""Record a PV-supervised SO-101 demonstration for grip-context training.

The action remains the established six joint positions. ``observation.state`` adds a deployable
one-hot grip context, while PV supplies masked privileged labels that are excluded from policy
observations at deployment. The independent evidence sidecar retains the full runtime audit trail.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
import json
from pathlib import Path
import os
import shutil
import time
import uuid

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.feature_utils import combine_feature_dicts

from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
from lerobot_teleoperator_so101_webcam.ee_control import joint_center
from lerobot_teleoperator_so101_webcam.ee_controller import (
    MAX_WRIST_ROLL_RANGE_DEG,
    WebcamEEController,
)
from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    GripperClosureLimits,
    GripperTelemetrySampler,
)
from lerobot_teleoperator_so101_webcam.hand_startup_gate import (
    HAND_STARTUP_DWELL_S,
    ContinuousHandStartupGate,
)
from lerobot_teleoperator_so101_webcam.paths import dataset_root as default_dataset_root
from lerobot_teleoperator_so101_webcam.paths import evidence_dir as default_evidence_dir
from lerobot_teleoperator_so101_webcam.paths import urdf_path
from pressurevision_integration.pv_grip_adapter import PVGripAdapter
from pressurevision_integration.pv_grip_controller import (
    PressureVisionGripRuntime,
    pressure_range_mapping_contract,
)
from pressurevision_integration.pv_object_profile import (
    load_object_profile,
    object_profile_sha256,
)
from pressurevision_integration.pv_pressure import (
    LatestFrameSource,
    PressureVisionSource,
    PressureVisionUDPSource,
)
from pressurevision_integration.pv_episode_review import (
    OUTCOME_FAILURE,
    ReviewFrame,
    interactive_review,
    outcome_record,
    write_review_artifacts,
)
from pressurevision_integration.pv_preview import (
    DEFAULT_PV_PREVIEW_SHARE,
    PressureVisionPreviewSource,
    draw_gripper_position_banner,
)
from pressurevision_integration.pv_shadow_telemetry import (
    PV_SHADOW_SCHEMA_VERSION,
    PVShadowTelemetryLogger,
    pv_shadow_sample,
)
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator


# Stable by-id symlinks: /dev/ttyACM* and /dev/video* indices flip across
# replugs. Each is overridable for a different rig.
ARM_PORT = os.environ.get(
    "SO101_ARM_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
)
ARM_ID = os.environ.get("SO101_ARM_ID", "so101_follower_1")
WORKSPACE_CAM_PATH = os.environ.get(
    "SO101_WORKSPACE_CAM",
    "/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0",
)
WORKSPACE_CAM_FOURCC = "YUYV"
SIDE_CAM_PATH = os.environ.get(
    "SO101_SIDE_CAM",
    "/dev/v4l/by-id/usb-Etron_Technology__Inc._USB2.0_Camera-video-index0",
)
SIDE_CAM_FOURCC = "YUYV"
HAND_CAMERA_INDEX = 0
DEFAULT_REPO_ID = "local/hand_tracking_pv_carton_dual_view"
DATASET_NAME = "hand_tracking_pv_carton_dual_view"
DEFAULT_DATASET_ROOT = default_dataset_root() / DATASET_NAME
DEFAULT_EVIDENCE_ROOT = default_evidence_dir() / DATASET_NAME
DEFAULT_TASK = "hand-tracking PV pick and place"
DEFAULT_EPISODES = 1
DEFAULT_EPISODE_SECONDS = 120
DEFAULT_FPS = 10
DEFAULT_PV_PORT = 8090
DEFAULT_LEVEL_MAX_AGE_MINUTES = 180.0

# The live controller keeps its historical two position-units/control-frame limit.  At 10 Hz the
# recorder uses six so that its gripper slew is comparable to the roughly 30 Hz live path.
LIVE_POSITION_PER_FRAME = 2.0
RECORDER_POSITION_PER_FRAME = 6.0
RECORDER_FPS = 10
FRONT_CAMERA_FPS = 10
# The Etron exposes 640x480 YUYV only at 30 Hz. The recorder samples its latest frame at 10 Hz.
SIDE_CAMERA_FPS = 30
PANEL_LABELS = (
    "hand-track",
    "shared-memory PV",
    "front (Creative overhead)",
    "side (Etron)",
)
RECORDER_WINDOW = "PV recorder: hand-track | PV | front | side"
GRIP_CONTEXTS = ("soft", "hard", "unknown")
GRIP_CONTEXT_FEATURES = tuple(f"grip_context.{name}" for name in GRIP_CONTEXTS)
PV_TEACHER_FEATURE = "observation.grip_intent_teacher"
PV_TEACHER_VALID_FEATURE = "observation.grip_intent_valid"
HUMAN_INTERVENTION_FEATURE = "observation.human_intervention"
PV_SOURCE_TIMESTAMP_FEATURE = "observation.grip_intent_source_timestamp_s"
PV_SENT_TIMESTAMP_FEATURE = "observation.grip_intent_sent_timestamp_s"
PV_RECEIVED_TIMESTAMP_FEATURE = "observation.grip_intent_received_timestamp_s"
PV_FRAME_AGE_FEATURE = "observation.grip_intent_frame_age_s"
PV_SEQUENCE_FEATURE = "observation.grip_intent_sequence"
PV_TIMING_FEATURES = (
    PV_SOURCE_TIMESTAMP_FEATURE,
    PV_SENT_TIMESTAMP_FEATURE,
    PV_RECEIVED_TIMESTAMP_FEATURE,
    PV_FRAME_AGE_FEATURE,
    PV_SEQUENCE_FEATURE,
)
PV_AUXILIARY_FEATURES = (
    PV_TEACHER_FEATURE,
    PV_TEACHER_VALID_FEATURE,
    HUMAN_INTERVENTION_FEATURE,
    *PV_TIMING_FEATURES,
)
PV_BAD_STATUSES = frozenset(
    {
        "pv_stale",
        "pv_unavailable",
        "pv_time_skew",
        "pressure_error",
        "pressure_unavailable",
    }
)


def _path_hash(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _load_levels(path: Path, *, max_age_minutes: float) -> dict:
    if not path.is_file():
        raise ValueError(f"levels file does not exist: {path}")
    try:
        levels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid levels file {path}: {exc}") from exc
    if not isinstance(levels, dict):
        raise ValueError("levels must contain a JSON object")
    try:
        n_levels = int(levels["n_levels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("levels must contain integer n_levels") from exc
    if n_levels < 2:
        raise ValueError("levels must contain at least two wire levels")
    fitted_levels = levels.get("levels")
    if fitted_levels is not None and (not isinstance(fitted_levels, list) or len(fitted_levels) < 2):
        raise ValueError("levels.levels must contain at least two fitted levels")
    age_s = max(0.0, time.time() - path.stat().st_mtime)
    if age_s > float(max_age_minutes) * 60.0:
        raise ValueError(
            f"levels are {age_s / 60.0:.1f} minutes old; maximum is {max_age_minutes:.1f} minutes"
        )
    return levels


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def dataset_root_mode(root: Path) -> str:
    """Classify a local dataset without letting LeRobot fall back to the Hub."""
    if not root.exists():
        return "create"
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"dataset root must be a directory: {root}")

    files = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if not files:
        return "reset_empty"

    info_path = Path("meta/info.json")
    tasks_path = Path("meta/tasks.parquet")
    if info_path in files:
        try:
            info = json.loads((root / info_path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid local dataset metadata at {root / info_path}: {exc}") from exc
        totals = (info.get("total_episodes"), info.get("total_frames"), info.get("total_tasks"))
        if totals == (0, 0, 0):
            return "reset_empty"

    if info_path in files and tasks_path in files:
        return "resume"

    missing = [str(path) for path in (info_path, tasks_path) if path not in files]
    detail = f"missing {', '.join(missing)}" if missing else "metadata is incomplete"
    raise ValueError(
        f"incomplete local dataset at {root}: {detail}; refusing to resume it or fall back to the Hub"
    )


def reset_empty_dataset_root(root: Path) -> None:
    if dataset_root_mode(root) != "reset_empty":
        raise ValueError(f"refusing to reset non-empty dataset root: {root}")
    shutil.rmtree(root)


def resolve_grip_context(pv_mapping: str, requested: str = "auto") -> str:
    expected = "soft" if pv_mapping in ("soft_direct", "carton_span") else "hard"
    if requested == "auto":
        return expected
    if requested not in GRIP_CONTEXTS:
        raise ValueError(f"grip context must be one of {GRIP_CONTEXTS} or 'auto'")
    if requested != expected:
        raise ValueError(
            f"--grip-context {requested} conflicts with --pv-mapping {pv_mapping}; expected {expected}"
        )
    return requested


def grip_context_observation(context: str) -> dict[str, float]:
    if context not in GRIP_CONTEXTS:
        raise ValueError(f"unknown grip context {context!r}")
    return {
        feature: float(name == context)
        for name, feature in zip(GRIP_CONTEXTS, GRIP_CONTEXT_FEATURES, strict=True)
    }


def resolve_gripper_closure_limits(args: argparse.Namespace) -> GripperClosureLimits | None:
    values = (args.max_load, args.max_current, args.max_position_lag)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError(
            "max load, max current, and max position lag must be provided together"
        )
    return GripperClosureLimits(*values)


def _scalar_feature(name: str, *, dtype: str = "float32") -> dict:
    return {"dtype": dtype, "shape": (1,), "names": [name]}


def validate_config(args: argparse.Namespace) -> dict:
    """Validate files and semantics without opening a camera, serial bus, UDP socket or mmap."""
    if args.fps != RECORDER_FPS:
        raise ValueError(f"recorder fps is fixed at {RECORDER_FPS} Hz")
    if args.episodes < 0:
        raise ValueError("episodes must not be negative")
    if args.episode_seconds <= 0:
        raise ValueError("episode seconds must be positive")
    if args.pv_port <= 0:
        raise ValueError("pv port must be positive")
    if args.max_level_age_minutes <= 0:
        raise ValueError("max level age minutes must be positive")
    resolve_gripper_closure_limits(args)
    if args.pv_mapping == "hard_profile" and not args.object_profile:
        raise ValueError("hard_profile requires --object-profile")
    if args.pv_mapping in ("soft_direct", "carton_span") and args.object_profile:
        raise ValueError(f"{args.pv_mapping} does not accept --object-profile")
    if not 0.0 <= args.wrist_roll_range_deg <= MAX_WRIST_ROLL_RANGE_DEG:
        raise ValueError(
            f"wrist roll range must be within 0..{MAX_WRIST_ROLL_RANGE_DEG:g} degrees"
        )
    if not 0.0 < args.wrist_roll_gain <= 4.0:
        raise ValueError("wrist roll gain must be within (0, 4]")
    grip_context = resolve_grip_context(args.pv_mapping, args.grip_context)

    levels = _load_levels(Path(args.levels), max_age_minutes=args.max_level_age_minutes)
    profile = None
    profile_hash = None
    if args.object_profile:
        try:
            profile = load_object_profile(args.object_profile)
            profile_hash = object_profile_sha256(args.object_profile)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid object profile: {exc}") from exc
        if profile.arm_id != args.arm_id:
            raise ValueError(
                f"object profile arm_id {profile.arm_id!r} does not match {args.arm_id!r}"
            )

    dataset_root = Path(args.dataset_root)
    dataset_mode = dataset_root_mode(dataset_root)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else DEFAULT_EVIDENCE_ROOT
    if _under(evidence_dir, dataset_root) or _under(dataset_root, evidence_dir):
        raise ValueError("evidence dir and dataset root must be independent directories")
    return {
        "levels": levels,
        "levels_sha256": _path_hash(args.levels),
        "profile": profile,
        "profile_sha256": profile_hash,
        "grip_context": grip_context,
        "dataset_root": dataset_root,
        "dataset_mode": dataset_mode,
        "evidence_dir": evidence_dir,
        "mapping_contract": pressure_range_mapping_contract(
            args.pv_mapping,
            object_profile=profile,
            max_grip_step=RECORDER_POSITION_PER_FRAME,
        ),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--root", dest="dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", "--repo", dest="repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-seconds", type=int, default=DEFAULT_EPISODE_SECONDS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--levels", type=Path, required=True)
    parser.add_argument(
        "--pv-mapping",
        choices=("soft_direct", "carton_span", "hard_profile"),
        default="carton_span",
    )
    parser.add_argument(
        "--grip-context",
        choices=("auto", *GRIP_CONTEXTS),
        default="auto",
        help="Deployable grip context stored in observation.state; auto maps soft_direct->soft and hard_profile->hard.",
    )
    parser.add_argument("--object-profile", "--pv-object-profile", dest="object_profile", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--pv-port", type=int, default=DEFAULT_PV_PORT)
    parser.add_argument("--pv-preview-share", type=Path, default=DEFAULT_PV_PREVIEW_SHARE)
    parser.add_argument("--max-level-age-minutes", type=float, default=DEFAULT_LEVEL_MAX_AGE_MINUTES)
    parser.add_argument("--max-load", type=float)
    parser.add_argument("--max-current", type=float)
    parser.add_argument("--max-position-lag", type=float)
    parser.add_argument("--arm-port", default=ARM_PORT)
    parser.add_argument("--arm-id", default=ARM_ID)
    parser.add_argument("--front-camera", default=WORKSPACE_CAM_PATH)
    parser.add_argument("--side-camera", default=SIDE_CAM_PATH)
    parser.add_argument("--hand-camera", type=int, default=HAND_CAMERA_INDEX)
    parser.add_argument(
        "--wrist-roll-range-deg",
        type=float,
        default=0.0,
        help="Opt-in relative wrist roll; keep zero until the W1 roll-accuracy check is accepted.",
    )
    parser.add_argument("--wrist-roll-gain", type=float, default=1.0)
    parser.add_argument("--no-oak", action="store_true", help="Use the ordinary webcam instead of OAK-D hand tracking.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the three-pane OpenCV window.")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--stream-preflight",
        action="store_true",
        help="Open the formal PV/OAK/front/side streams for 5 seconds without connecting the robot.",
    )
    disposition = parser.add_mutually_exclusive_group()
    disposition.add_argument("--keep-session", action="store_true", dest="keep_session")
    disposition.add_argument("--discard-session", action="store_false", dest="keep_session")
    parser.set_defaults(keep_session=None)
    args = parser.parse_args(argv)
    try:
        validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def build_training_features(robot) -> dict:
    """Return deployable observations plus PV-only auxiliary supervision."""
    action_pipeline = RobotProcessorPipeline(
        steps=[],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    observation_pipeline = RobotProcessorPipeline(
        steps=[],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=action_pipeline,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_pipeline,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    features[PV_TEACHER_FEATURE] = _scalar_feature("grip_intent_teacher")
    features[PV_TEACHER_VALID_FEATURE] = _scalar_feature("grip_intent_valid")
    features[HUMAN_INTERVENTION_FEATURE] = _scalar_feature("human_intervention")
    features[PV_SOURCE_TIMESTAMP_FEATURE] = _scalar_feature(
        "grip_intent_source_timestamp_s", dtype="float64"
    )
    features[PV_SENT_TIMESTAMP_FEATURE] = _scalar_feature(
        "grip_intent_sent_timestamp_s", dtype="float64"
    )
    features[PV_RECEIVED_TIMESTAMP_FEATURE] = _scalar_feature(
        "grip_intent_received_timestamp_s", dtype="float64"
    )
    features[PV_FRAME_AGE_FEATURE] = _scalar_feature("grip_intent_frame_age_s")
    features[PV_SEQUENCE_FEATURE] = _scalar_feature("grip_intent_sequence", dtype="int64")
    image_features = {
        key for key in features if key.startswith("observation.images.")
    }
    expected = {
        "action",
        "observation.state",
        *image_features,
        *PV_AUXILIARY_FEATURES,
    }
    if set(features) != expected:
        raise ValueError(f"PV recorder schema drift: expected {sorted(expected)}, got {sorted(features)}")
    if len(features["action"].get("names", ())) != 6:
        raise ValueError("PV recorder requires exactly six joint-position action values")
    state_names = tuple(features["observation.state"].get("names", ()))
    if not all(name in state_names for name in GRIP_CONTEXT_FEATURES):
        raise ValueError("training observation.state must contain the three grip_context fields")
    return features


def validate_dataset_schema(actual: dict, expected: dict) -> None:
    actual_training_features = set(actual) - set(DEFAULT_FEATURES)
    if actual_training_features != set(expected):
        raise ValueError(
            "dataset schema mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual_training_features)}"
        )
    for key in ("observation.state", "action"):
        if tuple(actual[key].get("names", ())) != tuple(expected[key].get("names", ())):
            raise ValueError(f"dataset {key} names do not match the grip-context schema")


# Backward-compatible descriptive alias for tests and callers that want to inspect the schema.
build_dataset_features = build_training_features


@dataclass
class TemperatureGuard:
    threshold_c: float = 55.0
    consecutive_limit: int = 2
    high_streak: int = 0
    stopped: bool = False
    samples: list[dict] | None = None
    last_observed_at_s: float | None = None
    last_observation_was_new: bool = False

    def __post_init__(self) -> None:
        if self.threshold_c <= 0 or self.consecutive_limit <= 0:
            raise ValueError("temperature guard limits must be positive")
        if self.samples is None:
            self.samples = []

    @property
    def should_stop(self) -> bool:
        return self.stopped

    def observe(self, temperatures, *, observed_at_s: float | None = None) -> bool:
        observed_at_s = time.time() if observed_at_s is None else float(observed_at_s)
        if self.last_observed_at_s is not None and observed_at_s <= self.last_observed_at_s:
            self.last_observation_was_new = False
            return self.stopped
        self.last_observed_at_s = observed_at_s
        self.last_observation_was_new = True
        if isinstance(temperatures, dict):
            values = {}
            for key, value in temperatures.items():
                if value is None:
                    continue
                try:
                    values[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        else:
            try:
                values = {"gripper": float(temperatures)} if temperatures is not None else {}
            except (TypeError, ValueError):
                values = {}
        values = {key: value for key, value in values.items() if np.isfinite(value)}
        high = bool(values) and any(value > self.threshold_c for value in values.values())
        self.high_streak = self.high_streak + 1 if high else 0
        if high:
            self.samples.append(
                {
                    "observed_at_s": observed_at_s,
                    "temperatures_c": values,
                    "high_streak": self.high_streak,
                }
            )
        if self.high_streak >= self.consecutive_limit:
            self.stopped = True
        return self.stopped


def pv_sample_invalid(reading) -> bool:
    """Return whether a PV sample is a save-blocking fault/stale reading."""
    if reading is None:
        return True
    if not bool(getattr(reading, "available", False)):
        return True
    if str(getattr(reading, "status", "")) in PV_BAD_STATUSES:
        return True
    if bool(getattr(reading, "fault_latched", False)):
        return True
    return False


def pv_supervision_from_reading(
    reading,
    *,
    observed_at_s: float | None = None,
    teacher_override: float | None = None,
) -> dict:
    """Build one frame's label and sender-to-dataset timing audit fields."""
    valid = bool(
        reading is not None
        and getattr(reading, "available", False)
        and getattr(reading, "fresh", True)
        and getattr(reading, "status", None) not in PV_BAD_STATUSES
        and (teacher_override is not None or getattr(reading, "active", False))
    )
    value = (
        float(teacher_override)
        if valid and teacher_override is not None
        else float(getattr(reading, "pressure_0_1", 0.0))
        if valid
        else 0.0
    )
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        valid, value = False, 0.0

    # PressureReading's own field names. These were read as `thermal_observed_at_s`,
    # `pv_sent_at_s`, `pv_received_at_s` and `pv_sequence` -- none of which the
    # dataclass has, three of them left over from the IR line. `getattr(..., None)`
    # returned None every frame, so the sequence, all three timestamps and the
    # derived frame age were written to every recorded episode as a constant 0.0:
    # five of the seven PV columns, silently. The sender was numbering packets
    # 298..918 the whole time.
    # `reading` is None until the first PV packet arrives, which the validity
    # check above already accounts for. Named access is deliberate for the rest:
    # it is what makes a wrong field name fail loudly instead of defaulting.
    if reading is None:
        source_t = sent_t = received_t = sequence = None
    else:
        source_t = reading.observed_at_s
        sent_t = reading.sent_at_s
        received_t = reading.received_at_s
        sequence = reading.sequence
    frame_t = time.time() if observed_at_s is None else float(observed_at_s)
    source_t = 0.0 if source_t is None else float(source_t)
    return {
        PV_TEACHER_FEATURE: np.asarray([value], dtype=np.float32),
        PV_TEACHER_VALID_FEATURE: np.asarray([float(valid)], dtype=np.float32),
        PV_SOURCE_TIMESTAMP_FEATURE: np.asarray([source_t], dtype=np.float64),
        PV_SENT_TIMESTAMP_FEATURE: np.asarray(
            [0.0 if sent_t is None else float(sent_t)], dtype=np.float64
        ),
        PV_RECEIVED_TIMESTAMP_FEATURE: np.asarray(
            [0.0 if received_t is None else float(received_t)], dtype=np.float64
        ),
        PV_FRAME_AGE_FEATURE: np.asarray(
            [0.0 if source_t == 0.0 else frame_t - source_t], dtype=np.float32
        ),
        PV_SEQUENCE_FEATURE: np.asarray(
            [0 if sequence is None else int(sequence)], dtype=np.int64
        ),
    }


DATASET_MAPPING_CONTRACT_NAME = "pv_mapping_contract.json"


def write_dataset_mapping_contract(dataset_root: Path, contract) -> Path | None:
    """Record the PV mapping contract inside the dataset's own meta/ directory.

    Returns the path written, or None when the run has no contract (the mappings
    that predate the range mapper). Rewriting it on every run is deliberate: an
    append to an existing dataset under a *different* contract would otherwise be
    invisible, and this makes the mismatch checkable.
    """
    if contract is None:
        return None
    meta_dir = Path(dataset_root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / DATASET_MAPPING_CONTRACT_NAME
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class EvidenceSession:
    """Collision-free audit directory with manifest, control sidecar and episode timeline."""

    def __init__(self, path: Path, *, config: argparse.Namespace, hashes: dict):
        self.path = Path(path)
        if self.path.exists():
            if not self.path.is_dir() or any(self.path.iterdir()):
                raise ValueError(f"refusing to overwrite non-empty evidence directory: {self.path}")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
        self.control_path = self.path / "control_sidecar.csv"
        self.episodes_path = self.path / "episodes.jsonl"
        self.outcomes_path = self.path / "episode_outcomes.jsonl"
        self.initial_states_path = self.path / "episode_initial_state.csv"
        self._control = self.control_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._control,
            fieldnames=("timestamp_s", "event", "episode", "status", "reason", "max_temperature_c", "temperature_high_streak"),
            lineterminator="\n",
        )
        self._writer.writeheader()
        self._control.flush()
        self.outcomes_path.open("x", encoding="utf-8").close()
        self._initial_states = self.initial_states_path.open("x", newline="", encoding="utf-8")
        self._initial_writer = csv.DictWriter(
            self._initial_states,
            fieldnames=(
                "episode",
                "observed_at_s",
                "initial_gripper_pos",
                "front_frame",
                "side_frame",
                "carton_x_px",
                "carton_y_px",
                "carton_yaw_deg",
                "carton_pose_source",
            ),
            lineterminator="\n",
        )
        self._initial_writer.writeheader()
        self._initial_states.flush()
        self.manifest = {
            "schema_version": 1,
            "session_id": self.path.name,
            "created_at_s": time.time(),
            "config": _jsonable(vars(config)),
            "hashes": hashes,
            "panel_columns": list(PANEL_LABELS),
            "dataset_training_fields": [
                "observation.state",
                "observation.images.front",
                "observation.images.side",
                *PV_AUXILIARY_FEATURES,
                "action",
            ],
            "deploy_observation_fields": [
                "observation.state",
                "observation.images.front",
                "observation.images.side",
            ],
            "privileged_training_fields": list(PV_AUXILIARY_FEATURES),
            "pv_sidecar_schema_version": PV_SHADOW_SCHEMA_VERSION,
            "episode_outcomes": self.outcomes_path.name,
            "status": "running",
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(_jsonable(self.manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def control(self, *, event: str, episode: int | None = None, status: str = "", reason: str = "", guard: TemperatureGuard | None = None) -> None:
        self._writer.writerow(
            {
                "timestamp_s": f"{time.time():.6f}",
                "event": event,
                "episode": "" if episode is None else int(episode),
                "status": status,
                "reason": reason,
                "max_temperature_c": "" if guard is None or not guard.samples else max(
                    max(sample["temperatures_c"].values()) for sample in guard.samples
                ),
                "temperature_high_streak": "" if guard is None else guard.high_streak,
            }
        )
        self._control.flush()

    def episode(self, *, number: int, status: str, started_at_s: float, ended_at_s: float, reason: str = "") -> None:
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "episode": int(number),
                        "status": status,
                        "started_at_s": started_at_s,
                        "ended_at_s": ended_at_s,
                        "reason": reason,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def outcome(self, record: dict) -> None:
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    def initial_state(
        self,
        *,
        number: int,
        observed_at_s: float,
        initial_gripper_pos: float,
        front_frame: np.ndarray,
        side_frame: np.ndarray,
    ) -> None:
        frames_dir = self.path / "episode_initial_frames"
        frames_dir.mkdir(exist_ok=True)
        front_path = frames_dir / f"episode-{number:06d}-front.png"
        side_path = frames_dir / f"episode-{number:06d}-side.png"
        if not cv2.imwrite(str(front_path), cv2.cvtColor(front_frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write initial front frame: {front_path}")
        if not cv2.imwrite(str(side_path), cv2.cvtColor(side_frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write initial side frame: {side_path}")
        self._initial_writer.writerow(
            {
                "episode": int(number),
                "observed_at_s": f"{observed_at_s:.6f}",
                "initial_gripper_pos": float(initial_gripper_pos),
                "front_frame": str(front_path.relative_to(self.path)),
                "side_frame": str(side_path.relative_to(self.path)),
                "carton_x_px": "",
                "carton_y_px": "",
                "carton_yaw_deg": "",
                "carton_pose_source": "pending_front_frame_extraction",
            }
        )
        self._initial_states.flush()

    def close(self, *, status: str, analyzer: dict | None = None) -> None:
        if not self._control.closed:
            self._control.close()
        if not self._initial_states.closed:
            self._initial_states.close()
        self.manifest["ended_at_s"] = time.time()
        self.manifest["status"] = status
        if analyzer is not None:
            self.manifest["analyzer"] = analyzer
        self._write_manifest()


def create_evidence_session(root: Path | str = DEFAULT_EVIDENCE_ROOT) -> Path:
    """Create a unique evidence session directory without replacing an existing one."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        path = root / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"could not allocate a unique evidence session below {root}")


def prepare_evidence_session(path: Path | str | None, *, config: argparse.Namespace, hashes: dict) -> EvidenceSession:
    if path is None:
        return EvidenceSession(create_evidence_session(), config=config, hashes=hashes)
    path = Path(path)
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"refusing to overwrite non-empty evidence directory: {path}")
        return EvidenceSession(path, config=config, hashes=hashes)
    path.parent.mkdir(parents=True, exist_ok=True)
    return EvidenceSession(path, config=config, hashes=hashes)


def snapshot_dataset(root: Path) -> Path | None:
    if not root.exists():
        return None
    backup = root.with_name(f"{root.name}.session-{uuid.uuid4().hex}.bak")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite dataset backup: {backup}")
    shutil.copytree(root, backup)
    return backup


def dispose_dataset_session(root: Path | str, backup: Path | str | None, *, keep: bool) -> None:
    """Keep or roll back exactly this session's dataset tree."""
    root, backup = Path(root), None if backup is None else Path(backup)
    if keep:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return
    if root.exists():
        shutil.rmtree(root)
    if backup is not None and backup.exists():
        shutil.move(str(backup), str(root))


def compose_recorder_panel(
    hand_frame: np.ndarray | None,
    pv_frame: np.ndarray | None,
    front_frame: np.ndarray | None,
    side_frame: np.ndarray | None,
    *,
    height: int = 480,
    width: int = 640,
) -> np.ndarray:
    """Return the frozen hand / PV / front / side operator layout."""
    frames = (hand_frame, pv_frame, front_frame, side_frame)
    panels = []
    for frame, label in zip(frames, PANEL_LABELS):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        if frame is not None:
            image = np.asarray(frame)
            if image.ndim == 3 and image.shape[2] == 3:
                scale = min(width / image.shape[1], height / image.shape[0])
                resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                y0 = (height - resized.shape[0]) // 2
                x0 = (width - resized.shape[1]) // 2
                panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.rectangle(panel, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def workspace_camera_frame(robot, name: str):
    camera = robot.cameras.get(name)
    if camera is None:
        return None
    try:
        frame = camera.read_latest(max_age_ms=1000)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def wait_for_continuous_hand_tracking(
    source,
    robot,
    pv_preview,
    *,
    preview: bool,
) -> None:
    """Keep the arm locked until new OAK frames contain a valid right hand for 3 s."""
    gate = ContinuousHandStartupGate()
    last_frame_id = None
    elapsed_s = 0.0
    if preview:
        cv2.namedWindow(RECORDER_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RECORDER_WINDOW, 2560, 540)
    print("ARM LOCKED: keep the right hand continuously visible for 3.0 s to enable startup motion.")
    while elapsed_s < HAND_STARTUP_DWELL_S:
        if getattr(source, "oak_failed", False):
            raise RuntimeError("OAK failed before the hand startup gate passed")
        sample = source.latest_sample()
        if sample.frame_id is not None and sample.frame_id != last_frame_id:
            elapsed_s = gate.update(
                hand_valid=bool(sample.wrist.valid and sample.landmarks.valid),
                observed_at_s=sample.observed_at_s,
            )
            last_frame_id = sample.frame_id
        if preview:
            panel = compose_recorder_panel(
                sample.preview_frame,
                pv_preview.read(),
                workspace_camera_frame(robot, "front"),
                workspace_camera_frame(robot, "side"),
            )
            cv2.putText(
                panel,
                f"ARM LOCKED: right hand {min(elapsed_s, HAND_STARTUP_DWELL_S):.1f}/3.0 s",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(RECORDER_WINDOW, panel)
            if cv2.waitKey(1) & 0xFF == 27:
                raise RuntimeError("startup cancelled before the first robot motion")
        time.sleep(0.01)
    print("ARM ENABLED: continuous right-hand detection reached 3.0 s.")


class ResilientSOFollower(SOFollower):
    def get_observation(self):
        for attempt in range(8):
            try:
                return super().get_observation()
            except ConnectionError:
                if attempt == 7:
                    raise
                time.sleep(0.003)


#: Where the gripper is sent on the way out. Torque-off alone does not open it:
#: the jaw is held by gear friction, so a session that ends mid-grasp -- ESC, a
#: PV fault, an exception -- leaves the object clamped. One deterministic open
#: command before the bus closes is what actually lets go.
SHUTDOWN_RELEASE_POS = 100.0
SHUTDOWN_RELEASE_SETTLE_S = 0.6


def release_gripper_before_disconnect(
    robot,
    *,
    position: float = SHUTDOWN_RELEASE_POS,
    settle_s: float = SHUTDOWN_RELEASE_SETTLE_S,
) -> bool:
    """Command the gripper open and give it time to move. Never raises.

    Best effort by design: this runs on every exit path including the ones that
    are already handling a failure, so it must not replace the original error
    with one of its own. Returns whether the command was actually sent.
    """
    if not getattr(robot, "is_connected", False):
        return False
    try:
        action = {
            key: float(value)
            for key, value in _read_positions(robot).items()
        }
    except Exception as exc:
        print(f"[cleanup] could not read pose before releasing the gripper: {exc}")
        return False
    if "gripper.pos" not in action:
        return False
    action["gripper.pos"] = float(position)
    try:
        robot.send_action(action)
    except Exception as exc:
        print(f"[cleanup] gripper release command failed: {exc}")
        return False
    # Hold the other joints where they are while the jaw travels; disconnecting
    # immediately would cut torque before the gripper has moved.
    time.sleep(settle_s)
    print(f"[cleanup] gripper released to {position:.0f} before disconnect.")
    return True


def disconnect_robot_safely(robot) -> None:
    """Best-effort cleanup for partially connected hardware during an aborted session."""
    try:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    except Exception as exc:
        print(f"[cleanup] robot disconnect failed: {exc}")
    for camera in getattr(robot, "cameras", {}).values():
        try:
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        except Exception as exc:
            print(f"[cleanup] camera disconnect failed: {exc}")
    bus = getattr(robot, "bus", None)
    try:
        if bus is not None and getattr(bus, "is_connected", False):
            bus.disconnect(disable_torque=False)
    except Exception as exc:
        print(f"[cleanup] robot bus disconnect failed: {exc}")


class PVRecorderRobot(ResilientSOFollower):
    """Finalize the audit sidecar around the exact command that record_loop sends."""

    def __init__(self, config, *, grip_context: str):
        super().__init__(config)
        self.grip_context = grip_context
        self._grip_context_observation = grip_context_observation(grip_context)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **super().observation_features,
            **{name: float for name in GRIP_CONTEXT_FEATURES},
        }

    def get_observation(self):
        return {**super().get_observation(), **self._grip_context_observation}

    def attach_recorder(self, teleop) -> None:
        self._pv_recorder_teleop = teleop

    def send_action(self, action):
        command_sent = False
        try:
            result = super().send_action(action)
            command_sent = True
            return result
        finally:
            teleop = getattr(self, "_pv_recorder_teleop", None)
            if teleop is not None:
                teleop.finalize_telemetry(command_sent=command_sent)


class PVRecorderTeleop(Teleoperator):
    config_class = SO101WebcamEEConfig
    name = "webcam_ee_joint_pv"

    def __init__(self, config, controller, pv, source, robot, pv_preview, sidecar, evidence, *, preview=True, motor_sampler=None, use_oak=True):
        super().__init__(config)
        self.config = config
        self.controller = controller
        #: The PV runtime the controller's gripper adapter wraps. The recorder
        #: reads PV state from here, not from the controller: the controller
        #: owns arm motion, this owns pressure.
        self.pv = pv
        self.source = source
        self.robot = robot
        self.pv_preview = pv_preview
        self.sidecar = sidecar
        self.evidence = evidence
        self.preview = bool(preview)
        self.motor_sampler = motor_sampler
        self.use_oak = bool(use_oak)
        self.temperature_guard = TemperatureGuard()
        self.events = None
        self.episode_active = False
        self.episode_valid = True
        self.invalid_reason = ""
        self.episode_started_at_s = None
        self.episode_number = None
        self._pending_sample = None
        self.last_key = 255
        self._connected = False
        self._window = RECORDER_WINDOW

    @property
    def action_features(self) -> dict:
        return {f"{motor}.pos": float for motor in self.controller.motors}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def set_events(self, events: dict) -> None:
        self.events = events

    def connect(self, calibrate: bool = True) -> None:
        self.source.start_oak() if self.use_oak else self.source.start(self.config.camera_index)
        if self.preview:
            cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window, 2560, 540)
        self._connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def begin_episode(self, number: int) -> None:
        self.episode_active = True
        self.episode_valid = True
        self.invalid_reason = ""
        self.episode_started_at_s = time.time()
        self.episode_number = int(number)
        self._pending_sample = None
        initial = self.robot.get_observation()
        self.evidence.initial_state(
            number=number,
            observed_at_s=time.time(),
            initial_gripper_pos=float(initial["gripper.pos"]),
            front_frame=initial["front"],
            side_frame=initial["side"],
        )
        self.evidence.control(event="episode_start", episode=number)

    def invalidate_episode(self, reason: str) -> None:
        if self.episode_valid:
            self.episode_valid = False
            self.invalid_reason = str(reason)
            self.evidence.control(event="episode_invalid", status="discard", reason=self.invalid_reason, guard=self.temperature_guard)
        if self.events is not None:
            self.events["stop_recording"] = True
            self.events["exit_early"] = True

    def end_episode(self, number: int, status: str) -> None:
        started = self.episode_started_at_s or time.time()
        self.evidence.episode(number=number, status=status, started_at_s=started, ended_at_s=time.time(), reason=self.invalid_reason)
        self.episode_active = False
        self.episode_number = None
        self._pending_sample = None

    def _poll_runtime(self) -> None:
        if self.motor_sampler is None:
            return
        telemetry = self.motor_sampler.poll(self.robot)
        if telemetry is None:
            return
        self.pv.set_telemetry(telemetry)
        stopped = self.temperature_guard.observe(
            {"gripper": telemetry.present_temperature}, observed_at_s=telemetry.observed_at_s
        )
        if not self.temperature_guard.last_observation_was_new:
            return
        if stopped:
            self.evidence.control(event="temperature_stop", status="stop", reason="two consecutive samples above 55C", guard=self.temperature_guard)
            self.invalidate_episode("temperature_over_55C_consecutive")
        elif self.temperature_guard.samples and self.temperature_guard.samples[-1]["high_streak"] == 1:
            self.evidence.control(event="temperature_anomaly", status="record", reason="single sample above 55C", guard=self.temperature_guard)

    def _camera_frame(self, name: str):
        return workspace_camera_frame(self.robot, name)

    def _handle_key(self, key: int) -> None:
        self.last_key = key
        if self.events is None:
            return
        if key in (27,):
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
        elif key in (32, 83):  # SPACE or OpenCV right arrow: finish/save
            self.events["exit_early"] = True
        elif key in (ord("r"), ord("R"), 81):  # R or OpenCV left arrow: re-record
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True

    def get_action(self) -> dict:
        if self.use_oak and getattr(self.source, "oak_failed", False):
            if self.episode_active:
                self.invalidate_episode("oak_failed")
            return dict(self.controller.cmd_state)
        previous_locked = self.pv.adjustment_locked
        previous_anchor = self.pv.adjustment_anchor_target
        wrist, landmarks = self.source.latest()
        control_observed_at_s = time.perf_counter()
        joint_action, state = self.controller.step(wrist, landmarks)
        if joint_action is None:
            joint_action = dict(self.controller.cmd_state)
        self._poll_runtime()

        reading = self.pv.last_pressure
        pv_frame = self.pv_preview.read() if self.pv_preview is not None else None
        decision = self.pv.last_pressure_control
        decision_fault = bool(decision is not None and getattr(decision, "fault_latched", False))
        if self.episode_active and state == "MOVING" and (
            pv_sample_invalid(reading) or decision_fault
        ):
            reason = "pressure_fault_latched" if decision_fault else getattr(reading, "status", "preview_stale")
            self.invalidate_episode(f"pv_{reason}")
        self._pending_sample = pv_shadow_sample(
            self.pv,
            control_observed_at_s=control_observed_at_s,
            state=state,
            pinch=self.controller.last_pinch,
        )

        locked = self.pv.adjustment_locked
        anchor = self.pv.adjustment_anchor_target
        if locked != previous_locked or (previous_anchor is not None and anchor is None):
            status = "locked" if locked else "adjusting" if anchor is not None else "reset"
            self.evidence.control(
                event="pv_adjustment",
                episode=self.episode_number,
                status=status,
                reason="" if anchor is None else f"anchor_gripper_pos={anchor:.3f}",
            )
            if locked:
                print(f"[pv] adjustment LOCKED at q={anchor:.2f}; touch PV to resume")
            elif anchor is not None:
                print(f"[pv] adjustment ACTIVE; release PV to lock q<={anchor:.2f}")

        if self.preview:
            hand_frame = self.source.latest_frame()
            front_frame = self._camera_frame("front")
            side_frame = self._camera_frame("side")
            panel = compose_recorder_panel(
                hand_frame,
                pv_frame,
                front_frame,
                side_frame,
            )
            height, width = panel.shape[:2]
            cv2.rectangle(panel, (0, height - 30), (width, height), (0, 0, 0), -1)
            banner_anchor = self.pv.adjustment_anchor_target
            if self.pv.adjustment_locked:
                pv_status = f"PV LOCKED q={banner_anchor:.1f}"
            elif banner_anchor is not None:
                pv_status = f"PV ADJUST q<={banner_anchor:.1f}"
            else:
                pv_status = "PV ADJUST LIVE"
            measured_gripper = (
                None
                if self.motor_sampler is None or self.motor_sampler.latest is None
                else self.motor_sampler.latest.observed_gripper_pos
            )
            draw_gripper_position_banner(
                panel,
                commanded=joint_action.get("gripper.pos"),
                observed=measured_gripper,
            )
            cv2.putText(
                panel,
                f"release/touch PV: lock/adjust ({pv_status})    R/LEFT: restart    SPACE/RIGHT: finish    ESC: stop",
                (12, height - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(self._window, panel)
            self._handle_key(cv2.waitKey(1) & 0xFF)
        return joint_action

    def finalize_telemetry(self, *, command_sent: bool) -> None:
        if self.sidecar is not None:
            self.sidecar.finalize(
                self._pending_sample,
                command_sent=command_sent,
                motor_telemetry=None if self.motor_sampler is None else self.motor_sampler.latest,
            )

    def training_grip_label(self) -> tuple[np.ndarray, np.ndarray]:
        supervision = self.training_grip_supervision()
        return supervision[PV_TEACHER_FEATURE], supervision[PV_TEACHER_VALID_FEATURE]

    def training_grip_supervision(self) -> dict:
        return pv_supervision_from_reading(
            self.pv.last_pressure,
            teacher_override=self.pv.adjustment_teacher,
        )

    def send_feedback(self, feedback: dict) -> None:
        return None

    def close_preview(self) -> None:
        if self.preview:
            self.preview = False
            cv2.destroyAllWindows()

    def disconnect(self) -> None:
        if not self._connected:
            return
        self.close_preview()
        try:
            self.source.stop()
        finally:
            self.controller.close()
            self._connected = False


class PVTeachingDatasetView:
    """Hide privileged labels from record_loop, then attach the current PV target at add_frame."""

    def __init__(self, dataset: LeRobotDataset, teleop: PVRecorderTeleop, *, human_intervention: bool = False):
        self.dataset = dataset
        self.teleop = teleop
        self.fps = dataset.fps
        self.features = {
            key: feature for key, feature in dataset.features.items() if key not in PV_AUXILIARY_FEATURES
        }
        self.human_intervention = human_intervention

    def add_frame(self, frame: dict) -> None:
        supervision = self.teleop.training_grip_supervision()
        self.dataset.add_frame(
            {
                **frame,
                **supervision,
                HUMAN_INTERVENTION_FEATURE: np.asarray(
                    [float(self.human_intervention)], dtype=np.float32
                ),
            }
        )


def episode_review_frames(dataset: LeRobotDataset) -> list[ReviewFrame]:
    """Expose the current unsaved episode through its already-written temp images."""
    writer = dataset.writer
    writer._wait_image_writer()
    buffer = writer.episode_buffer
    size = int(buffer["size"])
    action_names = list(dataset.features["action"]["names"])
    state_names = list(dataset.features["observation.state"]["names"])
    action_gripper = action_names.index("gripper.pos")
    state_gripper = state_names.index("gripper.pos")
    return [
        ReviewFrame(
            timestamp_s=float(buffer["timestamp"][index]),
            front_path=Path(buffer["observation.images.front"][index]),
            side_path=Path(buffer["observation.images.side"][index]),
            commanded_gripper_pos=float(
                np.asarray(buffer["action"][index]).reshape(-1)[action_gripper]
            ),
            observed_gripper_pos=float(
                np.asarray(buffer["observation.state"][index]).reshape(-1)[state_gripper]
            ),
            pv_teacher=float(
                np.asarray(buffer[PV_TEACHER_FEATURE][index]).reshape(-1)[0]
            ),
            pv_valid=bool(
                np.asarray(buffer[PV_TEACHER_VALID_FEATURE][index]).reshape(-1)[0]
            ),
        )
        for index in range(size)
    ]


def system_outcome_record(
    *,
    attempt: int,
    status: str,
    reason: str,
    review_video: Path | None,
    review_timeline: Path | None,
    evidence_root: Path,
) -> dict:
    return {
        "schema_version": 1,
        "attempt": int(attempt),
        "dataset_episode": None,
        "outcome": status,
        "promoted_to_training": False,
        "reason": reason,
        "review_video": (
            None if review_video is None else str(review_video.relative_to(evidence_root))
        ),
        "review_timeline": (
            None if review_timeline is None else str(review_timeline.relative_to(evidence_root))
        ),
        "annotation_method": "system",
    }


def _read_positions(robot, tries: int = 12) -> dict:
    for _ in range(tries):
        try:
            observation = robot.get_observation()
            return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.1)
    raise ConnectionError("arm position read kept failing")


def _ramp_to(robot, target: dict, *, steps: int = 30) -> None:
    start = _read_positions(robot)
    for alpha in np.linspace(0.0, 1.0, steps):
        command = {key: (1.0 - alpha) * start[key] + alpha * target[key] for key in target}
        robot.send_action(command)
        time.sleep(0.04)


def _build_pressure_source(port: int):
    # The UDP socket/thread is intentionally opened only after validate_config has returned.
    udp = PressureVisionUDPSource(port=port)
    latest = LatestFrameSource(udp)
    return PressureVisionSource(source=latest)


def run_analyzer(sidecar_path: Path, evidence_dir: Path) -> dict:
    """Use the existing analyzer to produce the command/readback CSV/PNG and summary."""
    try:
        from analyze_pv_relative_trial import (
            analyze_relative_trial,
            read_csv,
            write_gripper_position_artifacts,
            write_track_hold_artifacts,
        )

        rows = read_csv(sidecar_path) if sidecar_path.is_file() else []
        report = analyze_relative_trial(rows)
        report["artifacts"] = {
            "gripper_position": write_gripper_position_artifacts(rows, evidence_dir),
            "track_hold": write_track_hold_artifacts(rows, evidence_dir),
        }
    except Exception as exc:
        report = {"data_complete": False, "analyzer_error": f"{type(exc).__name__}: {exc}"}
    (evidence_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _choose_keep(args, recorded: int) -> bool:
    if recorded == 0:
        return False
    return args.keep_session is not False


def workspace_camera_configs(args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            index_or_path=args.front_camera,
            width=640,
            height=480,
            fps=RECORDER_FPS,
            fourcc=WORKSPACE_CAM_FOURCC,
            warmup_s=3,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=args.side_camera,
            width=640,
            height=480,
            fps=SIDE_CAMERA_FPS,
            fourcc=SIDE_CAM_FOURCC,
            warmup_s=3,
        ),
    }


def run_stream_preflight(args: argparse.Namespace, *, duration_s: float = 5.0) -> int:
    """Exercise the formal four-stream startup order without opening the robot bus."""
    validate_config(args)
    cfg = SO101WebcamEEConfig(camera_index=args.hand_camera)
    source = WebcamSource(
        WebcamWristEstimator(
            ScaleDepthStrategy(),
            workspace_size_m=cfg.workspace_size_m,
        )
    )
    cameras = {
        name: OpenCVCamera(camera_config)
        for name, camera_config in workspace_camera_configs(args).items()
    }
    preview = PressureVisionPreviewSource(args.pv_preview_share)
    frame_ids = set()
    camera_seen = {name: False for name in cameras}
    pv_seen = False
    with ExitStack() as resources:
        resources.callback(preview.close)
        resources.callback(source.stop)
        source.start_oak() if not args.no_oak else source.start(args.hand_camera)
        for camera in cameras.values():
            camera.connect()
            resources.callback(camera.disconnect)

        deadline = time.monotonic() + float(duration_s)
        while time.monotonic() < deadline:
            if not args.no_oak and getattr(source, "oak_failed", False):
                raise RuntimeError("OAK failed during stream preflight")
            sample = source.latest_sample()
            if sample.frame_id is not None:
                frame_ids.add(int(sample.frame_id))
            for name, camera in cameras.items():
                camera_seen[name] = camera_seen[name] or camera.read_latest(max_age_ms=1000) is not None
            pv_seen = pv_seen or preview.read() is not None
            time.sleep(0.02)

    missing = [name for name, seen in camera_seen.items() if not seen]
    if len(frame_ids) < 2:
        missing.append("oak" if not args.no_oak else "hand")
    if not pv_seen:
        missing.append("pv")
    if missing:
        raise RuntimeError(f"stream preflight missing fresh frames: {', '.join(missing)}")
    print(
        json.dumps(
            {
                "ok": True,
                "duration_s": float(duration_s),
                "hand_frame_count": len(frame_ids),
                "front": camera_seen["front"],
                "side": camera_seen["side"],
                "pv": pv_seen,
                "robot_connected": False,
                "commands_sent": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _open_evidence_session(args: argparse.Namespace, checked: dict) -> EvidenceSession:
    """Create or adopt this run's evidence directory, refusing to reuse a used one."""
    evidence_path = args.evidence_dir
    if evidence_path is None:
        evidence_path = create_evidence_session(DEFAULT_EVIDENCE_ROOT)
    elif Path(evidence_path).exists() and any(Path(evidence_path).iterdir()):
        raise ValueError(f"refusing to overwrite non-empty evidence directory: {evidence_path}")
    return prepare_evidence_session(
        evidence_path,
        config=args,
        hashes={
            "levels": checked["levels_sha256"],
            "calibration": checked["levels_sha256"],
            "object_profile": checked["profile_sha256"],
        },
    )


def run_recording(args: argparse.Namespace) -> int:
    checked = validate_config(args)
    evidence = _open_evidence_session(args, checked)
    dataset_root = Path(args.dataset_root)
    backup = None
    dataset = None
    recorded = 0
    sidecar_path = evidence.path / "pv_shadow.csv"
    try:
        backup = snapshot_dataset(dataset_root)
        dataset_mode = checked["dataset_mode"]
        if dataset_mode == "reset_empty":
            reset_empty_dataset_root(dataset_root)
        open_mode = "create" if dataset_mode == "reset_empty" else dataset_mode
        # --- 1. Devices: hand camera, OAK, arm, PV pressure source. The order
        # inside the ExitStack below is load-bearing; see its comments. ---
        cfg = SO101WebcamEEConfig(camera_index=args.hand_camera)
        source = WebcamSource(
            WebcamWristEstimator(
                ScaleDepthStrategy(),
                workspace_size_m=cfg.workspace_size_m,
            )
        )
        cameras = workspace_camera_configs(args)
        robot = PVRecorderRobot(
            SO101FollowerConfig(
                port=args.arm_port,
                id=args.arm_id,
                use_degrees=True,
                cameras=cameras,
                disable_torque_on_disconnect=False,
            ),
            grip_context=checked["grip_context"],
        )
        with ExitStack() as resources:
            # Registered first, so it runs LAST -- after every other teardown and
            # immediately before the bus is closed by disconnect_robot_safely.
            resources.callback(disconnect_robot_safely, robot)
            resources.callback(release_gripper_before_disconnect, robot)
            if not args.no_oak:
                # DepthAI startup blocks on this host when both workspace UVC
                # streams are already active. Boot OAK before robot.connect()
                # opens the front and side cameras.
                resources.callback(source.stop)
                source.start_oak()
            robot.connect(calibrate=False)
            pv_preview = PressureVisionPreviewSource(args.pv_preview_share)
            resources.callback(pv_preview.close)
            wait_for_continuous_hand_tracking(
                source,
                robot,
                pv_preview,
                preview=not args.no_preview,
            )
            motors = list(robot.bus.motors.keys())
            kin = RobotKinematics(urdf_path=str(urdf_path()), target_frame_name="gripper_frame_link", joint_names=motors)
            # --- 2. Control stack: PV runtime -> gripper adapter -> EE
            # controller -> teleop. Each one wraps the previous. ---
            pressure_source = _build_pressure_source(args.pv_port)
            resources.callback(pressure_source.close)
            sidecar = PVShadowTelemetryLogger(sidecar_path)
            resources.callback(sidecar.close)
            middle_gripper = joint_center(robot.bus.motors["gripper"].norm_mode.value)
            pv = PressureVisionGripRuntime(
                pressure_source,
                initial_gripper=middle_gripper,
                middle_gripper=middle_gripper,
                pressure_apply=True,
                object_profile=checked["profile"],
                object_profile_sha256=checked["profile_sha256"],
                pv_mapping=args.pv_mapping,
                pressure_max_grip_step=RECORDER_POSITION_PER_FRAME,
                gripper_closure_limits=resolve_gripper_closure_limits(args),
            )
            controller = WebcamEEController(
                robot,
                kin,
                cfg,
                use_oak=not args.no_oak,
                gripper=PVGripAdapter(pv),
                # The left hand is on the PressureVision pad, so a left fist is
                # not a gesture the operator can make. This is also what
                # local/evidence/ was recorded through.
                middle_gesture="right_v",
                wrist_roll_range_deg=args.wrist_roll_range_deg,
                wrist_roll_gain=args.wrist_roll_gain,
            )
            resources.callback(controller.close)
            resources.callback(pv.close)
            evidence.manifest["pv_mapping_contract"] = pv.mapping_contract
            evidence.manifest["control_rate_hz"] = RECORDER_FPS
            evidence.manifest["hand_startup_gate_s"] = HAND_STARTUP_DWELL_S
            evidence.manifest["camera_capture_fps"] = {
                "front": FRONT_CAMERA_FPS,
                "side": SIDE_CAMERA_FPS,
            }
            evidence.manifest["wrist_roll"] = {
                "range_deg": args.wrist_roll_range_deg,
                "gain": args.wrist_roll_gain,
            }
            # These six fields are only known once the runtime and the
            # controller exist, so the manifest is rewritten here rather than
            # at prepare_evidence_session time.
            evidence._write_manifest()
            positions = _read_positions(robot)
            ee_centre = kin.forward_kinematics(np.array([positions[f"{motor}.pos"] for motor in motors], dtype=float))[:3, 3]
            controller.build(ee_centre)
            controller.seed(positions)
            motor_sampler = GripperTelemetrySampler(interval_s=0.2)
            teleop = PVRecorderTeleop(
                cfg,
                controller,
                pv,
                source,
                robot,
                pv_preview,
                sidecar,
                evidence,
                preview=not args.no_preview,
                motor_sampler=motor_sampler,
                use_oak=not args.no_oak,
            )
            robot.attach_recorder(teleop)
            resources.callback(teleop.disconnect)
            # --- 3. Dataset: feature schema check -> create/resume -> the
            # mapping contract written beside the episodes. ---
            features = build_training_features(robot)
            image_features = {
                key for key in features if key.startswith("observation.images.")
            }
            required_images = {
                "observation.images.front",
                "observation.images.side",
            }
            if image_features != required_images:
                raise ValueError(
                    "formal PV recorder requires front+side image schema; "
                    f"got {sorted(image_features)}"
                )
            if open_mode == "resume":
                dataset = LeRobotDataset.resume(args.repo_id, root=dataset_root, image_writer_threads=4)
                validate_dataset_schema(dataset.features, features)
            else:
                dataset = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    fps=RECORDER_FPS,
                    features=features,
                    robot_type=robot.name,
                    root=dataset_root,
                    use_videos=True,
                    image_writer_threads=4,
                )
            # The mapping contract travels WITH the dataset, not only in the
            # evidence directory beside it. Schema v7's claim is that a recorded
            # grip can be reproduced from the episode alone; without the release
            # / zero / one positions and the filter cutoff, the teacher column is
            # a number with no scale. The evidence manifest keeps its own copy.
            write_dataset_mapping_contract(dataset_root, pv.mapping_contract)
            teaching_view = PVTeachingDatasetView(dataset, teleop)
            resources.callback(dataset.finalize)
            teleop.connect()
            listener, events = init_keyboard_listener()
            teleop.set_events(events)
            if listener is not None:
                resources.callback(listener.stop)
            identity_action = RobotProcessorPipeline(steps=[], to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action)
            identity_observation = RobotProcessorPipeline(steps=[], to_transition=observation_to_transition, to_output=transition_to_observation)
            attempt = dataset.num_episodes
            while recorded < args.episodes:
                number = attempt
                attempt += 1
                events.update(exit_early=False, rerecord_episode=False)
                if events["stop_recording"]:
                    break
                teleop.begin_episode(number)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=RECORDER_FPS,
                    teleop_action_processor=identity_action,
                    robot_action_processor=identity_action,
                    robot_observation_processor=identity_observation,
                    teleop=teleop,
                    dataset=teaching_view,
                    control_time_s=args.episode_seconds,
                    single_task=args.task,
                    display_data=False,
                )
                if events["rerecord_episode"] and teleop.episode_valid:
                    evidence.outcome(
                        system_outcome_record(
                            attempt=number,
                            status="rerecord",
                            reason="operator_rerecord",
                            review_video=None,
                            review_timeline=None,
                            evidence_root=evidence.path,
                        )
                    )
                    dataset.clear_episode_buffer()
                    teleop.end_episode(number, "rerecord")
                    print(
                        f"[record] attempt {number} discarded; "
                        f"restarting dataset episode {dataset.num_episodes}."
                    )
                    continue
                review_frames = []
                review_video = None
                review_timeline = None
                if dataset.has_pending_frames():
                    review_frames = episode_review_frames(dataset)
                    review_video, review_timeline = write_review_artifacts(
                        review_frames,
                        evidence.path,
                        attempt=number,
                    )
                if not teleop.episode_valid:
                    evidence.outcome(
                        system_outcome_record(
                            attempt=number,
                            status="invalid",
                            reason=teleop.invalid_reason,
                            review_video=review_video,
                            review_timeline=review_timeline,
                            evidence_root=evidence.path,
                        )
                    )
                    dataset.clear_episode_buffer()
                    teleop.end_episode(number, "discarded_pv_fault")
                    break
                if events["stop_recording"]:
                    evidence.outcome(
                        system_outcome_record(
                            attempt=number,
                            status="aborted",
                            reason="operator_abort",
                            review_video=review_video,
                            review_timeline=review_timeline,
                            evidence_root=evidence.path,
                        )
                    )
                    dataset.clear_episode_buffer()
                    teleop.end_episode(number, "aborted")
                    break
                if not review_frames or review_video is None or review_timeline is None:
                    raise RuntimeError("completed episode has no frames to review")
                decision = interactive_review(review_video, review_frames)
                if decision is None:
                    evidence.outcome(
                        system_outcome_record(
                            attempt=number,
                            status="aborted_review",
                            reason="operator_aborted_review",
                            review_video=review_video,
                            review_timeline=review_timeline,
                            evidence_root=evidence.path,
                        )
                    )
                    dataset.clear_episode_buffer()
                    teleop.end_episode(number, "aborted_review")
                    events["stop_recording"] = True
                    break
                promoted = decision["outcome"] != OUTCOME_FAILURE
                session_complete = args.episodes == 1 or (
                    promoted and recorded + 1 >= args.episodes
                )
                if session_complete:
                    # The review decision is final. Close the live teleop UI
                    # before video encoding/finalization so it cannot appear
                    # frozen during dataset shutdown.
                    teleop.close_preview()
                record = outcome_record(
                    attempt=number,
                    dataset_episode=dataset.num_episodes if promoted else None,
                    frames=review_frames,
                    review_video=review_video,
                    review_timeline=review_timeline,
                    evidence_root=evidence.path,
                    **decision,
                )
                evidence.outcome(record)
                if promoted:
                    dataset.save_episode()
                    recorded += 1
                    teleop.end_episode(number, decision["outcome"])
                else:
                    dataset.clear_episode_buffer()
                    teleop.end_episode(number, "discarded_operator_failure")
                if session_complete:
                    teleop.disconnect()
                    break
            # The session status written at close() below is keep/roll-back.
            # Whether the operator stopped early is already recorded per attempt
            # by evidence.outcome(), so it is not summarised a second time here.
            keep = _choose_keep(args, recorded)
        analyzer = run_analyzer(sidecar_path, evidence.path)
        dispose_dataset_session(dataset_root, backup, keep=keep)
        evidence.close(status="kept" if keep else "rolled_back", analyzer=analyzer)
        return 0
    except BaseException:
        try:
            dispose_dataset_session(dataset_root, backup, keep=False)
        finally:
            analyzer = run_analyzer(sidecar_path, evidence.path)
            evidence.close(status="error", analyzer=analyzer)
        raise


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_config:
        checked = validate_config(args)
        print(
            json.dumps(
                {
                    "ok": True,
                    "dataset_root": str(checked["dataset_root"]),
                    "dataset_mode": checked["dataset_mode"],
                    "evidence_dir": str(checked["evidence_dir"]),
                    "pv_mapping": args.pv_mapping,
                    "grip_context": checked["grip_context"],
                    "levels_sha256": checked["levels_sha256"],
                    "object_profile_sha256": checked["profile_sha256"],
                    "pv_mapping_contract": checked["mapping_contract"],
                    "camera_capture_fps": {
                        "front": FRONT_CAMERA_FPS,
                        "side": SIDE_CAMERA_FPS,
                    },
                    "devices_opened": False,
                    "deploy_observation_features": [
                        "observation.state",
                        "observation.images.front",
                        "observation.images.side",
                    ],
                    "privileged_training_features": list(PV_AUXILIARY_FEATURES),
                    "training_features": [
                        "observation.state",
                        "observation.images.front",
                        "observation.images.side",
                        *PV_AUXILIARY_FEATURES,
                        "action",
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.stream_preflight:
        return run_stream_preflight(args)
    return run_recording(args)


if __name__ == "__main__":
    raise SystemExit(main())
