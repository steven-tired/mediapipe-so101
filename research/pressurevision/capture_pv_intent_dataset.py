#!/usr/bin/env python3
"""Capture fixed-setup PressureVision data for light/medium/hard fine-tuning.

This is a robot-free dataset recorder.  The labels are the current operator's
three ordered grip intentions, not force measurements.  Each trial is split
into explicit phases so transition frames never inherit the final press label:

    released baseline -> ramp in -> steady hold -> release

Only the steady hold is written as ``row_type=sample`` and receives an ordinal
target.  Baseline, ramp, release, rejected, and aborted frames are retained as
``diagnostic_sample`` rows for abstention analysis.  They are not a fourth
class and must not be used by the ordinal loss.

The recorder reuses the camera, crop, exposure-lock, and released
PressureVision inference path from ``capture_labelled_press.py``.  It saves the
exact 480x384 BGR crop used by the network as lossless PNG plus periodic full
frames for setup auditing.  No controller or robot module is imported.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_labelled_press import measure, open_frames  # noqa: E402
from aim_pad_camera import scene_fingerprint  # noqa: E402
from pressurevision_probe import NETWORK_SIZE, load_model, overlay  # noqa: E402


LABELS = ("light", "medium", "hard")
WINDOW = "PressureVision intent dataset"
SCHEMA_VERSION = 1
DEFAULT_TARGET_ZONE = (NETWORK_SIZE[0] // 2, NETWORK_SIZE[1] // 2, 40)


def _crop(value: str) -> tuple[int, int, int, int]:
    try:
        box = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1") from exc
    if len(box) != 4:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1")
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("crop must have non-negative origin and positive area")
    return box


def _target_zone(value: str) -> tuple[int, int, int]:
    try:
        zone = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target zone must be cx,cy,r in 480x384 pixels") from exc
    if len(zone) != 3:
        raise argparse.ArgumentTypeError("target zone must be cx,cy,r in 480x384 pixels")
    cx, cy, radius = zone
    width, height = NETWORK_SIZE
    if radius <= 0 or cx - radius < 0 or cy - radius < 0:
        raise argparse.ArgumentTypeError("target zone must have positive radius and stay in frame")
    if cx + radius >= width or cy + radius >= height:
        raise argparse.ArgumentTypeError("target zone must stay inside the 480x384 model input")
    return zone


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("<workspace>/pressurevision"),
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--setup-id", required=True, help="stable name for this fixed rig")
    parser.add_argument(
        "--operator-id",
        required=True,
        help="stable pseudonym; the labels are operator-specific intentions",
    )
    parser.add_argument("--crop", type=_crop, required=True, help="x0,y0,x1,y1 from aiming")
    parser.add_argument("--surface", required=True, help="visible pad/surface description")
    parser.add_argument(
        "--target-zone",
        type=_target_zone,
        default=DEFAULT_TARGET_ZONE,
        metavar="CX,CY,R",
        help=(
            "fixed circular press guide in 480x384 model-input pixels "
            f"(default: {DEFAULT_TARGET_ZONE[0]},{DEFAULT_TARGET_ZONE[1]},{DEFAULT_TARGET_ZONE[2]})"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=15,
        help="complete randomized blocks; each block contains light, medium and hard once",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=0.5,
        help="released-pad diagnostic recorded before each prompted press",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="steady, operator-confirmed interval that receives the ordinal label",
    )
    parser.add_argument(
        "--release-seconds",
        type=float,
        default=0.75,
        help="unlabelled release transition retained for abstention evaluation",
    )
    parser.add_argument(
        "--full-frame-stride",
        type=int,
        default=5,
        help="also save every Nth full frame across all phases; 0 disables it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--mjpg", action="store_true")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--blind-pressure-feedback",
        action="store_true",
        help=(
            "show only RGB, target zone and protocol instructions; hide the "
            "PressureVision heatmap and all pressure/contact statistics"
        ),
    )
    parser.add_argument(
        "--visible-steady-seconds",
        type=float,
        default=0.0,
        help=(
            "With --blind-pressure-feedback, briefly show the PV heatmap and metrics "
            "at the start of each steady hold, then hide them again. These initial "
            "frames can be excluded from drift analysis."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved protocol and randomized schedule; open no camera/model and write nothing",
    )
    args = parser.parse_args(argv)

    if args.session_dir.exists():
        parser.error("--session-dir must not already exist")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for name in ("baseline_seconds", "hold_seconds", "release_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.visible_steady_seconds < 0:
        parser.error("--visible-steady-seconds must be non-negative")
    if args.visible_steady_seconds > args.hold_seconds:
        parser.error("--visible-steady-seconds cannot exceed --hold-seconds")
    if args.full_frame_stride < 0:
        parser.error("--full-frame-stride must be non-negative")
    return args


def trial_order(
    labels: tuple[str, ...],
    repeats: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Block-randomize the three grades so drift cannot become a class."""
    trials = []
    for block in range(repeats):
        block_labels = list(labels)
        rng.shuffle(block_labels)
        trials.extend((block, label) for label in block_labels)
    return trials


def frames_for_seconds(seconds: float, frame_interval_s: float) -> int:
    if seconds <= 0 or frame_interval_s <= 0:
        raise ValueError("seconds and frame_interval_s must be positive")
    return max(1, round(seconds / frame_interval_s))


def finalize_trial_rows(
    rows: list[dict],
    *,
    prompted_label: str,
    accepted: bool,
    trial_status: str,
) -> list[dict]:
    """Assign ordinal targets only to accepted steady frames."""
    ordinal = LABELS.index(prompted_label)
    finalized = []
    for raw in rows:
        row = dict(raw)
        trainable = accepted and row["phase"] == "steady"
        row.update(
            {
                "row_type": "sample" if trainable else "diagnostic_sample",
                "valid_for_ordinal_training": trainable,
                "target_label": prompted_label if trainable else None,
                "ordinal_target": ordinal if trainable else None,
                "trial_status": trial_status,
            }
        )
        finalized.append(row)
    return finalized


def protocol_summary(args) -> dict:
    schedule = trial_order(LABELS, args.repeats, np.random.default_rng(args.seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_identity": "pressurevision_fixed_setup_intent_v1",
        "setup_id": args.setup_id,
        "operator_id": args.operator_id,
        "labels": list(LABELS),
        "label_semantics": "ordered operator grip intention; not measured force",
        "target_zone_model_px": {
            "cx": args.target_zone[0],
            "cy": args.target_zone[1],
            "radius": args.target_zone[2],
        },
        "repeats": args.repeats,
        "trial_count": len(schedule),
        "phases": ["baseline", "ramp_in", "steady", "release"],
        "ordinal_training_phase": "steady only",
        "nontraining_policy": "baseline/ramp/release/rejected/aborted are diagnostic, not a class",
        "operator_pressure_feedback": (
            "initial_steady_visible_then_hidden"
            if args.blind_pressure_feedback and args.visible_steady_seconds > 0.0
            else "hidden"
            if args.blind_pressure_feedback
            else "visible"
        ),
        "visible_steady_seconds": args.visible_steady_seconds,
        "schedule": [
            {"block": block, "label": label} for block, label in schedule
        ],
    }


def _checkpoint_sha256(repo: Path) -> str:
    path = repo / "data/model/paper_59.pt"
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    if x1 > width or y1 > height:
        raise ValueError(f"crop {box} exceeds camera frame {width}x{height}")


def _draw(
    resized: np.ndarray,
    kpa: np.ndarray,
    lines: tuple[str, ...],
    target_zone: tuple[int, int, int],
    *,
    show_pressure_feedback: bool = True,
) -> None:
    input_panel = resized.copy()
    painted = overlay(resized, kpa) if show_pressure_feedback else resized.copy()
    cx, cy, radius = target_zone
    panels = (input_panel, painted) if show_pressure_feedback else (painted,)
    for panel in panels:
        cv2.circle(panel, (cx, cy), radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            panel,
            (cx, cy),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )
    for index, line in enumerate(lines):
        cv2.putText(
            painted,
            line,
            (12, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    display = np.hstack([input_panel, painted]) if show_pressure_feedback else painted
    cv2.imshow(WINDOW, display)


def _feedback_line(stats: dict, *, hidden: bool) -> str:
    if hidden:
        return "pressure feedback hidden for blinded capture"
    return (
        f"contact {stats['contact_px']} px  max {stats['max_kpa']:.1f} kPa  "
        f"sum {stats['sum_kpa']:.0f}"
    )


def run_session(args, *, camera_factory=None, key_source=None, rng=None) -> dict:
    rng = rng or np.random.default_rng(args.seed)
    schedule = trial_order(LABELS, args.repeats, rng)
    read_key = key_source or (lambda: cv2.waitKey(1) & 0xFF)

    frames_source = open_frames(args, camera_factory)
    try:
        for _ in range(5):
            frames_source.read()
        valid = 0
        started = time.perf_counter()
        while valid < 20:
            if frames_source.read() is not None:
                valid += 1
        frame_interval_s = (time.perf_counter() - started) / valid
        model, config = load_model(args.repo, args.device)
        first_frame = frames_source.read()
        if first_frame is None:
            raise RuntimeError("camera returned no frame after model load")
        _validate_crop(first_frame, args.crop)
        scene = scene_fingerprint(first_frame)
    except BaseException:
        frames_source.close()
        raise

    phase_counts = {
        "baseline": frames_for_seconds(args.baseline_seconds, frame_interval_s),
        "steady": frames_for_seconds(args.hold_seconds, frame_interval_s),
        "release": frames_for_seconds(args.release_seconds, frame_interval_s),
    }
    visible_steady_frames = (
        frames_for_seconds(args.visible_steady_seconds, frame_interval_s)
        if args.visible_steady_seconds > 0.0
        else 0
    )
    thresholds = config.FORCE_THRESHOLDS
    contact_thresh = float(config.CONTACT_THRESH)
    camera_label = "realsense_d435i" if args.realsense else f"uvc:/dev/video{args.camera}"

    session = args.session_dir
    (session / "frames").mkdir(parents=True, exist_ok=False)
    if args.full_frame_stride:
        (session / "frames_full").mkdir(parents=True, exist_ok=False)
    capture_path = session / "capture.jsonl"
    stream = capture_path.open("x", encoding="utf-8")

    metadata = {
        **protocol_summary(args),
        "row_type": "metadata",
        "role": "fine_tune_dataset_not_deployment_authorization",
        "surface": args.surface,
        "notes": args.notes,
        "crop": list(args.crop),
        "network_input_size": list(NETWORK_SIZE),
        "preprocess": "RGB ImageNet mean/std; frames stored as 480x384 BGR PNG",
        "camera": camera_label,
        "pixel_format": getattr(frames_source, "pixel_format", None),
        "exposure_locked": not args.no_lock,
        "scene": scene,
        "frame_interval_s": frame_interval_s,
        "phase_frame_counts": phase_counts,
        "visible_steady_frames": visible_steady_frames,
        "full_frame_stride": args.full_frame_stride,
        "pressurevision_checkpoint_sha256": _checkpoint_sha256(args.repo),
        "no_contact_semantics": "diagnostic invalid candidate; never a gripper-open command",
        "grasp_intent_missing_level_policy": "hold_previous_gripper_command",
        "split_policy": "whole-session only; never random frame split",
        "robot_or_controller_output": False,
        "operator_pressure_feedback": (
            "initial_steady_visible_then_hidden"
            if args.blind_pressure_feedback and args.visible_steady_seconds > 0.0
            else "hidden"
            if args.blind_pressure_feedback
            else "visible"
        ),
        "visible_steady_seconds": args.visible_steady_seconds,
    }
    stream.write(json.dumps(metadata, sort_keys=True) + "\n")
    stream.flush()

    frame_serial = 0
    written_rows = []
    completed_trials = 0
    rejected_trials = 0
    aborted = False

    def observe(
        *,
        trial_index: int,
        block: int,
        label: str,
        phase: str,
        phase_index: int,
        instruction: str,
        show_pressure_feedback: bool | None = None,
    ) -> tuple[dict | None, int]:
        nonlocal frame_serial
        bgr = frames_source.read()
        if bgr is None:
            return None, -1
        resized, kpa, stats = measure(
            model, thresholds, contact_thresh, bgr, args.crop, args.device
        )
        name = f"trial{trial_index:04d}_{label}_{phase}_{phase_index:04d}.png"
        frame_path = session / "frames" / name
        if not cv2.imwrite(str(frame_path), resized):
            raise RuntimeError(f"failed to write {frame_path}")
        full_name = None
        if args.full_frame_stride and frame_serial % args.full_frame_stride == 0:
            full_name = f"frames_full/{name}"
            if not cv2.imwrite(str(session / full_name), bgr):
                raise RuntimeError(f"failed to write {session / full_name}")
        frame_serial += 1
        visible = (
            not args.blind_pressure_feedback
            if show_pressure_feedback is None
            else show_pressure_feedback
        )
        _draw(
            resized,
            kpa,
            (
                f"trial {trial_index + 1}/{len(schedule)}  block {block + 1}",
                f"target {label.upper()}  phase {phase}",
                f"keep fingertip inside yellow zone {args.target_zone}",
                instruction,
                _feedback_line(stats, hidden=not visible),
                "q: abort session   r: reject current trial",
            ),
            args.target_zone,
            show_pressure_feedback=visible,
        )
        row = {
            "trial_index": trial_index,
            "block": block,
            "target_zone_model_px": list(args.target_zone),
            "prompted_label": label,
            "phase": phase,
            "phase_index": phase_index,
            "frame": f"frames/{name}",
            "frame_full": full_name,
            "t": time.time(),
            **stats,
        }
        return row, read_key()

    try:
        for trial_index, (block, label) in enumerate(schedule):
            # Operator explicitly establishes the released state before anything is saved.
            while True:
                bgr = frames_source.read()
                if bgr is None:
                    continue
                resized, kpa, stats = measure(
                    model, thresholds, contact_thresh, bgr, args.crop, args.device
                )
                _draw(
                    resized,
                    kpa,
                    (
                        f"trial {trial_index + 1}/{len(schedule)}  block {block + 1}",
                        f"RELEASE PAD; next target {label.upper()}",
                        f"next press stays inside yellow zone {args.target_zone}",
                        "SPACE: released and ready",
                        _feedback_line(stats, hidden=args.blind_pressure_feedback),
                        "q: abort session",
                    ),
                    args.target_zone,
                    show_pressure_feedback=not args.blind_pressure_feedback,
                )
                key = read_key()
                if key in (ord("q"), ord("Q")):
                    aborted = True
                    break
                if key == ord(" "):
                    break
            if aborted:
                break

            trial_rows = []
            trial_aborted = False
            rejected = False

            for phase_index in range(phase_counts["baseline"]):
                row, key = observe(
                    trial_index=trial_index,
                    block=block,
                    label=label,
                    phase="baseline",
                    phase_index=phase_index,
                    instruction="KEEP RELEASED",
                )
                if row is not None:
                    trial_rows.append(row)
                if key in (ord("q"), ord("Q")):
                    trial_aborted = aborted = True
                    break
                if key in (ord("r"), ord("R")):
                    rejected = True
                    break

            # The operator, not a timer, marks when the target level is stable.
            ramp_index = 0
            while not trial_aborted and not rejected:
                row, key = observe(
                    trial_index=trial_index,
                    block=block,
                    label=label,
                    phase="ramp_in",
                    phase_index=ramp_index,
                    instruction=f"PRESS {label.upper()}; SPACE only when stable",
                )
                if row is not None:
                    trial_rows.append(row)
                    ramp_index += 1
                if key in (ord("q"), ord("Q")):
                    trial_aborted = aborted = True
                    break
                if key in (ord("r"), ord("R")):
                    rejected = True
                    break
                if key == ord(" "):
                    break

            for phase_index in range(phase_counts["steady"]):
                if trial_aborted or rejected:
                    break
                row, key = observe(
                    trial_index=trial_index,
                    block=block,
                    label=label,
                    phase="steady",
                    phase_index=phase_index,
                    instruction=f"HOLD {label.upper()} STEADY",
                    show_pressure_feedback=(
                        not args.blind_pressure_feedback
                        or phase_index < visible_steady_frames
                    ),
                )
                if row is not None:
                    trial_rows.append(row)
                if key in (ord("q"), ord("Q")):
                    trial_aborted = aborted = True
                    break
                if key in (ord("r"), ord("R")):
                    rejected = True
                    break

            if not trial_aborted:
                for phase_index in range(phase_counts["release"]):
                    row, key = observe(
                        trial_index=trial_index,
                        block=block,
                        label=label,
                        phase="release",
                        phase_index=phase_index,
                        instruction="RELEASE NOW; these frames are diagnostic only",
                    )
                    if row is not None:
                        trial_rows.append(row)
                    if key in (ord("q"), ord("Q")):
                        trial_aborted = aborted = True
                        break
                    if key in (ord("r"), ord("R")):
                        rejected = True

            accepted = not trial_aborted and not rejected
            status = "complete" if accepted else "aborted" if trial_aborted else "rejected"
            finalized = finalize_trial_rows(
                trial_rows,
                prompted_label=label,
                accepted=accepted,
                trial_status=status,
            )
            for row in finalized:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            written_rows.extend(finalized)

            if accepted:
                completed_trials += 1
            elif rejected:
                rejected_trials += 1
            if aborted:
                break
    finally:
        frames_source.close()
        stream.close()
        cv2.destroyAllWindows()

    trainable = [row for row in written_rows if row["valid_for_ordinal_training"]]
    diagnostic = [row for row in written_rows if not row["valid_for_ordinal_training"]]
    per_label = {
        label: sum(row["target_label"] == label for row in trainable) for label in LABELS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_identity": "pressurevision_fixed_setup_intent_v1",
        "status": "aborted" if aborted else "complete",
        "setup_id": args.setup_id,
        "operator_id": args.operator_id,
        "labels": list(LABELS),
        "target_zone_model_px": list(args.target_zone),
        "operator_pressure_feedback": (
            "initial_steady_visible_then_hidden"
            if args.blind_pressure_feedback and args.visible_steady_seconds > 0.0
            else "hidden"
            if args.blind_pressure_feedback
            else "visible"
        ),
        "visible_steady_seconds": args.visible_steady_seconds,
        "scheduled_trials": len(schedule),
        "completed_trials": completed_trials,
        "rejected_trials": rejected_trials,
        "trainable_frame_count": len(trainable),
        "diagnostic_frame_count": len(diagnostic),
        "trainable_frames_per_label": per_label,
        "capture_jsonl_sha256": sha256(capture_path.read_bytes()).hexdigest(),
        "controller_or_robot_actuation": False,
        "verdict": "dataset_capture_only_not_model_or_control_validation",
    }
    with (session / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(protocol_summary(args), indent=2, sort_keys=True))
        return 0
    manifest = run_session(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
