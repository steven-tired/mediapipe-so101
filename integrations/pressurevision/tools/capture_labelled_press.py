#!/usr/bin/env python3
"""Record labelled press trials against a scale, and test light/hard separation.

The operator presses a pad resting on an ordinary kitchen scale, holding the
displayed weight at a target, and confirms each trial. The scale is never read
by software -- it does not need to be -- because the architecture this serves
makes force a discrete event rather than a continuous signal, so one verified
label per trial is what the question needs.

Targets default to 100 g and 330 g. The obvious choice was 330 g and 830 g,
matching PressureVision's own 3.24 N and 8.16 N prompted-force conditions, but
a fingertip concentrates those into roughly 32 and 81 kPa against a top bin of
64 kPa, so both saturate: session 02 separated them at only AUC 0.84 and the
peak estimate was pinned to the top bin on every single frame. At 100 g and
330 g the levels separate perfectly on contact area.

Where the scale is not part of the final rig, --intent-labels drops it and has
the operator press at their own idea of each level instead. That costs the force
ground truth, and the session says so. It is the honest trade when the scale
would sit under the pad only during calibration: its thickness raises the
pressing surface, and viewpoint is the variable that decides whether the model
responds at all, so calibrating on a geometry the teleop never runs in produces
anchors that do not transfer either.

Robot-free and read-only with respect to the robot: this records and analyses,
and drives nothing.
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
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from aim_pad_camera import (  # noqa: E402
    SETTLE_FRAMES,
    SETTLE_S,
    WARMUP_FRAMES,
    lock_camera,
    open_camera,
    pixel_format,
    scene_fingerprint,
)
from pressurevision_probe import (  # noqa: E402
    NETWORK_SIZE,
    load_model,
    overlay,
    preprocess,
    to_kpa,
)


GRAMS_TO_NEWTONS = 9.80665 / 1000.0
# 330 g and 830 g put a fingertip at 32 and 81 kPa against a 64 kPa top bin,
# so both saturate and only area carries information. 100 g and 330 g sit in
# the model's usable range: session 03 separated them perfectly on contact_px.
DEFAULT_TARGETS_G = (0, 100, 330)
SATURATION_WARN_FRACTION = 0.5
WINDOW = "labelled press capture"


def _targets(value: str) -> list[int]:
    grams = [int(part) for part in value.split(",") if part.strip()]
    if len(grams) < 2:
        raise argparse.ArgumentTypeError("give at least two targets in grams")
    if sorted(grams) != grams:
        raise argparse.ArgumentTypeError("targets must ascend")
    return grams


def _intent_labels(value: str) -> list[str]:
    labels = [part.strip() for part in value.split(",") if part.strip()]
    if len(labels) < 2:
        raise argparse.ArgumentTypeError("give at least two press levels, e.g. light,hard")
    if labels[0].lower() == "none" and len(labels) < 3:
        raise argparse.ArgumentTypeError(
            "a no-press baseline needs at least two press levels, e.g. none,light,hard"
        )
    return labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--crop",
        required=True,
        help="x0,y0,x1,y1 of the pad region, from the live aiming run",
    )
    parser.add_argument(
        "--targets-g",
        type=_targets,
        default=list(DEFAULT_TARGETS_G),
        help=(
            "scale readings to hold, in grams (default 0,100,330). Higher "
            "targets saturate the model's top pressure bin on a fingertip."
        ),
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help=(
            "how long to hold each press. Converted to frames from the rate the "
            "camera actually delivers, so a camera swap does not silently change "
            "the protocol: 8 frames is a sensible 1.1 s on the 7.5 fps C270 and a "
            "useless 0.27 s transient on a 30 fps one. Session 06 held 2 s and the "
            "operator's holds still decayed from 15/15 to 6/15 over 24 trials, "
            "dragging the light-hard d' from 3.4 to 1.6."
        ),
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=None,
        help="Override --hold-seconds with a fixed frame count.",
    )
    parser.add_argument(
        "--full-frame-stride",
        type=int,
        default=3,
        help=(
            "also save every Nth uncropped 1280x720 frame, 0 to disable. "
            "The 480x384 model crop cuts off the palm, so MediaPipe found a "
            "hand in only 30 percent of session 03's saved frames and posture "
            "could not be measured after the fact."
        ),
    )
    parser.add_argument(
        "--intent-labels",
        type=_intent_labels,
        default=None,
        metavar="light,hard",
        help="Label presses by intent instead of by scale reading, for rigs where the "
        "scale is not part of the final geometry. With light,hard, contact is gated "
        "externally and wire level zero stays reserved for the baseline. Prefix with "
        "none only when deliberately measuring the model's contact decision. The "
        "session records that it carries no force ground truth.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=2,
        help="UVC camera index for the overhead rig (default: the C270 on /dev/video2)",
    )
    parser.add_argument(
        "--mjpg",
        action="store_true",
        help="Ask the camera for MJPG (30 fps at 1280x720 against 7.5 uncompressed). "
        "JPEG is lossy and colour carries the blanching cue, so the sender must run "
        "under the same format this was captured with.",
    )
    parser.add_argument(
        "--realsense",
        action="store_true",
        help="Use a D435i colour stream instead. PressureVision is RGB-only, so this "
        "buys nothing but a different lens and height than the sender later streams.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip the exposure/white-balance lock. Colour carries the blanching cue, "
        "so an unlocked run drifts away from the anchors it produces.",
    )
    parser.add_argument("--surface", required=True, help="what the pad is")
    parser.add_argument("--scale-model", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.session_dir.exists():
        parser.error("--session-dir must not already exist")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.hold_seconds <= 0:
        parser.error("--hold-seconds must be positive")
    if args.hold_frames is not None and args.hold_frames < 1:
        parser.error("--hold-frames must be positive")
    args.contact_gated = False
    if args.intent_labels:
        args.contact_gated = args.intent_labels[0].lower() != "none"
        # Level zero remains the consumer's baseline. A contact-gated calibration
        # therefore starts at one; a session that explicitly records `none` starts
        # at zero and can still measure the contact decision as before.
        first_level = 1 if args.contact_gated else 0
        args.targets_g = list(range(first_level, first_level + len(args.intent_labels)))
        if args.scale_model:
            parser.error("--scale-model contradicts --intent-labels: no scale is read")
    return args


def trial_order(targets, repeats, rng):
    """Interleave the levels so slow drift cannot masquerade as an effect."""
    trials = []
    for block in range(repeats):
        block_targets = list(targets)
        rng.shuffle(block_targets)
        trials.extend((block, grams) for grams in block_targets)
    return trials


def measure(model, thresholds, contact_thresh, bgr, box, device):
    resized, tensor = preprocess(bgr, box)
    with torch.no_grad():
        logits = model(tensor.to(device))
    kpa = to_kpa(logits, thresholds)
    contact = kpa >= contact_thresh
    return resized, kpa, {
        "sum_kpa": float(kpa.sum()),
        "max_kpa": float(kpa.max()),
        "contact_px": int(contact.sum()),
        "mean_kpa_in_contact": float(kpa[contact].mean()) if contact.any() else 0.0,
    }


METRICS = ("contact_px", "sum_kpa", "mean_kpa_in_contact", "max_kpa")


def _per_trial(rows, targets, metric):
    """Collapse each held press to one number.

    The frames inside a trial are the same press half a second apart, so
    treating them as independent samples inflates any separability claim. The
    first analysis of session 02 made exactly that mistake.
    """
    trials: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        trials.setdefault(
            (row["trial_index"], row["target_g"]), []
        ).append(float(row[metric]))
    per_target = {grams: [] for grams in targets}
    for (_index, grams), values in sorted(trials.items()):
        per_target[grams].append(float(np.median(values)))
    return {grams: np.asarray(values) for grams, values in per_target.items()}


def separation_report(rows, targets, thresholds=None) -> dict:
    """Per-level distributions, and the cheapest honest separability check.

    Reports the rank-sum overlap between adjacent levels rather than a t-test,
    since a handful of trials per level says nothing about normality.
    """
    by_target = _per_trial(rows, targets, "sum_kpa")
    levels = {}
    for grams, values in by_target.items():
        levels[str(grams)] = {
            "n": int(values.size),
            "median_sum_kpa": float(np.median(values)) if values.size else None,
            "min_sum_kpa": float(values.min()) if values.size else None,
            "max_sum_kpa": float(values.max()) if values.size else None,
            "newtons": grams * GRAMS_TO_NEWTONS,
        }
    pairs = {}
    ordered = sorted(targets)
    for low, high in zip(ordered, ordered[1:]):
        a, b = by_target[low], by_target[high]
        if a.size == 0 or b.size == 0:
            pairs[f"{low}_vs_{high}"] = None
            continue
        # Probability that a random high-force trial exceeds a random low one.
        # 1.0 means the two sets do not overlap at all.
        wins = float((b[:, None] > a[None, :]).sum())
        ties = float((b[:, None] == a[None, :]).sum())
        pairs[f"{low}_vs_{high}"] = {
            "auc": (wins + 0.5 * ties) / (a.size * b.size),
            "separated": bool(a.max() < b.min()),
        }
    metrics = {}
    for metric in METRICS:
        values = _per_trial(rows, targets, metric)
        entry = {}
        for low, high in zip(ordered, ordered[1:]):
            a, b = values[low], values[high]
            if a.size == 0 or b.size == 0:
                entry[f"{low}_vs_{high}"] = None
                continue
            wins = float((b[:, None] > a[None, :]).sum())
            ties = float((b[:, None] == a[None, :]).sum())
            entry[f"{low}_vs_{high}"] = {
                "auc": (wins + 0.5 * ties) / (a.size * b.size),
                "separated": bool(a.max() < b.min()),
                "medians": [float(np.median(a)), float(np.median(b))],
            }
        metrics[metric] = entry

    # A metric pinned to the top bin carries no information about force, which
    # is what happened to max_kpa at 330 g and above.
    top_bin = float(max(thresholds)) if thresholds else None
    top_bin_fraction = {}
    for grams in targets:
        peaks = np.array(
            [row["max_kpa"] for row in rows if row["target_g"] == grams],
            dtype=float,
        )
        top_bin_fraction[str(grams)] = (
            float((peaks >= top_bin).mean())
            if peaks.size and top_bin is not None
            else None
        )
    saturated = [
        grams
        for grams in targets
        if grams > 0
        and top_bin_fraction[str(grams)] is not None
        and top_bin_fraction[str(grams)] >= SATURATION_WARN_FRACTION
    ]

    return {
        "unit_of_analysis": "one median per held trial, not per frame",
        "trials_per_level": {
            str(grams): int(by_target[grams].size) for grams in targets
        },
        "levels": levels,
        "adjacent_pairs": pairs,
        "by_metric": metrics,
        "top_bin_kpa": top_bin,
        "peak_at_top_bin_fraction": top_bin_fraction,
        "saturation_warning": (
            None
            if not saturated
            else f"peak pinned to the top bin at {saturated} g: intensity "
            "carries no force information there, prefer contact_px"
        ),
    }


class _UVCFrames:
    """Locked-exposure UVC camera, reusing aim_pad_camera's open + lock recipe.

    The overhead rig is a C270 on /dev/video2 and the aiming step already runs
    against it, so calibrating through the D435i would anchor kPa values taken
    through a different lens at a different height than the ones the sender
    later streams.
    """

    def __init__(self, index: int, *, lock: bool = True, mjpg: bool = False):
        self.index = index
        self.capture = open_camera(index, mjpg=mjpg)
        self.pixel_format = pixel_format(self.capture)
        if lock:
            for _ in range(WARMUP_FRAMES):     # exposure only sticks once frames flow
                self.capture.read()
            lock_camera(index)
            time.sleep(SETTLE_S)
            for _ in range(SETTLE_FRAMES):
                self.capture.read()

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        self.capture.release()


class _RealSenseFrames:
    def __init__(self):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def read(self):
        colour = self.pipeline.wait_for_frames().get_color_frame()
        return np.asanyarray(colour.get_data()) if colour else None

    def close(self) -> None:
        self.pipeline.stop()


def open_frames(args, camera_factory=None):
    if camera_factory is not None:
        return camera_factory()
    if args.realsense:
        return _RealSenseFrames()
    return _UVCFrames(args.camera, lock=not args.no_lock, mjpg=args.mjpg)


def run_session(args, *, camera_factory=None, key_source=None, rng=None) -> dict:
    rng = rng or np.random.default_rng()
    box = tuple(int(v) for v in args.crop.split(","))

    # Camera first: a busy device is the common failure, and opening it after the
    # model means waiting out a GPU load only to be told the camera was never free.
    # Nothing is created on disk yet either, so a failure here leaves no half-made
    # session directory to delete before retrying under the same name.
    frames_source = open_frames(args, camera_factory)
    try:
        # Timed rather than read off CAP_PROP_FPS: the property reports what the
        # driver negotiated, and the delivered rate is what decides how long a
        # hold of N frames actually lasts.
        for _ in range(5):
            frames_source.read()
        started = time.perf_counter()
        for _ in range(20):
            frames_source.read()
        frame_interval_s = (time.perf_counter() - started) / 20
        hold_frames = args.hold_frames or max(1, round(args.hold_seconds / frame_interval_s))
        print(
            f"[capture] camera delivers {1 / frame_interval_s:.1f} fps -> "
            f"{hold_frames} frames per {hold_frames * frame_interval_s:.2f} s hold"
        )
        model, config = load_model(args.repo, args.device)
    except BaseException:
        frames_source.close()
        raise
    thresholds = config.FORCE_THRESHOLDS
    contact_thresh = float(config.CONTACT_THRESH)

    session = args.session_dir
    (session / "frames").mkdir(parents=True, exist_ok=False)
    if args.full_frame_stride > 0:
        (session / "frames_full").mkdir(parents=True, exist_ok=False)
    capture_path = session / "capture.jsonl"
    stream = capture_path.open("x", encoding="utf-8")

    # A crop is only meaningful against the camera it was aimed with, so the
    # session has to say which one produced it.
    camera_label = "realsense_d435i" if args.realsense else f"uvc:/dev/video{args.camera}"
    scene = scene_fingerprint(frames_source.read())
    intent = args.intent_labels
    intent_by_target = dict(zip(args.targets_g, intent or []))

    trials = trial_order(args.targets_g, args.repeats, rng)
    rows = []
    aborted = False
    try:
        stream.write(
            json.dumps(
                {
                    "row_type": "metadata",
                    "experiment_identity": "pressurevision_labelled_press_v1",
                    "role": "feasibility_not_preregistered",
                    "surface": args.surface,
                    "scale_model": args.scale_model,
                    "targets_g": args.targets_g,
                    "targets_n": None if intent else [g * GRAMS_TO_NEWTONS for g in args.targets_g],
                    "intent_labels": intent,
                    "contact_gated": args.contact_gated,
                    "full_frame_stride": args.full_frame_stride,
                    "repeats": args.repeats,
                    "hold_frames": hold_frames,
                    "frame_interval_s": frame_interval_s,
                    "crop": list(box),
                    "camera": camera_label,
                    "exposure_locked": not args.no_lock,
                    "pixel_format": getattr(frames_source, "pixel_format", None),
                    "scene": scene,
                    "force_thresholds_kpa": thresholds,
                    "ground_truth": (
                        "operator pressed at their own idea of each level; no scale, no force truth"
                        if intent
                        else "operator held the scale reading at target"
                    ),
                    "robot_or_controller_output": False,
                },
                sort_keys=True,
            )
            + "\n"
        )
        # Flushed here, not left buffered: a session killed mid-run should still say
        # what it was trying to record, and every reader expects metadata on line 1.
        stream.flush()
        for index, (block, grams) in enumerate(trials):
            confirmed = False
            while not confirmed:
                bgr = frames_source.read()
                if bgr is None:
                    continue
                resized, kpa, stats = measure(
                    model, thresholds, contact_thresh, bgr, box, args.device
                )
                painted = overlay(resized, kpa)
                for line_index, line in enumerate(
                    (
                        f"trial {index + 1}/{len(trials)}  block {block + 1}",
                        (f"PRESS: {intent_by_target[grams].upper()}" if intent
                         else f"HOLD THE SCALE AT {grams} g")
                        + ("  (no contact)" if grams == 0 else ""),
                        f"contact {stats['contact_px']} px   "
                        f"max {stats['max_kpa']:.1f} kPa   "
                        f"sum {stats['sum_kpa']:.0f}",
                        "SPACE: record   q: abort",
                    )
                ):
                    cv2.putText(
                        painted,
                        line,
                        (12, 26 + line_index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW, np.hstack([resized, painted]))
                key = (key_source or (lambda: cv2.waitKey(1) & 0xFF))()
                if key in (ord("q"), ord("Q")):
                    aborted = True
                    break
                if key == ord(" "):
                    confirmed = True
            if aborted:
                break

            # A trial that started with contact and then lost it is the operator
            # lifting early, not a reading. Recording those frames anyway lets the
            # per-trial median walk through them: session 06 lost two trials to
            # exactly that and it cost the light-hard separation the d' gate.
            held_contact = False
            lifted_at = None
            for hold in range(hold_frames):
                bgr = frames_source.read()
                if bgr is None:
                    continue
                resized, kpa, stats = measure(
                    model, thresholds, contact_thresh, bgr, box, args.device
                )
                if stats["contact_px"] > 0:
                    held_contact = True
                elif held_contact:
                    lifted_at = hold
                    print(
                        f"[hold] trial {index} ({intent_by_target[grams] if intent else f'{grams} g'}): "
                        f"contact lost after {hold}/{hold_frames} frames -- "
                        "hold until the window stops counting"
                    )
                    break
                name = f"trial{index:03d}_{grams:04d}g_{hold:02d}.png"
                cv2.imwrite(str(session / "frames" / name), resized)
                full_name = None
                if args.full_frame_stride > 0 and hold % args.full_frame_stride == 0:
                    full_name = f"frames_full/{name}"
                    cv2.imwrite(str(session / full_name), bgr)
                row = {
                    "row_type": "sample",
                    "trial_index": index,
                    "block": block,
                    "target_g": grams,
                    "target_n": None if intent else grams * GRAMS_TO_NEWTONS,
                    "target_label": intent_by_target[grams] if intent else None,
                    "hold_index": hold,
                    "lifted_early": lifted_at is not None,
                    "frame": f"frames/{name}",
                    "frame_full": full_name,
                    "t": time.time(),
                    **stats,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                rows.append(row)
    finally:
        frames_source.close()
        stream.close()
        cv2.destroyAllWindows()

    report = separation_report(rows, args.targets_g, thresholds) if rows else {}
    manifest = {
        "experiment_identity": "pressurevision_labelled_press_v1",
        "role": "feasibility_not_preregistered",
        "status": "aborted" if aborted else "complete",
        "surface": args.surface,
        "scale_model": args.scale_model,
        "targets_g": args.targets_g,
        "repeats": args.repeats,
        "crop": list(box),
        "camera": camera_label,
        "exposure_locked": not args.no_lock,
        "pixel_format": getattr(frames_source, "pixel_format", None),
        "full_frame_stride": args.full_frame_stride,
        "sample_count": len(rows),
        "capture_jsonl_sha256": (
            sha256(capture_path.read_bytes()).hexdigest()
            if capture_path.is_file()
            else None
        ),
        "separation": report,
        "force_ground_truth": (
            "operator_intent_uncalibrated" if intent else "operator_held_scale_reading"
        ),
        "intent_labels": intent,
        "contact_gated": args.contact_gated,
        "controller_or_robot_actuation": False,
        "verdict": "not_a_formal_result",
    }
    with (session / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = run_session(args)
    print(json.dumps(manifest["separation"], indent=2, sort_keys=True))
    print(f"status {manifest['status']}, {manifest['sample_count']} samples -> {args.session_dir}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
