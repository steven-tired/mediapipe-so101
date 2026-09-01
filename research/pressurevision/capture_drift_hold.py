"""Hold one press level for minutes and measure how far the estimate drifts.

§3.11 of the handoff establishes that the per-frame error is a slowly-varying
bias, not white noise: a 15-frame rolling median improves separation only 1.5x
where white noise would give sqrt(15). Every measurement behind that finding
covers a hold of about half a second. This script extends the hold to minutes,
which is the timescale a real teleop session runs at, and answers one question:

    does the slow bias wander far enough to cross a level boundary?

If it does, a continuously-updated grip channel needs periodic re-referencing
and the operator has to re-zero mid-session. If it does not, the level set at
calibration stays valid for the whole session and the channel is much simpler.

Ground truth matters here for a second reason. A drifting estimate can mean the
measurement chain drifted OR that the operator's hand drifted, and those have
completely different fixes. Uncropped frames are saved periodically so the scale
display can be read afterwards (as in `evidence/scale_readings_manual.md`) and
the two separated.

Run, per level, with the hand already resting on the pad:

    env -u PYTHONPATH <workspace>/.venv-pressurevision/bin/python \\
        scripts/capture_drift_hold.py \\
        --session-dir <workspace>/scratch_lepton/pv_drift_01 \\
        --crop 138,289,660,705 \\
        --surface "white paper on kitchen scale" --scale-model Tombia
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics as st
import sys
import time

import cv2
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from pressurevision_probe import (  # noqa: E402
    load_model,
    overlay,
)
from capture_labelled_press import (  # noqa: E402
    GRAMS_TO_NEWTONS,
    METRICS,
    measure,
)


DEFAULT_TARGETS_G = (100, 330)
# The reference the drift is measured against, and the opening window discarded
# from it. The first run (pv_drift_01) showed a settling transient over roughly
# the first 10-20 s -- mean_kpa_in_contact rose from 8.0 to 11.9 at the 100 g
# level and then held -- so a reference taken inside that window charges the
# settling to drift. Reference after it, not during it.
DEFAULT_SETTLE_S = 25.0
# Separates slow bias from fast noise. Anything slower than this is "bias".
SLOW_WINDOW_S = 2.0
WINDOW = "drift hold"


def _targets(value: str) -> list[int]:
    grams = [int(v) for v in value.split(",") if v.strip()]
    if len(grams) < 2:
        raise argparse.ArgumentTypeError(
            "give at least two targets in grams; the separation between them is "
            "what the drift is judged against"
        )
    if grams != sorted(grams):
        raise argparse.ArgumentTypeError("targets must ascend")
    return grams


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("<workspace>/pressurevision"),
        help="PressureVision checkout holding config/ and data/model/",
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--crop",
        required=True,
        help="x0,y0,x1,y1 in source pixels; use the same crop as the sessions "
        "being compared against, or the numbers are not comparable",
    )
    parser.add_argument(
        "--targets-g",
        type=_targets,
        default=list(DEFAULT_TARGETS_G),
        help="scale readings to hold, in grams, ascending",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=120.0,
        help="how long to hold each level; the point is minutes, not seconds",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=DEFAULT_SETTLE_S,
        help="opening window used as the drift reference, and discarded from it",
    )
    parser.add_argument(
        "--full-frame-seconds",
        type=float,
        default=5.0,
        help="save an uncropped frame this often so the scale display can be "
        "read afterwards; 0 disables, which makes operator drift and "
        "measurement drift indistinguishable",
    )
    parser.add_argument("--surface", required=True, help="what the pad is")
    parser.add_argument("--scale-model", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _rolling_median(values, width):
    if width <= 1:
        return list(values)
    return [
        st.median(values[max(0, i - width + 1): i + 1])
        for i in range(len(values))
    ]


def drift_report(rows, targets, fps, settle_seconds=DEFAULT_SETTLE_S) -> dict:
    """Split each hold into a slow bias and a fast residual, and ask whether the
    slow part wanders far enough to cross a level boundary.

    The opening `settle_seconds` are excluded from both the reference and the
    excursion: a settling transient there is not drift, and referencing inside
    it makes the settling look like drift for the rest of the hold.
    """
    slow_width = max(1, int(round(SLOW_WINDOW_S * fps)))
    settle_frames = max(1, int(round(settle_seconds * fps)))
    report = {
        "fps": fps,
        "slow_window_frames": slow_width,
        "settle_frames_discarded": settle_frames,
        "metrics": {},
    }

    for metric in METRICS:
        per_level = {}
        for grams in targets:
            series = [r[metric] for r in rows if r["target_g"] == grams]
            if len(series) < settle_frames + slow_width * 2:
                continue
            slow = _rolling_median(series, slow_width)[slow_width - 1:]
            fast = [v - s for v, s in
                    zip(series[slow_width - 1:], slow)]
            # Everything below is measured on the settled part of the hold only.
            settled = slow[settle_frames:]
            reference = st.median(settled[:slow_width]) if settled else 0.0
            excursion = [s - reference for s in settled]
            per_level[grams] = {
                "reference": reference,
                "settled_min": min(settled) if settled else 0.0,
                "settled_max": max(settled) if settled else 0.0,
                "slow_sd": st.pstdev(settled) if settled else 0.0,
                "fast_sd": st.pstdev(fast[settle_frames:]) if settled else 0.0,
                "max_excursion": max(excursion, key=abs) if excursion else 0.0,
                "final_excursion": excursion[-1] if excursion else 0.0,
                "samples": len(series),
            }

        entry = {"per_level": per_level}
        levels = sorted(per_level)
        if len(levels) >= 2:
            separation = (
                per_level[levels[-1]]["reference"] - per_level[levels[0]]["reference"]
            )
            worst = max(
                (abs(per_level[g]["max_excursion"]) for g in levels), default=0.0
            )
            entry["separation"] = separation
            entry["worst_excursion"] = worst
            # A metric pinned to one bin at both levels separates them by
            # nothing. max_kpa did exactly this in pv_drift_01, sitting on the
            # 64 kPa top bin for the whole 330 g hold. Flag it rather than
            # letting a zero separation propagate into a ratio downstream.
            entry["saturated"] = separation == 0
            # A boundary sits halfway between levels, so half the separation is
            # the budget. Anything at or above 1.0 has crossed it.
            entry["excursion_over_budget"] = (
                worst / (abs(separation) / 2) if separation else None
            )
            # The excursion budget is indirect. What actually decides whether a
            # level survives a long hold is whether the two settled ranges stay
            # apart, so report that directly.
            lo_level, hi_level = levels[0], levels[-1]
            if per_level[lo_level]["reference"] > per_level[hi_level]["reference"]:
                lo_level, hi_level = hi_level, lo_level
            gap = (
                per_level[hi_level]["settled_min"] - per_level[lo_level]["settled_max"]
            )
            entry["settled_gap"] = gap
            entry["settled_ranges_overlap"] = gap <= 0
            slow_sds = [per_level[g]["slow_sd"] for g in levels]
            fast_sds = [per_level[g]["fast_sd"] for g in levels]
            mean_fast = st.mean(fast_sds)
            entry["slow_over_fast"] = (
                st.mean(slow_sds) / mean_fast if mean_fast else None
            )
        report["metrics"][metric] = entry

    return report


def run_session(args, *, camera_factory=None, key_source=None) -> dict:
    import pyrealsense2 as rs

    box = tuple(int(v) for v in args.crop.split(","))
    model, config = load_model(args.repo, args.device)
    thresholds = config.FORCE_THRESHOLDS
    contact_thresh = float(config.CONTACT_THRESH)

    session = args.session_dir
    (session / "frames_full").mkdir(parents=True, exist_ok=False)
    capture_path = session / "capture.jsonl"
    stream = capture_path.open("x", encoding="utf-8")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(rs_config)

    rows = []
    aborted = False
    try:
        stream.write(
            json.dumps(
                {
                    "row_type": "metadata",
                    "experiment_identity": "pressurevision_drift_hold_v1",
                    "role": "feasibility_not_preregistered",
                    "surface": args.surface,
                    "scale_model": args.scale_model,
                    "targets_g": args.targets_g,
                    "targets_n": [g * GRAMS_TO_NEWTONS for g in args.targets_g],
                    "hold_seconds": args.hold_seconds,
                    "settle_seconds": args.settle_seconds,
                    "full_frame_seconds": args.full_frame_seconds,
                    "crop": list(box),
                    "force_thresholds_kpa": thresholds,
                    "ground_truth": "operator held the scale reading at target; "
                    "uncropped frames saved periodically for post-hoc reading",
                    "robot_or_controller_output": False,
                },
                sort_keys=True,
            )
            + "\n"
        )

        for grams in args.targets_g:
            # Wait for the operator to settle at this level before timing starts,
            # so the reference window is not contaminated by the approach.
            armed = False
            while not armed:
                frames = pipeline.wait_for_frames()
                colour = frames.get_color_frame()
                if not colour:
                    continue
                bgr = np.asanyarray(colour.get_data())
                resized, kpa, sample = measure(
                    model, thresholds, contact_thresh, bgr, box, args.device
                )
                view = overlay(resized, kpa)
                cv2.putText(
                    view,
                    f"hold {grams} g steady, then SPACE to start "
                    f"({args.hold_seconds:.0f} s)",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    view,
                    f"contact_px {sample['contact_px']}  "
                    f"mean_kpa {sample['mean_kpa_in_contact']:.1f}",
                    (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW, view)
                key = (key_source() if key_source else cv2.waitKey(1)) & 0xFF
                if key == ord(" "):
                    armed = True
                elif key == ord("q"):
                    aborted = True
                    break
            if aborted:
                break

            started = time.time()
            last_full = -1e9
            reference = None
            index = 0
            while True:
                now = time.time()
                elapsed = now - started
                if elapsed >= args.hold_seconds:
                    break

                frames = pipeline.wait_for_frames()
                colour = frames.get_color_frame()
                if not colour:
                    continue
                bgr = np.asanyarray(colour.get_data())
                resized, kpa, sample = measure(
                    model, thresholds, contact_thresh, bgr, box, args.device
                )

                frame_full = None
                if args.full_frame_seconds > 0 and (
                    now - last_full >= args.full_frame_seconds
                ):
                    frame_full = f"frames_full/hold{grams:04d}g_{index:05d}.png"
                    cv2.imwrite(str(session / frame_full), bgr)
                    last_full = now

                row = {
                    "row_type": "sample",
                    "target_g": grams,
                    "target_n": grams * GRAMS_TO_NEWTONS,
                    "hold_index": index,
                    "t": now,
                    "elapsed_s": elapsed,
                    "frame_full": frame_full,
                    **sample,
                }
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                index += 1

                if elapsed >= args.settle_seconds and reference is None:
                    settled = [
                        r["mean_kpa_in_contact"]
                        for r in rows
                        if r["target_g"] == grams
                    ]
                    reference = st.median(settled)

                view = overlay(resized, kpa)
                remaining = args.hold_seconds - elapsed
                cv2.putText(
                    view,
                    f"{grams} g   {remaining:5.1f} s left",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if reference:
                    delta = 100.0 * (sample["mean_kpa_in_contact"] - reference) / reference
                    cv2.putText(
                        view,
                        f"mean_kpa {sample['mean_kpa_in_contact']:6.2f}  "
                        f"vs ref {reference:6.2f}  ({delta:+5.1f}%)",
                        (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow(WINDOW, view)
                key = (key_source() if key_source else cv2.waitKey(1)) & 0xFF
                if key == ord("q"):
                    aborted = True
                    break
            if aborted:
                break
    finally:
        pipeline.stop()
        stream.close()
        cv2.destroyAllWindows()

    ts = sorted(r["t"] for r in rows)
    dts = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 1.0]
    fps = (1.0 / st.median(dts)) if dts else 0.0

    held = {
        g: sum(1 for r in rows if r["target_g"] == g) for g in args.targets_g
    }
    complete = (
        not aborted
        and all(held.get(g, 0) > 0 for g in args.targets_g)
        and fps > 0
    )

    manifest = {
        "experiment_identity": "pressurevision_drift_hold_v1",
        "role": "feasibility_not_preregistered",
        "surface": args.surface,
        "scale_model": args.scale_model,
        "targets_g": args.targets_g,
        "hold_seconds": args.hold_seconds,
        "crop": list(box),
        "sample_count": len(rows),
        "samples_per_level": held,
        "capture_jsonl_sha256": sha256(capture_path.read_bytes()).hexdigest(),
        "controller_or_robot_actuation": False,
        "status": "complete" if complete else "aborted",
        "drift": (
            drift_report(rows, args.targets_g, fps, args.settle_seconds)
            if fps
            else {}
        ),
    }
    (session / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = run_session(args)
    print(json.dumps(manifest["drift"], indent=2, sort_keys=True))
    print(
        f"status {manifest['status']}, {manifest['sample_count']} samples "
        f"-> {args.session_dir}"
    )
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
