#!/usr/bin/env python3
"""Measure the pressing finger's posture from saved session frames.

Sessions 03 and 04 disagreed on light-versus-hard thresholds, and comparing
frames suggested the cause was posture: pad-flat in one, tip-on in the other.
That is a hypothesis about the data, so it has to be measured before anything
is built on it.

The feature is the **foreshortening** of the distal phalanx: the apparent
length of TIP-DIP divided by the apparent length of PIP-MCP, both in image
pixels. Rotating the finger out of the image plane shortens the first and
leaves the second largely alone, so the ratio falls as the finger goes from
pad-flat to tip-on. Being a ratio, it is invariant to hand size and to how
large the hand appears in the crop, which matters because those also differed.

MediaPipe's landmark `z` is reported alongside as a secondary check; it is a
more direct pitch signal but is not reliable in absolute terms.

Runs on saved PNGs. Reads only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# MediaPipe hand landmark indices for the index finger.
MCP, PIP, DIP, TIP = 5, 6, 7, 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        action="append",
        required=True,
        help="session directory holding frames/; repeat to compare sessions",
    )
    parser.add_argument(
        "--frames-subdir",
        default="frames",
        help=(
            "which saved frames to read: 'frames' is the 480x384 model crop, "
            "'frames_full' the uncropped sensor frame. Comparing the two "
            "separates a crop that cuts off the palm from the fingers "
            "occluding each other as they close."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="frames are 15 per held press and highly redundant",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser.parse_args(argv)


def _hands():
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


def posture_features(landmarks, width: int, height: int) -> dict:
    """Foreshortening of the distal phalanx, plus a z-based cross-check."""
    def pixel(index):
        point = landmarks[index]
        return np.array([point.x * width, point.y * height], dtype=float)

    distal = float(np.linalg.norm(pixel(TIP) - pixel(DIP)))
    proximal = float(np.linalg.norm(pixel(PIP) - pixel(MCP)))
    if proximal <= 1e-6:
        raise ValueError("degenerate proximal phalanx")
    # z is in roughly the same units as normalised x, negative towards camera.
    z_tip, z_dip = float(landmarks[TIP].z), float(landmarks[DIP].z)
    return {
        "distal_px": distal,
        "proximal_px": proximal,
        "foreshortening": distal / proximal,
        "z_tip_minus_dip": z_tip - z_dip,
    }


def parse_frame_name(name: str):
    """`trial007_0330g_04.png` -> (7, 330, 4)."""
    stem = Path(name).stem
    trial, grams, hold = stem.split("_")
    return int(trial[5:]), int(grams[:-1]), int(hold)


def measure_session(session: Path, stride: int, subdir: str = "frames") -> list[dict]:
    hands = _hands()
    rows = []
    try:
        for index, path in enumerate(sorted((session / subdir).glob("*.png"))):
            if index % stride:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            height, width = image.shape[:2]
            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            trial, grams, hold = parse_frame_name(path.name)
            row = {
                "session": session.name,
                "frame": path.name,
                "trial_index": trial,
                "target_g": grams,
                "hold_index": hold,
                "detected": bool(result.multi_hand_landmarks),
            }
            if result.multi_hand_landmarks:
                try:
                    row.update(
                        posture_features(
                            result.multi_hand_landmarks[0].landmark, width, height
                        )
                    )
                except ValueError:
                    row["detected"] = False
            rows.append(row)
    finally:
        hands.close()
    return rows


def summarise(rows) -> dict:
    sessions = sorted({row["session"] for row in rows})
    summary = {}
    for session in sessions:
        detected = [
            row for row in rows if row["session"] == session and row["detected"]
        ]
        total = len([row for row in rows if row["session"] == session])
        entry = {
            "frames": total,
            "detected": len(detected),
            "detection_rate": len(detected) / total if total else None,
        }
        for key in ("foreshortening", "z_tip_minus_dip"):
            values = np.array([row[key] for row in detected], dtype=float)
            entry[key] = (
                {
                    "median": float(np.median(values)),
                    "p05": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                }
                if values.size
                else None
            )
        # per level, since posture could plausibly vary with how hard you press
        entry["foreshortening_by_level"] = {}
        for grams in sorted({row["target_g"] for row in detected}):
            values = np.array(
                [row["foreshortening"] for row in detected if row["target_g"] == grams],
                dtype=float,
            )
            entry["foreshortening_by_level"][str(grams)] = (
                float(np.median(values)) if values.size else None
            )
        summary[session] = entry
    if len(sessions) == 2:
        a, b = (
            np.array(
                [row["foreshortening"] for row in rows
                 if row["session"] == session and row["detected"]],
                dtype=float,
            )
            for session in sessions
        )
        if a.size and b.size:
            wins = float((b[:, None] > a[None, :]).sum())
            ties = float((b[:, None] == a[None, :]).sum())
            summary["between_sessions"] = {
                "pair": sessions,
                "foreshortening_auc": (wins + 0.5 * ties) / (a.size * b.size),
                "note": (
                    "0.5 means posture is indistinguishable between sessions; "
                    "far from 0.5 means it differed"
                ),
            }
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = []
    for session in args.session:
        session_rows = measure_session(session, args.stride, args.frames_subdir)
        print(f"{session.name}: {len(session_rows)} frames sampled")
        rows.extend(session_rows)

    summary = summarise(rows)
    for session, entry in summary.items():
        if session == "between_sessions":
            continue
        fore = entry["foreshortening"]
        print(f"\n--- {session} ---")
        print(
            f"  detection {entry['detected']}/{entry['frames']} "
            f"({entry['detection_rate']:.1%})"
        )
        if fore:
            print(
                f"  foreshortening  median {fore['median']:.3f}   "
                f"p05-p95 [{fore['p05']:.3f}, {fore['p95']:.3f}]"
            )
            print(f"  by level {entry['foreshortening_by_level']}")
        z = entry["z_tip_minus_dip"]
        if z:
            print(f"  z(tip)-z(dip)   median {z['median']:+.4f}")
    if "between_sessions" in summary:
        between = summary["between_sessions"]
        print(
            f"\nbetween {between['pair'][0]} and {between['pair'][1]}: "
            f"foreshortening AUC {between['foreshortening_auc']:.3f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
