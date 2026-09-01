#!/usr/bin/env python3
"""Measure continuous-PV variation during operator-confirmed steady holds."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from aim_pad_camera import fingerprint_drift  # noqa: E402
from serve_pad_pressure import continuous_pressure, load_levels  # noqa: E402


def low_pass(samples: list[tuple[float, float]], cutoff_hz: float) -> list[float]:
    """Replay the relative mapper's time-aware first-order low-pass."""
    filtered = 0.0
    previous_t = None
    output = []
    for observed_at_s, value in samples:
        if previous_t is not None:
            dt_s = max(0.0, observed_at_s - previous_t)
            alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)
            filtered += alpha * (value - filtered)
        output.append(filtered)
        previous_t = observed_at_s
    return output


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    p05, median, p95 = np.quantile(array, (0.05, 0.5, 0.95))
    return {
        "median": float(median),
        "p05": float(p05),
        "p95": float(p95),
        "p95_minus_p05": float(p95 - p05),
    }


def analyze(
    rows: list[dict],
    metadata: dict,
    levels: dict,
    *,
    cutoff_hz: float,
    discard_s: float,
) -> tuple[list[dict], dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("row_type") == "sample":
            grouped.setdefault(int(row["trial_index"]), []).append(row)

    timeseries = []
    trial_summaries = []
    frame_interval_s = float(metadata.get("frame_interval_s") or 0.0)
    expected_duration_s = float(metadata.get("hold_frames") or 0) * frame_interval_s
    for trial_index, trial_rows in sorted(grouped.items()):
        trial_rows.sort(key=lambda row: float(row["t"]))
        t0 = float(trial_rows[0]["t"])
        samples = [
            (
                float(row["t"]),
                continuous_pressure(float(row[levels["metric"]]), levels),
            )
            for row in trial_rows
        ]
        filtered = low_pass(samples, cutoff_hz)
        kept_raw = []
        kept_filtered = []
        for row, (_, raw), smooth in zip(trial_rows, samples, filtered):
            relative_t = float(row["t"]) - t0
            kept = relative_t >= discard_s
            timeseries.append(
                {
                    "trial_index": trial_index,
                    "block": int(row["block"]),
                    "label": row["target_label"],
                    "time_s": relative_t,
                    "kept": int(kept),
                    "sum_kpa": float(row["sum_kpa"]),
                    "pressure_0_1": raw,
                    "pressure_0_1_3hz": smooth,
                }
            )
            if kept:
                kept_raw.append(raw)
                kept_filtered.append(smooth)
        if not kept_raw:
            raise ValueError(f"trial {trial_index} has no samples after {discard_s:g} s")
        recorded_duration_s = float(trial_rows[-1]["t"] - t0)
        complete_hold = (
            expected_duration_s <= 0.0
            or recorded_duration_s
            >= expected_duration_s - max(0.1, 3.0 * frame_interval_s)
        )
        trial_summaries.append(
            {
                "trial_index": trial_index,
                "block": int(trial_rows[0]["block"]),
                "label": trial_rows[0]["target_label"],
                "recorded_samples": len(trial_rows),
                "recorded_duration_s": recorded_duration_s,
                "complete_hold": complete_hold,
                "analyzed_samples": len(kept_raw),
                "raw": _distribution(kept_raw),
                "filtered_3hz": _distribution(kept_filtered),
                "raw_zero_fraction": float(np.mean(np.asarray(kept_raw) == 0.0)),
                "raw_one_fraction": float(np.mean(np.asarray(kept_raw) == 1.0)),
            }
        )

    by_label = {}
    for label in metadata.get("intent_labels") or sorted(
        {trial["label"] for trial in trial_summaries}
    ):
        all_trials = [trial for trial in trial_summaries if trial["label"] == label]
        trials = [trial for trial in all_trials if trial["complete_hold"]]
        raw_medians = [trial["raw"]["median"] for trial in trials]
        filtered_medians = [trial["filtered_3hz"]["median"] for trial in trials]
        by_label[label] = {
            "all_trial_indices": [trial["trial_index"] for trial in all_trials],
            "incomplete_trial_indices": [
                trial["trial_index"] for trial in all_trials if not trial["complete_hold"]
            ],
            "trial_indices": [trial["trial_index"] for trial in trials],
            "within_hold_raw_spans": [trial["raw"]["p95_minus_p05"] for trial in trials],
            "within_hold_filtered_3hz_spans": [
                trial["filtered_3hz"]["p95_minus_p05"] for trial in trials
            ],
            "raw_trial_medians": raw_medians,
            "filtered_3hz_trial_medians": filtered_medians,
            "between_repeat_raw_median_difference": (
                abs(raw_medians[1] - raw_medians[0]) if len(raw_medians) == 2 else None
            ),
            "between_repeat_filtered_3hz_median_difference": (
                abs(filtered_medians[1] - filtered_medians[0])
                if len(filtered_medians) == 2
                else None
            ),
        }

    drift = fingerprint_drift(metadata.get("scene") or {}, levels.get("scene") or {})
    summary = {
        "cutoff_hz": cutoff_hz,
        "discard_first_s": discard_s,
        "signal": "pressure_0_1 reconstructed with the supplied deployed levels",
        "scene_match": drift is None,
        "scene_mismatch": drift,
        "trials": trial_summaries,
        "by_label": by_label,
    }
    return timeseries, summary


def write_artifacts(timeseries: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "steady_hold_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(timeseries[0]))
        writer.writeheader()
        writer.writerows(timeseries)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["by_label"])
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, 2.8 * len(labels)), sharex=True)
    axes = np.atleast_1d(axes)
    colours = ("tab:blue", "tab:orange")
    for axis, label in zip(axes, labels):
        trial_indices = summary["by_label"][label]["all_trial_indices"]
        for repeat, trial_index in enumerate(trial_indices):
            trial = [row for row in timeseries if row["trial_index"] == trial_index]
            colour = colours[repeat % len(colours)]
            complete = trial_index not in summary["by_label"][label]["incomplete_trial_indices"]
            suffix = "" if complete else " (short)"
            axis.plot(
                [row["time_s"] for row in trial],
                [row["pressure_0_1"] for row in trial],
                color=colour,
                alpha=0.28,
                label=f"repeat {repeat + 1}{suffix} raw",
            )
            axis.plot(
                [row["time_s"] for row in trial],
                [row["pressure_0_1_3hz"] for row in trial],
                color=colour,
                linewidth=1.6,
                label=f"repeat {repeat + 1}{suffix} 3 Hz",
            )
        axis.axvspan(0.0, summary["discard_first_s"], color="grey", alpha=0.12)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel(f"{label}\nPV 0..1")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time since hold recording began (s)")
    figure.suptitle("Subjective steady holds: deployed continuous PV, raw vs 3 Hz")
    figure.tight_layout()
    figure.savefig(out_dir / "steady_hold_raw_vs_3hz.png", dpi=150)
    plt.close(figure)

    (out_dir / "steady_hold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cutoff-hz", type=float, default=3.0)
    parser.add_argument("--discard-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0.0:
        parser.error("--cutoff-hz must be positive and finite")
    if not math.isfinite(args.discard_seconds) or args.discard_seconds < 0.0:
        parser.error("--discard-seconds must be finite and non-negative")

    capture_path = args.session_dir / "capture.jsonl"
    lines = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("row_type") != "metadata":
        parser.error("capture must begin with a metadata row")
    levels = load_levels(args.levels)
    timeseries, summary = analyze(
        lines[1:],
        lines[0],
        levels,
        cutoff_hz=args.cutoff_hz,
        discard_s=args.discard_seconds,
    )
    summary["capture_jsonl_sha256"] = sha256(capture_path.read_bytes()).hexdigest()
    summary["levels_sha256"] = sha256(args.levels.read_bytes()).hexdigest()
    write_artifacts(timeseries, summary, args.out_dir or args.session_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
