#!/usr/bin/env python3
"""Measure second-scale PressureVision drift in fixed-position intent holds."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


def _trial_summary(rows: list[dict], discard_s: float, bin_s: float) -> tuple[list[dict], dict]:
    rows = sorted(rows, key=lambda row: float(row["t"]))
    t0 = float(rows[0]["t"])
    bins: dict[int, list[float]] = {}
    kept = []
    for row in rows:
        elapsed_s = float(row["t"]) - t0
        if elapsed_s < discard_s:
            continue
        value = float(row["sum_kpa"])
        kept.append(value)
        index = int((elapsed_s - discard_s) // bin_s)
        bins.setdefault(index, []).append(value)
    if not kept or len(bins) < 2:
        raise ValueError(f"trial {rows[0]['trial_index']} has too little post-discard data")

    bin_rows = [
        {
            "trial_index": int(rows[0]["trial_index"]),
            "block": int(rows[0]["block"]),
            "label": rows[0]["target_label"],
            "bin_index": index,
            "time_center_s": discard_s + (index + 0.5) * bin_s,
            "samples": len(values),
            "median_sum_kpa": float(np.median(values)),
        }
        for index, values in sorted(bins.items())
    ]
    medians = [row["median_sum_kpa"] for row in bin_rows]
    trial_median = float(np.median(kept))
    return bin_rows, {
        "trial_index": int(rows[0]["trial_index"]),
        "block": int(rows[0]["block"]),
        "label": rows[0]["target_label"],
        "analyzed_samples": len(kept),
        "one_second_bins": len(bin_rows),
        "trial_median_sum_kpa": trial_median,
        "slow_drift_range_sum_kpa": float(max(medians) - min(medians)),
        "slow_drift_fraction_of_trial_median": (
            None if trial_median == 0.0 else float((max(medians) - min(medians)) / trial_median)
        ),
        "last_minus_first_bin_sum_kpa": float(medians[-1] - medians[0]),
        "bin_medians_sum_kpa": medians,
    }


def analyze(rows: list[dict], metadata: dict, discard_s: float, bin_s: float) -> tuple[list[dict], dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if (
            row.get("row_type") == "sample"
            and row.get("phase") == "steady"
            and row.get("trial_status") == "complete"
        ):
            grouped.setdefault(int(row["trial_index"]), []).append(row)
    if not grouped:
        raise ValueError("capture has no completed steady samples")

    bin_rows = []
    trials = []
    for trial_rows in grouped.values():
        trial_bins, trial = _trial_summary(trial_rows, discard_s, bin_s)
        bin_rows.extend(trial_bins)
        trials.append(trial)

    labels = metadata.get("labels") or sorted({trial["label"] for trial in trials})
    by_label = {}
    for label in labels:
        selected = [trial for trial in trials if trial["label"] == label]
        repeat_medians = [trial["trial_median_sum_kpa"] for trial in selected]
        window_medians = [
            value for trial in selected for value in trial["bin_medians_sum_kpa"]
        ]
        by_label[label] = {
            "trial_indices": [trial["trial_index"] for trial in selected],
            "trial_medians_sum_kpa": repeat_medians,
            "between_repeat_median_range_sum_kpa": float(max(repeat_medians) - min(repeat_medians)),
            "slow_drift_ranges_sum_kpa": [
                trial["slow_drift_range_sum_kpa"] for trial in selected
            ],
            "slow_drift_fractions": [
                trial["slow_drift_fraction_of_trial_median"] for trial in selected
            ],
            "all_window_medians_min_max_sum_kpa": [
                float(min(window_medians)),
                float(max(window_medians)),
            ],
        }

    return bin_rows, {
        "metric": "sum_kpa",
        "discard_first_s": discard_s,
        "bin_seconds": bin_s,
        "mapping_or_clipping_applied": False,
        "operator_pressure_feedback": metadata.get("operator_pressure_feedback"),
        "trials": sorted(trials, key=lambda trial: trial["trial_index"]),
        "by_label": by_label,
    }


def write_artifacts(bin_rows: list[dict], summary: dict, out_dir: Path) -> None:
    csv_path = out_dir / "slow_drift_1s_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["by_label"])
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, 2.8 * len(labels)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, label in zip(axes, labels):
        trial_indices = summary["by_label"][label]["trial_indices"]
        for repeat, trial_index in enumerate(trial_indices, start=1):
            selected = [row for row in bin_rows if row["trial_index"] == trial_index]
            axis.plot(
                [row["time_center_s"] for row in selected],
                [row["median_sum_kpa"] for row in selected],
                marker="o",
                label=f"repeat {repeat}",
            )
        axis.set_ylabel(f"{label}\nsum_kpa")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time since steady hold began (s); first second excluded")
    figure.suptitle("PressureVision slow drift: one-second medians, no mapping or clipping")
    figure.tight_layout()
    figure.savefig(out_dir / "slow_drift_1s_bins.png", dpi=150)
    plt.close(figure)

    (out_dir / "slow_drift_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--discard-seconds", type=float, default=None)
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.discard_seconds is not None and (
        not math.isfinite(args.discard_seconds) or args.discard_seconds < 0.0
    ):
        parser.error("--discard-seconds must be finite and non-negative")
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0.0:
        parser.error("--bin-seconds must be positive and finite")

    capture_path = args.session_dir / "capture.jsonl"
    manifest_path = args.session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        parser.error("session manifest is not complete")
    digest = sha256(capture_path.read_bytes()).hexdigest()
    if digest != manifest.get("capture_jsonl_sha256"):
        parser.error("capture checksum does not match manifest")
    lines = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("row_type") != "metadata":
        parser.error("capture must begin with metadata")
    discard_s = (
        float(lines[0].get("visible_steady_seconds") or 0.0)
        if args.discard_seconds is None
        else args.discard_seconds
    )
    bin_rows, summary = analyze(lines[1:], lines[0], discard_s, args.bin_seconds)
    summary["capture_jsonl_sha256"] = digest
    write_artifacts(bin_rows, summary, args.session_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
