#!/usr/bin/env python3
"""Replay a time-aware first-order low-pass filter on relative PV telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def replay(rows: list[dict[str, str]], cutoff_hz: float) -> tuple[list[dict], dict]:
    samples = []
    filtered_target = None
    previous_t = None
    first_t = None
    previous_reference = None
    for row in rows:
        observed_at_s = _float(row, "control_observed_at_s")
        reference = _float(row, "relative_reference_pos")
        closure = _float(row, "relative_closure")
        if observed_at_s is None or reference is None or closure is None:
            continue
        if first_t is None:
            first_t = observed_at_s
            previous_t = observed_at_s
        raw_target = reference - closure
        dt_s = max(0.0, observed_at_s - previous_t)
        if filtered_target is None or reference != previous_reference:
            filtered_target = reference
        elif samples:
            alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)
            filtered_target += alpha * (raw_target - filtered_target)
        samples.append(
            {
                "time_s": observed_at_s - first_t,
                "dt_s": dt_s,
                "reference_pos": reference,
                "raw_closure": closure,
                "filtered_closure": reference - filtered_target,
                "raw_target_pos": raw_target,
                "filtered_target_pos": filtered_target,
            }
        )
        previous_t = observed_at_s
        previous_reference = reference

    raw_steps = [
        abs(current["raw_target_pos"] - previous["raw_target_pos"])
        for previous, current in zip(samples, samples[1:])
    ]
    filtered_steps = [
        abs(current["filtered_target_pos"] - previous["filtered_target_pos"])
        for previous, current in zip(samples, samples[1:])
    ]
    raw_variation = sum(raw_steps)
    filtered_variation = sum(filtered_steps)
    summary = {
        "cutoff_hz": cutoff_hz,
        "samples": len(samples),
        "raw": {
            "max_step": max(raw_steps),
            "p95_step": _quantile(raw_steps, 0.95),
            "total_variation": raw_variation,
        },
        "filtered": {
            "max_step": max(filtered_steps),
            "p95_step": _quantile(filtered_steps, 0.95),
            "total_variation": filtered_variation,
            "min_target_pos": min(row["filtered_target_pos"] for row in samples),
            "max_target_pos": max(row["filtered_target_pos"] for row in samples),
        },
        "variation_reduction_fraction": (
            None if raw_variation == 0.0 else 1.0 - filtered_variation / raw_variation
        ),
        "nominal_30hz": {
            "alpha": 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / 30.0),
            "rise_time_10_90_s": 2.197 / (2.0 * math.pi * cutoff_hz),
        },
    }
    return samples, summary


def write_artifacts(samples: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pv_3hz_filter.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4.8))
    times = [row["time_s"] for row in samples]
    axis.plot(times, [row["raw_target_pos"] for row in samples], label="raw relative target", alpha=0.55)
    axis.plot(times, [row["filtered_target_pos"] for row in samples], label="3 Hz low-pass", linewidth=2.0)
    axis.set_xlabel("Time since relative reference latch (s)")
    axis.set_ylabel("Proposed gripper position")
    axis.set_title("Relative PV target: raw vs time-aware 3 Hz low-pass")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "pv_3hz_filter.png", dpi=150)
    plt.close(figure)

    (output_dir / "pv_3hz_filter.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff-hz", type=float, default=3.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0.0:
        parser.error("--cutoff-hz must be positive and finite")
    with args.sidecar.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    samples, summary = replay(rows, args.cutoff_hz)
    if not samples:
        parser.error("sidecar has no relative reference samples")
    write_artifacts(samples, summary, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
