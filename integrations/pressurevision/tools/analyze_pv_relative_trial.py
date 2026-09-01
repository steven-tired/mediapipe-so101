#!/usr/bin/env python3
"""Summarize a PV mapped-control trial and generate gripper time-series artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


CONTACT_STATUSES = {"active", "pv_abstain_continuous"}


def _float(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unique_motor_rows(rows: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for row in rows:
        observed_at_s = _float(row, "motor_observed_at_s")
        if observed_at_s is None or observed_at_s in seen:
            continue
        seen.add(observed_at_s)
        unique.append(row)
    return unique


def write_gripper_position_artifacts(rows: list[dict], output_dir: str | Path) -> dict:
    """Write one row per motor read and a command-vs-readback time plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gripper_position.csv"
    plot_path = output_dir / "gripper_position.png"
    motor_rows = _unique_motor_rows(rows)
    first_t = (
        None if not motor_rows else _float(motor_rows[0], "motor_observed_at_s")
    )
    records = []
    for row in motor_rows:
        observed_at_s = _float(row, "motor_observed_at_s")
        commanded = _float(row, "commanded_gripper_pos")
        observed = _float(row, "observed_gripper_pos")
        records.append(
            {
                "time_s": (
                    None
                    if first_t is None or observed_at_s is None
                    else observed_at_s - first_t
                ),
                "commanded_gripper_pos": commanded,
                "observed_gripper_pos": observed,
                "absolute_error": (
                    None
                    if commanded is None or observed is None
                    else abs(commanded - observed)
                ),
            }
        )

    fieldnames = (
        "time_s",
        "commanded_gripper_pos",
        "observed_gripper_pos",
        "absolute_error",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5))
    times = [record["time_s"] for record in records]
    commanded = [record["commanded_gripper_pos"] for record in records]
    observed = [record["observed_gripper_pos"] for record in records]
    if records:
        axis.plot(times, commanded, label="commanded", linewidth=1.5)
        axis.plot(times, observed, label="observed", linewidth=1.5)
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No motor readback samples", ha="center", va="center")
    axis.set_xlabel("Time since first motor sample (s)")
    axis.set_ylabel("Gripper position")
    axis.set_title("SO-101 gripper command and readback")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return {"csv": str(csv_path), "plot": str(plot_path), "rows": len(records)}


def _track_hold_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("relative_track_hold_state") in {"HOLD", "TRACK"}
    ]


def analyze_track_hold(rows: list[dict]) -> dict:
    valid_rows = _track_hold_rows(rows)
    state_counts = {
        state: sum(row.get("relative_track_hold_state") == state for row in valid_rows)
        for state in ("HOLD", "TRACK")
    }
    transitions = {"hold_to_track": 0, "track_to_hold": 0}
    for previous, current in zip(valid_rows, valid_rows[1:]):
        pair = (
            previous.get("relative_track_hold_state"),
            current.get("relative_track_hold_state"),
        )
        if pair == ("HOLD", "TRACK"):
            transitions["hold_to_track"] += 1
        elif pair == ("TRACK", "HOLD"):
            transitions["track_to_hold"] += 1

    outputs = [
        value
        for row in valid_rows
        if (value := _float(row, "relative_track_hold_output")) is not None
    ]
    hold_ranges = []
    episode_outputs = []
    for row in valid_rows:
        if row.get("relative_track_hold_state") != "HOLD":
            if episode_outputs:
                hold_ranges.append(max(episode_outputs) - min(episode_outputs))
                episode_outputs = []
            continue
        output = _float(row, "relative_track_hold_output")
        if output is not None:
            episode_outputs.append(output)
    if episode_outputs:
        hold_ranges.append(max(episode_outputs) - min(episode_outputs))

    return {
        "valid_rows": len(valid_rows),
        "state_rows": state_counts,
        "transitions": transitions,
        "hold_episodes": len(hold_ranges),
        "max_within_hold_output_range": None if not hold_ranges else max(hold_ranges),
        "output_range": None if not outputs else [min(outputs), max(outputs)],
    }


def write_track_hold_artifacts(rows: list[dict], output_dir: str | Path) -> dict:
    """Write the live input, stabilized output, and TRACK/HOLD state over time."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "track_hold_timeseries.csv"
    plot_path = output_dir / "track_hold_timeseries.png"
    valid_rows = _track_hold_rows(rows)
    first_t = None if not valid_rows else _float(valid_rows[0], "control_observed_at_s")
    records = []
    for row in valid_rows:
        observed_at_s = _float(row, "control_observed_at_s")
        records.append(
            {
                "time_s": (
                    None
                    if first_t is None or observed_at_s is None
                    else observed_at_s - first_t
                ),
                "pressure": _float(row, "pressure"),
                "residual": _float(row, "relative_track_hold_residual"),
                "stabilized_output": _float(row, "relative_track_hold_output"),
                "track_hold_state": row.get("relative_track_hold_state", ""),
                "proposed_gripper_pos": _float(row, "proposed_gripper_pos"),
                "actual_gripper_pos": _float(row, "actual_gripper_pos"),
            }
        )

    fieldnames = tuple(records[0]) if records else (
        "time_s",
        "pressure",
        "residual",
        "stabilized_output",
        "track_hold_state",
        "proposed_gripper_pos",
        "actual_gripper_pos",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5))
    times = [record["time_s"] for record in records]
    if records:
        axis.plot(times, [record["pressure"] for record in records], label="PV input", alpha=0.7)
        axis.plot(
            times,
            [record["stabilized_output"] for record in records],
            label="TRACK/HOLD output",
            linewidth=1.8,
        )
        track_times = [
            record["time_s"]
            for record in records
            if record["track_hold_state"] == "TRACK"
        ]
        track_outputs = [
            record["stabilized_output"]
            for record in records
            if record["track_hold_state"] == "TRACK"
        ]
        axis.scatter(track_times, track_outputs, label="TRACK", s=8, color="tab:red")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No TRACK/HOLD samples", ha="center", va="center")
    axis.set_xlabel("Time since first TRACK/HOLD sample (s)")
    axis.set_ylabel("Normalized pressure / output")
    axis.set_ylim(-0.05, 1.05)
    pressure_modes = {row.get("pressure_mode", "") for row in valid_rows}
    run_mode = "apply" if any(mode.endswith("_apply") for mode in pressure_modes) else "shadow"
    axis.set_title(f"PressureVision stability-first {run_mode}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return {"csv": str(csv_path), "plot": str(plot_path), "rows": len(records)}


def analyze_relative_trial(rows: list[dict]) -> dict:
    moving = [row for row in rows if row.get("state") == "MOVING"]
    contact = [row for row in moving if row.get("pressure_status") in CONTACT_STATUSES]
    pressures = [value for row in contact if (value := _float(row, "pressure")) is not None]
    zero_count = sum(value <= 0.001 for value in pressures)

    motor_rows = _unique_motor_rows(rows)
    errors = []
    currents = []
    temperatures = []
    for row in motor_rows:
        commanded = _float(row, "commanded_gripper_pos")
        observed = _float(row, "observed_gripper_pos")
        if commanded is not None and observed is not None:
            errors.append(abs(commanded - observed))
        current = _float(row, "present_current")
        if current is not None:
            currents.append(abs(current))
        temperature = _float(row, "present_temperature")
        if temperature is not None:
            temperatures.append(temperature)

    complete = bool(pressures and errors and currents and temperatures)
    return {
        "data_complete": complete,
        "signal": {
            "contact_rows": len(pressures),
            "zero_fraction": None if not pressures else zero_count / len(pressures),
        },
        "tracking": {
            "motor_samples": len(errors),
            "median_absolute_error": None if not errors else median(errors),
            "max_absolute_error": None if not errors else max(errors),
        },
        "motor": {
            "samples": len(motor_rows),
            "peak_absolute_current": None if not currents else max(currents),
            "max_temperature_c": None if not temperatures else max(temperatures),
        },
        "track_hold": analyze_track_hold(rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read_csv(args.sidecar)
    report = analyze_relative_trial(rows)
    report["artifacts"] = {
        "gripper_position": write_gripper_position_artifacts(rows, args.out.parent),
        "track_hold": write_track_hold_artifacts(rows, args.out.parent),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["data_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
