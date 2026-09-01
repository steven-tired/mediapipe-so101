#!/usr/bin/env python3
"""Replay the stability-first PV TRACK/HOLD candidate on existing evidence."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from lerobot_teleoperator_so101_webcam.ee_control import GRIP_LATCH_ENTER
from pressurevision_integration.pv_relative_mapping import (
    RELATIVE_PRESSURE_LOW_PASS_HZ,
    TRACK_HOLD_BASELINE_HZ,
    TRACK_HOLD_ENTER_DELTA,
    TRACK_HOLD_EXIT_DELTA,
    TRACK_HOLD_MAX_OUTPUT_RATE_PER_S,
    TRACK_HOLD_SETTLE_S,
    TRACK_HOLD_WINDOW_S,
    PressureTrackHoldStabilizer,
)


LABEL_VALUES = {"light": 0.0, "medium": 0.5, "hard": 1.0}
DEFAULT_INTENT_EVENTS = (
    ("rise", 5.3),
    ("fall", 6.75),
    ("rise", 10.25),
    ("fall", 14.75),
    ("rise", 17.75),
    ("fall", 21.75),
    ("rise", 24.25),
    ("fall", 27.75),
    ("rise", 30.25),
    ("fall", 32.75),
    ("rise", 34.25),
    ("fall", 35.75),
)
SIGNAL_ONSET_DELTA = 0.05
OUTPUT_RESPONSE_DELTA = 0.025


def _low_pass(times: list[float], values: list[float], cutoff_hz: float) -> list[float]:
    if not times or len(times) != len(values):
        raise ValueError("times and values must be non-empty and equally sized")
    output = []
    filtered = float(values[0])
    previous_t = float(times[0])
    for observed_at_s, value in zip(times, values):
        observed_at_s = float(observed_at_s)
        dt_s = max(0.0, observed_at_s - previous_t)
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)
        filtered += alpha * (float(value) - filtered)
        output.append(filtered)
        previous_t = observed_at_s
    return output


def _completed_steady_trials(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if (
            row.get("row_type") == "sample"
            and row.get("phase") == "steady"
            and row.get("trial_status") == "complete"
        ):
            grouped.setdefault(int(row["trial_index"]), []).append(row)
    if not grouped:
        raise ValueError("capture has no completed steady trials")
    return grouped


def fit_three_point_anchors(
    trials: dict[int, list[dict]], discard_s: float
) -> dict[str, float]:
    anchors = {}
    for label in LABEL_VALUES:
        trial_medians = []
        for rows in trials.values():
            rows = sorted(rows, key=lambda row: float(row["t"]))
            if rows[0]["target_label"] != label:
                continue
            first_t = float(rows[0]["t"])
            values = [
                float(row["sum_kpa"])
                for row in rows
                if float(row["t"]) - first_t >= discard_s
            ]
            if values:
                trial_medians.append(float(np.median(values)))
        if not trial_medians:
            raise ValueError(f"capture has no post-discard {label} samples")
        anchors[label] = float(np.median(trial_medians))
    ordered = list(anchors.values())
    if not all(lower < upper for lower, upper in zip(ordered, ordered[1:])):
        raise ValueError("light/medium/hard anchors are not ordered")
    return anchors


def _map_sum_kpa(value: float, anchors: dict[str, float]) -> float:
    return float(
        np.interp(
            value,
            [anchors[label] for label in LABEL_VALUES],
            list(LABEL_VALUES.values()),
        )
    )


def replay_steady_holds(
    trials: dict[int, list[dict]],
    *,
    anchors: dict[str, float],
    discard_s: float,
    max_closure: float,
) -> tuple[list[dict], list[dict]]:
    records = []
    summaries = []
    for trial_index, rows in sorted(trials.items()):
        rows = sorted(rows, key=lambda row: float(row["t"]))
        first_t = float(rows[0]["t"])
        times = [float(row["t"]) - first_t for row in rows]
        mapped = [_map_sum_kpa(float(row["sum_kpa"]), anchors) for row in rows]
        filtered = _low_pass(times, mapped, RELATIVE_PRESSURE_LOW_PASS_HZ)
        selected = [index for index, t in enumerate(times) if t >= discard_s]
        if not selected:
            raise ValueError(f"trial {trial_index} has no post-discard samples")

        stabilizer = PressureTrackHoldStabilizer()
        outputs = []
        track_transitions = 0
        for index in selected:
            decision = stabilizer.update(filtered[index], times[index])
            track_transitions += decision.transition == "HOLD_TO_TRACK"
            outputs.append(decision.output_value)
            records.append(
                {
                    "dataset": "steady",
                    "segment": str(trial_index),
                    "direction": "",
                    "time_s": times[index],
                    "input_3hz": filtered[index],
                    "baseline": decision.baseline,
                    "robust_residual": decision.robust_residual,
                    "state": decision.state,
                    "output": decision.output_value,
                }
            )
        movement = (max(outputs) - min(outputs)) * max_closure
        summaries.append(
            {
                "trial_index": trial_index,
                "label": rows[0]["target_label"],
                "samples": len(outputs),
                "track_transitions": track_transitions,
                "gripper_movement": movement,
            }
        )
    return records, summaries


def load_intent_trace(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    moving = [row for row in rows if row.get("state") == "MOVING"]
    start = next(
        (
            index
            for index, row in enumerate(moving)
            if float(row["base_gripper_pos"]) <= GRIP_LATCH_ENTER
        ),
        None,
    )
    if start is None:
        raise ValueError("intent sidecar has no sustained right-grasp episode")
    moving = moving[start:]
    first_t = float(moving[0]["control_observed_at_s"])
    times = [float(row["control_observed_at_s"]) - first_t for row in moving]
    pressure = [float(row["pressure"]) for row in moving]
    return times, _low_pass(times, pressure, RELATIVE_PRESSURE_LOW_PASS_HZ)


def replay_intent_events(
    times: list[float],
    values: list[float],
    events=DEFAULT_INTENT_EVENTS,
) -> tuple[list[dict], list[dict]]:
    records = []
    summaries = []
    event_ends = [start for _, start in events[1:]] + [times[-1]]
    for event_index, ((direction, start_s), end_s) in enumerate(zip(events, event_ends)):
        start = int(np.searchsorted(times, start_s))
        stop = int(np.searchsorted(times, end_s, side="right"))
        event_times = times[start:stop]
        event_values = values[start:stop]
        if not event_times:
            raise ValueError(f"intent event {event_index} is outside the trace")
        stabilizer = PressureTrackHoldStabilizer()
        output = []
        for observed_at_s, value in zip(event_times, event_values):
            decision = stabilizer.update(value, observed_at_s)
            output.append(decision.output_value)
            records.append(
                {
                    "dataset": "intent",
                    "segment": str(event_index),
                    "direction": direction,
                    "time_s": observed_at_s,
                    "input_3hz": value,
                    "baseline": decision.baseline,
                    "robust_residual": decision.robust_residual,
                    "state": decision.state,
                    "output": decision.output_value,
                }
            )

        sign = 1.0 if direction == "rise" else -1.0
        signal_onset = next(
            (
                index
                for index, value in enumerate(event_values)
                if sign * (value - event_values[0]) >= SIGNAL_ONSET_DELTA
            ),
            None,
        )
        output_response = next(
            (
                index
                for index, value in enumerate(output)
                if sign * (value - output[0]) >= OUTPUT_RESPONSE_DELTA
            ),
            None,
        )
        response_s = (
            None
            if signal_onset is None or output_response is None
            else max(0.0, event_times[output_response] - event_times[signal_onset])
        )
        summaries.append(
            {
                "event_index": event_index,
                "direction": direction,
                "start_s": start_s,
                "end_s": end_s,
                "response_s": response_s,
            }
        )
    return records, summaries


def analyze(
    capture_rows: list[dict],
    metadata: dict,
    intent_rows: list[dict[str, str]],
    *,
    max_closure: float,
) -> tuple[list[dict], dict]:
    discard_s = float(metadata.get("visible_steady_seconds") or 0.0)
    trials = _completed_steady_trials(capture_rows)
    anchors = fit_three_point_anchors(trials, discard_s)
    steady_records, steady = replay_steady_holds(
        trials,
        anchors=anchors,
        discard_s=discard_s,
        max_closure=max_closure,
    )
    intent_times, intent_values = load_intent_trace(intent_rows)
    intent_records, intent = replay_intent_events(intent_times, intent_values)

    responses = [event["response_s"] for event in intent if event["response_s"] is not None]
    max_hold_movement = max(trial["gripper_movement"] for trial in steady)
    report = {
        "candidate": "stability_first_track_hold_v1",
        "runtime_integration": False,
        "actuation_authorized": False,
        "parameters": {
            "input_low_pass_hz": RELATIVE_PRESSURE_LOW_PASS_HZ,
            "drift_baseline_hz": TRACK_HOLD_BASELINE_HZ,
            "residual_median_window_s": TRACK_HOLD_WINDOW_S,
            "enter_track_delta": TRACK_HOLD_ENTER_DELTA,
            "return_hold_delta": TRACK_HOLD_EXIT_DELTA,
            "return_hold_settle_s": TRACK_HOLD_SETTLE_S,
            "max_output_rate_per_s": TRACK_HOLD_MAX_OUTPUT_RATE_PER_S,
            "max_closure": max_closure,
            "signal_onset_delta": SIGNAL_ONSET_DELTA,
            "output_response_delta": OUTPUT_RESPONSE_DELTA,
        },
        "three_point_anchors_sum_kpa": anchors,
        "metrics": {
            "stable_hold_max_gripper_movement": max_hold_movement,
            "deliberate_adjustment_response": {
                "events": len(intent),
                "responded": len(responses),
                "missed": len(intent) - len(responses),
                "median_s": None if not responses else float(np.median(responses)),
                "max_s": None if not responses else max(responses),
            },
        },
        "steady_trials": steady,
        "intent_events": intent,
        "verdict": "OFFLINE_CHARACTERIZED_NO_PRESET_ACCEPTANCE_GATE",
        "tradeoff": "very slow adjustments can be absorbed as drift",
    }
    return steady_records + intent_records, report


def write_artifacts(records: list[dict], report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pv_track_hold_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    steady = [row for row in records if row["dataset"] == "steady"]
    for segment in sorted({row["segment"] for row in steady}, key=int):
        selected = [row for row in steady if row["segment"] == segment]
        t0 = selected[0]["time_s"]
        axes[0].plot(
            [row["time_s"] - t0 for row in selected],
            [row["input_3hz"] for row in selected],
            color="tab:blue",
            alpha=0.22,
        )
        axes[0].plot(
            [row["time_s"] - t0 for row in selected],
            [row["output"] for row in selected],
            color="tab:orange",
            alpha=0.65,
        )
    axes[0].set_title("Steady holds: 3 Hz input (blue) and frozen output (orange)")
    axes[0].set_xlabel("Seconds after visible steady interval")
    axes[0].set_ylabel("normalized closure")

    intent = [row for row in records if row["dataset"] == "intent"]
    axes[1].plot(
        [row["time_s"] for row in intent],
        [row["input_3hz"] for row in intent],
        label="3 Hz input",
        alpha=0.5,
    )
    axes[1].plot(
        [row["time_s"] for row in intent],
        [row["output"] for row in intent],
        label="TRACK/HOLD output",
        linewidth=1.5,
    )
    axes[1].set_title("Attempt05 annotated adjustment segments (each starts in HOLD)")
    axes[1].set_xlabel("Seconds from sustained right-pinch start")
    axes[1].set_ylabel("normalized closure")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.05, 1.05)
    figure.tight_layout()
    figure.savefig(output_dir / "pv_track_hold_replay.png", dpi=150)
    plt.close(figure)

    (output_dir / "pv_track_hold_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady-session", type=Path, required=True)
    parser.add_argument("--intent-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-closure", type=float, default=2.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_closure) or args.max_closure <= 0.0:
        parser.error("--max-closure must be positive and finite")

    capture_path = args.steady_session / "capture.jsonl"
    manifest = json.loads((args.steady_session / "manifest.json").read_text(encoding="utf-8"))
    digest = sha256(capture_path.read_bytes()).hexdigest()
    if manifest.get("status") != "complete" or digest != manifest.get("capture_jsonl_sha256"):
        parser.error("steady session is incomplete or its checksum does not match")
    capture = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    if not capture or capture[0].get("row_type") != "metadata":
        parser.error("steady capture must begin with metadata")
    with args.intent_sidecar.open(newline="", encoding="utf-8") as handle:
        intent_rows = list(csv.DictReader(handle))

    records, report = analyze(
        capture[1:],
        capture[0],
        intent_rows,
        max_closure=args.max_closure,
    )
    report["evidence"] = {
        "steady_capture_sha256": digest,
        "intent_sidecar_sha256": sha256(args.intent_sidecar.read_bytes()).hexdigest(),
    }
    write_artifacts(records, report, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
