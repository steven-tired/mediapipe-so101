#!/usr/bin/env python3
"""Slow-scan one labeled rigid object and teach its light/hard position profile."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys
import time

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    read_gripper_telemetry,
    serialize_telemetry_snapshot,
    slow_close_waypoints,
    summarize_target_current,
)
from pressurevision_integration.pv_object_profile import (
    PressureVisionObjectProfile,
    save_object_profile,
)


DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00"
DEFAULT_ARM_ID = "so101_follower_1"
MAX_TEMPERATURE_C = 55.0
MAX_TEMPERATURE_HIGH_RUN = 1


def _max_consecutive_above(values: list[int | None], threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        if value is not None and float(value) > threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def profile_trial_summary(samples, target: float) -> dict:
    if not samples:
        raise ValueError(f"target {target}: no telemetry samples")
    base = summarize_target_current(samples)
    actual = [float(sample.gripper_pos) for sample in samples]
    result = {
        "target": float(target),
        **base,
        "median_actual_pos": sorted(actual)[len(actual) // 2],
        "position_error": sorted(actual)[len(actual) // 2] - float(target),
        "actual_span": max(actual) - min(actual),
    }
    result["temperature_over_limit_max_run"] = _max_consecutive_above(
        [sample.present_temperature for sample in samples],
        MAX_TEMPERATURE_C,
    )
    return result


def validate_profile_trials(
    records: list[dict],
    *,
    light_pos: float,
    hard_pos: float,
    max_current: float = 50.0,
    max_temperature_c: float = MAX_TEMPERATURE_C,
    max_actual_span: float = 1.0,
    min_repeats: int = 1,
) -> dict:
    selected_records = [record for record in records if record.get("selected_repeat") is not None]
    if selected_records:
        records = selected_records
    grouped: dict[float, list[dict]] = {}
    for record in records:
        grouped.setdefault(float(record["target"]), []).append(record)
    by_target = {}
    for target, target_records in grouped.items():
        if len(target_records) < min_repeats:
            continue
        by_target[target] = {
            "target": target,
            "repeats": len(target_records),
            "mean_current": mean(float(record["mean_current"]) for record in target_records),
            "mean_load": mean(float(record.get("mean_load", 0.0)) for record in target_records),
            "max_current": max(float(record["max_current"]) for record in target_records),
            "max_temperature": max(float(record["max_temperature"]) for record in target_records),
            "temperature_over_limit_max_run": max(
                int(record.get("temperature_over_limit_max_run", 0))
                for record in target_records
            ),
            "actual_span": max(float(record["actual_span"]) for record in target_records),
        }
    if hard_pos not in by_target or light_pos not in by_target:
        raise ValueError(
            f"light_pos and hard_pos each need {min_repeats} completed repeat summaries"
        )
    if not hard_pos < light_pos:
        raise ValueError("hard_pos must be lower than light_pos")
    light = by_target[light_pos]
    hard = by_target[hard_pos]
    reasons = []
    if light["max_current"] > max_current or hard["max_current"] > max_current:
        reasons.append(f"current exceeds {max_current:g}")
    if (
        light["temperature_over_limit_max_run"] > MAX_TEMPERATURE_HIGH_RUN
        or hard["temperature_over_limit_max_run"] > MAX_TEMPERATURE_HIGH_RUN
    ):
        reasons.append(
            f"temperature exceeds {max_temperature_c:g} C for more than "
            f"{MAX_TEMPERATURE_HIGH_RUN} consecutive sample"
        )
    if light["actual_span"] > max_actual_span or hard["actual_span"] > max_actual_span:
        reasons.append(f"actual position span exceeds {max_actual_span:g}")
    if reasons:
        raise ValueError("; ".join(reasons))
    return {
        "light": light,
        "hard": hard,
        "observed_current_gap": hard["mean_current"] - light["mean_current"],
        "observed_load_gap": hard["mean_load"] - light["mean_load"],
        "max_current": max_current,
        "max_temperature_c": max_temperature_c,
        "max_actual_span": max_actual_span,
        "min_repeats": min_repeats,
    }


def format_sweep_summary(summary: dict) -> str:
    return (
        f"[sweep] target={float(summary['target']):g} "
        f"actual={float(summary['median_actual_pos']):.2f} "
        f"load_mean={float(summary.get('mean_load', 0.0)):.2f} "
        f"current_mean={float(summary['mean_current']):.2f} "
        f"current_max={float(summary['max_current']):.2f} "
        f"temp_max={float(summary['max_temperature']):.1f}C "
        f"temp_high_run={int(summary['temperature_over_limit_max_run'])} "
        f"actual_span={float(summary['actual_span']):.2f}"
    )


def validate_profile_positions(
    light_pos: float,
    hard_pos: float,
    *,
    scanned_targets: set[float] | None = None,
) -> tuple[float, float]:
    light_pos = float(light_pos)
    hard_pos = float(hard_pos)
    if not hard_pos < light_pos < 95.0:
        raise ValueError("hard-pos < light-pos < 95 is required")
    if scanned_targets is not None:
        missing = sorted({light_pos, hard_pos} - scanned_targets, reverse=True)
        if missing:
            available = ", ".join(f"{target:g}" for target in sorted(scanned_targets, reverse=True))
            requested = ", ".join(f"{target:g}" for target in missing)
            raise ValueError(
                f"selected positions were not completed by the sweep: {requested}; "
                f"available targets: {available}"
            )
    return light_pos, hard_pos


def scan_targets(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not 0.0 <= stop < start <= 100.0:
        raise ValueError("0 <= scan-stop < scan-start <= 100 is required")
    if step <= 0.0:
        raise ValueError("scan-step must be positive")
    targets = []
    target = start
    while target >= stop - 1e-9:
        targets.append(round(target, 6))
        target -= step
    return targets


def prompt_profile_positions(summaries: list[dict], *, input_fn=None) -> tuple[float, float]:
    if input_fn is None:
        input_fn = input
    try:
        light_pos = float(input_fn("Enter light position from the completed sweep: ").strip())
        hard_pos = float(input_fn("Enter hard position from the completed sweep: ").strip())
    except ValueError as exc:
        raise ValueError("light and hard positions must be numeric") from exc
    scanned_targets = {float(summary["target"]) for summary in summaries}
    return validate_profile_positions(
        light_pos,
        hard_pos,
        scanned_targets=scanned_targets,
    )


def _connect(port: str, arm_id: str) -> SOFollower:
    robot = SOFollower(
        SO101FollowerConfig(
            port=port,
            id=arm_id,
            use_degrees=False,
            cameras={},
            disable_torque_on_disconnect=True,
        )
    )
    robot.connect(calibrate=False)
    return robot


def _send_gripper(robot: SOFollower, position: float) -> None:
    observation = robot.get_observation()
    action = {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
    action["gripper.pos"] = float(position)
    robot.send_action(action)


def _hold(
    robot: SOFollower,
    target: float,
    hold_s: float,
    steps: int,
    settle_s: float,
) -> tuple[list, list[dict]]:
    samples = []
    for waypoint in slow_close_waypoints(95.0, target, steps):
        _send_gripper(robot, waypoint)
        time.sleep(0.04)
    time.sleep(settle_s)
    started = time.perf_counter()
    while time.perf_counter() - started < hold_s:
        _send_gripper(robot, target)
        samples.append(read_gripper_telemetry(robot, target, time.perf_counter() - started))
        time.sleep(0.1)
    return samples, [serialize_telemetry_snapshot(sample, target=target, sample_index=index)
                     for index, sample in enumerate(samples)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--arm-id", default=DEFAULT_ARM_ID)
    parser.add_argument(
        "--light-pos",
        type=float,
        help="preselected light position; omit with --hard-pos to choose after the sweep",
    )
    parser.add_argument(
        "--hard-pos",
        type=float,
        help="preselected hard position; omit with --light-pos to choose after the sweep",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--scan-start", type=float, default=95.0)
    parser.add_argument("--scan-stop", type=float, default=20.0)
    parser.add_argument(
        "--scan-step",
        type=float,
        default=2.0,
        help="position spacing for the exploratory sweep; use 1 or less for a narrow rigid range",
    )
    parser.add_argument("--yes", action="store_true", help="skip the physical confirmation prompt")
    args = parser.parse_args(argv)
    protocol = {
        "hold_s": args.hold_s,
        "settle_s": args.settle_s,
        "repeats": args.repeats,
        "steps": args.steps,
        "max_temperature_c": MAX_TEMPERATURE_C,
        "max_temperature_high_run": MAX_TEMPERATURE_HIGH_RUN,
        "scan_start": args.scan_start,
        "scan_stop": args.scan_stop,
        "scan_step": args.scan_step,
    }
    if args.repeats < 3 or args.hold_s <= 0 or args.settle_s < 0 or args.steps <= 0:
        parser.error("repeats >= 3, hold-s > 0, settle-s >= 0 and steps > 0 are required")
    try:
        sweep_targets = scan_targets(args.scan_start, args.scan_stop, args.scan_step)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.light_pos is None) != (args.hard_pos is None):
        parser.error("--light-pos and --hard-pos must be provided together")
    if args.yes and args.light_pos is None:
        parser.error("--yes requires preselected --light-pos and --hard-pos")
    if args.light_pos is not None:
        try:
            validate_profile_positions(args.light_pos, args.hard_pos)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.yes and input("Object fixed, jaws clear, type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    robot = _connect(args.port, args.arm_id)
    summaries = []
    telemetry = []
    try:
        _send_gripper(robot, 95.0)
        time.sleep(1.0)
        for target in sweep_targets:
            samples, rows = _hold(
                robot,
                target,
                min(0.5, args.hold_s),
                args.steps,
                args.settle_s,
            )
            summary = profile_trial_summary(samples, target)
            summary["selected_repeat"] = None
            summaries.append(summary)
            telemetry.extend(rows)
            print(format_sweep_summary(summary), flush=True)
            _send_gripper(robot, 95.0)
            time.sleep(0.3)
            if (
                summaries[-1]["max_current"] >= 50.0
                or summaries[-1]["temperature_over_limit_max_run"]
                > MAX_TEMPERATURE_HIGH_RUN
            ):
                break
        scanned_targets = {float(summary["target"]) for summary in summaries}
        if args.light_pos is None:
            print("[sweep] complete; select light and hard from the targets above", flush=True)
            try:
                light_pos, hard_pos = prompt_profile_positions(summaries)
            except ValueError as exc:
                partial_evidence = {
                    "schema_version": 1,
                    "object_id": args.object_id,
                    "arm_id": args.arm_id,
                    "status": "NO_GO",
                    "failure_stage": "sweep_before_selected_repeats",
                    "validation_error": str(exc),
                    "protocol": protocol,
                    "summaries": summaries,
                    "selected": None,
                    "telemetry_samples": telemetry,
                }
                args.evidence.parent.mkdir(parents=True, exist_ok=True)
                args.evidence.write_text(
                    json.dumps(partial_evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"[object] wrote evidence {args.evidence}")
                parser.error(f"{exc}; evidence retained at {args.evidence}")
        else:
            try:
                light_pos, hard_pos = validate_profile_positions(
                    args.light_pos,
                    args.hard_pos,
                    scanned_targets=scanned_targets,
                )
            except ValueError as exc:
                partial_evidence = {
                    "schema_version": 1,
                    "object_id": args.object_id,
                    "arm_id": args.arm_id,
                    "status": "NO_GO",
                    "failure_stage": "sweep_before_selected_repeats",
                    "validation_error": str(exc),
                    "protocol": protocol,
                    "summaries": summaries,
                    "selected": None,
                    "telemetry_samples": telemetry,
                }
                args.evidence.parent.mkdir(parents=True, exist_ok=True)
                args.evidence.write_text(
                    json.dumps(partial_evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"[object] wrote evidence {args.evidence}")
                parser.error(f"{exc}; evidence retained at {args.evidence}")
        if not args.yes:
            confirmation = input(
                f"Confirm light={light_pos:g}, "
                f"hard={hard_pos:g} for this object (type YES): "
            ).strip()
            if confirmation != "YES":
                raise SystemExit("aborted after sweep confirmation")
        selected_targets = {light_pos, hard_pos}
        for repeat in range(args.repeats):
            for target in sorted(selected_targets, reverse=True):
                _send_gripper(robot, 95.0)
                time.sleep(0.5)
                samples, rows = _hold(
                    robot,
                    target,
                    args.hold_s,
                    args.steps,
                    args.settle_s,
                )
                summary = profile_trial_summary(samples, target)
                summary["selected_repeat"] = repeat + 1
                summaries.append(summary)
                telemetry.extend(rows)
                print(f"[repeat {repeat + 1}] {format_sweep_summary(summary)}", flush=True)
                _send_gripper(robot, 95.0)
                time.sleep(0.5)
    finally:
        try:
            _send_gripper(robot, 95.0)
        finally:
            robot.disconnect()

    validation_error = None
    try:
        selected = validate_profile_trials(
            summaries,
            light_pos=light_pos,
            hard_pos=hard_pos,
            min_repeats=args.repeats,
        )
    except ValueError as exc:
        selected = None
        validation_error = str(exc)
    evidence = {
        "schema_version": 1,
        "object_id": args.object_id,
        "arm_id": args.arm_id,
        "status": "NO_GO" if validation_error else "PASS",
        "validation_error": validation_error,
        "protocol": protocol,
        "summaries": summaries,
        "selected": selected,
        "telemetry_samples": telemetry,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[object] wrote evidence {args.evidence}")
    if validation_error is not None:
        raise ValueError(f"{validation_error}; evidence retained at {args.evidence}")
    evidence_hash = sha256(args.evidence.read_bytes()).hexdigest()
    profile = PressureVisionObjectProfile(
        object_id=args.object_id,
        arm_id=args.arm_id,
        open_pos=95.0,
        light_pos=light_pos,
        hard_pos=hard_pos,
        max_current=50.0,
        max_temperature_c=MAX_TEMPERATURE_C,
        sweep_evidence_sha256=evidence_hash,
    )
    save_object_profile(args.out, profile)
    print(f"[object] wrote profile {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
