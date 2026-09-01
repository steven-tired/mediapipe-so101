#!/usr/bin/env python3
"""Run fixed-position carton lift or static-closure trials with the SO-101.

The operator places the carton at the fixed pickup mark and presses Enter. The
robot then either performs the randomized lift protocol or closes and releases
without lifting for a static deformation check. Motor effort proxies are
recorded but never gate or modify motion.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from lerobot_teleoperator_so101_webcam.gripper_hardware import GripperTelemetrySampler
from lerobot_teleoperator_so101_webcam.paths import urdf_path


ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00"
ARM_ID = "so101_follower_1"
DEFAULT_TARGETS = (28.0, 27.0, 26.0, 25.0, 24.0)
STATIC_CLOSURE_TARGETS = (32.0, 29.0, 26.0, 23.0, 20.0)
POSE_SCHEMA_VERSION = 1
OUTCOME_FIELDS = (
    "trial_index",
    "block",
    "target",
    "outcome",
    "tilt_grade",
    "crease_grade",
    "first_slip_s_from_lift",
    "notes",
)
STATIC_OUTCOME_FIELDS = (
    "trial_index",
    "block",
    "target",
    "residual_deformation_grade",
    "functional_damage",
    "notes",
)


def parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    if len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("targets must be unique")
    if any(not math.isfinite(target) or not 0.0 <= target <= 100.0 for target in targets):
        raise argparse.ArgumentTypeError("targets must be finite values in [0, 100]")
    return targets


def build_trial_plan(
    targets: tuple[float, ...],
    repeats: int,
    seed: int,
    *,
    randomize: bool = True,
) -> list[dict[str, int | float]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    plan: list[dict[str, int | float]] = []
    for block in range(1, repeats + 1):
        block_targets = list(targets)
        if randomize:
            rng.shuffle(block_targets)
        for target in block_targets:
            plan.append(
                {
                    "trial_index": len(plan) + 1,
                    "block": block,
                    "target": float(target),
                }
            )
    return plan


def interpolate_pose(
    start: dict[str, float],
    end: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    if start.keys() != end.keys():
        raise ValueError("pose keys must match")
    alpha = float(alpha)
    return {key: (1.0 - alpha) * start[key] + alpha * end[key] for key in start}


def compute_lift_pose(
    kinematics: RobotKinematics,
    motors: list[str],
    pickup_pose: dict[str, float],
    lift_distance_m: float,
) -> tuple[dict[str, float], list[float], list[float]]:
    if lift_distance_m <= 0.0:
        raise ValueError("lift_distance_m must be positive")
    q_pick = np.array([pickup_pose[f"{motor}.pos"] for motor in motors], dtype=float)
    t_pick = kinematics.forward_kinematics(q_pick)
    t_desired = t_pick.copy()
    t_desired[2, 3] += lift_distance_m
    q_lift = kinematics.inverse_kinematics(
        q_pick,
        t_desired,
        position_weight=1.0,
        orientation_weight=0.0,
    )
    lift_pose = {f"{motor}.pos": float(q_lift[index]) for index, motor in enumerate(motors)}
    lift_pose["gripper.pos"] = float(pickup_pose["gripper.pos"])
    t_lift = kinematics.forward_kinematics(q_lift)
    pickup_xyz = t_pick[:3, 3].astype(float)
    lift_xyz = t_lift[:3, 3].astype(float)
    lateral_error = float(np.linalg.norm(lift_xyz[:2] - pickup_xyz[:2]))
    achieved_lift = float(lift_xyz[2] - pickup_xyz[2])
    if abs(achieved_lift - lift_distance_m) > 0.015 or lateral_error > 0.015:
        raise ValueError(
            "IK did not produce the requested vertical lift: "
            f"requested={lift_distance_m:.3f}m achieved={achieved_lift:.3f}m "
            f"lateral={lateral_error:.3f}m"
        )
    return lift_pose, pickup_xyz.tolist(), lift_xyz.tolist()


def parse_outcome_annotation(value: str) -> dict[str, str | int | float]:
    value = value.strip()
    if not value:
        return {
            "outcome": "unlabeled",
            "tilt_grade": "",
            "crease_grade": "",
            "first_slip_s_from_lift": "",
            "notes": "",
        }
    parts = [part.strip() for part in value.split(",", maxsplit=4)]
    parts.extend([""] * (5 - len(parts)))
    aliases = {"s": "success", "p": "slip", "d": "drop", "a": "partial"}
    outcome = aliases.get(parts[0].lower(), parts[0].lower())
    if outcome not in {"success", "slip", "drop", "partial"}:
        raise ValueError("outcome must be success, slip, drop, or partial")
    grades: list[int | str] = []
    for name, raw in (("tilt", parts[1]), ("crease", parts[2])):
        if raw == "":
            grades.append("")
            continue
        grade = int(raw)
        if grade not in (0, 1, 2):
            raise ValueError(f"{name} grade must be 0, 1, or 2")
        grades.append(grade)
    slip_time: float | str = ""
    if parts[3] != "":
        slip_time = float(parts[3])
        if not math.isfinite(slip_time) or slip_time < 0.0:
            raise ValueError("first slip time must be a non-negative finite number")
    return {
        "outcome": outcome,
        "tilt_grade": grades[0],
        "crease_grade": grades[1],
        "first_slip_s_from_lift": slip_time,
        "notes": parts[4],
    }


def parse_static_annotation(value: str) -> dict[str, str | int]:
    value = value.strip()
    if not value:
        return {
            "residual_deformation_grade": "",
            "functional_damage": "",
            "notes": "",
        }
    parts = [part.strip() for part in value.split(",", maxsplit=2)]
    parts.extend([""] * (3 - len(parts)))
    grade = int(parts[0])
    if grade not in (0, 1, 2):
        raise ValueError("residual deformation grade must be 0, 1, or 2")
    damage = {"n": "no", "no": "no", "y": "yes", "yes": "yes"}.get(parts[1].lower())
    if damage is None:
        raise ValueError("functional damage must be yes or no")
    return {
        "residual_deformation_grade": grade,
        "functional_damage": damage,
        "notes": parts[2],
    }


def summarize_results(
    plan: list[dict[str, int | float]],
    outcomes: list[dict[str, object]],
    telemetry_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_trial = {int(row["trial_index"]): row for row in outcomes}
    per_target: dict[str, dict[str, object]] = {}
    for target in sorted({float(row["target"]) for row in plan}, reverse=True):
        trials = [row for row in plan if float(row["target"]) == target]
        labels = [by_trial[int(row["trial_index"])] for row in trials if int(row["trial_index"]) in by_trial]
        labeled = [row for row in labels if row.get("outcome") != "unlabeled"]
        active = [
            row
            for row in telemetry_rows
            if float(row["target"]) == target and row["phase"] in {"settle", "lift", "hold", "lower"}
        ]

        def _mean(field: str) -> float | None:
            values = [float(row[field]) for row in active if row.get(field) not in (None, "")]
            return None if not values else sum(values) / len(values)

        successes = sum(row.get("outcome") == "success" for row in labeled)
        per_target[f"{target:g}"] = {
            "planned_trials": len(trials),
            "labeled_trials": len(labeled),
            "successes": successes,
            "success_rate": None if not labeled else successes / len(labeled),
            "mean_present_current": _mean("present_current"),
            "mean_present_load": _mean("present_load"),
            "mean_absolute_position_lag": _mean("absolute_position_lag"),
        }
    rates = [per_target[f"{target:g}"]["success_rate"] for target in sorted(map(float, per_target), reverse=True)]
    all_rates_available = all(rate is not None for rate in rates)
    empirical_monotonic = None
    if all_rates_available:
        empirical_monotonic = all(float(left) <= float(right) for left, right in zip(rates, rates[1:]))
    return {
        "completed_trials": len(outcomes),
        "planned_trials": len(plan),
        "per_target": per_target,
        "success_rate_non_decreasing_with_closure": empirical_monotonic,
    }


def summarize_static_results(
    plan: list[dict[str, int | float]],
    outcomes: list[dict[str, object]],
    telemetry_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_trial = {int(row["trial_index"]): row for row in outcomes}
    per_target: dict[str, dict[str, object]] = {}
    ordered_targets = sorted({float(row["target"]) for row in plan}, reverse=True)
    for target in ordered_targets:
        trials = [row for row in plan if float(row["target"]) == target]
        labels = [by_trial[int(row["trial_index"])] for row in trials if int(row["trial_index"]) in by_trial]
        labeled = [row for row in labels if row.get("residual_deformation_grade") not in (None, "")]
        active = [
            row
            for row in telemetry_rows
            if float(row["target"]) == target and row["phase"] == "hold"
        ]

        def _mean(field: str) -> float | None:
            values = [float(row[field]) for row in active if row.get(field) not in (None, "")]
            return None if not values else sum(values) / len(values)

        grades = [int(row["residual_deformation_grade"]) for row in labeled]
        per_target[f"{target:g}"] = {
            "planned_trials": len(trials),
            "labeled_trials": len(labeled),
            "mean_residual_deformation_grade": None if not grades else sum(grades) / len(grades),
            "functional_damage_count": sum(row.get("functional_damage") == "yes" for row in labeled),
            "mean_present_current": _mean("present_current"),
            "mean_present_load": _mean("present_load"),
            "mean_absolute_position_lag": _mean("absolute_position_lag"),
        }
    grades = [per_target[f"{target:g}"]["mean_residual_deformation_grade"] for target in ordered_targets]
    monotonic = None
    if all(grade is not None for grade in grades):
        monotonic = all(float(left) <= float(right) for left, right in zip(grades, grades[1:]))
    return {
        "mode": "static_closure",
        "completed_trials": len(outcomes),
        "planned_trials": len(plan),
        "per_target": per_target,
        "residual_deformation_non_decreasing_with_closure": monotonic,
    }


def build_trial_segments(
    *,
    static_closure: bool,
    pickup_open: dict[str, float],
    pickup_closed: dict[str, float],
    lift_closed: dict[str, float],
    close_s: float,
    settle_s: float,
    lift_s: float,
    hold_s: float,
    lower_s: float,
    release_s: float,
    recovery_s: float,
) -> tuple[tuple[str, dict[str, float], dict[str, float], float], ...]:
    if static_closure:
        return (
            ("close", pickup_open, pickup_closed, close_s),
            ("hold", pickup_closed, pickup_closed, hold_s),
            ("release", pickup_closed, pickup_open, release_s),
            ("recovery", pickup_open, pickup_open, recovery_s),
        )
    return (
        ("close", pickup_open, pickup_closed, close_s),
        ("settle", pickup_closed, pickup_closed, settle_s),
        ("lift", pickup_closed, lift_closed, lift_s),
        ("hold", lift_closed, lift_closed, hold_s),
        ("lower", lift_closed, pickup_closed, lower_s),
        ("release", pickup_closed, pickup_open, release_s),
    )


def _connect_robot(port: str, arm_id: str) -> SOFollower:
    robot = SOFollower(
        SO101FollowerConfig(
            port=port,
            id=arm_id,
            use_degrees=True,
            cameras={},
            disable_torque_on_disconnect=True,
        )
    )
    robot.connect(calibrate=False)
    return robot


def _read_positions(robot: SOFollower, tries: int = 12) -> dict[str, float]:
    for _ in range(tries):
        try:
            observation = robot.get_observation()
            return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.1)
    raise ConnectionError("arm position read kept failing; check the USB cable and port")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_event(
    writer: csv.DictWriter,
    session_start: float,
    *,
    trial_index: int | str = "",
    block: int | str = "",
    target: float | str = "",
    phase: str,
) -> None:
    writer.writerow(
        {
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "session_elapsed_s": round(time.perf_counter() - session_start, 6),
            "trial_index": trial_index,
            "block": block,
            "target": target,
            "phase": phase,
        }
    )


def _append_telemetry(
    rows: list[dict[str, object]],
    writer: csv.DictWriter,
    *,
    session_start: float,
    trial: dict[str, int | float],
    phase: str,
    command: dict[str, float],
    sample,
) -> None:
    lag = float(sample.observed_gripper_pos) - float(command["gripper.pos"])
    row: dict[str, object] = {
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "session_elapsed_s": round(time.perf_counter() - session_start, 6),
        "trial_index": int(trial["trial_index"]),
        "block": int(trial["block"]),
        "target": float(trial["target"]),
        "phase": phase,
        "commanded_gripper_pos": float(command["gripper.pos"]),
        "observed_gripper_pos": float(sample.observed_gripper_pos),
        "position_lag": lag,
        "absolute_position_lag": abs(lag),
        "present_current": sample.present_current,
        "present_load": sample.present_load,
        "present_temperature": sample.present_temperature,
    }
    rows.append(row)
    writer.writerow(row)


def _execute_segment(
    robot: SOFollower,
    *,
    start_pose: dict[str, float],
    end_pose: dict[str, float],
    duration_s: float,
    control_hz: float,
    telemetry: GripperTelemetrySampler | None = None,
    telemetry_rows: list[dict[str, object]] | None = None,
    telemetry_writer: csv.DictWriter | None = None,
    session_start: float | None = None,
    trial: dict[str, int | float] | None = None,
    phase: str = "move",
) -> None:
    steps = max(1, int(round(duration_s * control_hz)))
    period_s = 1.0 / control_hz
    last_sample_at = None if telemetry is None or telemetry.latest is None else telemetry.latest.observed_at_s
    for step in range(steps):
        tick_started = time.perf_counter()
        command = interpolate_pose(start_pose, end_pose, (step + 1) / steps)
        sent = {key: float(value) for key, value in robot.send_action(command).items()}
        if telemetry is not None:
            sample = telemetry.poll(robot)
            if sample is not None and sample.observed_at_s != last_sample_at:
                if telemetry_rows is None or telemetry_writer is None or session_start is None or trial is None:
                    raise ValueError("telemetry output arguments are required together")
                _append_telemetry(
                    telemetry_rows,
                    telemetry_writer,
                    session_start=session_start,
                    trial=trial,
                    phase=phase,
                    command=sent,
                    sample=sample,
                )
                last_sample_at = sample.observed_at_s
        remaining = period_s - (time.perf_counter() - tick_started)
        if remaining > 0:
            time.sleep(remaining)


def _capture_current_pose(args: argparse.Namespace) -> int:
    robot: SOFollower | None = None
    try:
        robot = _connect_robot(args.port, args.arm_id)
        motors = list(robot.bus.motors.keys())
        pickup_pose = _read_positions(robot)
        pickup_pose["gripper.pos"] = float(args.open_pos)
        kinematics = RobotKinematics(
            urdf_path=args.urdf or str(urdf_path()),
            target_frame_name="gripper_frame_link",
            joint_names=motors,
        )
        lift_pose, pickup_xyz, lift_xyz = compute_lift_pose(
            kinematics,
            motors,
            pickup_pose,
            args.lift_distance_m,
        )
        payload = {
            "schema_version": POSE_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm_id": args.arm_id,
            "lift_distance_m": args.lift_distance_m,
            "pickup_pose": pickup_pose,
            "lift_pose": lift_pose,
            "pickup_ee_xyz_m": pickup_xyz,
            "lift_ee_xyz_m": lift_xyz,
        }
        _write_json(args.pose_file, payload)
        print(f"captured fixed pickup pose; no motion command was sent: {args.pose_file}")
        print(f"pickup EE xyz: {np.round(pickup_xyz, 4)}")
        print(f"lift EE xyz:   {np.round(lift_xyz, 4)}")
        return 0
    finally:
        if robot is not None:
            robot.disconnect()


def _load_pose(path: Path, motors: list[str]) -> tuple[dict[str, float], dict[str, float], dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != POSE_SCHEMA_VERSION:
        raise ValueError(f"unsupported pose schema in {path}")
    expected = {f"{motor}.pos" for motor in motors}
    pickup = {key: float(value) for key, value in payload["pickup_pose"].items()}
    lift = {key: float(value) for key, value in payload["lift_pose"].items()}
    if pickup.keys() != expected or lift.keys() != expected:
        raise ValueError(f"pose motor keys do not match connected arm: {path}")
    return pickup, lift, payload


def _prompt_annotation() -> dict[str, str | int | float]:
    print("Label format: outcome,tilt,crease,first_slip_s,notes")
    print("outcome=s/slip/drop/partial; tilt and crease are 0/1/2; Enter leaves it for video review")
    print("success requires the entire carton base to remain visibly clear of the table; one-edge lift is partial")
    while True:
        try:
            return parse_outcome_annotation(input("Trial result: "))
        except (TypeError, ValueError) as exc:
            print(f"invalid label: {exc}")


def _prompt_static_annotation() -> dict[str, str | int]:
    print("Label format: residual_deformation_grade,functional_damage,notes")
    print("grade: 0=none, 1=mild/function unaffected, 2=geometry or function affected; damage=yes/no")
    while True:
        try:
            return parse_static_annotation(input("Static result: "))
        except (TypeError, ValueError) as exc:
            print(f"invalid label: {exc}")


def _run_trials(args: argparse.Namespace) -> int:
    if not args.pose_file.is_file():
        raise FileNotFoundError(
            f"fixed pickup pose not found: {args.pose_file}; position the empty gripper once, then run "
            "the launcher with --capture-current-pose"
        )
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    plan = build_trial_plan(
        args.targets,
        args.repeats,
        args.seed,
        randomize=not args.static_closure,
    )
    mode = "static_closure" if args.static_closure else "lift"
    _write_json(
        args.evidence_dir / "plan.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "seed": args.seed,
            "randomization": (
                "none; each block follows target order"
                if args.static_closure
                else "blocked; each block contains every target once"
            ),
            "mass_g": args.mass_g,
            "surface_id": args.surface_id,
            "grasp_mark": args.grasp_mark,
            "open_pos": args.open_pos,
            "close_s": args.close_s,
            "settle_s": args.settle_s,
            "lift_s": args.lift_s,
            "hold_s": args.hold_s,
            "lower_s": args.lower_s,
            "release_s": args.release_s,
            "recovery_s": args.recovery_s,
            "control_hz": args.control_hz,
            "telemetry_hz": args.telemetry_hz,
            "continuous": args.continuous,
            "targets": list(args.targets),
            "repeats": args.repeats,
            "trials": plan,
        },
    )

    robot: SOFollower | None = None
    pickup_open: dict[str, float] | None = None
    outcome_rows: list[dict[str, object]] = []
    telemetry_rows: list[dict[str, object]] = []
    session_start = time.perf_counter()
    event_fields = ("wall_time_utc", "session_elapsed_s", "trial_index", "block", "target", "phase")
    telemetry_fields = (
        "wall_time_utc",
        "session_elapsed_s",
        "trial_index",
        "block",
        "target",
        "phase",
        "commanded_gripper_pos",
        "observed_gripper_pos",
        "position_lag",
        "absolute_position_lag",
        "present_current",
        "present_load",
        "present_temperature",
    )
    events_path = args.evidence_dir / "events.csv"
    telemetry_path = args.evidence_dir / "telemetry.csv"
    outcomes_path = args.evidence_dir / (
        "static_outcomes.csv" if args.static_closure else "trial_outcomes.csv"
    )
    with (
        events_path.open("w", newline="") as events_handle,
        telemetry_path.open("w", newline="") as telemetry_handle,
        outcomes_path.open("w", newline="") as outcomes_handle,
    ):
        event_writer = csv.DictWriter(events_handle, fieldnames=event_fields)
        telemetry_writer = csv.DictWriter(telemetry_handle, fieldnames=telemetry_fields)
        outcome_writer = csv.DictWriter(
            outcomes_handle,
            fieldnames=STATIC_OUTCOME_FIELDS if args.static_closure else OUTCOME_FIELDS,
        )
        event_writer.writeheader()
        telemetry_writer.writeheader()
        outcome_writer.writeheader()
        try:
            robot = _connect_robot(args.port, args.arm_id)
            motors = list(robot.bus.motors.keys())
            pickup_open, lift_open, pose_payload = _load_pose(args.pose_file, motors)
            pickup_open["gripper.pos"] = float(args.open_pos)
            lift_open["gripper.pos"] = float(args.open_pos)
            _write_json(args.evidence_dir / "fixed_pose_used.json", pose_payload)
            protocol = "ordered static closures" if args.static_closure else "blinded lift trials"
            print(f"Prepared {len(plan)} {protocol}: {args.repeats} blocks x {len(args.targets)} targets")
            print("The robot will first move to the saved pickup pose with the gripper open.")
            print("Keep the workspace clear; Ctrl+C stops the session.")
            if input("Type START to begin robot motion: ").strip() != "START":
                raise SystemExit("aborted before motion")
            _write_event(event_writer, session_start, phase="session_start")
            _execute_segment(
                robot,
                start_pose=_read_positions(robot),
                end_pose=pickup_open,
                duration_s=args.initial_move_s,
                control_hz=args.control_hz,
            )

            for trial in plan:
                target = float(trial["target"])
                pickup_closed = dict(pickup_open)
                pickup_closed["gripper.pos"] = target
                lift_closed = dict(lift_open)
                lift_closed["gripper.pos"] = target
                print(
                    f"\nTrial {int(trial['trial_index'])}/{len(plan)}, block {int(trial['block'])}: "
                    f"place the {args.mass_g} g carton on the fixed mark and keep hands clear."
                )
                if not args.continuous:
                    input("Press Enter when the carton is placed: ")
                _write_event(event_writer, session_start, **trial, phase="placed")
                telemetry = GripperTelemetrySampler(interval_s=1.0 / args.telemetry_hz)

                segments = build_trial_segments(
                    static_closure=args.static_closure,
                    pickup_open=pickup_open,
                    pickup_closed=pickup_closed,
                    lift_closed=lift_closed,
                    close_s=args.close_s,
                    settle_s=args.settle_s,
                    lift_s=args.lift_s,
                    hold_s=args.hold_s,
                    lower_s=args.lower_s,
                    release_s=args.release_s,
                    recovery_s=args.recovery_s,
                )
                for phase, start_pose, end_pose, duration_s in segments:
                    _write_event(event_writer, session_start, **trial, phase=f"{phase}_start")
                    _execute_segment(
                        robot,
                        start_pose=start_pose,
                        end_pose=end_pose,
                        duration_s=duration_s,
                        control_hz=args.control_hz,
                        telemetry=telemetry,
                        telemetry_rows=telemetry_rows,
                        telemetry_writer=telemetry_writer,
                        session_start=session_start,
                        trial=trial,
                        phase=phase,
                    )
                    _write_event(event_writer, session_start, **trial, phase=f"{phase}_end")
                    events_handle.flush()
                    telemetry_handle.flush()

                if args.static_closure:
                    annotation = (
                        parse_static_annotation("")
                        if args.continuous
                        else _prompt_static_annotation()
                    )
                else:
                    annotation = parse_outcome_annotation("") if args.continuous else _prompt_annotation()
                outcome_row = {**trial, **annotation}
                outcome_rows.append(outcome_row)
                outcome_writer.writerow(outcome_row)
                outcomes_handle.flush()

            _write_event(event_writer, session_start, phase="session_complete")
        finally:
            if robot is not None:
                pending = sys.exc_info()[1]
                if pickup_open is not None:
                    try:
                        _execute_segment(
                            robot,
                            start_pose=_read_positions(robot),
                            end_pose=pickup_open,
                            duration_s=1.5,
                            control_hz=args.control_hz,
                        )
                    except BaseException as exc:
                        print(f"warning: could not return to open pickup pose during cleanup: {exc}")
                try:
                    robot.disconnect()
                except BaseException:
                    if pending is None:
                        raise

    summary = (
        summarize_static_results(plan, outcome_rows, telemetry_rows)
        if args.static_closure
        else summarize_results(plan, outcome_rows, telemetry_rows)
    )
    _write_json(args.evidence_dir / "summary.json", summary)
    print(f"session evidence: {args.evidence_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=ARM_PORT)
    parser.add_argument("--arm-id", default=ARM_ID)
    # Resolved lazily: urdf_path() raises when SO-ARM100 is not configured, and
    # that belongs at run time, not at import.
    parser.add_argument("--urdf", default=None,
                        help="SO-101 URDF (default: $SO101_URDF / $SO_ARM100_DIR)")
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--capture-current-pose", action="store_true")
    parser.add_argument("--static-closure", action="store_true", help="close, hold, release, and recover without lifting")
    parser.add_argument("--targets", type=parse_targets)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--mass-g", type=int, default=250)
    parser.add_argument("--surface-id", default="marked_carton_face_v1")
    parser.add_argument("--grasp-mark", default="carton_grasp_mark_v1")
    parser.add_argument("--open-pos", type=float, default=100.0)
    parser.add_argument("--lift-distance-m", type=float, default=0.10)
    parser.add_argument("--initial-move-s", type=float, default=3.0)
    parser.add_argument("--close-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--lift-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--lower-s", type=float, default=2.0)
    parser.add_argument("--release-s", type=float, default=1.0)
    parser.add_argument("--recovery-s", type=float, default=10.0)
    parser.add_argument("--control-hz", type=float, default=25.0)
    parser.add_argument("--telemetry-hz", type=float, default=5.0)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="run trials back-to-back without placement or annotation prompts; review video afterward",
    )
    return parser


def resolve_protocol_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.targets is None:
        args.targets = STATIC_CLOSURE_TARGETS if args.static_closure else DEFAULT_TARGETS
    if args.repeats is None:
        args.repeats = 1 if args.static_closure else 5
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_protocol_defaults(build_parser().parse_args(argv))
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0.0 <= args.open_pos <= 100.0:
        raise ValueError("open-pos must be in [0, 100]")
    for name in (
        "lift_distance_m",
        "initial_move_s",
        "close_s",
        "settle_s",
        "lift_s",
        "hold_s",
        "lower_s",
        "release_s",
        "recovery_s",
        "control_hz",
        "telemetry_hz",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.capture_current_pose:
        return _capture_current_pose(args)
    if args.evidence_dir is None:
        raise ValueError("--evidence-dir is required for trials")
    return _run_trials(args)


if __name__ == "__main__":
    raise SystemExit(main())
