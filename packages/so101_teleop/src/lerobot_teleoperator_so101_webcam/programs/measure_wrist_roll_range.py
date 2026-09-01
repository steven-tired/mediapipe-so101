"""Find how far `wrist_roll` can actually turn before its cable binds.

`tune_servo_pid.py:69` records that wrist_roll "has a one-sided cable-wrap
limit: rotating it past a point binds (draws stall current, sags the rail)",
and pins the joint rather than risk it. The URDF advertises 320 degrees of
travel. Nobody has measured what is usable, and unlocking roll for
orientation-commanded teleop needs that number -- it is the one precondition
that can damage hardware.

Method: cap the servo's torque first so a bind cannot pull hard, then step
outward from the startup angle a few degrees at a time, reading position, load,
current and temperature after each step. Binding shows up as the joint no
longer following the command -- position error growing while current rises --
which is detected and stops the sweep before the stall becomes sustained. Each
direction is swept separately and returns to the startup angle afterwards.

The measured limits are reported relative to the STARTUP angle, because that is
what the servo's own zero is referenced to and what a caller can re-derive.

SAFETY
  - Clear the space around the wrist. The gripper will rotate.
  - Keep the e-stop within reach.
  - Torque is capped for the duration and restored on exit, including on Ctrl-C.
  - Nothing else should be driving the arm.

Run:  python -m lerobot_teleoperator_so101_webcam.programs.measure_wrist_roll_range \\
          --out /tmp/wrist_roll_range_01.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

JOINT = "wrist_roll"
# URDF so101_new_calib: wrist_roll is -157.2 .. +162.8 deg. Never command past
# it whatever the bind detector says.
URDF_MIN_DEG = -157.2
URDF_MAX_DEG = 162.8

# Small enough that a bind is caught within a few degrees of travel.
DEFAULT_STEP_DEG = 3.0
# Cap on Torque_Limit (RAM reg 48, 0..1000) while sweeping. Enough to turn a
# free joint, not enough to fight a bound cable. servo_current.py documents the
# register.
DEFAULT_TORQUE_CAP = 300
# The joint is following if it gets within this of the command once settled.
FOLLOW_TOLERANCE_DEG = 2.0
# Consecutive lagging steps before calling it bound. One step can lag from
# ordinary settling; three in a row is the cable.
BINDING_STEPS = 3
SETTLE_S = 0.35
# This bus drops packets. teleop_viz_ee.py sets MAX_RELATIVE_TARGET = None
# specifically to avoid a second per-frame read on it, and a sweep that dies on
# the first "There is no status packet!" loses the whole run for a reason that
# has nothing to do with the joint. Retry the transfer, then retry the step.
DEFAULT_COMMS_RETRY = 3
STEP_ATTEMPTS = 3
STEP_RETRY_PAUSE_S = 0.4
# Returning to the start can be most of the joint's travel. A fixed wait is
# wrong: the first run swept 137 deg positive, then began the negative sweep
# while the joint was still in transit at 65 deg, and three "lagging" transit
# samples were read as a cable bind. Wait for arrival instead.
ARRIVAL_TOLERANCE_DEG = 1.5
ARRIVAL_POLL_S = 0.2
ARRIVAL_TIMEOUT_S = 20.0


def binding_verdict(samples, *, tolerance_deg=FOLLOW_TOLERANCE_DEG,
                    consecutive=BINDING_STEPS) -> dict:
    """Where did the joint stop following, given the per-step readings?

    `samples` are dicts with `commanded_deg` and `measured_deg`, in the order
    they were taken. Returns the last angle the joint actually reached while
    still following, and whether a bind was seen at all.
    """
    lagging = 0
    last_following = None
    for index, sample in enumerate(samples):
        error = abs(sample["commanded_deg"] - sample["measured_deg"])
        if error <= tolerance_deg:
            lagging = 0
            last_following = sample["measured_deg"]
            continue
        lagging += 1
        if lagging >= consecutive:
            return {
                "bound": True,
                "last_following_deg": last_following,
                "bound_at_index": index,
                "bound_at_commanded_deg": sample["commanded_deg"],
            }
    return {
        "bound": False,
        "last_following_deg": last_following,
        "bound_at_index": None,
        "bound_at_commanded_deg": None,
    }


def summarise(start_deg, positive, negative) -> dict:
    """Usable travel either side of the startup angle, and the total."""
    report = {"start_deg": start_deg, "positive": positive, "negative": negative}
    reach = {}
    for name, verdict in (("positive", positive), ("negative", negative)):
        last = verdict.get("last_following_deg")
        reach[name] = None if last is None else abs(last - start_deg)
    report["travel_deg"] = reach
    if reach["positive"] is not None and reach["negative"] is not None:
        report["total_travel_deg"] = reach["positive"] + reach["negative"]
        # tune_servo_pid.py calls the limit one-sided. Comparing magnitudes gets
        # this wrong: the first clean run gave 137 vs 104 deg, which is not a
        # 2x gap, yet only ONE side ended in a cable bind -- the other simply
        # ran out of URDF travel. The asymmetry that matters is the cause.
        bound_sides = [name for name, verdict in
                       (("positive", positive), ("negative", negative))
                       if verdict.get("bound")]
        report["cable_bound_sides"] = bound_sides
        report["one_sided"] = len(bound_sides) == 1
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-deg", type=float, default=DEFAULT_STEP_DEG)
    parser.add_argument("--torque-cap", type=int, default=DEFAULT_TORQUE_CAP)
    parser.add_argument(
        "--comms-retry",
        type=int,
        default=DEFAULT_COMMS_RETRY,
        help="per-transfer retries on the servo bus, which drops packets",
    )
    parser.add_argument(
        "--max-travel-deg",
        type=float,
        default=170.0,
        help="stop each direction here even if nothing binds; the URDF limits "
        "are enforced on top of this and cannot be exceeded",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt; the wrist WILL rotate",
    )
    return parser.parse_args(argv)


def _sweep(bus, start_deg, direction, args, log, record=None):
    """Step one direction until the joint stops following, then come back.

    Returns whatever was gathered even if the bus gives up, so a dropout costs
    the remainder of the sweep rather than the measurement.
    """
    samples = []
    comms_failed = None
    angle = start_deg
    while abs(angle + direction * args.step_deg - start_deg) <= args.max_travel_deg:
        target = angle + direction * args.step_deg
        if not (URDF_MIN_DEG <= target <= URDF_MAX_DEG):
            break
        try:
            sample = _step(bus, target, args.comms_retry)
        except ConnectionError as exc:
            comms_failed = str(exc)
            log(f"  bus gave up: {exc}")
            break
        measured = sample["measured_deg"]
        samples.append(sample)
        if record is not None:
            record(dict(sample, direction=direction))
        log(f"  cmd {target:7.1f}  meas {measured:7.1f}  err {abs(target-measured):5.2f}"
            f"  load {sample['load']}  cur {sample['current']}  temp {sample['temperature']}")
        verdict = binding_verdict(samples)
        if verdict["bound"]:
            log(f"  bound: stopped following at {target:.1f} deg")
            break
        angle = target

    log("  returning to start")
    if not _return_to_start(bus, start_deg, args, log):
        # Sweeping the other direction from the wrong place measures transit,
        # not the joint, so refuse rather than produce a confident wrong number.
        return binding_verdict(samples) | {
            "samples": samples,
            "comms_failed": comms_failed,
            "return_failed": True,
        }
    return binding_verdict(samples) | {
        "samples": samples,
        "comms_failed": comms_failed,
        "return_failed": False,
    }


def _return_to_start(bus, start_deg, args, log) -> bool:
    """Drive back to the startup angle and wait until it actually arrives."""
    try:
        bus.write("Goal_Position", JOINT, start_deg, num_retry=args.comms_retry)
    except ConnectionError as exc:
        log(f"  WARNING: could not command the return ({exc}); move it back by hand")
        return False
    deadline = time.time() + ARRIVAL_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(ARRIVAL_POLL_S)
        try:
            measured = float(bus.read("Present_Position", JOINT, num_retry=args.comms_retry))
        except ConnectionError:
            continue
        if abs(measured - start_deg) <= ARRIVAL_TOLERANCE_DEG:
            return True
    log(f"  WARNING: did not reach the start angle within {ARRIVAL_TIMEOUT_S:.0f}s")
    return False


def _safe_read(bus, register, num_retry=DEFAULT_COMMS_RETRY):
    try:
        return int(bus.read(register, JOINT, normalize=False, num_retry=num_retry))
    except Exception:
        return None


def _step(bus, target, num_retry):
    """Command one step and read it back, tolerating a dropped packet.

    Raises ConnectionError only after STEP_ATTEMPTS whole-step attempts, so a
    single lost packet costs a retry rather than the run.
    """
    last = None
    for attempt in range(STEP_ATTEMPTS):
        try:
            bus.write("Goal_Position", JOINT, target, num_retry=num_retry)
            time.sleep(SETTLE_S)
            measured = float(bus.read("Present_Position", JOINT, num_retry=num_retry))
            return {
                "commanded_deg": float(target),
                "measured_deg": measured,
                "load": _safe_read(bus, "Present_Load", num_retry),
                "current": _safe_read(bus, "Present_Current", num_retry),
                "temperature": _safe_read(bus, "Present_Temperature", num_retry),
                "step_attempts": attempt + 1,
            }
        except ConnectionError as exc:
            last = exc
            time.sleep(STEP_RETRY_PAUSE_S)
    raise ConnectionError(f"step to {target:.1f} deg failed {STEP_ATTEMPTS}x: {last}")


def run(args) -> dict:
    from servo_current import _connect

    robot = _connect()
    bus = robot.bus
    original_cap = None
    # Append each step as it is taken. The bus can give up mid-sweep, and the
    # steps already taken are the measurement.
    trace_path = args.out.with_suffix(".samples.jsonl")
    trace = trace_path.open("w", encoding="utf-8")

    def log(message):
        print(message, flush=True)

    def record(sample):
        trace.write(json.dumps(sample, sort_keys=True) + "\n")
        trace.flush()

    try:
        original_cap = int(bus.read("Torque_Limit", JOINT, normalize=False))
        bus.write("Torque_Limit", JOINT, args.torque_cap, normalize=False)
        log(f"Torque_Limit {JOINT}: {original_cap} -> {args.torque_cap} (restored on exit)")

        start_deg = float(bus.read("Present_Position", JOINT))
        log(f"startup angle {start_deg:.1f} deg; stepping {args.step_deg} deg at a time")

        log("sweeping positive")
        positive = _sweep(bus, start_deg, +1.0, args, log, record)
        log("sweeping negative")
        negative = _sweep(bus, start_deg, -1.0, args, log, record)
    finally:
        if original_cap is not None:
            try:
                bus.write("Torque_Limit", JOINT, original_cap, normalize=False)
                log(f"Torque_Limit {JOINT} restored to {original_cap}")
            except Exception as exc:
                log(f"WARNING: could not restore Torque_Limit: {exc}")
        trace.close()
        try:
            robot.disconnect()
        except Exception as exc:
            log(f"WARNING: disconnect failed: {exc}")

    report = summarise(start_deg, positive, negative) | {
        "experiment_identity": "so101_wrist_roll_range_v1",
        "role": "diagnostic_not_preregistered",
        "urdf_limits_deg": [URDF_MIN_DEG, URDF_MAX_DEG],
        "torque_cap": args.torque_cap,
        "step_deg": args.step_deg,
        "robot_or_controller_actuation": True,
        "samples_path": str(trace_path),
        # A direction cut short by the bus has not found the joint's limit, so
        # its travel is a LOWER BOUND, not the answer.
        "complete": not any(
            v.get("comms_failed") or v.get("return_failed")
            for v in (positive, negative)
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.yes:
        print(__doc__.split("SAFETY")[1].split("Run:")[0])
        if input("Clear around the wrist? Type yes to rotate it: ").strip() != "yes":
            print("aborted")
            return 1
    report = run(args)
    travel = report["travel_deg"]
    print(json.dumps({k: v for k, v in report.items() if k != "positive" and k != "negative"},
                     indent=2, sort_keys=True))
    print(f"\nusable travel: +{travel['positive']} / -{travel['negative']} deg "
          f"about the startup angle")
    if not report["complete"]:
        print("INCOMPLETE: the bus gave up mid-sweep, so a direction that did not "
              "bind is a lower bound on its travel, not its limit. Re-run it.")
    if report.get("one_sided"):
        print("one-sided, as tune_servo_pid.py describes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
