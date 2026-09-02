#!/usr/bin/env python
"""Calibrate the SO-101 gripper deadband on a held carton (gate 1 of 2026-09-01).

Why this exists: the 2026-08-31 A/B collection intervened with a `0.2` degree
step and recorded two `loosen_stable` outcomes. Both were null. In trial03 the
readback held at exactly `26.426` for all 162 remaining control steps while the
standing command-to-readback offset was `0.787` -- the step was about a quarter
of the offset and never moved the jaw at all. Every grip-head result that rests
on those slots is therefore measuring nothing.

So before any further loosening or tightening experiment, this tool answers two
questions on the actual hardware:

  1. What is the smallest commanded step that produces a readback change the
     encoder can resolve? That number sizes every later ramp and the `delta_q`
     a residual head would emit.
  2. Do `Present_Load` and `Present_Current` respond to grip depth at all?
     `Present_Load` is quantized to multiples of four and was a per-trial
     constant while holding, so a head trained on it may be fitting trial
     identity. If both channels are flat across a deliberate depth staircase,
     there is no effort sensor here and the tight bound has to stay a fixed
     floor.

The body joints are held at their own readback for the whole run; only the
gripper is commanded. Nothing is learned and nothing is deployed.

Procedure: close the jaw on the carton first (by hand, or leave it held after a
teleop or ACT run), then start this. It holds the starting command to measure
the noise floor, walks a staircase of each swept step size, and returns to the
starting command between sweeps.

Usage:
  ./scripts/run_gripper_deadband.sh --arm-enabled
  ./scripts/run_gripper_deadband.sh --arm-enabled --direction both --steps 0.5,1,2,3,5

SAFETY: `--arm-enabled` drives the gripper against a held object. Without it
this runs as a locked rehearsal that reads telemetry and sends nothing.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from ..gripper_hardware import (
    DeadbandStep,
    GRIPPER,
    breakout_offset,
    rank_correlation,
    read_gripper_telemetry,
    readback_spread,
    serialize_telemetry_snapshot,
    slow_close_waypoints,
    smallest_resolvable_step,
    summarize_target_current,
    tracking_ratio,
)
from ..paths import evidence_dir

ARM_PORT = os.environ.get(
    "SO101_ARM_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
)
ARM_ID = os.environ.get("SO101_ARM_ID", "so101_follower_1")

#: Higher gripper positions are more open, so tightening is a negative step.
DIRECTION_SIGN = {"tighten": -1.0, "loosen": +1.0}


def _connect(port: str):
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(SO101FollowerConfig(port=port, id=ARM_ID, use_degrees=True,
                                           cameras={}, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    return robot


def body_hold(robot) -> dict[str, float]:
    """Freeze the body at one reading taken before the sweep starts.

    Re-reading the body every cycle and commanding it back would integrate the
    standing command-to-readback offset into a slow drift -- the same offset
    this tool exists to measure on the gripper -- and it doubles the traffic on
    a bus that already drops status packets.
    """
    observation = robot.get_observation()
    return {
        f"{motor}.pos": float(observation[f"{motor}.pos"])
        for motor in robot.bus.motors
        if motor != GRIPPER
    }


def dwell(robot, *, body: dict[str, float], gripper_pos: float, seconds: float, hz: float,
          arm_enabled: bool, started_at: float, max_current: float, max_temperature: float,
          samples: list) -> list:
    """Hold one aperture, sampling telemetry, and stop early on effort or heat."""
    period = 1.0 / hz
    collected = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        cycle = time.perf_counter()
        if arm_enabled:
            robot.send_action({**body, f"{GRIPPER}.pos": float(gripper_pos)})
        snapshot = read_gripper_telemetry(robot, gripper_pos, cycle - started_at)
        collected.append(snapshot)
        samples.append(snapshot)
        current = snapshot.present_current
        if current is not None and abs(float(current)) >= max_current:
            print(f"  ! Present_Current {current} reached the {max_current:g} cap; stopping the dwell")
            break
        temperature = snapshot.present_temperature
        if temperature is not None and float(temperature) >= max_temperature:
            print(f"  ! Present_Temperature {temperature} reached the {max_temperature:g} C cap; "
                  "stopping the dwell")
            break
        remaining = period - (time.perf_counter() - cycle)
        if remaining > 0:
            time.sleep(remaining)
    if not collected:
        raise RuntimeError("dwell collected no telemetry; check the bus and the sample rate")
    return collected


def _median_pos(snapshots: list) -> float:
    positions = sorted(float(snapshot.gripper_pos) for snapshot in snapshots)
    middle = len(positions) // 2
    if len(positions) % 2:
        return positions[middle]
    return (positions[middle - 1] + positions[middle]) / 2.0


def sweep(robot, *, body: dict[str, float], step_size: float, direction: str,
          start_pos: float, args, started_at: float, samples: list) -> tuple[list[DeadbandStep], list[dict]]:
    """Walk a staircase of one step size and record what the jaw did at each tread."""
    sign = DIRECTION_SIGN[direction]
    # Every ramp covers the same commanded distance, so a small step simply
    # gets more treads. Fixing the tread *count* instead would walk the 0.5
    # ramp a sixth as far as the 3.0 ramp, and it would then report "never
    # broke out" for a step that was only ever given a fifth of the deadband
    # to work with.
    repeats = max(1, round(args.ramp_travel / step_size))
    commanded = start_pos
    previous = dwell(robot, body=body, gripper_pos=commanded, seconds=args.dwell_s, hz=args.hz,
                     arm_enabled=args.arm_enabled, started_at=started_at,
                     max_current=args.max_current, max_temperature=args.max_temperature,
                      samples=samples)
    steps: list[DeadbandStep] = []
    records: list[dict] = []
    for index in range(repeats):
        if abs((commanded + sign * step_size) - start_pos) > args.max_travel:
            print(f"  ! {args.max_travel:g} degree travel cap reached; ending this staircase")
            break
        commanded += sign * step_size
        current = dwell(robot, body=body, gripper_pos=commanded, seconds=args.dwell_s, hz=args.hz,
                        arm_enabled=args.arm_enabled, started_at=started_at,
                        max_current=args.max_current, max_temperature=args.max_temperature,
                      samples=samples)
        before, after = _median_pos(previous), _median_pos(current)
        step = DeadbandStep(
            step_size=step_size,
            commanded_delta=sign * step_size,
            readback_delta=after - before,
            readback_spread=readback_spread(current),
        )
        steps.append(step)
        effort = summarize_target_current(current)
        records.append({
            "direction": direction,
            "step_size": step_size,
            "tread_index": index,
            "commanded_pos": commanded,
            "readback_before": before,
            "readback_after": after,
            "readback_delta": step.readback_delta,
            "readback_spread": step.readback_spread,
            "command_to_readback_offset": after - commanded,
            "mean_load": effort["mean_load"],
            "mean_current": effort["mean_current"],
            "max_temperature": effort["max_temperature"],
        })
        print(f"  {direction} {step_size:>4g}  tread {index}  q_cmd={commanded:7.3f}  "
              f"q_read {before:7.3f} -> {after:7.3f}  d={step.readback_delta:+7.3f}  "
              f"load={effort['mean_load']:7.2f}  current={effort['mean_current']:6.2f}")
        previous = current
    return steps, records


def _effort_response(records: list[dict]) -> dict[str, float | int]:
    """Does either effort channel track commanded grip depth across the staircase?"""
    depth = [-record["commanded_pos"] for record in records]   # deeper = more closed
    loads = [record["mean_load"] for record in records]
    currents = [record["mean_current"] for record in records]
    return {
        "treads": len(records),
        "load_vs_depth_rho": rank_correlation(depth, loads),
        "current_vs_depth_rho": rank_correlation(depth, currents),
        "distinct_loads": len(set(loads)),
        "distinct_currents": len(set(currents)),
    }


def run(args) -> tuple[dict, list]:
    robot = _connect(args.port)
    started_at = time.perf_counter()
    samples: list = []
    try:
        start_pos = float(robot.get_observation()[f"{GRIPPER}.pos"])
        body = body_hold(robot)
        print(f"gripper readback at start: {start_pos:.3f}")

        if args.close_to is None and start_pos > args.held_below:
            # A free jaw has a different deadband from a loaded one -- it is
            # the carton that supplies most of the static friction this gate
            # measures -- so calibrating an open gripper would produce a
            # confident number for the wrong mechanism.
            raise RuntimeError(
                f"gripper is at {start_pos:.2f}, which is open: nothing is held, and the "
                f"deadband of a free jaw is not the one the ramps need. Place the carton in "
                f"the jaw and pass --close-to <position>, or leave a grasp held from a teleop "
                f"or ACT run. --held-below raises the threshold if this grasp really is above "
                f"{args.held_below:g}."
            )

        if not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "the gripper will move and stdin is not a terminal, so the confirmation "
                    "cannot be given. Run this from an interactive shell, or pass --yes."
                )
            input("Confirm the carton is in the jaw, then press Enter to begin (Ctrl-C aborts): ")

        if args.close_to is not None:
            print(f"closing {start_pos:.3f} -> {args.close_to:g} to take the carton")
            for waypoint in slow_close_waypoints(start_pos, args.close_to, args.close_steps):
                dwell(robot, body=body, gripper_pos=waypoint, seconds=args.close_dwell_s,
                      hz=args.hz, arm_enabled=args.arm_enabled, started_at=started_at,
                      max_current=args.max_current, max_temperature=args.max_temperature,
                      samples=samples)
            start_pos = float(robot.get_observation()[f"{GRIPPER}.pos"])
            print(f"gripper readback after closing: {start_pos:.3f}")

        # Settle BEFORE measuring, and throw the settle away. The first run
        # measured across it: the jaw was still arriving from the closing ramp,
        # moved 0.852 in the first half second, then held at exactly 31.2131
        # for the remaining 4.9 s at a peak-to-peak of 0.0000. The floor came
        # out as that one transient, which is also why it equalled the
        # command-to-readback offset exactly. A floor of 0.852 then demanded
        # more than 0.852 of travel per tread, and scored a loosen ramp that
        # tracked its command at 95% as having moved on none of its treads.
        print(f"settling {args.settle_s:g}s at {start_pos:.3f} before measuring")
        settle = dwell(robot, body=body, gripper_pos=start_pos, seconds=args.settle_s, hz=args.hz,
                       arm_enabled=args.arm_enabled, started_at=started_at,
                       max_current=args.max_current, max_temperature=args.max_temperature,
                       samples=samples)
        settle_spread = readback_spread(settle)
        print(f"  settle transient (discarded): {settle_spread:.3f}")

        print(f"holding {args.hold_s:g}s to measure the noise floor")
        hold = dwell(robot, body=body, gripper_pos=start_pos, seconds=args.hold_s, hz=args.hz,
                     arm_enabled=args.arm_enabled, started_at=started_at,
                     max_current=args.max_current, max_temperature=args.max_temperature,
                      samples=samples)
        noise_floor = readback_spread(hold)
        settled = _median_pos(hold)
        offset = settled - start_pos
        hold_effort = summarize_target_current(hold)
        print(f"  noise floor (peak-to-peak readback while held): {noise_floor:.3f}")
        print(f"  distinct readbacks while held: {len({s.gripper_pos for s in hold})}")
        print(f"  command-to-readback offset: {offset:+.3f}")
        print(f"  load mean {hold_effort['mean_load']:.2f}  current mean {hold_effort['mean_current']:.2f}")
        if noise_floor > args.floor_warn:
            # Not fatal: a genuinely restless jaw has a genuinely high floor,
            # and every step it then disqualifies really is unresolvable
            # against it. But it is far more often a jaw that has not finished
            # arriving, and then every result below is conservative by however
            # much of this is transient.
            print(f"  ! the floor is above {args.floor_warn:g} while the command is held. Either the "
                  "jaw is still settling -- raise --settle-s and re-run -- or it is genuinely "
                  "restless, and every step size below is judged against that.")

        directions = ["tighten", "loosen"] if args.direction == "both" else [args.direction]
        sweeps: dict[str, dict] = {}
        for direction in directions:
            print(f"\n{direction} sweep")
            all_steps: list[DeadbandStep] = []
            all_records: list[dict] = []
            ramps: list[dict] = []
            for step_size in args.steps:
                # Each ramp starts from wherever the jaw actually is. Returning
                # to the settled *command* does not return the jaw: the return
                # is itself smaller than the deadband it just walked past, so a
                # ramp that assumed it started at `settled` would attribute the
                # leftover offset to its own first tread.
                ramp_start = float(robot.get_observation()[f"{GRIPPER}.pos"])
                steps, records = sweep(robot, body=body, step_size=step_size,
                                       direction=direction, start_pos=ramp_start, args=args,
                                       started_at=started_at, samples=samples)
                all_steps.extend(steps)
                all_records.extend(records)
                ramps.append({
                    "step_size": step_size,
                    "ramp_start_readback": ramp_start,
                    "treads": len(steps),
                    "treads_that_moved": sum(
                        1 for step in steps
                        if abs(step.readback_delta) > noise_floor
                        and step.readback_delta * step.commanded_delta > 0.0
                    ),
                    "breakout_offset": breakout_offset(steps, noise_floor=noise_floor),
                    "tracking_ratio": tracking_ratio(steps) if steps else None,
                })
                moved = ramps[-1]["treads_that_moved"]
                print(f"  step {step_size:g}: {moved}/{len(steps)} treads moved, "
                      f"breakout={ramps[-1]['breakout_offset']}, "
                      f"tracking={ramps[-1]['tracking_ratio']}")
                print(f"  returning toward {settled:.3f}")
                dwell(robot, body=body, gripper_pos=settled, seconds=args.dwell_s, hz=args.hz,
                      arm_enabled=args.arm_enabled, started_at=started_at,
                      max_current=args.max_current, max_temperature=args.max_temperature,
                      samples=samples)
            resolvable = smallest_resolvable_step(all_steps, noise_floor=noise_floor)
            breakouts = [ramp["breakout_offset"] for ramp in ramps
                         if ramp["breakout_offset"] is not None]
            sweeps[direction] = {
                "smallest_resolvable_step": resolvable,
                "breakout_offsets": breakouts,
                "ramps": ramps,
                "effort_response": _effort_response(all_records) if all_records else None,
                "treads": all_records,
            }
            print(f"  smallest {direction} step that moved the jaw on every tread: "
                  + ("none of the swept sizes" if resolvable is None else f"{resolvable:g}"))
            if breakouts:
                print(f"  breakout deadband across ramps: {min(breakouts):.3f} to {max(breakouts):.3f}")
    finally:
        robot.disconnect()

    return {
        "arm_enabled": args.arm_enabled,
        "port": args.port,
        "swept_steps": args.steps,
        "ramp_travel": args.ramp_travel,
        "dwell_s": args.dwell_s,
        "hold_s": args.hold_s,
        "settle_s": args.settle_s,
        "settle_spread": settle_spread,
        "hz": args.hz,
        "max_travel": args.max_travel,
        "max_current": args.max_current,
        "max_temperature": args.max_temperature,
        "closed_to": args.close_to,
        "start_readback": start_pos,
        "settled_readback": settled,
        "command_to_readback_offset": offset,
        "noise_floor": noise_floor,
        "hold_effort": hold_effort,
        "sweeps": sweeps,
        "samples": len(samples),
    }, samples


def _steps(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("steps must be a comma-separated list of positive degrees")
    return sorted(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=ARM_PORT)
    ap.add_argument("--arm-enabled", action="store_true",
                    help="actually command the gripper; without it nothing is sent")
    ap.add_argument("--direction", choices=("tighten", "loosen", "both"), default="both",
                    help="a loosen ramp entered from a tightened state carries hysteresis, "
                         "so the two directions are swept and reported separately")
    ap.add_argument("--steps", type=_steps, default=[0.5, 1.0, 2.0, 3.0, 5.0],
                    help="commanded step sizes in degrees (default: the 2026-09-01 gate list)")
    ap.add_argument("--ramp-travel", type=float, default=6.0,
                    help="commanded degrees each ramp walks, whatever its step size")
    ap.add_argument("--dwell-s", type=float, default=2.0, help="hold at each tread")
    ap.add_argument("--hold-s", type=float, default=5.0, help="hold measured for the noise floor")
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="hold discarded before the noise floor is measured, so the closing "
                         "ramp's transient is not mistaken for jaw noise")
    ap.add_argument("--floor-warn", type=float, default=0.3,
                    help="warn when the measured noise floor exceeds this")
    ap.add_argument("--hz", type=float, default=10.0, help="command and sample rate")
    ap.add_argument("--max-travel", type=float, default=12.0,
                    help="degrees of gripper travel allowed from the settled start")
    # The same caps calibrate_pv_object_grip.py holds a grip to. Present_Current
    # is raw LSB and spanned only 0 to 16 across the 2026-08-31 trials, so 50 is
    # well clear of a holding grasp and well under a stall.
    ap.add_argument("--max-current", type=float, default=50.0,
                    help="abort a dwell when Present_Current reaches this raw value")
    ap.add_argument("--max-temperature", type=float, default=55.0,
                    help="abort a dwell when Present_Temperature reaches this many degrees C")
    ap.add_argument("--out", default=None, help="evidence directory (default: <evidence>/gripper_deadband/<stamp>)")
    ap.add_argument("--close-to", type=float, default=None,
                    help="ramp the jaw to this position first, to take the carton. Without it "
                         "the run needs a grasp already held from a teleop or ACT run.")
    ap.add_argument("--close-steps", type=int, default=25, help="waypoints in the closing ramp")
    ap.add_argument("--close-dwell-s", type=float, default=0.15, help="dwell per closing waypoint")
    ap.add_argument("--held-below", type=float, default=90.0,
                    help="gripper positions above this read as an empty, open jaw")
    ap.add_argument("--yes", action="store_true", help="skip the 'carton is held' confirmation")
    args = ap.parse_args()

    if args.close_to is not None and not 0.0 <= args.close_to <= 100.0:
        ap.error("--close-to must be a gripper position")
    if args.close_to is not None and args.close_steps < 1:
        ap.error("--close-steps must be at least 1")
    if args.settle_s < 0.0:
        ap.error("--settle-s must be non-negative")
    if args.ramp_travel <= 0.0:
        ap.error("--ramp-travel must be positive")
    if args.ramp_travel > args.max_travel:
        ap.error("--ramp-travel exceeds --max-travel; the ramps would stop at the cap")
    if not args.arm_enabled:
        print("LOCKED: no command will be sent. Re-run with --arm-enabled to move the jaw.\n")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (evidence_dir() / "gripper_deadband" / stamp)
    summary, samples = run(args)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "telemetry.jsonl", "w", encoding="utf-8") as fh:
        for index, snapshot in enumerate(samples):
            fh.write(json.dumps(serialize_telemetry_snapshot(
                snapshot, target=snapshot.goal_gripper_pos, sample_index=index)) + "\n")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(f"\nwrote {out}/summary.json and telemetry.jsonl")


if __name__ == "__main__":
    main()
