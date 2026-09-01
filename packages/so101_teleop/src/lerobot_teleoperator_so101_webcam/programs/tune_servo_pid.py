"""Objective, closed-loop auto-tuner for the SO-101 servo position-PID (P/I/D coefficients).

Why: LeRobot's configure() writes P=16, I=0, D=32 to every motor on every connect. With
I_Coefficient=0 a constant gravity load leaves a permanent steady-state position error, so the
most-loaded joint (shoulder_lift) droops and the EE/IK arm "stays on the table" even though the
commanded joint target climbs. This tuner finds better P/I/D using ONLY the servo's own encoder
(Present_Position) as the objective -- no human has to describe the behaviour.

What it does, per loaded motor (default shoulder_lift, then elbow_flex):
  1. Holds the arm at a gravity-loaded test pose (default = the all-zeros "open" pose the user
     keeps; shoulder_lift is near-horizontal there = high load).
  2. Measures steady-state error = goal - Present_Position, position noise (shakiness proxy),
     Present_Current (effort) and Present_Temperature (thermal safety).
  3. Recursive coordinate descent over I -> P -> D (the I term is the main lever for the gravity
     droop), keeping a change only if the objective improves without oscillation.
  4. Saturation guard: if Present_Current sits at the 50% cap while the error persists, the pose
     is genuinely torque-limited -- no P/I value can fix it -- so it reports and stops instead of
     chasing impossible gains. (Torque/current caps are LEFT at LeRobot's 50% defaults.)
  5. Saves the tuned per-motor P/I/D to so101_pid.json. teleop re-applies it after connect via
     servo_pid.apply_tuned_pid() (configure() would otherwise overwrite it).

P/I/D are EEPROM registers (Lock-protected), so each candidate is written with torque briefly
disabled (a few ms -> sub-degree dip) then re-enabled to measure. KEEP THE E-STOP IN REACH.

Run (stop other arm apps first):
  env -u PYTHONPATH python -m lerobot_teleoperator_so101_webcam.programs.tune_servo_pid
Options: --motors shoulder_lift,elbow_flex   --yes (skip the confirm prompt)   --dry-run (no writes)
"""

import sys
import time

import numpy as np

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from lerobot_teleoperator_so101_webcam.servo_pid import (
    DEFAULT_PID,
    DEFAULT_PID_PATH,
    read_pid,
    save_pid,
    write_pid,
)

ARM_PORT = "/dev/ttyACM0"
ARM_ID = "so101_follower_1"

MOTORS_TO_TUNE = ["shoulder_lift", "elbow_flex"]   # loaded joints, most-loaded first

# --- candidate grids (raw register units, 0-254) ---
I_GRID = [0, 1, 2, 4, 6, 8, 12, 16, 24]   # integral: the main lever for steady-state droop
P_GRID = [16, 20, 24, 28, 32]             # proportional: stiffen around the LeRobot default 16
D_GRID = [16, 24, 32, 40]                 # derivative: damp oscillation if P/I introduce it
DESCENT_ROUNDS = 2

# --- objective + guards ---
NOISE_FLOOR_DEG = 0.5     # std of Present_Position below this = "not shaking"
NOISE_WEIGHT = 2.0        # deg of objective penalty per deg of noise above the floor
CURRENT_CAP_RAW = 250     # = LeRobot's Protection_Current default (50% cap)
SATURATION_FRAC = 0.85    # current >= 85% of cap + persisting error => torque-limited
TEMP_WARN_C = 55
TEMP_ABORT_C = 60

HOLD_S = 1.5              # settle time at each candidate before measuring
SAMPLES = 12             # Present_Position samples for error + noise

# wrist_roll has a one-sided cable-wrap limit: rotating it past a point binds (draws stall current,
# sags the rail). The tuner never needs to rotate wrist_roll (it tunes shoulder_lift/elbow), so we
# PIN it at its startup angle and clamp every command so it can't move past that angle. None = auto:
# capture the current angle at startup; set a number to override the hold angle (degrees).
WRIST_ROLL_HOLD_DEG = None
_wrist_hold = None        # actual hold angle, set in main()


def clamp_pose(cmd):
    """Pin wrist_roll at its startup angle so the tuner can't drive it into the cable-wrap limit."""
    if _wrist_hold is not None and "wrist_roll.pos" in cmd:
        cmd = dict(cmd)
        cmd["wrist_roll.pos"] = _wrist_hold
    return cmd


def connect_arm():
    robot = SOFollower(SO101FollowerConfig(
        port=ARM_PORT, id=ARM_ID, use_degrees=True,
        max_relative_target=None, cameras={}, disable_torque_on_disconnect=False,
    ))
    for attempt in range(3):
        try:
            robot.connect(calibrate=False)
            return robot
        except Exception:
            if attempt == 2:
                raise
            try:
                robot.disconnect()
            except Exception:
                pass
            time.sleep(1.0)


def get_obs_retry(robot, tries=12):
    for _ in range(tries):
        try:
            obs = robot.get_observation()
            return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.08)
    raise ConnectionError("Arm position read kept failing -- check USB cable/port.")


def ramp_to(robot, goal, secs=1.5):
    """Gently ramp from the arm's current pose to `goal`, holding each step so slow joints follow."""
    start = get_obs_retry(robot)
    steps = max(int(secs / 0.04), 1)
    for a in np.linspace(0.0, 1.0, steps):
        cmd = {k: (1 - a) * start.get(k, goal[k]) + a * goal[k] for k in goal}
        robot.send_action(clamp_pose(cmd))
        time.sleep(0.04)


def set_pid_safely(robot, motor, pid, dry_run=False):
    """Write P/I/D (EEPROM, Lock-protected) with torque briefly disabled, then re-enable.

    The off-window is only the 3 register writes (tens of ms -> sub-degree dip). Other motors
    are toggled too (disable/enable take the whole bus) but they re-hold immediately.
    """
    if dry_run:
        return
    robot.bus.disable_torque()        # Lock=0 -> EEPROM writable
    try:
        write_pid(robot, motor, pid)
    finally:
        robot.bus.enable_torque()     # Lock=1 + torque on


def settle_and_measure(robot, pose, motor, hold_s=HOLD_S, samples=SAMPLES):
    """Hold `pose`, then measure the target motor's steady-state behaviour from its own sensors."""
    pose = clamp_pose(pose)
    deadline = time.time() + hold_s
    while time.time() < deadline:
        robot.send_action(pose)
        time.sleep(0.04)

    key = f"{motor}.pos"
    goal = float(pose[key])
    present, currents, temps = [], [], []
    for _ in range(samples):
        obs = get_obs_retry(robot)
        present.append(obs[key])
        try:
            currents.append(abs(int(robot.bus.read("Present_Current", motor, normalize=False, num_retry=5))))
            temps.append(int(robot.bus.read("Present_Temperature", motor, normalize=False, num_retry=5)))
        except (ConnectionError, RuntimeError, KeyError):
            pass   # dropped read or model lacks the register; current/temp guards just no-op then
        robot.send_action(pose)   # keep holding while sampling
        time.sleep(0.03)

    present = np.asarray(present, dtype=float)
    return {
        "goal": goal,
        "present": float(present.mean()),
        "error": goal - float(present.mean()),
        "abs_error": abs(goal - float(present.mean())),
        "noise": float(present.std()),
        "current": float(np.mean(currents)) if currents else 0.0,
        "temp": int(max(temps)) if temps else 0,
    }


def objective(m):
    """Lower = better: steady-state error, penalized for shakiness (Present_Position noise)."""
    return m["abs_error"] + NOISE_WEIGHT * max(0.0, m["noise"] - NOISE_FLOOR_DEG)


def thermal_check(m):
    if m["temp"] >= TEMP_ABORT_C:
        raise RuntimeError(f"ABORT: motor temperature {m['temp']}C >= {TEMP_ABORT_C}C. Let it cool.")
    if m["temp"] >= TEMP_WARN_C:
        print(f"    [warn] temperature {m['temp']}C -- approaching limit, slowing down")
        time.sleep(5.0)


def saturated(m):
    return m["current"] >= SATURATION_FRAC * CURRENT_CAP_RAW and m["abs_error"] > 2.0


def evaluate(robot, motor, pose, pid, dry_run=False):
    set_pid_safely(robot, motor, pid, dry_run=dry_run)
    m = settle_and_measure(robot, pose, motor)
    thermal_check(m)
    m["pid"] = dict(pid)
    m["J"] = objective(m)
    print(f"    P={pid['P_Coefficient']:>3} I={pid['I_Coefficient']:>3} D={pid['D_Coefficient']:>3}"
          f" | err={m['error']:+6.2f}deg noise={m['noise']:4.2f} cur={m['current']:5.0f}"
          f" temp={m['temp']:>2}C | J={m['J']:.2f}"
          + ("  <SATURATED>" if saturated(m) else ""))
    return m


def sweep(robot, motor, pose, best, reg, grid, dry_run=False):
    """Coordinate-descent over one register; return the best measurement found (kept or improved)."""
    best_m = best
    for val in grid:
        if val == best_m["pid"][reg]:
            continue
        cand = dict(best_m["pid"]); cand[reg] = val
        m = evaluate(robot, motor, pose, cand, dry_run=dry_run)
        if saturated(m) and m["abs_error"] >= best_m["abs_error"]:
            print(f"    -> {motor} torque-limited at the 50% cap; higher {reg} won't help. Stopping sweep.")
            break
        if m["J"] < best_m["J"]:
            best_m = m
    return best_m


def tune_motor(robot, motor, pose, dry_run=False):
    print(f"\n=== Tuning {motor} (baseline = LeRobot configure(): "
          f"P{DEFAULT_PID['P_Coefficient']}/I{DEFAULT_PID['I_Coefficient']}/D{DEFAULT_PID['D_Coefficient']}) ===")
    baseline = evaluate(robot, motor, pose, dict(DEFAULT_PID), dry_run=dry_run)
    best = baseline
    if dry_run:
        # No PID is written in dry-run, so sweeping would just re-measure the same state. Report
        # the baseline (verifies reads work + shows the current sag) and stop.
        print(f"  >> {motor}: baseline error {baseline['error']:+.2f} deg "
              f"(dry run -- no sweep, no writes)"
              + ("  <SATURATED at 50% cap>" if saturated(baseline) else ""))
        return baseline
    for rnd in range(DESCENT_ROUNDS):
        print(f"  -- round {rnd + 1}: integral (I) --")
        best = sweep(robot, motor, pose, best, "I_Coefficient", I_GRID, dry_run=dry_run)
        print(f"  -- round {rnd + 1}: proportional (P) --")
        best = sweep(robot, motor, pose, best, "P_Coefficient", P_GRID, dry_run=dry_run)
        if best["noise"] > NOISE_FLOOR_DEG:
            print(f"  -- round {rnd + 1}: derivative (D, damp shakiness) --")
            best = sweep(robot, motor, pose, best, "D_Coefficient", D_GRID, dry_run=dry_run)
    improved = baseline["abs_error"] - best["abs_error"]
    print(f"  >> {motor}: error {baseline['error']:+.2f} -> {best['error']:+.2f} deg "
          f"(reduced {improved:+.2f}); best PID {best['pid']}")
    if saturated(best):
        print(f"  >> NOTE: {motor} is torque-limited at the 50% cap at this pose -- P/I cannot fully "
              f"fix it. Revisit the torque-cap decision if this joint still won't hold.")
    return best


def main():
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    auto_yes = "--yes" in argv
    motors = MOTORS_TO_TUNE
    if "--motors" in argv:
        motors = argv[argv.index("--motors") + 1].split(",")

    if not auto_yes:
        print("This MOVES the arm and briefly drops torque to write EEPROM PID. Clear the workspace "
              "and keep the e-stop in reach.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted."); return

    robot = connect_arm()
    body = [m for m in robot.bus.motors if m != "gripper"]
    # Capture the current pose so we can PIN wrist_roll where it is now (its cable limits how far it
    # can rotate; driving it toward 0 binds -> stall current + rail sag).
    global _wrist_hold
    start_pose = get_obs_retry(robot)
    _wrist_hold = (WRIST_ROLL_HOLD_DEG if WRIST_ROLL_HOLD_DEG is not None
                   else start_pose.get("wrist_roll.pos", 0.0))
    # Loaded test pose = the all-zeros "open" pose (body 0 deg) the user keeps; shoulder_lift is
    # near-horizontal here = high gravity load = the regime that droops to the table. wrist_roll is
    # held at its current angle (NOT driven to 0) so the tuner never hits the cable-wrap limit.
    test_pose = {f"{m}.pos": 0.0 for m in body}
    test_pose["gripper.pos"] = 50.0
    test_pose["wrist_roll.pos"] = _wrist_hold
    print(f"wrist_roll pinned at {_wrist_hold:.1f} deg (cable-wrap limit) -- tuner will not rotate it.")

    print(f"PID file: {DEFAULT_PID_PATH}  (motors to tune: {motors}){'  [DRY RUN]' if dry_run else ''}")
    print("Ramping to the loaded test pose...")
    ramp_to(robot, test_pose, secs=2.0)

    results = {}
    try:
        for motor in motors:
            if motor not in robot.bus.motors:
                print(f"[skip] unknown motor '{motor}'"); continue
            try:
                before = read_pid(robot, motor)
            except (ConnectionError, RuntimeError):
                before = "<read dropped>"   # informational only; don't abort the run
            best = tune_motor(robot, motor, test_pose, dry_run=dry_run)
            results[motor] = best["pid"]
            # restore this motor to the chosen best before moving on
            set_pid_safely(robot, motor, best["pid"], dry_run=dry_run)
            print(f"  (live PID before tuning was {before})")
    finally:
        if results and not dry_run:
            path = save_pid(results)
            print(f"\nSaved tuned PID -> {path}")
            print("teleop will re-apply it after connect via servo_pid.apply_tuned_pid().")
        else:
            print("\nNo file written (dry run or no results).")
        print("Returning arm to rest and disconnecting...")
        try:
            ramp_to(robot, test_pose, secs=1.5)
        except Exception:
            pass
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
