"""Measure and cap per-servo current on the SO-101 (Feetech STS3215).

Reads drop ("no status packet") during grip+lift even though bus power is fine (12V/5A). The likely
cause is a servo hitting its own OVER-CURRENT PROTECTION: when it trips it stops answering serial,
so a sync_read of all ids fails. This tool lets you:
  1. MONITOR each servo's actual current (and temperature) -> find which one spikes/trips.
  2. CAP a servo's Torque_Limit (RAM reg 48, 0..1000 = % of max torque) -> bounds its stall current
     so it still holds/grips firmly but can't spike into the protection trip. RAM = resets on
     power-cycle, so nothing permanent.

No arm motion is commanded. Run while you hand-load the arm / close the gripper on a block to see
the spike, or run --monitor in one terminal during a teleop/record session in another.

Usage (env -u PYTHONPATH .../.venv-lerobot/bin/python servo_current.py ...):
  --monitor                 live Present_Current + Present_Temperature for all 6 servos
  --show                    print each servo's current Torque_Limit
  --set gripper=600 shoulder_lift=800   cap those servos' Torque_Limit (others unchanged)
  --reset                   restore all servos to full torque (1000)

Units: Present_Current is raw LSB (~6.5 mA/LSB on STS3215); compare relative values, watch for the
one that jumps when reads start dropping.
"""

import sys
import time

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ARM_PORT = "/dev/ttyACM0"
ARM_ID = "so101_follower_1"


def _connect():
    robot = SOFollower(SO101FollowerConfig(port=ARM_PORT, id=ARM_ID, use_degrees=True,
                                           cameras={}, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    return robot


def _read_all(bus, reg):
    try:
        return bus.sync_read(reg, normalize=False)
    except Exception as e:
        return {"_err": str(e)[:40]}


def monitor(robot):
    bus = robot.bus
    motors = list(bus.motors.keys())
    print("Watching Present_Current (Ctrl+C to stop). Tracks read SUCCESS RATE so you can compare")
    print("e.g. arm-on-hub vs arm-on-direct-USB-port. A healthy link should be ~100%.\n")
    print("           " + "  ".join(f"{m[:9]:>9}" for m in motors))
    ok = fail = 0
    while True:
        cur = _read_all(bus, "Present_Current")
        if "_err" in cur:
            fail += 1
        else:
            ok += 1
            print("current   " + "  ".join(f"{cur.get(m, 0):>9}" for m in motors))
        total = ok + fail
        print(f"  reads ok={ok} fail={fail}  success={100*ok/total:.0f}%"
              + ("   <-- LAST READ DROPPED" if "_err" in cur else ""))
        time.sleep(0.1)


def show(robot):
    bus = robot.bus
    for m in bus.motors:
        try:
            v = bus.read("Torque_Limit", m, normalize=False)
            mx = bus.read("Max_Torque_Limit", m, normalize=False)
            print(f"{m:>13}: Torque_Limit={v}  Max_Torque_Limit={mx}")
        except Exception as e:
            print(f"{m:>13}: read failed: {e}")


def set_limits(robot, pairs):
    bus = robot.bus
    for name, val in pairs.items():
        if name not in bus.motors:
            print(f"  ! unknown servo '{name}' (have: {', '.join(bus.motors)})")
            continue
        val = max(0, min(1000, int(val)))
        bus.write("Torque_Limit", name, val, normalize=False)
        print(f"  {name}: Torque_Limit -> {val}")
    print("Done (RAM; resets on power-cycle).")


def main():
    args = sys.argv[1:]
    robot = _connect()
    try:
        if "--monitor" in args:
            monitor(robot)
        elif "--show" in args:
            show(robot)
        elif "--reset" in args:
            set_limits(robot, {m: 1000 for m in robot.bus.motors})
        elif "--set" in args:
            pairs = {}
            for a in args[args.index("--set") + 1:]:
                if "=" in a:
                    k, v = a.split("=", 1)
                    pairs[k] = v
            if not pairs:
                print("nothing to set; e.g. --set gripper=600 shoulder_lift=800")
            else:
                set_limits(robot, pairs)
        else:
            print(__doc__)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
