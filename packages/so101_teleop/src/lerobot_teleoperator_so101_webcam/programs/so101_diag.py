#!/usr/bin/env python
"""SO-101 shoulder_lift fault diagnostics (run the steps in order).

Root cause under investigation: motor id 2 (shoulder_lift) trips its own torque
protection while holding the arm extended -> arm folds to the table. These commands
localize WHY (mechanical binding vs failing servo) without guessing.

Run each step with:
  ./scripts/run_so101_diag.sh <cmd>

Commands (in the order you should run them):
  health         dump torque/temp/current/gain/limit registers for all 6 joints
  ids            ping every motor id, show which drop off the bus
  lift           INSTRUMENTED full-lift trip test -- run right after a power-cycle (cold)
  relax          disable torque so you can move the joints BY HAND (binding check)
  setid OLD NEW  reassign one connected servo's id (for swapping in a spare). ONE servo only!
"""
import argparse
import sys
import time

import os

import numpy as np

# The by-id symlink, not ttyACM*: the index flips across replugs, and this tool
# exists to tell a failing servo from a bad connection -- pointing it at the
# wrong device would answer neither.
PORT = os.environ.get(
    "SO101_ARM_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
)
ARM_ID = "so101_follower_1"
NAMES = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
         "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}


# ---------- low-level helpers (retry-tolerant: this bus drops packets intermittently) ----------
def raw_bus(only=None):
    """Raw Feetech bus, no strict handshake (survives a transient motor dropout)."""
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode
    sel = only if only is not None else NAMES
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in sel.items()}
    bus = FeetechMotorsBus(port=PORT, motors=motors)
    bus._connect(handshake=False)
    return bus


def rd(bus, reg, motor, tries=8):
    for _ in range(tries):
        try:
            return bus.read(reg, motor)
        except Exception:
            time.sleep(0.03)
    return None


def wr(bus, reg, motor, val, tries=8):
    for _ in range(tries):
        try:
            bus.write(reg, motor, val)
            return True
        except Exception:
            time.sleep(0.04)
    return False


def connect_robot():
    """Full SOFollower (enables torque, position mode, stock gains, degrees), retrying drops."""
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    robot = SOFollower(SO101FollowerConfig(
        port=PORT, id=ARM_ID, use_degrees=True,
        max_relative_target=None, cameras={}, disable_torque_on_disconnect=False))
    for attempt in range(6):
        try:
            robot.connect(calibrate=False)
            return robot
        except Exception as e:
            try:
                robot.disconnect()
            except Exception:
                pass
            print(f"  connect retry {attempt + 1}/6 ({type(e).__name__})")
            time.sleep(1.0)
    sys.exit("Could not connect. A motor may be genuinely missing -> run the `ids` command.")


def read_pos(robot, tries=40):
    for _ in range(tries):
        try:
            return {k[:-4]: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")}
        except Exception:
            time.sleep(0.05)
    return None


# ----------------------------------- STEP: health -----------------------------------
def cmd_health(_):
    bus = raw_bus()
    regs = ["Torque_Enable", "Present_Temperature", "Present_Current", "Present_Load",
            "Present_Voltage", "P_Coefficient", "D_Coefficient", "Torque_Limit",
            "Protection_Current", "Overload_Torque", "Operating_Mode"]
    print("%-20s " % "register" + " ".join(f"{n[:9]:>9}" for n in NAMES))
    for r in regs:
        vals = [rd(bus, r, n) for n in NAMES]
        print(f"{r:20s} " + " ".join(f"{str(v):>9}" for v in vals))
    bus.disconnect(disable_torque=False)
    print("\nLook at the shoulder_lift column vs the others:")
    print("  Present_Voltage ~123 (=12.3V) on all = power is fine.")
    print("  shoulder_lift much HOTTER / higher current / Torque_Enable=0 = the failing joint.")


# ----------------------------------- STEP: ids -----------------------------------
def cmd_ids(_):
    bus = raw_bus()
    print("pinging motor ids 1..6 (x10) -- finds loose daisy-chain connectors:")
    miss = {i: 0 for i in NAMES.values()}
    for _ in range(10):
        for i in NAMES.values():
            try:
                bus.ping(i)
            except Exception:
                miss[i] += 1
        time.sleep(0.05)
    for n, i in NAMES.items():
        flag = "  <-- DROPPING/ABSENT" if miss[i] else "  ok"
        print(f"  id {i}  {n:14s} missed {miss[i]}/10{flag}")
    bus.disconnect(disable_torque=False)
    print("\nAny non-zero 'missed' = a flaky connector at or before that motor in the chain "
          "(reseat the cables on both sides).")


# ----------------------------------- STEP: lift -----------------------------------
def cmd_lift(_):
    robot = connect_robot()
    bus = robot.bus
    t0 = rd(bus, "Present_Temperature", "shoulder_lift")
    print(f"shoulder_lift start temp = {t0} C  (protection limit is 70 C)")
    if t0 and t0 >= 55:
        print("** Too hot for a fair test. Power OFF, let it cool below ~45 C, then rerun. **")

    def center(m):
        return 50.0 if str(bus.motors[m].norm_mode.value) == "range_0_100" else 0.0
    middle = {m: center(m) for m in NAMES}
    folded = {**middle, "shoulder_lift": -70.0, "elbow_flex": 85.0}

    # start from a real folded bottom so this is an honest bottom->top lift
    p = read_pos(robot) or folded
    print("settling to a folded start ...")
    for a in np.linspace(0, 1, 40):
        bus.sync_write("Goal_Position", {m: (1 - a) * p[m] + a * folded[m] for m in NAMES})
        time.sleep(0.03)
    time.sleep(0.5)
    p = read_pos(robot)
    print("folded at:", {k: round(v, 1) for k, v in p.items()} if p else "(read failed)")

    wr(bus, "Torque_Enable", "shoulder_lift", 1)
    print("\ncommanding a FULL lift to the middle pose (shoulder_lift -> 0). live samples:")
    peak = 0
    start = time.time()
    src = p or folded
    for a in np.linspace(0, 1, 140):
        bus.sync_write("Goal_Position", {m: (1 - a) * src[m] + a * middle[m] for m in NAMES})
        if int(a * 140) % 10 == 0:
            te = rd(bus, "Torque_Enable", "shoulder_lift", 2)
            cur = rd(bus, "Present_Current", "shoulder_lift", 2)
            pos = rd(bus, "Present_Position", "shoulder_lift", 2)
            if cur is not None:
                peak = max(peak, cur)
            el = time.time() - start
            ps = "  ----" if pos is None else f"{pos:+6.1f}"
            tag = "   <-- bus silent (current spike)" if pos is None else (
                  "   <-- TORQUE TRIPPED OFF" if te == 0 else "")
            print(f"  t={el:4.1f}s  TorqueEn={te}  current={cur}  pos={ps}{tag}")
        time.sleep(0.03)

    final = read_pos(robot)
    ft = rd(bus, "Present_Temperature", "shoulder_lift")
    sl = final["shoulder_lift"] if final else None
    print(f"\nfinal shoulder_lift = {sl}  (target 0)   temp {t0}->{ft} C   peak current {peak}")
    done = sl is not None and abs(sl) < 15
    print(f"LIFT COMPLETED: {done}")
    if not done:
        if peak >= 150:
            print(">>> HIGH current before it stalled -> the joint is mechanically HARD to turn "
                  "(binding) or OVERLOADED. Run `relax` and feel it by hand; check the payload.")
        else:
            print(">>> Stalled / tripped at LOW current -> the servo itself is faulty "
                  "(electronics or thermal). Swap motor 2 with a spare (see `setid`).")

    # leave it folded & safe
    p = read_pos(robot) or folded
    for a in np.linspace(0, 1, 40):
        bus.sync_write("Goal_Position", {m: (1 - a) * p[m] + a * folded[m] for m in NAMES})
        time.sleep(0.03)
    robot.disconnect()


# ----------------------------------- STEP: relax -----------------------------------
def cmd_relax(_):
    print("** SUPPORT THE ARM with your hand first -- it will go LIMP when torque drops. **")
    print("   (Ctrl-C now if you're not holding it.)")
    time.sleep(3)
    bus = raw_bus()
    for n in NAMES:
        wr(bus, "Torque_Enable", n, 0)
    bus.disconnect()
    print("Torque DISABLED on all joints. Now, by hand:")
    print("  - slowly rotate shoulder_lift through its full range.")
    print("  - a HEALTHY joint turns smoothly with light, even resistance.")
    print("  - GRINDING, a hard catch, or much stiffer than the elbow = mechanical binding (the fault).")
    print("  - also check the servo horn screw isn't over-tight and no cable is pinched/snagged.")
    print("Re-enable torque just by running `lift` again, or any normal teleop script.")


# ----------------------------------- STEP: setid -----------------------------------
def cmd_setid(args):
    old, new = int(args.old), int(args.new)
    print(f"Reassign servo id {old} -> {new}.")
    print("** ONLY ONE servo may be plugged into the bus right now (disconnect the daisy-chain). **")
    print("   Ctrl-C to abort; continuing in 4s ...")
    time.sleep(4)
    bus = raw_bus(only={"m": old})
    if not wr(bus, "Lock", "m", 0):
        sys.exit("Couldn't reach the servo at id %d. Is exactly one servo connected/powered?" % old)
    ok = wr(bus, "ID", "m", new)
    bus.disconnect()
    if not ok:
        sys.exit("ID write failed.")
    # the servo now answers on `new`; reconnect there to re-lock the EEPROM and verify
    bus2 = raw_bus(only={"m": new})
    wr(bus2, "Lock", "m", 1)
    present = False
    try:
        bus2.ping(new)
        present = True
    except Exception:
        pass
    bus2.disconnect()
    print(f"Done. Servo now answers on id {new}: {'verified' if present else 'NOT seen -- recheck'}.")
    print("Re-chain all motors and run `ids` to confirm 1..6 are all present.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    sub.add_parser("ids")
    sub.add_parser("lift")
    sub.add_parser("relax")
    sp = sub.add_parser("setid")
    sp.add_argument("old")
    sp.add_argument("new")
    args = ap.parse_args()
    {"health": cmd_health, "ids": cmd_ids, "lift": cmd_lift,
     "relax": cmd_relax, "setid": cmd_setid}[args.cmd](args)


if __name__ == "__main__":
    main()
