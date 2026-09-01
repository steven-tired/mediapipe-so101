"""Per-motor Feetech position-PID helpers (P/I/D coefficient registers).

Why this exists: LeRobot's ``SOFollower.configure()`` rewrites ``P_Coefficient=16,
I_Coefficient=0, D_Coefficient=32`` to EVERY motor on every ``connect()`` (see
``lerobot/.../so_follower.py``). With ``I_Coefficient=0`` a constant gravity load leaves a
permanent steady-state position error -- the most-loaded joint (``shoulder_lift``) droops, so
the EE/IK arm "stays on the table" even though the commanded joint target climbs.

``tune_servo_pid.py`` finds better per-motor P/I/D objectively (from the servo's own encoder),
saves them here, and teleop calls ``apply_tuned_pid()`` right AFTER ``robot.connect()`` to
re-apply them (otherwise ``configure()`` would have just overwritten them).

Registers are raw bytes (0-254), so all reads/writes pass ``normalize=False``.
"""

import json
import os

PID_REGISTERS = ("P_Coefficient", "I_Coefficient", "D_Coefficient")

# LeRobot configure() defaults (the baseline the tuner improves on). Kept here so the tuner can
# seed from / fall back to them without re-reading so_follower.py.
DEFAULT_PID = {"P_Coefficient": 16, "I_Coefficient": 0, "D_Coefficient": 32}

# Default location of the tuned-PID file (next to the teleop package).
DEFAULT_PID_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "so101_pid.json")


def read_pid(robot, motor: str, num_retry: int = 5) -> dict:
    """Read the live {P,I,D} coefficients of one motor (raw register units).

    ``num_retry`` defaults to 5 because this serial bus intermittently drops single reads
    ("There is no status packet!"); the rest of the codebase retries reads for the same reason.
    """
    return {reg: int(robot.bus.read(reg, motor, normalize=False, num_retry=num_retry))
            for reg in PID_REGISTERS}


def write_pid(robot, motor: str, pid: dict, num_retry: int = 5) -> None:
    """Write whichever of P/I/D coefficients are present in ``pid`` (raw register units)."""
    for reg in PID_REGISTERS:
        if reg in pid and pid[reg] is not None:
            robot.bus.write(reg, motor, int(pid[reg]), normalize=False, num_retry=num_retry)


def save_pid(table: dict, path: str = DEFAULT_PID_PATH) -> str:
    """Persist a {motor: {P_Coefficient,I_Coefficient,D_Coefficient}} table to JSON."""
    with open(path, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    return path


def load_pid(path: str = DEFAULT_PID_PATH) -> dict:
    """Load a tuned-PID table, or return {} if no file exists yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def apply_tuned_pid(robot, path: str = DEFAULT_PID_PATH) -> dict:
    """Re-apply saved per-motor P/I/D to the live arm; call AFTER ``robot.connect()``.

    Returns the table that was applied (empty dict if no tuned file exists, in which case the
    LeRobot configure() defaults remain in effect). Never raises on a missing file so teleop
    runs fine before the tuner has been run.
    """
    table = load_pid(path)
    applied = {}
    for motor, pid in table.items():
        if motor in robot.bus.motors:
            write_pid(robot, motor, pid)
            applied[motor] = pid
    return applied
