"""The package must be reachable without booting the robot stack.

`grip.proposal`, `gripper_hardware`, and `hand_startup_gate` are pure helpers,
but importing any of them imports the package `__init__`, which registers the
LeRobot plugin and therefore pulls in `lerobot.motors`, `lerobot.robots`, and a
serial stack. A robot-free consumer -- an offline analysis
run, or a private-side soak -- must be able to opt out.

This is not hypothetical: the guard existed before the repository split, was
dropped when this package was rewritten, and its absence broke the soak's
robot-free property the moment a shared module moved here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ENV = "LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT"

PROBE = """
import json, sys
import lerobot_teleoperator_so101_webcam as plugin
from lerobot_teleoperator_so101_webcam.grip.proposal import PressureProposalStateMachine
from lerobot_teleoperator_so101_webcam.gripper_hardware import GripperTelemetrySampler
from lerobot_teleoperator_so101_webcam.hand_startup_gate import ContinuousHandStartupGate
print(json.dumps({
    "robot_modules": sorted(
        name for name in sys.modules
        if name == "serial" or name.startswith("serial.")
        or name.startswith("lerobot.motors")
        or name.startswith("lerobot.robots")
        or name.startswith("lerobot.teleoperators")
    ),
    "has_plugin_classes": hasattr(plugin, "SO101Webcam"),
}))
"""


SRC_DIRS = [str(Path(__file__).resolve().parents[3] / part / "src")
            for part in ("packages/so101_teleop", "packages/webcam_input")]


def _probe(robot_free: bool):
    env = dict(os.environ)
    env.pop(ENV, None)
    # Prepend this checkout so the child cannot pick up an installed copy of the
    # package from a different tree -- the point is to probe *this* __init__.
    env["PYTHONPATH"] = os.pathsep.join(SRC_DIRS)
    if robot_free:
        env[ENV] = "1"
    done = subprocess.run([sys.executable, "-c", PROBE], env=env,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_robot_free_import_loads_the_helpers_without_the_robot_stack():
    payload = _probe(robot_free=True)

    assert payload["robot_modules"] == []


def test_robot_free_import_does_not_expose_the_plugin_classes():
    """The opt-out is real, not cosmetic: the classes genuinely are not imported."""
    assert _probe(robot_free=True)["has_plugin_classes"] is False


def test_a_normal_import_still_registers_the_plugin():
    """Nothing in a normal LeRobot run sets the variable, so registration must be intact."""
    payload = _probe(robot_free=False)

    assert payload["has_plugin_classes"] is True
    assert payload["robot_modules"], "a normal import is expected to load the robot stack"


def test_the_guard_is_declared_in_the_package():
    import lerobot_teleoperator_so101_webcam as plugin

    assert plugin.ROBOT_FREE_IMPORT_ENV == ENV
