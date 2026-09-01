"""LeRobot plugin: webcam hand-tracking teleoperator for the SO-101.

Importing this package registers the `so101_webcam` teleoperator choice.
LeRobot auto-imports it via register_third_party_plugins() (distribution name
starts with `lerobot_teleoperator_`).

**Robot-free import.** Setting `LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT=1`
skips the eager plugin imports, so a caller can reach the pure helpers in this
package (`grip.proposal`, `gripper_hardware`, `hand_startup_gate`) without
pulling in `lerobot.motors`, `lerobot.robots`, and a serial stack it will never
use. Analysis and soak runs that touch no hardware rely on this; nothing in a
normal LeRobot run sets the variable, so plugin registration is unaffected.
"""

import os

ROBOT_FREE_IMPORT_ENV = "LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT"

__all__ = ["SO101WebcamConfig", "SO101Webcam", "SO101WebcamEEConfig", "SO101WebcamEE"]

if not os.environ.get(ROBOT_FREE_IMPORT_ENV):
    from .config_so101_webcam import SO101WebcamConfig
    from .config_so101_webcam_ee import SO101WebcamEEConfig
    from .so101_webcam import SO101Webcam
    from .so101_webcam_ee import SO101WebcamEE
