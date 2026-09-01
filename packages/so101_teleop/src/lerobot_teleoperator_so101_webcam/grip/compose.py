"""Choose a gripper controller by mode name.

This is the only place that knows PressureVision might exist, and it imports the
adapter lazily so the core keeps working when PressureVision is not installed.
Nothing under `lerobot_teleoperator_so101_webcam` outside this module may import
`pressurevision_integration`.
"""

from __future__ import annotations

from .contract import GripperController
from .mediapipe import MediaPipeGripperController

__all__ = ["GRIPPER_MODES", "build_gripper", "add_gripper_mode_argument"]

GRIPPER_MODES = ("mediapipe", "pressurevision")


def build_gripper(
    mode: str = "mediapipe",
    *,
    zero_pos: float | None = None,
    one_pos: float | None = None,
) -> GripperController:
    """Build the controller for `mode`.

    `mediapipe` (the default) derives the command from pinch. `pressurevision`
    uses PV for grip strength only, inside the [`one_pos`, `zero_pos`] span; it
    still cannot open the gripper, because only an explicit MediaPipe release
    does that.
    """
    if mode == "mediapipe":
        return MediaPipeGripperController()

    if mode == "pressurevision":
        if zero_pos is None or one_pos is None:
            raise ValueError(
                "--gripper-mode pressurevision needs --grip-zero-pos and "
                "--grip-one-pos, the object profile's position span."
            )
        try:
            from pressurevision_integration.adapter import PressureVisionGripperController
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "--gripper-mode pressurevision requires the optional "
                "mediapipe-so101-pressurevision package. Install "
                "integrations/pressurevision, and run the PV sender as its own "
                "process."
            ) from exc
        return PressureVisionGripperController(zero_pos=zero_pos, one_pos=one_pos)

    raise ValueError(f"unknown gripper mode {mode!r}; expected one of {GRIPPER_MODES}")


def add_gripper_mode_argument(parser) -> None:
    """Add --gripper-mode and its span options to an argparse parser."""
    parser.add_argument("--gripper-mode", choices=GRIPPER_MODES, default="mediapipe",
                        help="mediapipe (default): pinch drives the gripper. "
                             "pressurevision: PV sets grip strength only.")
    parser.add_argument("--grip-zero-pos", type=float, default=None,
                        help="loosest PV-commanded gripper position (pressurevision mode)")
    parser.add_argument("--grip-one-pos", type=float, default=None,
                        help="firmest PV-commanded gripper position (pressurevision mode)")
