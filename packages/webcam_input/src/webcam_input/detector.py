"""Reuse the seniors' MediaPipe hand detector instead of reimplementing it.

`SingleHandDetector` (in the vr-dex-retargeting repo, under
example/vector_retargeting/single_hand_detector.py) already does the full
webcam → MediaPipe → 21 landmarks → MANO `joint_pos[21,3]` pipeline and returns
a wrist-orientation frame. It depends only on mediapipe + numpy.

It is resolved as an explicit dependency, in this order:

1. the ``VR_DEX_RETARGETING_DIR`` environment variable, pointing at the
   ``example/vector_retargeting`` directory;
2. ``single_hand_detector`` already importable, i.e. vr-dex-retargeting is
   installed in the environment.

There is deliberately no sibling-checkout default. This package does not assume
anything about what sits next to it on disk.
"""

import importlib.util
import os
import sys
from pathlib import Path


_HINT = (
    "Set VR_DEX_RETARGETING_DIR to the vr-dex-retargeting "
    "example/vector_retargeting directory, or install vr-dex-retargeting into "
    "this environment."
)


def detector_dir() -> Path | None:
    """The configured vr-dex-retargeting directory, or None if unset."""
    env = os.environ.get("VR_DEX_RETARGETING_DIR")
    return Path(env).expanduser().resolve() if env else None


_DIR = detector_dir()
if _DIR is not None:
    if not (_DIR / "single_hand_detector.py").exists():
        raise ImportError(
            f"VR_DEX_RETARGETING_DIR is {_DIR}, but single_hand_detector.py is "
            f"not there. {_HINT}"
        )
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))
elif importlib.util.find_spec("single_hand_detector") is None:
    raise ImportError(f"vr-dex-retargeting is not available. {_HINT}")

from single_hand_detector import (  # noqa: E402
    SingleHandDetector,
    OPERATOR2MANO_RIGHT,
    OPERATOR2MANO_LEFT,
)

__all__ = ["SingleHandDetector", "OPERATOR2MANO_RIGHT", "OPERATOR2MANO_LEFT", "detector_dir"]
