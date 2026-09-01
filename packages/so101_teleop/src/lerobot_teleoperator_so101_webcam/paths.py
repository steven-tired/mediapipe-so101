"""Path resolution for this repository.

Every location the programs need is resolved here, from the installed package
location and environment overrides. Nothing assumes a particular checkout
location or a sibling directory on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "local_dir", "dataset_root", "evidence_dir", "urdf_path"]


def repo_root() -> Path:
    """Repository root: .../packages/so101_teleop/src/<pkg>/paths.py -> up 4."""
    return Path(__file__).resolve().parents[4]


def local_dir() -> Path:
    """The git-ignored tree holding datasets, evidence, and checkpoints."""
    env = os.environ.get("SO101_LOCAL_DIR")
    return Path(env).expanduser().resolve() if env else repo_root() / "local"


def dataset_root() -> Path:
    """Where recorded LeRobot datasets are written."""
    env = os.environ.get("SO101_DATASET_ROOT")
    return Path(env).expanduser().resolve() if env else local_dir() / "datasets"


def evidence_dir() -> Path:
    """Where diagnostic logs and frames are written."""
    env = os.environ.get("SO101_EVIDENCE_DIR")
    return Path(env).expanduser().resolve() if env else local_dir() / "evidence"


def urdf_path() -> Path:
    """The SO-101 URDF.

    SO-ARM100 is an optional external dependency, not vendored here, so its
    location must be configured. Set SO101_URDF to the .urdf file, or
    SO_ARM100_DIR to a SO-ARM100 checkout.
    """
    env = os.environ.get("SO101_URDF")
    if env:
        return Path(env).expanduser().resolve()
    base = os.environ.get("SO_ARM100_DIR")
    if base:
        return (Path(base).expanduser()
                / "Simulation" / "SO101" / "so101_new_calib.urdf").resolve()
    raise RuntimeError(
        "SO-101 URDF not configured. Set SO101_URDF to the .urdf file, or "
        "SO_ARM100_DIR to a SO-ARM100 checkout "
        "(https://github.com/TheRobotStudio/SO-ARM100)."
    )
