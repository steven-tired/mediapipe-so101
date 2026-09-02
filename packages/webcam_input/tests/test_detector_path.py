"""Where the reused SingleHandDetector is found.

The worktree version of this test asserted a second resolution path: walk up
from `detector.__file__` and look for a sibling `LeFranX/vr-dex-retargeting`
checkout. That behaviour was deliberately dropped in the split -- nothing here
may assume a particular checkout location or a sibling directory on disk, which
is the same rule `paths.py` follows. So the environment variable is the only
way, and what matters is that it is honoured and that its absence fails with an
instruction rather than an import error deep in the module.
"""

import importlib
import sys

import pytest


def _reload_detector():
    sys.modules.pop("webcam_input.detector", None)
    return importlib.import_module("webcam_input.detector")


def _bootstrap(tmp_path):
    """A directory that looks enough like vr-dex-retargeting to be resolved."""
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "single_hand_detector.py").write_text(
        "class SingleHandDetector: pass\n"
        "OPERATOR2MANO_RIGHT = ()\n"
        "OPERATOR2MANO_LEFT = ()\n",
        encoding="utf-8",
    )
    return bootstrap


def test_the_environment_variable_locates_the_reused_detector(tmp_path, monkeypatch):
    bootstrap = _bootstrap(tmp_path)
    monkeypatch.syspath_prepend(str(bootstrap))
    monkeypatch.setenv("VR_DEX_RETARGETING_DIR", str(bootstrap))
    try:
        detector = _reload_detector()

        assert detector.detector_dir() == bootstrap.resolve()
    finally:
        sys.modules.pop("webcam_input.detector", None)
        sys.modules.pop("single_hand_detector", None)


def test_a_directory_without_the_detector_is_rejected_by_name(tmp_path, monkeypatch):
    """Pointing the variable at the wrong directory must say which file is
    missing; the alternative is an ImportError from inside the module."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("VR_DEX_RETARGETING_DIR", str(empty))
    try:
        with pytest.raises(Exception) as excinfo:
            _reload_detector()

        assert "single_hand_detector.py" in str(excinfo.value)
    finally:
        sys.modules.pop("webcam_input.detector", None)
        sys.modules.pop("single_hand_detector", None)
