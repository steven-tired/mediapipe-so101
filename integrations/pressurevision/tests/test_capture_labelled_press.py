"""Level bookkeeping for labelled press capture: ordinals, baselines, camera choice."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import capture_labelled_press as clp


def _args(tmp_path, *extra):
    return clp.parse_args(
        [
            "--session-dir", str(tmp_path / "session"),
            "--crop", "140,60,860,540",
            "--surface", "white paper on table",
            *extra,
        ]
    )


def test_the_overhead_uvc_camera_is_the_default(tmp_path):
    """PressureVision is RGB-only, and the crop is aimed with the C270."""
    args = _args(tmp_path)
    assert (args.camera, args.realsense) == (2, False)


def test_intent_labels_become_ordinals_keeping_zero_as_the_baseline(tmp_path):
    args = _args(tmp_path, "--intent-labels", "none,light,hard")
    assert args.targets_g == [0, 1, 2]
    assert args.intent_labels == ["none", "light", "hard"]
    assert args.contact_gated is False


def test_contact_gated_intent_labels_reserve_zero_for_the_external_gate(tmp_path):
    args = _args(tmp_path, "--intent-labels", "light,hard")
    assert args.targets_g == [1, 2]
    assert args.intent_labels == ["light", "hard"]
    assert args.contact_gated is True


def test_naming_a_scale_while_labelling_by_intent_is_refused(tmp_path):
    """The two describe different ground truths; recording both would misstate one."""
    with pytest.raises(SystemExit):
        _args(tmp_path, "--intent-labels", "none,light,hard", "--scale-model", "kitchen")


def test_gram_targets_survive_untouched_when_a_scale_is_used(tmp_path):
    args = _args(tmp_path, "--targets-g", "0,100,330", "--scale-model", "kitchen")
    assert args.targets_g == [0, 100, 330]
    assert args.intent_labels is None


def test_trial_order_interleaves_every_level_within_each_block(tmp_path):
    """Blocked interleaving is what stops slow drift masquerading as an effect."""
    import numpy as np

    trials = clp.trial_order([0, 1, 2], repeats=4, rng=np.random.default_rng(0))
    assert len(trials) == 12
    for block in range(4):
        assert sorted(g for b, g in trials if b == block) == [0, 1, 2]


def test_hold_length_is_a_duration_not_a_frame_count(tmp_path):
    """8 frames is 1.1 s on the 7.5 fps C270 and a 0.27 s transient on a 30 fps
    camera, so a camera swap would silently change the protocol."""
    args = _args(tmp_path)
    assert args.hold_seconds == 1.0
    assert args.hold_frames is None


def test_a_fixed_frame_count_still_overrides(tmp_path):
    args = _args(tmp_path, "--hold-frames", "12")
    assert args.hold_frames == 12


def test_a_nonpositive_hold_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--hold-seconds", "0")
    with pytest.raises(SystemExit):
        _args(tmp_path, "--hold-frames", "0")
