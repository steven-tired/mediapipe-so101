"""Sender-side wire format and frame reduction.

The consumer lives in another repo on another venv and writes its own codec, so
what keeps the two in step is that both round-trip SHARED_PV_TEST_PACKET. The
matching test is
webcam-input/.../tests/test_pv_pressure.py::test_shared_test_vector_decodes_to_known_values.
"""

import json
import mmap
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import serve_pad_pressure as sp


def _session(tmp_path, metadata=None):
    """A session dir that passes the emptiness guard; load_trials is stubbed anyway."""
    session = tmp_path / "session"
    session.mkdir()
    row = {"row_type": "metadata", **(metadata or {})}
    (session / "capture.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return session


def test_encoder_reproduces_the_shared_test_vector():
    fields = sp.SHARED_PV_TEST_PACKET.split(",")
    payload = sp.encode_pv_packet(
        sequence=int(fields[1]),
        source_observed_at_s=float(fields[2]),
        sent_at_s=float(fields[3]),
        mean_kpa_in_contact=float(fields[4]),
        max_kpa=float(fields[5]),
        sum_kpa=float(fields[6]),
        contact_px=int(fields[7]),
        off_x=float(fields[8]),
        off_y=float(fields[9]),
        crop_w=int(fields[10]),
        crop_h=int(fields[11]),
        pressure_0_1=float(fields[12]),
        level=int(fields[13]),
        n_levels=int(fields[14]),
    )
    assert payload == (sp.SHARED_PV_TEST_PACKET + "\n").encode("utf-8")


def test_no_contact_reports_zero_area_and_a_centred_offset():
    kpa = np.zeros((100, 200), dtype=np.float32)
    metrics = sp.reduce_frame(kpa, contact_thresh=1.0)
    assert metrics["contact_px"] == 0
    assert metrics["mean_kpa_in_contact"] == 0.0
    assert metrics["sum_kpa"] == 0.0
    assert (metrics["off_x"], metrics["off_y"]) == (0.0, 0.0)


def test_offsets_are_crop_fractions_not_pixels():
    """Same relative position in a differently sized crop must give the same offset."""
    small = np.zeros((100, 200), dtype=np.float32)
    small[25, 50] = 8.0            # quarter across, quarter down
    large = np.zeros((400, 800), dtype=np.float32)
    large[100, 200] = 8.0

    a = sp.reduce_frame(small, contact_thresh=1.0)
    b = sp.reduce_frame(large, contact_thresh=1.0)
    assert a["off_x"] == pytest.approx(b["off_x"]) == pytest.approx(-0.5)
    assert a["off_y"] == pytest.approx(b["off_y"]) == pytest.approx(-0.5)


def test_a_centred_contact_reports_zero_offset_and_its_mean():
    kpa = np.zeros((100, 200), dtype=np.float32)
    kpa[49:51, 99:101] = np.array([[4.0, 6.0], [6.0, 8.0]], dtype=np.float32)
    metrics = sp.reduce_frame(kpa, contact_thresh=1.0)
    assert metrics["contact_px"] == 4
    assert metrics["mean_kpa_in_contact"] == pytest.approx(6.0)
    assert metrics["max_kpa"] == pytest.approx(8.0)
    assert metrics["sum_kpa"] == pytest.approx(24.0)
    assert metrics["off_x"] == pytest.approx(0.0, abs=0.02)
    assert metrics["off_y"] == pytest.approx(0.0, abs=0.02)


def test_subthreshold_pressure_is_not_contact():
    kpa = np.full((100, 200), 0.5, dtype=np.float32)
    assert sp.reduce_frame(kpa, contact_thresh=1.0)["contact_px"] == 0


def test_levels_are_refused_when_the_session_fails_the_gate(tmp_path, monkeypatch):
    """An ungated anchor pair would make the consumer's 0..1 scale arbitrary."""
    session = _session(tmp_path)
    monkeypatch.setattr(
        sp, "trials_excluding_early_lifts",
        lambda *_: ({100: np.array([4.0, 9.0]), 330: np.array([5.0, 8.0])}, [])
    )
    with pytest.raises(SystemExit, match="calibration gate rejected"):
        sp.fit_levels(session)


def test_boundaries_sit_between_cleanly_separated_levels(tmp_path, monkeypatch):
    session = _session(tmp_path)
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                100: np.array([4.0, 4.1, 3.9, 4.0]),
                330: np.array([12.0, 12.1, 11.9, 12.0]),
            },
            [],
        ),
    )
    fitted = sp.fit_levels(session)
    assert fitted["levels"] == [100, 330]
    boundary = fitted["boundaries"][0]
    assert boundary["separated_in_fit"] is True
    assert boundary["edge"] == pytest.approx((4.1 + 11.9) / 2)


def test_a_no_press_level_gets_its_own_boundary(tmp_path, monkeypatch):
    """Every session carries a no-press level, and it must classify as level 0 --
    folding it in with the lightest press would grip on a hand barely touching."""
    session = _session(tmp_path)
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                0: np.array([0.0, 0.1, 0.0, 0.05]),
                100: np.array([6.0, 6.3, 6.1, 6.2]),
                330: np.array([13.0, 13.2, 12.9, 13.1]),
            },
            [],
        ),
    )
    fitted = sp.fit_levels(session)
    assert fitted["levels"] == [0, 100, 330]
    assert len(fitted["boundaries"]) == 2
    assert sp.level_index(0.05, fitted) == 0
    assert sp.level_index(6.15, fitted) == 1
    assert sp.level_index(13.05, fitted) == 2


def test_contact_gated_fit_ignores_no_press_and_reserves_wire_level_zero(tmp_path, monkeypatch):
    """Pinch owns contact, so a bad none/light split must not reject light/hard.

    Wire level zero remains the consumer's baseline; the two pressure grades are
    therefore levels one and two, not a re-numbered zero and one.
    """
    session = _session(
        tmp_path,
        {"targets_g": [0, 1, 2], "intent_labels": ["none", "light", "hard"]},
    )
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                0: np.array([0.0, 20.0, 5.0, 18.0]),
                1: np.array([6.0, 6.2, 5.8, 6.1]),
                2: np.array([13.0, 13.2, 12.9, 13.1]),
            },
            [],
        ),
    )

    fitted = sp.fit_levels(session, contact_gated=True)

    assert fitted["levels"] == [1, 2]
    assert fitted["wire_level_offset"] == 1
    assert fitted["n_levels"] == 3
    assert sp.level_index(6.0, fitted) == 1
    assert sp.level_index(13.0, fitted) == 2


def test_contact_gated_capture_metadata_enables_the_fit_without_a_cli_override(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, {"contact_gated": True})
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                1: np.array([6.0, 6.2, 5.8, 6.1]),
                2: np.array([13.0, 13.2, 12.9, 13.1]),
            },
            [],
        ),
    )

    fitted = sp.fit_levels(session)

    assert fitted["contact_gated"] is True
    assert fitted["wire_level_offset"] == 1
    assert fitted["n_levels"] == 3


def test_contact_gated_fit_writes_continuous_anchors_from_trial_medians(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, {"contact_gated": True})
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                1: np.array([4.0, 5.0, 6.0]),
                2: np.array([14.0, 15.0, 16.0]),
                3: np.array([29.0, 30.0, 31.0]),
            },
            [],
        ),
    )

    fitted = sp.fit_levels(session)

    assert fitted["continuous_anchors"] == [
        {"sum_kpa": 5.0, "pressure_0_1": 0.0},
        {"sum_kpa": 15.0, "pressure_0_1": 0.5},
        {"sum_kpa": 30.0, "pressure_0_1": 1.0},
    ]
    assert sp.continuous_pressure(10.0, fitted) == pytest.approx(0.25)


def test_a_reading_between_levels_abstains_rather_than_guessing(tmp_path, monkeypatch):
    """The whole reason for bands over a ramp: no answer beats a confident wrong one."""
    session = _session(tmp_path)
    monkeypatch.setattr(
        sp,
        "trials_excluding_early_lifts",
        lambda *_: (
            {
                0: np.array([0.0, 0.1, 0.0, 0.05]),
                1: np.array([4.0, 4.1, 3.9, 4.0]),
                2: np.array([12.0, 12.1, 11.9, 12.0]),
            },
            [],
        ),
    )
    fitted = sp.fit_levels(session, abstain_fraction=0.5)
    edge = fitted["boundaries"][1]["edge"]
    assert sp.level_index(edge, fitted) == sp.ABSTAIN_LEVEL
    # Abstaining can be switched off, and then the same reading decides.
    decisive = sp.fit_levels(session, abstain_fraction=0.0)
    assert sp.level_index(edge, decisive) != sp.ABSTAIN_LEVEL


def test_streaming_without_fitted_levels_is_a_usage_error():
    with pytest.raises(SystemExit):
        sp.parse_args([])


def test_levels_fitted_on_another_metric_are_refused(tmp_path):
    """A boundary in contact_px would be read as kPa and silently misclassify."""
    path = tmp_path / "levels.json"
    path.write_text('{"metric": "contact_px", "levels": [0, 1]}', encoding="utf-8")
    with pytest.raises(SystemExit, match="sender measures"):
        sp.load_levels(path)


def test_old_levels_are_refused_when_a_freshness_budget_is_requested(tmp_path):
    path = tmp_path / "levels.json"
    path.write_text(json.dumps({"metric": "sum_kpa", "levels": [1, 2]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="older"):
        sp.load_levels(path, max_age_s=-1.0)


def test_live_parser_exposes_strict_scene_and_freshness_guards():
    args = sp.parse_args([
        "--levels", "/tmp/levels.json",
        "--require-scene-match",
        "--max-level-age-minutes", "15",
    ])
    assert args.require_scene_match is True
    assert args.max_level_age_minutes == pytest.approx(15.0)


def test_live_parser_defaults_to_two_hour_freshness_budget():
    args = sp.parse_args(["--levels", "/tmp/levels.json"])
    assert args.max_level_age_minutes == pytest.approx(120.0)


def test_scene_gate_uses_componentwise_median_instead_of_one_outlier():
    reference = {"bright_fraction": 0.70, "bright_cx": 532.0, "bright_cy": 364.0}
    samples = [
        reference,
        {"bright_fraction": 0.64, "bright_cx": 499.0, "bright_cy": 367.0},
        {"bright_fraction": 0.67, "bright_cx": 511.0, "bright_cy": 369.0},
    ]

    assert sp.median_scene_fingerprint(samples) == {
        "bright_fraction": 0.67,
        "bright_cx": 511.0,
        "bright_cy": 367.0,
    }


def test_scene_gate_rejects_an_empty_sample_set():
    with pytest.raises(ValueError, match="at least one"):
        sp.median_scene_fingerprint([])


def test_live_parser_accepts_an_avi_recording_path():
    args = sp.parse_args([
        "--levels", "/tmp/levels.json",
        "--video-out", "/tmp/continuous.avi",
    ])
    assert args.video_out == Path("/tmp/continuous.avi")


def test_video_recording_is_not_offered_for_a_synthetic_stream():
    with pytest.raises(SystemExit):
        sp.parse_args(["--dry-run", "--video-out", "/tmp/continuous.avi"])


def test_live_panel_contains_both_views_and_annotations():
    resized = np.zeros((384, 480, 3), dtype=np.uint8)
    kpa = np.zeros((384, 480), dtype=np.float32)
    metrics = {
        "mean_kpa_in_contact": 0.0,
        "sum_kpa": 0.0,
        "contact_px": 0,
        "off_x": 0.0,
        "off_y": 0.0,
    }
    panel = sp.render_live_panel(resized, kpa, 0, 3, 0.0, metrics)
    assert panel.shape == (384, 960, 3)
    assert np.count_nonzero(panel) > 0


def test_shared_preview_writer_publishes_one_complete_frame(tmp_path):
    path = tmp_path / "pv-preview.mmap"
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape((4, 6, 3))
    writer = sp.SharedPreviewWriter(path)
    try:
        writer.publish(frame, observed_at_s=12.5)
        with path.open("rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            header = sp.PREVIEW_HEADER.unpack(mapped[:sp.PREVIEW_HEADER.size])
            pixels = bytes(mapped[sp.PREVIEW_HEADER_SIZE:])
            mapped.close()
    finally:
        writer.close()

    assert header == (sp.PREVIEW_MAGIC, 2, 12.5, 4, 6, 3, frame.nbytes)
    assert pixels == frame.tobytes()


def test_continuous_pressure_spans_the_measured_light_to_hard_support():
    levels = {
        "boundaries": [
            {"edge": 20.0, "gap": 12.0},
        ]
    }
    assert sp.continuous_pressure(14.0, levels) == 0.0
    assert sp.continuous_pressure(20.0, levels) == pytest.approx(0.5)
    assert sp.continuous_pressure(26.0, levels) == 1.0
    assert sp.continuous_pressure(100.0, levels) == 1.0


def test_continuous_pressure_uses_explicit_three_point_shadow_anchors():
    levels = {
        "continuous_anchors": [
            {"sum_kpa": 1530.0, "pressure_0_1": 0.0},
            {"sum_kpa": 20084.5, "pressure_0_1": 0.5},
            {"sum_kpa": 29339.25, "pressure_0_1": 1.0},
        ]
    }

    assert sp.continuous_pressure(0.0, levels) == 0.0
    assert sp.continuous_pressure(1530.0, levels) == 0.0
    assert sp.continuous_pressure(20084.5, levels) == 0.5
    assert sp.continuous_pressure(29339.25, levels) == 1.0
    assert sp.continuous_pressure(40000.0, levels) == 1.0


def test_continuous_pressure_rejects_unordered_explicit_anchors():
    levels = {
        "continuous_anchors": [
            {"sum_kpa": 20.0, "pressure_0_1": 0.0},
            {"sum_kpa": 10.0, "pressure_0_1": 1.0},
        ]
    }

    with pytest.raises(SystemExit, match="strictly increasing"):
        sp.continuous_pressure(15.0, levels)


def test_levels_out_without_a_session_is_a_usage_error():
    with pytest.raises(SystemExit):
        sp.parse_args(["--levels-out", "/tmp/a.json"])


def test_an_empty_session_says_so_instead_of_indexing_off_the_end(tmp_path):
    """A capture that died before recording used to surface as an IndexError deep
    in load_trials, pointing at the fitting step rather than the capture run."""
    session = tmp_path / "session"
    session.mkdir()
    (session / "capture.jsonl").touch()
    with pytest.raises(SystemExit, match="capture.jsonl is empty"):
        sp.fit_levels(session)


def test_a_session_that_was_never_run_says_so(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    with pytest.raises(SystemExit, match="no capture.jsonl"):
        sp.fit_levels(session)


def _capture(tmp_path, targets, trials):
    """trials: (trial_index, level, [contact_px per frame]) -> a capture.jsonl."""
    session = tmp_path / "session"
    session.mkdir()
    lines = [json.dumps({"row_type": "metadata", "targets_g": targets})]
    for trial, level, per_frame in trials:
        for hold, px in enumerate(per_frame):
            lines.append(json.dumps({
                "row_type": "sample", "trial_index": trial, "target_g": level,
                "hold_index": hold, "contact_px": px,
                "mean_kpa_in_contact": 0.0 if px == 0 else level * 10.0,
                "sum_kpa": 0.0 if px == 0 else level * 10.0 * px,
            }))
    (session / "capture.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session


def test_a_lift_that_breaks_the_median_drops_the_trial(tmp_path):
    """Six held frames of fifteen puts the median in the empty ones, reading 0 kPa
    for a press that happened -- session 06 failed the gate on two of these."""
    session = _capture(tmp_path, [0, 1], [
        (0, 1, [900] * 6 + [0] * 9),
    ])
    per_level, lifted = sp.trials_excluding_early_lifts(session, "sum_kpa")
    assert lifted == [(0, 1, 6, 15)]
    assert len(per_level[1]) == 0


def test_a_lift_the_median_survives_keeps_the_trial(tmp_path):
    """Nine of fifteen still summarises the press; dropping it would flatter the fit."""
    session = _capture(tmp_path, [0, 1], [
        (0, 1, [900] * 9 + [0] * 6),
    ])
    per_level, lifted = sp.trials_excluding_early_lifts(session, "sum_kpa")
    assert lifted == []
    assert per_level[1] == pytest.approx([9000.0])


def test_a_no_press_trial_whose_false_contact_faded_is_kept_as_evidence(tmp_path):
    """Session 06's no-press trials read a repeatable ~4 kPa blob. Dropping them
    would move the no-press boundary down onto readings it has to reject."""
    session = _capture(tmp_path, [0, 1], [
        (0, 0, [90] * 14 + [0]),
    ])
    per_level, lifted = sp.trials_excluding_early_lifts(session, "sum_kpa")
    assert lifted == []
    assert len(per_level[0]) == 1


# --- moved-rig guard ---------------------------------------------------------

def _scene(fraction, cx, cy):
    return {"bright_fraction": fraction, "bright_cx": cx, "bright_cy": cy}


def test_the_move_that_voided_session_06_is_caught():
    """The pad grew from 49% to 62% of the view and slid 78 px, and nothing noticed
    until two frames were put side by side."""
    complaint = sp.fingerprint_drift(_scene(0.623, 512, 349), _scene(0.492, 434, 345))
    assert complaint is not None
    assert "78 px" in complaint


def test_an_unmoved_rig_is_not_flagged():
    then = _scene(0.492, 434, 345)
    assert sp.fingerprint_drift(then, then) is None
    assert sp.fingerprint_drift(_scene(0.50, 445, 350), then) is None


def test_no_calibration_fingerprint_means_no_complaint():
    """Sessions recorded before the guard existed must still be usable."""
    assert sp.fingerprint_drift(_scene(0.9, 0, 0), {}) is None


def test_a_blank_view_does_not_divide_by_anything():
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    assert sp.scene_fingerprint(black)["bright_fraction"] >= 0.0
