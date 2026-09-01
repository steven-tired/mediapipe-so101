import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import aim_pad_camera as ap


FRAME = (720, 1280, 3)


def test_moving_the_crop_keeps_its_size():
    crop = (100, 100, 400, 340)
    moved = ap.apply_key(83, crop, FRAME)          # right arrow

    assert moved[2] - moved[0] == crop[2] - crop[0]
    assert moved[3] - moved[1] == crop[3] - crop[1]
    assert moved[0] > crop[0]


def test_an_unhandled_key_leaves_the_crop_alone():
    """The caller treats every other key as its own, so this must not move."""
    crop = (100, 100, 400, 340)
    assert ap.apply_key(ord("z"), crop, FRAME) == crop


def test_the_crop_cannot_be_pushed_off_the_frame():
    crop = (0, 0, 300, 240)
    for _ in range(50):
        crop = ap.apply_key(81, crop, FRAME)       # left, repeatedly

    assert crop[0] >= 0
    assert crop[2] - crop[0] >= ap.MIN_SIDE_PX


def test_the_crop_cannot_be_pushed_past_the_far_edge():
    crop = (1000, 600, 1280, 720)
    for _ in range(50):
        crop = ap.apply_key(83, crop, FRAME)       # right, repeatedly

    assert crop[2] <= FRAME[1]
    assert crop[3] <= FRAME[0]


def test_shrinking_stops_at_a_usable_size():
    crop = (100, 100, 400, 340)
    for _ in range(100):
        crop = ap.apply_key(ord("-"), crop, FRAME)

    assert crop[2] - crop[0] >= ap.MIN_SIDE_PX
    assert crop[3] - crop[1] >= ap.MIN_SIDE_PX


def test_width_and_height_are_adjustable_independently():
    crop = (100, 100, 400, 340)
    wider = ap.apply_key(ord("d"), crop, FRAME)
    taller = ap.apply_key(ord("s"), crop, FRAME)

    assert wider[2] > crop[2] and wider[3] == crop[3]
    assert taller[3] > crop[3] and taller[2] == crop[2]


def test_report_shows_whether_the_crop_is_being_upsampled():
    """Below 1.0 the network is being fed interpolation rather than detail."""
    small = ap.crop_report((0, 0, 240, 192), FRAME)
    large = ap.crop_report((0, 0, 960, 768), FRAME)

    assert small["source_px_per_net_px"][0] < 1.0
    assert large["source_px_per_net_px"][0] > 1.0


def test_report_does_not_impose_an_aspect_ratio():
    """The paper's own crops were 1.77 squashed into 480x384, so a crop that
    hugs a wide pad is correct and must not be reported as wrong."""
    report = ap.crop_report((235, 35, 915, 385), FRAME)

    assert report["aspect"] == pytest.approx(1.94, abs=0.01)
    assert "aspect" in report            # reported, not enforced


def test_moving_against_a_wall_stops_rather_than_shrinking():
    """Regression: clamping the resulting box let the near edge stop at zero
    while the far edge kept travelling, so holding an arrow shaved the crop."""
    crop = (0, 0, 300, 240)
    for _ in range(50):
        crop = ap.apply_key(81, crop, FRAME)

    assert crop == (0, 0, 300, 240)


def test_a_saved_crop_round_trips_through_the_report():
    """Resuming reads back the same key the report writes, so a session that
    ended without pressing return still starts where it left off."""
    report = ap.crop_report((320, 80, 960, 440), FRAME)
    resumed = ap.clamp_crop(report["crop"], FRAME)

    assert resumed == (320, 80, 960, 440)


def test_the_default_crop_differs_from_a_working_one():
    """Why losing the crop looked like the model breaking: the centred default
    sits 100 px below the crop that framed the pad, so it reads zero."""
    h, w = FRAME[0], FRAME[1]
    default = ap.clamp_crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4), FRAME)

    assert default == (320, 180, 960, 540)
    assert default != (320, 80, 960, 440)


# --- framing match ------------------------------------------------------------

def _scene(fraction, cx, cy):
    return {"bright_fraction": fraction, "bright_cx": cx, "bright_cy": cy}


def test_matched_framing_says_so():
    then = _scene(0.492, 434, 345)
    assert ap.framing_advice(then, then)[-1] == "MATCHED"


def test_a_pad_filling_more_of_the_view_means_pull_back():
    """Tightening the framing 10% flipped held light presses into the hard band."""
    advice = ap.framing_advice(_scene(0.67, 434, 345), _scene(0.492, 434, 345))
    assert advice[-1] == "PULL BACK"
    assert "+18 pts" in advice[0]


def test_a_centre_that_slid_right_means_slide_left():
    advice = ap.framing_advice(_scene(0.492, 512, 345), _scene(0.492, 434, 345))
    assert advice[-1] == "SLIDE LEFT"


def test_both_axes_are_reported_together():
    advice = ap.framing_advice(_scene(0.67, 512, 420), _scene(0.492, 434, 345))
    assert advice[-1] == "PULL BACK  SLIDE LEFT  SLIDE UP"


def test_small_wobble_is_not_worth_chasing():
    advice = ap.framing_advice(_scene(0.50, 445, 350), _scene(0.492, 434, 345))
    assert advice[-1] == "MATCHED"


# --- crop guidance ------------------------------------------------------------

def _pad_frame(pad_box, shape=(720, 1280)):
    """A dark frame with one bright rectangle standing in for the sheet."""
    frame = np.full((*shape, 3), 40, np.uint8)
    x0, y0, x1, y1 = pad_box
    frame[y0:y1, x0:x1] = 240
    return frame


def test_the_pad_is_found_where_it_was_drawn():
    assert ap.detect_pad(_pad_frame((200, 100, 900, 600))) == (200, 100, 900, 600)


def test_a_blank_view_finds_no_pad():
    assert ap.detect_pad(np.full((720, 1280, 3), 40, np.uint8)) is None


def test_a_crop_with_no_pad_edge_in_it_is_too_tight():
    """The framing that broke: all paper, no border. Same press, different band."""
    frame = _pad_frame((200, 100, 900, 600))
    fill, verdict = ap.pad_fill_verdict(frame, (300, 200, 800, 500))
    assert fill == pytest.approx(1.0)
    assert verdict.startswith("TOO TIGHT")


def test_a_crop_holding_the_pad_and_a_border_is_good():
    frame = _pad_frame((200, 100, 900, 600))
    fill, verdict = ap.pad_fill_verdict(frame, ap.suggested_crop((200, 100, 900, 600), frame.shape))
    assert ap.PAD_FILL_MIN <= fill <= ap.PAD_FILL_MAX
    assert verdict == "GOOD"


def test_a_crop_mostly_off_the_pad_is_too_loose():
    frame = _pad_frame((200, 100, 400, 300))
    _, verdict = ap.pad_fill_verdict(frame, (0, 0, 1280, 720))
    assert verdict.startswith("TOO LOOSE")


def test_the_suggestion_stays_inside_the_frame():
    frame = _pad_frame((0, 0, 1279, 719))
    x0, y0, x1, y1 = ap.suggested_crop((0, 0, 1279, 719), frame.shape)
    assert (x0, y0) >= (0, 0) and x1 <= 1280 and y1 <= 720
