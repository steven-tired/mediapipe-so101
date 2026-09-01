"""Decision logic for the PressureVision shadow source: gating, not numpy or sockets."""

from __future__ import annotations

import pytest

from pressurevision_integration import pv_pressure
from pressurevision_integration.pv_pressure import (
    SHARED_PV_TEST_PACKET,
    PressureVisionConfig,
    PressureVisionPacket,
    PressureVisionSource,
    ABSTAIN_LEVEL,
    area_quality,
    decode_pv_packet,
    encode_pv_packet,
    offset_quality,
)

NOW = 1_000.0


def _packet(**overrides) -> PressureVisionPacket:
    fields = dict(
        observed_at_s=NOW,
        sent_at_s=NOW,
        sequence=7,
        mean_kpa_in_contact=7.0,
        max_kpa=14.0,
        contact_px=900,
        off_x=0.0,
        off_y=0.0,
        sum_kpa=3712.0,
        crop_w=522,
        crop_h=416,
        pressure_0_1=1.0,
        level=2,
        n_levels=3,
    )
    fields.update(overrides)
    return PressureVisionPacket(**fields)


class _StubSource:
    """Replays packets; repeats the last one so a test can call update() repeatedly."""

    def __init__(self, *packets: PressureVisionPacket):
        self._packets = list(packets)

    def read(self) -> PressureVisionPacket:
        if not self._packets:
            raise RuntimeError("no packets")
        if len(self._packets) > 1:
            return self._packets.pop(0)
        return self._packets[0]


class _DeadSource:
    def read(self):
        raise RuntimeError("sender gone")


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(pv_pressure.time, "time", lambda: NOW)


def _source(*packets, **cfg_overrides) -> PressureVisionSource:
    return PressureVisionSource(
        source=_StubSource(*packets), config=PressureVisionConfig(**cfg_overrides)
    )


# --- wire format -------------------------------------------------------------

def test_shared_test_vector_decodes_to_known_values():
    """Pins the format the sender in hand-pressure/ encodes against."""
    packet = decode_pv_packet(SHARED_PV_TEST_PACKET.encode("utf-8"))
    assert packet.sequence == 42
    assert packet.observed_at_s == pytest.approx(1700000000.1)
    assert packet.sent_at_s == pytest.approx(1700000000.123456)
    assert packet.mean_kpa_in_contact == pytest.approx(7.25)
    assert packet.sum_kpa == pytest.approx(3712.0)
    assert packet.contact_px == 512
    assert packet.off_x == pytest.approx(-0.25)
    assert packet.off_y == pytest.approx(0.5)
    assert packet.pressure_0_1 == pytest.approx(0.625)
    assert (packet.level, packet.n_levels) == (1, 3)
    assert encode_pv_packet(packet) == (SHARED_PV_TEST_PACKET + "\n").encode("utf-8")


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"5,2,3,4\n", "expected 15 fields"),
        (b"99,0,0,0,0,0,0,0,0,0,10,10,0,0,3\n", "unsupported schema version"),
        (b"5,-1,0,0,0,0,0,0,0,0,10,10,0,0,3\n", "sequence must be non-negative"),
        (b"5,0,0,0,0,0,0,-5,0,0,10,10,0,0,3\n", "contact_px must be non-negative"),
        (b"5,0,nan,0,0,0,0,0,0,0,10,10,0,0,3\n", "non-finite"),
        (b"5,0,2,1,0,0,0,0,0,0,10,10,0,0,3\n", "sent_t must not precede"),
        (b"5,0,0,0,0,0,0,0,0,0,10,10,0,7,3\n", "outside 0..2"),
        (b"5,0,0,0,0,0,0,0,0,0,10,10,0,0,1\n", "n_levels must be at least 2"),
        (b"5,0,0,0,0,0,0,0,0,0,10,10,2,0,3\n", "pressure_0_1 must be"),
    ],
)
def test_decode_rejects_malformed(raw, match):
    with pytest.raises(ValueError, match=match):
        decode_pv_packet(raw)


# --- quality is trust, never magnitude ---------------------------------------

def test_area_quality_zeroes_outside_the_sanity_band():
    cfg = PressureVisionConfig(min_contact_area_px=40, max_contact_area_px=12000)
    assert area_quality(900, cfg) == 1.0
    assert area_quality(0, cfg) == 0.0
    assert area_quality(20, cfg) == pytest.approx(0.5)
    # A flat hand covering twice the ceiling is not a pinch contact.
    assert area_quality(24000, cfg) == 0.0


def test_the_measured_false_contact_is_rejected_on_area(frozen_clock):
    """Session 06's no-press trials made a repeatable 82-93 px blob at ~4 kPa; its
    real presses ran 464+ px. The default floor has to sit in that gap."""
    cfg = PressureVisionConfig()
    assert area_quality(93, cfg) < 0.5      # the false contact
    assert area_quality(464, cfg) == 1.0    # the lightest real press

    src = _source(_packet(contact_px=93, level=1))
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.quality < 0.5


def test_offset_quality_falls_off_away_from_the_crop_centre():
    cfg = PressureVisionConfig(trust_full_offset=0.5, max_trust_offset=0.8)
    assert offset_quality(0.0, 0.0, cfg) == 1.0
    assert offset_quality(0.5, 0.0, cfg) == 1.0
    assert offset_quality(-0.65, 0.0, cfg) == pytest.approx(0.5)
    assert offset_quality(0.0, 0.9, cfg) == 0.0
    # Radial: the same distance taken diagonally scores the same.
    assert offset_quality(0.65, 0.0, cfg) == pytest.approx(offset_quality(0.0, 0.65, cfg))


def test_a_light_press_is_still_high_quality(frozen_clock):
    """quality answers 'is this reading trustworthy', never 'is it pressing hard'."""
    src = _source(_packet(level=1, n_levels=3, pressure_0_1=0.25))
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.status == "active"
    assert reading.quality == 1.0
    assert reading.pressure_0_1 == pytest.approx(0.25)


def test_an_abstaining_sender_keeps_the_continuous_signal(frozen_clock):
    src = _source(_packet(level=ABSTAIN_LEVEL, pressure_0_1=0.625))
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.status == "pv_abstain_continuous"
    assert reading.active is True
    assert reading.quality == 1.0
    assert reading.pressure_0_1 == pytest.approx(0.625)


def test_level_zero_does_not_block_a_contact_gated_continuous_value(frozen_clock):
    """The diagnostic band must not override right-pinch intent plus pad contact."""
    src = _source(_packet(contact_px=900, level=0, pressure_0_1=0.2))
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert (reading.status, reading.active, reading.quality) == ("active", True, 1.0)
    assert reading.pressure_0_1 == pytest.approx(0.2)


# --- gating ------------------------------------------------------------------

def test_pinch_latches_contact_with_hysteresis(frozen_clock):
    src = _source(_packet(), near_contact_pinch=0.045, exit_contact_pinch=0.055)
    assert src.update(landmarks=None, pinch=0.04, enabled=True).status == "active"
    # Inside the deadband the latch holds.
    assert src.update(landmarks=None, pinch=0.05, enabled=True).status == "active"
    assert src.update(landmarks=None, pinch=0.06, enabled=True).status == "baseline"
    assert src.update(landmarks=None, pinch=0.05, enabled=True).status == "baseline"


def test_default_gate_accepts_attempt_11_recontact(frozen_clock):
    """Reproduces session-WIXMyE attempt 11 after its first PV lock."""
    src = _source(
        _packet(pressure_0_1=0.5),
        _packet(contact_px=0, pressure_0_1=0.0),
        _packet(pressure_0_1=0.9),
    )

    first = src.update(landmarks=None, pinch=0.03, enabled=True)
    released_pad = src.update(landmarks=None, pinch=0.05744, enabled=True)
    recontact = src.update(landmarks=None, pinch=0.045018, enabled=True)

    assert (first.active, first.pressure_0_1) == (True, pytest.approx(0.5))
    assert (released_pad.status, released_pad.active) == ("baseline", False)
    assert (recontact.active, recontact.pressure_0_1) == (True, pytest.approx(0.9))


def test_no_contact_is_a_confident_baseline_not_low_quality(frozen_clock):
    """contact_px == 0 must arm the proposal state machine, not latch a fault."""
    src = _source(_packet(contact_px=0))
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert (reading.status, reading.active, reading.quality) == ("baseline", False, 1.0)
    assert reading.available is True


def test_no_contact_returns_baseline_without_waiting_for_level_debounce(frozen_clock):
    src = _source(
        _packet(contact_px=900, level=2),
        _packet(contact_px=0, level=2, pressure_0_1=0.9),
        level_hold_frames=3,
    )
    assert src.update(landmarks=None, pinch=0.03, enabled=True).active is True

    reading = src.update(landmarks=None, pinch=0.03, enabled=True)

    assert reading.level == 2  # diagnostic level is still debounced
    assert (reading.status, reading.active, reading.quality) == ("baseline", False, 1.0)
    assert reading.pressure_0_1 == 0.0


def test_clutched_teleop_reports_baseline_and_drops_the_latch(frozen_clock):
    src = _source(_packet())
    assert src.update(landmarks=None, pinch=0.03, enabled=True).active is True
    assert src.update(landmarks=None, pinch=0.03, enabled=False).status == "baseline"
    # Re-enabling must require a fresh pinch crossing rather than resuming the grasp.
    assert src.update(landmarks=None, pinch=0.06, enabled=True).status == "baseline"


def test_sender_clock_ahead_is_rejected_rather_than_treated_as_fresh(monkeypatch):
    src = _source(_packet())
    monkeypatch.setattr(pv_pressure.time, "time", lambda: NOW - 1.0)
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.status == "pv_time_skew"
    assert reading.available is False


def test_missing_sender_degrades_instead_of_raising(frozen_clock):
    src = PressureVisionSource(source=_DeadSource())
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.status == "pv_unavailable"
    assert reading.available is False
    assert reading.roi_mode == "pv"


def test_reset_drops_the_latch(frozen_clock):
    src = _source(_packet())
    assert src.update(landmarks=None, pinch=0.03, enabled=True).active is True
    src.reset()
    assert src.update(landmarks=None, pinch=0.06, enabled=True).status == "baseline"


# --- the low-quality path reaches the existing fallback ----------------------

def test_a_single_stray_level_does_not_move_the_diagnostic_band(frozen_clock):
    src = _source(_packet(level=2), level_hold_frames=3)
    assert src.update(landmarks=None, pinch=0.03, enabled=True).level == 2
    src.source._packets = [_packet(level=1, pressure_0_1=0.4)]
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.level == 2
    assert reading.pressure_0_1 == pytest.approx(0.4)


def test_a_level_that_repeats_is_adopted(frozen_clock):
    src = _source(_packet(level=2), level_hold_frames=3)
    src.update(landmarks=None, pinch=0.03, enabled=True)
    src.source._packets = [_packet(level=1, pressure_0_1=0.4)]
    seen = [src.update(landmarks=None, pinch=0.03, enabled=True).level for _ in range(3)]
    assert seen == [2, 2, 1]


def test_flicker_between_two_levels_adopts_neither(frozen_clock):
    """Alternating packets never accumulate a run, so the held band stays put."""
    src = _source(_packet(level=2), level_hold_frames=3)
    src.update(landmarks=None, pinch=0.03, enabled=True)
    for level in (1, 0, 1, 0, 1, 0):
        src.source._packets = [_packet(level=level)]
        assert src.update(landmarks=None, pinch=0.03, enabled=True).level == 2


def test_reset_forgets_the_held_level(frozen_clock):
    src = _source(_packet(level=2), level_hold_frames=3)
    src.update(landmarks=None, pinch=0.03, enabled=True)
    src.reset()
    src.source._packets = [_packet(level=1, pressure_0_1=0.4)]
    reading = src.update(landmarks=None, pinch=0.03, enabled=True)
    assert reading.level == 1
    assert reading.pressure_0_1 == pytest.approx(0.4)


# Removed with this migration: test_stale_packet_is_unavailable,
# test_low_quality_is_available_but_below_the_proposal_gate, and
# test_shadow_proposal_diverges_from_the_legacy_command. All three exercised
# `ir_pressure_proposal`, the private IR shadow-proposal path, which is not part
# of the public PressureVision integration. They migrate to ir-camera-force
# together with that module.
