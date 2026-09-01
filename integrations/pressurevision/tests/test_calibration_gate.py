import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calibration_gate as cg


def _level(centre, spread, n=4):
    """n presses centred on `centre`, evenly spread by +/- spread."""
    if n == 1:
        return [centre]
    step = 2 * spread / (n - 1)
    return [centre - spread + i * step for i in range(n)]


def test_well_separated_levels_are_accepted():
    verdict = cg.check({100: _level(4.0, 0.3), 330: _level(12.0, 0.3)})

    assert verdict["accepted"] is True
    assert verdict["reasons"] == []
    assert verdict["boundaries"][0]["dprime"] > cg.MIN_DPRIME


def test_levels_too_close_are_refused_with_the_reason():
    verdict = cg.check({100: _level(4.0, 2.0), 330: _level(6.0, 2.0)})

    assert verdict["accepted"] is False
    assert "d'" in verdict["reasons"][0]
    assert "press further apart" in verdict["reasons"][0]


def test_pressing_the_levels_backwards_is_caught_before_the_dprime_test():
    """A negative gap means the operator pressed harder on the light level;
    reporting that as 'too close' would send them off tuning the wrong thing."""
    verdict = cg.check({100: _level(12.0, 0.3), 330: _level(4.0, 0.3)})

    assert verdict["accepted"] is False
    assert "did not read higher" in verdict["reasons"][0]
    assert not any("d'" in reason for reason in verdict["reasons"])


def test_too_few_presses_is_refused_because_d_prime_is_not_estimable():
    verdict = cg.check({100: [4.0], 330: [12.0]})

    assert verdict["accepted"] is False
    assert "no spread to judge" in verdict["reasons"][0]


def test_a_single_level_cannot_be_calibrated():
    assert cg.check({100: _level(4.0, 0.3)})["accepted"] is False


def test_identical_presses_do_not_divide_by_zero():
    verdict = cg.check({100: [4.0] * 4, 330: [12.0] * 4})

    assert verdict["boundaries"][0]["dprime"] == float("inf")
    assert verdict["accepted"] is True


def test_three_levels_are_each_checked():
    verdict = cg.check({
        0: _level(0.5, 0.1),
        100: _level(4.0, 0.3),
        330: _level(4.4, 0.3),      # too close to the middle level
    })

    assert verdict["accepted"] is False
    assert len(verdict["boundaries"]) == 2
    assert any("100 to 330" in reason for reason in verdict["reasons"])


@pytest.mark.parametrize(
    "session,metric,dprime,accepted",
    [
        # Measured on the captured sessions; the gate is calibrated to sit
        # between session 04's geometry and the sittings that work.
        ("pv_labelled_04", "mean_kpa_in_contact", 1.45, False),
        ("pv_labelled_04", "contact_px", 0.84, False),
        ("pv_labelled_03", "mean_kpa_in_contact", 3.11, True),
        ("pv_labelled_05", "mean_kpa_in_contact", 5.47, True),
        # contact_px is the metric §3.12 shows failing over a long hold, and
        # the gate independently refuses it on the marginal sittings.
        ("pv_labelled_03", "contact_px", 1.64, False),
    ],
)
def test_gate_threshold_matches_the_captured_sessions(session, metric, dprime, accepted):
    assert (dprime >= cg.MIN_DPRIME) is accepted
