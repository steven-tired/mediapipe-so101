import pytest

from pressurevision_integration.pv_trial_protocol import PVTrialProtocol


def test_default_sequence_repeats_open_light_open_hard():
    protocol = PVTrialProtocol(repetitions=2)
    protocol.start(10.0)
    assert protocol.expected(10.0)["trial_phase"] == "open"
    assert protocol.expected(12.1)["trial_phase"] == "light"
    assert protocol.expected(15.1)["trial_phase"] == "open"
    assert protocol.expected(17.1)["trial_phase"] == "hard"
    assert protocol.expected(20.1)["trial_index"] == 1
    assert protocol.expected(40.0) is None


def test_protocol_rejects_invalid_repetitions_and_duration():
    with pytest.raises(ValueError):
        PVTrialProtocol(repetitions=0)
