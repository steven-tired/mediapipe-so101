import json

import pytest

from pressurevision_integration.pv_object_profile import (
    PressureVisionObjectProfile,
    load_object_profile,
    object_profile_sha256,
    save_object_profile,
)


def _profile():
    return PressureVisionObjectProfile(
        object_id="hard_plastic_block",
        arm_id="so101_follower_1",
        open_pos=95.0,
        light_pos=28.0,
        hard_pos=26.0,
    )


def test_profile_maps_only_wire_levels_zero_one_two():
    profile = _profile()
    assert profile.target_for_level(0, 3) == 95.0
    assert profile.target_for_level(1, 3) == 28.0
    assert profile.target_for_level(2, 3) == 26.0
    assert profile.target_for_level(-1, 3) is None


def test_profile_interpolates_continuous_pressure_between_light_and_hard():
    profile = _profile()
    assert profile.target_for_pressure(0.0) == 28.0
    assert profile.target_for_pressure(0.5) == 27.0
    assert profile.target_for_pressure(1.0) == 26.0


def test_profile_rejects_wrong_order_but_accepts_short_rigid_range():
    with pytest.raises(ValueError, match="hard_pos < light_pos"):
        PressureVisionObjectProfile("x", "arm", 95, 26, 28)
    profile = PressureVisionObjectProfile("x", "arm", 95, 28, 27.5)
    assert profile.target_for_pressure(1.0) == 27.5


def test_profile_round_trips_and_hashes_exact_file(tmp_path):
    path = tmp_path / "object.json"
    save_object_profile(path, _profile())
    loaded = load_object_profile(path)
    assert loaded == _profile()
    assert object_profile_sha256(path) == object_profile_sha256(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["object_id"] == "hard_plastic_block"
    assert data["control_mode"] == "hard_profile"


def test_profile_requires_three_wire_levels_for_pressure_grades():
    with pytest.raises(ValueError, match="n_levels=3"):
        _profile().target_for_level(1, 2)
