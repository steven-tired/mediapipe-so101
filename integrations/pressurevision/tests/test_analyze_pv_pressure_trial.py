import csv
import json

import pytest

from analyze_pv_pressure_trial import analyze_shadow
from pressurevision_integration.pv_object_profile import (
    PressureVisionObjectProfile,
    save_object_profile,
)


def _sender_rows(count=31):
    return [{"t": str(index / 30.0)} for index in range(count)]


def _shadow_rows(*, stale=False, metadata=True):
    rows = [
        {"state": "MIDDLE", "actual_gripper_pos": "50"},
        {
            "state": "MOVING", "base_gripper_pos": "60", "actual_gripper_pos": "42",
            "proposed_gripper_pos": "52", "pressure_status": "baseline",
            "pressure_level": "0", "expected_level": "0",
        },
        {
            "state": "MOVING", "base_gripper_pos": "60", "actual_gripper_pos": "42",
            "proposed_gripper_pos": "50", "pressure_status": "active",
            "pressure_level": "1", "expected_level": "1",
        },
        {
            "state": "MOVING", "base_gripper_pos": "60", "actual_gripper_pos": "42",
            "proposed_gripper_pos": "48", "pressure_status": "active",
            "pressure_level": "2", "expected_level": "2",
        },
    ]
    if stale:
        rows.append({"state": "MOVING", "pressure_status": "pv_stale", "fault_latched": "true"})
    if metadata:
        for row in rows:
            row.update({"object_id": "rigid_block", "object_profile_sha256": "hash"})
    for row in rows:
        row.setdefault("fault_latched", "false")
    return rows


def test_shadow_gate_accepts_cadence_legacy_replay_and_proposal_divergence(tmp_path):
    profile_path = tmp_path / "profile.json"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block", "arm", 95, 30, 20),
    )
    # The fixture deliberately uses the real file hash in metadata.
    from pressurevision_integration.pv_object_profile import object_profile_sha256
    rows = _shadow_rows()
    for row in rows:
        row["object_profile_sha256"] = object_profile_sha256(profile_path)
    report = analyze_shadow(_sender_rows(), rows, profile_path=profile_path, require_protocol=True)
    assert report["accepted"]
    assert report["shadow"]["legacy_mismatches"] == 0
    assert report["shadow"]["proposal_difference_rows"] == 2


def test_shadow_gate_rejects_fault_stale_and_missing_metadata(tmp_path):
    profile_path = tmp_path / "profile.json"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block", "arm", 95, 30, 20),
    )
    report = analyze_shadow(
        _sender_rows(), _shadow_rows(stale=True, metadata=False), profile_path=profile_path,
        require_protocol=True,
    )
    assert not report["accepted"]
    assert any("stale" in reason for reason in report["reasons"])
    assert any("metadata" in reason for reason in report["reasons"])
