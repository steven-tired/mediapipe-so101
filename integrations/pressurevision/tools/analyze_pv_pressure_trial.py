#!/usr/bin/env python3
"""Analyze one PressureVision sender + teleop shadow pair without hardware assumptions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median, quantiles

from lerobot_teleoperator_so101_webcam.grip.proposal import (
    GRIP_CLOSE_ALPHA,
    GRIP_OPEN_ALPHA,
    GRIP_OVERDRIVE,
    apply_pressure_overdrive,
)
from pressurevision_integration.pv_object_profile import (
    load_object_profile,
    object_profile_sha256,
)


def _float(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row: dict, key: str) -> int | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sender_cadence(rows: list[dict]) -> dict:
    timestamps = sorted(t for row in rows if (t := _float(row, "t")) is not None)
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not intervals:
        return {"rows": len(rows), "median_hz": 0.0, "p95_interval_s": None, "max_interval_s": None}
    p95 = quantiles(intervals, n=20, method="inclusive")[18] if len(intervals) >= 2 else intervals[0]
    return {
        "rows": len(rows),
        "median_hz": 1.0 / median(intervals),
        "p95_interval_s": p95,
        "max_interval_s": max(intervals),
    }


def reconstruct_legacy_actual(rows: list[dict]) -> tuple[list[float], int]:
    """Replay the frozen legacy fixed-overdrive + asymmetric EMA from sidecar rows."""
    previous: float | None = None
    predicted: list[float] = []
    mismatches = 0
    for row in rows:
        state = row.get("state", "")
        if state == "MIDDLE":
            previous = None
            continue
        if state != "MOVING":
            continue
        base = _float(row, "base_gripper_pos")
        observed = _float(row, "actual_gripper_pos")
        if base is None or observed is None:
            continue
        raw = apply_pressure_overdrive(base, GRIP_OVERDRIVE, None)
        actual = raw if previous is None else (
            (GRIP_CLOSE_ALPHA * raw + (1.0 - GRIP_CLOSE_ALPHA) * previous)
            if raw < previous
            else (GRIP_OPEN_ALPHA * raw + (1.0 - GRIP_OPEN_ALPHA) * previous)
        )
        previous = actual
        predicted.append(actual)
        if abs(actual - observed) > 1e-5:
            mismatches += 1
    return predicted, mismatches


def analyze_shadow(
    sender_rows: list[dict],
    shadow_rows: list[dict],
    *,
    profile_path: str | Path | None = None,
    require_protocol: bool = False,
) -> dict:
    reasons: list[str] = []
    cadence = sender_cadence(sender_rows)
    if cadence["rows"] < 2 or cadence["median_hz"] < 25.0:
        reasons.append("sender cadence below 25 Hz")
    if cadence["p95_interval_s"] is None or cadence["p95_interval_s"] > 0.10:
        reasons.append("sender p95 interval exceeds 100 ms")
    if cadence["max_interval_s"] is None or cadence["max_interval_s"] > 0.40:
        reasons.append("sender max interval exceeds 400 ms")

    _, legacy_mismatches = reconstruct_legacy_actual(shadow_rows)
    moving_rows = [row for row in shadow_rows if row.get("state") == "MOVING"]
    active_rows = [
        row for row in moving_rows
        if row.get("pressure_status") == "active"
        and _int(row, "pressure_level") in (1, 2)
    ]
    differing = [
        row for row in active_rows
        if _float(row, "proposed_gripper_pos") is not None
        and _float(row, "actual_gripper_pos") is not None
        and abs(_float(row, "proposed_gripper_pos") - _float(row, "actual_gripper_pos")) > 0.5
    ]
    if not active_rows:
        reasons.append("no active level-1/2 PV rows")
    if active_rows and not differing:
        reasons.append("PV proposal never diverged from legacy actual")

    fault_rows = [row for row in shadow_rows if row.get("fault_latched") == "true"]
    stale_rows = [
        row for row in shadow_rows
        if row.get("pressure_status") in {"pv_stale", "pv_time_skew", "pv_unavailable"}
    ]
    if fault_rows:
        reasons.append(f"pressure fault latch in {len(fault_rows)} rows")
    if stale_rows:
        reasons.append(f"stale/unavailable PV rows: {len(stale_rows)}")

    protocol_rows = [row for row in moving_rows if _int(row, "expected_level") is not None]
    protocol_mismatches = sum(
        _int(row, "pressure_level") != _int(row, "expected_level")
        for row in protocol_rows
        if _int(row, "pressure_level") is not None
    )
    if require_protocol and not protocol_rows:
        reasons.append("guided protocol metadata is missing")
    if protocol_rows and protocol_mismatches / len(protocol_rows) > 0.20:
        reasons.append("guided protocol level mismatch exceeds 20%")

    profile_meta = {}
    if profile_path is not None:
        profile = load_object_profile(profile_path)
        profile_hash = object_profile_sha256(profile_path)
        profile_meta = {"object_id": profile.object_id, "sha256": profile_hash}
        ids = {row.get("object_id") for row in shadow_rows if row.get("object_id")}
        hashes = {row.get("object_profile_sha256") for row in shadow_rows if row.get("object_profile_sha256")}
        if ids and ids != {profile.object_id}:
            reasons.append("shadow object_id does not match profile")
        if hashes and hashes != {profile_hash}:
            reasons.append("shadow profile hash does not match profile")
        if not ids or not hashes:
            reasons.append("shadow profile metadata is missing")

    return {
        "accepted": not reasons and legacy_mismatches == 0,
        "reasons": (["legacy actual differs from frozen replay"] if legacy_mismatches else []) + reasons,
        "sender": cadence,
        "shadow": {
            "rows": len(shadow_rows),
            "moving_rows": len(moving_rows),
            "active_rows": len(active_rows),
            "proposal_difference_rows": len(differing),
            "legacy_mismatches": legacy_mismatches,
            "fault_rows": len(fault_rows),
            "stale_rows": len(stale_rows),
            "protocol_rows": len(protocol_rows),
            "protocol_mismatches": protocol_mismatches,
        },
        "profile": profile_meta,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender-log", type=Path, required=True)
    parser.add_argument("--shadow-sidecar", type=Path, required=True)
    parser.add_argument("--object-profile", type=Path)
    parser.add_argument("--require-protocol", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_shadow(
        read_csv(args.sender_log),
        read_csv(args.shadow_sidecar),
        profile_path=args.object_profile,
        require_protocol=args.require_protocol,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
