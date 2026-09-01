#!/usr/bin/env python3
"""Run the one-shot fixed-setup intent test with frozen candidates.

The primary candidate is the sum_kpa threshold baseline selected on fixed_04;
the already-frozen FPN ordinal checkpoint is reported as a secondary comparison.
This script fits nothing, changes no threshold, and refuses a test session that
appears in either artifact's train or validation split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("<workspace>")
LABELS = ("light", "medium", "hard")

sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_pv_intent_baseline import load_trials as audit_trials  # noqa: E402
from pressurevision_probe import load_model  # noqa: E402
from train_pv_fpn_ordinal import FPNOrdinalModel, evaluate_records  # noqa: E402
from train_pv_ordinal_head import (  # noqa: E402
    FEATURE_DIM,
    OrdinalPressureHead,
    classification_metrics,
    collect_records,
    resolve_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=WORKSPACE / "pressurevision")
    parser.add_argument("--test-session", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--fpn-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--evaluation-role",
        choices=("final_test", "blind_confirmation"),
        default="final_test",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def ensure_unseen_test_session(
    test_name: str,
    baseline_report: dict,
    fpn_checkpoint: dict,
) -> None:
    baseline_sessions = {
        audit["session"]
        for key in ("train_sessions", "eval_sessions")
        for audit in baseline_report.get(key, [])
    }
    neural_sessions = set(fpn_checkpoint.get("train_sessions", []))
    neural_sessions.add(fpn_checkpoint.get("validation_session"))
    if test_name in baseline_sessions or test_name in neural_sessions:
        raise SystemExit(f"{test_name} is not an unseen final-test session")


def scalar_predictions(
    labels: np.ndarray,
    values: np.ndarray,
    thresholds: tuple[float, float],
) -> dict:
    lower, upper = thresholds
    predicted = np.where(values < lower, 0, np.where(values < upper, 1, 2))
    return classification_metrics(labels, predicted)


def evaluate_scalar_baseline(session: Path, thresholds: tuple[float, float]) -> dict:
    with (session / "capture.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows = [row for row in rows if row.get("valid_for_ordinal_training")]
    frame_labels = np.asarray([int(row["ordinal_target"]) for row in rows])
    frame_values = np.asarray([float(row["sum_kpa"]) for row in rows])
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["trial_index"])].append(row)
    trial_labels = []
    trial_values = []
    for trial_rows in grouped.values():
        labels = {int(row["ordinal_target"]) for row in trial_rows}
        if len(labels) != 1:
            raise SystemExit("a final-test trial contains multiple labels")
        trial_labels.append(labels.pop())
        trial_values.append(statistics.median(float(row["sum_kpa"]) for row in trial_rows))
    return {
        "thresholds": {"light_medium": thresholds[0], "medium_hard": thresholds[1]},
        "frame": scalar_predictions(frame_labels, frame_values, thresholds),
        "trial": scalar_predictions(
            np.asarray(trial_labels), np.asarray(trial_values), thresholds
        ),
    }


def load_fpn_model(repo: Path, checkpoint: dict, device: str) -> FPNOrdinalModel:
    pressurevision, _config = load_model(repo, device)
    head = OrdinalPressureHead(
        input_dim=FEATURE_DIM,
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    pressurevision.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    head.load_state_dict(checkpoint["ordinal_head_state_dict"])
    model = FPNOrdinalModel(pressurevision, head).to(device)
    model.eval()
    return model


def run(args) -> dict:
    device = resolve_device(args.device)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    fpn_checkpoint = torch.load(args.fpn_checkpoint, map_location="cpu", weights_only=False)
    ensure_unseen_test_session(args.test_session.name, baseline_report, fpn_checkpoint)

    _trials, audit = audit_trials(args.test_session, "sum_kpa")
    records, record_audits = collect_records([args.test_session])
    thresholds = (
        float(baseline_report["thresholds"]["light_medium"]),
        float(baseline_report["thresholds"]["medium_hard"]),
    )
    baseline = evaluate_scalar_baseline(args.test_session, thresholds)
    model = load_fpn_model(args.repo, fpn_checkpoint, device)
    fpn = evaluate_records(model, records, batch_size=args.batch_size, device=device)

    status = (
        "blind_confirmation_complete_no_further_model_selection_authorized"
        if args.evaluation_role == "blind_confirmation"
        else "final_test_complete_no_further_model_selection_authorized"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "evaluation_role": args.evaluation_role,
        "task": "ordered_operator_grip_intent_not_force_measurement",
        "contact_policy": "MediaPipe owns contact",
        "test_session": audit,
        "test_session_record_audit": record_audits[0],
        "primary_candidate_selected_before_test": "sum_kpa_threshold_baseline",
        "selection_evidence": {
            "fixed_04_trial_accuracy": {
                "sum_kpa_threshold_baseline": 1.0,
                "fpn_ordinal": 1.0,
            },
            "fixed_04_frame_accuracy": {
                "sum_kpa_threshold_baseline": 0.9920987654320987,
                "fpn_ordinal": 0.9733333333333334,
            },
            "tie_break": "higher frame accuracy and lower complexity",
        },
        "artifacts_frozen_before_test": {
            "baseline_report": str(args.baseline_report),
            "fpn_checkpoint": str(args.fpn_checkpoint),
        },
        "sum_kpa_threshold_baseline": baseline,
        "fpn_ordinal_secondary_comparison": fpn,
        "device": device,
        "prohibitions": [
            "do not tune thresholds, architecture, epochs, or abstention on this test result",
            "do not describe either output as measured physical pressure",
            "do not treat this robot-free result as control authorization",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "FINAL primary sum_kpa trial/frame BA="
        f"{baseline['trial']['balanced_accuracy']:.3f}/"
        f"{baseline['frame']['balanced_accuracy']:.3f}"
    )
    print(
        "FINAL secondary FPN trial/frame BA="
        f"{fpn['trial']['balanced_accuracy']:.3f}/"
        f"{fpn['frame']['balanced_accuracy']:.3f}"
    )
    print(f"wrote {args.out}")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
