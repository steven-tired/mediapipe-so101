#!/usr/bin/env python3
"""Fit a trial-level sum_kpa ordinal baseline and score held-out sessions.

This is the first, non-neural baseline for the fixed-setup PressureVision
intent dataset.  Forty-five correlated frames from one steady hold collapse to
one median value before thresholds are fitted or scored.  Thresholds are fitted
from ``--train-session`` data only; ``--eval-session`` labels never influence
them.  Contact/no-contact is outside this task and remains MediaPipe's job.

Read-only with respect to cameras and robots.  ``--out`` writes only a derived
JSON report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


LABELS = ("light", "medium", "hard")
METRICS = ("sum_kpa", "contact_px", "mean_kpa_in_contact", "max_kpa")
COMPATIBILITY_FIELDS = (
    "experiment_identity",
    "setup_id",
    "operator_id",
    "crop",
    "surface",
    "target_zone_model_px",
    "network_input_size",
    "pixel_format",
    "preprocess",
    "pressurevision_checkpoint_sha256",
)


@dataclass(frozen=True)
class IntentTrial:
    session: str
    trial_index: int
    block: int
    label: int
    value: float
    frames: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-session",
        type=Path,
        action="append",
        required=True,
        help="Session used to fit thresholds; repeat for multiple sessions.",
    )
    parser.add_argument(
        "--eval-session",
        type=Path,
        action="append",
        required=True,
        help="Untouched held-out session; repeat to report each separately.",
    )
    parser.add_argument("--metric", choices=METRICS, default="sum_kpa")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _read_session(session: Path) -> tuple[list[dict], dict, dict]:
    capture = session / "capture.jsonl"
    manifest_path = session / "manifest.json"
    if not capture.is_file() or not manifest_path.is_file():
        raise SystemExit(f"{session}: capture.jsonl and manifest.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit(f"{session}: session status is not complete")
    digest = sha256(capture.read_bytes()).hexdigest()
    if digest != manifest.get("capture_jsonl_sha256"):
        raise SystemExit(f"{session}: capture.jsonl checksum does not match manifest")
    rows = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("row_type") != "metadata":
        raise SystemExit(f"{session}: first JSONL row must be metadata")
    return rows, rows[0], manifest


def load_trials(session: Path, metric: str) -> tuple[list[IntentTrial], dict]:
    """Load one median metric value per accepted steady trial."""
    rows, metadata, manifest = _read_session(session)
    grouped: dict[tuple[int, int, str], list[float]] = {}
    frame_counts = {label: 0 for label in LABELS}
    for row in rows[1:]:
        if not row.get("valid_for_ordinal_training"):
            continue
        label = row.get("target_label")
        if (
            row.get("row_type") != "sample"
            or row.get("phase") != "steady"
            or row.get("trial_status") != "complete"
            or label not in LABELS
            or row.get("ordinal_target") != LABELS.index(label)
        ):
            raise SystemExit(
                f"{session}: trial {row.get('trial_index')} has an invalid training row"
            )
        if metric not in row:
            raise SystemExit(f"{session}: training row lacks {metric}")
        key = (int(row["trial_index"]), int(row["block"]), label)
        grouped.setdefault(key, []).append(float(row[metric]))
        frame_counts[label] += 1

    trials = [
        IntentTrial(
            session=session.name,
            trial_index=trial_index,
            block=block,
            label=LABELS.index(label),
            value=float(np.median(values)),
            frames=len(values),
        )
        for (trial_index, block, label), values in sorted(grouped.items())
    ]
    trials_per_label = {
        label: sum(trial.label == index for trial in trials)
        for index, label in enumerate(LABELS)
    }
    if any(count == 0 for count in trials_per_label.values()):
        raise SystemExit(f"{session}: missing an intent grade: {trials_per_label}")
    if len(trials) != manifest.get("completed_trials"):
        raise SystemExit(
            f"{session}: {len(trials)} trainable trials != "
            f"{manifest.get('completed_trials')} completed trials"
        )
    return trials, {
        "session": session.name,
        "path": str(session.resolve()),
        "trials_per_label": trials_per_label,
        "frames_per_label": frame_counts,
        "metadata": {field: metadata.get(field) for field in COMPATIBILITY_FIELDS},
        "capture_jsonl_sha256": manifest["capture_jsonl_sha256"],
    }


def require_compatible(audits: list[dict]) -> None:
    if len({audit["path"] for audit in audits}) != len(audits):
        raise SystemExit("train and eval session paths must be disjoint")
    reference = audits[0]
    for audit in audits[1:]:
        mismatches = [
            field
            for field in COMPATIBILITY_FIELDS
            if audit["metadata"][field] != reference["metadata"][field]
        ]
        if mismatches:
            raise SystemExit(
                f"{audit['session']} is incompatible with {reference['session']}: "
                f"{mismatches}"
            )


def classify(value: float, thresholds: tuple[float, float]) -> int:
    lower, upper = thresholds
    if value < lower:
        return 0
    if value < upper:
        return 1
    return 2


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [float(np.mean(predictions[labels == label] == label)) for label in range(3)]
    return float(np.mean(recalls))


def fit_thresholds(trials: list[IntentTrial]) -> tuple[float, float]:
    """Choose two ordered cuts using training trials only."""
    values = np.asarray([trial.value for trial in trials], dtype=np.float64)
    labels = np.asarray([trial.label for trial in trials], dtype=int)
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("light, medium and hard training trials are required")
    unique = np.unique(values)
    cuts = np.concatenate(
        ([unique[0] - 1.0], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1.0])
    )
    best_score: tuple[float, float, float] | None = None
    best_thresholds: tuple[float, float] | None = None
    for lower_index, lower in enumerate(cuts[:-1]):
        for upper in cuts[lower_index + 1 :]:
            predicted = np.asarray(
                [classify(value, (float(lower), float(upper))) for value in values]
            )
            balanced = _balanced_accuracy(labels, predicted)
            ordinal_error = float(np.mean(np.abs(predicted - labels)))
            # Prefer wider medium bands only after classification quality ties.
            score = (balanced, -ordinal_error, float(upper - lower))
            if best_score is None or score > best_score:
                best_score = score
                best_thresholds = (float(lower), float(upper))
    assert best_thresholds is not None
    return best_thresholds


def score_trials(trials: list[IntentTrial], thresholds: tuple[float, float]) -> dict:
    confusion = np.zeros((3, 3), dtype=int)
    by_session: dict[str, list[IntentTrial]] = {}
    for trial in trials:
        confusion[trial.label, classify(trial.value, thresholds)] += 1
        by_session.setdefault(trial.session, []).append(trial)
    labels = np.asarray([trial.label for trial in trials], dtype=int)
    predictions = np.asarray([classify(trial.value, thresholds) for trial in trials])
    return {
        "trials": len(trials),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "mean_absolute_ordinal_error": float(np.mean(np.abs(predictions - labels))),
        "severe_error_rate": float(np.mean(np.abs(predictions - labels) == 2)),
        "confusion": {
            truth: {prediction: int(confusion[i, j]) for j, prediction in enumerate(LABELS)}
            for i, truth in enumerate(LABELS)
        },
        "per_session_accuracy": {
            session: float(
                np.mean(
                    [classify(trial.value, thresholds) == trial.label for trial in session_trials]
                )
            )
            for session, session_trials in sorted(by_session.items())
        },
    }


def class_summary(trials: list[IntentTrial]) -> dict:
    result = {}
    for index, label in enumerate(LABELS):
        values = np.asarray([trial.value for trial in trials if trial.label == index])
        result[label] = {
            "trials": int(values.size),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def evaluate(args) -> dict:
    train_trials: list[IntentTrial] = []
    eval_trials: list[IntentTrial] = []
    audits = []
    for session in args.train_session:
        trials, audit = load_trials(session, args.metric)
        train_trials.extend(trials)
        audits.append(audit)
    train_count = len(audits)
    for session in args.eval_session:
        trials, audit = load_trials(session, args.metric)
        eval_trials.extend(trials)
        audits.append(audit)
    require_compatible(audits)
    thresholds = fit_thresholds(train_trials)
    return {
        "schema_version": 1,
        "task": "ordered_operator_grip_intent_not_force_measurement",
        "split_policy": "whole_session_only",
        "contact_policy": "MediaPipe owns contact; baseline grades contact-confirmed frames only",
        "metric": args.metric,
        "train_sessions": audits[:train_count],
        "eval_sessions": audits[train_count:],
        "thresholds": {"light_medium": thresholds[0], "medium_hard": thresholds[1]},
        "train_class_summary": class_summary(train_trials),
        "eval_class_summary": class_summary(eval_trials),
        "train_score": score_trials(train_trials, thresholds),
        "eval_score": score_trials(eval_trials, thresholds),
    }


def _print_report(report: dict) -> None:
    thresholds = report["thresholds"]
    print(f"metric: {report['metric']}")
    print(
        "thresholds: "
        f"light/medium={thresholds['light_medium']:.1f}, "
        f"medium/hard={thresholds['medium_hard']:.1f}"
    )
    for split in ("train", "eval"):
        score = report[f"{split}_score"]
        print(
            f"{split}: trials={score['trials']} "
            f"accuracy={score['accuracy']:.3f} "
            f"balanced_accuracy={score['balanced_accuracy']:.3f} "
            f"severe_error_rate={score['severe_error_rate']:.3f}"
        )
        print(f"{split} confusion: {json.dumps(score['confusion'], sort_keys=True)}")


def main(argv=None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    _print_report(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
