#!/usr/bin/env python3
"""Compare frozen PressureVision light/hard baselines and a binary head.

Medium is excluded completely.  The original-model baseline fits one sum_kpa
threshold from training-session trial medians.  The adapted candidate freezes
the complete released PressureVision model, pools its 9-bin logits to 18D, and
trains a small LayerNorm/MLP binary head.  One whole validation session selects
the epoch; confirmation sessions are scored only after the head is frozen.

This is operator grip-intent classification, not force measurement.  Contact
remains MediaPipe's responsibility and no camera or robot is controlled.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("<workspace>")
BINARY_LABELS = ("light", "hard")

sys.path.insert(0, str(SCRIPT_DIR))
from train_pv_ordinal_head import (  # noqa: E402
    FEATURE_DIM,
    FeatureSet,
    cache_metadata,
    collect_records,
    extract_features,
    load_cache,
    resolve_device,
    save_cache,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=WORKSPACE / "pressurevision")
    parser.add_argument("--session-root", type=Path, default=WORKSPACE / "scratch_lepton")
    parser.add_argument("--train-sessions", required=True)
    parser.add_argument("--validation-session", required=True)
    parser.add_argument(
        "--confirmation-sessions",
        default="",
        help="Comma-separated frozen post-selection checks; never used for training/early stopping.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--development-cache", type=Path, required=True)
    parser.add_argument("--confirmation-cache", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.train_sessions = tuple(
        part.strip() for part in args.train_sessions.split(",") if part.strip()
    )
    args.confirmation_sessions = tuple(
        part.strip() for part in args.confirmation_sessions.split(",") if part.strip()
    )
    if not args.train_sessions:
        parser.error("--train-sessions must not be empty")
    roles = args.train_sessions + (args.validation_session,) + args.confirmation_sessions
    if len(roles) != len(set(roles)):
        parser.error("train, validation and confirmation sessions must be disjoint")
    if args.confirmation_sessions and args.confirmation_cache is None:
        parser.error("--confirmation-cache is required with --confirmation-sessions")
    for name in (
        "feature_batch_size",
        "head_batch_size",
        "epochs",
        "patience",
        "hidden_dim",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    return args


class BinaryPressureHead(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def binary_feature_indices(data: FeatureSet, sessions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return light/hard frame indices and remapped 0/1 labels; exclude medium."""
    session_mask = np.isin(data.sessions, np.asarray(sessions))
    grade_mask = np.isin(data.labels, np.asarray([0, 2]))
    indices = np.flatnonzero(session_mask & grade_mask)
    labels = (data.labels[indices] == 2).astype(np.int64)
    return indices, labels


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    confusion = np.zeros((2, 2), dtype=int)
    for actual, estimate in zip(truth, predicted):
        confusion[actual, estimate] += 1
    recalls = [
        float(confusion[index, index] / confusion[index].sum())
        if confusion[index].sum()
        else float("nan")
        for index in range(2)
    ]
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "confusion": {
            truth_name: {
                prediction_name: int(confusion[i, j])
                for j, prediction_name in enumerate(BINARY_LABELS)
            }
            for i, truth_name in enumerate(BINARY_LABELS)
        },
    }


def fit_sum_threshold(light: np.ndarray, hard: np.ndarray) -> float:
    light = np.asarray(light, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.float64)
    if not light.size or not hard.size:
        raise ValueError("both light and hard trials are required")
    if light.max() < hard.min():
        return float((light.max() + hard.min()) / 2.0)
    values = np.unique(np.concatenate((light, hard)))
    candidates = (values[:-1] + values[1:]) / 2.0
    target_midpoint = float((np.median(light) + np.median(hard)) / 2.0)
    best = None
    threshold = None
    labels = np.concatenate((np.zeros(light.size, dtype=int), np.ones(hard.size, dtype=int)))
    combined = np.concatenate((light, hard))
    for candidate in candidates:
        predictions = (combined >= candidate).astype(int)
        score = binary_metrics(labels, predictions)["balanced_accuracy"]
        key = (score, -abs(float(candidate) - target_midpoint))
        if best is None or key > best:
            best = key
            threshold = float(candidate)
    assert threshold is not None
    return threshold


def scalar_trial_values(session: Path) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    with (session / "capture.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row.get("target_label")
            if not row.get("valid_for_ordinal_training") or label not in BINARY_LABELS:
                continue
            grouped[(int(row["trial_index"]), label)].append(float(row["sum_kpa"]))
    labels = []
    values = []
    for (_trial, label), frames in sorted(grouped.items()):
        labels.append(BINARY_LABELS.index(label))
        values.append(statistics.median(frames))
    return np.asarray(labels, dtype=int), np.asarray(values, dtype=np.float64)


def scalar_frame_values(session: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    values = []
    with (session / "capture.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row.get("target_label")
            if not row.get("valid_for_ordinal_training") or label not in BINARY_LABELS:
                continue
            labels.append(BINARY_LABELS.index(label))
            values.append(float(row["sum_kpa"]))
    return np.asarray(labels, dtype=int), np.asarray(values, dtype=np.float64)


def evaluate_sum_threshold(session: Path, threshold: float) -> dict:
    frame_labels, frame_values = scalar_frame_values(session)
    trial_labels, trial_values = scalar_trial_values(session)
    return {
        "frame": binary_metrics(frame_labels, (frame_values >= threshold).astype(int)),
        "trial": binary_metrics(trial_labels, (trial_values >= threshold).astype(int)),
    }


def _trial_head_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    trial_indices: np.ndarray,
) -> dict:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (session, trial) in enumerate(zip(sessions.tolist(), trial_indices.tolist())):
        grouped[(str(session), int(trial))].append(index)
    truth = []
    predictions = []
    for indices in grouped.values():
        trial_labels = set(labels[indices].tolist())
        if len(trial_labels) != 1:
            raise ValueError("binary trial contains multiple labels")
        truth.append(trial_labels.pop())
        predictions.append(int(np.median(logits[indices]) >= 0.0))
    return binary_metrics(np.asarray(truth), np.asarray(predictions))


def evaluate_head(
    model: BinaryPressureHead,
    data: FeatureSet,
    sessions: tuple[str, ...],
    device: str,
) -> dict:
    indices, labels = binary_feature_indices(data, sessions)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data.features[indices]).to(device)).cpu().numpy()
    return {
        "frame": binary_metrics(labels, (logits >= 0.0).astype(int)),
        "trial": _trial_head_metrics(
            logits,
            labels,
            data.sessions[indices],
            data.trial_indices[indices],
        ),
    }


def train_head(data: FeatureSet, args, device: str) -> tuple[BinaryPressureHead, dict]:
    train_indices, train_labels = binary_feature_indices(data, args.train_sessions)
    validation_indices, validation_labels = binary_feature_indices(
        data, (args.validation_session,)
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = BinaryPressureHead(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.features[train_indices]),
            torch.from_numpy(train_labels.astype(np.float32)),
        ),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = torch.from_numpy(data.features[validation_indices]).to(device)
    validation_targets = torch.from_numpy(validation_labels.astype(np.float32)).to(device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * targets.shape[0]
            seen += targets.shape[0]
        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_features)
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    validation_logits, validation_targets
                ).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / seen,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }


def _load_or_extract(
    session_names: tuple[str, ...],
    *,
    cache: Path,
    args,
    device: str,
) -> tuple[FeatureSet, list[dict]]:
    paths = [args.session_root / name for name in session_names]
    records, audits = collect_records(paths)
    metadata = cache_metadata(args.repo, records, audits)
    data = None if args.refresh_cache else load_cache(cache, metadata)
    if data is None:
        data = extract_features(
            records,
            repo=args.repo,
            device=device,
            batch_size=args.feature_batch_size,
        )
        save_cache(cache, metadata, data)
        print(f"wrote feature cache {cache}", flush=True)
    else:
        print(f"loaded exact feature cache {cache}", flush=True)
    return data, audits


def run(args) -> dict:
    device = resolve_device(args.device)
    development_sessions = args.train_sessions + (args.validation_session,)
    development, development_audits = _load_or_extract(
        development_sessions,
        cache=args.development_cache,
        args=args,
        device=device,
    )

    train_scalar_labels = []
    train_scalar_values = []
    for name in args.train_sessions:
        labels, values = scalar_trial_values(args.session_root / name)
        train_scalar_labels.append(labels)
        train_scalar_values.append(values)
    scalar_labels = np.concatenate(train_scalar_labels)
    scalar_values = np.concatenate(train_scalar_values)
    threshold = fit_sum_threshold(
        scalar_values[scalar_labels == 0], scalar_values[scalar_labels == 1]
    )

    head, training = train_head(development, args, device)
    results = {
        "train": {
            "sum_kpa": {
                name: evaluate_sum_threshold(args.session_root / name, threshold)
                for name in args.train_sessions
            },
            "binary_head": evaluate_head(head, development, args.train_sessions, device),
        },
        "validation": {
            "sum_kpa": evaluate_sum_threshold(
                args.session_root / args.validation_session, threshold
            ),
            "binary_head": evaluate_head(
                head, development, (args.validation_session,), device
            ),
        },
        "confirmation": {},
    }
    confirmation_audits = []
    if args.confirmation_sessions:
        confirmation, confirmation_audits = _load_or_extract(
            args.confirmation_sessions,
            cache=args.confirmation_cache,
            args=args,
            device=device,
        )
        for name in args.confirmation_sessions:
            results["confirmation"][name] = {
                "sum_kpa": evaluate_sum_threshold(args.session_root / name, threshold),
                "binary_head": evaluate_head(head, confirmation, (name,), device),
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "binary_head_best.pt"
    torch.save(
        {
            "schema_version": 1,
            "task": "light_vs_hard_operator_grip_intent",
            "labels": BINARY_LABELS,
            "train_sessions": args.train_sessions,
            "validation_session": args.validation_session,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
        },
        checkpoint_path,
    )
    report = {
        "schema_version": 1,
        "status": "posthoc_binary_development_and_confirmation_not_new_final_test",
        "task": "light_vs_hard_operator_grip_intent_not_force_measurement",
        "excluded_grade": "medium; no medium frame or trial enters fit or score",
        "contact_policy": "MediaPipe owns contact",
        "split_policy": {
            "train": list(args.train_sessions),
            "validation": args.validation_session,
            "post_selection_confirmation": list(args.confirmation_sessions),
        },
        "sum_kpa_threshold": threshold,
        "binary_head": {
            "architecture": "18D LayerNorm -> Linear -> GELU -> Dropout -> binary logit",
            "parameters": sum(parameter.numel() for parameter in head.parameters()),
            "training": training,
            "checkpoint": str(checkpoint_path),
        },
        "development_session_audits": development_audits,
        "confirmation_session_audits": confirmation_audits,
        "results": results,
        "limitations": [
            "the decision to try binary classification was motivated by fixed_06",
            "fixed_05 and fixed_06 are post-selection confirmations, not untouched binary final tests",
            "a new blinded session is required for an untouched binary final test",
            "no FPN, encoder, contact classifier, or robot path was trained",
        ],
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sum_kpa light/hard threshold={threshold:.1f}")
    print(
        "validation trial BA original/head="
        f"{results['validation']['sum_kpa']['trial']['balanced_accuracy']:.3f}/"
        f"{results['validation']['binary_head']['trial']['balanced_accuracy']:.3f}"
    )
    for name, result in results["confirmation"].items():
        print(
            f"{name} trial BA original/head="
            f"{result['sum_kpa']['trial']['balanced_accuracy']:.3f}/"
            f"{result['binary_head']['trial']['balanced_accuracy']:.3f}"
        )
    print(f"wrote {report_path}")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
