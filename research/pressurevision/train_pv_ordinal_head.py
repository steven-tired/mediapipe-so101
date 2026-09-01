#!/usr/bin/env python3
"""Train the first fixed-setup PressureVision light/medium/hard model.

The released PressureVision network stays entirely frozen and in eval mode.
For each 480x384 RGB frame, its frozen 9-bin pressure logits are spatially
average- and max-pooled into 18 values.  A small LayerNorm/MLP head produces a
single severity score.  Two learned ordered thresholds turn that score into
light/medium/hard probabilities with an ordinal cumulative-link loss.

Development evaluation is leave-one-whole-session-out.  Frames or trials from
one session are never randomly split across training and validation.  This
script does not train contact/no-contact, calibrate abstention, unfreeze the
FPN, or interact with a camera or robot.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("<workspace>")
DEFAULT_REPO = WORKSPACE / "pressurevision"
DEFAULT_SESSION_ROOT = WORKSPACE / "scratch_lepton"
LABELS = ("light", "medium", "hard")
FEATURE_DIM = 18
CACHE_SCHEMA_VERSION = 1

sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_pv_intent_baseline import (  # noqa: E402
    load_trials as audit_trials,
    require_compatible,
)
from pressurevision_probe import load_model, preprocess  # noqa: E402


@dataclass(frozen=True)
class FrameRecord:
    session: str
    trial_index: int
    phase_index: int
    label: int
    path: Path

    @property
    def identity(self) -> str:
        return f"{self.session}:{self.trial_index}:{self.phase_index}:{self.label}"


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    trial_indices: np.ndarray
    phase_indices: np.ndarray

    def __post_init__(self):
        count = self.features.shape[0]
        if self.features.shape != (count, FEATURE_DIM):
            raise ValueError(f"features must be [N, {FEATURE_DIM}]")
        for values in (self.labels, self.sessions, self.trial_indices, self.phase_indices):
            if values.shape != (count,):
                raise ValueError("feature metadata must have one entry per frame")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument(
        "--sessions",
        help="Comma-separated sessions for leave-one-whole-session-out development.",
    )
    split.add_argument(
        "--train-sessions",
        help="Comma-separated training sessions for one explicit held-out validation.",
    )
    parser.add_argument(
        "--validation-session",
        default=None,
        help="Held-out session used only with --train-sessions.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/pv_intent_pressure_logits_v1.npz"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sessions is not None:
        args.sessions = tuple(part.strip() for part in args.sessions.split(",") if part.strip())
        args.train_sessions = None
        if len(args.sessions) < 2:
            parser.error("--sessions needs at least two whole sessions for cross-validation")
        if args.validation_session is not None:
            parser.error("--validation-session requires --train-sessions")
        selected_sessions = args.sessions
    else:
        args.train_sessions = tuple(
            part.strip() for part in args.train_sessions.split(",") if part.strip()
        )
        if not args.train_sessions:
            parser.error("--train-sessions must not be empty")
        if args.validation_session is None:
            parser.error("--validation-session is required with --train-sessions")
        if args.validation_session in args.train_sessions:
            parser.error("validation session must not be a training session")
        selected_sessions = args.train_sessions + (args.validation_session,)
    if len(set(selected_sessions)) != len(selected_sessions):
        parser.error("session list contains a duplicate")
    for name in ("feature_batch_size", "head_batch_size", "epochs", "patience", "hidden_dim"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    return args


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def collect_records(session_paths: list[Path]) -> tuple[list[FrameRecord], list[dict]]:
    """Audit sessions and return only accepted steady frame records."""
    records: list[FrameRecord] = []
    audits = []
    for session in session_paths:
        _trials, audit = audit_trials(session, "sum_kpa")
        audits.append(audit)
        with (session / "capture.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("valid_for_ordinal_training"):
                    continue
                label_name = row["target_label"]
                path = session / row["frame"]
                if not path.is_file():
                    raise SystemExit(f"missing training frame: {path}")
                records.append(
                    FrameRecord(
                        session=session.name,
                        trial_index=int(row["trial_index"]),
                        phase_index=int(row["phase_index"]),
                        label=LABELS.index(label_name),
                        path=path,
                    )
                )
    require_compatible(audits)
    identities = [record.identity for record in records]
    if len(identities) != len(set(identities)):
        raise SystemExit("duplicate session/trial/frame identity in training records")
    return records, audits


def cache_metadata(repo: Path, records: list[FrameRecord], audits: list[dict]) -> dict:
    identities = "\n".join(record.identity for record in records).encode("utf-8")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "feature": "frozen_pressure_logits_spatial_avg_plus_max_18d",
        "preprocess": "480x384 RGB ImageNet mean/std",
        "checkpoint_sha256": _sha256(repo / "data/model/paper_59.pt"),
        "capture_sha256": {
            audit["session"]: audit["capture_jsonl_sha256"] for audit in audits
        },
        "record_identity_sha256": sha256(identities).hexdigest(),
        "records": len(records),
    }


def _pool_avg_max(tensor: torch.Tensor) -> torch.Tensor:
    average = F.adaptive_avg_pool2d(tensor, 1).flatten(1)
    maximum = F.adaptive_max_pool2d(tensor, 1).flatten(1)
    return torch.cat((average, maximum), dim=1)


class FrozenPressureLogitExtractor:
    def __init__(self, repo: Path, device: str):
        self.model, _config = load_model(repo, device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    def batch(self, paths: list[Path]) -> np.ndarray:
        tensors = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"could not decode {path}")
            height, width = frame.shape[:2]
            _resized, tensor = preprocess(frame, (0, 0, width, height))
            tensors.append(tensor.squeeze(0))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            encoder_features = self.model.encoder(batch)
            decoder = self.model.decoder(encoder_features)
            logits = self.model.segmentation_head(decoder)
            features = _pool_avg_max(logits)
        if features.shape[1] != FEATURE_DIM:
            raise SystemExit(
                f"expected {FEATURE_DIM} pooled pressure-logit features, got {features.shape[1]}"
            )
        return features.cpu().numpy().astype(np.float32)


def extract_features(
    records: list[FrameRecord],
    *,
    repo: Path,
    device: str,
    batch_size: int,
) -> FeatureSet:
    extractor = FrozenPressureLogitExtractor(repo, device)
    collected = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        collected.append(extractor.batch([record.path for record in batch_records]))
        done = min(start + batch_size, len(records))
        if done == len(records) or done % max(batch_size, 256) == 0:
            print(f"feature extraction: {done}/{len(records)}", flush=True)
    return FeatureSet(
        features=np.concatenate(collected, axis=0),
        labels=np.asarray([record.label for record in records], dtype=np.int64),
        sessions=np.asarray([record.session for record in records]),
        trial_indices=np.asarray([record.trial_index for record in records], dtype=np.int64),
        phase_indices=np.asarray([record.phase_index for record in records], dtype=np.int64),
    )


def save_cache(path: Path, metadata: dict, data: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        features=data.features,
        labels=data.labels,
        sessions=data.sessions,
        trial_indices=data.trial_indices,
        phase_indices=data.phase_indices,
    )


def load_cache(path: Path, expected_metadata: dict) -> FeatureSet | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata != expected_metadata:
            return None
        return FeatureSet(
            features=archive["features"].astype(np.float32),
            labels=archive["labels"].astype(np.int64),
            sessions=archive["sessions"],
            trial_indices=archive["trial_indices"].astype(np.int64),
            phase_indices=archive["phase_indices"].astype(np.int64),
        )


class OrdinalPressureHead(nn.Module):
    """One severity score plus two explicitly ordered thresholds."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.normalise = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold_low = nn.Parameter(torch.tensor(-0.5))
        minimum_gap = 0.05
        self.register_buffer("minimum_gap", torch.tensor(minimum_gap))
        initial_raw_gap = math.log(math.expm1(1.0 - minimum_gap))
        self.threshold_gap_raw = nn.Parameter(torch.tensor(initial_raw_gap))

    def severity(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.normalise(features)).squeeze(-1)

    def thresholds(self) -> torch.Tensor:
        high = self.threshold_low + F.softplus(self.threshold_gap_raw) + self.minimum_gap
        return torch.stack((self.threshold_low, high))

    def cumulative_logits_from_score(self, score: torch.Tensor) -> torch.Tensor:
        return score[:, None] - self.thresholds()[None, :]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.severity(features)
        return score, self.cumulative_logits_from_score(score)

    def probabilities_from_score(self, score: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(self.cumulative_logits_from_score(score))
        return torch.stack(
            (1.0 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1], cumulative[:, 1]),
            dim=1,
        )

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return self.probabilities_from_score(self.severity(features))


def ordinal_loss(cumulative_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = torch.stack((labels > 0, labels > 1), dim=1).to(cumulative_logits.dtype)
    return F.binary_cross_entropy_with_logits(cumulative_logits, targets)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    confusion = np.zeros((3, 3), dtype=int)
    for actual, estimate in zip(truth, predicted):
        confusion[actual, estimate] += 1
    recalls = [
        float(confusion[index, index] / confusion[index].sum())
        if confusion[index].sum()
        else float("nan")
        for index in range(3)
    ]
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "mean_absolute_ordinal_error": float(np.mean(np.abs(truth - predicted))),
        "severe_error_rate": float(np.mean(np.abs(truth - predicted) == 2)),
        "confusion": {
            label: {prediction: int(confusion[i, j]) for j, prediction in enumerate(LABELS)}
            for i, label in enumerate(LABELS)
        },
    }


def trial_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    trial_indices: np.ndarray,
    model: OrdinalPressureHead,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, (session, trial) in enumerate(zip(sessions.tolist(), trial_indices.tolist())):
        grouped.setdefault((str(session), int(trial)), []).append(index)
    truth = []
    medians = []
    for indices in grouped.values():
        trial_labels = set(labels[indices].tolist())
        if len(trial_labels) != 1:
            raise ValueError("one trial contains multiple ordinal labels")
        truth.append(trial_labels.pop())
        medians.append(float(np.median(scores[indices])))
    score_tensor = torch.tensor(medians, dtype=torch.float32, device=device)
    with torch.inference_mode():
        predictions = model.probabilities_from_score(score_tensor).argmax(dim=1)
    return np.asarray(truth, dtype=int), predictions.cpu().numpy().astype(int)


def evaluate_subset(
    model: OrdinalPressureHead,
    data: FeatureSet,
    indices: np.ndarray,
    device: str,
) -> dict:
    model.eval()
    features = torch.from_numpy(data.features[indices]).to(device)
    labels = torch.from_numpy(data.labels[indices]).to(device)
    with torch.inference_mode():
        scores, logits = model(features)
        loss = float(ordinal_loss(logits, labels).item())
        frame_predictions = model.probabilities_from_score(scores).argmax(dim=1)
    trial_truth, trial_estimates = trial_predictions(
        scores.cpu().numpy(),
        data.labels[indices],
        data.sessions[indices],
        data.trial_indices[indices],
        model,
        device,
    )
    return {
        "loss": loss,
        "frame": classification_metrics(data.labels[indices], frame_predictions.cpu().numpy()),
        "trial": classification_metrics(trial_truth, trial_estimates),
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fold(
    data: FeatureSet,
    *,
    validation_session: str,
    args,
    device: str,
) -> tuple[OrdinalPressureHead, dict]:
    train_indices = np.flatnonzero(data.sessions != validation_session)
    validation_indices = np.flatnonzero(data.sessions == validation_session)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError(f"empty train or validation split for {validation_session}")
    if set(data.sessions[train_indices]) & set(data.sessions[validation_indices]):
        raise AssertionError("whole-session split was violated")

    fold_seed = args.seed + sorted(set(data.sessions.tolist())).index(validation_session)
    _seed_everything(fold_seed)
    model = OrdinalPressureHead(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(fold_seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.features[train_indices]),
            torch.from_numpy(data.labels[train_indices]),
        ),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _scores, logits = model(features)
            loss = ordinal_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
        validation = evaluate_subset(model, data, validation_indices, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_loss": validation["loss"],
                "validation_trial_balanced_accuracy": validation["trial"][
                    "balanced_accuracy"
                ],
            }
        )
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_result = evaluate_subset(model, data, train_indices, device)
    validation_result = evaluate_subset(model, data, validation_indices, device)
    thresholds = model.thresholds().detach().cpu().tolist()
    report = {
        "validation_session": validation_session,
        "train_sessions": sorted(set(data.sessions[train_indices].tolist())),
        "train_frames": int(train_indices.size),
        "validation_frames": int(validation_indices.size),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "thresholds": {"light_medium": thresholds[0], "medium_hard": thresholds[1]},
        "train": train_result,
        "validation": validation_result,
        "history": history,
    }
    return model, report


def _aggregate_fold_trials(folds: list[dict]) -> dict:
    confusion = np.zeros((3, 3), dtype=int)
    for fold in folds:
        block = fold["validation"]["trial"]["confusion"]
        for i, truth in enumerate(LABELS):
            for j, prediction in enumerate(LABELS):
                confusion[i, j] += block[truth][prediction]
    labels = []
    predictions = []
    for i in range(3):
        for j in range(3):
            labels.extend([i] * int(confusion[i, j]))
            predictions.extend([j] * int(confusion[i, j]))
    return classification_metrics(np.asarray(labels), np.asarray(predictions))


def run(args) -> dict:
    device = resolve_device(args.device)
    if args.sessions is not None:
        session_names = args.sessions
        validation_sessions = args.sessions
        status = "development_cross_validation_not_final_test"
        split_policy = "leave_one_whole_session_out"
    else:
        session_names = args.train_sessions + (args.validation_session,)
        validation_sessions = (args.validation_session,)
        status = "held_out_validation_for_model_selection_not_final_test"
        split_policy = "explicit_whole_train_sessions_and_one_held_out_validation_session"
    session_paths = [args.session_root / name for name in session_names]
    records, audits = collect_records(session_paths)
    metadata = cache_metadata(args.repo, records, audits)
    cached = None if args.refresh_cache else load_cache(args.cache, metadata)
    if cached is None:
        print(f"extracting frozen pressure logits on {device}", flush=True)
        data = extract_features(
            records,
            repo=args.repo,
            device=device,
            batch_size=args.feature_batch_size,
        )
        save_cache(args.cache, metadata, data)
        print(f"wrote feature cache {args.cache}", flush=True)
    else:
        data = cached
        print(f"loaded exact feature cache {args.cache}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    parameter_count = None
    for validation_session in validation_sessions:
        print(f"training fold: hold out {validation_session}", flush=True)
        model, fold = train_fold(
            data,
            validation_session=validation_session,
            args=args,
            device=device,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint = {
            "schema_version": 1,
            "architecture": "18D LayerNorm MLP single score two ordered thresholds",
            "labels": LABELS,
            "input_feature": metadata["feature"],
            "base_checkpoint_sha256": metadata["checkpoint_sha256"],
            "validation_session": validation_session,
            "train_sessions": fold["train_sessions"],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        checkpoint_path = args.out_dir / f"ordinal_head_holdout_{validation_session}.pt"
        torch.save(checkpoint, checkpoint_path)
        fold["checkpoint"] = str(checkpoint_path)
        folds.append(fold)
        print(
            f"{validation_session}: trial BA="
            f"{fold['validation']['trial']['balanced_accuracy']:.3f} "
            f"best_epoch={fold['best_epoch']}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": status,
        "task": "ordered_operator_grip_intent_not_force_measurement",
        "contact_policy": "MediaPipe owns contact",
        "split_policy": split_policy,
        "base_network": "released PressureVision fully frozen in eval mode",
        "feature": metadata["feature"],
        "head": "LayerNorm -> Linear -> GELU -> Dropout -> Linear severity score",
        "ordered_thresholds": "theta_2 = theta_1 + softplus(delta) + 0.05",
        "device": device,
        "sessions": list(session_names),
        "configured_train_sessions": (
            list(args.train_sessions) if args.train_sessions is not None else None
        ),
        "configured_validation_session": args.validation_session,
        "session_audits": audits,
        "feature_cache": str(args.cache),
        "head_parameters": parameter_count,
        "hyperparameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "head_batch_size": args.head_batch_size,
            "seed": args.seed,
        },
        "folds": folds,
        "aggregate_validation_trial": _aggregate_fold_trials(folds),
        "limitations": [
            (
                "a separate architecture/abstention validation session is still required"
                if args.sessions is not None
                else "this validation session must not be moved into the training split"
            ),
            "a separate final-test session must remain untouched",
            "no FPN/C5 parameters were trained",
            "no contact/no-contact classifier was trained",
        ],
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "aggregate held-session trial BA="
        f"{report['aggregate_validation_trial']['balanced_accuracy']:.3f}"
    )
    print(f"wrote {report_path}")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
