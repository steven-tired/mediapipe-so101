#!/usr/bin/env python3
"""Evaluate a frozen PressureVision feature head for light versus hard intent.

This is an offline, robot-free linear-probe audit. It asks whether the released
PressureVision encoder/FPN contains a more stable two-band signal than the
current `sum_kpa` reduction, without fine-tuning any of the 28.1M pretrained
parameters.

Frames inside one held press are correlated, so they are never train/test
samples. Frozen frame embeddings are collapsed to one median vector per trial;
all model selection is done on whole-session folds 03/04/05, before the intent
sessions 06/08 are evaluated. Contact/no-contact is deliberately excluded:
MediaPipe pinch owns contact, while this head grades only light versus hard.

The probe is deterministic NumPy ridge regression rather than a deep head. It
needs no sklearn/scipy dependency and its train statistics are stored in the
fit object, making accidental test-set scaling visible in unit tests.

Read-only with respect to cameras and robots. The optional cache and JSON report
contain derived offline features/results only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("<workspace>")
DEFAULT_REPO = WORKSPACE / "pressurevision"
DEFAULT_SESSION_ROOT = WORKSPACE / "scratch_lepton"
DEFAULT_SESSIONS = (
    "pv_labelled_03",
    "pv_labelled_04",
    "pv_labelled_05",
    "pv_labelled_06",
    "pv_labelled_08",
)
DEVELOPMENT_SESSIONS = ("pv_labelled_03", "pv_labelled_04", "pv_labelled_05")
INTENT_SESSIONS = ("pv_labelled_06", "pv_labelled_08")
METRIC_NAMES = ("contact_px", "sum_kpa", "mean_kpa_in_contact", "max_kpa")
FEATURE_NAMES = ("pressure_logits", "encoder_c5", "decoder")
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BinaryTrial:
    session: str
    trial_index: int
    target: int
    label: int  # 0 = lower loaded level, 1 = higher loaded level
    frame_paths: tuple[Path, ...]
    metrics: dict[str, float]

    @property
    def identity(self) -> str:
        return f"{self.session}:{self.trial_index}:{self.target}"


@dataclass(frozen=True)
class RidgeProbe:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    l2: float

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        return ((matrix - self.mean) / self.scale) @ self.weights + self.bias


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument(
        "--sessions",
        default=",".join(DEFAULT_SESSIONS),
        help="Comma-separated session directory names under --session-root.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/pressurevision_frozen_features_v1.npz"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--calibrate-trials", type=int, default=3)
    parser.add_argument("--abstain-fraction", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.sessions = tuple(part.strip() for part in args.sessions.split(",") if part.strip())
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.calibrate_trials < 1:
        parser.error("--calibrate-trials must be positive")
    if not 0.0 <= args.abstain_fraction <= 1.0:
        parser.error("--abstain-fraction must be in [0, 1]")
    missing = set(DEVELOPMENT_SESSIONS + INTENT_SESSIONS) - set(args.sessions)
    if missing:
        parser.error(f"required sessions omitted: {sorted(missing)}")
    return args


def _capture_rows(session: Path) -> list[dict]:
    capture = session / "capture.jsonl"
    if not capture.is_file():
        raise SystemExit(f"{session}: no capture.jsonl")
    rows = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"{capture}: empty")
    return rows


def _is_early_lift(rows: list[dict]) -> bool:
    contact = [int(row.get("contact_px", 0)) > 0 for row in rows]
    held = sum(contact)
    return bool(
        0 < held
        and all(contact[:held])
        and not any(contact[held:])
        and held * 2 < len(contact)
    )


def load_binary_trials(session: Path) -> tuple[list[BinaryTrial], dict]:
    """Load exactly the two nonzero levels, with one record per held press."""
    rows = _capture_rows(session)
    meta = rows[0]
    targets = sorted(int(target) for target in meta.get("targets_g", []))
    loaded_targets = [target for target in targets if target != 0]
    if len(loaded_targets) != 2:
        raise SystemExit(
            f"{session}: frozen two-band audit needs exactly two loaded targets, "
            f"found {loaded_targets}"
        )
    label_by_target = {loaded_targets[0]: 0, loaded_targets[1]: 1}

    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row.get("row_type") != "sample":
            continue
        key = (int(row["trial_index"]), int(row["target_g"]))
        grouped.setdefault(key, []).append(row)

    baseline_trials = 0
    lifted_trials: list[int] = []
    trials: list[BinaryTrial] = []
    for (trial_index, target), frames in sorted(grouped.items()):
        frames.sort(key=lambda row: int(row.get("hold_index", 0)))
        if target == 0:
            baseline_trials += 1
            continue
        if target not in label_by_target:
            continue
        if _is_early_lift(frames):
            lifted_trials.append(trial_index)
            continue
        missing_metrics = [name for name in METRIC_NAMES if name not in frames[0]]
        if missing_metrics:
            raise SystemExit(f"{session}: trial {trial_index} lacks {missing_metrics}")
        trials.append(
            BinaryTrial(
                session=session.name,
                trial_index=trial_index,
                target=target,
                label=label_by_target[target],
                frame_paths=tuple(session / frame["frame"] for frame in frames),
                metrics={
                    name: float(np.median([float(frame[name]) for frame in frames]))
                    for name in METRIC_NAMES
                },
            )
        )

    per_label = {str(label): sum(trial.label == label for trial in trials) for label in (0, 1)}
    if any(count == 0 for count in per_label.values()):
        raise SystemExit(f"{session}: missing a loaded class after exclusions: {per_label}")
    return trials, {
        "session": session.name,
        "targets": loaded_targets,
        "usable_trials_per_label": per_label,
        "excluded_baseline_trials": baseline_trials,
        "excluded_early_lift_trials": lifted_trials,
    }


def aggregate_trial_embedding(frame_embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("frame embeddings must be a non-empty [frames, features] matrix")
    return np.median(values, axis=0)


def fit_ridge_probe(values: np.ndarray, labels: np.ndarray, *, l2: float) -> RidgeProbe:
    """Fit a deterministic linear probe, using a dual solve when d >> n."""
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("values must be [trials, features] and labels [trials]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary labels are required")
    if l2 <= 0:
        raise ValueError("l2 must be positive")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    signed = y.astype(np.float64) * 2.0 - 1.0
    bias = float(signed.mean())
    centred = signed - bias
    # z.T @ inv(z @ z.T + lambda I) @ y avoids a 4096x4096 solve for C5.
    gram = z @ z.T
    alpha = np.linalg.solve(gram + l2 * np.eye(gram.shape[0]), centred)
    weights = z.T @ alpha
    return RidgeProbe(mean=mean, scale=scale, weights=weights, bias=bias, l2=l2)


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    low = scores[labels == 0]
    high = scores[labels == 1]
    if low.size == 0 or high.size == 0:
        return float("nan")
    wins = float((high[:, None] > low[None, :]).sum())
    ties = float((high[:, None] == low[None, :]).sum())
    return (wins + 0.5 * ties) / (low.size * high.size)


def _dprime(labels: np.ndarray, scores: np.ndarray) -> float:
    low = scores[labels == 0]
    high = scores[labels == 1]
    pooled = math.sqrt((float(np.var(low)) + float(np.var(high))) / 2.0)
    gap = float(np.median(high) - np.median(low))
    if pooled == 0.0:
        return float("inf") if gap > 0 else 0.0
    return gap / pooled


def binary_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float = 0.0) -> dict:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=np.float64)
    predicted = (values > threshold).astype(int)
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tp = int(((y == 1) & (predicted == 1)).sum())
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "trials": int(y.size),
        "roc_auc": _rank_auc(y, values),
        "dprime": _dprime(y, values),
        "balanced_accuracy": (tnr + tpr) / 2.0,
        "accuracy": float((predicted == y).mean()) if y.size else None,
        "confusion": {
            "light": {"light": tn, "hard": fp},
            "hard": {"light": fn, "hard": tp},
        },
    }


def fit_abstain_boundary(
    labels: np.ndarray, scores: np.ndarray, *, abstain_fraction: float
) -> dict:
    low = np.asarray(scores)[np.asarray(labels) == 0]
    high = np.asarray(scores)[np.asarray(labels) == 1]
    if low.size == 0 or high.size == 0:
        raise ValueError("both labels are required to fit an abstain boundary")
    separated = bool(low.max() < high.min())
    if separated:
        edge = (float(low.max()) + float(high.min())) / 2.0
        gap = float(high.min() - low.max())
    else:
        edge = (float(np.median(low)) + float(np.median(high))) / 2.0
        gap = float(low.max() - high.min())
    half = abstain_fraction * abs(gap) / 2.0
    return {
        "edge": edge,
        "abstain_lo": edge - half,
        "abstain_hi": edge + half,
        "separated_in_fit": separated,
    }


def abstain_metrics(labels: np.ndarray, scores: np.ndarray, boundary: dict) -> dict:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=np.float64)
    in_band = (
        (boundary["abstain_hi"] > boundary["abstain_lo"])
        & (values >= boundary["abstain_lo"])
        & (values <= boundary["abstain_hi"])
    )
    predictions = (values > boundary["edge"]).astype(int)
    decided = ~in_band
    correct = int(((predictions == y) & decided).sum())
    decided_count = int(decided.sum())
    return {
        "abstained": int(in_band.sum()),
        "coverage": decided_count / y.size if y.size else None,
        "decided": decided_count,
        "correct": correct,
        "accuracy_on_decided": correct / decided_count if decided_count else None,
        "accuracy_counting_abstain_as_wrong": correct / y.size if y.size else None,
    }


def calibration_split(
    trials: Iterable[BinaryTrial], *, per_label: int
) -> tuple[list[BinaryTrial], list[BinaryTrial]]:
    ordered = sorted(trials, key=lambda trial: trial.trial_index)
    seen = {0: 0, 1: 0}
    fit: list[BinaryTrial] = []
    test: list[BinaryTrial] = []
    for trial in ordered:
        if seen[trial.label] < per_label:
            fit.append(trial)
            seen[trial.label] += 1
        else:
            test.append(trial)
    if seen != {0: per_label, 1: per_label}:
        raise ValueError(f"not enough calibration trials: got {seen}, need {per_label} each")
    if any(not any(item.label == label for item in test) for label in (0, 1)):
        raise ValueError("calibration consumed all test trials for a label")
    return fit, test


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_avg_max(tensor: torch.Tensor) -> torch.Tensor:
    average = F.adaptive_avg_pool2d(tensor, 1).flatten(1)
    maximum = F.adaptive_max_pool2d(tensor, 1).flatten(1)
    return torch.cat((average, maximum), dim=1)


class FrozenFeatureExtractor:
    def __init__(self, repo: Path, device: str):
        sys.path.insert(0, str(SCRIPT_DIR))
        from pressurevision_probe import load_model, preprocess

        self.model, _config = load_model(repo, device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.preprocess = preprocess
        self.device = device

    def _batch(self, paths: list[Path]) -> dict[str, np.ndarray]:
        tensors = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"could not read captured frame {path}")
            height, width = frame.shape[:2]
            _resized, tensor = self.preprocess(frame, (0, 0, width, height))
            tensors.append(tensor.squeeze(0))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            encoder_features = self.model.encoder(batch)
            decoder = self.model.decoder(encoder_features)
            logits = self.model.segmentation_head(decoder)
            pooled = {
                "pressure_logits": _pool_avg_max(logits),
                "encoder_c5": _pool_avg_max(encoder_features[-1]),
                "decoder": _pool_avg_max(decoder),
            }
        return {name: value.cpu().numpy().astype(np.float32) for name, value in pooled.items()}

    def trial(self, trial: BinaryTrial, batch_size: int) -> dict[str, np.ndarray]:
        collected: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
        paths = list(trial.frame_paths)
        for start in range(0, len(paths), batch_size):
            batch = self._batch(paths[start : start + batch_size])
            for name in FEATURE_NAMES:
                collected[name].append(batch[name])
        return {
            name: aggregate_trial_embedding(np.concatenate(collected[name], axis=0))
            for name in FEATURE_NAMES
        }


def _cache_metadata(repo: Path, sessions: list[Path], trials: list[BinaryTrial]) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": _sha256(repo / "data/model/paper_59.pt"),
        "capture_sha256": {
            session.name: _sha256(session / "capture.jsonl") for session in sessions
        },
        "trial_identities": [trial.identity for trial in trials],
        "features": list(FEATURE_NAMES),
        "preprocess": "480x384 RGB ImageNet mean/std",
        "aggregation": "per-dimension median across frames in one held press",
    }


def save_cache(path: Path, metadata: dict, features: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{f"feature_{name}": value for name, value in features.items()},
    )


def load_cache(path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        actual = json.loads(str(data["metadata"].item()))
        if actual != expected:
            return None
        return {name: np.asarray(data[f"feature_{name}"]) for name in FEATURE_NAMES}


def _indices(trials: list[BinaryTrial], sessions: Iterable[str]) -> np.ndarray:
    wanted = set(sessions)
    return np.asarray([index for index, trial in enumerate(trials) if trial.session in wanted])


def _evaluate_fold(
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    l2: float,
    abstain_fraction: float,
) -> dict:
    probe = fit_ridge_probe(values[train_indices], labels[train_indices], l2=l2)
    train_scores = probe.decision_function(values[train_indices])
    test_scores = probe.decision_function(values[test_indices])
    boundary = fit_abstain_boundary(
        labels[train_indices], train_scores, abstain_fraction=abstain_fraction
    )
    metrics = binary_metrics(labels[test_indices], test_scores, threshold=boundary["edge"])
    return {
        **metrics,
        "abstain": abstain_metrics(labels[test_indices], test_scores, boundary),
        "fit_boundary": boundary,
    }


def _development_folds(
    values: np.ndarray,
    labels: np.ndarray,
    trials: list[BinaryTrial],
    *,
    l2: float,
    abstain_fraction: float,
) -> list[dict]:
    results = []
    for held_out in DEVELOPMENT_SESSIONS:
        train_sessions = [name for name in DEVELOPMENT_SESSIONS if name != held_out]
        result = _evaluate_fold(
            values,
            labels,
            _indices(trials, train_sessions),
            _indices(trials, [held_out]),
            l2=l2,
            abstain_fraction=abstain_fraction,
        )
        results.append({"held_out_session": held_out, **result})
    return results


def _finite_for_json(value):
    if isinstance(value, dict):
        return {key: _finite_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _select_frozen_candidate(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    trials: list[BinaryTrial],
    *,
    abstain_fraction: float,
) -> tuple[dict, list[dict]]:
    candidates = []
    for representation in FEATURE_NAMES:
        for l2 in L2_GRID:
            folds = _development_folds(
                features[representation],
                labels,
                trials,
                l2=l2,
                abstain_fraction=abstain_fraction,
            )
            balanced = [fold["balanced_accuracy"] for fold in folds]
            aucs = [fold["roc_auc"] for fold in folds]
            candidates.append(
                {
                    "representation": representation,
                    "l2": l2,
                    "min_balanced_accuracy": min(balanced),
                    "mean_balanced_accuracy": float(np.mean(balanced)),
                    "mean_roc_auc": float(np.mean(aucs)),
                    "folds": folds,
                }
            )
    # Worst-session performance comes first. The intent sessions remain untouched
    # by this selection, so these development folds may choose architecture/l2.
    selected = max(
        candidates,
        key=lambda item: (
            item["min_balanced_accuracy"],
            item["mean_balanced_accuracy"],
            item["mean_roc_auc"],
            -FEATURE_NAMES.index(item["representation"]),
            -item["l2"],
        ),
    )
    return selected, candidates


def _intent_transfer(
    values: np.ndarray,
    labels: np.ndarray,
    trials: list[BinaryTrial],
    *,
    l2: float,
    abstain_fraction: float,
) -> list[dict]:
    results = []
    for fit_session, test_session in (
        ("pv_labelled_06", "pv_labelled_08"),
        ("pv_labelled_08", "pv_labelled_06"),
    ):
        result = _evaluate_fold(
            values,
            labels,
            _indices(trials, [fit_session]),
            _indices(trials, [test_session]),
            l2=l2,
            abstain_fraction=abstain_fraction,
        )
        results.append({"fit_session": fit_session, "test_session": test_session, **result})
    return results


def _calibrated_intent(
    values: np.ndarray,
    labels: np.ndarray,
    trials: list[BinaryTrial],
    *,
    l2: float,
    per_label: int,
    abstain_fraction: float,
) -> list[dict]:
    index_by_id = {trial.identity: index for index, trial in enumerate(trials)}
    results = []
    for session in INTENT_SESSIONS:
        session_trials = [trial for trial in trials if trial.session == session]
        fit_trials, test_trials = calibration_split(session_trials, per_label=per_label)
        fit_indices = np.asarray([index_by_id[trial.identity] for trial in fit_trials])
        test_indices = np.asarray([index_by_id[trial.identity] for trial in test_trials])
        result = _evaluate_fold(
            values,
            labels,
            fit_indices,
            test_indices,
            l2=l2,
            abstain_fraction=abstain_fraction,
        )
        results.append(
            {
                "session": session,
                "calibration_trials": [trial.identity for trial in fit_trials],
                "test_trials": [trial.identity for trial in test_trials],
                **result,
            }
        )
    return results


def _print_pair(label: str, head: dict, baseline: dict) -> None:
    print(
        f"  {label}: head BA={head['balanced_accuracy']:.3f} "
        f"AUC={head['roc_auc']:.3f} d'={head['dprime']:.2f} "
        f"coverage={head['abstain']['coverage']:.3f}; "
        f"sum_kpa BA={baseline['balanced_accuracy']:.3f} "
        f"AUC={baseline['roc_auc']:.3f} d'={baseline['dprime']:.2f}"
    )


def run(args) -> dict:
    session_paths = [args.session_root / name for name in args.sessions]
    trials: list[BinaryTrial] = []
    audits = []
    for session in session_paths:
        loaded, audit = load_binary_trials(session)
        trials.extend(loaded)
        audits.append(audit)
    labels = np.asarray([trial.label for trial in trials], dtype=int)
    metadata = _cache_metadata(args.repo, session_paths, trials)

    features = None if args.refresh_cache else load_cache(args.cache, metadata)
    if features is None:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[features] extracting {len(trials)} trials on {device}")
        extractor = FrozenFeatureExtractor(args.repo, device)
        per_name: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
        for number, trial in enumerate(trials, start=1):
            extracted = extractor.trial(trial, args.batch_size)
            for name in FEATURE_NAMES:
                per_name[name].append(extracted[name])
            print(
                f"[features] {number:02d}/{len(trials)} {trial.identity} "
                f"({len(trial.frame_paths)} frames)",
                flush=True,
            )
        features = {name: np.stack(per_name[name]) for name in FEATURE_NAMES}
        save_cache(args.cache, metadata, features)
        print(f"[features] cached -> {args.cache}")
    else:
        print(f"[features] reused verified cache {args.cache}")

    # Hand-engineered baselines are rebuilt from capture JSON, not cached model output.
    sum_kpa = np.asarray([[trial.metrics["sum_kpa"]] for trial in trials], dtype=np.float32)
    pressure_metrics = np.asarray(
        [[trial.metrics[name] for name in METRIC_NAMES] for trial in trials], dtype=np.float32
    )
    all_features = {**features, "sum_kpa": sum_kpa, "pressure_metrics": pressure_metrics}

    selected, candidates = _select_frozen_candidate(
        features, labels, trials, abstain_fraction=args.abstain_fraction
    )
    representation, l2 = selected["representation"], selected["l2"]
    baseline_l2 = 1.0
    baseline_development = _development_folds(
        sum_kpa,
        labels,
        trials,
        l2=baseline_l2,
        abstain_fraction=args.abstain_fraction,
    )

    head_transfer = _intent_transfer(
        all_features[representation],
        labels,
        trials,
        l2=l2,
        abstain_fraction=args.abstain_fraction,
    )
    baseline_transfer = _intent_transfer(
        sum_kpa,
        labels,
        trials,
        l2=baseline_l2,
        abstain_fraction=args.abstain_fraction,
    )
    head_calibrated = _calibrated_intent(
        all_features[representation],
        labels,
        trials,
        l2=l2,
        per_label=args.calibrate_trials,
        abstain_fraction=args.abstain_fraction,
    )
    baseline_calibrated = _calibrated_intent(
        sum_kpa,
        labels,
        trials,
        l2=baseline_l2,
        per_label=args.calibrate_trials,
        abstain_fraction=args.abstain_fraction,
    )

    gate_reasons = []
    for head, baseline in zip(head_calibrated, baseline_calibrated):
        session = head["session"]
        if head["roc_auc"] < 0.90:
            gate_reasons.append(f"{session}: held-out AUC {head['roc_auc']:.3f} below 0.90")
        if head["dprime"] < 2.5:
            gate_reasons.append(f"{session}: held-out dprime {head['dprime']:.2f} below 2.5")
        if head["abstain"]["coverage"] < 0.80:
            gate_reasons.append(
                f"{session}: coverage {head['abstain']['coverage']:.3f} below 0.80"
            )
        decided_accuracy = head["abstain"]["accuracy_on_decided"]
        if decided_accuracy is None or decided_accuracy < 0.90:
            gate_reasons.append(
                f"{session}: decided accuracy {decided_accuracy} below 0.90"
            )
        if head["balanced_accuracy"] < baseline["balanced_accuracy"]:
            gate_reasons.append(
                f"{session}: head BA {head['balanced_accuracy']:.3f} below "
                f"sum_kpa {baseline['balanced_accuracy']:.3f}"
            )

    result = {
        "experiment": "frozen_pressurevision_two_band_head_v1",
        "role": "offline_proof_of_concept_not_deployment_authorization",
        "unit_of_analysis": "one median embedding per held press",
        "contact_policy": "excluded; MediaPipe pinch owns contact",
        "cache_metadata": metadata,
        "session_audit": audits,
        "feature_dimensions": {name: int(value.shape[1]) for name, value in features.items()},
        "development": {
            "sessions": list(DEVELOPMENT_SESSIONS),
            "selected": selected,
            "all_candidates": candidates,
            "sum_kpa_baseline_l2": baseline_l2,
            "sum_kpa_folds": baseline_development,
        },
        "intent_transfer": {
            "head": head_transfer,
            "sum_kpa": baseline_transfer,
        },
        "intent_per_session_calibration": {
            "calibration_trials_per_label": args.calibrate_trials,
            "head": head_calibrated,
            "sum_kpa": baseline_calibrated,
        },
        "fixed_rig_two_band_gate": {
            "accepted": not gate_reasons,
            "reasons": gate_reasons,
            "meaning": (
                "Only frozen-feature, current-operator, fixed-rig two-band feasibility; "
                "not cross-operator, three-band, continuous-force, or actuation evidence."
            ),
        },
    }
    result = _finite_for_json(result)

    print(
        f"\n[selected] {representation}, l2={l2:g}; development worst BA "
        f"{selected['min_balanced_accuracy']:.3f}, mean BA "
        f"{selected['mean_balanced_accuracy']:.3f}"
    )
    print("[intent transfer]")
    for head, baseline in zip(head_transfer, baseline_transfer):
        _print_pair(f"{head['fit_session']} -> {head['test_session']}", head, baseline)
    print("[three-press-per-level calibration]")
    for head, baseline in zip(head_calibrated, baseline_calibrated):
        _print_pair(head["session"], head, baseline)
    verdict = result["fixed_rig_two_band_gate"]
    print(f"[verdict] {'GO' if verdict['accepted'] else 'NO-GO'}")
    for reason in verdict["reasons"]:
        print(f"  - {reason}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[report] wrote {args.out}")
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
