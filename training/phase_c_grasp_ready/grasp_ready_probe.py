#!/usr/bin/env python
"""Offline dual-view probe for autonomous grasp-readiness observability."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


GRIPPER_INDEX = 5
OPEN_THRESHOLD = 90.0
HELDOUT_EPISODES = (0, 5, 10, 16, 23, 26)
CAMERA_KEYS = ("observation.images.front", "observation.images.side")


@dataclass(frozen=True)
class LabelSet:
    ready_positive: np.ndarray
    open_negative: np.ndarray
    event_rows: np.ndarray


@dataclass(frozen=True)
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        logits = normalized @ self.weight + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def fit_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int = 120,
) -> LinearHead:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = torch.from_numpy((features - mean) / scale)
    targets = torch.from_numpy(labels).reshape(-1, 1)

    torch.manual_seed(seed)
    layer = torch.nn.Linear(features.shape[1], 1)
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(normalized), targets)
        loss.backward()
        optimizer.step()

    return LinearHead(
        mean=mean,
        scale=scale,
        weight=layer.weight.detach().numpy().reshape(-1).copy(),
        bias=float(layer.bias.detach().item()),
    )


def zero_false_threshold(scores: np.ndarray, negative_mask: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    negative_scores = scores[np.asarray(negative_mask, dtype=bool)]
    if len(negative_scores) == 0:
        return 0.5
    return float(np.nextafter(negative_scores.max(), np.inf))


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    np.savez(path, **cache)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def build_episode_folds(
    episodes: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    episodes = np.asarray(episodes)
    return [
        (
            int(episode),
            np.flatnonzero(episodes != episode),
            np.flatnonzero(episodes == episode),
        )
        for episode in np.unique(episodes)
    ]


def evaluate_scores(
    episodes: np.ndarray,
    frames: np.ndarray,
    labels: LabelSet,
    scores: np.ndarray,
    *,
    threshold: float,
    legacy_open_phase: np.ndarray | None = None,
) -> dict[str, object]:
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    scores = np.asarray(scores)
    triggered = scores >= threshold
    false_trigger = triggered & labels.open_negative
    if legacy_open_phase is None:
        legacy_open_phase = labels.open_negative
    legacy_open_phase = np.asarray(legacy_open_phase, dtype=bool)
    sustained_pairs = 0
    for episode in np.unique(episodes):
        rows = np.flatnonzero((episodes == episode) & false_trigger)
        episode_frames = np.sort(frames[rows])
        sustained_pairs += int(np.sum(np.diff(episode_frames) == 1))

    event_hits = 0
    initial_hits = 0
    first_hit_offsets: list[int] = []
    seen_episodes: set[int] = set()
    for event_row in labels.event_rows:
        episode = int(episodes[event_row])
        event_frame = int(frames[event_row])
        window = np.flatnonzero(
            (episodes == episode)
            & (frames >= event_frame - 2)
            & (frames <= event_frame + 2)
        )
        hit_rows = window[triggered[window]]
        hit = len(hit_rows) > 0
        event_hits += int(hit)
        if episode not in seen_episodes:
            seen_episodes.add(episode)
            initial_hits += int(hit)
            if hit:
                first_hit_offsets.append(int(frames[hit_rows].min() - event_frame))

    median_offset = (
        float(np.median(first_hit_offsets)) if first_hit_offsets else None
    )
    return {
        "initial_event_hits": initial_hits,
        "initial_events": len(seen_episodes),
        "all_event_hits": event_hits,
        "all_events": int(len(labels.event_rows)),
        "first_hit_offsets": first_hit_offsets,
        "median_first_hit_offset": median_offset,
        "open_negative_rows": int(labels.open_negative.sum()),
        "false_trigger_frames": int(false_trigger.sum()),
        "sustained_false_trigger_pairs": sustained_pairs,
        "legacy_open_phase_rows": int(legacy_open_phase.sum()),
        "legacy_false_trigger_frames": int((triggered & legacy_open_phase).sum()),
    }


def extract_feature_cache(
    *,
    dataset_root: Path,
    repo_id: str,
    cache_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights, resnet18

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=list(range(30)),
        video_backend="torchcodec",
        return_uint8=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
    )
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.to(device).eval()
    preprocess = weights.transforms()

    collected: dict[str, list[np.ndarray]] = {
        "episode": [],
        "frame": [],
        "state": [],
        "action": [],
        "front": [],
        "side": [],
    }
    with torch.inference_mode():
        for batch in loader:
            images = torch.cat(
                [preprocess(batch[key]) for key in CAMERA_KEYS],
                dim=0,
            ).to(device, non_blocking=True)
            embeddings = backbone(images).cpu().numpy().astype(np.float32, copy=False)
            count = len(batch["episode_index"])
            collected["episode"].append(batch["episode_index"].numpy().astype(np.int64))
            collected["frame"].append(batch["frame_index"].numpy().astype(np.int64))
            collected["state"].append(batch["observation.state"].numpy().astype(np.float32))
            collected["action"].append(batch["action"].numpy().astype(np.float32))
            collected["front"].append(embeddings[:count])
            collected["side"].append(embeddings[count:])

    cache = {key: np.concatenate(parts) for key, parts in collected.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_cache(cache_path, cache)
    return cache


def _subset_labels(cache: dict[str, np.ndarray], rows: np.ndarray) -> LabelSet:
    return make_labels(
        cache["episode"][rows],
        cache["frame"][rows],
        cache["state"][rows],
        cache["action"][rows],
    )


def _legacy_open_phase(cache: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return (
        (cache["state"][rows, GRIPPER_INDEX] >= OPEN_THRESHOLD)
        & (cache["action"][rows, GRIPPER_INDEX] >= OPEN_THRESHOLD)
    )


def _cross_validated_scores(
    features: np.ndarray,
    cache: dict[str, np.ndarray],
    labels: LabelSet,
    train_rows: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> np.ndarray:
    train_episodes = cache["episode"][train_rows]
    scores = np.full(len(train_rows), np.nan, dtype=np.float64)
    selected = labels.ready_positive | labels.open_negative
    for fold_index, (_, fold_train, fold_validation) in enumerate(
        build_episode_folds(train_episodes)
    ):
        fit_rows = fold_train[selected[fold_train]]
        head = fit_linear_head(
            features[train_rows[fit_rows]],
            labels.ready_positive[fit_rows].astype(np.float32),
            seed=seed + fold_index,
            epochs=epochs,
        )
        scores[fold_validation] = head.predict_proba(features[train_rows[fold_validation]])
    return scores


def _fit_final_head(
    train_features: np.ndarray,
    train_labels: LabelSet,
    *,
    seed: int,
    epochs: int,
) -> LinearHead:
    selected = train_labels.ready_positive | train_labels.open_negative
    return fit_linear_head(
        train_features[selected],
        train_labels.ready_positive[selected].astype(np.float32),
        seed=seed,
        epochs=epochs,
    )


def _write_prediction_csv(
    path: Path,
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    labels: LabelSet,
    scores: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "episode",
                "frame",
                "ready_positive",
                "open_negative",
                "legacy_open_phase",
                "static_score",
                "history_score",
            ],
        )
        writer.writeheader()
        for local_row, source_row in enumerate(rows):
            writer.writerow(
                {
                    "episode": int(cache["episode"][source_row]),
                    "frame": int(cache["frame"][source_row]),
                    "ready_positive": int(labels.ready_positive[local_row]),
                    "open_negative": int(labels.open_negative[local_row]),
                    "legacy_open_phase": int(
                        cache["state"][source_row, GRIPPER_INDEX] >= OPEN_THRESHOLD
                        and cache["action"][source_row, GRIPPER_INDEX] >= OPEN_THRESHOLD
                    ),
                    "static_score": float(scores["static"][local_row]),
                    "history_score": float(scores["history"][local_row]),
                }
            )


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def evaluate_cache(
    *,
    cache_path: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
) -> dict[str, object]:
    cache = load_cache(cache_path)
    features = assemble_features(cache)
    heldout_mask = np.isin(cache["episode"], HELDOUT_EPISODES)
    train_rows = np.flatnonzero(~heldout_mask)
    heldout_rows = np.flatnonzero(heldout_mask)
    train_labels = _subset_labels(cache, train_rows)
    heldout_labels = _subset_labels(cache, heldout_rows)

    contract = {
        "rows": int(len(cache["episode"])),
        "episodes": int(len(np.unique(cache["episode"]))),
        "train_episodes": int(len(np.unique(cache["episode"][train_rows]))),
        "heldout_episodes": [int(value) for value in np.unique(cache["episode"][heldout_rows])],
        "train_events": int(len(train_labels.event_rows)),
        "heldout_events": int(len(heldout_labels.event_rows)),
        "train_open_negative_rows": int(train_labels.open_negative.sum()),
        "heldout_open_negative_rows": int(heldout_labels.open_negative.sum()),
        "train_legacy_open_phase_rows": int(_legacy_open_phase(cache, train_rows).sum()),
        "heldout_legacy_open_phase_rows": int(
            _legacy_open_phase(cache, heldout_rows).sum()
        ),
    }
    expected = {
        "rows": 6640,
        "episodes": 30,
        "train_episodes": 24,
        "heldout_episodes": list(HELDOUT_EPISODES),
        "train_events": 25,
        "heldout_events": 6,
        "train_open_negative_rows": 1368,
        "heldout_open_negative_rows": 273,
        "train_legacy_open_phase_rows": 1443,
        "heldout_legacy_open_phase_rows": 291,
    }
    if contract != expected:
        raise ValueError(f"dataset contract mismatch: expected {expected}, got {contract}")

    output_dir.mkdir(parents=True, exist_ok=False)
    _save_json(
        output_dir / "label_manifest.json",
        {
            **contract,
            "cache_path": str(cache_path.resolve()),
            "ready_window": [-2, 2],
            "feature_sources": {
                "static": ["front_rgb", "side_rgb", "body_state_0_4"],
                "history": [
                    "front_rgb_t",
                    "side_rgb_t",
                    "front_rgb_t_minus_2",
                    "side_rgb_t_minus_2",
                    "body_state_0_4_t",
                    "body_state_0_4_delta",
                ],
            },
            "excluded_inputs": [
                "gripper_state",
                "action",
                "pressurevision",
                "human_pinch",
                "episode_index",
                "frame_index",
                "timestamp",
            ],
            "feature_widths": {key: int(value.shape[1]) for key, value in features.items()},
        },
    )

    train_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    train_metrics: dict[str, dict[str, object]] = {}
    for variant in ("static", "history"):
        variant_scores = _cross_validated_scores(
            features[variant],
            cache,
            train_labels,
            train_rows,
            seed=seed,
            epochs=epochs,
        )
        threshold = zero_false_threshold(variant_scores, train_labels.open_negative)
        train_scores[variant] = variant_scores
        thresholds[variant] = threshold
        train_metrics[variant] = evaluate_scores(
            cache["episode"][train_rows],
            cache["frame"][train_rows],
            train_labels,
            variant_scores,
            threshold=threshold,
            legacy_open_phase=_legacy_open_phase(cache, train_rows),
        )

    winner = max(
        ("static", "history"),
        key=lambda variant: (
            int(train_metrics[variant]["initial_event_hits"]),
            variant == "static",
        ),
    )

    heldout_scores: dict[str, np.ndarray] = {}
    heldout_metrics: dict[str, dict[str, object]] = {}
    final_heads: dict[str, LinearHead] = {}
    for variant in ("static", "history"):
        head = _fit_final_head(
            features[variant][train_rows],
            train_labels,
            seed=seed + 1000,
            epochs=epochs,
        )
        variant_scores = head.predict_proba(features[variant][heldout_rows])
        final_heads[variant] = head
        heldout_scores[variant] = variant_scores
        heldout_metrics[variant] = evaluate_scores(
            cache["episode"][heldout_rows],
            cache["frame"][heldout_rows],
            heldout_labels,
            variant_scores,
            threshold=thresholds[variant],
            legacy_open_phase=_legacy_open_phase(cache, heldout_rows),
        )

    winning_metrics = heldout_metrics[winner]
    median_offset = winning_metrics["median_first_hit_offset"]
    passed = bool(
        winning_metrics["initial_event_hits"] >= 5
        and winning_metrics["false_trigger_frames"] == 0
        and winning_metrics["sustained_false_trigger_pairs"] == 0
        and median_offset is not None
        and -2 <= median_offset <= 2
        and all(np.isfinite(values).all() for values in heldout_scores.values())
    )
    summary = {
        "seed": seed,
        "epochs": epochs,
        "winner_frozen_from_train_oof": winner,
        "thresholds": thresholds,
        "train_oof": train_metrics,
        "heldout": heldout_metrics,
        "gate": "PASS" if passed else "FAIL",
        "gate_requirements": {
            "initial_event_hits": "at_least_5_of_6",
            "false_trigger_frames": 0,
            "sustained_false_trigger_pairs": 0,
            "median_first_hit_offset": "-2_to_2",
        },
    }
    _write_prediction_csv(
        output_dir / "train_oof_predictions.csv",
        cache,
        train_rows,
        train_labels,
        train_scores,
    )
    _write_prediction_csv(
        output_dir / "heldout_predictions.csv",
        cache,
        heldout_rows,
        heldout_labels,
        heldout_scores,
    )
    _save_json(output_dir / "summary.json", summary)
    if passed:
        from safetensors.numpy import save_file

        head = final_heads[winner]
        save_file(
            {
                "mean": head.mean,
                "scale": head.scale,
                "weight": head.weight,
                "bias": np.asarray([head.bias], dtype=np.float32),
            },
            output_dir / "probe_head.safetensors",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_extract_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--dataset-root",
            type=Path,
            default=Path("datasets/hand_tracking_pv_carton_phase_b"),
        )
        command.add_argument("--repo-id", default="stevenzenith/hand_tracking_pv_carton_phase_b")
        command.add_argument(
            "--cache-path",
            type=Path,
            default=Path("training/phase_c_grasp_ready/cache/resnet18_phase_b.npz"),
        )
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--num-workers", type=int, default=0)

    extract = subparsers.add_parser("extract")
    add_extract_arguments(extract)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache-path", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--epochs", type=int, default=120)
    run = subparsers.add_parser("run")
    add_extract_arguments(run)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--epochs", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"extract", "run"}:
        extract_feature_cache(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            cache_path=args.cache_path,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    if args.command in {"evaluate", "run"}:
        summary = evaluate_cache(
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


def assemble_features(
    cache: dict[str, np.ndarray],
    *,
    history_offset: int = 2,
) -> dict[str, np.ndarray]:
    episodes = np.asarray(cache["episode"])
    frames = np.asarray(cache["frame"])
    front = np.asarray(cache["front"], dtype=np.float32)
    side = np.asarray(cache["side"], dtype=np.float32)
    body = np.asarray(cache["state"], dtype=np.float32)[:, :GRIPPER_INDEX]
    history_rows = np.empty(len(episodes), dtype=np.int64)

    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(frames[rows])]
        for local_index, row in enumerate(rows):
            history_rows[row] = rows[max(0, local_index - history_offset)]

    static = np.concatenate([front, side, body], axis=1)
    history = np.concatenate(
        [
            front,
            side,
            front[history_rows],
            side[history_rows],
            body,
            body - body[history_rows],
        ],
        axis=1,
    )
    return {"static": static, "history": history}


def find_close_events(gripper_actions: np.ndarray) -> list[int]:
    actions = np.asarray(gripper_actions)
    return [
        index
        for index in range(1, len(actions))
        if actions[index - 1] >= 99.0 and OPEN_THRESHOLD <= actions[index] < 99.0
    ]


def make_labels(
    episodes: np.ndarray,
    frames: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
) -> LabelSet:
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    states = np.asarray(states)
    actions = np.asarray(actions)
    ready = np.zeros(len(episodes), dtype=bool)
    event_rows: list[int] = []

    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        order = np.argsort(frames[rows])
        rows = rows[order]
        for local_index in find_close_events(actions[rows, GRIPPER_INDEX]):
            event_rows.append(int(rows[local_index]))
            start = max(0, local_index - 2)
            stop = min(len(rows), local_index + 3)
            ready[rows[start:stop]] = True

    open_negative = (
        (states[:, GRIPPER_INDEX] >= OPEN_THRESHOLD)
        & (actions[:, GRIPPER_INDEX] >= OPEN_THRESHOLD)
        & ~ready
    )
    return LabelSet(
        ready_positive=ready,
        open_negative=open_negative,
        event_rows=np.asarray(sorted(event_rows), dtype=np.int64),
    )


if __name__ == "__main__":
    main()
