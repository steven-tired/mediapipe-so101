#!/usr/bin/env python
"""Evaluate ACT checkpoints on held-out LeRobot episodes without augmentation."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION


DEFAULT_DATASET_ROOT = Path(
    "/home/zhuokai/hand-teleop/datasets/hand_tracking_pv_carton_phase_b"
)
DEFAULT_REPO_ID = "stevenzenith/hand_tracking_pv_carton_phase_b"
DEFAULT_EPISODES = [0, 5, 10, 16, 23, 26]
HORIZONS = (1, 5, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--episodes",
        default=",".join(str(ep) for ep in DEFAULT_EPISODES),
        help="Comma-separated sanitized episode indices.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for checkpoint_dir in (run_dir / "checkpoints").glob("[0-9]*"):
        model_dir = checkpoint_dir / "pretrained_model"
        if checkpoint_dir.name.isdigit() and (model_dir / "model.safetensors").is_file():
            checkpoints.append((int(checkpoint_dir.name), model_dir))
    if not checkpoints:
        raise FileNotFoundError(f"No numeric ACT checkpoints found under {run_dir / 'checkpoints'}")
    return sorted(checkpoints)


def make_validation_dataset(
    first_checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    episodes: list[int],
) -> LeRobotDataset:
    policy_cfg = PreTrainedConfig.from_pretrained(first_checkpoint)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    return LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend="torchcodec",
        return_uint8=True,
    )


def add_metric(store: dict, episode: int, name: str, total: float, count: int) -> None:
    store[episode][name][0] += total
    store[episode][name][1] += count


def episode_balanced_mean(store: dict, name: str) -> float:
    values = [metrics[name][0] / metrics[name][1] for metrics in store.values() if metrics[name][1]]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: LeRobotDataset,
    loader: DataLoader,
    device: str,
    max_batches: int | None,
) -> dict[str, float | int]:
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = device
    policy_cfg.use_amp = device.startswith("cuda")
    policy_cfg.pretrained_path = checkpoint
    policy = ACTPolicy.from_pretrained(checkpoint, config=policy_cfg).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    metrics = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            raw_action = batch[ACTION].clone()
            action_is_pad = batch["action_is_pad"].bool().clone()
            episode_indices = batch["episode_index"].reshape(-1).tolist()
            for camera_key in dataset.meta.camera_keys:
                if batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float().div_(255.0)

            processed = preprocessor(batch)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                prediction = policy.predict_action_chunk(processed)
            prediction = postprocessor(prediction)

            for row, episode in enumerate(episode_indices):
                target = raw_action[row]
                pred = prediction[row]
                valid = ~action_is_pad[row]
                for horizon in HORIZONS:
                    horizon_valid = valid[:horizon]
                    error = (pred[:horizon] - target[:horizon]).abs()
                    add_metric(
                        metrics,
                        int(episode),
                        f"mae_h{horizon}",
                        error[horizon_valid].sum().item(),
                        int(horizon_valid.sum().item()) * error.shape[-1],
                    )

                h1_valid = valid[:1]
                h1_error = (pred[:1] - target[:1]).abs()
                add_metric(
                    metrics,
                    int(episode),
                    "body_mae_h1",
                    h1_error[h1_valid, :5].sum().item(),
                    int(h1_valid.sum().item()) * 5,
                )
                add_metric(
                    metrics,
                    int(episode),
                    "gripper_mae_h1",
                    h1_error[h1_valid, 5].sum().item(),
                    int(h1_valid.sum().item()),
                )
                close_h1 = h1_valid & (target[:1, 5] < 90.0)
                add_metric(
                    metrics,
                    int(episode),
                    "gripper_close_mae_h1",
                    h1_error[close_h1, 5].sum().item(),
                    int(close_h1.sum().item()),
                )
                close_h20 = valid[:20] & (target[:20, 5] < 90.0)
                h20_error = (pred[:20] - target[:20]).abs()
                add_metric(
                    metrics,
                    int(episode),
                    "gripper_close_mae_h20",
                    h20_error[close_h20, 5].sum().item(),
                    int(close_h20.sum().item()),
                )
                sample_count += 1

    row = {
        "samples": sample_count,
        "episodes": len(metrics),
        "mae_h1": episode_balanced_mean(metrics, "mae_h1"),
        "mae_h5": episode_balanced_mean(metrics, "mae_h5"),
        "mae_h20": episode_balanced_mean(metrics, "mae_h20"),
        "body_mae_h1": episode_balanced_mean(metrics, "body_mae_h1"),
        "gripper_mae_h1": episode_balanced_mean(metrics, "gripper_mae_h1"),
        "gripper_close_mae_h1": episode_balanced_mean(metrics, "gripper_close_mae_h1"),
        "gripper_close_mae_h20": episode_balanced_mean(metrics, "gripper_close_mae_h20"),
    }
    gripper_selection_mae = row["gripper_close_mae_h1"]
    if math.isnan(gripper_selection_mae):
        gripper_selection_mae = row["gripper_mae_h1"]
    row["selection_score"] = (row["body_mae_h1"] + gripper_selection_mae) / 2

    del policy, preprocessor, postprocessor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; held-out evaluation requires the local GPU.")

    episodes = [int(value) for value in args.episodes.split(",")]
    checkpoints = find_checkpoints(args.run_dir)
    dataset = make_validation_dataset(checkpoints[0][1], args.dataset_root, args.repo_id, episodes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    for step, checkpoint in checkpoints:
        row = evaluate_checkpoint(checkpoint, dataset, loader, args.device, args.max_batches)
        row = {"step": step, **row}
        rows.append(row)
        print(
            f"step={step:06d} score={row['selection_score']:.3f} "
            f"h1={row['mae_h1']:.3f} h5={row['mae_h5']:.3f} h20={row['mae_h20']:.3f} "
            f"body_h1={row['body_mae_h1']:.3f} close_grip_h1={row['gripper_close_mae_h1']:.3f}",
            flush=True,
        )

    output_csv = args.output_csv or args.run_dir / "heldout_checkpoint_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: row["selection_score"])
    print(f"wrote {output_csv}")
    print(
        f"offline candidate: step={best['step']:06d}, score={best['selection_score']:.3f}. "
        "This ranks checkpoints; it does not prove closed-loop grasp success."
    )


if __name__ == "__main__":
    main()
