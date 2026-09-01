#!/usr/bin/env python
"""Minimal ACT fine-tuning loop using reviewed recovery labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES


CHECKPOINT = Path(
    "/home/zhuokai/hand-teleop/training/phase_c_act_augmented/outputs/"
    "act_phase_c_aug_holdout_20260824_150345/checkpoints/050000/pretrained_model"
)
PHASE_B_ROOT = Path("/home/zhuokai/hand-teleop/datasets/hand_tracking_pv_carton_phase_b")
MIDDLE_ROOT = Path("/home/zhuokai/hand-teleop/datasets/hand_tracking_pv_carton_middle_standard")
PHASE_B_EPISODES = [1, 9, 14, 20, 26]
MIDDLE_EPISODES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]

# episode: (J, K, q_stable); absent episodes are reviewed no-slip demonstrations.
PHASE_B_RECOVERY = {
    14: (13.3, 15.9, 23.033655166625977),
    26: (10.3, 15.9, 23.0723819732666),
}
MIDDLE_RECOVERY = {
    7: (12.7, 17.8, 25.6190242767334),
    10: (10.2, 16.9, 25.11787986755371),
    11: (15.6, 21.2, 21.398475646972656),
}


class ReviewedDataset(Dataset):
    def __init__(self, dataset: LeRobotDataset, recovery: dict[int, tuple[float, float, float]]):
        self.dataset = dataset
        self.recovery = recovery
        self.fps = dataset.fps

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self.dataset[index]
        episode = int(source["episode_index"])
        future_t = source["timestamp"] + torch.arange(source[ACTION].shape[0]) / self.fps
        recovery_mask = torch.zeros_like(future_t, dtype=torch.bool)
        stable_mask = torch.zeros_like(future_t, dtype=torch.bool)
        q_stable = 0.0
        if episode in self.recovery:
            j_time, k_time, q_stable = self.recovery[episode]
            recovery_mask = (future_t >= j_time - 1e-3) & (future_t <= k_time + 1e-3)
            stable_mask = (future_t >= k_time - 1e-3) & (source[ACTION][:, 5] < 90.0)

        return {
            "observation.state": source["observation.state"][:6],
            "observation.images.front": source["observation.images.front"],
            "observation.images.side": source["observation.images.side"],
            ACTION: source[ACTION],
            "action_is_pad": source["action_is_pad"],
            "recovery_mask": recovery_mask,
            "stable_mask": stable_mask,
            "q_stable": torch.tensor(q_stable, dtype=torch.float32),
        }


def load_dataset(
    repo_id: str,
    root: Path,
    episodes: list[int],
    recovery: dict[int, tuple[float, float, float]],
    config: PreTrainedConfig,
    image_transforms=None,
) -> ReviewedDataset:
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        image_transforms=image_transforms,
        video_backend="torchcodec",
        return_uint8=image_transforms is None,
    )
    return ReviewedDataset(dataset, recovery)


def save_checkpoint(policy, preprocessor, postprocessor, output_dir: Path, step: int) -> None:
    model_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    preprocessor.save_pretrained(model_dir)
    postprocessor.save_pretrained(model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    config.device = args.device
    config.use_amp = args.device.startswith("cuda")
    image_transforms = None
    if args.from_scratch:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.transforms import ImageTransforms

        train_config = TrainPipelineConfig.from_pretrained(CHECKPOINT)
        image_transforms = ImageTransforms(train_config.dataset.image_transforms)
        config.pretrained_path = None
        policy = ACTPolicy(config)
    else:
        policy = ACTPolicy.from_pretrained(CHECKPOINT, config=config)
    policy = policy.to(args.device).train()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    train_data = ConcatDataset(
        [
            load_dataset(
                "stevenzenith/hand_tracking_pv_carton_phase_b",
                PHASE_B_ROOT,
                PHASE_B_EPISODES,
                PHASE_B_RECOVERY,
                config,
                image_transforms,
            ),
            load_dataset(
                "stevenzenith/hand_tracking_pv_carton_middle_standard",
                MIDDLE_ROOT,
                MIDDLE_EPISODES,
                MIDDLE_RECOVERY,
                config,
                image_transforms,
            ),
        ]
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    action_stats = preprocessor.steps[-1].stats[ACTION]
    q_mean = torch.tensor(action_stats["mean"][5], device=args.device)
    q_std = torch.tensor(action_stats["std"][5], device=args.device)

    print(f"train_frames={len(train_data)} train_episodes=15 recovered_episodes=5")
    batches = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        recovery_mask = batch.pop("recovery_mask").to(args.device)
        stable_mask = batch.pop("stable_mask").to(args.device)
        q_stable = batch.pop("q_stable").to(args.device)
        for key in config.image_features:
            batch[key] = batch[key].float().div_(255.0)
        batch = preprocessor(batch)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.use_amp):
            model_batch = dict(batch)
            model_batch[OBS_IMAGES] = [batch[key] for key in config.image_features]
            actions_hat, (mu_hat, log_sigma_x2_hat) = policy.model(model_batch)
            abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            valid = ~batch["action_is_pad"]
            l1_loss = abs_err[valid].mean()
            kld_loss = (
                -0.5 * (1 + log_sigma_x2_hat - mu_hat.square() - log_sigma_x2_hat.exp())
            ).sum(-1).mean()
            recovery_valid = recovery_mask & valid
            recovery_loss = (
                abs_err[..., 5][recovery_valid].mean() if recovery_valid.any() else l1_loss.new_zeros(())
            )
            stable_valid = stable_mask & valid
            q_stable_norm = (q_stable - q_mean) / q_std
            overclose = F.relu(q_stable_norm[:, None] - actions_hat[..., 5])
            overclose_loss = (
                overclose[stable_valid].mean() if stable_valid.any() else l1_loss.new_zeros(())
            )
            loss = l1_loss + config.kl_weight * kld_loss + recovery_loss + 0.25 * overclose_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0:
            print(
                f"step={step} loss={loss.detach().item():.4f} l1={l1_loss.detach().item():.4f} "
                f"recovery={recovery_loss.detach().item():.4f} "
                f"overclose={overclose_loss.detach().item():.4f}",
                flush=True,
            )
        if step % args.save_freq == 0 or step == args.steps:
            save_checkpoint(policy, preprocessor, postprocessor, args.output_dir, step)


if __name__ == "__main__":
    main()
