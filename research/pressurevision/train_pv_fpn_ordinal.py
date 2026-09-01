#!/usr/bin/env python3
"""Fine-tune only the PressureVision FPN decoder plus the ordinal head.

This is stage two of the fixed-setup intent model.  It starts from a selected
head-only checkpoint, keeps the complete encoder/C5 and the released 1x1
9-bin segmentation head frozen, and updates only the FPN decoder plus the
18D ordinal head.  The encoder is always in eval mode, so all 53 encoder
BatchNorm modules retain their released running statistics.

Training and validation are explicit whole sessions.  A small, changing set of
steady frames is sampled per trial each epoch to avoid treating 45 adjacent
frames as independent observations.  Final metrics use every steady frame and
one median severity decision per trial.  No camera, robot, contact classifier,
abstention calibration, C5 layer, or final-test session is touched.
"""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("<workspace>")
DEFAULT_REPO = WORKSPACE / "pressurevision"
DEFAULT_SESSION_ROOT = WORKSPACE / "scratch_lepton"
LABELS = ("light", "medium", "hard")

sys.path.insert(0, str(SCRIPT_DIR))
from pressurevision_probe import load_model, preprocess  # noqa: E402
from train_pv_ordinal_head import (  # noqa: E402
    FEATURE_DIM,
    FrameRecord,
    OrdinalPressureHead,
    classification_metrics,
    collect_records,
    ordinal_loss,
    resolve_device,
    trial_predictions,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--train-sessions", required=True)
    parser.add_argument("--validation-session", required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-trial", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.train_sessions = tuple(
        part.strip() for part in args.train_sessions.split(",") if part.strip()
    )
    if not args.train_sessions:
        parser.error("--train-sessions must not be empty")
    if args.validation_session in args.train_sessions:
        parser.error("validation session must not be a training session")
    if len(set(args.train_sessions)) != len(args.train_sessions):
        parser.error("--train-sessions contains a duplicate")
    for name in ("batch_size", "frames_per_trial", "epochs", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.decoder_learning_rate <= 0 or args.head_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_head_checkpoint(
    path: Path,
    *,
    base_checkpoint_sha256: str,
    train_sessions: tuple[str, ...],
    validation_session: str,
) -> tuple[OrdinalPressureHead, dict]:
    if not path.is_file():
        raise SystemExit(f"missing head checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise SystemExit("head checkpoint was built from a different PressureVision checkpoint")
    if set(checkpoint.get("train_sessions", [])) != set(train_sessions):
        raise SystemExit("head checkpoint training sessions do not match --train-sessions")
    if checkpoint.get("validation_session") != validation_session:
        raise SystemExit("head checkpoint validation session does not match")
    head = OrdinalPressureHead(
        input_dim=FEATURE_DIM,
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head, checkpoint


def _pool_avg_max(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            F.adaptive_avg_pool2d(tensor, 1).flatten(1),
            F.adaptive_max_pool2d(tensor, 1).flatten(1),
        ),
        dim=1,
    )


class FPNOrdinalModel(nn.Module):
    """Frozen encoder/head with a trainable FPN decoder and ordinal head."""

    def __init__(self, pressurevision: nn.Module, ordinal_head: OrdinalPressureHead):
        super().__init__()
        self.pressurevision = pressurevision
        self.ordinal_head = ordinal_head
        for parameter in self.pressurevision.parameters():
            parameter.requires_grad_(False)
        for parameter in self.pressurevision.decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.ordinal_head.parameters():
            parameter.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pressurevision.encoder.eval()
        self.pressurevision.segmentation_head.eval()
        self.pressurevision.decoder.train(mode)
        self.ordinal_head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoder_features = self.pressurevision.encoder(images)
        decoder = self.pressurevision.decoder(encoder_features)
        logits = self.pressurevision.segmentation_head(decoder)
        features = _pool_avg_max(logits)
        if features.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"expected {FEATURE_DIM} pooled logits, got {features.shape[1]}")
        return self.ordinal_head(features)


class FrameDataset(Dataset):
    def __init__(self, records: list[FrameRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        frame = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not decode {record.path}")
        height, width = frame.shape[:2]
        _resized, tensor = preprocess(frame, (0, 0, width, height))
        return tensor.squeeze(0), record.label


def select_trial_frames(
    records: list[FrameRecord],
    *,
    per_trial: int,
    seed: int,
    random: bool,
) -> list[FrameRecord]:
    """Select equal numbers from every trial; never split a trial by session."""
    grouped: dict[tuple[str, int], list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault((record.session, record.trial_index), []).append(record)
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted(grouped):
        trial = sorted(grouped[key], key=lambda item: item.phase_index)
        count = min(per_trial, len(trial))
        if random:
            indices = np.sort(rng.choice(len(trial), size=count, replace=False))
        else:
            indices = np.linspace(0, len(trial) - 1, num=count, dtype=int)
        selected.extend(trial[int(index)] for index in indices)
    return selected


def _autocast(device: str):
    return torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.float16 if device.startswith("cuda") else torch.bfloat16,
        enabled=device.startswith("cuda"),
    )


def evaluate_records(
    model: FPNOrdinalModel,
    records: list[FrameRecord],
    *,
    batch_size: int,
    device: str,
) -> dict:
    model.eval()
    loader = DataLoader(
        FrameDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    scores = []
    predictions = []
    total_loss = 0.0
    seen = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device):
                batch_scores, logits = model(images)
                loss = ordinal_loss(logits, labels)
                probabilities = model.ordinal_head.probabilities_from_score(batch_scores)
            scores.append(batch_scores.float().cpu().numpy())
            predictions.append(probabilities.argmax(dim=1).cpu().numpy())
            total_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
    score_values = np.concatenate(scores)
    frame_predictions = np.concatenate(predictions)
    labels = np.asarray([record.label for record in records], dtype=int)
    trial_truth, trial_estimates = trial_predictions(
        score_values,
        labels,
        np.asarray([record.session for record in records]),
        np.asarray([record.trial_index for record in records]),
        model.ordinal_head,
        device,
    )
    return {
        "loss": total_loss / seen,
        "frame": classification_metrics(labels, frame_predictions),
        "trial": classification_metrics(trial_truth, trial_estimates),
    }


def _trainable_counts(model: FPNOrdinalModel) -> dict:
    return {
        "encoder": sum(p.numel() for p in model.pressurevision.encoder.parameters() if p.requires_grad),
        "fpn_decoder": sum(
            p.numel() for p in model.pressurevision.decoder.parameters() if p.requires_grad
        ),
        "released_1x1_head": sum(
            p.numel()
            for p in model.pressurevision.segmentation_head.parameters()
            if p.requires_grad
        ),
        "ordinal_head": sum(p.numel() for p in model.ordinal_head.parameters() if p.requires_grad),
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args) -> dict:
    device = resolve_device(args.device)
    session_names = args.train_sessions + (args.validation_session,)
    paths = [args.session_root / name for name in session_names]
    records, audits = collect_records(paths)
    train_records = [record for record in records if record.session in args.train_sessions]
    validation_records = [
        record for record in records if record.session == args.validation_session
    ]
    if not train_records or not validation_records:
        raise SystemExit("explicit train/validation split produced an empty side")

    base_checkpoint = args.repo / "data/model/paper_59.pt"
    base_sha = _sha256(base_checkpoint)
    ordinal_head, source_head = load_head_checkpoint(
        args.head_checkpoint,
        base_checkpoint_sha256=base_sha,
        train_sessions=args.train_sessions,
        validation_session=args.validation_session,
    )
    pressurevision, _config = load_model(args.repo, device)
    model = FPNOrdinalModel(pressurevision, ordinal_head).to(device)
    counts = _trainable_counts(model)
    if counts["encoder"] or counts["released_1x1_head"]:
        raise AssertionError(f"frozen boundary failed: {counts}")
    print(f"trainable parameters: {counts}", flush=True)

    _seed_everything(args.seed)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.pressurevision.decoder.parameters(),
                "lr": args.decoder_learning_rate,
            },
            {"params": model.ordinal_head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    fixed_validation = select_trial_frames(
        validation_records,
        per_trial=args.frames_per_trial,
        seed=args.seed,
        random=False,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        sampled_train = select_trial_frames(
            train_records,
            per_trial=args.frames_per_trial,
            seed=args.seed + epoch,
            random=True,
        )
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            FrameDataset(sampled_train),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
        )
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                _scores, logits = model(images)
                loss = ordinal_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
        validation = evaluate_records(
            model,
            fixed_validation,
            batch_size=args.batch_size,
            device=device,
        )
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
        print(
            f"epoch {epoch:02d}: train_loss={total_loss / seen:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_trial_BA={validation['trial']['balanced_accuracy']:.3f}",
            flush=True,
        )
        if validation["loss"] < best_loss - 1e-5:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                "decoder": copy.deepcopy(model.pressurevision.decoder.state_dict()),
                "ordinal_head": copy.deepcopy(model.ordinal_head.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break
    assert best_state is not None
    model.pressurevision.decoder.load_state_dict(best_state["decoder"])
    model.ordinal_head.load_state_dict(best_state["ordinal_head"])

    print("evaluating every steady frame at the selected epoch", flush=True)
    train_result = evaluate_records(
        model, train_records, batch_size=args.batch_size, device=device
    )
    validation_result = evaluate_records(
        model, validation_records, batch_size=args.batch_size, device=device
    )
    thresholds = model.ordinal_head.thresholds().detach().cpu().tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "fpn_ordinal_best.pt"
    torch.save(
        {
            "schema_version": 1,
            "stage": "fpn_decoder_plus_ordinal_head",
            "base_checkpoint_sha256": base_sha,
            "source_head_checkpoint": str(args.head_checkpoint),
            "source_head_validation_session": source_head["validation_session"],
            "train_sessions": args.train_sessions,
            "validation_session": args.validation_session,
            "labels": LABELS,
            "hidden_dim": model.ordinal_head.hidden_dim,
            "dropout": model.ordinal_head.dropout,
            "decoder_state_dict": {
                key: value.detach().cpu()
                for key, value in model.pressurevision.decoder.state_dict().items()
            },
            "ordinal_head_state_dict": {
                key: value.detach().cpu()
                for key, value in model.ordinal_head.state_dict().items()
            },
        },
        checkpoint_path,
    )
    report = {
        "schema_version": 1,
        "status": "held_out_validation_for_model_selection_not_final_test",
        "stage": "fpn_decoder_plus_ordinal_head",
        "task": "ordered_operator_grip_intent_not_force_measurement",
        "contact_policy": "MediaPipe owns contact",
        "split_policy": "train fixed_01-03; validate fixed_04; fixed_05 unopened",
        "device": device,
        "train_sessions": list(args.train_sessions),
        "validation_session": args.validation_session,
        "session_audits": audits,
        "trainable_parameters": counts,
        "frozen_policy": {
            "encoder_and_C5": "requires_grad false; eval mode; BatchNorm state frozen",
            "fpn_decoder": "trainable; GroupNorm; no BatchNorm modules",
            "released_1x1_9bin_head": "requires_grad false; eval mode",
            "ordinal_head": "trainable",
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "frames_per_trial_per_epoch": args.frames_per_trial,
            "epochs": args.epochs,
            "patience": args.patience,
            "decoder_learning_rate": args.decoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "mixed_precision": device.startswith("cuda"),
        },
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "thresholds": {"light_medium": thresholds[0], "medium_hard": thresholds[1]},
        "train_all_steady_frames": train_result,
        "validation_all_steady_frames": validation_result,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "limitations": [
            "fixed_04 is model-selection validation, not a final test",
            "fixed_05 remains sealed and must be evaluated only after architecture is frozen",
            "no encoder, C5, released pressure head, contact classifier, or robot path was trained",
        ],
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "validation trial BA="
        f"{validation_result['trial']['balanced_accuracy']:.3f} "
        f"best_epoch={best_epoch}"
    )
    print(f"wrote {report_path}")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
