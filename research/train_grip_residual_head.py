#!/usr/bin/env python
"""Train an offline action-conditioned grip stability and effort scorer."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lerobot_teleoperator_so101_webcam.grip.runtime import (
    GRIP_CANDIDATE_DELTAS,
    GRIP_CANDIDATE_FEATURES,
    GripCandidateHead,
    grip_visual_features,
)
from lerobot_teleoperator_so101_webcam.paths import evidence_dir


EVIDENCE_ROOT = evidence_dir() / "phase_c_recovery_minimal"
DEFAULT_OUTPUT = Path("/tmp/grip_candidate_stability_effort_20260831.pt")
HISTORY_STEPS = 4
HIDDEN_DIM = 32
VISUAL_GAP_FRAMES = 5
MAX_PHASE_SAMPLES = 8
MAX_EVENT_SAMPLES = 4
EFFORT_HORIZON_S = 0.8
BOUNDARY_PAIR_MARGIN = 0.5
MIN_LOAD_FOR_LOOSEN = 60.0
BOUNDARY_TRIALS = {1001: 1, 1003: 3, 1004: 4}
HEAD_ONLY_TRIALS = {2001: 1, 2002: 2}
HELDOUT_TRIALS = {29, 40, 42, 45, 1004, 2002}

# stability: (target, phase). pre=before first intervention; post=after last.
# These are reviewed physical labels; invalid trials are absent.
HOLD_STABILITY_LABELS = {
    10: (0.0, "post"),
    13: (0.0, "pre"),
    15: (1.0, "post"),
    18: (0.0, "all"),
    19: (1.0, "all"),
    20: (1.0, "post"),
    21: (0.0, "pre"),
    22: (0.0, "all"),
    23: (0.0, "pre"),
    24: (0.0, "pre"),
    25: (0.0, "all"),
    26: (0.0, "all"),
    27: (1.0, "post"),
    28: (1.0, "post"),
    29: (0.0, "pre"),
    31: (1.0, "post"),
    32: (1.0, "all"),
    33: (1.0, "pre"),
    34: (1.0, "all"),
    36: (0.0, "post"),
    37: (0.0, "all"),
    39: (1.0, "post"),
    40: (1.0, "all"),
    41: (1.0, "post"),
    42: (0.0, "all"),
    43: (0.0, "post"),
    44: (1.0, "all"),
    45: (1.0, "post"),
}

# Only post-fix interventions whose sign is relative to the prior executed ACT q.
CANDIDATE_EVENT_LABELS = {
    27: ("loosen", 1.0),
    28: ("loosen", 1.0),
    29: ("tighten", 0.0),
    31: ("loosen", 1.0),
    36: ("tighten", 0.0),
    39: ("tighten", 1.0),
    41: ("loosen", 1.0),
    43: ("tighten", 0.0),
    45: ("loosen", 1.0),
    2001: ("loosen", 0.0),
    2002: ("loosen", 0.0),
}

# Deployment-matched view: the learned head starts only after ACT has lifted.
# Tighten remains a deterministic one-step rollback, not a learned candidate.
POST_LIFT_HOLD_STABILITY_LABELS = {
    18: (0.0, "all"),
    22: (0.0, "all"),
    25: (0.0, "all"),
    27: (1.0, "post"),
    28: (1.0, "post"),
    31: (1.0, "post"),
    32: (1.0, "all"),
    33: (1.0, "pre"),
    34: (1.0, "all"),
    37: (0.0, "all"),
    40: (1.0, "all"),
    41: (1.0, "post"),
    42: (0.0, "all"),
    44: (1.0, "all"),
    45: (1.0, "post"),
}
POST_LIFT_CANDIDATE_EVENT_LABELS = {
    trial: label
    for trial, label in CANDIDATE_EVENT_LABELS.items()
    if label[0] == "loosen"
}


@dataclass
class Sample:
    trial: int
    frame_indices: tuple[int, ...]
    features: np.ndarray
    delta_q: float
    stable: float
    effort_load_mean: float | None
    source: str


def trial_path(trial: int) -> Path:
    if trial in BOUNDARY_TRIALS:
        number = BOUNDARY_TRIALS[trial]
        return EVIDENCE_ROOT / f"grip_boundary_trial{number:02d}_20260831"
    if trial in HEAD_ONLY_TRIALS:
        number = HEAD_ONLY_TRIALS[trial]
        return EVIDENCE_ROOT / f"grip_candidate_head_only_trial{number:02d}_20260831"
    date = "20260827" if trial <= 20 else "20260828"
    return EVIDENCE_ROOT / f"grip_intervention_trial{trial}_{date}"


def load_rows(trial: int) -> list[dict]:
    with (trial_path(trial) / "control.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def telemetry_rows(rows: list[dict]) -> list[dict]:
    result = []
    for frame_index, row in enumerate(rows):
        telemetry = row.get("gripper_telemetry")
        if telemetry is None:
            continue
        result.append(
            {
                "frame_index": frame_index,
                "time": float(row["elapsed_s"]),
                "features": np.asarray(
                    [
                        row["predicted_action"]["gripper"],
                        row["bus_target"]["gripper"],
                        row["readback_state"]["gripper"],
                        telemetry["present_current"],
                        telemetry["present_load"],
                        telemetry["position_lag"],
                    ],
                    dtype=np.float32,
                ),
                "command": float(row["bus_target"]["gripper"]),
                "actual": float(row["readback_state"]["gripper"]),
            }
        )
    return result


def evenly_spaced(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return [indices[index] for index in np.unique(positions)]


def load_mean_between(telemetry: list[dict], start_s: float, end_s: float) -> float | None:
    values = [
        abs(float(item["features"][4]))
        for item in telemetry
        if start_s <= item["time"] <= end_s
    ]
    return float(np.mean(values)) if values else None


def numeric_sample(
    trial: int,
    telemetry: list[dict],
    index: int,
    *,
    delta_q: float,
    stable: float,
    effort_load_mean: float | None,
    source: str,
) -> Sample | None:
    if index < HISTORY_STEPS - 1:
        return None
    history = telemetry[index - HISTORY_STEPS + 1 : index + 1]
    frame_indices = tuple(item["frame_index"] for item in history)
    if frame_indices[0] < VISUAL_GAP_FRAMES:
        return None
    return Sample(
        trial=trial,
        frame_indices=frame_indices,
        features=np.stack([item["features"] for item in history]),
        delta_q=delta_q,
        stable=stable,
        effort_load_mean=effort_load_mean,
        source=source,
    )


def build_numeric_samples(
    hold_labels: dict[int, tuple[float, str]] = HOLD_STABILITY_LABELS,
    event_labels: dict[int, tuple[str, float]] = CANDIDATE_EVENT_LABELS,
) -> list[Sample]:
    trial_ids = sorted(set(hold_labels) | set(event_labels))
    samples: list[Sample] = []
    for trial in trial_ids:
        rows = load_rows(trial)
        telemetry = telemetry_rows(rows)
        times = [item["time"] for item in telemetry]
        direction_events = [
            (row, row["grip_intervention"]["direction"])
            for row in rows
            if (row.get("grip_intervention") or {}).get("label_valid")
            and row["grip_intervention"]["direction"] in {"tighten", "loosen"}
        ]
        direction_events.extend(
            (row, "loosen")
            for row in rows
            if (row.get("grip_candidate_trial") or {}).get("control", {}).get("action")
            == "loosen"
        )
        first_active = next(
            (
                float(row["elapsed_s"])
                for row in rows
                if (row.get("grip_intervention") or {}).get("active")
            ),
            None,
        )
        last_direction = (
            max(float(row["elapsed_s"]) for row, _ in direction_events)
            if direction_events
            else None
        )
        closed = [
            index
            for index, item in enumerate(telemetry)
            if item["command"] < 32.0 and item["actual"] < 32.0
        ]

        if trial in hold_labels:
            target, phase = hold_labels[trial]
            candidates = closed
            if phase == "pre" and first_active is not None:
                candidates = [index for index in candidates if telemetry[index]["time"] < first_active]
            elif phase == "post" and last_direction is not None:
                candidates = [
                    index for index in candidates if telemetry[index]["time"] > last_direction + 0.5
                ]
            candidates = candidates[len(candidates) // 3 :]
            for index in evenly_spaced(candidates, MAX_PHASE_SAMPLES):
                sample = numeric_sample(
                    trial,
                    telemetry,
                    index,
                    delta_q=0.0,
                    stable=target,
                    effort_load_mean=load_mean_between(
                        telemetry,
                        telemetry[index]["time"],
                        telemetry[index]["time"] + EFFORT_HORIZON_S,
                    ),
                    source=f"reviewed_hold_{phase}",
                )
                if sample is not None:
                    samples.append(sample)

        if trial in event_labels:
            direction, target = event_labels[trial]
            delta_q = -0.2 if direction == "tighten" else 0.2
            matching = [
                row for row, event_direction in direction_events if event_direction == direction
            ]
            event_indices = []
            for row in matching:
                # The event row already contains the adjusted command. Score from the
                # preceding telemetry context so delta_q is not leaked into q_cmd.
                index = bisect_left(times, float(row["elapsed_s"])) - 1
                if index >= HISTORY_STEPS - 1:
                    event_indices.append((index, float(row["elapsed_s"])))
            selected = evenly_spaced(list(range(len(event_indices))), MAX_EVENT_SAMPLES)
            for selected_index in selected:
                index, event_time = event_indices[selected_index]
                sample = numeric_sample(
                    trial,
                    telemetry,
                    index,
                    delta_q=delta_q,
                    stable=target,
                    effort_load_mean=load_mean_between(
                        telemetry,
                        event_time,
                        event_time + EFFORT_HORIZON_S,
                    ),
                    source=f"reviewed_{direction}_event",
                )
                if sample is not None:
                    samples.append(sample)
                    if trial in HEAD_ONLY_TRIALS:
                        samples.append(
                            numeric_sample(
                                trial,
                                telemetry,
                                index,
                                delta_q=0.0,
                                stable=1.0,
                                effort_load_mean=load_mean_between(
                                    telemetry,
                                    max(0.0, event_time - EFFORT_HORIZON_S),
                                    event_time,
                                ),
                                source="head_only_pre_loosen_stable_hold",
                            )
                        )
    samples.extend(build_boundary_samples())
    return samples


def build_boundary_samples() -> list[Sample]:
    samples = []
    for trial in BOUNDARY_TRIALS:
        path = trial_path(trial)
        outcome = json.loads((path / "outcome.json").read_text())
        rows = load_rows(trial)
        telemetry = telemetry_rows(rows)
        times = [item["time"] for item in telemetry]
        last_stable = float(outcome["last_stable_target_q"])
        first_unstable = float(outcome["first_unstable_target_q"])
        for row in rows:
            intervention = row.get("grip_intervention") or {}
            if not intervention.get("label_valid") or intervention.get("direction") != "loosen":
                continue
            target_q = float(row["bus_target"]["gripper"])
            index = bisect_left(times, float(row["elapsed_s"])) - 1
            if target_q <= last_stable + 0.01:
                stable = 1.0
                source = "boundary_safe_loosen"
            elif abs(target_q - first_unstable) <= 0.01:
                stable = 0.0
                source = "boundary_failure_loosen"
            else:
                continue
            sample = numeric_sample(
                trial,
                telemetry,
                index,
                delta_q=0.2,
                stable=stable,
                effort_load_mean=load_mean_between(
                    telemetry,
                    float(row["elapsed_s"]),
                    float(row["elapsed_s"]) + EFFORT_HORIZON_S,
                ),
                source=source,
            )
            if sample is not None:
                samples.append(sample)
                if source == "boundary_failure_loosen":
                    samples.append(
                        numeric_sample(
                            trial,
                            telemetry,
                            index,
                            delta_q=0.0,
                            stable=1.0,
                            effort_load_mean=load_mean_between(
                                telemetry,
                                float(row["elapsed_s"]) - EFFORT_HORIZON_S,
                                float(row["elapsed_s"]),
                            ),
                            source="boundary_last_stable_hold",
                        )
                    )
    return samples


def read_selected_rgb(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    for index in range(max(indices) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"video ended before frame {index}: {path}")
        if index in indices:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return frames


def add_visual_features(samples: list[Sample]) -> None:
    for trial in sorted({sample.trial for sample in samples}):
        trial_samples = [sample for sample in samples if sample.trial == trial]
        current_indices = {
            frame_index for sample in trial_samples for frame_index in sample.frame_indices
        }
        needed_indices = current_indices | {
            frame_index - VISUAL_GAP_FRAMES for frame_index in current_indices
        }
        path = trial_path(trial)
        front = read_selected_rgb(path / "front.avi", needed_indices)
        side = read_selected_rgb(path / "side.avi", needed_indices)
        visual_by_index = {
            index: grip_visual_features(
                previous_front_rgb=front[index - VISUAL_GAP_FRAMES],
                current_front_rgb=front[index],
                previous_side_rgb=side[index - VISUAL_GAP_FRAMES],
                current_side_rgb=side[index],
            )
            for index in current_indices
        }
        for sample in trial_samples:
            visual_history = np.stack(
                [visual_by_index[frame_index] for frame_index in sample.frame_indices]
            )
            sample.features = np.concatenate((sample.features, visual_history), axis=1)


def tensorize(samples: list[Sample], mean: torch.Tensor, std: torch.Tensor):
    features = torch.tensor(np.stack([sample.features for sample in samples]))
    features = (features - mean) / std
    delta_q = torch.tensor([sample.delta_q for sample in samples], dtype=torch.float32)
    targets = torch.tensor([sample.stable for sample in samples], dtype=torch.float32)
    effort = torch.tensor(
        [
            float("nan") if sample.effort_load_mean is None else sample.effort_load_mean
            for sample in samples
        ],
        dtype=torch.float32,
    )
    return features, delta_q, targets, effort


def metrics(head, features, delta_q, targets) -> dict:
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(features, delta_q))
    prevalence = float(targets.mean())
    return {
        "samples": int(targets.numel()),
        "stable": int(targets.sum()),
        "accuracy": float(((probabilities >= 0.5) == targets.bool()).float().mean()),
        "brier": float((probabilities - targets).square().mean()),
        "constant_prevalence_brier": float((targets - prevalence).square().mean()),
    }


def effort_metrics(
    head,
    features,
    delta_q,
    targets,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict:
    valid = torch.isfinite(targets)
    with torch.inference_mode():
        predictions = head(features[valid], delta_q[valid]) * target_std + target_mean
    selected = targets[valid]
    return {
        "samples": int(valid.sum()),
        "mae": float((predictions - selected).abs().mean()),
        "constant_train_mean_mae": float((selected - target_mean).abs().mean()),
    }


def boundary_pair_indices(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    hold_indices = []
    loosen_indices = []
    for index, sample in enumerate(samples):
        if sample.source != "boundary_failure_loosen":
            continue
        hold_index = next(
            other_index
            for other_index, other in enumerate(samples)
            if other.trial == sample.trial
            and other.frame_indices == sample.frame_indices
            and other.source == "boundary_last_stable_hold"
        )
        hold_indices.append(hold_index)
        loosen_indices.append(index)
    return torch.tensor(hold_indices), torch.tensor(loosen_indices)


def support_counts(samples: list[Sample]) -> dict[str, dict[str, int]]:
    result = {}
    for delta in GRIP_CANDIDATE_DELTAS:
        selected = [sample for sample in samples if sample.delta_q == delta]
        result[f"{delta:+.1f}"] = {
            "unstable": sum(sample.stable == 0.0 for sample in selected),
            "stable": sum(sample.stable == 1.0 for sample in selected),
        }
    return result


def heldout_trial_scores(
    head,
    features,
    samples: list[Sample],
    candidate_deltas: tuple[float, ...] = GRIP_CANDIDATE_DELTAS,
) -> dict[str, dict]:
    result = {}
    candidates = torch.tensor(candidate_deltas, dtype=torch.float32)
    with torch.inference_mode():
        for trial in sorted({sample.trial for sample in samples}):
            indices = [index for index, sample in enumerate(samples) if sample.trial == trial]
            trial_features = features[indices]
            scores = []
            for delta in candidates:
                repeated_delta = delta.repeat(len(indices))
                scores.append(torch.sigmoid(head(trial_features, repeated_delta)).mean().item())
            observed = torch.sigmoid(
                head(
                    trial_features,
                    torch.tensor([samples[index].delta_q for index in indices]),
                )
            )
            result[str(trial)] = {
                "samples": len(indices),
                "observed_delta_q": sorted({samples[index].delta_q for index in indices}),
                "observed_targets": sorted({samples[index].stable for index in indices}),
                "observed_probability_mean": float(observed.mean()),
                "candidate_probability_means": {
                    f"{delta:+.1f}": score
                    for delta, score in zip(candidate_deltas, scores, strict=True)
                },
            }
    return result


def heldout_boundary_pair_score(
    stability_head,
    effort_head,
    features,
    samples: list[Sample],
    *,
    effort_mean: torch.Tensor,
    effort_std: torch.Tensor,
    minimum_probability: float = 0.65,
    minimum_load_for_loosen: float = MIN_LOAD_FOR_LOOSEN,
) -> dict:
    index = next(
        index
        for index, sample in enumerate(samples)
        if sample.trial == 1004 and sample.source == "boundary_failure_loosen"
    )
    context = features[index : index + 1]
    with torch.inference_mode():
        hold = float(torch.sigmoid(stability_head(context, torch.tensor([0.0]))))
        loosen = float(torch.sigmoid(stability_head(context, torch.tensor([0.2]))))
        hold_effort = float(
            effort_head(context, torch.tensor([0.0])) * effort_std + effort_mean
        )
        loosen_effort = float(
            effort_head(context, torch.tensor([0.2])) * effort_std + effort_mean
        )
    probabilities = {0.0: hold, 0.2: loosen}
    efforts = {0.0: hold_effort, 0.2: loosen_effort}
    present_load = abs(float(samples[index].features[-1, 4]))
    eligible = [
        delta
        for delta in (0.0, 0.2)
        if probabilities[delta] >= minimum_probability
        and (delta == 0.0 or present_load > minimum_load_for_loosen)
    ]
    selected = min(eligible, key=efforts.get) if eligible else 0.0
    return {
        "hold_stability_probability": hold,
        "loosen_stability_probability": loosen,
        "hold_predicted_load_mean": hold_effort,
        "loosen_predicted_load_mean": loosen_effort,
        "present_load": present_load,
        "load_gate": minimum_load_for_loosen,
        "ranking_pass": loosen < hold,
        "selected_delta_q": selected,
        "pass": selected == 0.0,
    }


def heldout_loosen_selection_score(
    stability_head,
    effort_head,
    features,
    samples: list[Sample],
    *,
    effort_mean: torch.Tensor,
    effort_std: torch.Tensor,
    minimum_probability: float = 0.65,
    minimum_load_for_loosen: float = MIN_LOAD_FOR_LOOSEN,
) -> dict:
    selections = []
    for index, sample in enumerate(samples):
        if sample.trial != 45 or sample.source != "reviewed_loosen_event":
            continue
        context = features[index : index + 1]
        probabilities = {}
        efforts = {}
        with torch.inference_mode():
            for delta in (0.0, 0.2):
                candidate = torch.tensor([delta])
                probabilities[delta] = float(
                    torch.sigmoid(stability_head(context, candidate))
                )
                efforts[delta] = float(
                    effort_head(context, candidate) * effort_std + effort_mean
                )
        eligible = [
            delta
            for delta in (0.0, 0.2)
            if probabilities[delta] >= minimum_probability
            and (
                delta == 0.0
                or abs(float(sample.features[-1, 4])) > minimum_load_for_loosen
            )
        ]
        selected = min(eligible, key=efforts.get) if eligible else 0.0
        selections.append(selected)
    return {
        "trial": 45,
        "events": len(selections),
        "selected_loosen": sum(delta == 0.2 for delta in selections),
        "selected_hold": sum(delta == 0.0 for delta in selections),
        "pass": bool(selections) and all(delta == 0.2 for delta in selections),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--post-lift-only", action="store_true")
    parser.add_argument("--no-load-gate", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    hold_labels = (
        POST_LIFT_HOLD_STABILITY_LABELS if args.post_lift_only else HOLD_STABILITY_LABELS
    )
    event_labels = (
        POST_LIFT_CANDIDATE_EVENT_LABELS
        if args.post_lift_only
        else CANDIDATE_EVENT_LABELS
    )
    samples = build_numeric_samples(hold_labels, event_labels)
    add_visual_features(samples)
    train_samples = [sample for sample in samples if sample.trial not in HELDOUT_TRIALS]
    heldout_samples = [sample for sample in samples if sample.trial in HELDOUT_TRIALS]
    train_raw = torch.tensor(np.stack([sample.features for sample in train_samples]))
    feature_mean = train_raw.flatten(0, 1).mean(dim=0)
    feature_std = train_raw.flatten(0, 1).std(dim=0).clamp_min(1e-6)
    train_x, train_delta, train_targets, train_effort = tensorize(
        train_samples, feature_mean, feature_std
    )
    heldout_x, heldout_delta, heldout_targets, heldout_effort = tensorize(
        heldout_samples, feature_mean, feature_std
    )
    train_effort_valid = torch.isfinite(train_effort)
    effort_mean = train_effort[train_effort_valid].mean()
    effort_std = train_effort[train_effort_valid].std().clamp_min(1e-6)

    delta_indices = torch.tensor(
        [int(round(float(delta) / 0.2)) + 1 for delta in train_delta]
    )
    delta_counts = torch.bincount(delta_indices, minlength=len(GRIP_CANDIDATE_DELTAS))
    expected_delta_indices = {1, 2} if args.post_lift_only else {0, 1, 2}
    if set(torch.nonzero(delta_counts, as_tuple=False).flatten().tolist()) != expected_delta_indices:
        raise ValueError(f"unexpected candidate actions in training: {delta_counts.tolist()}")
    target_indices = train_targets.long()
    cell_counts = torch.zeros((len(GRIP_CANDIDATE_DELTAS), 2), dtype=torch.int64)
    for delta_index, target_index in zip(delta_indices, target_indices, strict=True):
        cell_counts[delta_index, target_index] += 1
    if any(torch.any(cell_counts[index] == 0) for index in expected_delta_indices):
        raise ValueError(f"missing candidate outcome in training: {cell_counts.tolist()}")
    cell_weights = torch.ones_like(cell_counts, dtype=torch.float32)
    active_counts = cell_counts[sorted(expected_delta_indices)].float()
    cell_weights[sorted(expected_delta_indices)] = (
        active_counts.sum() / (active_counts.numel() * active_counts)
    ).sqrt()
    stability_head = GripCandidateHead(history_steps=HISTORY_STEPS, hidden_dim=HIDDEN_DIM)
    effort_head = GripCandidateHead(history_steps=HISTORY_STEPS, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.AdamW(
        [*stability_head.parameters(), *effort_head.parameters()],
        lr=1e-3,
        weight_decay=1e-4,
    )
    pair_hold_indices, pair_loosen_indices = boundary_pair_indices(train_samples)

    for step in range(1, args.steps + 1):
        logits = stability_head(train_x, train_delta)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            train_targets,
            reduction="none",
        )
        stability_loss = (losses * cell_weights[delta_indices, target_indices]).mean()
        effort_predictions = effort_head(
            train_x[train_effort_valid], train_delta[train_effort_valid]
        )
        effort_targets = (
            train_effort[train_effort_valid] - effort_mean
        ) / effort_std
        effort_loss = F.smooth_l1_loss(effort_predictions, effort_targets)
        hold_logits = stability_head(
            train_x[pair_hold_indices], train_delta[pair_hold_indices]
        )
        loosen_logits = stability_head(
            train_x[pair_loosen_indices], train_delta[pair_loosen_indices]
        )
        pair_loss = F.relu(BOUNDARY_PAIR_MARGIN - (hold_logits - loosen_logits)).mean()
        loss = stability_loss + effort_loss + pair_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(
                f"step={step} loss={loss.item():.4f} "
                f"stability={stability_loss.item():.4f} "
                f"effort={effort_loss.item():.4f} pair={pair_loss.item():.4f}"
            )

    train_metrics = metrics(stability_head, train_x, train_delta, train_targets)
    heldout_metrics = metrics(stability_head, heldout_x, heldout_delta, heldout_targets)
    train_effort_metrics = effort_metrics(
        effort_head,
        train_x,
        train_delta,
        train_effort,
        target_mean=effort_mean,
        target_std=effort_std,
    )
    heldout_effort_metrics = effort_metrics(
        effort_head,
        heldout_x,
        heldout_delta,
        heldout_effort,
        target_mean=effort_mean,
        target_std=effort_std,
    )
    train_support = support_counts(train_samples)
    supported_deltas = [
        delta
        for delta in GRIP_CANDIDATE_DELTAS
        if all(train_support[f"{delta:+.1f}"][label] > 0 for label in ("unstable", "stable"))
    ]
    active_candidate_deltas = (0.0, 0.2) if args.post_lift_only else GRIP_CANDIDATE_DELTAS
    trial_scores = heldout_trial_scores(
        stability_head, heldout_x, heldout_samples, active_candidate_deltas
    )
    minimum_load_for_loosen = -1.0 if args.no_load_gate else MIN_LOAD_FOR_LOOSEN
    boundary_pair_score = heldout_boundary_pair_score(
        stability_head,
        effort_head,
        heldout_x,
        heldout_samples,
        effort_mean=effort_mean,
        effort_std=effort_std,
        minimum_load_for_loosen=minimum_load_for_loosen,
    )
    loosen_selection_score = heldout_loosen_selection_score(
        stability_head,
        effort_head,
        heldout_x,
        heldout_samples,
        effort_mean=effort_mean,
        effort_std=effort_std,
        minimum_load_for_loosen=minimum_load_for_loosen,
    )
    offline_gate_pass = bool(
        boundary_pair_score["pass"]
        and loosen_selection_score["pass"]
        and heldout_metrics["brier"] < heldout_metrics["constant_prevalence_brier"]
    )
    print(f"delta_counts={delta_counts.tolist()}")
    print(f"candidate_outcome_counts={cell_counts.tolist()}")
    print(f"train_support={train_support}")
    print(f"auto_supported_deltas={supported_deltas}")
    print(f"train_metrics={train_metrics}")
    print(f"heldout_metrics={heldout_metrics}")
    print(f"train_effort_metrics={train_effort_metrics}")
    print(f"heldout_effort_metrics={heldout_effort_metrics}")
    print(f"heldout_trial_scores={json.dumps(trial_scores, sort_keys=True)}")
    print(f"heldout_boundary_pair_score={boundary_pair_score}")
    print(f"heldout_loosen_selection_score={loosen_selection_score}")
    print(f"offline_gate_pass={offline_gate_pass}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "action_conditioned_grip_stability_effort_v1",
            "training_view": (
                "post_lift_hold_loosen" if args.post_lift_only else "three_action_all_reviewed"
            ),
            "history_steps": HISTORY_STEPS,
            "hidden_dim": HIDDEN_DIM,
            "feature_names": GRIP_CANDIDATE_FEATURES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "visual_gap_frames": VISUAL_GAP_FRAMES,
            "candidate_deltas": active_candidate_deltas,
            "stability_model_state_dict": stability_head.state_dict(),
            "effort_model_state_dict": effort_head.state_dict(),
            "effort_target": "present_load_abs_mean_next_0p8s",
            "effort_mean": effort_mean,
            "effort_std": effort_std,
            "boundary_pair_margin": BOUNDARY_PAIR_MARGIN,
            "train_trials": sorted({sample.trial for sample in train_samples}),
            "heldout_trials": sorted({sample.trial for sample in heldout_samples}),
            "train_support": train_support,
            "candidate_outcome_counts": cell_counts,
            "auto_supported_deltas": supported_deltas,
            "selection_policy": {
                "order": "stability_constraint_then_minimum_predicted_load",
                "uncertain_action": "hold",
                "minimum_probability": 0.65,
                "minimum_present_load_for_loosen": minimum_load_for_loosen,
                "load_gate_enabled": not args.no_load_gate,
                "load_gate_provenance": (
                    "disabled"
                    if args.no_load_gate
                    else "selected_from_historical_boundary_trials_1_3_4"
                ),
                "loosen_requires_stable_lift_seen": True,
                "tighten": (
                    "deterministic_rollback_only"
                    if args.post_lift_only
                    else "learned_candidate"
                ),
            },
            "deployment_allowed": False,
            "train_metrics": train_metrics,
            "heldout_metrics": heldout_metrics,
            "train_effort_metrics": train_effort_metrics,
            "heldout_effort_metrics": heldout_effort_metrics,
            "heldout_trial_scores": trial_scores,
            "heldout_boundary_pair_score": boundary_pair_score,
            "heldout_loosen_selection_score": loosen_selection_score,
            "offline_gate_pass": offline_gate_pass,
            "steps": args.steps,
            "seed": 42,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
