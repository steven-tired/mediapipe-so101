#!/usr/bin/env python
"""Phase C gripper-event diagnostics shared by ACT, SmolVLA, and Diffusion."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import numpy as np
from pathlib import Path
import torch


GRIPPER_INDEX = 5
GRIPPER_THRESHOLD = 90.0
PRE_LIFT_GRIPPER_ACTION = 32.5


def parse_named_assignment(text: str) -> tuple[str, str]:
    """Parse NAME=VALUE while leaving every character after the first '=' intact."""
    name, separator, value = text.partition("=")
    if not separator or not name or not value:
        raise ValueError(f"expected NAME=VALUE, got {text!r}")
    return name, value


def summarize_event_counts(counts: dict[str, int]) -> dict[str, float | None]:
    """Convert event hit/count pairs to rates without hiding absent event classes."""
    def rate(numerator: str, denominator: str) -> float | None:
        count = counts[denominator]
        return None if count == 0 else counts[numerator] / count

    return {
        "close_direction_recall": rate("close_direction_hit", "close_onset"),
        "close_crossing_recall": rate("close_crossing_hit", "close_crossing_target"),
        "open_false_close_rate": rate("open_false_close", "open_phase"),
        "release_direction_recall": rate("release_direction_hit", "release_onset"),
        "release_crossing_recall": rate("release_crossing_hit", "release_crossing_target"),
    }


def current_target_index(n_obs_steps: int) -> int:
    """Return the action index aligned with the newest observation."""
    if n_obs_steps < 1:
        raise ValueError("n_obs_steps must be positive")
    return n_obs_steps - 1


def translate_images(images: torch.Tensor, *, dx: int, dy: int) -> torch.Tensor:
    """Translate image tensors with zero fill and no wraparound."""
    height, width = images.shape[-2:]
    shifted = torch.zeros_like(images)

    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)
    target_x_start = max(0, dx)
    target_x_end = min(width, width + dx)
    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)
    target_y_start = max(0, dy)
    target_y_end = min(height, height + dy)

    if source_x_start < source_x_end and source_y_start < source_y_end:
        shifted[..., target_y_start:target_y_end, target_x_start:target_x_end] = images[
            ..., source_y_start:source_y_end, source_x_start:source_x_end
        ]
    return shifted


def set_current_gripper(state: torch.Tensor, value: float) -> torch.Tensor:
    """Replace only the gripper coordinate aligned with the newest observation."""
    changed = state.clone()
    if changed.ndim == 3:
        changed[:, -1, GRIPPER_INDEX] = value
    else:
        changed[:, GRIPPER_INDEX] = value
    return changed


def predict_physical_chunk(
    policy,
    processed_batch: dict[str, object],
    postprocessor,
    *,
    seed: int,
) -> torch.Tensor:
    """Predict the physical-unit chunk visible to the runtime execution queue."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in processed_batch.items()
        if key != "action"
    }
    with torch.inference_mode():
        if policy.config.type in {"act", "smolvla"}:
            return postprocessor(policy.predict_action_chunk(batch))
        if policy.config.type == "diffusion":
            if batch["observation.state"].ndim == 3:
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in policy.config.image_features], dim=-4
                )
                raw_chunk = policy.diffusion.generate_actions(batch)
                start = policy.config.n_obs_steps - 1
                end = start + policy.config.n_action_steps
                return postprocessor(raw_chunk[:, start:end])
            policy.reset()
            raw_actions = [policy.select_action(batch) for _ in range(policy.config.n_action_steps)]
            return postprocessor(torch.stack(raw_actions, dim=1))
    raise ValueError(f"unsupported policy type: {policy.config.type}")


def earliest_threshold_offset(gripper_chunks: np.ndarray, *, threshold: float) -> np.ndarray:
    """Return the first below-threshold offset per chunk, or -1 when absent."""
    gripper_chunks = np.asarray(gripper_chunks)
    below = gripper_chunks < threshold
    first = below.argmax(axis=1)
    return np.where(below.any(axis=1), first, -1)


def find_pre_lift_anchor_rows(
    episodes: np.ndarray,
    frames: np.ndarray,
    actions: np.ndarray,
    *,
    close_event_rows: np.ndarray,
    tight_action_threshold: float = PRE_LIFT_GRIPPER_ACTION,
) -> np.ndarray:
    """Find the first tight-grip command after each initial close event.

    The formal dataset has no end-effector height or lift annotation. This
    action-defined anchor is therefore a pre-lift proxy, not a lift detector.
    """
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    actions = np.asarray(actions)
    anchors: list[int] = []
    for event_row in np.asarray(close_event_rows, dtype=np.int64):
        episode = episodes[event_row]
        candidates = np.flatnonzero(
            (episodes == episode)
            & (frames >= frames[event_row])
            & (actions[:, GRIPPER_INDEX] <= tight_action_threshold)
        )
        if len(candidates):
            anchors.append(int(candidates[np.argmin(frames[candidates])]))
    return np.asarray(anchors, dtype=np.int64)


def derive_event_labels(
    episodes: np.ndarray,
    frames: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive close-event, readiness, pure-open, and pre-lift proxy labels."""
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    states = np.asarray(states)
    actions = np.asarray(actions)
    ready = np.zeros(len(episodes), dtype=bool)
    close_event_rows: list[int] = []
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(frames[rows])]
        grip = actions[rows, GRIPPER_INDEX]
        local_events = np.flatnonzero(
            (grip[:-1] >= 99.0)
            & (grip[1:] >= GRIPPER_THRESHOLD)
            & (grip[1:] < 99.0)
        ) + 1
        for local_event in local_events:
            event_row = int(rows[local_event])
            close_event_rows.append(event_row)
            start = max(0, int(local_event) - 2)
            stop = min(len(rows), int(local_event) + 3)
            ready[rows[start:stop]] = True
    close_events = np.asarray(close_event_rows, dtype=np.int64)
    pure_open = (
        (states[:, GRIPPER_INDEX] >= GRIPPER_THRESHOLD)
        & (actions[:, GRIPPER_INDEX] >= GRIPPER_THRESHOLD)
        & ~ready
    )
    return {
        "close_event_rows": close_events,
        "ready_positive": ready,
        "pure_open_negative": pure_open,
        "pre_lift_anchor_rows": find_pre_lift_anchor_rows(
            episodes,
            frames,
            actions,
            close_event_rows=close_events,
        ),
    }


def make_episode_audit_rows(
    episodes: np.ndarray,
    frames: np.ndarray,
    states: np.ndarray,
    target_actions: np.ndarray,
    predicted_actions: np.ndarray,
    action_is_pad: np.ndarray,
    *,
    target_index: int,
    executed_steps: int,
    labels: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Summarize body, event timing, false close, release, and pre-lift per episode."""
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    states = np.asarray(states)
    target_actions = np.asarray(target_actions)
    predicted_actions = np.asarray(predicted_actions)
    action_is_pad = np.asarray(action_is_pad, dtype=bool)
    steps = min(executed_steps, predicted_actions.shape[1])
    crossing = (predicted_actions[:, :steps, GRIPPER_INDEX] < GRIPPER_THRESHOLD).any(axis=1)
    first_body_mae = np.abs(
        predicted_actions[:, 0, :GRIPPER_INDEX]
        - target_actions[:, target_index, :GRIPPER_INDEX]
    ).mean(axis=1)
    hold_body_mae = np.abs(
        states[:, :GRIPPER_INDEX]
        - target_actions[:, target_index, :GRIPPER_INDEX]
    ).mean(axis=1)
    first_gripper_mae = np.abs(
        predicted_actions[:, 0, GRIPPER_INDEX]
        - target_actions[:, target_index, GRIPPER_INDEX]
    )
    target_grip = target_actions[:, target_index, GRIPPER_INDEX]
    release_target = (
        (states[:, GRIPPER_INDEX] < GRIPPER_THRESHOLD)
        & (target_grip >= states[:, GRIPPER_INDEX] + 4.0)
        & (target_grip >= GRIPPER_THRESHOLD)
    )
    release_hit = release_target & (
        predicted_actions[:, :steps, GRIPPER_INDEX] >= GRIPPER_THRESHOLD
    ).any(axis=1)

    rows: list[dict[str, object]] = []
    for episode in np.unique(episodes):
        episode_rows = np.flatnonzero(episodes == episode)
        episode_rows = episode_rows[np.argsort(frames[episode_rows])]
        event_candidates = labels["close_event_rows"][
            episodes[labels["close_event_rows"]] == episode
        ]
        event_row = int(event_candidates[0]) if len(event_candidates) else None
        crossing_rows = episode_rows[crossing[episode_rows]]
        first_crossing_row = int(crossing_rows[0]) if len(crossing_rows) else None
        first_offset = (
            int(frames[first_crossing_row] - frames[event_row])
            if event_row is not None and first_crossing_row is not None
            else None
        )
        if first_offset is None:
            timing = "missing"
        elif first_offset < -2:
            timing = "early"
        elif first_offset > 2:
            timing = "late"
        else:
            timing = "on_time"
        event_window_hit = bool(
            event_row is not None
            and crossing[
                episode_rows[
                    (frames[episode_rows] >= frames[event_row] - 2)
                    & (frames[episode_rows] <= frames[event_row] + 2)
                ]
            ].any()
        )
        false_rows = episode_rows[
            crossing[episode_rows] & labels["pure_open_negative"][episode_rows]
        ]
        sustained_pairs = int(np.sum(np.diff(frames[false_rows]) == 1))
        anchor_candidates = labels["pre_lift_anchor_rows"][
            episodes[labels["pre_lift_anchor_rows"]] == episode
        ]
        anchor_row = int(anchor_candidates[0]) if len(anchor_candidates) else None
        rows.append(
            {
                "episode": int(episode),
                "samples": int(len(episode_rows)),
                "close_event_frame": int(frames[event_row]) if event_row is not None else None,
                "event_window_hit": event_window_hit,
                "first_crossing_frame": (
                    int(frames[first_crossing_row]) if first_crossing_row is not None else None
                ),
                "first_crossing_offset": first_offset,
                "first_crossing_timing": timing,
                "pure_open_negative_frames": int(
                    labels["pure_open_negative"][episode_rows].sum()
                ),
                "pure_false_close_frames": int(len(false_rows)),
                "sustained_false_close_pairs": sustained_pairs,
                "body_mae_h1": float(first_body_mae[episode_rows].mean()),
                "body_p95_h1": float(np.quantile(first_body_mae[episode_rows], 0.95)),
                "body_hold_baseline_mae_h1": float(hold_body_mae[episode_rows].mean()),
                "gripper_mae_h1": float(first_gripper_mae[episode_rows].mean()),
                "release_crossing_target": int(release_target[episode_rows].sum()),
                "release_crossing_hit": int(release_hit[episode_rows].sum()),
                "pre_lift_anchor_frame": (
                    int(frames[anchor_row]) if anchor_row is not None else None
                ),
                "pre_lift_body_mae_h1": (
                    float(first_body_mae[anchor_row]) if anchor_row is not None else None
                ),
                "pre_lift_gripper_mae_h1": (
                    float(first_gripper_mae[anchor_row]) if anchor_row is not None else None
                ),
                "pre_lift_gripper_closed": (
                    bool(predicted_actions[anchor_row, 0, GRIPPER_INDEX] < GRIPPER_THRESHOLD)
                    if anchor_row is not None
                    else None
                ),
            }
        )
    return rows


def make_temporal_consistency_rows(
    predicted_actions: np.ndarray,
    action_is_pad: np.ndarray,
    *,
    episodes: np.ndarray,
    frames: np.ndarray,
    query_stride: int,
    gripper_threshold: float = GRIPPER_THRESHOLD,
) -> list[dict[str, object]]:
    """Compare contiguous chunks where they predict the same future times.

    This is a deterministic ACT analogue of Sentinel/STAC overlap comparison;
    it is not MMD because ACT inference uses one deterministic chunk here.
    """
    if query_stride < 1:
        raise ValueError("query_stride must be positive")
    predicted_actions = np.asarray(predicted_actions)
    action_is_pad = np.asarray(action_is_pad, dtype=bool)
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    rows: list[dict[str, object]] = []

    lookup = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episodes, frames, strict=True))
    }
    for first_index, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        second_index = lookup.get((int(episode), int(frame) + query_stride))
        if second_index is None or query_stride >= predicted_actions.shape[1]:
            continue
        first = predicted_actions[first_index, query_stride:]
        second = predicted_actions[second_index, :-query_stride]
        valid = ~(
            action_is_pad[first_index, query_stride:]
            | action_is_pad[second_index, :-query_stride]
        )
        if not valid.any():
            continue
        first = first[valid]
        second = second[valid]
        first_grip = first[:, GRIPPER_INDEX]
        second_grip = second[:, GRIPPER_INDEX]
        first_offset = earliest_threshold_offset(
            predicted_actions[first_index : first_index + 1, :, GRIPPER_INDEX],
            threshold=gripper_threshold,
        )[0]
        second_offset = earliest_threshold_offset(
            predicted_actions[second_index : second_index + 1, :, GRIPPER_INDEX],
            threshold=gripper_threshold,
        )[0]
        both_have_close = first_offset >= 0 and second_offset >= 0
        rows.append(
            {
                "episode": int(episode),
                "from_frame": int(frame),
                "to_frame": int(frame) + query_stride,
                "query_stride": query_stride,
                "overlap_steps": int(valid.sum()),
                "body_overlap_mae_deg": float(
                    np.abs(first[:, :GRIPPER_INDEX] - second[:, :GRIPPER_INDEX]).mean()
                ),
                "gripper_overlap_mae_deg": float(np.abs(first_grip - second_grip).mean()),
                "gripper_threshold_disagreement_rate": float(
                    ((first_grip < gripper_threshold) != (second_grip < gripper_threshold)).mean()
                ),
                "close_available_disagreement": bool((first_offset >= 0) != (second_offset >= 0)),
                "absolute_close_frame_shift": (
                    int(abs((int(frame) + int(first_offset)) - (
                        int(frame) + query_stride + int(second_offset)
                    )))
                    if both_have_close
                    else None
                ),
            }
        )
    return rows


def summarize_temporal_consistency_rows(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize deterministic overlap distances and Sentinel-style cumulative signal."""
    if not rows:
        return {
            "transitions": 0,
            "body_overlap_mae_deg": None,
            "body_overlap_p95_deg": None,
            "gripper_overlap_mae_deg": None,
            "gripper_overlap_p95_deg": None,
            "threshold_disagreement_rate": None,
            "close_available_disagreement_count": 0,
            "mean_absolute_close_frame_shift": None,
            "episode_cumulative_body_mae": {},
        }
    body = np.asarray([row["body_overlap_mae_deg"] for row in rows], dtype=np.float64)
    gripper = np.asarray(
        [row["gripper_overlap_mae_deg"] for row in rows], dtype=np.float64
    )
    threshold = np.asarray(
        [row["gripper_threshold_disagreement_rate"] for row in rows], dtype=np.float64
    )
    close_shifts = [
        float(row["absolute_close_frame_shift"])
        for row in rows
        if row["absolute_close_frame_shift"] is not None
    ]
    cumulative = {
        str(int(episode)): float(
            sum(
                float(row["body_overlap_mae_deg"])
                for row in rows
                if int(row["episode"]) == int(episode)
            )
        )
        for episode in sorted({int(row["episode"]) for row in rows})
    }
    return {
        "transitions": len(rows),
        "body_overlap_mae_deg": float(body.mean()),
        "body_overlap_p95_deg": float(np.quantile(body, 0.95)),
        "gripper_overlap_mae_deg": float(gripper.mean()),
        "gripper_overlap_p95_deg": float(np.quantile(gripper, 0.95)),
        "threshold_disagreement_rate": float(threshold.mean()),
        "close_available_disagreement_count": int(
            sum(bool(row["close_available_disagreement"]) for row in rows)
        ),
        "mean_absolute_close_frame_shift": (
            float(np.mean(close_shifts)) if close_shifts else None
        ),
        "episode_cumulative_body_mae": cumulative,
    }


def make_screening_row(
    summary: dict[str, object],
    episode_rows: list[dict[str, object]],
    *,
    hold_body_mae: float,
    hold_body_p95: float,
    temporal_summary: dict[str, object],
) -> dict[str, object]:
    """Apply the documented, non-weighted checkpoint screening gates."""
    on_time = sum(row["first_crossing_timing"] == "on_time" for row in episode_rows)
    event_hits = sum(bool(row["event_window_hit"]) for row in episode_rows)
    false_close = sum(int(row["pure_false_close_frames"]) for row in episode_rows)
    sustained_false = sum(
        int(row["sustained_false_close_pairs"]) for row in episode_rows
    )
    release_target = sum(int(row["release_crossing_target"]) for row in episode_rows)
    release_hit = sum(int(row["release_crossing_hit"]) for row in episode_rows)
    pre_lift_values = [
        row["pre_lift_gripper_closed"]
        for row in episode_rows
        if row["pre_lift_gripper_closed"] is not None
    ]
    body = summary["body_prefix"]
    body_mean = float(body["body_mae_executed_prefix"])
    body_p95 = float(body["body_p95_executed_prefix"])
    timing_gate = on_time >= 5 and event_hits >= 5
    pure_false_gate = false_close == 0
    release_gate = release_target >= 12 and release_hit >= 11
    pre_lift_gate = len(pre_lift_values) >= 6 and all(pre_lift_values)
    body_gate = body_mean < hold_body_mae and body_p95 < hold_body_p95
    gates = (timing_gate, pure_false_gate, release_gate, pre_lift_gate, body_gate)
    return {
        "policy": summary["policy"],
        "checkpoint": summary["checkpoint"],
        "on_time_episodes": on_time,
        "event_window_hit_episodes": event_hits,
        "pure_false_close_frames": false_close,
        "sustained_false_close_pairs": sustained_false,
        "release_crossing_hit": release_hit,
        "release_crossing_target": release_target,
        "pre_lift_closed_episodes": int(sum(bool(value) for value in pre_lift_values)),
        "pre_lift_anchor_episodes": len(pre_lift_values),
        "body_mae_h1": body_mean,
        "body_p95_h1": body_p95,
        "body_hold_baseline_mae_h1": hold_body_mae,
        "body_hold_baseline_p95_h1": hold_body_p95,
        "temporal_body_overlap_p95_deg": temporal_summary["body_overlap_p95_deg"],
        "timing_gate": timing_gate,
        "pure_false_close_gate": pure_false_gate,
        "release_gate": release_gate,
        "pre_lift_gate": pre_lift_gate,
        "body_beats_hold_gate": body_gate,
        "gates_passed": int(sum(gates)),
        "joint_gate": bool(
            all(gates)
        ),
    }


def nearest_event_neighbors(
    episodes: np.ndarray,
    frames: np.ndarray,
    states: np.ndarray,
    front_features: np.ndarray,
    side_features: np.ndarray,
    *,
    event_rows: np.ndarray,
    heldout_episodes: set[int],
) -> list[dict[str, object]]:
    """Find separate visual and body-state train neighbors for held-out events."""
    episodes = np.asarray(episodes)
    frames = np.asarray(frames)
    states = np.asarray(states)
    event_rows = np.asarray(event_rows, dtype=np.int64)
    heldout_mask = np.isin(episodes[event_rows], list(heldout_episodes))
    heldout_rows = event_rows[heldout_mask]
    train_rows = event_rows[~heldout_mask]
    if not len(train_rows):
        raise ValueError("nearest-neighbor analysis requires training events")

    visual = np.concatenate([front_features, side_features], axis=1).astype(np.float64)
    norms = np.linalg.norm(visual, axis=1, keepdims=True)
    visual = np.divide(visual, norms, out=np.zeros_like(visual), where=norms > 0)
    body = states[:, :GRIPPER_INDEX].astype(np.float64)
    body_mean = body[train_rows].mean(axis=0)
    body_scale = body[train_rows].std(axis=0)
    body_scale[body_scale == 0] = 1.0
    body = (body - body_mean) / body_scale

    rows: list[dict[str, object]] = []
    for heldout_row in heldout_rows:
        visual_distances = 1.0 - visual[train_rows] @ visual[heldout_row]
        body_distances = np.sqrt(
            np.mean((body[train_rows] - body[heldout_row]) ** 2, axis=1)
        )
        visual_row = train_rows[int(np.argmin(visual_distances))]
        body_row = train_rows[int(np.argmin(body_distances))]
        rows.append(
            {
                "heldout_episode": int(episodes[heldout_row]),
                "heldout_frame": int(frames[heldout_row]),
                "visual_neighbor_episode": int(episodes[visual_row]),
                "visual_neighbor_frame": int(frames[visual_row]),
                "visual_cosine_distance": float(visual_distances.min()),
                "body_neighbor_episode": int(episodes[body_row]),
                "body_neighbor_frame": int(frames[body_row]),
                "body_standardized_distance": float(body_distances.min()),
            }
        )
    return rows


def body_prefix_mae_per_sample(
    target_actions: np.ndarray,
    predicted_actions: np.ndarray,
    action_is_pad: np.ndarray,
    *,
    executed_steps: int,
) -> np.ndarray:
    """Return five-body-joint MAE for each actually executed, non-pad prefix."""
    target_actions = np.asarray(target_actions)
    predicted_actions = np.asarray(predicted_actions)
    action_is_pad = np.asarray(action_is_pad)
    steps = min(executed_steps, target_actions.shape[1], predicted_actions.shape[1])
    valid = ~action_is_pad[:, :steps]
    errors = np.abs(
        predicted_actions[:, :steps, :GRIPPER_INDEX]
        - target_actions[:, :steps, :GRIPPER_INDEX]
    )
    totals = (errors * valid[:, :, None]).sum(axis=(1, 2))
    counts = valid.sum(axis=1) * GRIPPER_INDEX
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def summarize_body_prefix_maes(
    per_sample_mae: np.ndarray,
    close_onset: np.ndarray,
) -> dict[str, float | None]:
    """Summarize body accuracy for the full set and grasp-onset subset."""
    per_sample_mae = np.asarray(per_sample_mae, dtype=np.float64)
    close_onset = np.asarray(close_onset, dtype=bool)
    finite = np.isfinite(per_sample_mae)
    close_finite = finite & close_onset
    return {
        "body_mae_executed_prefix": (
            None if not finite.any() else float(per_sample_mae[finite].mean())
        ),
        "body_p95_executed_prefix": (
            None if not finite.any() else float(np.percentile(per_sample_mae[finite], 95))
        ),
        "body_mae_close_onset_prefix": (
            None if not close_finite.any() else float(per_sample_mae[close_finite].mean())
        ),
    }


def compute_event_metrics(
    state: np.ndarray,
    target_actions: np.ndarray,
    predicted_actions: np.ndarray,
    *,
    target_index: int,
    executed_steps: int,
) -> dict[str, int]:
    """Count grasp/release events using only actions the runtime would execute."""
    state = np.asarray(state)
    target_actions = np.asarray(target_actions)
    predicted_actions = np.asarray(predicted_actions)

    state_grip = state[:, GRIPPER_INDEX]
    target_grip = target_actions[:, target_index, GRIPPER_INDEX]
    predicted_grip = predicted_actions[:, :, GRIPPER_INDEX]
    executed_grip = predicted_grip[:, :executed_steps]
    first_grip = executed_grip[:, 0]

    close_onset = (state_grip >= GRIPPER_THRESHOLD) & (target_grip <= state_grip - 4.0)
    close_direction_hit = close_onset & (first_grip <= state_grip - 2.0)
    close_crossing_target = close_onset & (target_grip < GRIPPER_THRESHOLD)
    close_crossing_hit = close_crossing_target & (executed_grip < GRIPPER_THRESHOLD).any(axis=1)
    close_in_chunk = close_onset & (predicted_grip < GRIPPER_THRESHOLD).any(axis=1)
    close_in_executed_prefix = close_onset & (executed_grip < GRIPPER_THRESHOLD).any(axis=1)

    open_phase = (state_grip >= GRIPPER_THRESHOLD) & (target_grip >= GRIPPER_THRESHOLD)
    open_false_close = open_phase & (executed_grip < GRIPPER_THRESHOLD).any(axis=1)
    closed_hold = (state_grip < GRIPPER_THRESHOLD) & (target_grip < GRIPPER_THRESHOLD)

    release_onset = (state_grip < GRIPPER_THRESHOLD) & (target_grip >= state_grip + 4.0)
    release_direction_hit = release_onset & (first_grip >= state_grip + 2.0)
    release_crossing_target = release_onset & (target_grip >= GRIPPER_THRESHOLD)
    release_crossing_hit = release_crossing_target & (
        executed_grip >= GRIPPER_THRESHOLD
    ).any(axis=1)

    masks = {
        "close_onset": close_onset,
        "close_direction_hit": close_direction_hit,
        "close_crossing_target": close_crossing_target,
        "close_crossing_hit": close_crossing_hit,
        "close_in_chunk": close_in_chunk,
        "close_in_executed_prefix": close_in_executed_prefix,
        "open_phase": open_phase,
        "open_false_close": open_false_close,
        "closed_hold": closed_hold,
        "release_onset": release_onset,
        "release_direction_hit": release_direction_hit,
        "release_crossing_target": release_crossing_target,
        "release_crossing_hit": release_crossing_hit,
    }
    return {name: int(mask.sum()) for name, mask in masks.items()}


def make_event_rows(
    state: np.ndarray,
    target_actions: np.ndarray,
    predicted_actions: np.ndarray,
    *,
    episodes: np.ndarray,
    frames: np.ndarray,
    target_index: int,
    executed_steps: int,
) -> list[dict[str, int | float | str | bool]]:
    """Create auditable per-frame rows for event and execution-prefix analysis."""
    state = np.asarray(state)
    target_actions = np.asarray(target_actions)
    predicted_actions = np.asarray(predicted_actions)
    state_grip = state[:, GRIPPER_INDEX]
    target = target_actions[:, target_index]
    target_grip = target[:, GRIPPER_INDEX]
    predicted_h1 = predicted_actions[:, 0]
    offsets = earliest_threshold_offset(
        predicted_actions[:, :, GRIPPER_INDEX],
        threshold=GRIPPER_THRESHOLD,
    )

    rows = []
    for index in range(len(state)):
        if state_grip[index] >= GRIPPER_THRESHOLD and target_grip[index] <= state_grip[index] - 4.0:
            phase = "close_onset"
        elif state_grip[index] < GRIPPER_THRESHOLD and target_grip[index] >= state_grip[index] + 4.0:
            phase = "release_onset"
        elif state_grip[index] >= GRIPPER_THRESHOLD and target_grip[index] >= GRIPPER_THRESHOLD:
            phase = "open"
        elif state_grip[index] < GRIPPER_THRESHOLD and target_grip[index] < GRIPPER_THRESHOLD:
            phase = "closed_hold"
        else:
            phase = "transition_other"

        offset = int(offsets[index])
        rows.append(
            {
                "episode": int(episodes[index]),
                "frame": int(frames[index]),
                "phase": phase,
                "state_grip": float(state_grip[index]),
                "target_grip": float(target_grip[index]),
                "predicted_grip_h1": float(predicted_h1[index, GRIPPER_INDEX]),
                "d_t": offset,
                "within_executed_prefix": 0 <= offset < executed_steps,
                "body_mae_h1": float(
                    np.abs(predicted_h1[index, :GRIPPER_INDEX] - target[index, :GRIPPER_INDEX]).mean()
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Repeat for each checkpoint to evaluate.",
    )
    parser.add_argument(
        "--action-steps",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Override the executed prefix for a named ACT/SmolVLA/Diffusion checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/zhuokai/hand-teleop/datasets/hand_tracking_pv_carton_phase_b"),
    )
    parser.add_argument("--repo-id", default="stevenzenith/hand_tracking_pv_carton_phase_b")
    parser.add_argument("--episodes", default="0,5,10,16,23,26")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--probe-frames",
        help="Run only targeted probes at comma-separated EPISODE:FRAME keys.",
    )
    parser.add_argument("--gripper-values", default="100,95,90,85,80")
    parser.add_argument("--shift-pixels", default="0,20,40,60")
    parser.add_argument("--temporal-offsets", default="-2,-1,1,2")
    parser.add_argument(
        "--audit-feature-cache",
        type=Path,
        help="Optional ResNet feature cache used for event neighbors and full audit artifacts.",
    )
    parser.add_argument(
        "--episode-map",
        type=Path,
        default=Path(
            "/home/zhuokai/hand-teleop/datasets/hand_tracking_pv_carton_phase_b/source_episode_map.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_policy_bundle(checkpoint: Path, *, device: str, action_steps: int | None):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = device
    if action_steps is not None:
        config.n_action_steps = action_steps
    if config.type == "diffusion":
        config.noise_scheduler_type = "DDIM"
        config.num_inference_steps = 10
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return config, policy, preprocessor, postprocessor


def make_validation_dataset(
    config,
    *,
    dataset_root: Path,
    repo_id: str,
    episodes: list[int],
):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    return LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        image_transforms=None,
        video_backend="torchcodec",
        return_uint8=True,
    )


def evaluate_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    episodes: list[int],
    device: str,
    action_steps: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_batches: int | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    from torch.utils.data import DataLoader

    config, policy, preprocessor, postprocessor = load_policy_bundle(
        checkpoint,
        device=device,
        action_steps=action_steps,
    )
    dataset = make_validation_dataset(
        config,
        dataset_root=dataset_root,
        repo_id=repo_id,
        episodes=episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    counts = {
        "close_onset": 0,
        "close_direction_hit": 0,
        "close_crossing_target": 0,
        "close_crossing_hit": 0,
        "close_in_chunk": 0,
        "close_in_executed_prefix": 0,
        "open_phase": 0,
        "open_false_close": 0,
        "closed_hold": 0,
        "release_onset": 0,
        "release_direction_hit": 0,
        "release_crossing_target": 0,
        "release_crossing_hit": 0,
    }
    event_rows: list[dict[str, object]] = []
    body_prefix_maes: list[float] = []
    body_close_onset: list[bool] = []
    trace_parts: dict[str, list[np.ndarray]] = {
        "episode": [],
        "frame": [],
        "state": [],
        "target_actions": [],
        "predicted_actions": [],
        "action_is_pad": [],
    }
    sample_count = 0
    target_index = current_target_index(config.n_obs_steps)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch["observation.state"] = batch["observation.state"][
            ..., : config.input_features["observation.state"].shape[-1]
        ]
        raw_state = batch["observation.state"].clone()
        if raw_state.ndim == 3:
            raw_state = raw_state[:, -1]
        raw_target = batch["action"].clone()
        valid = ~batch["action_is_pad"][:, target_index].bool()
        episode_indices = batch["episode_index"].reshape(-1)
        frame_indices = batch["frame_index"].reshape(-1)
        for camera_key in dataset.meta.camera_keys:
            if batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float().div_(255.0)
        processed = preprocessor(batch)
        predicted = predict_physical_chunk(
            policy,
            processed,
            postprocessor,
            seed=seed + batch_index,
        ).detach().cpu()
        executed_steps = min(config.n_action_steps, predicted.shape[1])

        state_np = raw_state[valid].cpu().numpy()
        target_np = raw_target[valid].cpu().numpy()
        predicted_np = predicted[valid].cpu().numpy()
        prefix_mae = body_prefix_mae_per_sample(
            target_np,
            predicted_np,
            batch["action_is_pad"][valid].cpu().numpy(),
            executed_steps=executed_steps,
        )
        state_grip = state_np[:, GRIPPER_INDEX]
        target_grip = target_np[:, target_index, GRIPPER_INDEX]
        close_onset = (state_grip >= GRIPPER_THRESHOLD) & (
            target_grip <= state_grip - 4.0
        )
        body_prefix_maes.extend(prefix_mae.tolist())
        body_close_onset.extend(close_onset.tolist())
        batch_counts = compute_event_metrics(
            state_np,
            target_np,
            predicted_np,
            target_index=target_index,
            executed_steps=executed_steps,
        )
        for key, value in batch_counts.items():
            counts[key] += value
        rows = make_event_rows(
            state_np,
            target_np,
            predicted_np,
            episodes=episode_indices[valid].cpu().numpy(),
            frames=frame_indices[valid].cpu().numpy(),
            target_index=target_index,
            executed_steps=executed_steps,
        )
        for row in rows:
            row.update(
                {
                    "policy": name,
                    "policy_type": config.type,
                    "checkpoint": str(checkpoint),
                    "seed": seed,
                    "n_obs_steps": config.n_obs_steps,
                    "executed_steps": executed_steps,
                }
            )
        event_rows.extend(rows)
        trace_parts["episode"].append(episode_indices[valid].cpu().numpy())
        trace_parts["frame"].append(frame_indices[valid].cpu().numpy())
        trace_parts["state"].append(state_np)
        trace_parts["target_actions"].append(target_np)
        trace_parts["predicted_actions"].append(predicted_np)
        trace_parts["action_is_pad"].append(
            batch["action_is_pad"][valid].cpu().numpy()
        )
        sample_count += int(valid.sum())

    summary = {
        "policy": name,
        "policy_type": config.type,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "episodes": episodes,
        "samples": sample_count,
        "n_obs_steps": config.n_obs_steps,
        "chunk_size": getattr(config, "chunk_size", None),
        "horizon": getattr(config, "horizon", None),
        "executed_steps": config.n_action_steps,
        "target_index": target_index,
        "counts": counts,
        "rates": summarize_event_counts(counts),
        "body_prefix": summarize_body_prefix_maes(
            np.asarray(body_prefix_maes),
            np.asarray(body_close_onset),
        ),
    }
    del dataset, loader, policy, preprocessor, postprocessor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    trace = {key: np.concatenate(parts, axis=0) for key, parts in trace_parts.items()}
    return summary, event_rows, trace


def _clone_batch(batch: dict[str, object]) -> dict[str, object]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _collect_probe_batches(dataset, keys: set[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    from torch.utils.data import DataLoader

    found = {}
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        key = (
            int(batch["episode_index"].reshape(-1)[0]),
            int(batch["frame_index"].reshape(-1)[0]),
        )
        if key in keys:
            found[key] = _clone_batch(batch)
            if len(found) == len(keys):
                break
    missing = keys - set(found)
    if missing:
        raise ValueError(f"probe frames are absent from the selected dataset: {sorted(missing)}")
    return found


def evaluate_probes(
    *,
    name: str,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    probe_frames: list[tuple[int, int]],
    device: str,
    action_steps: int | None,
    seed: int,
    gripper_values: list[float],
    shift_pixels: list[int],
    temporal_offsets: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config, policy, preprocessor, postprocessor = load_policy_bundle(
        checkpoint,
        device=device,
        action_steps=action_steps,
    )
    episodes = sorted({episode for episode, _ in probe_frames})
    dataset = make_validation_dataset(
        config,
        dataset_root=dataset_root,
        repo_id=repo_id,
        episodes=episodes,
    )
    needed = set(probe_frames)
    needed.update(
        (episode, frame + offset)
        for episode, frame in probe_frames
        for offset in temporal_offsets
    )
    batches = _collect_probe_batches(dataset, needed)
    target_index = current_target_index(config.n_obs_steps)
    executed_steps = config.n_action_steps
    rows = []

    def run_variant(
        base: dict[str, object],
        *,
        episode: int,
        frame: int,
        variant_type: str,
        variant: str,
        gripper_input: float | None = None,
        image_source: dict[str, object] | None = None,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        batch = _clone_batch(base)
        batch["observation.state"] = batch["observation.state"][
            ..., : config.input_features["observation.state"].shape[-1]
        ]
        raw_state = batch["observation.state"].clone()
        current_state = raw_state[:, -1] if raw_state.ndim == 3 else raw_state
        target = batch["action"][:, target_index]
        if gripper_input is not None:
            batch["observation.state"] = set_current_gripper(
                batch["observation.state"], gripper_input
            )
        if image_source is not None:
            for camera_key in dataset.meta.camera_keys:
                batch[camera_key] = image_source[camera_key].clone()
        for camera_key in dataset.meta.camera_keys:
            if dx or dy:
                batch[camera_key] = translate_images(batch[camera_key], dx=dx, dy=dy)
            if batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float().div_(255.0)
        processed = preprocessor(batch)
        predicted = predict_physical_chunk(
            policy,
            processed,
            postprocessor,
            seed=seed,
        ).detach().cpu()
        predicted_np = predicted.numpy()
        offset = int(
            earliest_threshold_offset(
                predicted_np[:, :, GRIPPER_INDEX], threshold=GRIPPER_THRESHOLD
            )[0]
        )
        rows.append(
            {
                "policy": name,
                "policy_type": config.type,
                "episode": episode,
                "frame": frame,
                "variant_type": variant_type,
                "variant": variant,
                "state_grip": float(current_state[0, GRIPPER_INDEX]),
                "gripper_input": (
                    float(current_state[0, GRIPPER_INDEX])
                    if gripper_input is None
                    else float(gripper_input)
                ),
                "target_grip": float(target[0, GRIPPER_INDEX]),
                "predicted_grip_h1": float(predicted_np[0, 0, GRIPPER_INDEX]),
                "d_t": offset,
                "within_executed_prefix": 0 <= offset < executed_steps,
                "body_mae_h1": float(
                    np.abs(
                        predicted_np[0, 0, :GRIPPER_INDEX]
                        - target[0, :GRIPPER_INDEX].cpu().numpy()
                    ).mean()
                ),
                "predicted_gripper_chunk": json.dumps(
                    predicted_np[0, :, GRIPPER_INDEX].tolist()
                ),
            }
        )

    for episode, frame in probe_frames:
        base = batches[(episode, frame)]
        for value in gripper_values:
            run_variant(
                base,
                episode=episode,
                frame=frame,
                variant_type="gripper_input",
                variant=f"q={value:g}",
                gripper_input=value,
            )
        for pixels in shift_pixels:
            shifts = [(0, 0)] if pixels == 0 else [(pixels, 0), (-pixels, 0), (0, pixels), (0, -pixels)]
            for dx, dy in shifts:
                run_variant(
                    base,
                    episode=episode,
                    frame=frame,
                    variant_type="synchronized_image_shift",
                    variant=f"dx={dx},dy={dy}",
                    dx=dx,
                    dy=dy,
                )
        for offset in temporal_offsets:
            run_variant(
                base,
                episode=episode,
                frame=frame,
                variant_type="synchronized_time_offset",
                variant=f"frames={offset:+d}",
                image_source=batches[(episode, frame + offset)],
            )

    summary = {
        "policy": name,
        "policy_type": config.type,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "probe_frames": [f"{episode}:{frame}" for episode, frame in probe_frames],
        "executed_steps": executed_steps,
        "rows": len(rows),
    }
    del dataset, policy, preprocessor, postprocessor
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_episode_metadata(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["episode_index"]): row
            for row in csv.DictReader(handle)
        }


def _add_episode_metadata(
    row: dict[str, object],
    metadata: dict[int, dict[str, str]],
    *,
    episode_key: str,
    prefix: str = "",
) -> None:
    episode = int(row[episode_key])
    for key in (
        "source_episode_index",
        "table_position",
        "grasp_depth",
        "repeat_index",
        "jaw_offset_mm",
        "session_id",
        "outcome",
    ):
        row[f"{prefix}{key}"] = metadata[episode][key]


def _summarize_temporal_episode_rows(
    rows: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    result = {}
    for episode in sorted({int(row["episode"]) for row in rows}):
        episode_rows = [row for row in rows if int(row["episode"]) == episode]
        summary = summarize_temporal_consistency_rows(episode_rows)
        result[episode] = {
            "temporal_body_overlap_mae_deg": summary["body_overlap_mae_deg"],
            "temporal_body_overlap_p95_deg": summary["body_overlap_p95_deg"],
            "temporal_gripper_overlap_mae_deg": summary["gripper_overlap_mae_deg"],
            "temporal_threshold_disagreement_rate": summary[
                "threshold_disagreement_rate"
            ],
            "temporal_close_available_disagreements": summary[
                "close_available_disagreement_count"
            ],
        }
    return result


def _write_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    *,
    policies: list[str],
    episodes: list[int],
    metric: str,
    title: str,
    color_map: str,
) -> None:
    import matplotlib.pyplot as plt

    values = np.full((len(policies), len(episodes)), np.nan, dtype=np.float64)
    lookup = {
        (str(row["policy"]), int(row["episode"])): row.get(metric)
        for row in rows
    }
    for policy_index, policy in enumerate(policies):
        for episode_index, episode in enumerate(episodes):
            value = lookup.get((policy, episode))
            if value is not None:
                values[policy_index, episode_index] = float(value)

    width = max(7.0, 0.7 * len(episodes) + 3.0)
    height = max(6.0, 0.38 * len(policies) + 2.5)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap=color_map)
    axis.set_xticks(range(len(episodes)), labels=[str(value) for value in episodes])
    axis.set_yticks(range(len(policies)), labels=policies)
    axis.set_xlabel("held-out episode")
    axis.set_ylabel("checkpoint")
    axis.set_title(title)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_audit_artifacts(
    output_dir: Path,
    summaries: list[dict[str, object]],
    traces: dict[str, dict[str, np.ndarray]],
    *,
    feature_cache_path: Path,
    episode_map_path: Path,
) -> None:
    """Write the event screen, episode heatmaps, neighbors, and STAC-style audit."""
    metadata = load_episode_metadata(episode_map_path)
    episode_rows_all: list[dict[str, object]] = []
    temporal_rows_all: list[dict[str, object]] = []
    temporal_summaries: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []

    for summary in summaries:
        policy = str(summary["policy"])
        trace = traces[policy]
        target_index = int(summary["target_index"])
        executed_steps = int(summary["executed_steps"])
        current_actions = trace["target_actions"][:, target_index]
        labels = derive_event_labels(
            trace["episode"],
            trace["frame"],
            trace["state"],
            current_actions,
        )
        episode_rows = make_episode_audit_rows(
            trace["episode"],
            trace["frame"],
            trace["state"],
            trace["target_actions"],
            trace["predicted_actions"],
            trace["action_is_pad"],
            target_index=target_index,
            executed_steps=executed_steps,
            labels=labels,
        )
        temporal_rows = make_temporal_consistency_rows(
            trace["predicted_actions"],
            trace["action_is_pad"],
            episodes=trace["episode"],
            frames=trace["frame"],
            query_stride=1,
        )
        temporal_summary = summarize_temporal_consistency_rows(temporal_rows)
        temporal_by_episode = _summarize_temporal_episode_rows(temporal_rows)
        for row in episode_rows:
            row.update(
                {
                    "policy": policy,
                    "checkpoint": summary["checkpoint"],
                    **temporal_by_episode[int(row["episode"])],
                }
            )
            _add_episode_metadata(row, metadata, episode_key="episode")
        for row in temporal_rows:
            row.update({"policy": policy, "checkpoint": summary["checkpoint"]})
        temporal_summary.update({"policy": policy, "checkpoint": summary["checkpoint"]})

        target_body = trace["target_actions"][:, target_index, :GRIPPER_INDEX]
        hold_errors = np.abs(trace["state"][:, :GRIPPER_INDEX] - target_body).mean(axis=1)
        screening_rows.append(
            make_screening_row(
                summary,
                episode_rows,
                hold_body_mae=float(hold_errors.mean()),
                hold_body_p95=float(np.quantile(hold_errors, 0.95)),
                temporal_summary=temporal_summary,
            )
        )
        episode_rows_all.extend(episode_rows)
        temporal_rows_all.extend(temporal_rows)
        temporal_summaries.append(temporal_summary)

    screening_rows.sort(
        key=lambda row: (
            -int(bool(row["joint_gate"])),
            -int(row["gates_passed"]),
            -int(row["on_time_episodes"]),
            -int(row["event_window_hit_episodes"]),
            int(row["pure_false_close_frames"]),
            float(row["body_mae_h1"]),
            float(row["temporal_body_overlap_p95_deg"]),
        )
    )
    for rank, row in enumerate(screening_rows, start=1):
        row["screening_rank"] = rank

    write_csv(output_dir / "screening.csv", screening_rows)
    write_csv(output_dir / "episode_metrics.csv", episode_rows_all)
    write_csv(output_dir / "temporal_consistency.csv", temporal_rows_all)
    (output_dir / "temporal_summary.json").write_text(
        json.dumps(temporal_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with np.load(feature_cache_path) as stored:
        cache = {key: stored[key] for key in stored.files}
    feature_labels = derive_event_labels(
        cache["episode"], cache["frame"], cache["state"], cache["action"]
    )
    heldout_episodes = {int(value) for value in summaries[0]["episodes"]}
    neighbor_rows = nearest_event_neighbors(
        cache["episode"],
        cache["frame"],
        cache["state"],
        cache["front"],
        cache["side"],
        event_rows=feature_labels["close_event_rows"],
        heldout_episodes=heldout_episodes,
    )
    train_event_rows = feature_labels["close_event_rows"][
        ~np.isin(
            cache["episode"][feature_labels["close_event_rows"]],
            list(heldout_episodes),
        )
    ]
    train_reference_rows: list[dict[str, object]] = []
    for episode in np.unique(cache["episode"][train_event_rows]):
        train_reference_rows.extend(
            nearest_event_neighbors(
                cache["episode"],
                cache["frame"],
                cache["state"],
                cache["front"],
                cache["side"],
                event_rows=train_event_rows,
                heldout_episodes={int(episode)},
            )
        )
    visual_reference = np.asarray(
        [row["visual_cosine_distance"] for row in train_reference_rows], dtype=np.float64
    )
    body_reference = np.asarray(
        [row["body_standardized_distance"] for row in train_reference_rows],
        dtype=np.float64,
    )
    for row in neighbor_rows:
        row["visual_train_loo_percentile"] = float(
            100.0 * np.mean(visual_reference <= float(row["visual_cosine_distance"]))
        )
        row["body_train_loo_percentile"] = float(
            100.0 * np.mean(body_reference <= float(row["body_standardized_distance"]))
        )
        _add_episode_metadata(row, metadata, episode_key="heldout_episode")
        _add_episode_metadata(
            row,
            metadata,
            episode_key="visual_neighbor_episode",
            prefix="visual_neighbor_",
        )
        _add_episode_metadata(
            row,
            metadata,
            episode_key="body_neighbor_episode",
            prefix="body_neighbor_",
        )
    write_csv(output_dir / "nearest_neighbors.csv", neighbor_rows)

    condition_rows: list[dict[str, object]] = []
    for policy in [str(summary["policy"]) for summary in summaries]:
        policy_rows = [row for row in episode_rows_all if row["policy"] == policy]
        for factor in ("table_position", "grasp_depth", "jaw_offset_mm"):
            for value in sorted({str(row[factor]) for row in policy_rows}):
                group = [row for row in policy_rows if str(row[factor]) == value]
                condition_rows.append(
                    {
                        "policy": policy,
                        "factor": factor,
                        "value": value,
                        "episodes": len(group),
                        "on_time_rate": float(
                            np.mean([row["first_crossing_timing"] == "on_time" for row in group])
                        ),
                        "body_mae_h1": float(np.mean([row["body_mae_h1"] for row in group])),
                        "pure_false_close_frames": int(
                            sum(int(row["pure_false_close_frames"]) for row in group)
                        ),
                        "temporal_body_overlap_mae_deg": float(
                            np.mean([row["temporal_body_overlap_mae_deg"] for row in group])
                        ),
                    }
                )
    write_csv(output_dir / "condition_metrics.csv", condition_rows)

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir()
    policies = [str(summary["policy"]) for summary in summaries]
    episodes = sorted(heldout_episodes)
    for metric, title, color_map in (
        ("body_mae_h1", "Body first-action MAE (deg)", "magma"),
        ("first_crossing_offset", "First gripper crossing offset (frames)", "coolwarm"),
        ("pure_false_close_frames", "Pure-open false-close frames", "Reds"),
        (
            "temporal_body_overlap_mae_deg",
            "Adjacent-chunk body overlap MAE (deg)",
            "viridis",
        ),
        (
            "temporal_threshold_disagreement_rate",
            "Adjacent-chunk gripper threshold disagreement rate",
            "plasma",
        ),
    ):
        _write_heatmap(
            heatmap_dir / f"{metric}.png",
            episode_rows_all,
            policies=policies,
            episodes=episodes,
            metric=metric,
            title=title,
            color_map=color_map,
        )

    method = {
        "screening": {
            "weighted_score": False,
            "gate_order": [
                "at least 5/6 first crossings and event-window hits within t-2:t+2",
                "zero false-close frames on pure-open negatives outside every t-2:t+2 band",
                "at least 11/12 release threshold crossings",
                "closed gripper at all six action-defined pre-lift proxy anchors",
                "body mean and p95 first-action MAE both beat copying current body state",
            ],
            "rank_after_gates": [
                "number of component gates passed descending",
                "on-time episodes descending",
                "event-window hits descending",
                "pure false-close frames ascending",
                "body first-action MAE ascending",
                "temporal body-overlap p95 ascending",
            ],
        },
        "pre_lift_proxy": (
            "first demonstration frame after initial close onset whose gripper action is <=32.5 deg; "
            "the formal dataset has no end-effector height/contact label, so this is not lift detection"
        ),
        "temporal_consistency": (
            "deterministic STAC-style comparison of adjacent ACT chunks aligned on identical absolute "
            "future timesteps; reports physical-degree body/gripper MAE and 90-deg threshold disagreement; "
            "not Sentinel MMD and not a calibrated runtime failure detector"
        ),
        "nearest_neighbor": (
            "dual-view ResNet18 features use concatenated cosine distance; five body joints use RMS "
            "Euclidean distance after train-event standardization; visual/body neighbors are selected separately"
        ),
        "source": "https://proceedings.mlr.press/v270/agia25a.html",
    }
    (output_dir / "audit_method.json").write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            raise ValueError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)

    policies = dict(parse_named_assignment(value) for value in args.policy)
    action_steps = {
        name: int(value) for name, value in map(parse_named_assignment, args.action_steps)
    }
    unknown_overrides = set(action_steps) - set(policies)
    if unknown_overrides:
        raise ValueError(f"action-step override has no policy: {sorted(unknown_overrides)}")
    episodes = [int(value) for value in args.episodes.split(",")]

    if args.probe_frames:
        probe_frames = [
            tuple(int(part) for part in value.split(":"))
            for value in args.probe_frames.split(",")
        ]
        probe_summaries = []
        probe_rows = []
        for name, checkpoint_text in policies.items():
            summary, rows = evaluate_probes(
                name=name,
                checkpoint=Path(checkpoint_text),
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                probe_frames=probe_frames,
                device=args.device,
                action_steps=action_steps.get(name),
                seed=args.seed,
                gripper_values=[float(value) for value in args.gripper_values.split(",")],
                shift_pixels=[int(value) for value in args.shift_pixels.split(",")],
                temporal_offsets=[int(value) for value in args.temporal_offsets.split(",")],
            )
            probe_summaries.append(summary)
            probe_rows.extend(rows)
            print(json.dumps(summary, sort_keys=True), flush=True)
        (args.output_dir / "probe_summary.json").write_text(
            json.dumps(probe_summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(args.output_dir / "probes.csv", probe_rows)
        print(f"wrote {args.output_dir}")
        return

    summaries = []
    event_rows = []
    traces: dict[str, dict[str, np.ndarray]] = {}
    for name, checkpoint_text in policies.items():
        summary, rows, trace = evaluate_checkpoint(
            name=name,
            checkpoint=Path(checkpoint_text),
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            episodes=episodes,
            device=args.device,
            action_steps=action_steps.get(name),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            max_batches=args.max_batches,
        )
        summaries.append(summary)
        event_rows.extend(rows)
        traces[name] = trace
        print(json.dumps(summary, sort_keys=True), flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "events.csv", event_rows)
    if args.audit_feature_cache is not None:
        trace_dir = args.output_dir / "traces"
        trace_dir.mkdir()
        for name, trace in traces.items():
            np.savez_compressed(trace_dir / f"{name}.npz", **trace)
        write_audit_artifacts(
            args.output_dir,
            summaries,
            traces,
            feature_cache_path=args.audit_feature_cache,
            episode_map_path=args.episode_map,
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
