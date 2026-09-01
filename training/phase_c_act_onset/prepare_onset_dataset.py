#!/usr/bin/env python
"""Create a train-only Phase C dataset with pre-onset gripper-state replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


GRIPPER_INDEX = 5


def select_preonset_frames(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    chunk_size: int,
    threshold: float,
) -> np.ndarray:
    """Select open-state frames whose current action chunk contains closure."""
    state_grip = np.asarray(states)[:, GRIPPER_INDEX]
    action_grip = np.asarray(actions)[:, GRIPPER_INDEX]
    selected = np.zeros(len(state_grip), dtype=bool)
    for index in range(len(state_grip)):
        end = min(len(action_grip), index + chunk_size)
        selected[index] = state_grip[index] >= threshold and np.any(
            action_grip[index:end] < threshold
        )
    return selected


def replace_selected_gripper(
    states: np.ndarray,
    selected: np.ndarray,
    *,
    values: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    """Cycle replacement values across selected rows without changing body state."""
    replaced = np.asarray(states, dtype=np.float32).copy()
    counts = {f"{value:g}": 0 for value in values}
    for replacement_index, row_index in enumerate(np.flatnonzero(selected)):
        value = values[replacement_index % len(values)]
        replaced[row_index, GRIPPER_INDEX] = value
        counts[f"{value:g}"] += 1
    return replaced, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--values", default="90,95,100")
    return parser.parse_args()


def _copy_dataset(source: Path, output: Path) -> None:
    def copy_or_link(src: str, dst: str):
        if f"{os.sep}videos{os.sep}" in src:
            return os.link(src, dst)
        return shutil.copy2(src, dst)

    shutil.copytree(source, output, copy_function=copy_or_link)


def _state_column(states: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(states.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, states.shape[1])


def _update_gripper_stats(stats_path: Path, gripper_values: np.ndarray) -> None:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    state_stats = stats["observation.state"]
    state_stats["min"][GRIPPER_INDEX] = float(gripper_values.min())
    state_stats["max"][GRIPPER_INDEX] = float(gripper_values.max())
    state_stats["mean"][GRIPPER_INDEX] = float(gripper_values.mean())
    state_stats["std"][GRIPPER_INDEX] = float(gripper_values.std())
    for key, quantile in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        state_stats[key][GRIPPER_INDEX] = float(np.quantile(gripper_values, quantile))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    values = tuple(float(value) for value in args.values.split(","))
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    _copy_dataset(args.source, args.output)

    changed_rows = 0
    episode_counts = {}
    replacement_counts = {f"{value:g}": 0 for value in values}
    all_gripper_values = []
    source_files = sorted((args.source / "data").rglob("*.parquet"))
    for source_path in source_files:
        relative = source_path.relative_to(args.source)
        output_path = args.output / relative
        source_table = pq.read_table(source_path)
        states = np.asarray(source_table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(source_table["action"].to_pylist(), dtype=np.float32)
        selected = select_preonset_frames(
            states,
            actions,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
        )
        replaced, counts = replace_selected_gripper(states, selected, values=values)
        state_index = source_table.schema.get_field_index("observation.state")
        output_table = source_table.set_column(
            state_index,
            "observation.state",
            _state_column(replaced),
        )
        pq.write_table(output_table, output_path)

        episode = int(source_table["episode_index"][0].as_py())
        episode_counts[str(episode)] = int(selected.sum())
        changed_rows += int(selected.sum())
        for key, count in counts.items():
            replacement_counts[key] += count
        all_gripper_values.append(replaced[:, GRIPPER_INDEX])

    gripper_values = np.concatenate(all_gripper_values)
    _update_gripper_stats(args.output / "meta/stats.json", gripper_values)
    video_files = sorted((args.source / "videos").rglob("*.mp4"))
    videos_hardlinked = all(
        source_path.stat().st_ino == (args.output / source_path.relative_to(args.source)).stat().st_ino
        for source_path in video_files
    )
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "rule": {
            "state_gripper_at_least": args.threshold,
            "future_chunk_size": args.chunk_size,
            "future_action_gripper_below": args.threshold,
            "replacement_values": list(values),
        },
        "total_rows": int(len(gripper_values)),
        "changed_rows": changed_rows,
        "episode_changed_rows": episode_counts,
        "replacement_counts": replacement_counts,
        "video_files": len(video_files),
        "videos_hardlinked": videos_hardlinked,
        "unchanged_contract": [
            "action",
            "observation.state[0:5]",
            "videos",
            "episode metadata",
            "tasks",
        ],
    }
    (args.output / "onset_state_replacement_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
