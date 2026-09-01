#!/usr/bin/env python3
"""Build the filtered Phase B LeRobot dataset used for policy training.

The output contains only the 30 formal Phase B episodes, reindexed in ascending
source-episode order. It keeps front/side video, six-joint state, and six-joint
action; audit-only PV fields and the constant grip-context state are removed.
Surplus video frames beyond each parquet-indexed episode are trimmed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from lerobot_teleoperator_so101_webcam.paths import dataset_root

import av
import numpy as np
import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.dataset_tools import (
    _keep_episodes_from_video_with_av,
    _write_parquet,
    delete_episodes,
    modify_tasks,
    remove_feature,
)
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_SOURCE_ROOT = Path(
    str(dataset_root() / "hand_tracking_pv_carton_dual_view")
)
DEFAULT_OUTPUT_ROOT = Path(
    str(dataset_root() / "hand_tracking_pv_carton_phase_b")
)
DEFAULT_MAP = Path(
    str(Path(__file__).parents[1] / "configs" / "phase_b_250g_actual_episode_map.csv")
)
DEFAULT_REPO_ID = "stevenzenith/hand_tracking_pv_carton_phase_b"
TASK = (
    "Gently grasp and lift the 250 g paper carton, tighten the gripper if it "
    "slips, then return it to the table and release it."
)

POLICY_FEATURES = {
    "action",
    "observation.state",
    "observation.images.front",
    "observation.images.side",
}
REQUIRED_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
STATE_KEY = "observation.state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--actual-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def load_formal_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 30:
        raise ValueError(f"Expected 30 formal rows, found {len(rows)} in {path}")
    if {int(row["group_no"]) for row in rows} != set(range(1, 31)):
        raise ValueError("Formal group numbers must be exactly 1..30")
    source_episodes = [int(row["dataset_episode"]) for row in rows]
    if len(set(source_episodes)) != 30:
        raise ValueError("Formal source episode indices must be unique")
    if any(row["status"] != "recorded" for row in rows):
        raise ValueError("Every formal row must have status=recorded")
    return rows


def video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def trim_surplus_video_frames(dataset: LeRobotDataset) -> list[dict[str, int | str]]:
    trimmed: list[dict[str, int | str]] = []
    for episode_index in range(dataset.meta.total_episodes):
        episode = dataset.meta.episodes[episode_index]
        expected = int(episode["length"])
        for video_key in dataset.meta.video_keys:
            relative_path = dataset.meta.get_video_file_path(episode_index, video_key)
            video_path = dataset.root / relative_path
            observed = video_frame_count(video_path)
            if observed < expected:
                raise ValueError(
                    f"{video_key} episode {episode_index} has {observed} frames, expected {expected}"
                )
            if observed == expected:
                continue

            encoder = VideoEncoderConfig.from_video_info(
                dataset.meta.info.features[video_key].get("info")
            )
            trimmed_path = video_path.with_suffix(".trimmed.mp4")
            _keep_episodes_from_video_with_av(
                video_path,
                trimmed_path,
                [(0, expected)],
                dataset.meta.fps,
                encoder,
            )
            trimmed_count = video_frame_count(trimmed_path)
            if trimmed_count != expected:
                raise ValueError(
                    f"Trimmed {video_key} episode {episode_index} has {trimmed_count} frames, "
                    f"expected {expected}"
                )
            os.replace(trimmed_path, video_path)
            trimmed.append(
                {
                    "episode_index": episode_index,
                    "video_key": video_key,
                    "frames_before": observed,
                    "frames_after": expected,
                }
            )
    return trimmed


def slice_state_to_six_joints(dataset: LeRobotDataset) -> None:
    state_feature = dataset.meta.info.features[STATE_KEY]
    if tuple(state_feature["shape"]) != (9,):
        raise ValueError(f"Expected nine-dimensional source state, got {state_feature['shape']}")

    state_feature["shape"] = [6]
    state_feature["names"] = state_feature["names"][:6]
    write_info(dataset.meta.info, dataset.root)

    for parquet_path in sorted((dataset.root / "data").rglob("*.parquet")):
        frame_table = pd.read_parquet(parquet_path)
        frame_table[STATE_KEY] = frame_table[STATE_KEY].map(
            lambda value: np.asarray(value, dtype=np.float32)[:6]
        )
        temporary_path = parquet_path.with_suffix(".temporary.parquet")
        _write_parquet(frame_table, temporary_path, dataset.meta)
        os.replace(temporary_path, parquet_path)

    state_stats_prefix = f"stats/{STATE_KEY}/"
    for parquet_path in sorted((dataset.root / "meta" / "episodes").rglob("*.parquet")):
        episode_table = pd.read_parquet(parquet_path)
        for column in episode_table.columns:
            if column.startswith(state_stats_prefix) and not column.endswith("/count"):
                episode_table[column] = episode_table[column].map(
                    lambda value: np.asarray(value)[:6]
                )
        temporary_path = parquet_path.with_suffix(".temporary.parquet")
        episode_table.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, parquet_path)

    stats_path = dataset.root / "meta" / "stats.json"
    with stats_path.open() as handle:
        stats = json.load(handle)
    for stat_name, value in stats[STATE_KEY].items():
        if stat_name != "count":
            stats[STATE_KEY][stat_name] = value[:6]
    with stats_path.open("w") as handle:
        json.dump(stats, handle, indent=4)
        handle.write("\n")


def write_source_map(
    output_root: Path, formal_rows: list[dict[str, str]], source_episodes: list[int]
) -> None:
    rows_by_source = {int(row["dataset_episode"]): row for row in formal_rows}
    fieldnames = [
        "episode_index",
        "source_episode_index",
        "group_no",
        "table_position",
        "grasp_depth",
        "repeat_index",
        "jaw_offset_mm",
        "session_id",
        "attempt",
        "outcome",
    ]
    with (output_root / "source_episode_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode_index, source_episode in enumerate(source_episodes):
            row = rows_by_source[source_episode]
            writer.writerow(
                {
                    "episode_index": episode_index,
                    "source_episode_index": source_episode,
                    **{name: row[name] for name in fieldnames[2:]},
                }
            )


def validate_parquet(dataset: LeRobotDataset) -> None:
    expected_features = POLICY_FEATURES | REQUIRED_FEATURES
    if set(dataset.meta.features) != expected_features:
        raise ValueError(
            f"Unexpected output features: {sorted(set(dataset.meta.features) - expected_features)}; "
            f"missing: {sorted(expected_features - set(dataset.meta.features))}"
        )
    if dataset.meta.total_episodes != 30 or dataset.meta.total_frames != 6640:
        raise ValueError(
            f"Expected 30 episodes / 6640 frames, got "
            f"{dataset.meta.total_episodes} / {dataset.meta.total_frames}"
        )
    if tuple(dataset.meta.info.features[STATE_KEY]["shape"]) != (6,):
        raise ValueError("Output state is not six-dimensional")

    expected_index = 0
    observed_frames = 0
    for parquet_path in sorted((dataset.root / "data").rglob("*.parquet")):
        frame_table = pd.read_parquet(parquet_path)
        states = np.stack(frame_table[STATE_KEY].to_numpy())
        actions = np.stack(frame_table["action"].to_numpy())
        if states.shape[1:] != (6,) or actions.shape[1:] != (6,):
            raise ValueError(f"Unexpected state/action shape in {parquet_path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite state/action value in {parquet_path}")
        indices = frame_table["index"].to_numpy()
        if not np.array_equal(indices, np.arange(expected_index, expected_index + len(indices))):
            raise ValueError(f"Non-contiguous global index in {parquet_path}")
        expected_index += len(indices)
        observed_frames += len(frame_table)
    if observed_frames != 6640:
        raise ValueError(f"Parquet contains {observed_frames} frames, expected 6640")


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")

    formal_rows = load_formal_rows(args.actual_map)
    source_episodes = sorted(int(row["dataset_episode"]) for row in formal_rows)

    source = LeRobotDataset(
        repo_id="local/hand_tracking_pv_carton_dual_view",
        root=args.source_root,
    )
    if source.meta.total_episodes != 54 or source.meta.total_frames != 13369:
        raise ValueError(
            f"Unexpected source snapshot: {source.meta.total_episodes} episodes / "
            f"{source.meta.total_frames} frames"
        )

    episodes_to_delete = sorted(set(range(source.meta.total_episodes)) - set(source_episodes))
    with tempfile.TemporaryDirectory(prefix="phase_b_training_subset_") as temporary_dir:
        subset_root = Path(temporary_dir) / "subset"
        subset = delete_episodes(
            source,
            episode_indices=episodes_to_delete,
            output_dir=subset_root,
            repo_id="local/hand_tracking_pv_carton_phase_b_subset",
        )
        features_to_remove = sorted(
            set(subset.meta.features) - POLICY_FEATURES - REQUIRED_FEATURES
        )
        training = remove_feature(
            subset,
            feature_names=features_to_remove,
            output_dir=args.output_root,
            repo_id=args.repo_id,
        )

    modify_tasks(training, new_task=TASK)
    trimmed = trim_surplus_video_frames(training)
    slice_state_to_six_joints(training)
    write_source_map(args.output_root, formal_rows, source_episodes)

    manifest = {
        "repo_id": args.repo_id,
        "source_root": str(args.source_root),
        "source_episode_indices": source_episodes,
        "episode_order": "ascending source_episode_index",
        "episodes": 30,
        "frames": 6640,
        "fps": 10,
        "task": TASK,
        "policy_features": sorted(POLICY_FEATURES),
        "trimmed_videos": trimmed,
    }
    with (args.output_root / "phase_b_training_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    validate_parquet(training)
    print(f"Built {args.repo_id} at {args.output_root}")
    print("Episodes: 30; frames: 6640; features: action, six-joint state, front, side")
    print(f"Trimmed {len(trimmed)} video files with surplus frames")


if __name__ == "__main__":
    main()
