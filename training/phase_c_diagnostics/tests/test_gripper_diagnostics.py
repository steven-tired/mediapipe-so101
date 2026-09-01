import numpy as np
import pytest
import torch

from training.phase_c_diagnostics.gripper_diagnostics import (
    body_prefix_mae_per_sample,
    compute_event_metrics,
    current_target_index,
    derive_event_labels,
    earliest_threshold_offset,
    find_pre_lift_anchor_rows,
    make_event_rows,
    make_episode_audit_rows,
    make_screening_row,
    make_temporal_consistency_rows,
    nearest_event_neighbors,
    parse_named_assignment,
    predict_physical_chunk,
    set_current_gripper,
    summarize_body_prefix_maes,
    summarize_event_counts,
    summarize_temporal_consistency_rows,
    translate_images,
)


def test_episode_audit_keeps_event_hit_separate_from_early_false_close():
    episodes = np.zeros(10, dtype=np.int64)
    frames = np.arange(10)
    states = np.zeros((10, 6), dtype=np.float32)
    states[:, 5] = [100, 100, 100, 100, 100, 98, 92, 80, 40, 33]
    actions = np.zeros((10, 6), dtype=np.float32)
    actions[:, 5] = [100, 100, 100, 100, 94, 88, 82, 40, 32, 100]
    predicted = np.zeros((10, 2, 6), dtype=np.float32)
    predicted[:, :, 5] = 100
    predicted[1, 0, 5] = 89
    predicted[3, 0, 5] = 89
    predicted[8, 0, 5] = 70
    predicted[9, 0, 5] = 100
    target = actions[:, None, :]

    labels = derive_event_labels(episodes, frames, states, actions)
    rows = make_episode_audit_rows(
        episodes,
        frames,
        states,
        target,
        predicted,
        np.zeros((10, 2), dtype=bool),
        target_index=0,
        executed_steps=1,
        labels=labels,
    )

    assert labels["close_event_rows"].tolist() == [4]
    assert np.flatnonzero(labels["pure_open_negative"]).tolist() == [0, 1]
    assert labels["pre_lift_anchor_rows"].tolist() == [8]
    assert rows[0]["event_window_hit"] is True
    assert rows[0]["first_crossing_timing"] == "early"
    assert rows[0]["first_crossing_offset"] == -3
    assert rows[0]["pure_false_close_frames"] == 1
    assert rows[0]["pre_lift_gripper_closed"] is True
    assert rows[0]["release_crossing_hit"] == 1


def test_screening_gate_requires_timing_safety_release_prelift_and_body_baseline():
    episode_rows = [
        {
            "first_crossing_timing": "on_time" if episode < 5 else "late",
            "event_window_hit": True,
            "pure_false_close_frames": 0,
            "sustained_false_close_pairs": 0,
            "release_crossing_target": 2,
            "release_crossing_hit": 2 if episode < 5 else 1,
            "pre_lift_gripper_closed": True,
        }
        for episode in range(6)
    ]
    summary = {
        "policy": "act_050k",
        "checkpoint": "/tmp/050000",
        "body_prefix": {
            "body_mae_executed_prefix": 1.0,
            "body_p95_executed_prefix": 1.5,
        },
    }

    row = make_screening_row(
        summary,
        episode_rows,
        hold_body_mae=2.0,
        hold_body_p95=3.0,
        temporal_summary={"body_overlap_p95_deg": 4.0},
    )

    assert row["timing_gate"] is True
    assert row["pure_false_close_gate"] is True
    assert row["release_gate"] is True
    assert row["pre_lift_gate"] is True
    assert row["body_beats_hold_gate"] is True
    assert row["joint_gate"] is True
    assert row["gates_passed"] == 5


def test_find_pre_lift_anchor_rows_uses_first_tight_command_after_close_event():
    episodes = np.array([0, 0, 0, 0, 1, 1, 1])
    frames = np.array([0, 1, 2, 3, 0, 1, 2])
    actions = np.zeros((7, 6), dtype=np.float32)
    actions[:, 5] = [100, 94, 40, 32, 100, 94, 30]

    anchors = find_pre_lift_anchor_rows(
        episodes,
        frames,
        actions,
        close_event_rows=np.array([1, 5]),
    )

    assert anchors.tolist() == [3, 6]


def test_temporal_consistency_aligns_predictions_for_same_future_time():
    predicted = np.zeros((2, 3, 6), dtype=np.float32)
    predicted[0, :, :5] = np.array([0, 1, 2])[:, None]
    predicted[1, :, :5] = np.array([3, 2, 4])[:, None]
    predicted[0, :, 5] = [100, 89, 80]
    predicted[1, :, 5] = [91, 80, 70]

    rows = make_temporal_consistency_rows(
        predicted,
        np.zeros((2, 3), dtype=bool),
        episodes=np.array([4, 4]),
        frames=np.array([10, 11]),
        query_stride=1,
    )

    assert len(rows) == 1
    assert rows[0]["episode"] == 4
    assert rows[0]["from_frame"] == 10
    assert rows[0]["to_frame"] == 11
    assert rows[0]["overlap_steps"] == 2
    assert rows[0]["body_overlap_mae_deg"] == pytest.approx(1.0)
    assert rows[0]["gripper_overlap_mae_deg"] == pytest.approx(1.0)
    assert rows[0]["gripper_threshold_disagreement_rate"] == pytest.approx(0.5)
    assert rows[0]["absolute_close_frame_shift"] == 1


def test_temporal_consistency_summary_preserves_tail_and_cumulative_signal():
    summary = summarize_temporal_consistency_rows(
        [
            {
                "episode": 0,
                "body_overlap_mae_deg": 1.0,
                "gripper_overlap_mae_deg": 2.0,
                "gripper_threshold_disagreement_rate": 0.0,
                "close_available_disagreement": False,
                "absolute_close_frame_shift": 1,
            },
            {
                "episode": 0,
                "body_overlap_mae_deg": 3.0,
                "gripper_overlap_mae_deg": 6.0,
                "gripper_threshold_disagreement_rate": 0.5,
                "close_available_disagreement": True,
                "absolute_close_frame_shift": None,
            },
        ]
    )

    assert summary["transitions"] == 2
    assert summary["body_overlap_mae_deg"] == pytest.approx(2.0)
    assert summary["body_overlap_p95_deg"] == pytest.approx(2.9)
    assert summary["gripper_overlap_mae_deg"] == pytest.approx(4.0)
    assert summary["threshold_disagreement_rate"] == pytest.approx(0.25)
    assert summary["close_available_disagreement_count"] == 1
    assert summary["mean_absolute_close_frame_shift"] == pytest.approx(1.0)
    assert summary["episode_cumulative_body_mae"] == {"0": pytest.approx(4.0)}


def test_nearest_event_neighbors_separates_visual_and_body_matches():
    episodes = np.array([0, 1, 2])
    frames = np.array([5, 6, 7])
    front = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    side = front.copy()
    states = np.zeros((3, 6), dtype=np.float32)
    states[1, :5] = 10.0

    rows = nearest_event_neighbors(
        episodes,
        frames,
        states,
        front,
        side,
        event_rows=np.array([0, 1, 2]),
        heldout_episodes={0},
    )

    assert rows == [
        {
            "heldout_episode": 0,
            "heldout_frame": 5,
            "visual_neighbor_episode": 1,
            "visual_neighbor_frame": 6,
            "visual_cosine_distance": pytest.approx(0.0),
            "body_neighbor_episode": 2,
            "body_neighbor_frame": 7,
            "body_standardized_distance": pytest.approx(0.0),
        }
    ]


def test_body_prefix_mae_per_sample_uses_executed_non_pad_actions_only():
    target = np.zeros((2, 3, 6), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[0, 0, :5] = 1.0
    prediction[0, 1, :5] = 3.0
    prediction[1, 0, :5] = 2.0
    prediction[1, 1, :5] = 100.0
    action_is_pad = np.array(
        [
            [False, False, False],
            [False, True, False],
        ]
    )

    result = body_prefix_mae_per_sample(
        target,
        prediction,
        action_is_pad,
        executed_steps=2,
    )

    assert result.tolist() == pytest.approx([2.0, 2.0])


def test_summarize_body_prefix_maes_reports_all_tail_and_close_window():
    result = summarize_body_prefix_maes(
        np.array([1.0, 2.0, 10.0, np.nan]),
        np.array([True, False, True, False]),
    )

    assert result == pytest.approx(
        {
            "body_mae_executed_prefix": 13.0 / 3.0,
            "body_p95_executed_prefix": 9.2,
            "body_mae_close_onset_prefix": 5.5,
        }
    )


def test_event_metrics_distinguish_onset_from_closed_hold():
    state = np.array(
        [
            [0, 0, 0, 0, 0, 100],
            [0, 0, 0, 0, 0, 70],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [[0, 0, 0, 0, 0, 84]],
            [[0, 0, 0, 0, 0, 68]],
        ],
        dtype=np.float32,
    )
    prediction = np.array(
        [
            [[0, 0, 0, 0, 0, 88]],
            [[0, 0, 0, 0, 0, 69]],
        ],
        dtype=np.float32,
    )

    result = compute_event_metrics(
        state,
        target,
        prediction,
        target_index=0,
        executed_steps=1,
    )

    assert result["close_onset"] == 1
    assert result["close_direction_hit"] == 1
    assert result["close_crossing_target"] == 1
    assert result["close_crossing_hit"] == 1
    assert result["closed_hold"] == 1


def test_earliest_threshold_offset_reports_first_close_and_missing():
    grip = np.array(
        [
            [100, 97, 89, 70],
            [100, 99, 98, 97],
        ],
        dtype=np.float32,
    )

    offsets = earliest_threshold_offset(grip, threshold=90)

    assert offsets.tolist() == [2, -1]


def test_event_metrics_report_close_outside_executed_prefix():
    state = np.array([[0, 0, 0, 0, 0, 100]], dtype=np.float32)
    target = np.array([[[0, 0, 0, 0, 0, 84]]], dtype=np.float32)
    prediction = np.array(
        [[[0, 0, 0, 0, 0, 99], [0, 0, 0, 0, 0, 89]]],
        dtype=np.float32,
    )

    result = compute_event_metrics(
        state,
        target,
        prediction,
        target_index=0,
        executed_steps=1,
    )

    assert result["close_crossing_hit"] == 0
    assert result["close_in_chunk"] == 1
    assert result["close_in_executed_prefix"] == 0


def test_translation_zero_fills_without_wraparound():
    image = torch.arange(12).reshape(1, 1, 3, 4)

    shifted = translate_images(image, dx=2, dy=0)

    assert torch.equal(shifted[..., :2], torch.zeros_like(shifted[..., :2]))
    assert torch.equal(shifted[..., 2:], image[..., :2])


def test_translation_supports_negative_vertical_shift():
    image = torch.arange(12).reshape(1, 1, 3, 4)

    shifted = translate_images(image, dx=0, dy=-1)

    assert torch.equal(shifted[..., -1, :], torch.zeros_like(shifted[..., -1, :]))
    assert torch.equal(shifted[..., :2, :], image[..., 1:, :])


def test_set_current_gripper_changes_only_newest_observation():
    state = torch.zeros(1, 2, 6)
    state[:, 0, 5] = 77.0

    changed = set_current_gripper(state, 85.0)

    assert changed[0, 0, 5] == 77.0
    assert changed[0, 1, 5] == 85.0
    assert state[0, 1, 5] == 0.0


@pytest.mark.parametrize(("n_obs_steps", "expected"), [(1, 0), (2, 1)])
def test_current_target_index_matches_observation_history(n_obs_steps, expected):
    assert current_target_index(n_obs_steps) == expected


def test_event_rows_preserve_episode_frame_phase_and_execution_offset():
    state = np.array([[0, 0, 0, 0, 0, 100]], dtype=np.float32)
    target = np.array([[[1, 2, 3, 4, 5, 84]]], dtype=np.float32)
    prediction = np.array(
        [[[1, 2, 3, 4, 5, 99], [1, 2, 3, 4, 5, 89]]],
        dtype=np.float32,
    )

    rows = make_event_rows(
        state,
        target,
        prediction,
        episodes=np.array([5]),
        frames=np.array([64]),
        target_index=0,
        executed_steps=1,
    )

    assert rows == [
        {
            "episode": 5,
            "frame": 64,
            "phase": "close_onset",
            "state_grip": 100.0,
            "target_grip": 84.0,
            "predicted_grip_h1": 99.0,
            "d_t": 1,
            "within_executed_prefix": False,
            "body_mae_h1": 0.0,
        }
    ]


def test_predict_physical_chunk_uses_full_act_chunk_and_postprocessor():
    class Config:
        type = "act"
        n_action_steps = 1

    class Policy:
        config = Config()

        def predict_action_chunk(self, batch):
            return torch.tensor([[[1.0] * 6, [2.0] * 6]])

    chunk = predict_physical_chunk(
        Policy(),
        {"observation.state": torch.zeros(1, 6)},
        lambda action: action + 10,
        seed=7,
    )

    assert chunk.shape == (1, 2, 6)
    assert torch.equal(chunk[0, :, 0], torch.tensor([11.0, 12.0]))


def test_predict_physical_chunk_reconstructs_diffusion_execution_queue():
    class Config:
        type = "diffusion"
        n_action_steps = 2

    class Policy:
        config = Config()

        def __init__(self):
            self.index = 0

        def reset(self):
            self.index = 0

        def select_action(self, batch):
            value = float(self.index + 1)
            self.index += 1
            return torch.full((1, 6), value)

    chunk = predict_physical_chunk(
        Policy(),
        {"observation.state": torch.zeros(1, 6)},
        lambda action: action,
        seed=7,
    )

    assert chunk.shape == (1, 2, 6)
    assert torch.equal(chunk[0, :, 0], torch.tensor([1.0, 2.0]))


def test_predict_physical_chunk_uses_offline_diffusion_history_once():
    class Config:
        type = "diffusion"
        n_obs_steps = 2
        n_action_steps = 2
        image_features = {"observation.images.front": object(), "observation.images.side": object()}

    class Diffusion:
        def generate_actions(self, batch):
            assert batch["observation.state"].shape == (1, 2, 6)
            assert batch["observation.images"].shape == (1, 2, 2, 3, 4, 5)
            return torch.arange(4.0).reshape(1, 4, 1).expand(-1, -1, 6)

    class Policy:
        config = Config()
        diffusion = Diffusion()

    chunk = predict_physical_chunk(
        Policy(),
        {
            "observation.state": torch.zeros(1, 2, 6),
            "observation.images.front": torch.zeros(1, 2, 3, 4, 5),
            "observation.images.side": torch.zeros(1, 2, 3, 4, 5),
        },
        lambda action: action + 10,
        seed=7,
    )

    assert chunk.shape == (1, 2, 6)
    assert torch.equal(chunk[0, :, 0], torch.tensor([11.0, 12.0]))


def test_parse_named_assignment_preserves_path_colons():
    name, value = parse_named_assignment("diff90=/tmp/checkpoint:with-colon")

    assert name == "diff90"
    assert value == "/tmp/checkpoint:with-colon"


def test_summarize_event_counts_keeps_denominators_visible():
    summary = summarize_event_counts(
        {
            "close_onset": 4,
            "close_direction_hit": 3,
            "close_crossing_target": 2,
            "close_crossing_hit": 1,
            "open_phase": 5,
            "open_false_close": 1,
            "release_onset": 0,
            "release_direction_hit": 0,
            "release_crossing_target": 0,
            "release_crossing_hit": 0,
        }
    )

    assert summary["close_direction_recall"] == 0.75
    assert summary["close_crossing_recall"] == 0.5
    assert summary["open_false_close_rate"] == 0.2
    assert summary["release_direction_recall"] is None
