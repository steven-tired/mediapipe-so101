import subprocess
import sys

import numpy as np

from training.phase_c_grasp_ready.grasp_ready_probe import (
    LabelSet,
    _fit_final_head,
    assemble_features,
    build_episode_folds,
    evaluate_scores,
    fit_linear_head,
    find_close_events,
    load_cache,
    make_labels,
    save_cache,
    zero_false_threshold,
)


def test_close_events_and_ready_bands_stay_inside_episode():
    episodes = np.array([0] * 8 + [1] * 5)
    frames = np.array(list(range(8)) + list(range(5)))
    actions = np.zeros((13, 6), dtype=np.float32)
    states = np.zeros((13, 6), dtype=np.float32)
    actions[:, 5] = [100, 100, 94, 88, 100, 94, 88, 32, 100, 100, 94, 88, 32]
    states[:, 5] = [100, 100, 100, 94, 88, 100, 94, 88, 100, 100, 100, 94, 88]

    assert find_close_events(actions[:8, 5]) == [2, 5]

    labels = make_labels(episodes, frames, states, actions)

    assert labels.event_rows.tolist() == [2, 5, 10]
    assert np.flatnonzero(labels.ready_positive).tolist() == list(range(13))
    assert not labels.open_negative.any()


def test_feature_assembly_excludes_gripper_and_clamps_history_per_episode():
    cache = {
        "episode": np.array([0, 0, 0, 1, 1]),
        "frame": np.array([0, 1, 2, 0, 1]),
        "front": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32),
        "side": np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=np.float32),
        "state": np.array(
            [
                [1, 2, 3, 4, 5, 100],
                [2, 3, 4, 5, 6, 90],
                [3, 4, 5, 6, 7, 80],
                [4, 5, 6, 7, 8, -1000],
                [5, 6, 7, 8, 9, 1000],
            ],
            dtype=np.float32,
        ),
    }

    features = assemble_features(cache, history_offset=2)
    changed = {key: value.copy() for key, value in cache.items()}
    changed["state"][:, 5] += 12345
    changed_features = assemble_features(changed, history_offset=2)

    assert features["static"].shape == (5, 9)
    assert features["history"].shape == (5, 18)
    np.testing.assert_array_equal(features["static"], changed_features["static"])
    np.testing.assert_array_equal(features["history"], changed_features["history"])
    np.testing.assert_array_equal(features["history"][2, 4:8], [1, 2, 11, 12])
    np.testing.assert_array_equal(features["history"][3, 4:8], [7, 8, 17, 18])
    np.testing.assert_array_equal(features["history"][2, -5:], [2, 2, 2, 2, 2])
    np.testing.assert_array_equal(features["history"][3, -5:], [0, 0, 0, 0, 0])


def test_linear_head_is_deterministic_and_separates_literal_training_rows():
    features = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.float32)

    first = fit_linear_head(features, labels, seed=7, epochs=200)
    second = fit_linear_head(features, labels, seed=7, epochs=200)
    scores = first.predict_proba(features)

    np.testing.assert_array_equal(first.mean, [0, 0])
    np.testing.assert_allclose(first.scale, [np.sqrt(2.5), 1], rtol=1e-6)
    np.testing.assert_allclose(first.weight, second.weight, rtol=0, atol=0)
    np.testing.assert_allclose(first.bias, second.bias, rtol=0, atol=0)
    assert np.isfinite(scores).all()
    assert scores[:2].max() < scores[2:].min()


def test_zero_false_threshold_is_strictly_above_largest_negative_score():
    scores = np.array([0.1, 0.7, 0.7, 0.9], dtype=np.float64)
    negative = np.array([True, True, False, False])

    threshold = zero_false_threshold(scores, negative)

    assert threshold > 0.7
    assert not (scores[2] >= threshold)
    assert scores[3] >= threshold


def test_event_metrics_use_initial_ready_window_and_detect_sustained_false_trigger():
    episodes = np.array([0] * 6 + [1] * 6)
    frames = np.array(list(range(6)) * 2)
    labels = LabelSet(
        ready_positive=np.array(
            [False, True, True, True, True, True, False, True, True, True, True, True]
        ),
        open_negative=np.array(
            [True, False, False, False, False, False, True, False, False, False, False, False]
        ),
        event_rows=np.array([3, 9]),
    )
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.2, 0.1, 0.3, 0.1, 0.95, 0.7, 0.2, 0.1])

    legacy_open_phase = np.array(
        [True, False, True, False, False, False, True, False, False, False, False, False]
    )
    metrics = evaluate_scores(
        episodes,
        frames,
        labels,
        scores,
        threshold=0.8,
        legacy_open_phase=legacy_open_phase,
    )

    assert metrics["initial_event_hits"] == 2
    assert metrics["initial_events"] == 2
    assert metrics["first_hit_offsets"] == [-1, -1]
    assert metrics["median_first_hit_offset"] == -1.0
    assert metrics["false_trigger_frames"] == 0
    assert metrics["sustained_false_trigger_pairs"] == 0
    assert metrics["legacy_open_phase_rows"] == 3
    assert metrics["legacy_false_trigger_frames"] == 1

    false_labels = LabelSet(
        ready_positive=np.zeros(4, dtype=bool),
        open_negative=np.ones(4, dtype=bool),
        event_rows=np.array([], dtype=np.int64),
    )
    false_metrics = evaluate_scores(
        np.array([0, 0, 0, 0]),
        np.array([0, 1, 2, 3]),
        false_labels,
        np.array([0.9, 0.95, 0.1, 0.9]),
        threshold=0.8,
    )
    assert false_metrics["false_trigger_frames"] == 3
    assert false_metrics["sustained_false_trigger_pairs"] == 1


def test_feature_cache_round_trip_preserves_rows_shapes_and_dtypes(tmp_path):
    cache = {
        "episode": np.array([2, 2], dtype=np.int64),
        "frame": np.array([0, 1], dtype=np.int64),
        "state": np.arange(12, dtype=np.float32).reshape(2, 6),
        "action": np.arange(12, 24, dtype=np.float32).reshape(2, 6),
        "front": np.arange(8, dtype=np.float32).reshape(2, 4),
        "side": np.arange(8, 16, dtype=np.float32).reshape(2, 4),
    }
    path = tmp_path / "features.npz"

    save_cache(path, cache)
    loaded = load_cache(path)

    assert loaded.keys() == cache.keys()
    for key in cache:
        np.testing.assert_array_equal(loaded[key], cache[key])
        assert loaded[key].dtype == cache[key].dtype


def test_episode_folds_hold_out_each_episode_without_row_leakage():
    episodes = np.array([2, 2, 5, 5, 5, 9])

    folds = build_episode_folds(episodes)

    assert [episode for episode, _, _ in folds] == [2, 5, 9]
    validation_rows = []
    for episode, train_rows, validation in folds:
        assert not np.any(episodes[train_rows] == episode)
        assert np.all(episodes[validation] == episode)
        validation_rows.extend(validation.tolist())
    assert sorted(validation_rows) == list(range(6))


def test_evaluate_cli_reaches_dataset_contract_check(tmp_path):
    cache_path = tmp_path / "tiny.npz"
    output_dir = tmp_path / "out"
    save_cache(
        cache_path,
        {
            "episode": np.array([0, 0], dtype=np.int64),
            "frame": np.array([0, 1], dtype=np.int64),
            "state": np.zeros((2, 6), dtype=np.float32),
            "action": np.zeros((2, 6), dtype=np.float32),
            "front": np.zeros((2, 4), dtype=np.float32),
            "side": np.zeros((2, 4), dtype=np.float32),
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            "training/phase_c_grasp_ready/grasp_ready_probe.py",
            "evaluate",
            "--cache-path",
            str(cache_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd="/home/zhuokai/hand-teleop",
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "dataset contract mismatch" in completed.stderr
    assert "NameError" not in completed.stderr


def test_final_head_accepts_only_training_subset_index_space():
    train_features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    train_labels = LabelSet(
        ready_positive=np.array([False, False, True, True]),
        open_negative=np.array([True, True, False, False]),
        event_rows=np.array([2]),
    )

    head = _fit_final_head(train_features, train_labels, seed=3, epochs=20)

    assert np.isfinite(head.predict_proba(train_features)).all()
