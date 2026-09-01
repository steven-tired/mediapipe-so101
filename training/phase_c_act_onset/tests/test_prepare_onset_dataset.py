import numpy as np

from training.phase_c_act_onset.prepare_onset_dataset import (
    replace_selected_gripper,
    select_preonset_frames,
)


def test_select_preonset_frames_uses_future_chunk_and_open_state():
    states = np.zeros((4, 6), dtype=np.float32)
    states[:, 5] = [100, 100, 100, 89]
    actions = np.zeros((4, 6), dtype=np.float32)
    actions[:, 5] = [100, 100, 80, 80]

    selected = select_preonset_frames(states, actions, chunk_size=3, threshold=90)

    assert selected.tolist() == [True, True, True, False]


def test_replace_selected_gripper_cycles_values_and_preserves_body():
    states = np.arange(30, dtype=np.float32).reshape(5, 6)
    selected = np.array([True, False, True, True, False])

    replaced, counts = replace_selected_gripper(states, selected, values=(90.0, 95.0))

    assert replaced[:, 5].tolist() == [90.0, 11.0, 95.0, 90.0, 29.0]
    assert np.array_equal(replaced[:, :5], states[:, :5])
    assert counts == {"90": 2, "95": 1}
