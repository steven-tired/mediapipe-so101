from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot_teleoperator_so101_webcam.grip.runtime import (
    GRIP_CANDIDATE_FEATURES,
    GripCandidateHead,
    GripFeedbackConfig,
    GripFeedbackController,
    GripResidualHead,
    GripResidualShadow,
    append_grip_context,
    grip_visual_features,
    grip_context_vector,
    pv_teacher_label,
    select_grip_candidate,
    select_stability_effort_candidate,
)


def test_context_is_one_hot_and_only_appended_for_new_policy():
    base = np.arange(6, dtype=np.float32)
    assert grip_context_vector("soft").tolist() == [1.0, 0.0, 0.0]
    assert append_grip_context(base, context="hard", expected_dim=6).tolist() == base.tolist()
    assert append_grip_context(base, context="hard", expected_dim=9).tolist()[-3:] == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="expects observation.state dim"):
        append_grip_context(base, context="hard", expected_dim=8)


def test_feedback_controller_uses_action_as_gate_and_actual_position_as_feedback():
    controller = GripFeedbackController(GripFeedbackConfig(light_pos=40.0, hard_pos=20.0, max_step=2.0))
    assert controller.update(policy_target=80.0, grip_intent=1.0, actual_pos=80.0) == 80.0
    assert controller.update(policy_target=20.0, grip_intent=0.5, actual_pos=50.0) == 48.0
    assert controller.grasp_latched
    assert controller.update(policy_target=20.0, grip_intent=0.5, actual_pos=31.0) == 30.0
    assert controller.update(policy_target=80.0, grip_intent=1.0, actual_pos=30.0) == 80.0
    assert not controller.grasp_latched


def test_force_grasp_allows_pv_takeover_of_a_failed_open_policy_command():
    controller = GripFeedbackController(GripFeedbackConfig(light_pos=40.0, hard_pos=20.0, max_step=5.0))
    command = controller.update(
        policy_target=80.0,
        grip_intent=1.0,
        actual_pos=45.0,
        force_grasp=True,
    )
    assert command == 40.0
    assert controller.grasp_latched


def test_residual_shadow_waits_for_history_and_only_returns_a_prediction(tmp_path):
    head = GripResidualHead(history_steps=2, hidden_dim=4)
    for parameter in head.parameters():
        parameter.data.zero_()
    head.output.bias.data[:] = torch.tensor([0.0, 0.0, 2.0, 1.0])
    checkpoint = tmp_path / "grip_residual.pt"
    torch.save(
        {
            "history_steps": 2,
            "hidden_dim": 4,
            "feature_mean": torch.zeros(6),
            "feature_std": torch.ones(6),
            "model_state_dict": head.state_dict(),
        },
        checkpoint,
    )
    shadow = GripResidualShadow.from_checkpoint(checkpoint)

    assert shadow.observe(
        policy_target=25.0,
        command_target=25.0,
        actual_pos=27.0,
        present_current=8.0,
        present_load=130.0,
        position_lag=2.0,
    ) is None
    prediction = shadow.observe(
        policy_target=24.0,
        command_target=24.0,
        actual_pos=26.0,
        present_current=9.0,
        present_load=140.0,
        position_lag=2.0,
    )

    assert prediction["direction"] == "loosen"
    assert prediction["delta_q_sign"] == 1
    assert prediction["grasp_stable_probability"] == pytest.approx(0.7310586)
    assert prediction["prediction_for"] == "next_control_step"


def test_candidate_head_scores_one_delta_per_context():
    head = GripCandidateHead(history_steps=2, hidden_dim=4)
    features = torch.zeros((3, 2, len(GRIP_CANDIDATE_FEATURES)))
    scores = head(features, torch.tensor([-0.2, 0.0, 0.2]))
    assert scores.shape == (3,)


def test_candidate_selection_holds_when_uncertain_or_before_lift():
    scores = {-0.2: 0.72, 0.0: 0.60, 0.2: 0.90}
    assert select_grip_candidate(
        scores,
        supported_deltas={-0.2, 0.0, 0.2},
        stable_lift_seen=False,
    ) == -0.2
    assert select_grip_candidate(
        scores,
        supported_deltas={-0.2, 0.0, 0.2},
        stable_lift_seen=True,
    ) == 0.2
    assert select_grip_candidate(
        {-0.2: 0.64, 0.0: 0.60, 0.2: 0.62},
        supported_deltas={-0.2, 0.0, 0.2},
        stable_lift_seen=True,
    ) == 0.0
    assert select_grip_candidate(
        scores,
        supported_deltas={-0.2, 0.0},
        stable_lift_seen=True,
    ) == -0.2


def test_stability_effort_selection_applies_strict_load_gate():
    probabilities = {0.0: 0.70, 0.2: 0.75}
    predicted_loads = {0.0: 100.0, 0.2: 80.0}
    assert select_stability_effort_candidate(
        probabilities,
        predicted_loads,
        present_load=61.0,
        minimum_probability=0.65,
        minimum_load_for_loosen=60.0,
    ) == 0.2
    assert select_stability_effort_candidate(
        probabilities,
        predicted_loads,
        present_load=60.0,
        minimum_probability=0.65,
        minimum_load_for_loosen=60.0,
    ) == 0.0


def test_visual_features_include_motion_and_shape_change():
    previous = np.zeros((120, 160, 3), dtype=np.uint8)
    current = previous.copy()
    current[70:100, 40:120] = 255
    features = grip_visual_features(
        previous_front_rgb=previous,
        current_front_rgb=current,
        previous_side_rgb=previous,
        current_side_rgb=current,
    )
    assert features.shape == (14,)
    assert np.isfinite(features).all()
    assert features[0] > 0.0


def test_pv_teacher_label_masks_inactive_and_bad_samples():
    good = SimpleNamespace(active=True, available=True, fresh=True, status="active", pressure_0_1=0.7)
    target, valid = pv_teacher_label(good)
    assert target.tolist() == pytest.approx([0.7])
    assert valid.tolist() == [1.0]
    target, valid = pv_teacher_label(SimpleNamespace(active=False, available=True, pressure_0_1=0.8))
    assert target.tolist() == [0.0]
    assert valid.tolist() == [0.0]
