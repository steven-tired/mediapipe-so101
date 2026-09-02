from collections import deque
import json

import pytest
import torch

from deploy_so101_grip_ee import (
    DeploymentEvidence,
    GripInterventionController,
    PolicyChunkTrace,
    apply_gripper_close_offset,
    hold_body_action,
    read_present_positions,
)


def test_grip_intervention_uses_point_two_steps_and_preserves_act_release():
    intervention = GripInterventionController(step=0.2)

    command, label = intervention.update(policy_target=24.0, actual_pos=26.0)
    assert command == 24.0
    assert not label["label_valid"]

    intervention.request_toggle()
    command, label = intervention.update(policy_target=23.0, actual_pos=26.0)
    assert command == 24.0
    assert label["direction"] == "hold"
    assert label["label_valid"]
    assert label["paused"]
    assert intervention.paused

    intervention.request_steps(-1)
    command, label = intervention.update(policy_target=22.0, actual_pos=25.8)
    assert command == pytest.approx(23.8)
    assert label["direction"] == "tighten"
    assert label["delta_q"] == pytest.approx(-0.2)

    intervention.request_steps(1)
    command, label = intervention.update(policy_target=21.0, actual_pos=25.8)
    assert command == pytest.approx(24.0)
    assert label["direction"] == "loosen"
    assert label["delta_q"] == pytest.approx(0.2)

    intervention.request_toggle()
    command, label = intervention.update(policy_target=21.0, actual_pos=25.8)
    assert command == pytest.approx(24.0)
    assert label["paused"]
    assert label["resume_after_cycle"]
    assert intervention.finish_cycle()
    assert intervention.active
    assert not intervention.paused

    command, label = intervention.update(policy_target=90.0, actual_pos=26.0)
    assert command == 90.0
    assert label["direction"] == "release"
    assert not intervention.active


def test_policy_chunk_trace_records_queue_chunk_and_execution_index():
    class Config:
        type = "act"
        n_action_steps = 3
        temporal_ensemble_coeff = None

    class Policy:
        config = Config()

        def __init__(self):
            self._action_queue = deque()

        def select_action(self, batch):
            if not self._action_queue:
                self._action_queue.extend(
                    [torch.full((1, 6), value) for value in (1.0, 2.0, 3.0)]
                )
            return self._action_queue.popleft()

    trace = PolicyChunkTrace(Policy(), lambda action: action + 10.0)

    first_action, first = trace.select({"observation.state": torch.zeros(1, 6)})
    second_action, second = trace.select({"observation.state": torch.zeros(1, 6)})

    assert torch.equal(first_action, torch.full((1, 6), 11.0))
    assert first["chunk_id"] == 0
    assert first["execution_index"] == 0
    assert torch.equal(first["raw_normalized_chunk"][0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(first["denormalized_chunk"][0, :, 0], torch.tensor([11.0, 12.0, 13.0]))
    assert torch.equal(second_action, torch.full((1, 6), 12.0))
    assert second["chunk_id"] == 0
    assert second["execution_index"] == 1


def test_deployment_evidence_records_full_action_chain(tmp_path):
    evidence = DeploymentEvidence(
        tmp_path / "trace",
        policy="checkpoint",
        task="task",
        arm_enabled=True,
        start_mode="current",
        policy_action_steps=2,
        motors=["shoulder", "gripper"],
        image_features=set(),
        gripper_close_offset=2.0,
        gripper_telemetry_hz=5.0,
        grip_residual_shadow_model="grip_residual.pt",
        grip_intervention_step=0.2,
        action_step_repeat=2,
    )
    evidence.add(
        step=0,
        elapsed_s=0.1,
        inference_ms=2.0,
        observation={},
        state=torch.tensor([1.0, 100.0]).numpy(),
        predicted_action=torch.tensor([2.0, 90.0]).numpy(),
        planned_action=torch.tensor([2.0, 85.0]).numpy(),
        action_trace={
            "chunk_id": 3,
            "execution_index": 1,
            "raw_normalized_action": torch.tensor([[0.2, -0.1]]),
            "raw_normalized_chunk": torch.tensor([[[0.1, 0.0], [0.2, -0.1]]]),
            "denormalized_chunk": torch.tensor([[[1.0, 100.0], [2.0, 90.0]]]),
        },
        bus_action={"shoulder.pos": 2.0, "gripper.pos": 86.0},
        readback_state={"shoulder.pos": 1.5, "gripper.pos": 95.0},
        gripper_telemetry={
            "sample_elapsed_s": 0.09,
            "present_current": 12,
            "present_load": 34,
            "position_lag": 9.0,
            "absolute_position_lag": 9.0,
        },
        grip_residual_shadow={
            "direction": "hold",
            "delta_q_sign": 0,
            "grasp_stable_probability": 0.75,
            "prediction_for": "next_control_step",
        },
        grip_intervention={
            "active": True,
            "direction": "loosen",
            "delta_q": 0.2,
            "label_valid": True,
        },
        command_sent=True,
    )
    evidence.close(status="complete", elapsed_s=0.1, steps=1, commands_sent=1)

    row = json.loads((tmp_path / "trace" / "control.jsonl").read_text())
    assert row["policy_chunk_id"] == 3
    assert row["policy_execution_index"] == 1
    assert row["raw_normalized_chunk"][1]["gripper"] == pytest.approx(-0.1)
    assert row["denormalized_chunk"][1]["gripper"] == 90.0
    assert row["planned_action"]["gripper"] == 85.0
    assert row["bus_target"]["gripper"] == 86.0
    assert row["readback_state"]["gripper"] == 95.0
    assert row["gripper_telemetry"]["present_current"] == 12
    assert row["gripper_telemetry"]["absolute_position_lag"] == 9.0
    assert row["grip_residual_shadow"]["direction"] == "hold"
    assert row["grip_intervention"]["delta_q"] == 0.2
    assert row["planned_action"]["gripper"] == 85.0
    manifest = json.loads((tmp_path / "trace" / "manifest.json").read_text())
    assert manifest["gripper_close_offset"] == 2.0
    assert manifest["gripper_telemetry_hz"] == 5.0
    assert manifest["grip_residual_shadow_model"] == "grip_residual.pt"
    assert manifest["grip_intervention_step"] == 0.2
    assert manifest["action_step_repeat"] == 2


def test_hold_body_action_preserves_only_policy_gripper():
    action = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).numpy()
    state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).numpy()

    held = hold_body_action(action, state, gripper_index=5)

    assert held.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 60.0]


def test_gripper_close_offset_changes_only_close_actions():
    close_action = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 24.0]).numpy()
    open_action = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 95.0]).numpy()

    adjusted_close = apply_gripper_close_offset(close_action, gripper_index=5, offset=2.0)
    adjusted_open = apply_gripper_close_offset(open_action, gripper_index=5, offset=2.0)

    assert adjusted_close.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 26.0]
    assert adjusted_open.tolist() == open_action.tolist()
    assert close_action[-1] == 24.0


def test_read_present_positions_retries_one_transient_bus_failure():
    class Bus:
        def __init__(self):
            self.calls = 0

        def sync_read(self, register):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient missing status packet")
            return {"shoulder": 1.5, "gripper": 95.0}

    bus = Bus()

    positions = read_present_positions(bus, tries=3, retry_delay_s=0.0)

    assert positions == {"shoulder": 1.5, "gripper": 95.0}
    assert bus.calls == 2
