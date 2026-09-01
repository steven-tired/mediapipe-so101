import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_policy_grip_aux.common import (
    GRIP_CONTEXT_FEATURES,
    HUMAN_INTERVENTION_FEATURE,
    PV_TEACHER_FEATURE,
    PV_TEACHER_VALID_FEATURE,
    PV_TIMING_FEATURES,
)
from lerobot_policy_grip_aux.configuration_grip_aux_act import GripAuxACTConfig
from lerobot_policy_grip_aux.modeling_grip_aux_act import GripAuxACTPolicy
from lerobot_policy_grip_aux.configuration_grip_aux_diffusion import GripAuxDiffusionConfig
from lerobot_policy_grip_aux.modeling_grip_aux_diffusion import GripAuxDiffusionPolicy


def _features():
    return {
        OBS_STATE: {"dtype": "float32", "shape": (9,), "names": [
            *(f"joint{i}.pos" for i in range(6)),
            *GRIP_CONTEXT_FEATURES,
        ]},
        "observation.images.front": {"dtype": "video", "shape": (3, 64, 64)},
        PV_TEACHER_FEATURE: {"dtype": "float32", "shape": (1,)},
        PV_TEACHER_VALID_FEATURE: {"dtype": "float32", "shape": (1,)},
        HUMAN_INTERVENTION_FEATURE: {"dtype": "float32", "shape": (1,)},
        **{
            name: {"dtype": "float32", "shape": (1,)}
            for name in PV_TIMING_FEATURES
        },
        ACTION: {"dtype": "float32", "shape": (6,)},
    }


def _config():
    return GripAuxACTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (9,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            PV_TEACHER_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
            PV_TEACHER_VALID_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
            HUMAN_INTERVENTION_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
        chunk_size=2,
        n_action_steps=1,
        use_vae=False,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )


def test_config_keeps_privileged_labels_out_of_deploy_inputs():
    config = _config()
    config.set_dataset_feature_metadata(_features())
    assert OBS_STATE in config.input_features
    assert "observation.images.front" in config.input_features
    assert PV_TEACHER_FEATURE not in config.input_features
    assert PV_TEACHER_VALID_FEATURE not in config.input_features
    assert HUMAN_INTERVENTION_FEATURE not in config.input_features
    assert not set(PV_TIMING_FEATURES) & set(config.input_features)


def test_auxiliary_head_uses_both_frozen_camera_views():
    features = _features()
    features["observation.images.side"] = {
        "dtype": "video",
        "shape": (3, 64, 64),
    }
    config = _config()
    config.input_features["observation.images.side"] = PolicyFeature(
        FeatureType.VISUAL, (3, 64, 64)
    )
    config.set_dataset_feature_metadata(features)
    policy = GripAuxACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 9),
        "observation.images.front": torch.randn(2, 3, 64, 64),
        "observation.images.side": torch.randn(2, 3, 64, 64),
    }

    assert len(policy.grip_intent_head.extra_image_encoders) == 1
    assert policy.predict_grip_intent(batch).shape == (2, 1)


def test_act_forward_adds_masked_grip_auxiliary_loss():
    config = _config()
    config.set_dataset_feature_metadata(_features())
    policy = GripAuxACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 9),
        "observation.images.front": torch.randn(2, 3, 64, 64),
        ACTION: torch.randn(2, 2, 6),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        PV_TEACHER_FEATURE: torch.tensor([[0.25], [0.75]]),
        PV_TEACHER_VALID_FEATURE: torch.tensor([[1.0], [0.0]]),
        HUMAN_INTERVENTION_FEATURE: torch.zeros(2, 1),
    }
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["grip_aux_valid"] == 1
    assert metrics["grip_aux_loss"] >= 0.0
    prediction = policy.predict_grip_intent(batch)
    assert prediction.shape == (2, 1)
    assert torch.all((0.0 <= prediction) & (prediction <= 1.0))


def test_diffusion_forward_keeps_six_dimensional_action_and_adds_auxiliary_loss():
    config = GripAuxDiffusionConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (9,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            PV_TEACHER_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
            PV_TEACHER_VALID_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
            HUMAN_INTERVENTION_FEATURE: PolicyFeature(FeatureType.STATE, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (6,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        drop_n_last_frames=0,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        spatial_softmax_num_keypoints=4,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        diffusion_step_embed_dim=32,
        num_train_timesteps=4,
    )
    config.set_dataset_feature_metadata(_features())
    policy = GripAuxDiffusionPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 2, 9),
        "observation.images.front": torch.randn(2, 2, 3, 64, 64),
        ACTION: torch.randn(2, 8, 6),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        PV_TEACHER_FEATURE: torch.tensor([[[0.25], [0.5]], [[0.5], [0.75]]]),
        PV_TEACHER_VALID_FEATURE: torch.tensor([[[1.0], [1.0]], [[1.0], [0.0]]]),
        HUMAN_INTERVENTION_FEATURE: torch.zeros(2, 2, 1),
    }
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert policy.config.action_feature.shape == (6,)
    assert metrics["grip_aux_valid"] == 1
    assert metrics["grip_aux_loss"] >= 0.0


def test_grip_aux_checkpoint_round_trip_keeps_custom_head(tmp_path):
    config = _config()
    config.set_dataset_feature_metadata(_features())
    policy = GripAuxACTPolicy(config)
    policy.save_pretrained(tmp_path)

    loaded_config = PreTrainedConfig.from_pretrained(tmp_path)
    loaded = get_policy_class(loaded_config.type).from_pretrained(
        tmp_path,
        config=loaded_config,
    )
    assert loaded_config.type == "grip_aux_act"
    assert hasattr(loaded, "grip_intent_head")
    expected = policy.grip_intent_head.regressor[-1].weight
    actual = loaded.grip_intent_head.regressor[-1].weight
    assert torch.equal(expected, actual)
