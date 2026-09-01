from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import OBS_STATE

PV_TEACHER_FEATURE = "observation.grip_intent_teacher"
PV_TEACHER_VALID_FEATURE = "observation.grip_intent_valid"
HUMAN_INTERVENTION_FEATURE = "observation.human_intervention"
PV_TIMING_FEATURES = (
    "observation.grip_intent_source_timestamp_s",
    "observation.grip_intent_sent_timestamp_s",
    "observation.grip_intent_received_timestamp_s",
    "observation.grip_intent_frame_age_s",
    "observation.grip_intent_sequence",
)
PV_SUPERVISION_FEATURES = (
    PV_TEACHER_FEATURE,
    PV_TEACHER_VALID_FEATURE,
    HUMAN_INTERVENTION_FEATURE,
)
PRIVILEGED_FEATURES = (
    *PV_SUPERVISION_FEATURES,
    *PV_TIMING_FEATURES,
)
GRIP_CONTEXT_FEATURES = (
    "grip_context.soft",
    "grip_context.hard",
    "grip_context.unknown",
)


def configure_privileged_features(config, dataset_features: dict) -> None:
    missing = [key for key in PV_SUPERVISION_FEATURES if key not in dataset_features]
    if missing:
        raise ValueError(f"grip-aux policy requires dataset features: {missing}")
    state_names = tuple(dataset_features.get(OBS_STATE, {}).get("names", ()))
    missing_context = [name for name in GRIP_CONTEXT_FEATURES if name not in state_names]
    if missing_context:
        raise ValueError(f"observation.state is missing grip context fields: {missing_context}")
    config.input_features = {
        key: feature
        for key, feature in (config.input_features or {}).items()
        if key not in PRIVILEGED_FEATURES
    }


def _last_step(value: Tensor) -> Tensor:
    if value.ndim in (3, 5):
        return value[:, -1]
    return value


class GripIntentHead(nn.Module):
    """Small deployable head over all policy RGB views plus state/grip_context."""

    def __init__(self, state_dim: int, image_channels: tuple[int, ...]):
        super().__init__()
        if not image_channels:
            raise ValueError("grip intent head requires at least one image feature")

        def image_encoder(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=5, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )

        # Keep the first encoder's historical state-dict name so existing
        # one-camera checkpoints still load unchanged.
        self.image_encoder = image_encoder(image_channels[0])
        self.extra_image_encoders = nn.ModuleList(
            image_encoder(channels) for channels in image_channels[1:]
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * len(image_channels) + state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, images: list[Tensor], state: Tensor) -> Tensor:
        encoders = (self.image_encoder, *self.extra_image_encoders)
        if len(images) != len(encoders):
            raise ValueError("grip intent image count does not match configured image features")
        state = _last_step(state)
        encoded = [
            encoder(_last_step(image)).flatten(1)
            for encoder, image in zip(encoders, images, strict=True)
        ]
        return torch.sigmoid(self.regressor(torch.cat((*encoded, state), dim=-1)))


@dataclass(frozen=True)
class GripAuxResult:
    loss: Tensor
    prediction: Tensor
    valid_count: int
    mae: float


def grip_aux_loss(policy, batch: dict[str, Tensor]) -> GripAuxResult:
    if PV_TEACHER_FEATURE not in batch or PV_TEACHER_VALID_FEATURE not in batch:
        raise KeyError("PV teacher target or valid mask is missing from the training batch")
    images = [batch[key] for key in policy.config.image_features]
    prediction = policy.grip_intent_head(images, batch[OBS_STATE])
    target = _last_step(batch[PV_TEACHER_FEATURE]).reshape_as(prediction).float()
    valid = _last_step(batch[PV_TEACHER_VALID_FEATURE]).reshape_as(prediction).float().clamp(0.0, 1.0)
    per_item = F.smooth_l1_loss(prediction, target, reduction="none")
    valid_count_tensor = valid.sum()
    loss = (per_item * valid).sum() / valid_count_tensor.clamp_min(1.0)
    mae = ((prediction.detach() - target).abs() * valid).sum() / valid_count_tensor.clamp_min(1.0)
    return GripAuxResult(
        loss=loss,
        prediction=prediction,
        valid_count=int(valid_count_tensor.detach().item()),
        mae=float(mae.detach().item()),
    )


def build_grip_intent_head(config) -> GripIntentHead:
    if config.robot_state_feature is None:
        raise ValueError("grip-aux policy requires observation.state")
    if not config.image_features:
        raise ValueError("grip-aux policy requires at least one image observation")
    image_channels = tuple(feature.shape[0] for feature in config.image_features.values())
    return GripIntentHead(config.robot_state_feature.shape[0], image_channels)


@torch.no_grad()
def predict_grip_intent(policy, batch: dict[str, Tensor]) -> Tensor:
    policy.eval()
    images = [batch[key] for key in policy.config.image_features]
    return policy.grip_intent_head(images, batch[OBS_STATE])
