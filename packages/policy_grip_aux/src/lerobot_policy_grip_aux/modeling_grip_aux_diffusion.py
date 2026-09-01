from itertools import chain

from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from .common import build_grip_intent_head, grip_aux_loss, predict_grip_intent
from .configuration_grip_aux_diffusion import GripAuxDiffusionConfig


class GripAuxDiffusionPolicy(DiffusionPolicy):
    config_class = GripAuxDiffusionConfig
    name = "grip_aux_diffusion"

    def __init__(self, config: GripAuxDiffusionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.grip_intent_head = build_grip_intent_head(config)

    def get_optim_params(self):
        return chain(self.diffusion.parameters(), self.grip_intent_head.parameters())

    def forward(self, batch: dict[str, Tensor]):
        base_loss, _ = super().forward(batch)
        grip = grip_aux_loss(self, batch)
        total = base_loss + self.config.grip_aux_weight * grip.loss
        return total, {
            "action_loss": base_loss.item(),
            "grip_aux_loss": grip.loss.item(),
            "grip_aux_mae": grip.mae,
            "grip_aux_valid": grip.valid_count,
        }

    def predict_grip_intent(self, batch: dict[str, Tensor]) -> Tensor:
        return predict_grip_intent(self, batch)
