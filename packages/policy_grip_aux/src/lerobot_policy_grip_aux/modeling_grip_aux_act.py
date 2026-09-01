from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy

from .common import build_grip_intent_head, grip_aux_loss, predict_grip_intent
from .configuration_grip_aux_act import GripAuxACTConfig


class GripAuxACTPolicy(ACTPolicy):
    config_class = GripAuxACTConfig
    name = "grip_aux_act"

    def __init__(self, config: GripAuxACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.grip_intent_head = build_grip_intent_head(config)

    def forward(self, batch: dict[str, Tensor]):
        base_loss, metrics = super().forward(batch)
        grip = grip_aux_loss(self, batch)
        total = base_loss + self.config.grip_aux_weight * grip.loss
        return total, {
            **metrics,
            "action_loss": base_loss.item(),
            "grip_aux_loss": grip.loss.item(),
            "grip_aux_mae": grip.mae,
            "grip_aux_valid": grip.valid_count,
        }

    def predict_grip_intent(self, batch: dict[str, Tensor]) -> Tensor:
        return predict_grip_intent(self, batch)
