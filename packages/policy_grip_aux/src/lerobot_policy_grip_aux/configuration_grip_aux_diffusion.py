from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from .common import configure_privileged_features


@PreTrainedConfig.register_subclass("grip_aux_diffusion")
@dataclass
class GripAuxDiffusionConfig(DiffusionConfig):
    grip_aux_weight: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.grip_aux_weight <= 0:
            raise ValueError("grip_aux_weight must be positive")

    def set_dataset_feature_metadata(self, dataset_features: dict) -> None:
        configure_privileged_features(self, dataset_features)
