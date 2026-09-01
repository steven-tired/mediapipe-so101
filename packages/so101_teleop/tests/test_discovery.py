import lerobot_teleoperator_so101_webcam  # noqa: F401  (registers the choice)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


def test_so101_webcam_is_registered():
    # draccus ChoiceRegistry knows the type once the package is imported
    assert "so101_webcam" in TeleoperatorConfig.get_known_choices()


def test_factory_builds_so101_webcam():
    from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
    from lerobot_teleoperator_so101_webcam.so101_webcam import SO101Webcam

    teleop = make_teleoperator_from_config(SO101WebcamConfig())
    assert isinstance(teleop, SO101Webcam)
    assert teleop.name == "so101_webcam"
