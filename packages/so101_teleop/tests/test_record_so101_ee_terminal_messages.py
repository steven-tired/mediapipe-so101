from pathlib import Path


RECORDER = (Path(__file__).parents[1] / "src" / "lerobot_teleoperator_so101_webcam"
            / "programs" / "record_so101_ee.py")


def test_recorder_uses_terminal_messages_instead_of_text_to_speech():
    source = RECORDER.read_text()

    assert "log_say" not in source
    assert 'print("Stop recording")' in source
