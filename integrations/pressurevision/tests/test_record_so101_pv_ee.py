import argparse
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from pressurevision_integration.protocol import PressureReading
from webcam_input.types import LandmarksData, WebcamSample, WristData

recorder = importlib.import_module("record_so101_pv_ee")
review = importlib.import_module("pressurevision_integration.pv_episode_review")


def _levels(tmp_path: Path) -> Path:
    path = tmp_path / "levels.json"
    path.write_text(
        json.dumps(
            {
                "n_levels": 3,
                "levels": [1, 2],
                "metric": "sum_kpa",
                "continuous_anchors": [
                    {"sum_kpa": 1.0, "pressure_0_1": 0.0},
                    {"sum_kpa": 2.0, "pressure_0_1": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, *extra: str):
    return recorder.parse_args(
        [
            "--levels",
            str(_levels(tmp_path)),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--check-config",
            "--max-load", "240",
            "--max-current", "35",
            "--max-position-lag", "5.0",
            *extra,
        ]
    )


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "block.profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "object_id": "block",
                "arm_id": recorder.ARM_ID,
                "open_pos": 60,
                "light_pos": 30,
                "hard_pos": 20,
                "control_mode": "hard_profile",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_soft_cli_accepts_config_without_devices(tmp_path):
    args = _args(tmp_path, "--pv-mapping", "soft_direct")
    checked = recorder.validate_config(args)
    assert checked["profile"] is None
    assert args.fps == 10
    assert args.max_level_age_minutes == pytest.approx(180.0)


def test_carton_recorder_defaults_to_no_gripper_closure_limiter(tmp_path):
    args = recorder.parse_args(
        [
            "--levels",
            str(_levels(tmp_path)),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--check-config",
        ]
    )

    assert recorder.resolve_gripper_closure_limits(args) is None


def test_carton_recorder_rejects_partial_gripper_closure_limits(tmp_path):
    with pytest.raises(SystemExit):
        recorder.parse_args(
            [
                "--levels",
                str(_levels(tmp_path)),
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--evidence-dir",
                str(tmp_path / "evidence"),
                "--check-config",
                "--max-load",
                "240",
            ]
        )


def test_hard_cli_requires_profile(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--pv-mapping", "hard_profile")


def test_soft_cli_rejects_profile(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--pv-mapping", "soft_direct", "--object-profile", str(_profile(tmp_path)))


def test_levels_are_required(tmp_path):
    with pytest.raises(SystemExit):
        recorder.parse_args(["--check-config", "--dataset-root", str(tmp_path / "d")])


def test_hard_cli_loads_and_hashes_profile(tmp_path):
    args = _args(tmp_path, "--pv-mapping", "hard_profile", "--object-profile", str(_profile(tmp_path)))
    checked = recorder.validate_config(args)
    assert checked["profile"].object_id == "block"
    assert len(checked["profile_sha256"]) == 64


def test_schema_adds_deployable_grip_context_and_privileged_pv_labels():
    class FakeRobot:
        action_features = {f"joint{i}.pos": float for i in range(6)}
        observation_features = {
            **action_features,
            **{name: float for name in recorder.GRIP_CONTEXT_FEATURES},
            "front": (480, 640, 3),
            "side": (480, 640, 3),
        }

    features = recorder.build_training_features(FakeRobot())
    assert set(features) == {
        "action",
        "observation.state",
        "observation.images.front",
        "observation.images.side",
        *recorder.PV_AUXILIARY_FEATURES,
    }
    assert len(features["action"]["names"]) == 6
    assert len(features["observation.state"]["names"]) == 9
    assert set(recorder.GRIP_CONTEXT_FEATURES).issubset(features["observation.state"]["names"])


def test_resume_schema_ignores_lerobot_default_features():
    expected = {
        "observation.state": {"names": ["joint.pos"]},
        "action": {"names": ["joint.pos"]},
    }
    actual = {**expected, **recorder.DEFAULT_FEATURES}

    recorder.validate_dataset_schema(actual, expected)


def test_resume_schema_still_rejects_nondefault_extra_features():
    expected = {
        "observation.state": {"names": ["joint.pos"]},
        "action": {"names": ["joint.pos"]},
    }
    actual = {**expected, **recorder.DEFAULT_FEATURES, "observation.extra": {}}

    with pytest.raises(ValueError, match="observation.extra"):
        recorder.validate_dataset_schema(actual, expected)


def test_grip_context_is_one_hot_and_auto_matches_mapping():
    assert recorder.resolve_grip_context("soft_direct") == "soft"
    assert recorder.resolve_grip_context("carton_span") == "soft"
    assert recorder.resolve_grip_context("hard_profile") == "hard"
    values = recorder.grip_context_observation("unknown")
    assert sum(values.values()) == 1.0
    assert values["grip_context.unknown"] == 1.0
    with pytest.raises(ValueError, match="conflicts"):
        recorder.resolve_grip_context("hard_profile", "soft")


def test_recorder_position_step_is_six_and_live_default_is_two():
    assert recorder.RECORDER_FPS == 10
    assert recorder.FRONT_CAMERA_FPS == 10
    assert recorder.SIDE_CAMERA_FPS == 30
    assert recorder.RECORDER_POSITION_PER_FRAME == 6.0
    assert recorder.LIVE_POSITION_PER_FRAME == 2.0


def test_formal_recorder_keeps_wrist_roll_opt_in_until_accuracy_check(tmp_path):
    args = _args(tmp_path)
    assert args.pv_mapping == "carton_span"
    assert args.wrist_roll_range_deg == 0.0
    assert args.wrist_roll_gain == 1.0


def test_recorder_layout_is_a_fixed_two_by_two_grid():
    panel = recorder.compose_recorder_panel(
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
        None,
        None,
        height=40,
        width=50,
    )
    assert panel.shape == (80, 100, 3)   # 2 rows x 2 cols of 40x50
    assert recorder.PANEL_LABELS == (
        "hand-track",
        "shared-memory PV",
        "front (Creative overhead)",
        "side (Etron)",
    )


def test_pv_stale_or_fault_blocks_episode():
    class Reading:
        available = False
        status = "pv_stale"

    assert recorder.pv_sample_invalid(Reading())


def test_valid_pv_baseline_does_not_block_episode():
    class Reading:
        available = True
        status = "baseline"

    assert not recorder.pv_sample_invalid(Reading())


def test_transient_missing_operator_preview_does_not_invalidate_episode():
    class Reading:
        available = True
        status = "baseline"
        fault_latched = False

    class Controller:
        cmd_state = {"gripper.pos": 42.0}
        last_pinch = 0.04

        def step(self, wrist, landmarks):
            return dict(self.cmd_state), "MOVING"

    # PV state now lives on the runtime, not the controller: the controller
    # owns arm motion, the runtime owns pressure.
    class Runtime:
        last_pressure = Reading()
        last_pressure_control = None
        adjustment_locked = False
        adjustment_anchor_target = None

    teleop = recorder.PVRecorderTeleop.__new__(recorder.PVRecorderTeleop)
    teleop.use_oak = True
    teleop.source = argparse.Namespace(
        oak_failed=False,
        latest=lambda: (argparse.Namespace(), argparse.Namespace()),
    )
    teleop.controller = Controller()
    teleop.pv = Runtime()
    teleop.motor_sampler = None
    teleop.pv_preview = argparse.Namespace(read=lambda: None)
    teleop.preview = False
    teleop.preview_video_out = None
    teleop.episode_active = True
    teleop.episode_valid = True
    teleop.invalid_reason = ""
    teleop.temperature_guard = recorder.TemperatureGuard()
    teleop.events = {"stop_recording": False, "exit_early": False}
    teleop.pv_fault_release_deadline_s = None

    assert teleop.get_action() == {"gripper.pos": 42.0}
    assert teleop.episode_valid
    assert not teleop.events["stop_recording"]
    assert teleop.pv_fault_release_deadline_s is None


@pytest.mark.parametrize("key", [ord("r"), ord("R"), 81])
def test_recording_restart_key_discards_current_attempt(key):
    teleop = recorder.PVRecorderTeleop.__new__(recorder.PVRecorderTeleop)
    teleop.events = {
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
    }

    teleop._handle_key(key)

    assert teleop.events == {
        "stop_recording": False,
        "exit_early": True,
        "rerecord_episode": True,
    }


def test_locked_pv_teacher_uses_effective_latched_value_while_raw_pad_is_released():
    # A real PressureReading, not a Namespace shaped to whatever the code asked
    # for. The provenance fields were read under four names the dataclass does
    # not have (`pv_sequence`, `pv_sent_at_s`, `pv_received_at_s`,
    # `thermal_observed_at_s`); a Namespace answers `getattr(..., None)` just as
    # happily as the real object does, so every recorded episode got five
    # constant-zero columns and this test stayed green.
    reading = PressureReading(
        pressure_0_1=0.0,
        active=False,
        quality=1.0,
        available=True,
        status="baseline",
        sequence=41,
        observed_at_s=100.0,
        sent_at_s=100.01,
        received_at_s=100.02,
    )
    teleop = recorder.PVRecorderTeleop.__new__(recorder.PVRecorderTeleop)
    teleop.pv = argparse.Namespace(
        last_pressure=reading,
        adjustment_teacher=0.5,
    )

    supervision = teleop.training_grip_supervision()

    assert supervision[recorder.PV_TEACHER_FEATURE] == pytest.approx([0.5])
    assert supervision[recorder.PV_TEACHER_VALID_FEATURE] == pytest.approx([1.0])
    # The provenance must reach the frame, not be defaulted away.
    assert supervision[recorder.PV_SEQUENCE_FEATURE] == pytest.approx([41])
    assert supervision[recorder.PV_SOURCE_TIMESTAMP_FEATURE] == pytest.approx([100.0])
    assert supervision[recorder.PV_SENT_TIMESTAMP_FEATURE] == pytest.approx([100.01])
    assert supervision[recorder.PV_RECEIVED_TIMESTAMP_FEATURE] == pytest.approx([100.02])


def test_rerecord_discards_buffer_before_building_review_artifacts():
    source = inspect.getsource(recorder.run_recording)
    rerecord = source.index('if events["rerecord_episode"] and teleop.episode_valid:')
    review = source.index("review_frames = []", rerecord)
    branch = source[rerecord:review]

    assert "dataset.clear_episode_buffer()" in branch
    assert "review_video=None" in branch
    assert "review_timeline=None" in branch


def test_recorder_hand_gate_uses_new_frames_and_sends_no_action(monkeypatch):
    # The real sample type: this gate is the arm lock, so a field it reads must
    # be a field the publisher actually sends.
    samples = [
        WebcamSample(
            preview_frame=None,
            wrist=WristData(
                position=np.zeros(3),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                fist_state="open",
                valid=True,
            ),
            landmarks=LandmarksData(landmarks=np.zeros((21, 3)), valid=True),
            observed_at_s=10.0 + index,
            frame_id=index,
        )
        for index in range(4)
    ]

    class Source:
        oak_failed = False

        def latest_sample(self):
            return samples.pop(0)

    class Robot:
        cameras = {}

        def send_action(self, action):
            raise AssertionError("startup gate sent an action")

    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)
    recorder.wait_for_continuous_hand_tracking(
        Source(),
        Robot(),
        argparse.Namespace(),
        preview=False,
    )


def test_recording_hand_gate_seeds_controller_from_current_pose_without_ready_motion():
    source = inspect.getsource(recorder.run_recording)
    assert (
        source.index("robot.connect(calibrate=False)")
        < source.index("wait_for_continuous_hand_tracking")
        < source.index("positions = _read_positions(robot)")
        < source.index("controller.build(ee_centre)")
        < source.index("controller.seed(positions)")
    )
    assert "_ramp_to(robot, controller.middle_pose)" not in source


def test_final_review_closes_teleop_before_session_finalization():
    source = inspect.getsource(recorder.run_recording)
    start = source.index("session_complete =")
    review_branch = source[start : source.index("keep = _choose_keep(", start)]

    assert "args.episodes == 1" in review_branch
    assert review_branch.index("teleop.close_preview()") < review_branch.index(
        "dataset.save_episode()"
    )
    assert review_branch.index("dataset.save_episode()") < review_branch.index(
        "teleop.disconnect()"
    )


def test_recorder_teleop_disconnect_is_idempotent_and_closes_window_first(monkeypatch):
    calls = []
    teleop = object.__new__(recorder.PVRecorderTeleop)
    teleop.preview = True
    teleop._preview_writer = None
    teleop._connected = True
    teleop.source = type("Source", (), {"stop": lambda self: calls.append("source")})()
    teleop.controller = type(
        "Controller", (), {"close": lambda self: calls.append("controller")}
    )()
    monkeypatch.setattr(recorder.cv2, "destroyAllWindows", lambda: calls.append("windows"))

    teleop.disconnect()
    teleop.disconnect()

    assert calls == ["windows", "source", "controller"]
    assert teleop.is_connected is False


def test_temperature_single_anomaly_is_recorded_but_not_stop():
    guard = recorder.TemperatureGuard()
    assert not guard.observe(56.0)
    assert len(guard.samples) == 1
    assert guard.high_streak == 1


def test_temperature_two_consecutive_samples_stop():
    guard = recorder.TemperatureGuard()
    assert not guard.observe(56.0)
    assert guard.observe(57.0)
    assert guard.should_stop


def test_temperature_normal_sample_breaks_consecutive_streak():
    guard = recorder.TemperatureGuard()
    guard.observe(56.0)
    guard.observe(54.0)
    assert guard.high_streak == 0
    assert not guard.should_stop


def test_temperature_guard_counts_distinct_telemetry_samples_only():
    guard = recorder.TemperatureGuard()
    assert not guard.observe(61.0, observed_at_s=10.0)
    assert guard.high_streak == 1
    assert not guard.observe(61.0, observed_at_s=10.0)
    assert not guard.last_observation_was_new
    assert guard.high_streak == 1
    assert guard.observe(61.0, observed_at_s=10.2)


def test_keep_session_does_not_keep_zero_episode_dataset():
    assert not recorder._choose_keep(argparse.Namespace(keep_session=True), recorded=0)


def test_completed_episode_is_kept_without_terminal_prompt(monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("completed recording must not prompt in the terminal"),
    )

    assert recorder._choose_keep(argparse.Namespace(keep_session=None), recorded=1)
    assert recorder._choose_keep(argparse.Namespace(keep_session=True), recorded=1)
    assert not recorder._choose_keep(argparse.Namespace(keep_session=False), recorded=1)


def test_oak_failure_invalidates_episode_before_using_stale_hand_sample():
    class Evidence:
        def control(self, **kwargs):
            return None

    class Controller:
        cmd_state = {"gripper.pos": 42.0}

        def step(self, wrist, landmarks):
            raise AssertionError("stale OAK sample was used")

    teleop = recorder.PVRecorderTeleop.__new__(recorder.PVRecorderTeleop)
    teleop.use_oak = True
    teleop.preview_video_out = None
    teleop.source = argparse.Namespace(oak_failed=True)
    teleop.controller = Controller()
    teleop.episode_active = True
    teleop.episode_valid = True
    teleop.invalid_reason = ""
    teleop.temperature_guard = recorder.TemperatureGuard()
    teleop.events = {"stop_recording": False, "exit_early": False}
    teleop.evidence = Evidence()

    assert teleop.get_action() == {"gripper.pos": 42.0}
    assert teleop.invalid_reason == "oak_failed"
    assert teleop.events["stop_recording"] and teleop.events["exit_early"]


def test_evidence_session_is_unique_and_does_not_overwrite(tmp_path):
    first = recorder.create_evidence_session(tmp_path / "evidence")
    second = recorder.create_evidence_session(tmp_path / "evidence")
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_prepare_evidence_rejects_nonempty_directory(tmp_path):
    path = tmp_path / "evidence"
    path.mkdir()
    (path / "existing.txt").write_text("keep", encoding="utf-8")
    config = argparse.Namespace(value=1)
    with pytest.raises(ValueError):
        recorder.prepare_evidence_session(path, config=config, hashes={})


def test_launcher_parent_artifacts_do_not_occupy_recorder_evidence_directory(tmp_path):
    session = tmp_path / "session-abc"
    session.mkdir()
    (session / "config-check.txt").write_text("ok\n", encoding="utf-8")
    (session / "pv_sender.csv").write_text("header\n", encoding="utf-8")
    recorder_path = session / "recorder"
    evidence = recorder.prepare_evidence_session(
        recorder_path,
        config=argparse.Namespace(value=1),
        hashes={},
    )
    evidence.close(status="test")
    assert (recorder_path / "manifest.json").is_file()


def test_discard_rolls_back_previous_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    backup = recorder.snapshot_dataset(root)
    (root / "new.txt").write_text("new", encoding="utf-8")
    recorder.dispose_dataset_session(root, backup, keep=False)
    assert (root / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (root / "new.txt").exists()


def test_keep_removes_only_session_backup(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    backup = recorder.snapshot_dataset(root)
    (root / "new.txt").write_text("new", encoding="utf-8")
    recorder.dispose_dataset_session(root, backup, keep=True)
    assert (root / "old.txt").exists() and (root / "new.txt").exists()
    assert backup is not None and not backup.exists()


def test_dataset_root_modes_do_not_resume_zero_episode_shell(tmp_path):
    root = tmp_path / "dataset"
    assert recorder.dataset_root_mode(root) == "create"

    (root / "meta").mkdir(parents=True)
    assert recorder.dataset_root_mode(root) == "reset_empty"

    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0, "total_tasks": 0}),
        encoding="utf-8",
    )
    assert recorder.dataset_root_mode(root) == "reset_empty"

    pending = root / "images" / "observation.images.front" / "episode-000000"
    pending.mkdir(parents=True)
    (pending / "frame-000000.png").write_bytes(b"unsaved frame")
    assert recorder.dataset_root_mode(root) == "reset_empty"

    (root / "meta" / "tasks.parquet").write_bytes(b"parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 1, "total_tasks": 1}),
        encoding="utf-8",
    )
    assert recorder.dataset_root_mode(root) == "resume"


def test_incomplete_nonempty_dataset_is_rejected_before_devices(tmp_path):
    args = _args(tmp_path)
    root = Path(args.dataset_root)
    (root / "data").mkdir(parents=True)
    (root / "data" / "partial.bin").write_bytes(b"do not remove")

    with pytest.raises(ValueError, match="incomplete local dataset"):
        recorder.validate_config(args)
    assert (root / "data" / "partial.bin").read_bytes() == b"do not remove"


def test_empty_dataset_reset_is_transactionally_recoverable(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = root / "meta" / "info.json"
    info.write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0, "total_tasks": 0}),
        encoding="utf-8",
    )

    backup = recorder.snapshot_dataset(root)
    recorder.reset_empty_dataset_root(root)
    assert not root.exists()
    recorder.dispose_dataset_session(root, backup, keep=False)
    assert json.loads(info.read_text(encoding="utf-8"))["total_episodes"] == 0


def test_recording_resets_empty_shell_before_robot_construction_and_restores_on_failure(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    root = Path(args.dataset_root)
    (root / "meta").mkdir(parents=True)
    info = root / "meta" / "info.json"
    info.write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0, "total_tasks": 0}),
        encoding="utf-8",
    )

    def fail_before_devices(*args, **kwargs):
        assert not root.exists()
        raise RuntimeError("construction probe")

    monkeypatch.setattr(recorder, "PVRecorderRobot", fail_before_devices)
    with pytest.raises(RuntimeError, match="construction probe"):
        recorder.run_recording(args)

    assert json.loads(info.read_text(encoding="utf-8"))["total_frames"] == 0
    assert not list(tmp_path.glob("dataset.session-*.bak"))


def test_check_config_does_not_construct_robot(tmp_path, monkeypatch):
    called = False

    def fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("device construction during config check")

    monkeypatch.setattr(recorder, "PVRecorderRobot", fail)
    args = _args(tmp_path)
    assert args.check_config
    assert recorder.main(
        [
            "--check-config",
            "--levels",
            str(_levels(tmp_path)),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--max-load", "240",
            "--max-current", "35",
            "--max-position-lag", "5.0",
        ]
    ) == 0
    assert not called


def test_stream_preflight_starts_oak_before_workspace_cameras_without_robot(
    tmp_path, monkeypatch, capsys
):
    events = []

    class Source:
        oak_failed = False

        def __init__(self, estimator):
            self.frame_id = 0

        def start_oak(self):
            events.append("oak_start")

        def latest_sample(self):
            self.frame_id += 1
            return argparse.Namespace(frame_id=self.frame_id)

        def stop(self):
            events.append("oak_stop")

    class Camera:
        def __init__(self, config):
            self.name = Path(config.index_or_path).name

        def connect(self):
            events.append(f"camera_connect:{self.name}")

        def read_latest(self, max_age_ms):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def disconnect(self):
            events.append(f"camera_disconnect:{self.name}")

    class Preview:
        def __init__(self, path):
            pass

        def read(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            events.append("preview_close")

    monkeypatch.setattr(recorder, "WebcamSource", Source)
    monkeypatch.setattr(recorder, "OpenCVCamera", Camera)
    monkeypatch.setattr(recorder, "PressureVisionPreviewSource", Preview)
    monkeypatch.setattr(
        recorder,
        "PVRecorderRobot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("robot constructed")),
    )
    args = _args(
        tmp_path,
        "--stream-preflight",
        "--front-camera", str(tmp_path / "front"),
        "--side-camera", str(tmp_path / "side"),
    )

    assert recorder.run_stream_preflight(args, duration_s=0.06) == 0
    assert events.index("oak_start") < events.index("camera_connect:front")
    assert events.index("oak_start") < events.index("camera_connect:side")
    assert json.loads(capsys.readouterr().out)["commands_sent"] == 0


def test_evidence_manifest_lists_training_fields(tmp_path):
    path = recorder.create_evidence_session(tmp_path / "evidence")
    config = argparse.Namespace(value=1)
    session = recorder.EvidenceSession(path, config=config, hashes={"levels": "abc"})
    session.close(status="rolled_back")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["deploy_observation_fields"] == [
        "observation.state",
        "observation.images.front",
        "observation.images.side",
    ]
    assert manifest["privileged_training_fields"] == list(recorder.PV_AUXILIARY_FEATURES)
    assert manifest["pv_sidecar_schema_version"] == "7"
    assert manifest["dataset_training_fields"] == [
        "observation.state",
        "observation.images.front",
        "observation.images.side",
        *recorder.PV_AUXILIARY_FEATURES,
        "action",
    ]


def test_evidence_session_appends_episode_outcomes(tmp_path):
    path = recorder.create_evidence_session(tmp_path / "evidence")
    session = recorder.EvidenceSession(
        path,
        config=argparse.Namespace(value=1),
        hashes={},
    )
    session.outcome({"attempt": 3, "outcome": "failure"})
    session.close(status="test")

    records = [
        json.loads(line)
        for line in (path / "episode_outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"attempt": 3, "outcome": "failure"}]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["episode_outcomes"] == "episode_outcomes.jsonl"


def test_episode_review_frames_waits_for_images_and_extracts_gripper_fields(tmp_path):
    class Writer:
        waited = False
        episode_buffer = {
            "size": 2,
            "timestamp": [0.0, 0.1],
            "observation.images.front": [tmp_path / "front0.png", tmp_path / "front1.png"],
            "observation.images.side": [tmp_path / "side0.png", tmp_path / "side1.png"],
            "action": [np.asarray([1, 32]), np.asarray([2, 29])],
            "observation.state": [np.asarray([3, 31]), np.asarray([4, 28])],
            recorder.PV_TEACHER_FEATURE: [np.asarray([0.2]), np.asarray([0.7])],
            recorder.PV_TEACHER_VALID_FEATURE: [np.asarray([1]), np.asarray([0])],
        }

        def _wait_image_writer(self):
            self.waited = True

    class Dataset:
        writer = Writer()
        features = {
            "action": {"names": ["shoulder.pos", "gripper.pos"]},
            "observation.state": {"names": ["shoulder.pos", "gripper.pos"]},
        }

    frames = recorder.episode_review_frames(Dataset())
    assert Dataset.writer.waited
    assert [frame.commanded_gripper_pos for frame in frames] == [32.0, 29.0]
    assert [frame.observed_gripper_pos for frame in frames] == [31.0, 28.0]
    assert [frame.pv_valid for frame in frames] == [True, False]


def _review_frames(tmp_path: Path):
    frames = []
    for index, q in enumerate((32.0, 32.0, 29.0)):
        front = tmp_path / f"front-{index}.png"
        side = tmp_path / f"side-{index}.png"
        assert recorder.cv2.imwrite(
            str(front), np.full((24, 32, 3), 20 + index, dtype=np.uint8)
        )
        assert recorder.cv2.imwrite(
            str(side), np.full((24, 32, 3), 40 + index, dtype=np.uint8)
        )
        frames.append(
            recorder.ReviewFrame(
                timestamp_s=index / 10,
                front_path=front,
                side_path=side,
                commanded_gripper_pos=q,
                observed_gripper_pos=q + 0.5,
                pv_teacher=index / 2,
                pv_valid=True,
            )
        )
    return frames


def test_review_artifacts_and_operator_recovered_decision(tmp_path):
    frames = _review_frames(tmp_path)
    video, timeline = recorder.write_review_artifacts(frames, tmp_path, attempt=7)
    keys = iter(
        (
            ord(" "),
            ord("j"),
            ord("."),
            ord("k"),
            ord("r"),
            ord("1"),
            ord("n"),
            ord("t"),
            ord("t"),
            13,
        )
    )

    decision = recorder.interactive_review(
        video,
        frames,
        key_source=lambda: next(keys),
        show=False,
    )

    assert video.is_file() and timeline.is_file()
    assert "0.200000,29.0,29.5,1.0,1,1" in timeline.read_text(encoding="utf-8")
    assert decision == {
        "outcome": "success_recovered_slip",
        "slip_index": 0,
        "stable_index": 2,
        "failure_reasons": set(),
        "table_contact": "brief",
        "residual_grade": 1,
        "functional_damage": False,
    }
    for frame in frames:
        frame.front_path.unlink()
        frame.side_path.unlink()
    capture = recorder.cv2.VideoCapture(str(video))
    ok, image = capture.read()
    capture.release()
    assert ok and image is not None


def test_manual_review_labels_have_semantic_colors():
    tokens = dict(
        review.review_status_tokens(
            outcome=review.OUTCOME_RECOVERED,
            residual_grade=1,
            functional_damage=False,
            table_contact="brief",
            slip_index=10,
            stable_index=20,
            failure_reasons=set(),
        )
    )

    assert tokens["outcome=success_recovered_slip"] == review.RECOVERED_LABEL_COLOR
    assert tokens["grade=1"] == review.WARNING_LABEL_COLOR
    assert tokens["damage=N"] == review.GOOD_LABEL_COLOR
    assert tokens["table=brief"] == review.WARNING_LABEL_COLOR
    assert tokens["slip=10"] == review.SLIP_LABEL_COLOR
    assert tokens["stable=20"] == review.RECOVERED_LABEL_COLOR
    assert tokens["failures=-"] == review.UNSET_LABEL_COLOR


def test_failure_review_labels_are_red():
    tokens = dict(
        review.review_status_tokens(
            outcome=review.OUTCOME_FAILURE,
            residual_grade=2,
            functional_damage=True,
            table_contact="supported",
            slip_index=None,
            stable_index=None,
            failure_reasons={"full_detach"},
        )
    )

    for key in ("outcome=failure", "grade=2", "damage=Y", "table=supported"):
        assert tokens[key] == review.FAILURE_LABEL_COLOR
    assert tokens["failures=full_detach"] == review.FAILURE_LABEL_COLOR


def test_review_space_replays_from_first_frame_after_reaching_end(tmp_path):
    frames = _review_frames(tmp_path)
    video, _ = recorder.write_review_artifacts(frames, tmp_path, attempt=8)
    keys = iter(
        (
            ord("x"),
            ord("x"),
            ord("x"),
            ord(" "),
            ord("j"),
            ord("f"),
            ord("d"),
            ord("t"),
            ord("0"),
            ord("n"),
            13,
        )
    )

    decision = recorder.interactive_review(
        video,
        frames,
        key_source=lambda: next(keys),
        show=False,
    )

    assert decision is not None
    assert decision["slip_index"] == 0


@pytest.mark.parametrize("table_contact", ("none", "brief", "supported"))
def test_only_acceptable_success_outcomes_are_promoted(tmp_path, table_contact):
    frames = _review_frames(tmp_path)
    video, timeline = recorder.write_review_artifacts(frames, tmp_path, attempt=1)
    recovered = recorder.outcome_record(
        attempt=1,
        dataset_episode=4,
        outcome="success_recovered_slip",
        slip_index=0,
        stable_index=2,
        failure_reasons=set(),
        table_contact=table_contact,
        residual_grade=1,
        functional_damage=False,
        frames=frames,
        review_video=video,
        review_timeline=timeline,
        evidence_root=tmp_path,
    )
    failure = recorder.outcome_record(
        attempt=2,
        dataset_episode=None,
        outcome="failure",
        slip_index=None,
        stable_index=None,
        failure_reasons={"full_detach"},
        table_contact=table_contact,
        residual_grade=0,
        functional_damage=False,
        frames=frames,
        review_video=video,
        review_timeline=timeline,
        evidence_root=tmp_path,
    )

    assert recovered["promoted_to_training"] and recovered["dataset_episode"] == 4
    assert not failure["promoted_to_training"] and failure["dataset_episode"] is None
    assert recovered["table_contact"] == table_contact
    assert recovered["teacher_tighten_candidate_s"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("failure_reasons", "residual_grade", "functional_damage"),
    [({"full_detach"}, 1, False), (set(), 2, False), (set(), 1, True)],
)
def test_bad_success_labels_are_rejected(
    tmp_path, failure_reasons, residual_grade, functional_damage
):
    frames = _review_frames(tmp_path)
    video, timeline = recorder.write_review_artifacts(frames, tmp_path, attempt=1)
    with pytest.raises(ValueError):
        recorder.outcome_record(
            attempt=1,
            dataset_episode=0,
            outcome="success_no_slip",
            slip_index=None,
            stable_index=None,
            failure_reasons=failure_reasons,
            table_contact="none",
            residual_grade=residual_grade,
            functional_damage=functional_damage,
            frames=frames,
            review_video=video,
            review_timeline=timeline,
            evidence_root=tmp_path,
        )


def test_table_contact_must_be_recorded_but_does_not_force_failure(tmp_path):
    frames = _review_frames(tmp_path)
    video, timeline = recorder.write_review_artifacts(frames, tmp_path, attempt=1)
    with pytest.raises(ValueError, match="select table contact"):
        recorder.outcome_record(
            attempt=1,
            dataset_episode=0,
            outcome="success_recovered_slip",
            slip_index=0,
            stable_index=2,
            failure_reasons=set(),
            table_contact=None,
            residual_grade=1,
            functional_damage=False,
            frames=frames,
            review_video=video,
            review_timeline=timeline,
            evidence_root=tmp_path,
        )


def test_dataset_view_attaches_current_teacher_label_without_exposing_it_to_record_loop():
    class Dataset:
        fps = 10
        features = {
            "observation.state": {},
            "action": {},
            **{key: {} for key in recorder.PV_AUXILIARY_FEATURES},
        }

        def add_frame(self, frame):
            self.frame = frame

    class Teleop:
        def training_grip_supervision(self):
            return {
                recorder.PV_TEACHER_FEATURE: np.asarray([0.75], dtype=np.float32),
                recorder.PV_TEACHER_VALID_FEATURE: np.asarray([1.0], dtype=np.float32),
                recorder.PV_SOURCE_TIMESTAMP_FEATURE: np.asarray([10.0], dtype=np.float64),
                recorder.PV_SENT_TIMESTAMP_FEATURE: np.asarray([10.01], dtype=np.float64),
                recorder.PV_RECEIVED_TIMESTAMP_FEATURE: np.asarray([10.02], dtype=np.float64),
                recorder.PV_FRAME_AGE_FEATURE: np.asarray([0.03], dtype=np.float32),
                recorder.PV_SEQUENCE_FEATURE: np.asarray([12], dtype=np.int64),
            }

    dataset = Dataset()
    view = recorder.PVTeachingDatasetView(dataset, Teleop(), human_intervention=True)
    assert not set(recorder.PV_AUXILIARY_FEATURES) & set(view.features)
    view.add_frame({"observation.state": np.zeros(9), "action": np.zeros(6), "task": "x"})
    assert dataset.frame[recorder.PV_TEACHER_FEATURE].tolist() == [0.75]
    assert dataset.frame[recorder.PV_TEACHER_VALID_FEATURE].tolist() == [1.0]
    assert dataset.frame[recorder.PV_SEQUENCE_FEATURE].tolist() == [12]
    assert dataset.frame[recorder.HUMAN_INTERVENTION_FEATURE].tolist() == [1.0]


REPO = Path(__file__).resolve().parents[3]


def test_launcher_orders_check_sender_recorder_and_cleanup():
    script = (REPO / "scripts" / "run_record_pv_ee.sh").read_text(encoding="utf-8")
    assert "--check-config" in script
    assert 'RECORDER_EVIDENCE="${SESSION}/recorder"' in script
    assert '--evidence-dir "${RECORDER_EVIDENCE}"' in script
    assert '>"${SESSION}/config-check.txt"' in script
    assert '--log "${SENDER_LOG}"' in script
    assert '--camera "${PV_CAMERA}"' in script
    assert 'FRONT_CAMERA="${PV_FRONT_CAMERA:-${DEFAULT_FRONT_CAMERA}}"' in script
    assert '--front-camera "${FRONT_CAMERA}"' in script
    assert '--side-camera "${SIDE_CAMERA}"' in script
    assert 'front_camera_path' in script and 'pv_camera_path' in script
    assert 'PV_FRONT_CAMERA and PV_CAMERA resolve to the same device' in script
    assert '--crop "${PV_CROP}"' in script
    assert "--mjpg" in script
    assert "--require-scene-match" in script
    assert 'MAX_LEVEL_AGE_MINUTES="${PV_MAX_LEVEL_AGE_MINUTES:-180}"' in script
    assert '--max-level-age-minutes "${MAX_LEVEL_AGE_MINUTES}"' in script
    assert '--grip-context "${GRIP_CONTEXT}"' in script
    assert '--max-load "${MAX_LOAD}"' in script
    assert '--max-current "${MAX_CURRENT}"' in script
    assert '--max-position-lag "${MAX_POSITION_LAG}"' in script
    assert "must be provided together" in script
    assert 'if [[ -n "${MAX_LOAD}" ]]; then' in script
    assert "--video-out" in script
    assert "trap cleanup EXIT INT TERM" in script
    assert "kill \"${sender_pid}\"" in script and "kill \"${recorder_pid}\"" in script
    assert "rm -rf" not in script


def test_launcher_hides_the_gpu_from_the_recorder_but_not_the_sender():
    """The recorder is CPU-only; the sender runs the PressureVision network and
    needs the GPU. Exporting CUDA_VISIBLE_DEVICES for the whole script is what
    put an earlier smoke run on CPU without anyone noticing."""
    script = (REPO / "scripts" / "run_record_pv_ee.sh").read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="" "$PYTHON" "${RECORDER}"' in script
    assert "hide_gpu" not in script


def test_launcher_runs_this_checkout_not_an_installed_copy():
    script = (REPO / "scripts" / "run_record_pv_ee.sh").read_text(encoding="utf-8")

    assert "_common.sh" in script
    assert "/home/" not in script.replace(
        "/dev/v4l/by-id", ""
    ) or "/home/" not in script


def test_evidence_and_dataset_paths_are_rejected_when_nested(tmp_path):
    levels = _levels(tmp_path)
    with pytest.raises(SystemExit):
        recorder.parse_args(
            [
                "--check-config",
                "--levels",
                str(levels),
                "--dataset-root",
                str(tmp_path / "root"),
                "--evidence-dir",
                str(tmp_path / "root" / "evidence"),
                "--max-load", "240",
                "--max-current", "35",
                "--max-position-lag", "5.0",
            ]
        )


def test_the_recorder_parks_on_the_right_v_not_a_left_fist():
    """The left hand is pressing the PV pad. A recorder that kept the fist
    clutch would be unparkable in practice, and would put newly recorded
    episodes on different operator semantics from local/evidence/."""
    source = inspect.getsource(recorder.run_recording)

    assert 'middle_gesture="right_v"' in source


def test_the_mapping_contract_is_written_into_the_dataset(tmp_path):
    """Schema v7 claims a recorded grip is reproducible from the episode alone.

    It was written only into the evidence directory's manifest. A dataset copied
    or uploaded on its own carried a teacher column with no scale attached: the
    release / zero / one positions and the filter cutoff all lived elsewhere.
    """
    contract = {
        "mapping": "carton_span",
        "release_pos": 100.0,
        "pressure_zero_pos": 32.0,
        "pressure_one_pos": 20.0,
        "cutoff_hz": 1.0,
        "stabilize": False,
    }

    written = recorder.write_dataset_mapping_contract(tmp_path, contract)

    assert written == tmp_path / "meta" / recorder.DATASET_MAPPING_CONTRACT_NAME
    assert json.loads(written.read_text()) == contract


def test_a_run_without_a_contract_writes_nothing(tmp_path):
    """The mappings that predate the range mapper must not leave an empty file."""
    assert recorder.write_dataset_mapping_contract(tmp_path, None) is None
    assert not (tmp_path / "meta" / recorder.DATASET_MAPPING_CONTRACT_NAME).exists()


def test_the_recorder_writes_the_contract_before_the_first_episode():
    """Writing it at finalize would lose it on a run that crashes mid-episode."""
    source = inspect.getsource(recorder.run_recording)
    write = source.index("write_dataset_mapping_contract(")
    first_save = source.index("dataset.save_episode()")
    assert write < first_save


class _ReleaseRobot:
    """Minimal follower double: remembers the last action and whether it closed."""

    def __init__(self, *, connected=True, gripper=22.0, raise_on_send=False):
        self.is_connected = connected
        self.gripper = gripper
        self.raise_on_send = raise_on_send
        self.actions = []

    def get_observation(self):
        return {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": self.gripper,
            "observation.images.front": object(),
        }

    def send_action(self, action):
        if self.raise_on_send:
            raise ConnectionError("bus went away")
        self.actions.append(dict(action))


def test_the_gripper_is_opened_before_the_bus_closes(monkeypatch):
    """Torque-off does not open the jaw; gear friction keeps the object held.

    A session ending mid-grasp -- ESC, a PV fault, an exception -- used to leave
    the carton clamped until someone pried it out.
    """
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)
    robot = _ReleaseRobot(gripper=22.0)

    assert recorder.release_gripper_before_disconnect(robot) is True

    assert len(robot.actions) == 1
    action = robot.actions[0]
    assert action["gripper.pos"] == recorder.SHUTDOWN_RELEASE_POS
    # The other joints are held where they are: this is a release, not a move.
    assert action["shoulder_pan.pos"] == 1.0
    assert action["elbow_flex.pos"] == 3.0
    assert "observation.images.front" not in action


def test_a_disconnected_robot_is_left_alone(monkeypatch):
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)
    robot = _ReleaseRobot(connected=False)

    assert recorder.release_gripper_before_disconnect(robot) is False
    assert robot.actions == []


def test_a_failing_release_never_masks_the_original_failure(monkeypatch):
    """It runs on teardown paths that are already handling an error."""
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)
    robot = _ReleaseRobot(raise_on_send=True)

    assert recorder.release_gripper_before_disconnect(robot) is False


def test_the_release_runs_before_the_disconnect():
    """ExitStack unwinds in reverse: the release must be registered second."""
    source = inspect.getsource(recorder.run_recording)
    disconnect = source.index("resources.callback(disconnect_robot_safely, robot)")
    release = source.index("resources.callback(release_gripper_before_disconnect, robot)")
    assert disconnect < release, (
        "the release is registered after the disconnect, so it would run first "
        "and the bus would still be closed on a clamped gripper"
    )


def test_a_frame_before_the_first_pv_packet_still_builds_a_row():
    """`reading` is None until the sender's first packet lands.

    Switching the provenance fields from `getattr(..., None)` to named access
    fixed five constant-zero columns and broke this: the recorder crashed on the
    first frame of the next session, mid-episode, on a connected robot.
    """
    supervision = recorder.pv_supervision_from_reading(None)

    assert supervision[recorder.PV_TEACHER_VALID_FEATURE] == pytest.approx([0.0])
    assert supervision[recorder.PV_SEQUENCE_FEATURE] == pytest.approx([0])
    assert supervision[recorder.PV_SOURCE_TIMESTAMP_FEATURE] == pytest.approx([0.0])
    # Every feature this function owns must be present, or the schema breaks.
    # (human_intervention is contributed elsewhere, not here.)
    for feature in (recorder.PV_TEACHER_FEATURE, recorder.PV_TEACHER_VALID_FEATURE,
                    *recorder.PV_TIMING_FEATURES):
        assert feature in supervision
