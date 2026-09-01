"""Record SO-101 pick-and-place demos with LeRobot's standard `record_loop`.

The arm is driven by the SAME validated control as the live teleop (`teleop_viz_ee.py`): both go
through `WebcamEEController`, so the recorded behaviour matches what you tuned (gripper points down,
slew-limited, gripper EMA, OAK hand-tracking). The teleoperator returns post-IK JOINT TARGETS, so
the recorded ACTION = joints (deploy needs no IK) and OBSERVATION = joint state + Logitech image.

Episode control is LeRobot's standard keyboard UX (arrow keys via init_keyboard_listener):
  right arrow = end/continue, left arrow = re-record this episode, ESC = stop.

Run:  ./scripts/run_record_ee.sh      (delete the dataset dir to re-record)
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.utils.feature_utils import combine_feature_dicts

from ..config_so101_webcam_ee import SO101WebcamEEConfig
from ..ee_controller import WebcamEEController
from ..grip.compose import add_gripper_mode_argument, build_gripper
from ..grip.mediapipe import MediaPipeGripperController
from ..paths import dataset_root, urdf_path
from .teleop_viz_ee import disconnect_robot_safely
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator
from webcam_input.depth import ScaleDepthStrategy

OBS_READ_RETRIES = 8   # per-frame observation read attempts before giving up


class ResilientSOFollower(SOFollower):
    """SOFollower whose per-frame observation read RETRIES through transient serial drops.

    record_loop calls get_observation() every frame, which does bus.sync_read("Present_Position")
    with num_retry=0 -- so a single dropped status packet (common on this Feetech bus under motor
    load, see so101-serial-brownout) crashes the whole recording. The live teleop avoids per-frame
    reads entirely (open-loop), and its at-rest read helper already retries; this applies the SAME
    proven pattern to the recorder's mandatory per-frame read.
    """

    def get_observation(self):
        for attempt in range(OBS_READ_RETRIES):
            try:
                return super().get_observation()
            except ConnectionError:
                if attempt == OBS_READ_RETRIES - 1:
                    raise
                time.sleep(0.003)

# Stable by-id symlink: the arm's /dev/ttyACM* index flips across replugs/rewiring (ACM0 <-> ACM1),
# but this serial-number path is constant. (Same path deploy_so101_ee.py uses.)
ARM_PORT = os.environ.get(
    "SO101_ARM_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
)
ARM_ID = "so101_follower_1"
# URDF is resolved lazily in _run_recording(); importing must not need a configured robot.
WORKSPACE_CAM_PATH = os.environ.get("SO101_WORKSPACE_CAM", "/dev/video2")
# YUYV (uncompressed), NOT MJPG: the C270's MJPG stream drops bytes over USB -> constant libjpeg
# "Corrupt JPEG data" warnings (and corrupted recorded frames). YUYV reads a clean 640x480@30
# (verified) with no JPEG decode at all. This matches the default format the earlier path used.
WORKSPACE_CAM_FOURCC = "YUYV"
HF_REPO_ID = "local/hand_tracking_pick_place"
DATASET_ROOT = str(dataset_root() / "hand_tracking_ee")
# 10, not 30: record_loop reads the bus EVERY frame (get_observation), unlike the open-loop live
# teleop which never reads during motion. Under GRIPPING load the motors brown out and drop reads,
# so the read retries and the loop falls to ~9 Hz. 10 Hz is what this bus sustains under load -> the
# loop runs STEADY instead of spamming "running slower", and 10 Hz is fine for a DP pick-place
# dataset. (Higher/steadier would need a higher bus baudrate or more motor power -- hardware.)
FPS = 10
NUM_EPISODES = 5
EPISODE_TIME_SEC = 120   # fallback auto-finish; normally you end an episode with RIGHT/SPACE
TASK = "hand-tracking pick and place"
class WebcamEEJointTeleop(Teleoperator):
    """LeRobot teleoperator that emits post-IK JOINT TARGETS from the shared controller.

    Wraps an OAK-backed WebcamSource + WebcamEEController. get_action() returns the same joint dict
    the live teleop sends; on HOLD (hand lost) it repeats the last commanded joints so the arm holds.
    """

    config_class = SO101WebcamEEConfig
    name = "webcam_ee_joint"

    def __init__(
        self,
        config,
        controller: WebcamEEController,
        source: WebcamSource,
        robot=None,
        use_oak: bool = False,
    ):
        super().__init__(config)
        self.config = config
        self._ctl = controller
        self._source = source
        self._robot = robot
        self._use_oak = use_oak
        # The workspace "bird-view" camera the dataset records (observation.images.front). Shown next
        # to the hand-cam so the operator can keep the BLOCK inside what the policy actually sees.
        self._front_cam = robot.cameras.get("front") if robot is not None else None
        self._connected = False
        self._status = ""          # banner shown on the preview (set by main per phase)
        self.last_key = 255        # last key from the preview window (for the SPACE start-gate)

    @property
    def action_features(self) -> dict:
        return {f"{m}.pos": float for m in self._ctl.motors}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        try:
            if self._use_oak:
                self._source.start_oak()
            else:
                self._source.start()
            # record_loop is headless (no display), so we show the hand-cam preview ourselves -- same
            # view the live teleop had: landmarks + control state. Window updates from get_action().
            self._win = "recording: hand-cam (control)  |  bird-view (what's recorded)"
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, 1280, 540)
            self._connected = True
        except Exception:
            self.disconnect()
            raise

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def set_status(self, status: str) -> None:
        self._status = status

    def get_action(self) -> dict:
        wrist, landmarks = self._source.latest()
        joint_act, state = self._ctl.step(wrist, landmarks)
        if joint_act is None:                      # HOLD: repeat last command so the arm holds
            joint_act = dict(self._ctl.cmd_state)
        # Preview: show the per-phase banner (set by main) + the live control state on the hand-cam
        # frame. waitKey(1) pumps the GUI and captures keys (last_key) for main's SPACE start-gate.
        frame = self._source.latest_frame()
        if frame is not None:
            color = {"MOVING": (0, 200, 0), "MIDDLE": (0, 165, 255), "HOLD": (0, 0, 255)}[state]
            banner = f"hand: {state}"
            # The right-V middle gesture is an optional controller feature. This
            # controller clutches on a left fist and does not implement it, so the
            # indicator appears only when a controller actually exposes it rather
            # than advertising a gesture that does nothing.
            if hasattr(self._ctl, "middle_gesture_active"):
                banner += "  middle(right-V): " + (
                    "ACTIVE" if self._ctl.middle_gesture_active
                    else "PENDING" if getattr(self._ctl, "middle_gesture_seen", False)
                    else "off"
                )
            cv2.putText(frame, self._status, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, banner, (8, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.imshow(self._win, self._with_birdview(frame))
            self.last_key = cv2.waitKey(1) & 0xFF
        return joint_act

    def _with_birdview(self, hand_frame):
        """Append the workspace (bird-view) cam frame to the RIGHT of the hand-cam, so the operator can
        keep the block inside the frame the policy records. Uses read_latest() -- the same frame
        get_observation records -- a non-blocking peek at the camera's background grab thread, so it
        adds no extra bus/camera load. Falls back to the hand-cam alone if the frame isn't ready."""
        if self._front_cam is None:
            return hand_frame
        try:
            bird = self._front_cam.read_latest(max_age_ms=1000)   # RGB; thread runs since robot.connect()
        except Exception:
            return hand_frame                                     # stale / not-ready -> hand-cam only
        bird = cv2.cvtColor(bird, cv2.COLOR_RGB2BGR)              # LeRobot frames are RGB; cv2 wants BGR
        h = hand_frame.shape[0]
        bird = cv2.resize(bird, (int(bird.shape[1] * h / bird.shape[0]), h))
        cv2.putText(bird, "BIRD VIEW - keep BLOCK in frame", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return np.hstack([hand_frame, bird])

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        # Each stage runs even if an earlier one raises; the first failure propagates.
        with ExitStack() as closing:
            closing.callback(setattr, self, "_connected", False)
            closing.callback(cv2.destroyAllWindows)
            closing.callback(self._ctl.close)
            self._source.stop()


def _read_positions(robot, tries=12):
    for _ in range(tries):
        try:
            obs = robot.get_observation()
            return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.1)
    raise ConnectionError("Arm position read kept failing -- check the USB cable/port.")


def _ramp_to(robot, target_pose, steps=30, holds=3):
    """Move gently to target_pose, repeating each step so slow joints can follow (anti-droop)."""
    start = _read_positions(robot)
    for a in np.linspace(0.0, 1.0, steps):
        cmd = {k: (1 - a) * start[k] + a * target_pose[k] for k in target_pose}
        for _ in range(holds):
            robot.send_action(cmd)
            time.sleep(0.04)
    for _ in range(50):
        robot.send_action(dict(target_pose))
        time.sleep(0.04)


def _identity(kind):
    if kind == "action":
        return RobotProcessorPipeline(steps=[], to_transition=robot_action_observation_to_transition,
                                      to_output=transition_to_robot_action)
    return RobotProcessorPipeline(steps=[], to_transition=observation_to_transition,
                                  to_output=transition_to_observation)


def build_dataset_features(robot, teleop, action_pipeline, observation_pipeline):
    if teleop.action_features != robot.action_features:
        raise ValueError("teleoperator and robot action features must match")
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=action_pipeline,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_pipeline,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def _close_and_dispose_recording_session(
    resources,
    *,
    dataset,
    ep: int,
    resume: bool,
    backup_dir: str,
    already: int,
) -> None:
    import shutil

    cleanup_error = None
    try:
        resources.close()
    except BaseException as exc:
        cleanup_error = exc

    disposition_error = None
    try:
        if cleanup_error is not None:
            keep = False
            print("\n[record] cleanup failed; forcing session discard.")
        elif ep == 0:
            keep = False
            print("\n[record] no episodes recorded this session.")
        else:
            try:
                ans = input(
                    f"\n>>> Include this session's {ep} episode(s) in the dataset? [Y/n]: "
                ).strip().lower()
            except EOFError:
                ans = "y"
            keep = ans not in ("n", "no")

        if keep:
            print(
                f"[record] KEPT -- dataset now has {dataset.num_episodes} episode(s) "
                f"at {DATASET_ROOT}"
            )
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
        else:
            if os.path.exists(DATASET_ROOT):
                shutil.rmtree(DATASET_ROOT)
            if resume:
                if not os.path.exists(backup_dir):
                    raise RuntimeError(
                        f"cannot restore discarded recording session: backup missing at {backup_dir}"
                    )
                shutil.move(backup_dir, DATASET_ROOT)
                print(
                    f"[record] DISCARDED session -- restored previous {already} episode(s)."
                )
            else:
                print("[record] DISCARDED session -- dataset removed (it was newly created).")
    except BaseException as exc:
        disposition_error = exc

    if cleanup_error is not None:
        if disposition_error is not None:
            cleanup_error.add_note(f"dataset disposition also failed: {disposition_error}")
            raise cleanup_error from disposition_error
        raise cleanup_error
    if disposition_error is not None:
        raise disposition_error


def _run_recording(resources: ExitStack, gripper=None, *, use_oak: bool = False) -> None:
    import shutil
    resume = os.path.exists(DATASET_ROOT)   # append to an existing dataset instead of overwriting

    cfg = SO101WebcamEEConfig()
    # warmup_s=3: the C270 needs >1s to deliver its first MJPG frame; the default warmup_s=1 times
    # out cam.connect() (verified). 3 gives margin so connect doesn't flake.
    cameras = {"front": OpenCVCameraConfig(index_or_path=WORKSPACE_CAM_PATH, width=640, height=480,
                                           fps=FPS, fourcc=WORKSPACE_CAM_FOURCC, warmup_s=3)}
    robot = ResilientSOFollower(SO101FollowerConfig(port=ARM_PORT, id=ARM_ID, use_degrees=True,
                                                    cameras=cameras, disable_torque_on_disconnect=False))
    resources.callback(disconnect_robot_safely, robot)
    robot.connect(calibrate=False)   # LeRobot default servo PID is good enough (verified)

    motors = list(robot.bus.motors.keys())
    kin = RobotKinematics(urdf_path=str(urdf_path()), target_frame_name="gripper_frame_link", joint_names=motors)
    # The gripper controller is the seam. The default derives the command from
    # MediaPipe pinch; the optional PressureVision adapter is injected here by the
    # composition entry point instead, and can only change grip STRENGTH.
    controller = WebcamEEController(
        robot,
        kin,
        cfg,
        use_oak=use_oak,
        gripper=gripper or MediaPipeGripperController(),
    )
    resources.callback(controller.close)

    # Move to the down ready pose, then build the workspace box around the resulting EE pose + seed.
    _ramp_to(robot, controller.middle_pose)
    obs0 = _read_positions(robot)
    ee_centre = kin.forward_kinematics(np.array([obs0[f"{m}.pos"] for m in motors], float))[:3, 3]
    controller.build(ee_centre)
    controller.seed(obs0)
    print(f"EE centre (ready FK): {np.round(ee_centre, 3)}  down rotvec: {np.round(controller.r_down, 3)}")

    source = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=cfg.workspace_size_m))
    teleop = WebcamEEJointTeleop(cfg, controller, source, robot=robot, use_oak=use_oak)

    ident_act, ident_obs = _identity("action"), _identity("obs")
    backup_dir = DATASET_ROOT + ".bak"
    if resume:
        # Snapshot the existing dataset BEFORE appending, so a "discard this session" at the end can
        # restore it exactly (cheap copy; the dataset is small). Append-mode shares the same features.
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(DATASET_ROOT, backup_dir)
    dataset = None
    ep = 0
    already = 0
    session_error = None
    try:
        if resume:
            dataset = LeRobotDataset.resume(
                HF_REPO_ID,
                root=DATASET_ROOT,
                image_writer_threads=4,
            )
        else:
            dataset = LeRobotDataset.create(
                repo_id=HF_REPO_ID, fps=FPS,
                features=build_dataset_features(robot, teleop, ident_act, ident_obs),
                robot_type=robot.name, root=DATASET_ROOT, use_videos=True, image_writer_threads=4,
            )
        resources.callback(dataset.finalize)
        already = dataset.num_episodes
        if resume:
            print(f"[record] RESUMING existing dataset: {already} episode(s) already there")

        # Silence the repetitive "Record loop running slower than target FPS" warning: it's expected
        # on this serial bus under gripping load (see FPS note) and the frames still record.
        import logging
        logging.getLogger().addFilter(
            lambda r: "running slower" not in r.getMessage())

        teleop.connect()
        resources.callback(teleop.disconnect)
        listener, events = init_keyboard_listener()
        if listener is not None:
            resources.callback(listener.stop)
        SPACE, ESC = 32, 27
        print("\n" + "=" * 66)
        print(f"  RECORDING {NUM_EPISODES} new episode(s) this session  ->  {DATASET_ROOT}")
        print(f"  Dataset already has {already}; will total {already + NUM_EPISODES} when done.")
        print("  Controls (RIGHT/LEFT/ESC work anywhere; SPACE needs the preview focused):")
        print("    RIGHT arrow (or SPACE) : START an episode / FINISH+SAVE the current one")
        print("    LEFT arrow             : DISCARD & re-record the current episode")
        print("    ESC                    : STOP the session")
        print("  Between episodes the arm still follows your hand so you can arrange the block.")
        print("=" * 66 + "\n")

        def wait_for_start(ep_human):
            """Keep the arm live while waiting for START; return False when the user quits."""
            teleop.set_status(
                f"READY  ep {ep_human}/{NUM_EPISODES}   ->  RIGHT/SPACE = start    ESC = quit"
            )
            teleop.last_key = 255
            events["exit_early"] = False
            while True:
                robot.send_action(teleop.get_action())
                if getattr(source, "oak_failed", False):
                    events["stop_recording"] = True
                    return False
                if events["stop_recording"] or teleop.last_key == ESC:
                    events["stop_recording"] = True
                    return False
                if events["exit_early"] or teleop.last_key == SPACE:
                    events["exit_early"] = False
                    return True

        while ep < NUM_EPISODES and not events["stop_recording"]:
            total_n = already + ep + 1   # this episode's number in the whole dataset
            print(f"--- Session {ep + 1}/{NUM_EPISODES} (dataset ep #{total_n}): arrange the block, "
                  "then press RIGHT/SPACE to start (ESC to quit) ---")
            if not wait_for_start(ep + 1):
                break
            print(f"Recording episode {total_n}")
            print(f">>> RECORDING dataset ep #{total_n} (session {ep + 1}/{NUM_EPISODES}) ...  "
                  "RIGHT = finish & save, LEFT = redo")
            teleop.set_status(f"REC  #{total_n}  (session {ep + 1}/{NUM_EPISODES})   ->  "
                              "RIGHT = finish    LEFT = redo    ESC = stop")
            events["exit_early"] = False
            record_loop(robot=robot, events=events, fps=FPS,
                        teleop_action_processor=ident_act, robot_action_processor=ident_act,
                        robot_observation_processor=ident_obs, teleop=teleop, dataset=dataset,
                        control_time_s=EPISODE_TIME_SEC, single_task=TASK, display_data=False)
            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                print(f"~~~ episode {ep + 1} discarded -- re-record it")
                continue
            if events["stop_recording"]:
                dataset.clear_episode_buffer()      # ESC mid-episode: drop the partial, don't save
                print("--- stop requested; current (unfinished) episode discarded")
                break
            dataset.save_episode()
            ep += 1
            print(f"Saved episode {total_n}")
            print(f"*** SAVED dataset ep #{total_n}  (session {ep}/{NUM_EPISODES}, total now {already + ep})"
                  + ("  -- session done!" if ep == NUM_EPISODES else ""))
    except BaseException as exc:
        session_error = exc
        raise
    finally:
        stop_error = None
        try:
            print("Stop recording")
        except BaseException as exc:
            stop_error = exc

        disposition_error = None
        try:
            _close_and_dispose_recording_session(
                resources,
                dataset=dataset,
                ep=ep,
                resume=resume,
                backup_dir=backup_dir,
                already=already,
            )
        except BaseException as exc:
            disposition_error = exc

        if session_error is not None:
            if stop_error is not None:
                session_error.add_note(f"stop announcement also failed: {stop_error}")
            if disposition_error is not None:
                session_error.add_note(
                    f"resource cleanup or dataset disposition also failed: {disposition_error}"
                )
                raise session_error from disposition_error
            if stop_error is not None:
                raise session_error from stop_error
        elif disposition_error is not None:
            if stop_error is not None:
                disposition_error.add_note(f"stop announcement also failed: {stop_error}")
            raise disposition_error from stop_error
        elif stop_error is not None:
            raise stop_error


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--oak", action="store_true",
                    help="use the OAK-D stereo depth camera instead of the monocular webcam")
    add_gripper_mode_argument(ap)
    args = ap.parse_args()
    gripper = build_gripper(args.gripper_mode,
                            zero_pos=args.grip_zero_pos, one_pos=args.grip_one_pos)
    with ExitStack() as resources:
        _run_recording(resources, gripper=gripper, use_oak=args.oak)


if __name__ == "__main__":
    main()
