"""Webcam END-EFFECTOR teleop of the SO-101 + live hand-cam panel, single process.

Shows the hand-camera feed with landmarks drawn, overlaying control state (MOVING/MIDDLE/HOLD),
the EE target delta, and the joint targets being sent. Right hand moves the gripper in space
(differential, reference latched on the clutch rising edge); LEFT hand OPEN = run, fist = MIDDLE,
lost hand = HOLD; pinch = gripper.

Control lives in the SHARED `WebcamEEController` (same code the recorder uses), so live teleop and
recording can't diverge. Runs the robot with use_degrees=True. Keep the e-stop within reach.

Run (stop other camera apps first; default = OAK-D stereo depth):
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb python -m lerobot_teleoperator_so101_webcam.programs.teleop_viz_ee
Add --no-oak to fall back to the monocular webcam (CAMERA_INDEX) when no OAK-D is attached.

To RECORD a dataset, use record_so101_ee.py (LeRobot record_loop) -- not this script.
"""

import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator

from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
from lerobot_teleoperator_so101_webcam.ee_control import gripper_pos_from_pinch
from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController

from ..paths import urdf_path

ARM_PORT = "/dev/ttyACM0"
ARM_ID = "so101_follower_1"
CAMERA_INDEX = 0
# Resolved lazily inside main(): urdf_path() raises when SO-ARM100 is not configured,
# and importing this module must not require a configured robot.
# None on purpose: setting it makes send_action do a SECOND per-frame Present_Position read on a
# flaky bus. Per-step motion is capped by the controller's EMA + slew-limit instead.
MAX_RELATIVE_TARGET = None
MAX_COMM_FAILURES = 10      # consecutive serial write failures before giving up
_THUMB_TIP = 4
_INDEX_TIP = 8


def read_positions(robot, tries=12):
    """Read joint positions, retrying through intermittent serial read drops."""
    for _ in range(tries):
        try:
            obs = robot.get_observation()
            return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.1)
    raise ConnectionError("Arm position read kept failing -- check the USB cable/port "
                          "(try a direct port instead of the shared hub).")


def disconnect_robot_safely(robot) -> None:
    """Close fully or partially connected LeRobot resources without masking failures."""
    try:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    except Exception as exc:
        print(f"[cleanup] robot disconnect failed: {exc}")

    for camera in getattr(robot, "cameras", {}).values():
        try:
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        except Exception as exc:
            print(f"[cleanup] camera disconnect failed: {exc}")

    bus = getattr(robot, "bus", None)
    try:
        if bus is not None and getattr(bus, "is_connected", False):
            disable_torque = getattr(
                getattr(robot, "config", None),
                "disable_torque_on_disconnect",
                False,
            )
            bus.disconnect(disable_torque=disable_torque)
    except Exception as exc:
        print(f"[cleanup] robot bus disconnect failed: {exc}")


def main():
    # OAK-D by default: its stereo depth is metric, where the monocular fallback
    # only infers scale. --no-oak is the escape when the device is not attached.
    use_oak = "--no-oak" not in sys.argv
    cfg = SO101WebcamEEConfig(camera_index=CAMERA_INDEX)
    robot = SOFollower(SO101FollowerConfig(
        port=ARM_PORT, id=ARM_ID, use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET, cameras={},
        disable_torque_on_disconnect=False,  # hold pose on exit instead of collapsing
    ))
    for attempt in range(3):
        try:
            robot.connect(calibrate=False)
            break
        except Exception as e:
            if attempt == 2:
                raise
            # connect() opens the bus first; a hiccup after that leaves the port open and the
            # guarded robot.disconnect() won't close it. Close the bus directly before retrying.
            print(f"[connect] attempt {attempt + 1} failed: {e!r} -- cleaning up, retrying")
            try:
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=False)
            except Exception:
                pass
            time.sleep(1.0)
    # LeRobot default servo PID is good enough (verified) -- no tuned-PID re-apply.

    motors = list(robot.bus.motors.keys())
    kin = RobotKinematics(urdf_path=str(urdf_path()), target_frame_name="gripper_frame_link", joint_names=motors)
    controller = WebcamEEController(robot, kin, cfg, use_oak=use_oak)

    # Ramp gently to the down ready pose (repeat each step so slow joints can follow), then build
    # the workspace box around the resulting EE pose and seed the controller's open-loop state.
    start = read_positions(robot)
    for a in np.linspace(0.0, 1.0, 30):
        cmd = {k: (1 - a) * start[k] + a * controller.middle_pose[k] for k in controller.middle_pose}
        for _ in range(3):
            robot.send_action(cmd)
            time.sleep(0.04)
    for _ in range(50):
        robot.send_action(dict(controller.middle_pose))
        time.sleep(0.04)

    obs0 = read_positions(robot)
    ee_centre = kin.forward_kinematics(np.array([obs0[f"{m}.pos"] for m in motors], float))[:3, 3]
    controller.build(ee_centre)
    controller.seed(obs0)
    print(f"EE centre (ready FK): {np.round(ee_centre, 3)}  down rotvec: {np.round(controller.r_down, 3)}")

    # Camera + depth source: the OAK-D (clean stereo depth) unless --no-oak asked for the webcam.
    if use_oak:
        from webcam_input.depth import OAKDepthStrategy
        from webcam_input.oak_camera import OAKCamera
        oak_depth = OAKDepthStrategy(radius_px=6, ema_alpha=0.4)
        depth_strategy = oak_depth
        cam = OAKCamera(rgb_size=(640, 480), fps=30)
        cam.start()

        def read_frame():
            rgb, depth = cam.read()
            oak_depth.update_depth(depth)
            return rgb

        def release_cam():
            cam.stop()
    else:
        depth_strategy = ScaleDepthStrategy()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            robot.disconnect()
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX} (another app using it?)")

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def release_cam():
            cap.release()

    src = WebcamSource(WebcamWristEstimator(depth_strategy, workspace_size_m=cfg.workspace_size_m))
    print(f"Camera source: {'OAK-D (stereo depth)' if use_oak else 'laptop webcam (monocular)'}")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.8, min_tracking_confidence=0.8,
    )
    draw = mp.solutions.drawing_utils

    comm_failures = 0
    joint_act = None
    print("Right hand moves the gripper; LEFT open=run / fist=MIDDLE / lost-hand=HOLD; pinch=gripper. q to quit.")
    win = "so101_webcam_ee teleop (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)

    try:
        while True:
            frame = read_frame()
            if frame is None:
                continue
            src.image_shape = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            right, left = WebcamSource.split_results(results)
            wrist, landmarks = src.process_hands(right, left)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

            # All control (down orientation, slew, EMA, clutch/HOLD) lives in the shared controller.
            joints, state = controller.step(wrist, landmarks)
            try:
                if joints is not None:           # MOVING or MIDDLE: send the targets
                    robot.send_action(joints)
                    joint_act = joints
                    comm_failures = 0
                # else HOLD: send nothing; arm holds its last commanded pose.
            except ConnectionError as e:
                comm_failures += 1
                print(f"[serial] write {comm_failures}/{MAX_COMM_FAILURES}: {e}")
                if comm_failures >= MAX_COMM_FAILURES:
                    print("[serial] too many consecutive write failures -- check arm/USB. Stopping.")
                    break

            # --- overlay ---
            lm = np.asarray(landmarks.landmarks, dtype=float)
            pinch = float(np.linalg.norm(lm[_THUMB_TIP] - lm[_INDEX_TIP]))
            color = {"MOVING": (0, 200, 0), "MIDDLE": (0, 165, 255), "HOLD": (0, 0, 255)}[state]
            lines = [
                f"right_valid={wrist.valid}  clutch(left_fist)={wrist.fist_state}",
                f"CONTROL: {state}  gripper={gripper_pos_from_pinch(pinch, cfg) if state == 'MOVING' else 0.0:5.1f}",
            ]
            if joint_act is not None:
                lines += [f"{m:>13}={joint_act.get(m + '.pos', float('nan')):+7.1f}" for m in motors]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color if i < 2 else (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        release_cam()
        hands.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
