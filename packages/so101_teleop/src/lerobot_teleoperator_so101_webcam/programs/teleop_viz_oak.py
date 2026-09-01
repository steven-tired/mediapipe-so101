"""Teleop the SO-101 from an OAK-D (RGB + REAL stereo depth) with a live diagnostic window.

Same differential control as teleop_viz.py, but the hand camera is the OAK-D and the wrist's
z (depth) comes from aligned stereo depth instead of the noisy monocular ScaleDepthStrategy --
which was the root cause of the shaking / bad reach. RIGHT hand moves the arm; LEFT hand OPEN =
run, fist = freeze; pinch = gripper. 'q' quits. Keep the e-stop within reach.

Run (stop other OAK/camera apps first):
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb python -m lerobot_teleoperator_so101_webcam.programs.teleop_viz_oak
"""

import cv2
import mediapipe as mp
import numpy as np

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from webcam_input.depth import OAKDepthStrategy
from webcam_input.oak_camera import OAKCamera
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator

from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
from lerobot_teleoperator_so101_webcam.control import (
    MOTORS,
    REST_ACTION,
    clamp_joints,
    ema,
    hand_roll,
    rate_limit,
    retarget_delta,
)

ARM_PORT = "/dev/ttyACM0"
ARM_ID = "so101_follower_1"
# Writes-only (no per-frame Present_Position read) -- see teleop_viz.py for the why.
MAX_RELATIVE_TARGET = None


def main():
    cfg = SO101WebcamConfig()
    robot = SOFollower(
        SO101FollowerConfig(
            port=ARM_PORT, id=ARM_ID, use_degrees=False,
            max_relative_target=MAX_RELATIVE_TARGET, cameras={},
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect(calibrate=False)
    print("Robot connected (OAK depth). RIGHT hand moves the arm; LEFT hand OPEN to run, fist to freeze.")

    try:
        obs0 = robot.get_observation()
        seed = {f"{m}.pos": float(obs0[f"{m}.pos"]) for m in MOTORS}
    except (ConnectionError, KeyError):
        seed = dict(REST_ACTION)

    oak_depth = OAKDepthStrategy(radius_px=6, ema_alpha=0.4)
    src = WebcamSource(WebcamWristEstimator(oak_depth, workspace_size_m=cfg.workspace_size_m))
    cam = OAKCamera(rgb_size=(640, 480), fps=30)
    cam.start()
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.8, min_tracking_confidence=0.8,
    )
    draw = mp.solutions.drawing_utils
    last = seed
    comm_failures = 0
    max_comm_failures = 10
    prev_moving = False
    hand_ref = None
    roll_ref = 0.0
    arm_ref = dict(last)

    try:
        while True:
            rgb, depth = cam.read()
            oak_depth.update_depth(depth)            # feed real metric depth for this frame
            src.image_shape = rgb.shape[:2]
            results = hands.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            right, left = WebcamSource.split_results(results)
            wrist, landmarks = src.process_hands(right, left)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    draw.draw_landmarks(rgb, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

            moving = wrist.valid and wrist.fist_state != "closed"
            if moving:
                if not prev_moving:
                    hand_ref = np.asarray(wrist.position, dtype=float).copy()
                    roll_ref = hand_roll(wrist.quaternion)
                    arm_ref = dict(last)
                t = retarget_delta(wrist.position, wrist.quaternion, landmarks.landmarks,
                                   hand_ref, roll_ref, arm_ref, cfg)
                t = clamp_joints(rate_limit(ema(t, last, cfg.smoothing), last, cfg.max_delta))
                last = t
            prev_moving = moving
            action = last
            try:
                robot.send_action(action)
                comm_failures = 0
            except ConnectionError as e:
                comm_failures += 1
                print(f"[serial] {comm_failures}/{max_comm_failures} comm failure: {e}")
                if comm_failures >= max_comm_failures:
                    print("[serial] too many consecutive failures -- check arm power/USB. Stopping.")
                    break

            lines = [
                f"right_valid={wrist.valid}  clutch(left_fist)={wrist.fist_state}",
                f"CONTROL: {'MOVING' if moving else 'HOLD'}  wrist_z={wrist.position[2]:+.3f}m",
            ] + [f"{m:>13}={action[m + '.pos']:+7.1f}" for m in MOTORS]
            color = (0, 200, 0) if moving else (0, 0, 255)
            for i, line in enumerate(lines):
                cv2.putText(rgb, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color if i < 2 else (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("so101_webcam_oak teleop (q to quit)", rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        hands.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
