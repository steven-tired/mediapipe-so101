"""Teleop the SO-101 with the webcam AND show a live diagnostic window in ONE process.

Single camera (no contention), drives the real arm, and overlays right-hand `valid`,
clutch (left-fist) state, MOVING/HOLD, and the 6 joint targets being sent.

Controls: RIGHT hand moves the arm; LEFT hand OPEN = run, LEFT fist CLOSED = freeze.
Press 'q' in the window (or Ctrl+C) to stop. Keep the e-stop within reach.

Run (stop teleop/preview/viz first so camera 0 is free):
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb python -m lerobot_teleoperator_so101_webcam.programs.teleop_viz
"""

import cv2
import mediapipe as mp
import numpy as np

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from webcam_input.depth import ScaleDepthStrategy
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
CAMERA_INDEX = 0
# max_relative_target is LEFT None ON PURPOSE: when set, send_action does a per-frame
# sync_read("Present_Position") (see so_follower.py) which is what kept dropping packets
# ("no status packet") on this serial setup. We already cap per-step motion in software
# (rate_limit(max_delta) + clamp_joints), so the robot-side read-back is redundant. Running
# on writes-only makes teleop robust to the flaky reads.
MAX_RELATIVE_TARGET = None


def main():
    cfg = SO101WebcamConfig()
    robot = SOFollower(
        SO101FollowerConfig(
            port=ARM_PORT, id=ARM_ID, use_degrees=False,
            max_relative_target=MAX_RELATIVE_TARGET, cameras={},
            disable_torque_on_disconnect=False,  # hold pose on exit instead of collapsing
        )
    )
    robot.connect(calibrate=False)
    print("Robot connected. RIGHT hand moves the arm; LEFT hand OPEN to run, fist to freeze.")

    # Seed the software rate-limiter from the arm's ACTUAL pose (one read) so the first commands
    # ramp gently from where the arm really is, not from REST. Falls back to REST if the read drops.
    try:
        obs0 = robot.get_observation()
        seed = {f"{m}.pos": float(obs0[f"{m}.pos"]) for m in MOTORS}
    except (ConnectionError, KeyError):
        seed = dict(REST_ACTION)

    src = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=cfg.workspace_size_m))
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        robot.disconnect()
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX} (teleop/preview/viz still running?)")
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
            ok, frame = cap.read()
            if not ok:
                continue
            src.image_shape = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            right, left = WebcamSource.split_results(results)
            wrist, landmarks = src.process_hands(right, left)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

            moving = wrist.valid and wrist.fist_state != "closed"
            if moving:
                if not prev_moving:
                    # Clutch rising edge: latch hand reference + the arm's CURRENT pose so motion
                    # is differential (relative) -- the arm stays put and moves only as the hand
                    # moves, instead of diving to an absolute pose.
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
                f"CONTROL: {'MOVING' if moving else 'HOLD'}",
            ] + [f"{m:>13}={action[m + '.pos']:+7.1f}" for m in MOTORS]
            color = (0, 200, 0) if moving else (0, 0, 255)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color if i < 2 else (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("so101_webcam teleop (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
