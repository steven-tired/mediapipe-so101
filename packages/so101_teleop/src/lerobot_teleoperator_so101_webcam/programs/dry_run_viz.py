"""Combined diagnostic: ONE camera window showing detection AND control output.

Shows the hand skeletons plus, as text overlay: right-hand `valid`, the clutch
(left-fist) state, whether control is MOVING or HOLD, and the 6 joint targets the
SO101Webcam teleoperator would send. No robot. Uses camera 0 -> make sure teleop and
the preview are both STOPPED first (camera can only be opened by one process).

Run:
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb python -m lerobot_teleoperator_so101_webcam.programs.dry_run_viz
"""

import cv2
import mediapipe as mp

from webcam_input.depth import ScaleDepthStrategy
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator

from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
from lerobot_teleoperator_so101_webcam.control import (
    MOTORS,
    REST_ACTION,
    clamp_joints,
    ema,
    rate_limit,
    retarget,
)


def main(camera_index: int = 0):
    cfg = SO101WebcamConfig()
    src = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy(), workspace_size_m=cfg.workspace_size_m))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index} (is teleop/preview still running?)")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.8, min_tracking_confidence=0.8,
    )
    draw = mp.solutions.drawing_utils
    last = dict(REST_ACTION)

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
            t = retarget(wrist.position, wrist.quaternion, landmarks.landmarks, cfg)
            t = clamp_joints(rate_limit(ema(t, last, cfg.smoothing), last, cfg.max_delta))
            last = t
        action = last

        lines = [
            f"right_valid={wrist.valid}  clutch(left_fist)={wrist.fist_state}",
            f"CONTROL: {'MOVING' if moving else 'HOLD (no right hand or clutch closed)'}",
        ] + [f"{m:>13}={action[m + '.pos']:+7.1f}" for m in MOTORS]
        color = (0, 200, 0) if moving else (0, 0, 255)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color if i < 2 else (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("so101_webcam diagnostic (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
