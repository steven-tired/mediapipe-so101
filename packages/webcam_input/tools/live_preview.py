"""Live webcam preview for the webcam-input pipeline.

Runs the REAL pipeline (one MediaPipe Hands(2) + reused SingleHandDetector math +
our process_hands) and overlays:
  - the detected hand skeletons (right = control, left = clutch)
  - the computed VR-frame wrist position (x,y,z), the fist/clutch state, and a
    dedicated LEFT-hand clutch readout

Windowed (default): live window; press 'q' to quit. Headless: --no-window
[--seconds N] [--save PATH] prints values and optionally saves an annotated frame.

Run (PYTHONPATH hygiene keeps ROS from shadowing the venv):
  env -u PYTHONPATH PYTHONPATH=packages/webcam_input/src python packages/webcam_input/tools/live_preview.py
"""

import argparse
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # prefer XWayland for the cv2 window

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402

from webcam_input.detector import SingleHandDetector  # noqa: E402
from webcam_input.depth import ScaleDepthStrategy  # noqa: E402
from webcam_input.gestures import count_curled_fingers  # noqa: E402
from webcam_input.wrist_estimator import WebcamWristEstimator  # noqa: E402
from webcam_input.webcam_source import WebcamSource  # noqa: E402

# BGR colors
GREEN, RED, GRAY, ORANGE, YELLOW, WHITE = (
    (0, 255, 0), (0, 0, 255), (180, 180, 180), (0, 165, 255), (0, 255, 255), (255, 255, 255),
)


def _draw_panel(frame, rows):
    """rows: list of (text, color). Draws a translucent panel, one line per row."""
    x0, y0, line_h = 14, 16, 24
    pad_w, pad_h = 470, y0 + line_h * len(rows) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (6 + pad_w, 6 + pad_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, (text, color) in enumerate(rows):
        y = y0 + line_h * (i + 1)
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _build_rows(right, left, wrist):
    """Organize what LeFranX consumes into a clean panel: right=arm+fingers, left=clutch."""
    rows = [("RIGHT hand  ->  arm pose + XHand fingers", WHITE)]
    if right is not None:
        p, q = wrist.position, wrist.quaternion
        rcurl = count_curled_fingers(np.asarray(right[0], dtype=float))
        rows += [
            (f"  wrist pos (VR, m):  x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}", GREEN),
            (f"  wrist quat:         ({q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f})", GREEN),
            (f"  fingers: 21 joints streaming  (curl {rcurl}/4)", GREEN),
        ]
    else:
        rows.append(("  NOT DETECTED", RED))

    rows.append(("LEFT hand  ->  clutch (pause gate)", WHITE))
    if left is not None:
        curls = count_curled_fingers(np.asarray(left[0], dtype=float))
        if curls >= 3:
            rows.append((f"  CLOSED -> PAUSE        (curl {curls}/4)", ORANGE))
        else:
            rows.append((f"  OPEN   -> active       (curl {curls}/4)", GREEN))
    else:
        rows.append(("  not visible -> holding last state", GRAY))

    if right is None:
        rows.append(("STATUS: waiting for right hand", RED))
    elif wrist.fist_state == "closed":
        rows.append(("STATUS: PAUSED (left fist)", YELLOW))
    else:
        rows.append(("STATUS: ACTIVE -> would record", GREEN))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-window", action="store_true", help="headless: print values, no GUI")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = until 'q')")
    ap.add_argument("--save", type=str, default="", help="save the last annotated frame to this path")
    args = ap.parse_args()

    source = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy()))
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.8, min_tracking_confidence=0.8,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    start = time.time()
    last_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            source.image_shape = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)
            right, left = WebcamSource.split_results(results)
            wrist, landmarks = source.process_hands(right=right, left=left)

            for hand_lms in (results.multi_hand_landmarks or []):
                SingleHandDetector.draw_skeleton_on_image(frame, hand_lms, style="white")

            rows = _build_rows(right, left, wrist)
            _draw_panel(frame, rows)
            last_frame = frame

            if args.no_window:
                print(" | ".join(text.strip() for text, _ in rows), flush=True)
            else:
                cv2.imshow("webcam-input live (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds and (time.time() - start) >= args.seconds:
                break
    finally:
        cap.release()
        hands.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        if args.save and last_frame is not None:
            cv2.imwrite(args.save, last_frame)
            print(f"saved annotated frame -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
