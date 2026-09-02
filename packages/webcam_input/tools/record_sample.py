"""Capture a few seconds of the pipeline's payloads to an .npz — a CHECK that we are
producing exactly the data LeFranX will record, NOT the real recorder.

The real dataset recorder is LeFranX/LeRobot (Phase B). This just dumps, per frame:
  - right_position  (N, 3)   wrist position, VR frame, meters       -> arm
  - right_quaternion(N, 4)   wrist orientation [x,y,z,w], VR frame  -> arm
  - right_joints    (N, 21, 3) MANO finger joints                   -> XHand
  - clutch          (N,)     "open"/"closed"                        -> pause gate
  - valid           (N,)     right hand tracked this frame

Run (put your RIGHT hand up; LEFT fist toggles clutch):
  env -u PYTHONPATH PYTHONPATH=packages/webcam_input/src python packages/webcam_input/tools/record_sample.py --seconds 5
Then verify:
  env -u PYTHONPATH PYTHONPATH=packages/webcam_input/src python packages/webcam_input/tools/record_sample.py --inspect /tmp/webcam_sample.npz
"""

import argparse
import time

import numpy as np


def inspect(path):
    d = np.load(path, allow_pickle=True)
    valid = d["valid"]
    n = len(valid)
    nv = int(valid.sum())
    print(f"file: {path}")
    print(f"frames: {n}   valid (right hand tracked): {nv}")
    print("fields & shapes:")
    for k in ("right_position", "right_quaternion", "right_joints", "clutch", "valid"):
        print(f"  {k:18s} {d[k].shape} {d[k].dtype}")
    print(f"clutch states seen: {sorted(set(d['clutch'].tolist()))}")
    if nv:
        i = int(np.argmax(valid))  # first valid frame
        print("first valid frame:")
        print("  position :", np.round(d["right_position"][i], 3))
        print("  quaternion:", np.round(d["right_quaternion"][i], 3))
        print("  joints[0..2]:", np.round(d["right_joints"][i][:3], 3).tolist())
        print("  clutch   :", d["clutch"][i])
    else:
        print("No valid frames — was your RIGHT hand in view? (re-run with hand up)")


def record(seconds, out, camera):
    import cv2
    import mediapipe as mp
    from webcam_input.depth import ScaleDepthStrategy
    from webcam_input.wrist_estimator import WebcamWristEstimator
    from webcam_input.webcam_source import WebcamSource

    source = WebcamSource(WebcamWristEstimator(ScaleDepthStrategy()))
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.8, min_tracking_confidence=0.8)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}")

    positions, quats, joints, clutch, valid = [], [], [], [], []
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            source.image_shape = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            right, left = WebcamSource.split_results(hands.process(rgb))
            wrist, landmarks = source.process_hands(right=right, left=left)
            positions.append(np.asarray(wrist.position, dtype=float))
            quats.append(np.asarray(wrist.quaternion, dtype=float))
            joints.append(np.asarray(landmarks.landmarks, dtype=float))
            clutch.append(wrist.fist_state)
            valid.append(bool(wrist.valid))
    finally:
        cap.release()
        hands.close()

    np.savez(
        out,
        right_position=np.array(positions),
        right_quaternion=np.array(quats),
        right_joints=np.array(joints),
        clutch=np.array(clutch),
        valid=np.array(valid),
    )
    print(f"saved {len(valid)} frames -> {out}")
    inspect(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", type=str, default="/tmp/webcam_sample.npz")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--inspect", type=str, default="", help="inspect an existing .npz and exit")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.inspect)
    else:
        record(args.seconds, args.out, args.camera)


if __name__ == "__main__":
    main()
