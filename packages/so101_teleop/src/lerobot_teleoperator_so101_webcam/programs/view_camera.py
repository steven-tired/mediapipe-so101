"""Live camera preview for aligning the SO-101 policy camera.

Shows the deploy-resolution feed with reference markers from successful DP100
training frames, detected cube/tag centers, and pixel offsets for camera setup.

This tool does NOT move the arm. It is what `--profile` means in the deploy
procedure: DP50 was trained on episodes 0-49 with the tag around (421, 143),
while DP100 and ACT100 match the later layout around (416, 297). Aligning to
the wrong profile is a visual domain shift the policy sees but nothing reports.

Run:
  ./scripts/view_camera.sh                 # /dev/video2 / camera index 2
  ./scripts/view_camera.sh 0               # other camera index
  ./scripts/view_camera.sh --profile dp50  # align for the DP50 checkpoint
  ./scripts/view_camera.sh 4 --reference-image /path/to/reference.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
DEFAULT_CAMERA_INDEX = 2
DEFAULT_PROFILE = "dp100"
PROFILE_PATH = Path(__file__).with_name("camera_alignment_profiles.json")
TARGET_CUBE_CENTER = (320, 220)
TARGET_TAG_CENTER = (416, 297)
TOLERANCE_PX = 30


@dataclass(frozen=True)
class AlignmentProfile:
    name: str
    cube_target: tuple[int, int]
    tag_target: tuple[int, int]
    tolerance_px: int
    description: str = ""


@dataclass(frozen=True)
class Detection:
    center: tuple[int, int]
    box: tuple[int, int, int, int]
    area: int


@dataclass(frozen=True)
class AlignmentStatus:
    ok: bool
    cube_delta: tuple[int, int] | None
    tag_delta: tuple[int, int] | None


def load_alignment_profile(name: str = DEFAULT_PROFILE) -> AlignmentProfile:
    with PROFILE_PATH.open() as f:
        profiles = json.load(f)
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown camera alignment profile '{name}'. Available: {available}")
    data = profiles[name]
    return AlignmentProfile(
        name=name,
        cube_target=tuple(data["cube_target"]),
        tag_target=tuple(data["tag_target"]),
        tolerance_px=int(data.get("tolerance_px", TOLERANCE_PX)),
        description=str(data.get("description", "")),
    )


def _largest_component(mask: np.ndarray, min_area: int) -> Detection | None:
    n, _labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centers[i]
        candidate = Detection(
            center=(int(round(cx)), int(round(cy))),
            box=(int(x), int(y), int(w), int(h)),
            area=int(area),
        )
        if best is None or candidate.area > best.area:
            best = candidate
    return best


def detect_red_cube(frame: np.ndarray) -> Detection | None:
    """Detect the red cube top/side in the deployment camera frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([14, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([165, 45, 35]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(low_red | high_red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25 or y < 35 or w > 120 or h > 120:
            continue
        cx, cy = centers[i]
        candidate = Detection(
            center=(int(round(cx)), int(round(cy))),
            box=(int(x), int(y), int(w), int(h)),
            area=int(area),
        )
        if best is None or candidate.area > best.area:
            best = candidate
    return best


def detect_apriltag_square(frame: np.ndarray) -> Detection | None:
    """Detect the black AprilTag square used as the place target."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray < 65

    n, _labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 1500:
            continue
        if not (40 < w < 180 and 40 < h < 180):
            continue
        if not (0.5 < (w / max(h, 1)) < 1.8):
            continue
        cx, cy = centers[i]
        if cy < 100:
            continue
        candidates.append(
            Detection(
                center=(int(round(cx)), int(round(cy))),
                box=(int(x), int(y), int(w), int(h)),
                area=int(area),
            )
        )

    right_half = [c for c in candidates if c.center[0] > 220]
    candidates = right_half or candidates
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.area)


def alignment_status(
    cube_center: tuple[int, int] | None,
    tag_center: tuple[int, int] | None,
    cube_target: tuple[int, int] = TARGET_CUBE_CENTER,
    tag_target: tuple[int, int] = TARGET_TAG_CENTER,
    tolerance_px: int = TOLERANCE_PX,
) -> AlignmentStatus:
    cube_delta = None
    tag_delta = None
    ok = True
    if cube_center is None:
        ok = False
    else:
        cube_delta = (cube_center[0] - cube_target[0], cube_center[1] - cube_target[1])
        ok = ok and abs(cube_delta[0]) <= tolerance_px and abs(cube_delta[1]) <= tolerance_px

    if tag_center is None:
        ok = False
    else:
        tag_delta = (tag_center[0] - tag_target[0], tag_center[1] - tag_target[1])
        ok = ok and abs(tag_delta[0]) <= tolerance_px and abs(tag_delta[1]) <= tolerance_px

    return AlignmentStatus(ok=ok, cube_delta=cube_delta, tag_delta=tag_delta)


def _draw_cross(
    frame: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    tolerance_px: int,
) -> None:
    cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 28, 2)
    cv2.circle(frame, center, tolerance_px, color, 1)
    cv2.putText(
        frame,
        label,
        (center[0] + 8, max(18, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_detection(frame: np.ndarray, detection: Detection | None, color: tuple[int, int, int], label: str) -> None:
    if detection is None:
        return
    x, y, w, h = detection.box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.drawMarker(frame, detection.center, color, cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.putText(
        frame,
        label,
        (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_delta(
    frame: np.ndarray,
    center: tuple[int, int] | None,
    target: tuple[int, int],
    delta: tuple[int, int] | None,
    color: tuple[int, int, int],
) -> None:
    if center is None or delta is None:
        return
    cv2.arrowedLine(frame, center, target, color, 2, tipLength=0.16)
    mid = ((center[0] + target[0]) // 2, (center[1] + target[1]) // 2)
    cv2.putText(
        frame,
        f"dx={delta[0]:+d} dy={delta[1]:+d}",
        (mid[0] + 4, max(18, mid[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_alignment_overlay(frame: np.ndarray, profile: AlignmentProfile | None = None) -> np.ndarray:
    profile = profile or load_alignment_profile(DEFAULT_PROFILE)
    cube = detect_red_cube(frame)
    tag = detect_apriltag_square(frame)
    status = alignment_status(
        cube.center if cube else None,
        tag.center if tag else None,
        cube_target=profile.cube_target,
        tag_target=profile.tag_target,
        tolerance_px=profile.tolerance_px,
    )

    overlay = frame.copy()
    h, w = overlay.shape[:2]
    for x in (w // 3, 2 * w // 3):
        cv2.line(overlay, (x, 0), (x, h), (0, 150, 0), 1)
    for y in (h // 3, 2 * h // 3):
        cv2.line(overlay, (0, y), (w, y), (0, 150, 0), 1)

    _draw_cross(overlay, profile.cube_target, (0, 255, 255), "target cube", profile.tolerance_px)
    _draw_cross(overlay, profile.tag_target, (255, 255, 0), "target tag", profile.tolerance_px)
    _draw_detection(overlay, cube, (0, 0, 255), "cube now")
    _draw_detection(overlay, tag, (0, 255, 0), "tag now")
    _draw_delta(overlay, cube.center if cube else None, profile.cube_target, status.cube_delta, (0, 0, 255))
    _draw_delta(overlay, tag.center if tag else None, profile.tag_target, status.tag_delta, (0, 255, 0))

    banner_color = (0, 180, 0) if status.ok else (0, 0, 220)
    status_text = "ALIGNMENT OK" if status.ok else "ADJUST CAMERA"
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"{status_text}  {profile.name}: cube {profile.cube_target} tag {profile.tag_target}",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    if cube is None or tag is None:
        missing = []
        if cube is None:
            missing.append("cube")
        if tag is None:
            missing.append("tag")
        cv2.putText(
            overlay,
            f"missing: {', '.join(missing)}",
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_reference_overlay(
    frame: np.ndarray,
    reference: np.ndarray,
    *,
    alpha: float = 0.35,
    label: str = "reference",
) -> np.ndarray:
    """Blend a successful reference frame over the live image for scene alignment."""
    if reference.shape[:2] != frame.shape[:2]:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 1.0 - alpha, reference, alpha, 0.0)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"A2 TARGET GHOST: {label}  (align carton edges, q=quit)",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index} -- is teleop or another app using it?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", nargs="?", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="camera alignment profile: dp100 or dp50")
    parser.add_argument("--reference-image", type=Path, help="successful frame to blend over the live feed")
    parser.add_argument("--reference-alpha", type=float, default=0.35)
    parser.add_argument("--reference-label", default="A2 success")
    args = parser.parse_args()
    if not 0.0 < args.reference_alpha < 1.0:
        raise SystemExit("--reference-alpha must be between 0 and 1")
    reference = None
    profile = None
    if args.reference_image is not None:
        reference = cv2.imread(str(args.reference_image))
        if reference is None:
            raise SystemExit(f"Could not read reference image: {args.reference_image}")
    else:
        try:
            profile = load_alignment_profile(args.profile)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    cap = open_camera(args.camera)
    mode = args.reference_label if reference is not None else f"alignment {profile.name}"
    win = f"camera {args.camera} {mode} (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIDTH, HEIGHT)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if reference is not None:
            view = draw_reference_overlay(
                frame,
                reference,
                alpha=args.reference_alpha,
                label=args.reference_label,
            )
        else:
            view = draw_alignment_overlay(frame, profile)
        cv2.imshow(win, view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
