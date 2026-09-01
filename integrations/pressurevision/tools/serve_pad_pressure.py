#!/usr/bin/env python3
"""Stream PressureVision pad readings to the SO-101 teleop over localhost UDP.

PressureVision needs a GPU and a package set (`segmentation-models-pytorch`,
`timm`) that `.venv-lerobot` does not have, so the model runs here on
`.venv-pressurevision` and ships scalar metrics to the teleop process instead of
being imported into it. Inference is ~12 ms/frame on the 4060; on CPU it is
~107 ms, which cannot keep up with a 30 Hz control loop, so --device cpu is for
debugging only.

Robot-free: this sends numbers to a socket and drives nothing. The teleop
consumes them in shadow mode, where they change no robot command.

The wire format is duplicated rather than imported: the consumer lives in
another repo on another venv. SHARED_PV_TEST_PACKET is the same literal on both
sides and each side round-trips it in its own tests, which is what actually
keeps the two encoders honest.

  env -u PYTHONPATH python \
    scripts/serve_pad_pressure.py --preview
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import time

# Before cv2: this build is Qt5, the desktop is Wayland, and the Qt platform is
# chosen at import. Left on Wayland the preview window can fail to map, which
# looks like the model never loaded -- and with no window there is no key to
# press, so the usual exit is closing the terminal and leaving the sender holding
# the camera. teleop_viz_ee.py prescribes the same xcb hint for the same reason.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aim_pad_camera import (  # noqa: E402
    SETTLE_FRAMES,
    SETTLE_S,
    WARMUP_FRAMES,
    fingerprint_drift,
    lock_camera,
    open_camera,
    pixel_format,
    scene_fingerprint,
)
from calibration_gate import check as check_calibration  # noqa: E402
from evaluate_press_thresholds import (  # noqa: E402
    ABSTAIN,
    classify,
    fit_thresholds,
    load_trials,
    score,
)
from pressurevision_probe import (  # noqa: E402
    crop_box,
    load_model,
    overlay,
    preprocess,
    to_kpa,
)

DEFAULT_REPO = None
DEFAULT_CAMERA = 2          # the overhead Logi watching the pad, not the teleop webcam
DEFAULT_PORT = 8090
DEFAULT_PREVIEW_SHARE = Path("/tmp/pressurevision-preview-v1.mmap")
# sum_kpa, not mean_kpa_in_contact. The mean saturates: every contact pixel takes a
# log-spaced bin edge topping out at 64 kPa, so light and hard sat about one bin apart
# and a live session pressing slightly harder put every frame above the top boundary --
# 0 of 517 frames landed in the middle band. sum_kpa keeps growing with area as well as
# intensity and separated light from hard best on the calibration set too (d' 3.85 vs
# 3.37). It is tied to the crop, but so is the calibration already.
ANCHOR_METRIC = "sum_kpa"
WINDOW = "PressureVision UDP sender (q to quit)"

# Mirrored in webcam-input/.../pv_pressure.py -- keep both literals identical.
PV_PACKET_SCHEMA_VERSION = 5
SHARED_PV_TEST_PACKET = (
    "5,42,1700000000.100000,1700000000.123456,7.250000,14.000000,3712.000000,512,"
    "-0.250000,0.500000,522,416,0.625000,1,3"
)
ABSTAIN_LEVEL = -1
DEFAULT_ABSTAIN_FRACTION = 0.5
SCENE_FINGERPRINT_FRAMES = 30
PREVIEW_MAGIC = b"PVPREV1\0"
PREVIEW_HEADER = struct.Struct("<8sQdIIII")
PREVIEW_HEADER_SIZE = 64


class SharedPreviewWriter:
    """Publish the latest rendered PV panel without coupling it to control UDP."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = None
        self._map = None
        self._payload_size = 0
        self._sequence = 0

    def _open(self, payload_size: int) -> None:
        self.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w+b")
        self._file.truncate(PREVIEW_HEADER_SIZE + payload_size)
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        self._payload_size = payload_size

    def publish(self, panel: np.ndarray, *, observed_at_s: float | None = None) -> None:
        frame = np.ascontiguousarray(panel, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("preview panel must be HxWx3 uint8")
        payload_size = int(frame.nbytes)
        if self._map is None or payload_size != self._payload_size:
            self._open(payload_size)

        self._sequence += 1
        if self._sequence % 2 == 0:
            self._sequence += 1
        timestamp = time.monotonic() if observed_at_s is None else float(observed_at_s)
        header = (
            PREVIEW_MAGIC,
            self._sequence,
            timestamp,
            int(frame.shape[0]),
            int(frame.shape[1]),
            int(frame.shape[2]),
            payload_size,
        )
        self._map[:PREVIEW_HEADER.size] = PREVIEW_HEADER.pack(*header)
        self._map[PREVIEW_HEADER_SIZE:PREVIEW_HEADER_SIZE + payload_size] = frame.tobytes()
        self._sequence += 1
        header = (PREVIEW_MAGIC, self._sequence, timestamp, *header[3:])
        self._map[:PREVIEW_HEADER.size] = PREVIEW_HEADER.pack(*header)

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._payload_size = 0


def encode_pv_packet(
    *,
    sequence: int,
    source_observed_at_s: float,
    sent_at_s: float,
    mean_kpa_in_contact: float,
    max_kpa: float,
    sum_kpa: float,
    contact_px: int,
    off_x: float,
    off_y: float,
    crop_w: int,
    crop_h: int,
    pressure_0_1: float,
    level: int,
    n_levels: int,
) -> bytes:
    return (
        ",".join(
            (
                str(PV_PACKET_SCHEMA_VERSION),
                str(int(sequence)),
                f"{source_observed_at_s:.6f}",
                f"{sent_at_s:.6f}",
                f"{mean_kpa_in_contact:.6f}",
                f"{max_kpa:.6f}",
                f"{sum_kpa:.6f}",
                str(int(contact_px)),
                f"{off_x:.6f}",
                f"{off_y:.6f}",
                str(int(crop_w)),
                str(int(crop_h)),
                f"{pressure_0_1:.6f}",
                str(int(level)),
                str(int(n_levels)),
            )
        )
        + "\n"
    ).encode("utf-8")


def reduce_frame(kpa: np.ndarray, contact_thresh: float) -> dict:
    """Collapse a per-pixel kPa map to what the consumer gates on.

    off_x / off_y are the contact centroid relative to the crop centre, each
    normalized by the crop half-width / half-height. Fractions rather than
    pixels because the crop is resized to the network's 480x384 input, so
    position within the crop is what decides where a pixel lands in the model's
    field of view -- which is the axis the response falloff was measured along
    (docs/HANDOFF.md section 7.0). Signed and separate rather than one radius so
    an off-centre or anisotropic falloff can be fitted later without changing
    the wire format.
    """
    contact = kpa >= contact_thresh
    contact_px = int(contact.sum())
    height, width = kpa.shape[:2]
    if contact_px == 0:
        return {
            "mean_kpa_in_contact": 0.0,
            "max_kpa": float(kpa.max()),
            "sum_kpa": float(kpa.sum()),
            "contact_px": 0,
            "off_x": 0.0,
            "off_y": 0.0,
        }
    ys, xs = np.nonzero(contact)
    return {
        "mean_kpa_in_contact": float(kpa[contact].mean()),
        "max_kpa": float(kpa.max()),
        "sum_kpa": float(kpa.sum()),
        "contact_px": contact_px,
        "off_x": float(xs.mean() / (width / 2.0) - 1.0),
        "off_y": float(ys.mean() / (height / 2.0) - 1.0),
    }


def trials_excluding_early_lifts(session: Path, metric: str):
    """Per-level trial values, minus trials the operator stopped pressing partway.

    A hold that shows contact and then loses it for the rest of the trial is a
    lift, and once fewer than half the frames hold contact the trial's median
    lands in the empty ones: session 06 lost a light and a hard trial to exactly
    that, which pulled the light-hard d' from 3.4 down to 1.6 and failed the gate
    on a session whose medians were 11 and 20 kPa apart.

    Only trials whose median the lift actually broke are dropped. A hold that ran
    9 of 15 frames still summarises the press correctly, and a no-press trial
    whose false contact faded is evidence about that level, not a failed press --
    dropping either would flatter the fit.

    Detected from the frame trace rather than from the label, so it applies the
    same way to every level and works on sessions recorded before the capture
    script learned to stop a hold early.
    """
    rows = [
        json.loads(line)
        for line in (session / "capture.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    trials: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row.get("row_type") != "sample":
            continue
        trials.setdefault((row["trial_index"], row["target_g"]), []).append(row)

    per_level: dict[int, list[float]] = {level: [] for level in meta["targets_g"]}
    lifted = []
    for (trial, level), frames in sorted(trials.items()):
        contact = [frame["contact_px"] > 0 for frame in frames]
        held = sum(contact)
        # Contact present, then gone for the whole tail, for so much of the hold
        # that the median no longer lands on a frame with any press in it.
        lift = 0 < held and all(contact[:held]) and not any(contact[held:])
        if lift and held * 2 < len(contact):
            lifted.append((trial, level, held, len(contact)))
            continue
        per_level[level].append(float(np.median([float(f[metric]) for f in frames])))
    return {level: np.asarray(values) for level, values in per_level.items()}, lifted


def fit_levels(
    session: Path,
    abstain_fraction: float = DEFAULT_ABSTAIN_FRACTION,
    *,
    contact_gated: bool | None = None,
) -> dict:
    """Fit press-level boundaries from a labelled session, gated on separability.

    Bands rather than a continuous kPa scale because the control output is banded
    anyway: the gripper takes a position, not a force, so a fitted 0..1 ramp would
    be precision neither the model (its top bin saturates) nor the arm can use.
    What the boundaries add over a ramp is abstention -- a reading between two
    levels says so instead of inventing a number in the middle.

    Rejecting rather than defaulting is the point: ungated boundaries make every
    level the sender ships arbitrary, and a shadow log written against arbitrary
    levels cannot be compared to anything later.
    """
    capture = session / "capture.jsonl"
    if not capture.is_file():
        raise SystemExit(f"{session}: no capture.jsonl -- was the session ever run?")
    if capture.stat().st_size == 0:
        raise SystemExit(
            f"{session}: capture.jsonl is empty -- the capture run died before recording "
            "anything. Delete the directory and re-run the capture."
        )
    _meta = json.loads((session / "capture.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if contact_gated is None:
        contact_gated = bool(_meta.get("contact_gated", False))
    per_level, lifted = trials_excluding_early_lifts(session, ANCHOR_METRIC)
    for trial, level, held, total in lifted:
        print(
            f"[levels] excluding trial {trial} (level {level}): contact lost after "
            f"{held}/{total} frames -- the operator lifted mid-hold, so the trial's "
            "median walks through frames with no press in them"
        )
    per_level = {level: values for level, values in per_level.items() if len(values)}
    if contact_gated:
        # MediaPipe pinch owns contact. The model is fitted only on pressure grades;
        # wire level zero remains reserved for the consumer's inactive baseline.
        per_level.pop(0, None)
    if len(per_level) < 2:
        raise SystemExit(f"{session}: need at least two levels, found {sorted(per_level)}")
    verdict = check_calibration({level: list(values) for level, values in per_level.items()})
    if not verdict["accepted"]:
        raise SystemExit(
            f"{session}: calibration gate rejected this session: " + "; ".join(verdict["reasons"])
        )
    thresholds = fit_thresholds(per_level, abstain_fraction)
    fitted = {
        "schema_version": 3,
        "metric": ANCHOR_METRIC,
        "abstain_fraction": abstain_fraction,
        "contact_gated": bool(contact_gated),
        "wire_level_offset": 1 if contact_gated else 0,
        "n_levels": len(per_level) + (1 if contact_gated else 0),
        "session": str(session),
        "pixel_format": _meta.get("pixel_format"),
        "scene": _meta.get("scene"),
        "created_at_s": time.time(),
        "scoring_on_fit": score(per_level, thresholds),
        **thresholds,
    }
    if contact_gated:
        ordered_levels = sorted(per_level)
        denominator = len(ordered_levels) - 1
        fitted["continuous_anchors"] = [
            {
                "sum_kpa": float(np.median(per_level[level])),
                "pressure_0_1": index / denominator,
            }
            for index, level in enumerate(ordered_levels)
        ]
        _continuous_anchor_points(fitted)
    return fitted


def write_levels(path: Path, levels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(levels, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_levels(path: Path, *, max_age_s: float | None = None) -> dict:
    path = Path(path)
    levels = json.loads(path.read_text(encoding="utf-8"))
    if max_age_s is not None:
        age_s = max(0.0, time.time() - path.stat().st_mtime)
        if age_s > max_age_s:
            raise SystemExit(
                f"{path}: fitted levels are older by {age_s / 60.0:.1f} minutes; "
                f"maximum is {max_age_s / 60.0:.1f} minutes -- recapture"
            )
    if levels.get("metric") != ANCHOR_METRIC:
        raise SystemExit(f"{path}: fitted on {levels.get('metric')!r}, sender measures {ANCHOR_METRIC!r}")
    if len(levels.get("levels", [])) < 2:
        raise SystemExit(f"{path}: needs at least two levels")
    offset = int(levels.get("wire_level_offset", 0))
    if offset not in (0, 1):
        raise SystemExit(f"{path}: unsupported wire_level_offset {offset}")
    expected_n_levels = len(levels["levels"]) + offset
    n_levels = int(levels.get("n_levels", expected_n_levels))
    if n_levels != expected_n_levels:
        raise SystemExit(
            f"{path}: n_levels {n_levels} does not match {len(levels['levels'])} fitted "
            f"levels plus wire offset {offset}"
        )
    levels["wire_level_offset"] = offset
    levels["n_levels"] = n_levels
    if levels.get("continuous_anchors") is not None:
        _continuous_anchor_points(levels)
    return levels


def level_index(value: float, levels: dict) -> int:
    """Ordinal level for this reading, or ABSTAIN_LEVEL inside an abstain band."""
    predicted = classify(float(value), levels)
    if predicted == ABSTAIN:
        return ABSTAIN_LEVEL
    return int(levels.get("wire_level_offset", 0)) + levels["levels"].index(predicted)


def _continuous_anchor_points(levels: dict) -> tuple[list[float], list[float]]:
    anchors = levels.get("continuous_anchors")
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise SystemExit("continuous_anchors must contain at least two points")
    try:
        raw = [float(anchor["sum_kpa"]) for anchor in anchors]
        normalized = [float(anchor["pressure_0_1"]) for anchor in anchors]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid continuous_anchors: {exc}") from exc
    if not all(np.isfinite(value) for value in raw + normalized):
        raise SystemExit("continuous_anchors must be finite")
    if not all(lower < upper for lower, upper in zip(raw, raw[1:])):
        raise SystemExit("continuous_anchors sum_kpa must be strictly increasing")
    if normalized[0] != 0.0 or normalized[-1] != 1.0:
        raise SystemExit("continuous_anchors pressure_0_1 must span 0..1")
    if not all(lower < upper for lower, upper in zip(normalized, normalized[1:])):
        raise SystemExit("continuous_anchors pressure_0_1 must be strictly increasing")
    return raw, normalized


def continuous_pressure(value: float, levels: dict) -> float:
    """Normalize calibrated light-to-hard support onto 0..1.

    Explicit continuous anchors support the three-point shadow candidate. The
    older fitted-level format falls back to outer gap edges. Values beyond
    either support saturate; right-hand pinch owns contact/no-contact.
    """
    if levels.get("continuous_anchors") is not None:
        raw, normalized = _continuous_anchor_points(levels)
        return float(np.interp(float(value), raw, normalized))
    boundaries = levels.get("boundaries") or []
    if not boundaries:
        raise SystemExit("fitted levels have no boundaries for continuous pressure")
    first, last = boundaries[0], boundaries[-1]
    low = float(first["edge"]) - 0.5 * float(first["gap"])
    high = float(last["edge"]) + 0.5 * float(last["gap"])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise SystemExit(f"invalid continuous pressure anchors: {low:g}, {high:g}")
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def median_scene_fingerprint(samples: list[dict]) -> dict:
    """Reject transient MJPG/brightness outliers without relaxing the move gate."""
    if not samples:
        raise ValueError("scene fingerprint needs at least one sample")
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in ("bright_fraction", "bright_cx", "bright_cy")
    }


def render_live_panel(
    resized: np.ndarray,
    kpa: np.ndarray,
    level: int,
    n_levels: int,
    pressure_0_1: float,
    metrics: dict,
) -> np.ndarray:
    """Side-by-side camera/model view with the continuous signal burned in."""
    camera = resized.copy()
    pressure = overlay(resized, kpa)
    panel = np.hstack((camera, pressure))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    level_text = "ABSTAIN" if level == ABSTAIN_LEVEL else f"L{level}/{n_levels - 1}"
    cv2.putText(
        panel,
        f"{level_text}  p={pressure_0_1:.2f}  sum_kpa={metrics['sum_kpa']:7.0f}  "
        f"mean={metrics['mean_kpa_in_contact']:5.1f}  px={metrics['contact_px']:5d}  "
        f"off=({metrics['off_x']:+.2f},{metrics['off_y']:+.2f})",
        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, "camera crop", (10, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, "PressureVision overlay", (camera.shape[1] + 10, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    parser.add_argument("--crop", default=None, help="x0,y0,x1,y1 in source pixels (default: centred)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda", help="cpu is ~9x slower; debugging only")
    parser.add_argument("--preview", action="store_true", help="Show the pressure overlay.")
    parser.add_argument(
        "--preview-share",
        type=Path,
        default=DEFAULT_PREVIEW_SHARE,
        help="Publish the live PV panel for the hand-track window through this local mmap file.",
    )
    parser.add_argument(
        "--mjpg",
        action="store_true",
        help="Ask the camera for MJPG. The C270 caps uncompressed 1280x720 at 7.5 fps "
        "against 30 for MJPG, but JPEG is lossy and colour carries the blanching cue, "
        "so the calibration has to be recaptured under whichever format is used.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip the exposure/white-balance lock. Only for debugging: it drifts "
        "the sender away from the anchors its kPa are scaled against.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Append every packet to this CSV. For working out why a live run "
        "disagrees with the calibration it was fitted on.",
    )
    parser.add_argument(
        "--log-frames",
        type=Path,
        help="With --log, also save the frame and its pressure overlay whenever the "
        "level changes, plus one a second. Numbers say a band was wrong; only the "
        "picture says what the model was looking at.",
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        help="Record the camera crop, PressureVision overlay, and live metrics to an MJPG AVI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Send synthetic packets without opening the camera or the model.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N frames; 0 = unlimited.")
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Fit press-level boundaries from this labelled capture session and exit.",
    )
    parser.add_argument("--levels-out", type=Path, help="Where --session-dir writes its levels.")
    parser.add_argument(
        "--contact-gated",
        action="store_true",
        help="When fitting, drop target 0 and reserve wire level 0 for an external "
        "contact gate such as MediaPipe pinch.",
    )
    parser.add_argument(
        "--levels",
        type=Path,
        help="Fitted levels to classify against while streaming. Required unless --dry-run.",
    )
    parser.add_argument(
        "--max-level-age-minutes",
        type=float,
        default=120.0,
        help="Refuse fitted levels older than this while streaming (default: 120).",
    )
    parser.add_argument(
        "--require-scene-match",
        action="store_true",
        help="Exit before sending if the live scene or pixel format differs from calibration.",
    )
    parser.add_argument(
        "--abstain-fraction",
        type=float,
        default=DEFAULT_ABSTAIN_FRACTION,
        help="Width of the abstain band around each boundary, as a fraction of the "
        "gap between levels. 0 disables abstaining.",
    )
    args = parser.parse_args(argv)
    if args.port <= 0:
        parser.error("--port must be positive")
    if args.limit < 0:
        parser.error("--limit must not be negative")
    if args.abstain_fraction < 0:
        parser.error("--abstain-fraction must not be negative")
    if args.max_level_age_minutes <= 0:
        parser.error("--max-level-age-minutes must be positive")
    if args.session_dir and not args.session_dir.is_dir():
        parser.error(f"--session-dir is not a directory: {args.session_dir}")
    if args.levels_out and not args.session_dir:
        parser.error("--levels-out requires --session-dir")
    if args.contact_gated and not args.session_dir:
        parser.error("--contact-gated is only valid with --session-dir")
    if args.log_frames and not args.log:
        parser.error("--log-frames requires --log")
    if args.video_out and args.video_out.suffix.lower() != ".avi":
        parser.error("--video-out must end in .avi")
    if args.video_out and (args.session_dir or args.dry_run):
        parser.error("--video-out is only valid for live camera streaming")
    if not args.session_dir and not args.dry_run and not args.levels:
        parser.error("--levels is required to stream; fit one with --session-dir first")
    return args


def run_dry(args, sender: socket.socket, target, counter: list) -> int:
    """Replay the shared test vector at 30 Hz so the consumer can be wired up offline."""
    fields = SHARED_PV_TEST_PACKET.split(",")
    sent = 0
    while True:
        payload = encode_pv_packet(
            sequence=sent + 1,
            source_observed_at_s=time.time(),
            sent_at_s=time.time(),
            mean_kpa_in_contact=float(fields[4]),
            max_kpa=float(fields[5]),
            sum_kpa=float(fields[6]),
            contact_px=int(fields[7]),
            off_x=float(fields[8]),
            off_y=float(fields[9]),
            crop_w=int(fields[10]),
            crop_h=int(fields[11]),
            pressure_0_1=float(fields[12]),
            level=int(fields[13]),
            n_levels=int(fields[14]),
        )
        sender.sendto(payload, target)
        sent = counter[0] = sent + 1
        if args.preview:
            print(f"[dry-run] {payload.decode('utf-8').rstrip()}")
        if args.limit and sent >= args.limit:
            return sent
        time.sleep(1.0 / 30.0)


def run_live(args, sender: socket.socket, target, counter: list) -> int:
    if args.video_out and args.video_out.exists():
        raise SystemExit(f"refusing to overwrite existing recording: {args.video_out}")
    levels = load_levels(args.levels, max_age_s=args.max_level_age_minutes * 60.0)
    n_levels = levels["n_levels"]
    model, config = load_model(args.repo, args.device)
    contact_thresh = float(config.CONTACT_THRESH)
    thresholds = config.FORCE_THRESHOLDS

    # Same open + exposure-lock recipe the aiming and calibration steps use. An
    # unlocked sender drifts in colour, and colour is what carries the blanching
    # cue, so its kPa would wander away from the anchors it is scaled against.
    capture = open_camera(args.camera, mjpg=args.mjpg)
    log = None          # bound before the try, or the finally can raise instead of clean up
    video = None
    preview_writer = SharedPreviewWriter(args.preview_share)
    try:
        if not args.no_lock:
            for _ in range(WARMUP_FRAMES):     # the lock only sticks once frames flow
                capture.read()
            lock_camera(args.camera)
            time.sleep(SETTLE_S)
            for _ in range(SETTLE_FRAMES):
                capture.read()
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"camera {args.camera} opened but returned no frame")
        box = crop_box(frame.shape, args.crop)
        crop_w, crop_h = box[2] - box[0], box[3] - box[1]
        print(
            f"[pv] checking {SCENE_FINGERPRINT_FRAMES} empty-scene frames; "
            "keep hands out of view"
        )
        scene_samples = [scene_fingerprint(frame)]
        for _ in range(SCENE_FINGERPRINT_FRAMES - 1):
            ok, frame = capture.read()
            if not ok:
                raise SystemExit(f"camera {args.camera} returned no frame during scene check")
            scene_samples.append(scene_fingerprint(frame))
        drift = fingerprint_drift(
            median_scene_fingerprint(scene_samples), levels.get("scene") or {}
        )
        if drift:
            if args.require_scene_match:
                raise SystemExit(
                    f"{args.levels}: scene mismatch ({drift}); recapture before streaming"
                )
            print(f"[pv] WARNING: the rig has moved since calibration -- {drift}.")
            print("[pv]          The crop, the boundaries and the response falloff are all "
                  "tied to where the pad sits in frame. Re-aim and recapture.")
        fourcc = pixel_format(capture)
        fitted_on = levels.get("pixel_format")
        if fitted_on and fitted_on != fourcc:
            if args.require_scene_match:
                raise SystemExit(
                    f"{args.levels}: calibrated on {fitted_on}, live camera is {fourcc}; "
                    "recapture under the live format"
                )
            print(
                f"[pv] WARNING: levels were fitted on {fitted_on} frames but the camera "
                f"is delivering {fourcc}. Colour carries the blanching cue, so the "
                "boundaries may not hold -- recapture the calibration under this format."
            )
        print(
            f"[pv] crop={box} -> {crop_w}x{crop_h}, {fourcc}, {n_levels} levels, "
            f"sending to {target[0]}:{target[1]}  (ctrl-c to stop)"
        )

        if args.log_frames:
            args.log_frames.mkdir(parents=True, exist_ok=True)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("w", encoding="utf-8")
            log.write(
                "sequence,source_t,sent_t,mean_kpa_in_contact,max_kpa,sum_kpa,contact_px,off_x,off_y,"
                "pressure_0_1,level\n"
            )
        if args.video_out:
            args.video_out.parent.mkdir(parents=True, exist_ok=True)
        if args.preview:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        sent = 0
        last_logged_level, last_frame_saved_s = None, 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            source_observed_at_s = time.time()
            resized, tensor = preprocess(frame, box)
            with torch.no_grad():
                logits = model(tensor.to(args.device))
            kpa = to_kpa(logits, thresholds)
            metrics = reduce_frame(kpa, contact_thresh)
            level = (
                0
                if metrics["contact_px"] == 0
                else level_index(metrics[ANCHOR_METRIC], levels)
            )
            pressure_0_1 = (
                0.0
                if metrics["contact_px"] == 0
                else continuous_pressure(metrics[ANCHOR_METRIC], levels)
            )
            sent_at_s = time.time()
            sequence = sent + 1
            sender.sendto(
                encode_pv_packet(
                    sequence=sequence,
                    source_observed_at_s=source_observed_at_s,
                    sent_at_s=sent_at_s,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    pressure_0_1=pressure_0_1,
                    level=level,
                    n_levels=n_levels,
                    **metrics,
                ),
                target,
            )
            sent = counter[0] = sent + 1
            if args.log_frames is not None and (level != last_logged_level or
                                                time.time() - last_frame_saved_s >= 1.0):
                stem = args.log_frames / f"{sent:05d}_L{level}_px{metrics['contact_px']}"
                cv2.imwrite(f"{stem}_full.png", frame)
                cv2.imwrite(f"{stem}_pv.png", overlay(resized, kpa))
                last_logged_level, last_frame_saved_s = level, time.time()
            if log is not None:
                log.write(
                    f"{sequence},{source_observed_at_s:.6f},{sent_at_s:.6f},"
                    f"{metrics['mean_kpa_in_contact']:.3f},"
                    f"{metrics['max_kpa']:.3f},{metrics['sum_kpa']:.1f},{metrics['contact_px']},"
                    f"{metrics['off_x']:.4f},{metrics['off_y']:.4f},"
                    f"{pressure_0_1:.6f},{level}\n"
                )
                log.flush()

            panel = render_live_panel(
                resized, kpa, level, n_levels, pressure_0_1, metrics
            )
            preview_writer.publish(panel)
            if args.video_out:
                if video is None:
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    if not np.isfinite(fps) or fps <= 0:
                        fps = 30.0
                    video = cv2.VideoWriter(
                        str(args.video_out),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        fps,
                        (panel.shape[1], panel.shape[0]),
                    )
                    if not video.isOpened():
                        raise SystemExit(f"could not open video writer: {args.video_out}")
                    print(
                        f"[pv] recording {panel.shape[1]}x{panel.shape[0]} "
                        f"at {fps:.1f} fps to {args.video_out}"
                    )
                video.write(panel)

            if args.preview:
                # A window closed with the title-bar X leaves imshow drawing into
                # nothing, so the sender streams on with no way to stop it short of
                # killing the terminal -- which is how it ends up still holding the
                # camera for the next run.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    print("[pv] preview window closed")
                    return sent
                cv2.imshow(WINDOW, panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return sent
            if args.limit and sent >= args.limit:
                return sent
    finally:
        if log is not None:
            log.close()
        if video is not None:
            video.release()
        preview_writer.close()
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.session_dir:
        levels = fit_levels(
            args.session_dir,
            args.abstain_fraction,
            contact_gated=True if args.contact_gated else None,
        )
        for boundary in levels["boundaries"]:
            print(
                f"[levels] {boundary['below_g']} | {boundary['above_g']}  "
                f"edge={boundary['edge']:.1f} {ANCHOR_METRIC}  abstain "
                f"[{boundary['abstain_lo']:.2f}, {boundary['abstain_hi']:.2f}]  "
                f"{'separated' if boundary['separated_in_fit'] else 'OVERLAPPING'}"
            )
        fit = levels["scoring_on_fit"]
        print(
            f"[levels] on its own fit trials: {fit['correct']}/{fit['decided']} decided correctly, "
            f"{fit['abstained']}/{fit['trials']} abstained "
            "(optimistic -- this is the data it was fitted on)"
        )
        if args.levels_out:
            write_levels(args.levels_out, levels)
            print(f"[levels] wrote {args.levels_out}")
        return 0

    # SIGTERM skips finally blocks, so a `kill` would leave the camera open and
    # the next run waiting on a device nothing appears to hold. Turning it into
    # the same exception Ctrl-C raises lets the existing cleanup run.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)
    # Counted through a cell so an interrupt still reports the real total; the
    # return value is lost when the exception unwinds past it.
    counter = [0]
    try:
        if args.dry_run:
            run_dry(args, sender, target, counter)
        else:
            run_live(args, sender, target, counter)
    except KeyboardInterrupt:
        print("\n[pv] interrupted")
    finally:
        sender.close()
    print(f"[pv] sent {counter[0]} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
