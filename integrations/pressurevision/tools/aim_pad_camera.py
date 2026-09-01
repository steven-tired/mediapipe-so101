"""Aim a plain USB camera at the pressure pad and settle on a crop.

The rig moved to an overhead Logitech on a fixed mount, so the crop that the
D435i sittings used no longer describes anything. This is the loop for deriving
a new one: it draws the crop over the live feed, shows what the network would
actually receive, and — with `--model` — what the network says about it.

Watch the response, not the box. `pressurevision_probe.py` records why: on
near-horizontal desk frames the released model reported contact on 2 of 123
images, while on the paper's own demo frame the same code lights every
fingertip. Viewpoint is the variable that decides whether this works at all, so
aim by pressing and watching rather than by measuring the frame.

What the crop should look like, from the paper's own convention: images are
"cropped to include a 50-pixel border around the pressure sensor, and are
resized to 480x384". The sensor is 230x130 mm, aspect 1.77, so their crops were
squashed into 480x384 rather than matched to its 1.25. **Hug the pad; do not
chase an aspect ratio.** What has to be reproduced is the pad filling the crop.

Resolution is not the constraint. Their Table 2 loses only 4% of Volumetric IoU
going from 480x384 down to 120x96, while dropping colour at full resolution
costs more than twice that — so a crop far smaller than the frame is fine, and
white balance matters more than pixels. Lock the camera before trusting
anything seen here:

  v4l2-ctl -d /dev/video2 -c white_balance_automatic=0 \\
      -c white_balance_temperature=4000 -c auto_exposure=1

Keys:
  arrows      move the crop
  a/d w/s     widen/narrow, taller/shorter
  +/-         scale both
  g           snap the crop to the suggested one (pad plus a border)
  p           cycle the position label 1..9, for mapping where on the pad the
              model responds -- the cast shadow it relies on falls differently
              across the surface, so response is not expected to be uniform
  0 1 2       snapshot the current press as no-contact / light / hard,
              into --snapshots, so someone not at the desk can look at the
              crop and at whether the levels actually separate
  space       print the crop as a --crop argument
  return      save it to --out and quit
  q           quit without saving

Run:
  cd integrations/pressurevision
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb ../.venv-pressurevision/bin/python \\
      scripts/aim_pad_camera.py --model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import os

# Set before cv2 loads Qt: five lines of font warnings per start otherwise, and
# the message that actually says what went wrong scrolls away behind them.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pressurevision_probe import (  # noqa: E402
    NETWORK_SIZE,
    load_model,
    overlay,
    preprocess,
    to_kpa,
)

DEFAULT_CAMERA_INDEX = 2          # the overhead Logitech
DEFAULT_SIZE = (1280, 720)
WINDOW = "aim pad camera: source | network input"
MOVE_PX = 10
RESIZE_PX = 10
MIN_SIDE_PX = 80

# Measured on this rig with the pad filling the frame: the clipping cliff is
# steep -- at gain 64 the blown fraction goes from 0% at exposure 80 to 60% at
# 120 -- and a blown frame carries no pressure cue at all, so the model reads
# zero. gain 32 / exposure 60 lands at mean 163, p99 196, nothing clipped.
CAMERA_LOCK = {
    "auto_exposure": 1,             # manual
    "gain": 32,
    "exposure_time_absolute": 60,
    "white_balance_automatic": 0,
    "white_balance_temperature": 4000,
}
# These have to be applied once frames are already FLOWING. Setting them before
# opening the device does not stick, and neither does setting them straight
# after VideoCapture() -- streaming has to have started. In both of those cases
# v4l2-ctl still reports the values just written, so the settings look applied
# while the image stays blown, which is a combination that will mislead anyone
# who checks the controls rather than the pixels. Measured on this rig:
# locking before any read left 58% of pixels clipped; the identical lock after
# 25 frames had flowed left 0%.
# A previous run's file descriptor can outlive the process briefly, so opening
# straight after one exits fails for a second or two.
OPEN_ATTEMPTS = 10
OPEN_RETRY_S = 0.6
WARMUP_FRAMES = 25
SETTLE_S = 1.0
SETTLE_FRAMES = 30


# Coarse enough to ignore a hand entering the frame, tight enough to catch a rig
# that was picked up and put back down. The move that voided session 06 was 13
# points of area and 78 px.
MAX_FRACTION_DRIFT = 0.06
MAX_CENTROID_DRIFT_PX = 40.0


def scene_fingerprint(bgr) -> dict:
    """Coarse description of what the camera is looking at.

    Not a calibration check -- a moved-rig check. The crop, the fitted boundaries
    and the response falloff are all tied to where the pad sits in frame, and
    nothing noticed when it shifted: an afternoon went into chasing compression,
    lighting and the choice of metric before a side-by-side of two frames showed
    the pad had grown from 49% to 62% of the view and slid 78 px right. These
    three numbers caught it, so these three numbers are what get stored.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = grey > (float(grey.max()) * 0.75)
    ys, xs = np.nonzero(bright)
    if xs.size == 0:
        return {"bright_fraction": 0.0, "bright_cx": 0.0, "bright_cy": 0.0}
    return {
        "bright_fraction": float(bright.mean()),
        "bright_cx": float(xs.mean()),
        "bright_cy": float(ys.mean()),
    }


def fingerprint_drift(now: dict, then: dict) -> str | None:
    """Return a complaint if the scene has moved enough to void the boundaries."""
    if not then:
        return None
    d_frac = abs(now["bright_fraction"] - then["bright_fraction"])
    d_px = float(np.hypot(now["bright_cx"] - then["bright_cx"], now["bright_cy"] - then["bright_cy"]))
    if d_frac <= MAX_FRACTION_DRIFT and d_px <= MAX_CENTROID_DRIFT_PX:
        return None
    return (
        f"the pad covers {now['bright_fraction'] * 100:.0f}% of the view against "
        f"{then['bright_fraction'] * 100:.0f}% at calibration, and its centre has moved "
        f"{d_px:.0f} px"
    )


# What fraction of the crop the pad should cover. PressureVision's own convention is
# "cropped to include a 50-pixel border around the pressure sensor", and the numbers
# agree: session 06 calibrated cleanly with the pad at 85% of the crop, while a later
# run framed it at 100% -- no border at all -- and the same held presses read a
# different band. Tightening a working crop by 10% moved light presses into the hard
# band; by 40% it read no contact at all.
PAD_FILL_TARGET = 0.85
PAD_FILL_MIN = 0.70
PAD_FILL_MAX = 0.93
# A relative threshold alone calls any uniform view a pad, including a dark one --
# paper under working light peaks well above this.
MIN_PAD_PEAK = 120


def detect_pad(frame) -> tuple[int, int, int, int] | None:
    """Bounding box of the bright region, i.e. the sheet of paper."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    peak = float(grey.max())
    if peak < MIN_PAD_PEAK:
        return None
    bright = (grey > peak * 0.75).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, x + w, y + h)


def suggested_crop(pad, frame_shape, fill=PAD_FILL_TARGET):
    """A crop holding the pad with a border, since the model was trained with one."""
    if pad is None:
        return None
    x0, y0, x1, y1 = pad
    margin = (1.0 / fill) ** 0.5 - 1.0        # grow both sides so area ratio lands on fill
    dx, dy = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
    return clamp_crop((int(x0 - dx), int(y0 - dy), int(x1 + dx), int(y1 + dy)), frame_shape)


def pad_fill_verdict(frame, crop) -> tuple[float, str]:
    """What share of the crop's pixels are pad, and whether that leaves a border.

    Measured per pixel, not from the pad's bounding box: the sheet is larger than
    the crop, so any crop placed inside it intersects the box completely and every
    framing would score 100%. What has to be seen is whether the crop still
    contains some pad edge. Session 06 calibrated cleanly at 85%; a later run at
    100% -- all paper, no edge in view -- read a different band for the same press.
    """
    x0, y0, x1, y1 = crop
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, "NO PAD FOUND"
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    peak = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).max())
    fill = float((grey > peak * 0.75).mean())
    if fill > PAD_FILL_MAX:
        return fill, "TOO TIGHT -- leave a border (press g)"
    if fill < PAD_FILL_MIN:
        return fill, "TOO LOOSE -- pad should fill most of it (press g)"
    return fill, "GOOD"


def framing_advice(now: dict, then: dict) -> list[str]:
    """Which way to move the camera to get back to the calibrated framing.

    The model's reading is dominated by framing: on one set of held light presses,
    tightening the crop 10% flipped them from the light band to the hard band, and
    tightening it 40% read no contact at all. So a calibration only describes the
    framing it was made in, and getting back to that framing beats recapturing.
    """
    d_frac = now["bright_fraction"] - then["bright_fraction"]
    dx = now["bright_cx"] - then["bright_cx"]
    dy = now["bright_cy"] - then["bright_cy"]
    lines = [
        f"pad fills {now['bright_fraction'] * 100:.0f}% vs {then['bright_fraction'] * 100:.0f}% "
        f"calibrated  ({d_frac * 100:+.0f} pts)",
        f"centre off by ({dx:+.0f}, {dy:+.0f}) px",
    ]
    moves = []
    if abs(d_frac) > MAX_FRACTION_DRIFT:
        moves.append("PULL BACK" if d_frac > 0 else "MOVE CLOSER")
    if abs(dx) > MAX_CENTROID_DRIFT_PX:
        moves.append("SLIDE LEFT" if dx > 0 else "SLIDE RIGHT")
    if abs(dy) > MAX_CENTROID_DRIFT_PX:
        moves.append("SLIDE UP" if dy > 0 else "SLIDE DOWN")
    lines.append("MATCHED" if not moves else "  ".join(moves))
    return lines


def load_reference_scene(path: Path) -> dict:
    """Read the stored framing from a levels file or a capture session."""
    if path.is_dir():
        path = path / "capture.jsonl"
    text = path.read_text(encoding="utf-8")
    data = json.loads(text.splitlines()[0] if path.name.endswith(".jsonl") else text)
    scene = data.get("scene")
    if not scene:
        raise SystemExit(f"{path} carries no scene fingerprint (recorded before they existed)")
    return scene


def clamp_crop(crop, frame_shape) -> tuple[int, int, int, int]:
    """Keep the crop inside the frame and above a usable size."""
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = crop
    x1 = min(max(x1, x0 + MIN_SIDE_PX), width)
    y1 = min(max(y1, y0 + MIN_SIDE_PX), height)
    x0 = max(0, min(x0, x1 - MIN_SIDE_PX))
    y0 = max(0, min(y0, y1 - MIN_SIDE_PX))
    return int(x0), int(y0), int(x1), int(y1)


def _move(crop, dx, dy, frame_shape):
    """Translate the crop, stopping at the frame edge.

    Clamping the resulting box instead would shrink the crop against a wall:
    the near edge stops at zero while the far edge keeps travelling, so holding
    an arrow key silently shaved the box down.
    """
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = crop
    dx = min(max(dx, -x0), width - x1)
    dy = min(max(dy, -y0), height - y1)
    return int(x0 + dx), int(y0 + dy), int(x1 + dx), int(y1 + dy)


def apply_key(key: int, crop, frame_shape):
    """Move or resize the crop. Returns the new crop, unchanged if `key` is not
    one of ours, so the caller can treat every other key as its own."""
    x0, y0, x1, y1 = crop
    if key in (81, ord("h")):                      # left
        return _move(crop, -MOVE_PX, 0, frame_shape)
    if key in (83, ord("l")):                      # right
        return _move(crop, +MOVE_PX, 0, frame_shape)
    if key in (82, ord("k")):                      # up
        return _move(crop, 0, -MOVE_PX, frame_shape)
    if key in (84, ord("j")):                      # down
        return _move(crop, 0, +MOVE_PX, frame_shape)
    if key == ord("d"):
        x1 += RESIZE_PX
    elif key == ord("a"):
        x1 -= RESIZE_PX
    elif key == ord("s"):
        y1 += RESIZE_PX
    elif key == ord("w"):
        y1 -= RESIZE_PX
    elif key in (ord("+"), ord("=")):
        x1, y1 = x1 + RESIZE_PX, y1 + RESIZE_PX
    elif key == ord("-"):
        x1, y1 = x1 - RESIZE_PX, y1 - RESIZE_PX
    else:
        return crop
    return clamp_crop((x0, y0, x1, y1), frame_shape)


def crop_report(crop, frame_shape) -> dict:
    """The numbers worth reading while aiming."""
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    net_w, net_h = NETWORK_SIZE
    return {
        "crop": [x0, y0, x1, y1],
        "size": [w, h],
        "aspect": w / h if h else float("nan"),
        # Below 1.0 the crop is being upsampled into the network, which buys
        # nothing; well above it, detail is being thrown away.
        "source_px_per_net_px": [w / net_w, h / net_h],
        "frame_fraction": (w * h) / (frame_shape[0] * frame_shape[1]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument(
        "--crop",
        default=None,
        help="x0,y0,x1,y1 to start from; default is a centred box",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="where return writes the chosen crop")
    parser.add_argument(
        "--model",
        action="store_true",
        help="also run PressureVisionNet and overlay its output — this is what "
        "the crop should actually be judged on",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--match",
        type=Path,
        default=None,
        metavar="LEVELS_OR_SESSION",
        help="Show live how far the framing is from the one a calibration was made in, "
        "and which way to move the camera. Restoring the framing beats recapturing: "
        "the reading is dominated by it.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path("/tmp/pad_snapshots"),
        help="where the 0/1/2 keys write frames and metrics",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="leave the camera on auto exposure and white balance",
    )
    return parser.parse_args(argv)


def camera_holder(index: int) -> str | None:
    """Who has /dev/videoN open, if anyone. Best effort."""
    try:
        out = subprocess.run(["fuser", f"/dev/video{index}"], capture_output=True,
                             text=True, timeout=5).stdout.split()
    except Exception:
        return None
    for pid in out:
        try:
            cmd = pathlib_read(f"/proc/{pid}/cmdline")
            return f"pid {pid}: {cmd}"
        except Exception:
            return f"pid {pid}"
    return None


def pathlib_read(path: str) -> str:
    return Path(path).read_bytes().decode(errors="replace").replace("\x00", " ").strip()


def pixel_format(capture) -> str:
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()


def open_camera(index: int, size=DEFAULT_SIZE, attempts=OPEN_ATTEMPTS, mjpg=False):
    """Open the camera, waiting out a previous run that has not let go yet.

    A device released microseconds ago still refuses the next open, and the
    resulting failure is a single line behind a wall of Qt warnings — which
    reads as the model having crashed rather than as the camera being busy.
    """
    for attempt in range(attempts):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            if mjpg:
                # The C270 caps uncompressed 1280x720 at 7.5 fps -- a USB2 bandwidth
                # limit, not a driver setting -- against 30 fps for MJPG. Set before
                # the size, or the driver keeps the format it already negotiated.
                # JPEG is lossy and colour is what carries the blanching cue, so a
                # calibration fitted under one format does not transfer to the other.
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            return capture
        capture.release()
        if attempt == 0:
            holder = camera_holder(index)
            print(f"/dev/video{index} is busy"
                  + (f" — {holder}" if holder else "")
                  + f"; retrying for {attempts * OPEN_RETRY_S:.0f}s", flush=True)
        time.sleep(OPEN_RETRY_S)
    holder = camera_holder(index)
    raise SystemExit(
        f"could not open /dev/video{index} after {attempts} attempts"
        + (f" — still held by {holder}" if holder else
           " — nothing appears to hold it; try replugging")
    )


def lock_camera(index: int, settings=CAMERA_LOCK) -> None:
    """Pin exposure, gain and white balance, with the device already open.

    Auto white balance drifts with the scene, and colour is what carries the
    blanching cue -- the paper's own ablation loses more Volumetric IoU
    dropping to monochrome than dropping resolution by 16x.
    """
    argv = ["v4l2-ctl", "-d", f"/dev/video{index}"]
    for name, value in settings.items():
        argv += ["-c", f"{name}={value}"]
    subprocess.run(argv, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(args) -> int:
    capture = open_camera(args.camera)
    if not args.no_lock:
        for _ in range(WARMUP_FRAMES):     # start streaming; see CAMERA_LOCK
            capture.read()
        lock_camera(args.camera)
        time.sleep(SETTLE_S)
        for _ in range(SETTLE_FRAMES):
            capture.read()

    model = thresholds = contact_thresh = None
    if args.model:
        import torch  # noqa: F401  (imported for the no_grad context below)

        model, config = load_model(args.repo, args.device)
        thresholds = config.FORCE_THRESHOLDS
        contact_thresh = float(config.CONTACT_THRESH)

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("camera opened but returned no frame")
    if args.crop:
        crop = clamp_crop(tuple(int(v) for v in args.crop.split(",")), frame.shape)
    elif args.out and args.out.exists():
        # Resuming where the last session left off. Without this, quitting
        # without pressing return silently reverts to a centred box that does
        # not frame the pad, and the model reads zero for a reason that looks
        # like the model having stopped working.
        crop = clamp_crop(json.loads(args.out.read_text())["crop"], frame.shape)
        print(f"resumed crop {crop} from {args.out}")
    else:
        h, w = frame.shape[:2]
        crop = clamp_crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4), frame.shape)

    reference = load_reference_scene(args.match) if args.match else None
    if reference is not None:
        print(f"matching framing from {args.match}: "
              f"pad {reference['bright_fraction'] * 100:.0f}% of view, "
              f"centre ({reference['bright_cx']:.0f},{reference['bright_cy']:.0f})")

    args.snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_log = (args.snapshots / "snapshots.jsonl").open("a", encoding="utf-8")
    counts = {"none": 0, "light": 0, "hard": 0}
    position = 1

    saved = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            x0, y0, x1, y1 = crop
            report = crop_report(crop, frame.shape)
            pad = detect_pad(frame)
            fill, verdict = pad_fill_verdict(frame, crop)

            patch = frame[y0:y1, x0:x1]
            if model is not None:
                import torch

                resized, tensor = preprocess(frame, crop)
                with torch.no_grad():
                    kpa = to_kpa(model(tensor.to(args.device)), thresholds)
                panel = overlay(resized, kpa)
                contact_px = int((kpa >= contact_thresh).sum())
                status = (f"contact {contact_px} px   max {kpa.max():5.1f} kPa"
                          f"   mean_in_contact "
                          f"{kpa[kpa >= contact_thresh].mean() if contact_px else 0:5.1f}")
            else:
                panel = cv2.resize(patch, NETWORK_SIZE, interpolation=cv2.INTER_AREA)
                status = "no model: --model shows what the network makes of this"

            view = frame.copy()
            if pad is not None:
                # Blue: the pad as found. Yellow dashed-ish: where the crop should sit.
                cv2.rectangle(view, pad[:2], pad[2:], (255, 160, 0), 1)
                want = suggested_crop(pad, frame.shape)
                if want is not None:
                    cv2.rectangle(view, want[:2], want[2:], (0, 220, 220), 1)
                    cv2.putText(view, "suggested", (want[0] + 4, want[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            blown = 100.0 * float((grey >= 254).mean())
            lines = [
                f"blown {blown:5.2f}%   position {position}"
                + ("   <-- no cue survives this" if blown > 1.0 else ""),
                f"crop {x0},{y0},{x1},{y1}   {report['size'][0]}x{report['size'][1]}"
                f"   aspect {report['aspect']:.2f}",
                f"source px per net px  {report['source_px_per_net_px'][0]:.2f}"
                f" x {report['source_px_per_net_px'][1]:.2f}",
                f"pad fills {fill * 100:4.1f}% of crop   {verdict}",
                status,
            ]
            match_lines = []
            if reference is not None:
                match_lines = framing_advice(scene_fingerprint(frame), reference)
            for index, text in enumerate(lines):
                cv2.putText(view, text, (8, 26 + 26 * index),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for index, text in enumerate(match_lines):
                # Red until the framing is back, since a mismatched frame reads a
                # different pressure for the same press.
                matched = text == "MATCHED"
                cv2.putText(view, text, (8, 26 + 26 * (len(lines) + index)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75 if index == len(match_lines) - 1 else 0.6,
                            (0, 255, 0) if matched else (0, 80, 255), 2)

            side = cv2.resize(panel, (view.shape[0] * panel.shape[1] // panel.shape[0],
                                      view.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW, np.hstack([view, side]))

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == ord("g"):
                want = suggested_crop(detect_pad(frame), frame.shape)
                if want is not None:
                    crop = want
                    print(f"snapped to suggested crop {crop}")
                continue
            if key == ord("p"):
                position = position % 9 + 1
                continue
            if key in (ord("0"), ord("1"), ord("2")):
                label = {"0": "none", "1": "light", "2": "hard"}[chr(key)]
                index = counts[label]
                counts[label] += 1
                stem = f"p{position}_{label}_{index:02d}"
                cv2.imwrite(str(args.snapshots / f"{stem}_source.png"), view)
                cv2.imwrite(str(args.snapshots / f"{stem}_net.png"), panel)
                row = dict(report, label=label, index=index,
                           position=position, blown_percent=blown)
                if model is not None:
                    contact = kpa >= contact_thresh
                    if contact.any():
                        ys, xs = contact.nonzero()
                        row["contact_centre"] = [float(xs.mean()), float(ys.mean())]
                    row.update(
                        contact_px=int(contact.sum()),
                        sum_kpa=float(kpa.sum()),
                        max_kpa=float(kpa.max()),
                        mean_kpa_in_contact=(
                            float(kpa[contact].mean()) if contact.any() else 0.0
                        ),
                    )
                snapshot_log.write(json.dumps(row, sort_keys=True) + "\n")
                snapshot_log.flush()
                print(f"{stem}  {row.get('mean_kpa_in_contact', float('nan')):.1f} kPa"
                      f"  {row.get('contact_px', 0)} px", flush=True)
                continue
            if key == ord(" "):
                print(f"--crop {x0},{y0},{x1},{y1}", flush=True)
                continue
            if key in (13, 10):
                saved = report
                break
            crop = apply_key(key, crop, frame.shape)
    finally:
        snapshot_log.close()
        capture.release()
        cv2.destroyAllWindows()

    # Always write the crop that was actually in use, however the loop ended.
    # The one that gets lost is the one nobody remembered to press return on.
    final = saved if saved is not None else crop_report(crop, frame.shape)
    x0, y0, x1, y1 = final["crop"]
    print(f"--crop {x0},{y0},{x1},{y1}")
    if args.out:
        args.out.write_text(json.dumps(final, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"saved -> {args.out}"
              + ("" if saved is not None else "  (on exit, not via return)"))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
