#!/usr/bin/env python3
"""Run pretrained PressureVisionNet over our own RGB frames.

A feasibility probe, not an experiment. It answers one question: does the
released model produce a pressure response on images from our rig at all, and
does that response track contact?

PressureVisionNet was trained on a fixed capture rig -- four OptiTrack 1080p
cameras looking at a white vinyl or wood-textured planar sensor, under five
controlled lighting conditions -- and the authors recommend a camera about
60 cm above a white table angled 45 degrees down. Our D435i frames are a
near-horizontal view of a wooden desk, so a large domain gap is expected and a
null result here does not condemn the approach; it prices the gap.

Outputs are in image space, so the totals are arbitrary units rather than
newtons: the released kPa-to-newton conversion assumes the 1.25 mm sensel
pitch, which does not apply to our pixels. Separability is what matters here,
not calibration.

Robot-free and read-only: reads PNGs, writes a CSV and figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch


NETWORK_SIZE = (480, 384)  # width, height
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="PressureVision checkout holding config/ and data/model/",
    )
    parser.add_argument("--frames", type=Path)
    parser.add_argument(
        "--live",
        action="store_true",
        help="stream the D435i colour image instead of reading PNGs, for aiming",
    )
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--crop",
        default=None,
        help="x0,y0,x1,y1 in source pixels; default is a centred 480:384 crop",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--montage",
        type=int,
        default=8,
        help="how many evenly spaced frames to render side by side",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


PV_REPO_ENV = "SO101_PV_REPO"


def resolve_repo(repo: Path | None) -> Path:
    """Locate the released PressureVision checkout holding config/ and weights.

    The checkout is an external upstream clone, not vendored here, so this repo
    carries no default path to it -- the migration dropped the old hardcoded
    /home/... default and nothing replaced it, which left every program that
    loads the network dying on `None / "config/paper.yml"` at startup. An
    explicit --repo wins; otherwise the environment names it.
    """
    if repo is not None:
        return Path(repo)
    from os import environ

    named = environ.get(PV_REPO_ENV)
    if not named:
        raise SystemExit(
            f"no PressureVision checkout: pass --repo or set {PV_REPO_ENV} to the "
            "clone holding config/paper.yml and data/model/paper_59.pt"
        )
    return Path(named)


def load_model(repo: Path | None, device: str):
    """Build the released architecture directly.

    `prediction.model_builder.build_model` imports the dataset loader, which
    needs PressureVisionDB on disk; the probe only needs the weights.
    """
    repo = resolve_repo(repo)
    if not (repo / "config/paper.yml").is_file():
        raise SystemExit(f"{repo}: not a PressureVision checkout (no config/paper.yml)")
    sys.path.insert(0, str(repo))
    import segmentation_models_pytorch as smp
    import yaml

    # Their load_config resolves './config' relative to the working directory,
    # so read the released config directly instead of chdir-ing into the repo.
    with (repo / "config/paper.yml").open(encoding="utf-8") as handle:
        config = SimpleNamespace(**yaml.safe_load(handle))
    checkpoint = repo / "data/model/paper_59.pt"
    if not checkpoint.is_file():
        raise SystemExit(
            f"missing {checkpoint}; run recording.downloader.download_model_checkpoint"
        )
    model = smp.FPN(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,
        classes=config.NUM_FORCE_CLASSES,
        activation=None,
        in_channels=config.NETWORK_INPUT_CHANNELS,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, config


def crop_box(shape, crop):
    height, width = shape[:2]
    if crop is not None:
        x0, y0, x1, y1 = (int(value) for value in crop.split(","))
        return x0, y0, x1, y1
    aspect = NETWORK_SIZE[0] / NETWORK_SIZE[1]
    box_width = min(width, int(height * aspect))
    box_height = int(box_width / aspect)
    x0 = (width - box_width) // 2
    y0 = (height - box_height) // 2
    return x0, y0, x0 + box_width, y0 + box_height


def preprocess(bgr, box):
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1]
    resized = cv2.resize(crop, NETWORK_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.tensor(
        normalised.transpose(2, 0, 1).astype("float32")
    ).unsqueeze(0)
    return resized, tensor


def to_kpa(logits, thresholds):
    """Class argmax to the lower edge of each log-spaced pressure bin."""
    classes = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    table = np.asarray(thresholds, dtype=np.float32)
    return table[np.clip(classes, 0, len(table) - 1)]


def overlay(resized_bgr, kpa, max_vis=20.0):
    scaled = np.clip(kpa * (255.0 / max_vis), 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    mask = (kpa > 0)[..., None]
    return np.where(mask, cv2.addWeighted(resized_bgr, 0.35, colour, 0.65, 0), resized_bgr)


WINDOW = "PressureVision probe: input | estimated pressure"


def run_live(args) -> int:
    """Aim the camera against live model response.

    The released model is sensitive to viewpoint: on our near-horizontal desk
    frames it reported contact on 2 of 123 images, while on the paper's own
    demo frame the same code lights up every fingertip. The authors recommend
    roughly 60 cm above a white table at 45 degrees down, so this loop exists
    to find that geometry by watching the response rather than by recording
    first and analysing later.
    """
    import pyrealsense2 as rs

    model, config = load_model(args.repo, args.device)
    thresholds = config.FORCE_THRESHOLDS
    contact_thresh = float(config.CONTACT_THRESH)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(rs_config)
    intrinsics = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    width, height = intrinsics.width, intrinsics.height
    aspect = NETWORK_SIZE[0] / NETWORK_SIZE[1]

    cv2.namedWindow(WINDOW)
    state = {
        "cx": width // 2,
        "cy": height // 2,
        "half_w": min(width, int(height * aspect)) // 2,
    }
    for name, key, maximum in (
        ("centre x", "cx", width),
        ("centre y", "cy", height),
        ("half width", "half_w", width // 2),
    ):
        cv2.createTrackbar(
            name,
            WINDOW,
            state[key],
            maximum,
            lambda value, key=key: state.__setitem__(key, max(value, 16)),
        )

    saved = 0
    peak_kpa = 0.0
    peak_contact = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            colour = frames.get_color_frame()
            if not colour:
                continue
            bgr = np.asanyarray(colour.get_data())
            half_w = state["half_w"]
            half_h = int(half_w / aspect)
            x0 = max(0, state["cx"] - half_w)
            x1 = min(width, state["cx"] + half_w)
            y0 = max(0, state["cy"] - half_h)
            y1 = min(height, state["cy"] + half_h)
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            resized, tensor = preprocess(bgr, (x0, y0, x1, y1))
            with torch.no_grad():
                logits = model(tensor.to(args.device))
            kpa = to_kpa(logits, thresholds)
            contact = kpa >= contact_thresh
            contact_px = int(contact.sum())
            peak_kpa = max(peak_kpa, float(kpa.max()))
            peak_contact = max(peak_contact, contact_px)
            painted = overlay(resized, kpa)

            # A view of the whole sensor with the crop drawn on it, because the
            # trackbars are otherwise blind: the first aiming session framed the
            # hand at the very bottom edge, clipped, over a quarter of the crop.
            guide = cv2.resize(bgr, NETWORK_SIZE, interpolation=cv2.INTER_AREA)
            scale_x = NETWORK_SIZE[0] / width
            scale_y = NETWORK_SIZE[1] / height
            cv2.rectangle(
                guide,
                (int(x0 * scale_x), int(y0 * scale_y)),
                (int(x1 * scale_x), int(y1 * scale_y)),
                (0, 255, 255),
                2,
            )
            for index, line in enumerate(
                (
                    f"contact px {contact_px}   peak {peak_contact}",
                    f"max {kpa.max():.1f} kPa   peak {peak_kpa:.1f}   "
                    f"sum {kpa.sum():.0f}",
                    f"crop {x0},{y0},{x1},{y1}",
                    "s: save   r: reset peak   q: quit",
                )
            ):
                cv2.putText(
                    painted,
                    line,
                    (12, 26 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            display = np.hstack([guide, resized, painted])
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                peak_kpa, peak_contact = 0.0, 0
            if key in (ord("s"), ord("S")) and args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out / f"live_{saved:03d}.png"), display)
                print(
                    f"saved live_{saved:03d}.png  crop {(x0, y0, x1, y1)}  "
                    f"contact {contact_px} px  max {kpa.max():.1f} kPa"
                )
                saved += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.live:
        return run_live(args)
    if args.frames is None or args.out is None:
        raise SystemExit("--frames and --out are required unless --live is set")
    paths = sorted(args.frames.glob(args.glob))[:: args.stride]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no frames matched {args.frames}/{args.glob}")

    model, config = load_model(args.repo, args.device)
    thresholds = config.FORCE_THRESHOLDS
    contact_thresh = float(config.CONTACT_THRESH)
    args.out.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(paths[0]))
    box = crop_box(first.shape, args.crop)
    print(f"{len(paths)} frames, crop {box}, contact threshold {contact_thresh} kPa")

    rows = []
    montage_at = set(
        np.linspace(0, len(paths) - 1, min(args.montage, len(paths))).astype(int)
    )
    panels = []
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        resized, tensor = preprocess(bgr, box)
        with torch.no_grad():
            logits = model(tensor.to(args.device))
        kpa = to_kpa(logits, thresholds)
        contact = kpa >= contact_thresh
        rows.append(
            {
                "frame": path.name,
                "index": index,
                "sum_kpa": float(kpa.sum()),
                "max_kpa": float(kpa.max()),
                "contact_px": int(contact.sum()),
                "mean_kpa_in_contact": (
                    float(kpa[contact].mean()) if contact.any() else 0.0
                ),
            }
        )
        if index in montage_at:
            panels.append(np.vstack([resized, overlay(resized, kpa)]))

    with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if panels:
        cv2.imwrite(str(args.out / "montage.png"), np.hstack(panels))

    contact_px = np.array([row["contact_px"] for row in rows], float)
    sum_kpa = np.array([row["sum_kpa"] for row in rows], float)
    print(f"contact_px: zero on {int((contact_px == 0).sum())}/{len(rows)} frames, "
          f"max {contact_px.max():.0f}, median {np.median(contact_px):.0f}")
    print(f"sum_kpa   : min {sum_kpa.min():.1f}, median {np.median(sum_kpa):.1f}, "
          f"max {sum_kpa.max():.1f}")
    print(f"wrote {args.out}/frames.csv and montage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
