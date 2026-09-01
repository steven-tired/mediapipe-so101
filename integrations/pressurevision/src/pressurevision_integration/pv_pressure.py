"""PressureVision pressure source for the SO-101 webcam teleop shadow path.

PressureVision needs a GPU and a package set (`segmentation-models-pytorch`, `timm`)
that `.venv-lerobot` does not have, so the model runs in a separate process
(`hand-pressure/scripts/serve_pad_pressure.py`, on `.venv-pressurevision`) and ships
scalar metrics here over localhost UDP: a model
crash cannot take the arm down, and `.venv-lerobot` keeps carrying recording and the
three deployed policies untouched.

This module only decodes packets and gates them into a `PressureReading`. The
gripper proposal, the quality fallback, and the shadow/apply split all stay in
`ee_controller` and the gripper contract, which work against that type.

Telemetry note: the shadow CSV used by the private IR project reuses this
packet's timestamps; here the fields are named for what they are, so the
columns are named for the IR rig. On this path `roi_mode` is `"pv"` and the
`observed_at_s` / `age_s` fields carry the PressureVision
packet's timestamp and age. `roi_mode` is what tells the two apart.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .protocol import PressureReading, inactive_pressure

ROI_MODE = "pv"

# Wire format, shared with hand-pressure/scripts/serve_pad_pressure.py. Each side
# writes its own codec on purpose -- the two live in different repos on different
# venvs -- and SHARED_PV_TEST_PACKET pins the format from both ends.
PV_PACKET_SCHEMA_VERSION = 5
PV_PACKET_FIELDS = (
    "schema_version",
    "sequence",
    "source_t",
    "sent_t",
    "mean_kpa_in_contact",
    "max_kpa",
    "sum_kpa",
    "contact_px",
    "off_x",
    "off_y",
    "crop_w",
    "crop_h",
    "pressure_0_1",
    "level",
    "n_levels",
)
SHARED_PV_TEST_PACKET = (
    "5,42,1700000000.100000,1700000000.123456,7.250000,14.000000,3712.000000,512,"
    "-0.250000,0.500000,522,416,0.625000,1,3"
)
ABSTAIN_LEVEL = -1


@dataclass(frozen=True)
class PressureVisionPacket:
    """One sender frame, reduced to scalars.

    `off_x` / `off_y` place the contact centroid relative to the crop centre, each
    normalized by the crop half-width / half-height, so 0 is dead centre and +-1 is
    the edge. They are signed and separate rather than one radial number because the
    only measurement of the response falloff (HANDOFF section 7.0) is a single line
    through the crop -- whether it is radially symmetric or centred at all is
    unknown, and a signed pair can be refitted later without a wire format change.
    Fractions, not pixels, because the sender resizes the crop to the network's
    480x384 input, so position within the crop is what decides where a pixel lands
    in the model's field of view.
    """

    observed_at_s: float
    mean_kpa_in_contact: float
    max_kpa: float
    # What the sender classifies on. The mean saturates -- contact pixels take
    # log-spaced bin edges capped at 64 kPa -- so a session pressing slightly
    # harder than the calibration put every frame above the top boundary. sum_kpa
    # keeps growing with area as well as intensity.
    sum_kpa: float
    contact_px: int
    off_x: float
    off_y: float
    crop_w: int
    crop_h: int
    # Sender-normalized light..hard severity. Right-hand pinch separately owns
    # whether the gripper is engaged, so zero here means the calibrated light
    # target rather than "open".
    pressure_0_1: float
    # The sender still classifies for diagnostics because it owns the fitted
    # boundaries. -1 means the continuous value landed in the old abstain band;
    # it no longer blocks the continuous pressure path.
    level: int
    n_levels: int
    sequence: int = 0
    sent_at_s: float | None = None
    received_at_s: float | None = None


@dataclass(frozen=True)
class PressureVisionConfig:
    # Contact area sanity band. Below the floor is speckle; above the ceiling is a
    # flat hand or an arm resting on the pad, which HANDOFF section 7.0 records as
    # reading contact in 185/186 frames.
    #
    # The floor is measured, not guessed. Session 06's no-press trials produced a
    # repeatable false contact of 82-93 px at ~4 kPa, while its real presses ran
    # 464-835 px (light) and 1082-1313 px (hard). 250 sits in that 5x gap, so the
    # false positive is rejected on area before the level boundary ever sees it.
    min_contact_area_px: int = 250
    max_contact_area_px: int = 12000
    # Radial contact offset (hypot of off_x/off_y): fully trusted inside the first,
    # untrusted past the second, linear between. The model's response is measured
    # only near the crop centre.
    trust_full_offset: float = 0.5
    max_trust_offset: float = 0.8
    # Three sender periods of margin. The overhead C270 is capped at 7.5 fps in
    # uncompressed YUYV at 1280x720 -- measured 134 ms per frame, p95 141, max 147 --
    # so a 250 ms window sat barely above the sender's own jitter and turned any
    # hiccup into pv_stale. Still a staleness guard, not a rate requirement: a
    # 30 fps MJPG sender is well inside this too.
    max_pv_age_s: float = 0.40
    # Match the range mapper's q<=30 grasp entry and q>=65 explicit release
    # under the default 0.02..0.12 pinch map.
    near_contact_pinch: float = 0.050
    exit_contact_pinch: float = 0.085
    # Consecutive packets that must agree before the diagnostic level label is
    # adopted. This does not delay pressure_0_1, which remains continuous.
    level_hold_frames: int = 3


def _finite(text: str) -> float:
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"non-finite value {text!r}")
    return value


def encode_pv_packet(packet: PressureVisionPacket) -> bytes:
    return (
        ",".join(
            (
                str(PV_PACKET_SCHEMA_VERSION),
                str(int(packet.sequence)),
                f"{packet.observed_at_s:.6f}",
                f"{(packet.observed_at_s if packet.sent_at_s is None else packet.sent_at_s):.6f}",
                f"{packet.mean_kpa_in_contact:.6f}",
                f"{packet.max_kpa:.6f}",
                f"{packet.sum_kpa:.6f}",
                str(int(packet.contact_px)),
                f"{packet.off_x:.6f}",
                f"{packet.off_y:.6f}",
                str(int(packet.crop_w)),
                str(int(packet.crop_h)),
                f"{packet.pressure_0_1:.6f}",
                str(int(packet.level)),
                str(int(packet.n_levels)),
            )
        )
        + "\n"
    ).encode("utf-8")


def decode_pv_packet(raw: bytes) -> PressureVisionPacket:
    text = raw.decode("utf-8", errors="strict").strip()
    parts = text.split(",")
    if len(parts) != len(PV_PACKET_FIELDS):
        raise ValueError(f"expected {len(PV_PACKET_FIELDS)} fields, got {len(parts)}")
    version = int(parts[0])
    if version != PV_PACKET_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {version}")
    sequence = int(parts[1])
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    source_t = _finite(parts[2])
    sent_t = _finite(parts[3])
    if sent_t < source_t:
        raise ValueError("sent_t must not precede source_t")
    contact_px = int(parts[7])
    if contact_px < 0:
        raise ValueError("contact_px must be non-negative")
    pressure_0_1 = _finite(parts[12])
    if not 0.0 <= pressure_0_1 <= 1.0:
        raise ValueError("pressure_0_1 must be in [0, 1]")
    level, n_levels = int(parts[13]), int(parts[14])
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2")
    if not (level == ABSTAIN_LEVEL or 0 <= level < n_levels):
        raise ValueError(f"level {level} outside 0..{n_levels - 1} and not abstain")
    return PressureVisionPacket(
        observed_at_s=source_t,
        mean_kpa_in_contact=_finite(parts[4]),
        max_kpa=_finite(parts[5]),
        sum_kpa=_finite(parts[6]),
        contact_px=contact_px,
        off_x=_finite(parts[8]),
        off_y=_finite(parts[9]),
        crop_w=int(parts[10]),
        crop_h=int(parts[11]),
        pressure_0_1=pressure_0_1,
        level=level,
        n_levels=n_levels,
        sequence=sequence,
        sent_at_s=sent_t,
    )


def area_quality(contact_px: int, cfg: PressureVisionConfig) -> float:
    """How much to trust a contact blob of this size. Not how hard it is pressing."""
    if contact_px >= cfg.min_contact_area_px:
        if contact_px <= cfg.max_contact_area_px:
            return 1.0
        # Ramp to zero across one more ceiling's worth of area.
        excess = (contact_px - cfg.max_contact_area_px) / max(1, cfg.max_contact_area_px)
        return float(np.clip(1.0 - excess, 0.0, 1.0))
    return float(np.clip(contact_px / max(1, cfg.min_contact_area_px), 0.0, 1.0))


def offset_quality(off_x: float, off_y: float, cfg: PressureVisionConfig) -> float:
    """How much to trust a contact this far from the crop centre."""
    radius = float(np.hypot(off_x, off_y))
    if radius <= cfg.trust_full_offset:
        return 1.0
    span = cfg.max_trust_offset - cfg.trust_full_offset
    if span <= 0.0:
        return 0.0
    return float(np.clip(1.0 - (radius - cfg.trust_full_offset) / span, 0.0, 1.0))


class PressureVisionUDPSource:
    """Read one packet per datagram from localhost UDP.

    This never gives up on silence: the sender is a separate
    process the operator may start after the teleop, and LatestFrameSource retires
    its producer thread permanently on the first exception. Waiting instead means a
    late sender is picked up, and a missing one degrades to `pv_unavailable`.
    """

    def __init__(self, *, bind_ip: str = "127.0.0.1", port: int = 8090, poll_s: float = 0.5):
        self.poll_s = poll_s
        self._sock: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_ip, port))
        self.port = self._sock.getsockname()[1]

    def read(self) -> PressureVisionPacket:
        while True:
            sock = self._sock
            if sock is None:
                raise RuntimeError("pressurevision udp source is closed")
            sock.settimeout(self.poll_s)
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                raise RuntimeError(f"pressurevision udp socket error: {exc}") from exc
            try:
                return replace(decode_pv_packet(data), received_at_s=time.time())
            except (ValueError, UnicodeDecodeError):
                # Tolerate stray datagrams rather than killing the producer thread.
                continue

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            sock.close()


class PressureVisionSource:
    """Duck-typed pressure_source: update(landmarks, pinch=, enabled=) -> PressureReading.

    pinch decides WHEN contact counts, PressureVision decides HOW HARD. A resting
    flat hand reads contact (HANDOFF section 7.0) but is not a pinch pose, so the
    gate keeps it out of the grip. The cost is that PressureVision cannot announce a
    contact the pinch detector missed.
    """

    def __init__(self, *, source, config: PressureVisionConfig | None = None):
        self.source = source
        self.config = config or PressureVisionConfig()
        self._latched = False
        self._level: int | None = None
        self._candidate: int | None = None
        self._candidate_seen = 0

    def reset(self) -> None:
        self._latched = False
        self._level = None
        self._candidate = None
        self._candidate_seen = 0

    def _debounce(self, level: int) -> int:
        """Adopt a new level only once it has repeated, and hold the old one meanwhile."""
        if self._level is None:
            self._level = level
            self._candidate, self._candidate_seen = None, 0
            return level
        if level == self._level:
            self._candidate, self._candidate_seen = None, 0
            return level
        if level == self._candidate:
            self._candidate_seen += 1
        else:
            self._candidate, self._candidate_seen = level, 1
        if self._candidate_seen >= self.config.level_hold_frames:
            self._level = level
            self._candidate, self._candidate_seen = None, 0
        return self._level

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def _reading(
        self,
        *,
        pressure_0_1: float,
        active: bool,
        quality: float,
        status: str,
        packet: PressureVisionPacket,
        age_s: float | None,
        available: bool = True,
        level: int | None = None,
        n_levels: int | None = None,
    ) -> PressureReading:
        return PressureReading(
            pressure_0_1=pressure_0_1,
            active=active,
            quality=quality,
            available=available,
            status=status,
            roi=None,
            roi_mode=ROI_MODE,
            observed_at_s=packet.observed_at_s,
            age_s=age_s,
            level=level,
            n_levels=n_levels,
            sequence=packet.sequence,
            sent_at_s=packet.sent_at_s,
            received_at_s=packet.received_at_s,
        )

    def update(self, landmarks, pinch: float, enabled: bool) -> PressureReading:
        cfg = self.config
        try:
            packet = self.source.read()
        except Exception:
            self._latched = False
            return inactive_pressure("pv_unavailable", available=False, roi_mode=ROI_MODE)

        contact_present = packet.contact_px > 0
        level = self._debounce(0 if not contact_present else packet.level)
        age_s = time.time() - packet.observed_at_s
        if age_s < 0.0:
            self._latched = False
            return self._reading(
                pressure_0_1=0.0, active=False, quality=0.0, status="pv_time_skew",
                packet=packet, age_s=age_s, available=False,
                level=level, n_levels=packet.n_levels,
            )
        if age_s > cfg.max_pv_age_s:
            self._latched = False
            return self._reading(
                pressure_0_1=0.0, active=False, quality=0.0, status="pv_stale",
                packet=packet, age_s=age_s, available=False,
                level=level, n_levels=packet.n_levels,
            )

        if not enabled:
            self._latched = False
            return self._reading(
                pressure_0_1=0.0, active=False, quality=1.0, status="baseline",
                packet=packet, age_s=age_s,
                level=level, n_levels=packet.n_levels,
            )

        # Contact presence and the continuous value are immediate.  The debounced
        # level above is telemetry only; letting it gate this branch delayed the
        # continuous proposal and briefly turned a disappearing blob into a
        # low-quality active reading.
        if not contact_present:
            self._latched = False
            return self._reading(
                pressure_0_1=0.0, active=False, quality=1.0, status="baseline",
                packet=packet, age_s=age_s,
                level=level, n_levels=packet.n_levels,
            )

        quality = min(
            area_quality(packet.contact_px, cfg),
            offset_quality(packet.off_x, packet.off_y, cfg),
        )

        if pinch <= cfg.near_contact_pinch:
            self._latched = True
        elif pinch >= cfg.exit_contact_pinch:
            self._latched = False

        if not self._latched:
            return self._reading(
                pressure_0_1=0.0, active=False, quality=quality, status="baseline",
                packet=packet, age_s=age_s,
                level=level, n_levels=packet.n_levels,
            )

        return self._reading(
            pressure_0_1=packet.pressure_0_1,
            active=True,
            quality=quality,
            status=(
                "pv_abstain_continuous"
                if packet.level == ABSTAIN_LEVEL
                else "active"
            ),
            packet=packet,
            age_s=age_s,
            level=level,
            n_levels=packet.n_levels,
        )
