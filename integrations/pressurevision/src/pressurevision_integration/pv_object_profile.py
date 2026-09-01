"""Validated light-to-hard gripper positions for one labeled rigid object."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PressureVisionObjectProfile:
    object_id: str
    arm_id: str
    open_pos: float
    light_pos: float
    hard_pos: float
    max_current: float = 50.0
    max_temperature_c: float = 55.0
    control_mode: str = "hard_profile"
    schema_version: int = 1
    sweep_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported object profile schema {self.schema_version}")
        if not self.object_id.strip() or not self.arm_id.strip():
            raise ValueError("object_id and arm_id are required")
        if self.control_mode != "hard_profile":
            raise ValueError("object profile control_mode must be 'hard_profile'")
        values = (self.open_pos, self.light_pos, self.hard_pos)
        if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in values):
            raise ValueError("gripper positions must be finite values in [0, 100]")
        if not self.hard_pos < self.light_pos < self.open_pos:
            raise ValueError("hard_pos < light_pos < open_pos is required")
        if not math.isfinite(self.max_current) or self.max_current <= 0:
            raise ValueError("max_current must be positive")
        if not math.isfinite(self.max_temperature_c) or self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be positive")

    def target_for_level(self, level: int, n_levels: int) -> float | None:
        if level == 0:
            return self.open_pos
        if n_levels != 3:
            raise ValueError(f"object profile expects n_levels=3, got {n_levels}")
        if level == 1:
            return self.light_pos
        if level == 2:
            return self.hard_pos
        return None

    def target_for_pressure(self, pressure_0_1: float) -> float:
        """Interpolate calibrated light..hard gripper positions continuously."""
        pressure_0_1 = float(pressure_0_1)
        if not math.isfinite(pressure_0_1) or not 0.0 <= pressure_0_1 <= 1.0:
            raise ValueError("pressure_0_1 must be finite and in [0, 1]")
        return self.light_pos + pressure_0_1 * (self.hard_pos - self.light_pos)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "object_id": self.object_id,
            "arm_id": self.arm_id,
            "open_pos": self.open_pos,
            "light_pos": self.light_pos,
            "hard_pos": self.hard_pos,
            "max_current": self.max_current,
            "max_temperature_c": self.max_temperature_c,
            "control_mode": self.control_mode,
            "sweep_evidence_sha256": self.sweep_evidence_sha256,
        }


def load_object_profile(path: str | Path) -> PressureVisionObjectProfile:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PressureVisionObjectProfile(
            schema_version=int(data.get("schema_version", 0)),
            object_id=str(data["object_id"]),
            arm_id=str(data["arm_id"]),
            open_pos=float(data["open_pos"]),
            light_pos=float(data["light_pos"]),
            hard_pos=float(data["hard_pos"]),
            max_current=float(data.get("max_current", 50.0)),
            max_temperature_c=float(data.get("max_temperature_c", 55.0)),
            control_mode=str(data.get("control_mode", "hard_profile")),
            sweep_evidence_sha256=data.get("sweep_evidence_sha256"),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid object profile {path}: {exc}") from exc


def save_object_profile(path: str | Path, profile: PressureVisionObjectProfile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_profile_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
