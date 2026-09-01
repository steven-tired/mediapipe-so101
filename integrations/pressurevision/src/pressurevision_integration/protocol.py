"""The pressure-reading type this integration produces.

Historically this type lived in the private IR module and the PressureVision
path borrowed it, which is why its timestamp fields were named `thermal_*` even
when they carried PressureVision timestamps. Here it is defined on its own terms
with sensor-neutral names. The private IR project keeps its own richer version;
neither imports the other.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PressureROI", "PressureReading", "inactive_pressure"]


@dataclass(frozen=True)
class PressureROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y_end), slice(self.x, self.x_end)


@dataclass(frozen=True)
class PressureReading:
    """One pressure estimate, plus enough provenance to judge whether to trust it.

    `pressure_0_1` is only meaningful when `active` is true and `available` is
    true. A consumer that ignores those flags will read 0.0 as "no pressure"
    when it actually means "no measurement".
    """

    pressure_0_1: float
    active: bool
    quality: float
    available: bool
    status: str
    roi: PressureROI | None = None
    roi_mode: str | None = None
    #: When the sensor produced this reading, on the sender's clock.
    observed_at_s: float | None = None
    #: How old the reading was when it was turned into this object.
    age_s: float | None = None
    fresh: bool = True
    level: int | None = None
    n_levels: int | None = None
    sequence: int | None = None
    sent_at_s: float | None = None
    received_at_s: float | None = None


def inactive_pressure(
    status: str,
    *,
    available: bool = True,
    quality: float = 0.0,
    roi: PressureROI | None = None,
    roi_mode: str | None = None,
    observed_at_s: float | None = None,
    age_s: float | None = None,
    fresh: bool = True,
) -> PressureReading:
    """A reading that carries no usable pressure, with the reason in `status`."""
    return PressureReading(
        pressure_0_1=0.0,
        active=False,
        quality=quality,
        available=available,
        status=status,
        roi=roi,
        roi_mode=roi_mode,
        observed_at_s=observed_at_s,
        age_s=age_s,
        fresh=fresh,
    )
