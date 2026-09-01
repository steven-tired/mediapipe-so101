"""Deterministic guided sequence for PressureVision shadow validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PVTrialPhase:
    name: str
    expected_level: int
    duration_s: float


DEFAULT_PHASES = (
    PVTrialPhase("open", 0, 2.0),
    PVTrialPhase("light", 1, 3.0),
    PVTrialPhase("open", 0, 2.0),
    PVTrialPhase("hard", 2, 3.0),
)


class PVTrialProtocol:
    """Clock-driven protocol metadata; it never drives the robot or edits levels."""

    def __init__(self, *, repetitions: int = 5, phases=DEFAULT_PHASES):
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not phases:
            raise ValueError("at least one phase is required")
        if any(phase.duration_s <= 0 for phase in phases):
            raise ValueError("phase durations must be positive")
        self.repetitions = int(repetitions)
        self.phases = tuple(phases)
        self._started_at_s: float | None = None

    @property
    def total_duration_s(self) -> float:
        return self.repetitions * sum(phase.duration_s for phase in self.phases)

    def start(self, now_s: float) -> None:
        self._started_at_s = float(now_s)

    def expected(self, now_s: float) -> dict[str, float | int | str | bool] | None:
        if self._started_at_s is None:
            self.start(now_s)
        elapsed = float(now_s) - self._started_at_s
        if elapsed < 0.0 or elapsed >= self.total_duration_s:
            return None
        cycle_duration = sum(phase.duration_s for phase in self.phases)
        cycle_index = min(self.repetitions - 1, int(elapsed // cycle_duration))
        within_cycle = elapsed - cycle_index * cycle_duration
        phase_offset = 0.0
        phase_index = len(self.phases) - 1
        phase = self.phases[-1]
        for index, candidate in enumerate(self.phases):
            if within_cycle < phase_offset + candidate.duration_s:
                phase_index, phase = index, candidate
                break
            phase_offset += candidate.duration_s
        return {
            "trial_index": cycle_index,
            "phase_index": phase_index,
            "trial_phase": phase.name,
            "expected_level": phase.expected_level,
            "elapsed_s": elapsed,
            "complete": False,
        }


def default_trial_protocol(repetitions: int = 5) -> PVTrialProtocol:
    return PVTrialProtocol(repetitions=repetitions)
