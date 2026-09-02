"""`so101_diag ids` must report a motor that does not answer.

`MotorsBus.ping()` takes `raise_on_error=False` by default and returns None for
a motor that never replies. The command counted only exceptions, so it printed
"missed 0/10  ok" for all six ids at the very moment `robot.connect()` was
failing its handshake with "Missing motor IDs: - 5". The one command whose job
is to find a dropping servo could not fail.
"""

import sys
import types
from pathlib import Path

import pytest

MODULE = "lerobot_teleoperator_so101_webcam.programs.so101_diag"


@pytest.fixture
def diag(monkeypatch):
    import importlib

    module = importlib.import_module(MODULE)
    return module


class FakeBus:
    """Answers for every id except the absent one -- by returning None, not raising."""

    def __init__(self, absent):
        self.absent = absent

    def ping(self, motor_id):
        return None if motor_id == self.absent else 777

    def disconnect(self, disable_torque=False):
        pass


def test_a_silent_motor_is_reported_as_absent(diag, monkeypatch, capsys):
    monkeypatch.setattr(diag, "raw_bus", lambda: FakeBus(absent=5))
    monkeypatch.setattr(diag.time, "sleep", lambda seconds: None)

    diag.cmd_ids(None)

    out = capsys.readouterr().out
    absent = [line for line in out.splitlines() if "DROPPING/ABSENT" in line]
    assert len(absent) == 1, out
    assert " id 5 " in absent[0] and "missed 10/10" in absent[0]


def test_the_motors_that_answer_are_not_flagged(diag, monkeypatch, capsys):
    monkeypatch.setattr(diag, "raw_bus", lambda: FakeBus(absent=5))
    monkeypatch.setattr(diag.time, "sleep", lambda seconds: None)

    diag.cmd_ids(None)

    out = capsys.readouterr().out
    for motor_id in (1, 2, 3, 4, 6):
        line = next(l for l in out.splitlines() if f" id {motor_id} " in l)
        assert "missed 0/10" in line and "model=777" in line and "ok" in line


def test_an_exception_still_counts_as_a_miss(diag, monkeypatch, capsys):
    class Raising(FakeBus):
        def ping(self, motor_id):
            if motor_id == self.absent:
                raise RuntimeError("no response")
            return 777

    monkeypatch.setattr(diag, "raw_bus", lambda: Raising(absent=3))
    monkeypatch.setattr(diag.time, "sleep", lambda seconds: None)

    diag.cmd_ids(None)

    out = capsys.readouterr().out
    assert "missed 10/10" in next(l for l in out.splitlines() if " id 3 " in l)
