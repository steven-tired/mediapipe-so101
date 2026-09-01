import pytest
import numpy as np
from dataclasses import dataclass

from lerobot_teleoperator_so101_webcam.ee_control import (
    EE_ACTION_KEYS,
    ee_action_from_hand,
    gripper_pos_from_pinch,
    gripper_vel_from_pinch,
    joint_center,
)


@dataclass
class Cfg:
    ee_x_idx: int = 0; ee_x_sign: float = 1.0
    ee_y_idx: int = 1; ee_y_sign: float = 1.0
    ee_z_idx: int = 2; ee_z_sign: float = 1.0
    grip_pinch_min: float = 0.02; grip_pinch_max: float = 0.12; grip_sign: float = 1.0


def test_keys_present_and_orientation_zero():
    a = ee_action_from_hand(np.zeros(3), 0.07, True, Cfg())
    assert set(a) == set(EE_ACTION_KEYS)
    assert a["target_wx"] == 0.0 and a["target_wy"] == 0.0 and a["target_wz"] == 0.0
    assert a["enabled"] is True


def test_bounded_roll_is_forwarded_on_the_tool_axis():
    a = ee_action_from_hand(
        np.zeros(3), 0.07, True, Cfg(), roll_delta=np.deg2rad(20.0)
    )

    assert a["target_wx"] == 0.0
    assert a["target_wy"] == 0.0
    assert a["target_wz"] == pytest.approx(np.deg2rad(20.0))


def test_disabled_hand_cannot_change_roll():
    a = ee_action_from_hand(
        np.zeros(3), 0.07, False, Cfg(), roll_delta=np.deg2rad(20.0)
    )

    assert a["target_wz"] == 0.0


def test_displacement_maps_to_targets_with_sign():
    a = ee_action_from_hand(np.array([0.1, -0.2, 0.3]), 0.07, True, Cfg())
    assert a["target_x"] == 0.1
    assert a["target_y"] == -0.2
    assert a["target_z"] == 0.3
    b = ee_action_from_hand(np.array([0.1, 0, 0]), 0.07, True, Cfg(ee_x_sign=-1.0))
    assert b["target_x"] == -0.1


def test_pinch_closes_open_opens():
    cfg = Cfg()
    assert gripper_vel_from_pinch(cfg.grip_pinch_min, cfg) > 0   # tight pinch -> close
    assert gripper_vel_from_pinch(cfg.grip_pinch_max, cfg) < 0   # open hand -> open
    assert -1.0 <= gripper_vel_from_pinch(0.5, cfg) <= 1.0       # clipped


def test_disabled_passthrough():
    a = ee_action_from_hand(np.array([0.1, 0.1, 0.1]), 0.07, False, Cfg())
    assert a["enabled"] is False
    assert a["gripper_vel"] == 0.0


def test_disabled_freezes_gripper_even_with_tight_pinch():
    # Regression for the clutch/dropout bug: a zeroed/garbage pinch reading must NOT
    # be interpreted as a tight pinch and drive the gripper closed while disabled.
    cfg = Cfg()
    a = ee_action_from_hand(np.zeros(3), cfg.grip_pinch_min, False, cfg)
    assert a["gripper_vel"] == 0.0


def test_enabled_tight_pinch_still_closes():
    cfg = Cfg()
    a = ee_action_from_hand(np.zeros(3), cfg.grip_pinch_min, True, cfg)
    assert a["gripper_vel"] > 0.0


def test_gripper_pos_absolute_maps_pinch_to_0_100():
    cfg = Cfg()
    assert gripper_pos_from_pinch(cfg.grip_pinch_min, cfg) == 0.0    # tight pinch -> closed
    assert gripper_pos_from_pinch(cfg.grip_pinch_max, cfg) == 100.0  # open hand -> open
    mid = 0.5 * (cfg.grip_pinch_min + cfg.grip_pinch_max)
    assert abs(gripper_pos_from_pinch(mid, cfg) - 50.0) < 1e-6       # midway
    # out-of-range pinch is clipped, never rails past the joint limits
    assert gripper_pos_from_pinch(-1.0, cfg) == 0.0
    assert gripper_pos_from_pinch(1.0, cfg) == 100.0


def test_config_defaults_apply_vr_to_robot_transform():
    # The shipped config defaults must implement LeFranX's VR->robot axis map so that
    # hand height (VR y) drives robot height (EE z) -- otherwise you cannot lower to pick.
    from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
    cfg = SO101WebcamEEConfig()
    vr = np.array([0.11, 0.22, 0.33])  # VR [x=right, y=up, z=depth]
    a = ee_action_from_hand(vr, 0.07, True, cfg)
    assert a["target_x"] == vr[2]    # robot forward  = VR depth
    assert a["target_y"] == -vr[0]   # robot left     = -VR right
    assert a["target_z"] == vr[1]    # robot up       = VR height  (raise hand -> raise gripper)


def test_joint_center_calibration_midpoints():
    # body joints (DEGREES) and RANGE_M100_100 centre at 0; gripper (RANGE_0_100) at 50
    assert joint_center("degrees") == 0.0
    assert joint_center("range_m100_100") == 0.0
    assert joint_center("range_0_100") == 50.0


# --- grip ratchet (arm B) ---

from lerobot_teleoperator_so101_webcam.ee_control import (  # noqa: E402
    GRIP_LATCH_ENTER,
    GRIP_LATCH_EXIT,
    GRIP_LATCH_EXIT_FRAMES,
    grip_ratchet,
)

RATCHET = dict(close_alpha=0.7, open_alpha=0.15)


def _run(sequence, smoothed=None, latched=False, open_frames=0):
    """Feed raw grip commands through the ratchet; return (commands, state)."""
    out = []
    for raw in sequence:
        smoothed, latched, open_frames = grip_ratchet(
            raw, smoothed, latched, open_frames, **RATCHET
        )
        out.append(smoothed)
    return out, (smoothed, latched, open_frames)


def test_first_frame_seeds_without_latching():
    (cmd,), (_, latched, frames) = _run([80.0])
    assert cmd == 80.0
    assert latched is False and frames == 0


def test_closing_past_the_enter_threshold_latches():
    _, (cmd, latched, _) = _run([80.0] + [0.0] * 12)
    assert cmd < GRIP_LATCH_ENTER
    assert latched is True


def test_latched_grip_ignores_a_jittered_open_reading():
    """The documented failure: pinch tracking jitters mid-lift and the claw
    loosens. Latched, a burst of 'open' frames must not move the command."""
    _, state = _run([80.0] + [0.0] * 12)
    firm = state[0]
    # Four open frames is one short of the debounce, so nothing is released.
    commands, (cmd, latched, _) = _run([100.0] * (GRIP_LATCH_EXIT_FRAMES - 1), *state)

    assert latched is True
    assert cmd == pytest.approx(firm)
    assert all(c == pytest.approx(firm) for c in commands)


def test_latched_grip_still_closes_further():
    """Squeezing harder is deliberate and readable, so it must still track."""
    _, state = _run([80.0] + [25.0] * 6)
    firm = state[0]
    _, (cmd, latched, _) = _run([0.0] * 6, *state)

    assert latched is True
    assert cmd < firm


def test_a_sustained_open_releases_and_the_command_follows():
    _, state = _run([80.0] + [0.0] * 12)
    _, (cmd, latched, frames) = _run([100.0] * (GRIP_LATCH_EXIT_FRAMES + 6), *state)

    assert latched is False and frames == 0
    assert cmd > GRIP_LATCH_ENTER


def test_an_interrupted_open_burst_does_not_release():
    """Debounce: four open frames, a closed frame, then four more."""
    _, state = _run([80.0] + [0.0] * 12)
    firm = state[0]
    burst = [100.0] * (GRIP_LATCH_EXIT_FRAMES - 1) + [0.0]
    _, state = _run(burst, *state)
    _, (cmd, latched, _) = _run([100.0] * (GRIP_LATCH_EXIT_FRAMES - 1), *state)

    assert latched is True
    assert cmd <= firm


def test_tracked_mode_is_what_the_ratchet_replaces():
    """Arm A: the same open burst does move the command, which is the
    behaviour GRIP_OPEN_ALPHA exists to slow down rather than stop."""
    _, state = _run([80.0] + [0.0] * 12)
    firm = state[0]
    smoothed = firm
    for raw in [100.0] * (GRIP_LATCH_EXIT_FRAMES - 1):
        alpha = RATCHET["close_alpha"] if raw < smoothed else RATCHET["open_alpha"]
        smoothed = alpha * raw + (1 - alpha) * smoothed

    assert smoothed > firm


def test_release_does_not_immediately_relatch():
    """Regression: a released grasp leaves the command near zero and it climbs
    out slowly, so testing the command alone re-latched on the very next frame
    and the release never took effect."""
    _, state = _run([80.0] + [0.0] * 12)
    commands, (_, latched, _) = _run([100.0] * 20, *state)

    assert latched is False
    # Held flat while the release debounces, then monotonically out -- never
    # dipping back, which is what a mid-release re-latch would look like.
    assert all(b >= a for a, b in zip(commands, commands[1:]))
    # Frames before the debounce completes are held; the frame that completes
    # it is already moving.
    held = commands[: GRIP_LATCH_EXIT_FRAMES - 1]
    assert all(c == pytest.approx(held[0]) for c in held)
    assert commands[GRIP_LATCH_EXIT_FRAMES - 1] > held[0]
    assert commands[-1] > GRIP_LATCH_EXIT


def test_a_second_grasp_after_a_release_latches_again():
    _, state = _run([80.0] + [0.0] * 12)
    _, state = _run([100.0] * 20, *state)          # release, hand fully open
    _, (cmd, latched, _) = _run([0.0] * 12, *state)  # close again

    assert latched is True
    assert cmd < GRIP_LATCH_ENTER


# --- pinch -> command mapping ---

from lerobot_teleoperator_so101_webcam.ee_control import (  # noqa: E402
    GRIP_SPAN_PINCH_MAX,
    raw_grip_from_pinch,
)

OVERDRIVE = 18.0


def test_overdrive_map_flattens_the_closed_end():
    """The defect that motivated span mode: every pinch from the floor up to
    the overdrive point commands the same thing, so a partial loosening near a
    firm grip is not representable before any ratchet gets involved."""
    cfg = Cfg()
    flat = [
        raw_grip_from_pinch(p, cfg, grip_map="overdrive", overdrive=OVERDRIVE)
        for p in (0.020, 0.026, 0.032, 0.038)
    ]

    assert flat == [0.0, 0.0, 0.0, 0.0]
    assert raw_grip_from_pinch(0.044, cfg, grip_map="overdrive", overdrive=OVERDRIVE) > 0.0


def test_span_map_is_monotone_all_the_way_to_closure():
    cfg = Cfg()
    commands = [
        raw_grip_from_pinch(p, cfg, grip_map="span")
        for p in (0.020, 0.026, 0.032, 0.038, 0.044)
    ]

    assert commands[0] == 0.0
    assert all(b > a for a, b in zip(commands, commands[1:]))


def test_span_map_reaches_full_open_at_the_narrowed_maximum():
    cfg = Cfg()
    assert raw_grip_from_pinch(GRIP_SPAN_PINCH_MAX, cfg, grip_map="span") == pytest.approx(100.0)
    # Past it the command saturates rather than running away.
    assert raw_grip_from_pinch(0.20, cfg, grip_map="span") == pytest.approx(100.0)


def test_span_costs_clamping_authority_at_a_moderate_pinch():
    """The trade, stated as a test: the resolution near closure is paid for by
    a moderate pinch no longer commanding a full clamp."""
    cfg = Cfg()
    moderate = 0.030

    assert raw_grip_from_pinch(moderate, cfg, grip_map="overdrive", overdrive=OVERDRIVE) == 0.0
    assert raw_grip_from_pinch(moderate, cfg, grip_map="span") > 0.0
