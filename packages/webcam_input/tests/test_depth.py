import numpy as np
from webcam_input.depth import OAKDepthStrategy, ScaleDepthStrategy, sample_depth_m


def test_smaller_hand_is_farther():
    s = ScaleDepthStrategy(ref_bone_px=120.0, ref_distance_m=0.5)
    lm = np.zeros((21, 2))
    lm_big = lm.copy(); lm_big[0] = [0.5, 0.5]; lm_big[9] = [0.5, 0.25]
    lm_small = lm.copy(); lm_small[0] = [0.5, 0.5]; lm_small[9] = [0.5, 0.4375]
    z_big = s.estimate_z(lm_big, image_shape=(480, 640))
    z_small = s.estimate_z(lm_small, image_shape=(480, 640))
    assert z_small > z_big


def test_missing_bone_returns_last_good():
    s = ScaleDepthStrategy(ref_bone_px=120.0, ref_distance_m=0.5)
    lm = np.zeros((21, 2)); lm[0] = [0.5, 0.5]; lm[9] = [0.5, 0.25]
    z0 = s.estimate_z(lm, image_shape=(480, 640))
    z1 = s.estimate_z(np.zeros((21, 2)), image_shape=(480, 640))
    assert z1 == z0


# --- OAK stereo depth (real metric z) ---

def _depth_frame(value_mm, h=480, w=640):
    return np.full((h, w), value_mm, dtype=np.uint16)


def test_sample_depth_returns_metres_at_wrist():
    d = _depth_frame(0)
    d[235:245, 315:325] = 600          # 600 mm patch around image centre
    z = sample_depth_m(d, (0.5, 0.5), radius_px=8, min_m=0.1, max_m=2.0)
    assert abs(z - 0.6) < 1e-6          # 600 mm -> 0.6 m


def test_sample_depth_rejects_holes_and_out_of_range():
    d = _depth_frame(0)                 # all holes (0)
    assert sample_depth_m(d, (0.5, 0.5), radius_px=8) is None
    d2 = _depth_frame(50)              # 0.05 m, below min -> rejected
    assert sample_depth_m(d2, (0.5, 0.5), radius_px=8, min_m=0.1, max_m=2.0) is None


def test_sample_depth_median_ignores_zero_holes():
    d = _depth_frame(0)
    d[235:245, 315:325] = 700
    d[235:240, 315:320] = 0            # half the window is holes
    z = sample_depth_m(d, (0.5, 0.5), radius_px=8)
    assert abs(z - 0.7) < 1e-6         # median over valid pixels only


def test_oak_strategy_smooths_and_holds_on_hole():
    s = OAKDepthStrategy(radius_px=8, ema_alpha=1.0, default_z=0.5)   # alpha=1 -> no lag
    lm = np.zeros((21, 2)); lm[0] = [0.5, 0.5]
    s.update_depth(_depth_frame(800))
    assert abs(s.estimate_z(lm, (480, 640)) - 0.8) < 1e-6
    s.update_depth(_depth_frame(0))                                   # all holes
    assert abs(s.estimate_z(lm, (480, 640)) - 0.8) < 1e-6            # holds last good


def test_oak_strategy_default_before_any_frame():
    s = OAKDepthStrategy(default_z=0.42)
    lm = np.zeros((21, 2)); lm[0] = [0.5, 0.5]
    assert s.estimate_z(lm, (480, 640)) == 0.42
