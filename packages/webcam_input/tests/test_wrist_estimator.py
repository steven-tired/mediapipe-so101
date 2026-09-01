import numpy as np
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.wrist_estimator import WebcamWristEstimator


def _image():
    lm = np.zeros((21, 2)); lm[0] = [0.5, 0.5]; lm[9] = [0.5, 0.25]
    return lm


def test_quaternion_is_unit():
    est = WebcamWristEstimator(depth=ScaleDepthStrategy())
    pos, quat = est.estimate(np.eye(3), _image(), image_shape=(480, 640))
    assert pos.shape == (3,) and quat.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_identity_frame_maps_to_180deg_about_x():
    # _CAMERA_TO_VR = diag(1,-1,-1) is a 180° rotation about x → quat [1,0,0,0]
    est = WebcamWristEstimator(depth=ScaleDepthStrategy())
    _, quat = est.estimate(np.eye(3), _image(), image_shape=(480, 640))
    np.testing.assert_allclose(np.abs(quat), [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_moving_wrist_right_increases_vr_x():
    est = WebcamWristEstimator(depth=ScaleDepthStrategy(), workspace_size_m=0.4)
    img_l = _image(); img_l[0] = [0.3, 0.5]
    img_r = _image(); img_r[0] = [0.7, 0.5]
    pos_l, _ = est.estimate(np.eye(3), img_l, (480, 640))
    pos_r, _ = est.estimate(np.eye(3), img_r, (480, 640))
    assert pos_r[0] > pos_l[0]


def test_moving_wrist_up_increases_vr_y():
    est = WebcamWristEstimator(depth=ScaleDepthStrategy(), workspace_size_m=0.4)
    img_low = _image(); img_low[0] = [0.5, 0.7]
    img_high = _image(); img_high[0] = [0.5, 0.3]
    pos_low, _ = est.estimate(np.eye(3), img_low, (480, 640))
    pos_high, _ = est.estimate(np.eye(3), img_high, (480, 640))
    assert pos_high[1] > pos_low[1]
