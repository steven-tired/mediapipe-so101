"""Deploy a trained Diffusion Policy on the SO-101 — autonomous pick-and-place.

The recorded ACTION was post-IK joint targets, so deployment needs **no IK and no
webcam/hand-tracking**: the policy maps (front-cam image + joint state) -> 6 joint targets
directly. This mirrors `record_so101_ee.py` MINUS the teleop:

  * same robot/camera setup (SO101Follower, Logitech /dev/video2 YUYV as observation.images.front),
  * same down-ready start pose (controller.middle_pose) the demos started from,
  * LeRobot's own predict_action() (loads the saved normalizers, so units/scaling match training).

Run:  ./scripts/run_deploy_ee.sh                 (defaults: Hub policy, cuda, 60 s)
  or:  python deploy_so101_ee.py --policy stevenzenith/dp_pickplace --device cuda \
                                 --port /dev/ttyACM1 --duration 60

SAFETY: this moves the arm AUTONOMOUSLY. Clear the workspace, place one block, keep a hand on
the power switch. Ctrl-C stops cleanly and leaves torque ON so the arm holds pose (won't droop).
"""

import argparse
import os
import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

# Reuse the recorder's hardened robot + motion helpers (resilient per-frame read, gentle ramp).
from .record_so101_ee import ResilientSOFollower, _read_positions, _ramp_to
from ..ee_control import joint_center
from ..ee_controller import MIDDLE_WRIST_DOWN_DEG

ARM_ID = "so101_follower_1"
WORKSPACE_CAM_PATH = os.environ.get("SO101_WORKSPACE_CAM", "/dev/video2")
# = observation.images.front, opened by PATH so a changing /dev/video index does not matter.
WORKSPACE_CAM_FOURCC = "YUYV"        # MJPG drops bytes over USB -> corrupt frames; YUYV is clean
FPS = 10                             # match the dataset's record rate
DEFAULT_POLICY = "stevenzenith/dp_pickplace"
TASK = "hand-tracking pick and place"


def _load_policy(repo: str, device: torch.device):
    """Load the trained policy + its saved pre/post processors (normalizers) from a Hub repo/dir."""
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.pretrained_path = repo
    policy = get_policy_class(cfg.type).from_pretrained(repo, config=cfg)
    policy.to(device).eval()
    policy.reset()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=repo)
    return policy, pre, post


def _apply_diffusion_overrides(policy, scheduler, steps, tag="deploy"):
    """Optionally swap the diffusion sampler and/or set denoising steps. DDIM samples well in
    ~10 steps even for a DDPM-trained model (DDPM itself needs ~100 -> slow). No effect on ACT."""
    if not hasattr(policy, "diffusion"):
        return
    d = policy.diffusion
    if scheduler is not None:
        from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
        c = d.noise_scheduler.config
        d.noise_scheduler = _make_noise_scheduler(
            scheduler.upper(),
            num_train_timesteps=c.num_train_timesteps, beta_start=c.beta_start, beta_end=c.beta_end,
            beta_schedule=c.beta_schedule, clip_sample=c.clip_sample,
            clip_sample_range=c.clip_sample_range, prediction_type=c.prediction_type)
        print(f"[{tag}] diffusion scheduler -> {scheduler.upper()}")
    if steps is not None:
        d.num_inference_steps = steps
        print(f"[{tag}] diffusion denoising steps -> {steps}")


def _ready_pose(robot) -> dict:
    """The same down-ready pose the demos started from: centred joints, wrist pitched down 90 deg."""
    pose = {f"{m}.pos": joint_center(robot.bus.motors[m].norm_mode.value)
            for m in robot.bus.motors}
    pose["wrist_flex.pos"] = MIDDLE_WRIST_DOWN_DEG
    return pose


def policy_camera_keys(policy) -> list[str]:
    """The `observation.images.*` keys a policy expects, in config order."""
    features = getattr(getattr(policy, "config", None), "image_features", None) or []
    return [str(k) for k in features]


def _check_policy_cameras(policy, cameras) -> None:
    """Fail before touching the arm if the policy wants cameras we do not provide.

    Without this the mismatch surfaces as a KeyError inside the first inference —
    after the arm has already ramped to the ready pose and started moving.
    """
    wanted = policy_camera_keys(policy)
    have = {f"observation.images.{name}" for name in cameras}
    missing = [k for k in wanted if k not in have]
    if missing:
        raise SystemExit(
            f"[deploy] policy expects {wanted} but this program provides "
            f"{sorted(have)}.\n"
            f"          Missing: {missing}\n"
            f"          This deployment path is single-camera. Use a policy trained "
            f"on one camera, or extend the camera set to match."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="HF Hub repo id or local dir")
    # Stable by-id symlink: the arm's /dev/ttyACM* index changes across replugs (was ACM1, then ACM0),
    # but this serial-number path is constant. Pass --port /dev/ttyACMx to override.
    ap.add_argument("--port",
                    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00",
                    help="SO-101 serial port (default: stable by-id symlink)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to run before stopping")
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="Diffusion only: # of denoising steps per inference. DDPM needs ~100 (slow); "
                         "with --scheduler ddim, ~10 is enough. No effect on ACT.")
    ap.add_argument("--scheduler", choices=["ddpm", "ddim"], default=None,
                    help="Diffusion only: override the sampler. 'ddim' + '--inference-steps 10' gives "
                         "~10x faster inference at ~full quality (the right speed fix). 'ddpm' is the "
                         "trained default.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[deploy] loading policy {args.policy} on {device} ...")
    policy, pre, post = _load_policy(args.policy, device)
    _apply_diffusion_overrides(policy, args.scheduler, args.inference_steps)

    cameras = {"front": OpenCVCameraConfig(index_or_path=WORKSPACE_CAM_PATH, width=640, height=480,
                                           fps=FPS, fourcc=WORKSPACE_CAM_FOURCC, warmup_s=3)}
    _check_policy_cameras(policy, cameras)
    robot = ResilientSOFollower(SO101FollowerConfig(port=args.port, id=ARM_ID, use_degrees=True,
                                                    cameras=cameras,
                                                    disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    motors = list(robot.bus.motors.keys())

    try:
        # Start from the same down-ready pose the policy always saw at episode start.
        print("[deploy] ramping to down-ready pose ...")
        _ramp_to(robot, _ready_pose(robot))
        _read_positions(robot)        # settle / verify the bus reads

        print(f"[deploy] RUNNING autonomously for {args.duration:.0f}s  (Ctrl-C to stop) ...")
        period = 1.0 / FPS
        t_end = time.perf_counter() + args.duration
        n = 0
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            obs = robot.get_observation()                       # {"<m>.pos":..., "front": HxWx3 RGB}
            state = np.array([obs[f"{m}.pos"] for m in motors], dtype=np.float32)
            policy_obs = {"observation.state": state,
                          "observation.images.front": obs["front"]}
            action = predict_action(policy_obs, policy, device, pre, post, use_amp=False,
                                    task=TASK, robot_type=robot.name)
            a = action.cpu().numpy().reshape(-1)
            robot.send_action({f"{m}.pos": float(a[i]) for i, m in enumerate(motors)})
            n += 1
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
        print(f"[deploy] done — {n} steps (~{n / args.duration:.1f} Hz).")
    except KeyboardInterrupt:
        print("\n[deploy] stopped by user.")
    finally:
        # disconnect with torque held (disable_torque_on_disconnect=False) so the arm keeps its pose.
        robot.disconnect()


if __name__ == "__main__":
    main()
