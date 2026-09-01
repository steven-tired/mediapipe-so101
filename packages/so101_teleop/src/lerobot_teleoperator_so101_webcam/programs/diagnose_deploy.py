"""Diagnose a misbehaving deploy: log webcam input, joint state, servo load/temp, and the
policy's PREDICTED action each step — without (by default) moving the arm.

Why: if the arm "drifts one way with no object", the cause is usually one of:
  (a) frozen/black camera  -> policy sees a stale image -> same action every step -> drift
  (b) out-of-distribution scene (wrong camera pose / no object) -> degenerate biased action
  (c) a servo fault (overheat/overload/torque off) -> a joint sags or won't hold
This captures the data to tell them apart.

SAFE by default (--observe): ramps to the ready pose, holds it (torque on), and only READS +
runs inference, logging what the policy WOULD command. Pass --move to actually send actions.

Run (user, supervise with a hand on the power switch):
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb python \
    -m lerobot_teleoperator_so101_webcam.programs.diagnose_deploy \
      --policy <path-or-hub-id> --duration 15
"""

import argparse
import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

from ..paths import evidence_dir
from .record_so101_ee import ResilientSOFollower, _read_positions, _ramp_to
from .deploy_so101_ee import (_load_policy, _apply_diffusion_overrides, _ready_pose, ARM_ID,
                              WORKSPACE_CAM_PATH, WORKSPACE_CAM_FOURCC, FPS, DEFAULT_POLICY, TASK)


def _safe_read(bus, reg):
    try:
        return bus.sync_read(reg)
    except Exception as e:
        return {"_err": str(e)[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--inference-steps", type=int, default=None, help="diffusion denoising steps")
    ap.add_argument("--scheduler", choices=["ddpm", "ddim"], default=None,
                    help="diffusion sampler override; 'ddim' + '--inference-steps 10' = fast")
    ap.add_argument("--move", action="store_true", help="actually send actions (default: observe only)")
    ap.add_argument("--csv", default=str(evidence_dir() / "diag_log.csv"))
    ap.add_argument("--frame-dir", default=str(evidence_dir() / "diag_frames"))
    args = ap.parse_args()

    import os, cv2
    os.makedirs(args.frame_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[diag] loading {args.policy} on {device} ...")
    policy, pre, post = _load_policy(args.policy, device)
    _apply_diffusion_overrides(policy, args.scheduler, args.inference_steps, tag="diag")

    cameras = {"front": OpenCVCameraConfig(index_or_path=WORKSPACE_CAM_PATH, width=640, height=480,
                                           fps=FPS, fourcc=WORKSPACE_CAM_FOURCC, warmup_s=3)}
    robot = ResilientSOFollower(SO101FollowerConfig(port=args.port, id=ARM_ID, use_degrees=True,
                                                    cameras=cameras, disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    motors = list(robot.bus.motors.keys())
    print(f"[diag] motors present: {motors}")

    # one-time health snapshot
    te = _safe_read(robot.bus, "Torque_Enable")
    tmp = _safe_read(robot.bus, "Present_Temperature")
    print(f"[diag] torque_enable: {te}")
    print(f"[diag] temperature  : {tmp}")

    print("[diag] ramping to ready pose ...")
    _ramp_to(robot, _ready_pose(robot))
    _read_positions(robot)

    mode = "MOVE (sending actions)" if args.move else "OBSERVE (arm holds, not sending)"
    print(f"[diag] running {args.duration:.0f}s in {mode} ...")

    cols = (["step", "t", "cam_mean", "cam_dframe"]
            + [f"{m}_{s}" for m in motors for s in ("pos", "act", "delta", "load", "temp")])
    f = open(args.csv, "w"); f.write(",".join(cols) + "\n")

    prev_gray = None
    deltas_accum = {m: [] for m in motors}
    cam_means, cam_dframes = [], []
    period = 1.0 / FPS
    t0 = time.perf_counter(); t_end = t0 + args.duration; n = 0
    try:
        while time.perf_counter() < t_end:
            tstep = time.perf_counter()
            obs = robot.get_observation()
            state = np.array([obs[f"{m}.pos"] for m in motors], dtype=np.float32)
            img = obs["front"]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cam_mean = float(gray.mean())
            dframe = float(np.abs(gray.astype(np.int16) - prev_gray).mean()) if prev_gray is not None else -1.0
            prev_gray = gray.astype(np.int16)

            action = predict_action({"observation.state": state, "observation.images.front": img},
                                    policy, device, pre, post, use_amp=False,
                                    task=TASK, robot_type=robot.name)
            a = action.cpu().numpy().reshape(-1)
            delta = a - state

            load = _safe_read(robot.bus, "Present_Load")
            temp = _safe_read(robot.bus, "Present_Temperature")

            row = [n, round(tstep - t0, 3), round(cam_mean, 1), round(dframe, 2)]
            for i, m in enumerate(motors):
                row += [round(float(state[i]), 2), round(float(a[i]), 2), round(float(delta[i]), 2),
                        load.get(m, ""), temp.get(m, "")]
                deltas_accum[m].append(float(delta[i]))
            f.write(",".join(map(str, row)) + "\n")
            cam_means.append(cam_mean);
            if dframe >= 0: cam_dframes.append(dframe)

            if args.move:
                robot.send_action({f"{m}.pos": float(a[i]) for i, m in enumerate(motors)})
            if n % FPS == 0:
                cv2.imwrite(f"{args.frame_dir}/frame_{n:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            n += 1
            dt = time.perf_counter() - tstep
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[diag] stopped by user.")
    finally:
        f.close()
        robot.disconnect()

    # ---- summary ----
    print("\n========== DIAGNOSIS SUMMARY ==========")
    cm = np.array(cam_means)
    print(f"camera brightness  : mean={cm.mean():.1f}  min={cm.min():.1f}  max={cm.max():.1f}"
          f"   {'<<< DARK/BLACK?' if cm.mean() < 25 else ''}")
    if cam_dframes:
        dfa = np.array(cam_dframes)
        frozen = dfa.mean() < 0.5
        print(f"frame-to-frame diff: mean={dfa.mean():.2f}  max={dfa.max():.2f}"
              f"   {'<<< CAMERA FROZEN (stale feed!)' if frozen else '(feed is live)'}")
    print(f"steps={n}  achieved {n/args.duration:.1f} Hz   mode={'MOVE' if args.move else 'OBSERVE'}")
    print("\nper-joint PREDICTED action bias (mean delta = direction policy keeps pushing):")
    print(f"  {'joint':<14}{'mean_d':>8}{'std_d':>8}{'|max_d|':>8}   one-way?")
    for m in motors:
        d = np.array(deltas_accum[m]); md = d.mean()
        oneway = abs(md) > 2.0 and abs(md) > 1.5 * (d.std() + 1e-6)  # consistent push >> jitter
        print(f"  {m:<14}{md:>8.2f}{d.std():>8.2f}{np.abs(d).max():>8.2f}   "
              f"{'<<< DRIFTS ' + ('+' if md>0 else '-') if oneway else ''}")
    print(f"\nfull per-step log: {args.csv}\nsaved frames: {args.frame_dir}/")
    print("=======================================")


if __name__ == "__main__":
    main()
