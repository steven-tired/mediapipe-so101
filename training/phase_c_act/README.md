# Phase C local ACT training

This directory trains the first Phase C baseline on the local RTX 4060. It does
not modify the LeRobot checkout or the sanitized dataset.

## Frozen baseline contract

- Dataset: `stevenzenith/hand_tracking_pv_carton_phase_b`, loaded from the local
  sanitized tree only (30 episodes / 6640 frames / 10 Hz).
- Inputs: current `observation.images.front`, current
  `observation.images.side`, and current 6-D `observation.state`.
- Target: 6-D `action`.
- Policy: native LeRobot `act`, 20-step action chunk (about 2 seconds at 10 Hz),
  `n_action_steps=1`, temporal ensembling disabled.
- Runtime: CUDA AMP, batch size 8 initially, no W&B and no Hub upload.

## Commands

```bash
# Twenty-update CUDA and data-decoding smoke test.
bash /home/zhuokai/hand-teleop/training/phase_c_act/run_act.sh smoke

# 50,000-update training run after the smoke test passes.
bash /home/zhuokai/hand-teleop/training/phase_c_act/run_act.sh full
```

If the measured GPU memory from the smoke test requires a smaller batch, set
the chosen value explicitly for both runs:

```bash
ACT_BATCH_SIZE=4 bash /home/zhuokai/hand-teleop/training/phase_c_act/run_act.sh smoke
ACT_BATCH_SIZE=4 bash /home/zhuokai/hand-teleop/training/phase_c_act/run_act.sh full
```

Each invocation creates a timestamped model directory under `outputs/` and a
matching persistent console log under `logs/`.
