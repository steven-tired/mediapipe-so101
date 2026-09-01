# Training status

Honest summary of what has been trained and what that does and does not show.
Read `CLAIMS_AND_GATES.md` first; it defines the evidence levels used here.

## Tracked here

`training/` holds only what is needed to reproduce a run: scripts, notebooks,
Kaggle kernel metadata, and the two root notebooks. Every checkpoint, log, cache,
and run output was moved to `local/training_runs/` and `local/checkpoints/`, which
Git ignores. `docs/LOCAL_ARTIFACT_MIGRATION_MANIFEST.tsv` records where each went.

## Runs

Phase C explored several policy families on the PressureVision carton task:

| Run | What it is |
| --- | --- |
| `phase_c_act`, `phase_c_act_augmented`, `phase_c_act_onset` | ACT variants |
| `phase_c_diffusion_checkpoints`, `phase_c_diffusion_kaggle` | diffusion policy |
| `phase_c_smolvla_checkpoints`, `phase_c_smolvla_kaggle`, `phase_c_smolvla_extend_kaggle` | SmolVLA |
| `phase_c_recovery_minimal`, `phase_c_grasp_ready`, `phase_c_grip_residual`, `phase_c_diagnostics` | targeted probes rather than full policies |

Training ran on Kaggle T4s; a run killed at the 12-hour limit is resumed from a
checkpoint dataset rather than restarted.

## What is not claimed

**Offline metrics are not robot success.** Checkpoint selection here was driven by
offline evaluation on held-out episodes. That is *locked inference* evidence: it
does not establish closed-loop success rate, and no number in `local/` should be
quoted as one.

The controlled comparison these runs feed is unfinished. The W0 protocol froze
2026-08-06, W1 (fixed-position trials) is complete, the W3 pilot has not run, and
the v1.1 amendment has not been adopted. No conclusion about PressureVision-assisted
grip versus MediaPipe-only grip follows from what is here.

Deployment speed is the one firm operational result: diffusion wrappers run DDIM at
10 steps (~9 Hz) instead of 100-step DDPM (~3.6 Hz), which is what made autonomous
deployment usable at all.
