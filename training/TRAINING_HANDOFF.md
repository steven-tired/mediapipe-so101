# Phase C training and evaluation handoff

Updated: 2026-08-27 CDT

This is the canonical handoff for Phase C policy training and deployment
evaluation. Checkpoint storage details are kept in
[`PHASE_C_CHECKPOINTS.md`](PHASE_C_CHECKPOINTS.md).

## Current decision

- None of the retained ACT, SmolVLA, or Diffusion checkpoints is approved for
  an autonomous grasp rollout. The 2026-08-25 tests localize the shared failure
  to unreliable open-to-close onset prediction, not to action
  postprocessing, bus clipping, or a failed gripper motor.
- The bounded ACT aug-50k onset-state-replacement repair completed and is a
  NO-GO. Its best checkpoint, 4k, remained at 1/12 held-out close crossings in
  the one-step executed prefix, equal to the parent, while the other four
  checkpoints scored 0/12. Do not deploy these checkpoints.
- The replacement reduced the gripper-input copy slope but made the policy
  more conservatively open instead of transferring control to the image. The
  subsequent frozen grasp-ready signal probe also failed: static and two-frame
  history heads both scored 0/24 train-OOF and 0/6 held-out initial events at
  the zero-hard-negative-false-trigger threshold. Current data is insufficient
  for a deployment-safe autonomous close-ready decision. Do not continue
  reweighting the same near-duplicate successful trajectories; collect targeted
  hard-negative and recovery fragments next.
- Do not select Diffusion 20k from the older locked sweep. Under the corrected
  two-frame offline contract it had a 45.36% open-phase false-close rate.
  Diffusion 90k can close, but repeated identical-scene sampling was unstable.
- A later fixed-middle recovery branch is now the active bounded experiment:
  `act_middle_labels_20260826_attempt01/checkpoints/002000`. It has produced
  real lifts under the standard middle setup, but also no-lift, slip, and
  no-release outcomes. This is not approval for general autonomous deployment.
- Grip-residual collection started on 2026-08-27. The numeric head and
  shadow-inference path exist, but no head checkpoint has been trained yet.
  Continue reviewed collection before fitting it.

## Artifact state

The existing final policies are private on Hugging Face:

| Policy | Final step | Private repository |
| --- | ---: | --- |
| ACT baseline | 80k | `stevenzenith/act_carton_phase_c_80k` |
| SmolVLA | 80k effective | `stevenzenith/smolvla_carton_phase_c_80k` |
| Diffusion | 90k | `stevenzenith/diffusion_carton_phase_c_90k` |

Local checkpoint candidates are retained at multiple steps. Their exact paths
and SmolVLA effective-step mapping are in `PHASE_C_CHECKPOINTS.md`. Artifact
loading was verified, but loading alone is not physical evaluation.

The ACT image-augmentation run completed all 80k steps and retained checkpoints
every 5k. Held-out evaluation over episodes `0, 5, 10, 16, 23, 26` selected
50k by the original scalar score (`2.798963`), narrowly ahead of 40k
(`2.821648`). This score did not measure executed-prefix closure, which is why
the additional event diagnostics below were required.

## Robot evaluation on 2026-08-24

Only trials with `trial_valid: true` below count. No policy completed a grasp
and lift today.

| Policy | Valid trials | Full successes | Observed failure |
| --- | ---: | ---: | --- |
| ACT baseline 80k | 1 | 0 | Body joints moved, but the gripper stayed open and the carton stayed on the table. |
| SmolVLA 30k effective | 2 | 0 | One source-12 run contacted the carton top without grasping; the post-repair episode-25 matched run diverged during approach. |
| SmolVLA 80k effective | 1 | 0 | Repeatedly approached without forming a grasp. |
| Diffusion | 0 | n/a | Locked inference only; no command was sent to the arm. |

### ACT baseline 80k

The sole valid 30-second rollout used the current robot state and a source
episode-12 visual/start reference. The policy sent 298 commands at 9.93 Hz,
but its predicted gripper target stayed in `[99.35, 100.83]`; gripper readback
stayed in `[98.43, 99.21]`. The carton never left the table.

Evidence:

- `evidence/phase_c_policy_deploy/act80_front_deep_currentstart_20260824_attempt01/manifest.json`
- `evidence/phase_c_policy_deploy/act80_front_deep_currentstart_20260824_attempt01/outcome.json`
- `evidence/phase_c_policy_deploy/act80_front_deep_currentstart_20260824_attempt01/control.jsonl`
- dual-view videos in the same directory

The locked cable A/B/C exercise showed that visual cable changes altered the
predicted joint trajectory, but it did not support the stronger claim that the
cable alone caused the rollout failure. Across all three source-12 locked
conditions, no predicted gripper target was below 90; gripper ranges were
`[97.13, 100.77]`, `[97.27, 100.93]`, and `[97.94, 100.69]`. The pairwise
all-joint action RMSE was 2.35 to 4.16. Treat the cable as a possible visual
covariate, not an established root cause.

Evidence directories:

- `evidence/phase_c_policy_deploy/act80_cable_ab_source12_20260824_A_free_locked`
- `evidence/phase_c_policy_deploy/act80_cable_ab_source12_20260824_B_fixed_locked`
- `evidence/phase_c_policy_deploy/act80_cable_ab_source12_20260824_C_rerouted_locked`

### Gripper hardware check

The gripper fasteners were repaired before the final SmolVLA episode-25
rollout. An enabled open-close-open sweep commanded `90 -> 20 -> 90` and
observed readbacks `87.02 -> 22.49 -> 87.08`. This establishes that the
repaired gripper could track a closing command during the check. It does not
prove load-bearing grasp performance.

Evidence:

- `evidence/phase_c_policy_deploy/gripper_repair_sweep_20260824_attempt01/sweep.json`

### SmolVLA

The valid source-12 30k rollout contacted/pressed the carton top but did not
grasp. Its predicted gripper target stayed in `[91.31, 103.88]`. The valid 80k
rollout approached without grasping and predicted `[93.23, 103.37]`. After the
gripper repair, the episode-25 matched 30k rollout still diverged from the
demonstrated approach and predicted `[95.30, 103.39]`. None of these valid
rollouts issued a gripper target below 90.

Evidence:

- `evidence/phase_c_policy_deploy/smolvla30k_source12_currentstart_chunk4_20260824_attempt01/outcome.json`
- `evidence/phase_c_policy_deploy/smolvla80k_source12_currentstart_chunk2_20260824_attempt02/outcome.json`
- `evidence/phase_c_policy_deploy/smolvla30k_episode25_chunk4_postrepair_20260824_attempt02/outcome.json`
- control logs and dual-view videos in those directories

The locked episode-25 contract test used five seeds. For 30k, median
source-vs-demo action RMSE was 3.143 and median source-vs-live prediction RMSE
was 1.397. For 80k, they were 2.886 and 1.494. The source and live observations
therefore produced relatively similar predictions, but neither checkpoint
matched the demonstration closely enough to recover the grasp. This weakens a
pure start-state mismatch explanation and does not show an advantage from 80k.

Evidence:

- `evidence/phase_c_policy_deploy/smolvla30k_episode25_chunk4_postrepair_20260824_attempt02/contract_test.json`
- `evidence/phase_c_policy_deploy/smolvla30k_episode25_chunk4_postrepair_20260824_attempt02/contract_test_80k.json`

### Diffusion

All Diffusion observations so far were locked (`arm_enabled: false`). GPU
inference with DDIM-10 ran at about 9.0 Hz. In the matched source-12 locked
sweep, the predicted gripper results were:

| Checkpoint | Frames | Predicted gripper range | Targets below 90 |
| ---: | ---: | --- | ---: |
| 20k | 90 | `[51.91, 100.00]` | 23 |
| 40k | 90 | `[96.63, 100.00]` | 0 |
| 60k | 90 | `[97.88, 100.00]` | 0 |
| 80k | 90 | `[98.98, 100.00]` | 0 |
| 90k | 273 | `[99.01, 100.00]` | 0 |

The 90k DDPM-100 test achieved only 3.47 Hz, while the 90k DDIM-10 test
achieved 9.07 Hz. These locked results nominate 20k for a physical test; they
do not establish trajectory quality or task success.

Evidence directories:

- `evidence/phase_c_policy_deploy/diffusion020000_source12_locked_gpu_ddim10_20260824_attempt01`
- `evidence/phase_c_policy_deploy/diffusion040000_source12_locked_gpu_ddim10_20260824_attempt01`
- `evidence/phase_c_policy_deploy/diffusion060000_source12_locked_gpu_ddim10_20260824_attempt01`
- `evidence/phase_c_policy_deploy/diffusion080000_source12_locked_gpu_ddim10_20260824_attempt01`
- `evidence/phase_c_policy_deploy/diffusion90_source12_locked_gpu_20260824_attempt01`
- `evidence/phase_c_policy_deploy/diffusion90_source12_locked_gpu_ddpm100_20260824_attempt01`

## Gripper-onset diagnostics on 2026-08-25

The unified evaluator is
`training/phase_c_diagnostics/gripper_diagnostics.py`. It uses the policy's
actual observation history and execution-prefix length, preserves physical
action units, and records the first predicted chunk index below 90 degrees as
`d_t` (`-1` means no crossing). The fixed held-out set contains 1,214 valid
samples: 18 close-onset frames, 291 open-phase frames, 899 closed-hold frames,
and 75 release-onset frames.

| Policy | Close direction | Close crossing | Close in executed prefix | Open false close | Release direction | Release crossing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT aug 50k | 3/18 | 1/12 | 1/18 | 0/291 | 51/75 | 11/12 |
| SmolVLA 30k effective | 7/18 | 10/12 | 15/18 | 24/291 | 67/75 | 12/12 |
| SmolVLA 80k effective | 9/18 | 7/12 | 10/18 | 5/291 | 68/75 | 12/12 |
| Diffusion 20k | 11/18 | 10/12 | 14/18 | 132/291 | 28/75 | 10/12 |
| Diffusion 40k | 10/18 | 10/12 | 13/18 | 62/291 | 55/75 | 11/12 |
| Diffusion 90k | 17/18 | 12/12 | 18/18 | 42/291 | 72/75 | 12/12 |

The ACT baseline 80k scored 18/18 close direction, 10/12 close crossing,
10/18 close in its executed prefix, and 0/291 open false closes, but it trained
on these episodes. It is retained only as a capacity reference, not a held-out
result.

On episode-0 frames `55/60/64/68`, scanning the current gripper input through
`100/95/90/85/80` produced first-action slopes of `0.68-1.18` across all five
tested policy variants. This is direct evidence of an `action ~= current
gripper state` shortcut. Synchronized whole-image shifts of 20 pixels delayed
Diffusion 90k frame-64 closure from `d_t=0` to `d_t=2-4`; `-40 px` horizontal
and `+60 px` vertical shifts removed the crossing. Synchronized `+/-1` and
`+/-2` frame offsets did not remove Diffusion 90k's frame-64 crossing, so the
current evidence favors visual/spatial sensitivity over a small timing offset.

Offline evidence:

- `evidence/phase_c_policy_diagnostics/offline_20260825/act_events`
- `evidence/phase_c_policy_diagnostics/offline_20260825/smol_events`
- `evidence/phase_c_policy_diagnostics/offline_20260825/diffusion_events`
- `evidence/phase_c_policy_diagnostics/offline_20260825/targeted_probes`

The deploy evidence schema was extended to record normalized execution chunks,
chunk and execution indices, denormalized chunks, post-override plans,
`robot.send_action()` bus targets, and immediate motor readback. In the
gripper-only hardware branches, body targets equaled body readback at every
step. Predicted gripper targets also equaled returned bus targets; no clip or
postprocessing loss was found.

Diffusion 90k branch results near demonstration episode-0 poses:

- frame 55: chunk `[100, 100, 100, 100, 98.43, 92.86, 86.63, 79.94]`;
  the five-cycle branch stopped before the first crossing at index 6.
- frame 60: targets `99.09 -> 97.10 -> 92.40 -> 85.55 -> 79.23`; bus targets
  matched, while readback reached only `90.43`, establishing secondary motor
  tracking lag rather than a lost command.
- frame 64: live chunk `[98.94, 99.59, 99.51, 99.86, 99.75, 95.84, 91.01,
  84.53]`; the crossing moved to index 7 although the offline sample crossed
  at index 0.
- frame 68: the already-closed branch produced `[67.90, 63.23, 57.62, 50.76,
  43.43, ...]`, and readback reached `56.66`. The model can maintain a grasp
  state once closure has already begun.

With the same live image, body pose, and `99.21`-degree gripper state, five
independent Diffusion 90k DDIM-10 chunks had `d_t = [7, -1, -1, 5, -1]`.
DDIM-50 made inference slower and all three sampled chunks had `d_t=-1`.
Increasing denoising steps is therefore not the repair. Locked ACT aug 50k and
SmolVLA 80k checks in the same scene produced gripper execution prefixes
`[101.04]` and `[97.15, 97.40]`, respectively.

Hardware evidence:

- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_ep0_f55_gripper_only_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_ep0_f60_gripper_only_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_ep0_f64_gripper_only_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_ep0_f68_gripper_only_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_locked_5chunks_f64_scene`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/diff90_locked_3chunks_ddim50_f64_scene`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/aug50_locked_f64_scene`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/smol80_locked_f64_scene`

After the branch tests, a zero-policy-step ramp released the carton. The final
read-only position check was `[-2.20, 12.75, 1.27, 76.35, -2.15, 99.21]`
degrees in motor order. This is a point-in-time check, not a persistent robot
state guarantee. No complete autonomous grasp-and-lift success was claimed.

## Joint ACT execution-prefix audit on 2026-08-25

The earlier prefix-only interpretation was incomplete because it counted a
future gripper crossing without scoring the five body actions that would be
executed with it. The existing evaluator now reports mean and 95th-percentile
five-joint body MAE over the actual non-padded execution prefix, plus body MAE
inside the close-onset subset. Focused verification is `16 passed`; no second
diagnostic program was added.

The old `open_phase` compatibility mask is also not a valid false-close gate:
its 291 held-out rows overlap 18 readiness-positive rows. Prefix selection now
uses the 273 mutually exclusive pure hard negatives outside every `t-2:t+2`
readiness band. Offline prefix replay remains teacher-forced and is not a
closed-loop success rate. A valid physical trial still requires a valid setup,
an appropriate body approach, actual gripper closure/readback, and carton lift;
excluded setup or operator-interrupted attempts never enter that denominator.

Held-out ACT aug-50k results:

| Executed prefix | Close crossing | Pure false-close frames | Sustained false pairs | Body MAE / p95 | Close-window body MAE | First-crossing timing across six episodes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1/12 | 0/273 | 0 | 2.422 / 5.722 deg | 2.359 deg | 1 on-time, 5 late |
| 2 | 3/12 | 0/273 | 0 | 2.756 / 6.517 deg | 2.676 deg | 1 on-time, 5 late |
| 4 | 6/12 | 6/273 | 2 | 3.443 / 8.349 deg | 3.332 deg | 1 on-time, 1 early, 4 late |
| 6 | 8/12 | 20/273 | 11 | 4.131 / 9.719 deg | 3.894 deg | 2 on-time, 2 early, 2 late |
| 8 | 10/12 | 26/273 | 17 | 4.796 / 10.560 deg | 4.348 deg | 1 on-time, 2 early, 3 late |
| 12 | 10/12 | 54/273 | 43 | 5.914 / 12.086 deg | 5.006 deg | 2 on-time, 2 early, 2 late |
| 20 | 12/12 | 90/273 | 81 | 7.201 / 14.059 deg | 5.548 deg | 2 on-time, 3 early, 1 late |

The first-crossing replay starts a new chunk at episode frame 0 and then every
`k` frames, stopping at the first predicted gripper target below 90 degrees.
It therefore tests the recorded-observation scheduling consequence of prefix
`k`, but cannot model the observation drift caused by executing an incorrect
body trajectory.

The in-sample ACT baseline remains only a capacity reference: prefix 2 had
12/12 close crossings, 0/273 pure false-close frames, all 6/6 first crossings
inside the readiness band, and body MAE/p95 `0.747/1.523` degrees. The rejected
onset-replacement 4k checkpoint did not improve the joint result: prefix 4 had
5/12 crossings, 9/273 pure false-close frames, no on-time episode, and body
MAE/p95 `3.465/8.451` degrees.

In the current live scene, the robot body state was within about one degree of
demonstration episode-0 frame 64 and the carton occupied a similar grasp
region in both views. Locked aug-50k prefixes `1/2/4/6/8/12/20` all completed
with zero commands. Prefixes through 12 remained above 90 degrees. The locked
20-step chunk finally crossed at indices 18-19 (`92.05 -> 88.57 -> 85.94`),
but its simultaneous open-loop plan reached about 23.77 degrees of
shoulder-lift and 14.44 degrees of wrist-flex displacement from the current
readback, so full-6D execution was rejected.

One bounded enabled test executed that 20-step chunk with `--gripper-only`,
holding every body target exactly equal to the current readback. All 20 targets
reached the bus, but the newly sampled live chunk ended at 93.07 degrees and
gripper readback only moved `99.21 -> 97.11`; it neither crossed 90 nor grasped
the carton. A zero-policy-step ramp then reopened the gripper, and a final
ARM-LOCKED readback measured `99.34` degrees with zero commands sent.

Evidence:

- `evidence/phase_c_policy_diagnostics/offline_20260825/act_prefix_joint_sweep`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/aug50_live_prefix{1,2,4,6,8,12,20}_locked_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/aug50_live_prefix20_gripper_only_enabled_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/aug50_live_prefix20_release_to100_a1`
- `evidence/phase_c_policy_diagnostics/hardware_20260825/aug50_live_postrelease_locked_readback_a1`

No ACT execution prefix is approved for a full autonomous rollout. Increasing
the prefix recovers gripper crossings only by accepting progressively worse
body imitation, early closures, and sustained hard-negative triggers; the one
gripper-only live attempt also failed to reproduce the locked crossing.

## ACT onset-state-replacement repair on 2026-08-25

The first isolated repair from ACT aug 50k is complete and rejected. The
train-only derived dataset contains the same 24 episodes / 5,426 rows and
changes only state dimension 5 on 509 rows whose future 20-step action chunk
contains an open-to-close onset. Replacement values cycle deterministically
through `90/95/100` (`170/170/169` rows). Actions, the first five state
dimensions, unselected gripper states, metadata, and all video content are
unchanged; the 48 video files are hardlinks to the source data. The six held-out
episodes were never added to this dataset.

- Source: `datasets/hand_tracking_pv_carton_phase_b_train24`
- Derived dataset:
  `datasets/hand_tracking_pv_carton_phase_b_train24_onset_qreplace`
- Builder and unit tests:
  `training/phase_c_act_onset/prepare_onset_dataset.py` and
  `training/phase_c_act_onset/tests/test_prepare_onset_dataset.py`
- Dataset evidence: `onset_state_replacement_manifest.json` and
  `verification.json` at the derived dataset root
- Smoke config/output:
  `training/phase_c_act_onset/runs/act_phase_c_onset_qreplace_from_aug50_20260825_132506/smoke_config.json`
  and
  `training/phase_c_act_onset/outputs/act_phase_c_onset_qreplace_from_aug50_20260825_132506_smoke`
- Full config/output:
  `training/phase_c_act_onset/runs/act_phase_c_onset_qreplace_from_aug50_20260825_132506/full_config.json`
  and
  `training/phase_c_act_onset/outputs/act_phase_c_onset_qreplace_from_aug50_20260825_132506`

The two dataset unit tests passed. The 20-step GPU smoke completed and saved a
loadable checkpoint. The separate 5k RTX 4060 run completed in `23:57` at
about 3.48 step/s, saved all five 1k checkpoints, and ended at loss `0.049`.

Evaluation used the unchanged original Phase B dataset and held-out episodes
`0,5,10,16,23,26`:

| Checkpoint | Close direction | Close crossing / executed prefix | Open false close | Release direction | Release crossing | Body MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent ACT aug 50k | 3/18 | 1/12 | 0/291 | 51/75 | 11/12 | 2.421826 |
| Onset 1k | 0/18 | 0/12 | 0/291 | 46/75 | 10/12 | 2.525081 |
| Onset 2k | 1/18 | 0/12 | 0/291 | 48/75 | 12/12 | 2.466546 |
| Onset 3k | 1/18 | 0/12 | 0/291 | 50/75 | 12/12 | 2.507517 |
| Onset 4k | 3/18 | 1/12 | 0/291 | 51/75 | 11/12 | 2.473058 |
| Onset 5k | 1/18 | 0/12 | 0/291 | 52/75 | 11/12 | 2.446019 |

The best 4k result merely reproduced the parent's close timing and increased
overall body MAE by 2.1%; it does not pass the retention gate. On episode-0
frame 64, its first-action gripper-input slope fell from `0.8115` to `0.5361`,
but its `q=80` first action became more open (`85.44 -> 90.58`) and its crossing
moved later (`d_t=0 -> 2`). At the live-like `q=100` input the crossing also
moved later (`d_t=6 -> 7`). This refutes the hypothesis that state replacement
alone would transfer onset control to visual geometry. No locked or enabled
robot test was run with the rejected repair.

Offline evidence:

- `evidence/phase_c_policy_diagnostics/offline_20260825/act_onset_qreplace_1k_to_5k_events`
- `evidence/phase_c_policy_diagnostics/offline_20260825/act_onset_qreplace_4k_targeted_probes`

## Autonomous grasp-ready signal probe on 2026-08-25

The frozen offline probe tested whether the existing observations contain an
episode-general signal for when closure is appropriate. It used only frozen
ImageNet ResNet-18 embeddings from front/side RGB and the first five body motor
positions. The history variant additionally used the same image features from
two frames earlier and the five-dimensional body delta. Gripper state, action,
PressureVision, human pinch, episode/frame identity, and timestamps were
excluded from model inputs.

The fixed split remained 24 training episodes / 5,426 frames and held-out
episodes `0,5,10,16,23,26` / 1,214 frames. The train set contains 25 closure
events; held-out contains six. A label-audit correction was frozen before any
held-out scoring: the legacy open-phase masks contain 1,443 train / 291
held-out rows, but 75 / 18 of those rows overlap the `t-2:t` readiness-positive
windows. Zero-false calibration therefore uses the 1,368 train / 273 held-out
pure hard negatives outside every readiness band. The full legacy masks remain
compatibility diagnostics only.

| Variant | Train-OOF threshold | Train initial hits | Train pure false | Held-out initial hits | Held-out pure false | Held-out legacy false |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.999979496 | 0/24 | 0/1,368 | 0/6 | 0/273 | 0/291 |
| Two-frame history | 0.999999762 | 0/24 | 0/1,368 | 0/6 | 0/273 | 0/291 |

The result is **FAIL**. No head was saved and no robot test was run. The
train/held-out positive-versus-pure-negative rank AUCs were `0.622/0.745` for
static and `0.619/0.713` for history, so the recordings contain a weak ranking
signal but not a safe trigger boundary. As a non-gating diagnostic, relaxing
single-frame zero-false to zero consecutive two-frame false triggers still
gave train hits `0/24` static and `1/24` history, and held-out hits `0/6` for
both. Short history did not resolve the onset ambiguity.

Exact command:

```bash
HF_HOME=local/training_runs/.cache env -u PYTHONPATH .venv-lerobot/bin/python training/phase_c_grasp_ready/grasp_ready_probe.py evaluate --cache-path training/phase_c_grasp_ready/cache/resnet18_phase_b.npz --output-dir evidence/phase_c_gripper_signal_probe/offline_20260825_grasp_ready_v1 --seed 0 --epochs 120
```

Artifacts:

- Probe and tests: `training/phase_c_grasp_ready/grasp_ready_probe.py` and
  `training/phase_c_grasp_ready/tests/test_grasp_ready_probe.py`
- Frozen feature cache:
  `training/phase_c_grasp_ready/cache/resnet18_phase_b.npz` (6,640 rows,
  two 512-dimensional image embeddings per row)
- Sealed result:
  `evidence/phase_c_gripper_signal_probe/offline_20260825_grasp_ready_v1`
- A pre-held-out failed CLI attempt is retained at
  `evidence/phase_c_gripper_signal_probe/offline_20260825_grasp_ready_failed_preheldout_attempt01`;
  it contains only a label manifest and no predictions.

Artifact checks passed: both JSON files parse, train/held-out CSVs contain
5,426 / 1,214 unique episode-frame rows with finite scores, the failed gate
has no `probe_head.safetensors`, Python compilation passes, and all nine unit
tests pass.

## Excluded ACT attempts

The following runs are retained as evidence but must not enter a success-rate
denominator:

| Attempt | Exclusion |
| --- | --- |
| `act80_front_deep_center_smoke_20260824_attempt01` | Carton was placed too far from the intended start. |
| `act80_front_deep_center_smoke_20260824_attempt02` | Operator interrupted the rollout. |

Their `outcome.json` files record `trial_valid: false` and the exclusion reason.

## Completed ACT image-augmentation run

This is a new run, separate from the failed baseline ACT 80k deployment.

- Dataset: `datasets/hand_tracking_pv_carton_phase_b_train24`
- Training set: 24 episodes / 5,426 frames
- Held-out set: episodes `0, 5, 10, 16, 23, 26`, covering all six
  table-position x grasp-depth cells and all five jaw offsets
- Policy: ACT, 52M parameters, chunk size 20, one executed action step
- Batch: 8 with AMP
- Image augmentation: at most one mild transform per sample, plus weighted
  identity, covering brightness/contrast/saturation/sharpness/affine/random
  erasing
- Target: 80k, save every 5k, log every 100

The first complete checkpoint is 5k and contains model, optimizer, RNG, and
`training_step.json` state:

`training/phase_c_act_augmented/outputs/act_phase_c_aug_holdout_20260824_150345/checkpoints/005000`

Training resumed from that checkpoint at 16:56 CDT and completed at 22:55 CDT.
The final log records global step 80k, loss `0.042`, and checkpoint save at
`080000`; all 5k checkpoints from `005000` through `080000` are present.

- Resume log: `training/phase_c_act_augmented/runs/act_phase_c_aug_holdout_20260824_150345/resume_to_80k.log`
- Original 0-to-5k log: `training/phase_c_act_augmented/runs/act_phase_c_aug_holdout_20260824_150345/train_80k.log`
- Split evidence: `training/phase_c_act_augmented/runs/act_phase_c_aug_holdout_20260824_150345/split_manifest.json`
- Notebook and evaluator: `training/phase_c_act_augmented/train_act_augmented_local.ipynb` and `training/phase_c_act_augmented/evaluate_checkpoints.py`

Checkpoint metrics are in
`training/phase_c_act_augmented/outputs/act_phase_c_aug_holdout_20260824_150345/heldout_checkpoint_metrics.csv`.
Training loss and the original scalar selection score are not sufficient
gripper-onset metrics; use the event evaluator above for subsequent runs.

## SmolVLA retraining resource note

The previous SmolVLA run used two Tesla T4 processes, batch 2 per GPU and
effective batch 4. Trainer logs reported `1.88-1.89 GB` GPU memory per process.
The configuration froze the vision encoder and trained about 100M of 450M
total parameters (`train_expert_only=true`). A single 8 GB RTX 4060 therefore
has ample memory for batch 2 under the same model/data contract; allow at least
4 GB per GPU for runtime headroom. This estimate does not apply if the VLM or
vision encoder is unfrozen, image resolution is increased, or batch size is
substantially increased.

Evidence:

- `training/phase_c_smolvla_checkpoints/phase-c-smolvla-carton.log`
- `training/phase_c_smolvla_checkpoints/phase-c-smolvla-extend-carton.log`
- `training/phase_c_smolvla_checkpoints/phase_c_smolvla/outputs/smolvla_phase_c_20k_20260822_185909/checkpoints/020000/pretrained_model/train_config.json`

## Fixed-middle labeled ACT recovery on 2026-08-26

The fixed-middle supplemental dataset contains 13 episodes / 2,902 frames.
All 13 episodes are accounted for: middle episodes `0,1,2,3,4,5,6,7,10,11`
are in training, while `8,9,12` form the grouped held-out set. Training also
uses Phase B episodes `1,9,14,20,26`, for 15 training episodes / 3,387 frames
in total. The five reviewed recovery segments are:

| Source episode | Recovery interval J-K (s) | Reviewed stable q |
| --- | ---: | ---: |
| Phase B 14 | 13.3-15.9 | 23.034 |
| Phase B 26 | 10.3-15.9 | 23.072 |
| Middle 7 | 12.7-17.8 | 25.619 |
| Middle 10 | 10.2-16.9 | 25.118 |
| Middle 11 | 15.6-21.2 | 21.398 |

The minimal trainer is `training/train_middle_labeled_act.py`. It starts from
ACT aug-50k, preserves the parent's preprocessing and normal ACT L1 plus KL
loss, adds gripper L1 over each reviewed `J-K` recovery interval, and adds a
`0.25`-weighted penalty after `K` when the prediction is tighter than the
reviewed stable q. The run used batch 8, AMP, seed 42, 5,000 steps, and saved
every 1,000 steps:

`training/phase_c_recovery_minimal/act_middle_labels_20260826_attempt01`

Evaluation stayed on middle episodes `8,9,12` (654 frames):

| Policy | Body MAE / p95 | Close in prefix | Open false close | Release crossing |
| --- | ---: | ---: | ---: | ---: |
| Parent aug-50k | 1.750 / 3.641 | 1 | 0/172 | 4/4 |
| Fine-tune 1k | 1.252 / 2.771 | 6 | 12/172 | 4/4 |
| Fine-tune 2k | 1.355 / 2.951 | 2 | 2/172 | 4/4 |
| Fine-tune 3k | 1.355 / 2.817 | 3 | 3/172 | 4/4 |
| Fine-tune 4k | 1.404 / 2.866 | 1 | 0/172 | 4/4 |
| Fine-tune 5k | 1.428 / 2.986 | 1 | 3/172 | 4/4 |

Fine-tune 2k was selected by the operator as a bounded physical candidate,
not as a universal offline winner. The active checkpoint is:

`training/phase_c_recovery_minimal/act_middle_labels_20260826_attempt01/checkpoints/002000/pretrained_model`

A from-scratch comparison was also completed through 50k at
`training/phase_c_recovery_minimal/act_middle_labels_from_scratch_20260826_attempt01`.
It is a NO-GO: scratch 5k-35k falsely closed on all 172 held-out open-phase
frames, and scratch 40k/45k/50k still produced 74/108/100 false-close frames.
Its body MAE remained about `2.76-4.88`, versus `1.355` for fine-tune 2k. It
was not deployed.

The current fixed-middle physical contract is standard start joints
`[1.32,-38.42,42.68,86.20,0.92,99.34]`, executed prefix 14, 300 ACT steps,
and no gripper offset (`delta_q=0`). This setting produced real lifts and some
complete grasp-place-release loops, but also no-lift, partial-lift, slip, and
no-release outcomes. Prefix 14 was qualitatively a little better than prefix
12, but the trials do not establish a stable success rate. A forced looser
offset of `+1` disrupted the learned close-to-lift sequence and was rolled
back; do not pin q or restore that offset.

Offline evidence:

- `training/phase_c_recovery_minimal/offline_heldout_20260826_attempt01`
- `training/phase_c_recovery_minimal/offline_scratch_heldout_20260827_attempt01`

## Grip residual head preparation and collection on 2026-08-27

This subsection records the original collection state and is superseded by the
2026-08-28 result below. No head output has controlled the motor.
The small implemented head uses four control steps of six numeric inputs:
policy q target, commanded q target, q readback, `Present_Current`,
`Present_Load`, and position lag. It predicts `tighten/hold/loosen` plus a
grasp-stability probability. `--grip-residual-shadow-model` is log-only.

The intervention recorder now logs current/load/lag and uses the following
review protocol: press `c` to pause and preserve the buffered ACT chunk, press
`[` to tighten by `0.2` q or `]` to loosen by `0.2` q, then press `c` to
resume. The window continuously displays q target and q readback. Paused
cycles do not consume the 300 ACT-step budget. From trial 19 onward,
`--action-step-repeat 2` repeats each ACT target for two control frames, which
slows trajectory progression to about one half while cameras, keys, and
readback remain at 10 Hz.

Four trials are currently retained with reviewed outcomes:

| Trial | Intervention | Reviewed result | Head use |
| ---: | --- | --- | --- |
| 13 | Tighten x6 | Direction helped, but carton later slipped | Tighten label plus stability negative |
| 15 | Tighten x5 | Lifted without slip; did not release | Tighten label; exclude from full-loop success |
| 18 | None | Lifted, then slipped | Stability negative only; not a direction label |
| 19 | None, half speed | Stable lift with acceptable force | Stable positive; derive hold phase carefully |

Trial 10 has five loosen inputs and trial 20 has two loosen inputs at half
speed, but their operator outcomes were not recorded. They remain pending and
must not be silently counted. Trial 11 was explicitly discarded; trial 14 is
a lifted baseline rather than a head example; trials 7/8/9/12/16 did not lift;
trial 17 is unknown and excluded.

Evidence is under
`evidence/phase_c_recovery_minimal/grip_intervention_trial*_20260827`. The
implementation is in `grip_runtime.py` and `deploy_so101_ee.py` inside the
`ir-hand-pressure-so101-teleop` worktree. Focused software verification passed
11 tests; that verifies logging/control semantics, not physical grasp quality.

The collection target is eight reviewed valid trials, aiming for three
tighten-needed, three loosen-needed, and two stable-hold examples. Direct key
events supply direction labels only when compatible with the reviewed trial
outcome. A later slip invalidates a blanket stable/hold label; a no-key slip is
a stability negative, not an inferred direction. Effort should only rank
otherwise successful stable grasps. The head must not be expected to repair a
bad approach or a body trajectory that never lifts.

On 2026-08-28, trial-24 log review found that the original `c` latch copied
lagging q readback rather than the current ACT q command. Immediately before
the latch ACT commanded `24.03` while q read was `26.36`; the latch therefore
loosened the command by `+2.33`. Three subsequent `[` presses ended at `25.76`,
still `+1.73` looser than ACT. Pre-fix intervention key directions are not
valid policy-residual direction labels. Preserve their raw telemetry and
reviewed stability outcomes, but exclude their direction labels from head
training. No-intervention stability/slip trials remain usable.

The latch now starts from the current ACT q command, so `[`/`]` are true
`-0.2/+0.2` residual steps and pausing no longer cancels the policy's closure
effort. Focused verification passed 11 tests and Python compilation. This is
software evidence only; one corrected physical trial must confirm q command
continuity before collecting replacement direction labels or training a head.

## Superseded 2026-08-25 next gates

The following plan predates the fixed-middle fine-tune and grip-residual
collection above. Retain it as history; use the current gates after it.

1. Keep the onset-state-replacement run as a valid negative result; do not run
   it on hardware or extend it merely for more steps.
2. Keep the grasp-ready probe as a valid negative result. Do not relax its
   threshold after seeing held-out scores, save a head, or run it on hardware.
3. Before the next physical capture, extend the raw recorder contract to log
   synchronized right-hand pinch distance / `right_grasp_active` as
   training-only teacher metadata. Continue recording dual RGB, body/action,
   and later PressureVision validity/value; do not add pinch to deployment
   inputs.
4. Collect targeted short fragments instead of full near-duplicate successes:
   for each of the six table-position by grasp-depth cells, retain one correctly
   aligned close and one misaligned near-carton approach followed by human
   correction and close. Preserve the hard-negative dwell before correction.
   Add a fresh sealed validation episode per cell because episodes
   `0,5,10,16,23,26` have now been observed and cannot remain untouched for the
   next iteration.
5. Re-run the same grouped static/history probe with thresholds selected only
   from training episodes. Existing held-out scores may be reported as
   historical comparison but must not select the next model or threshold.
6. Separately evaluate approach quality from a standard start. The current
   frame-64 gripper-only evidence proves an onset failure near a demonstrated
   pose; it does not prove that any policy reaches the correct grasp pose from
   the start.

## Superseded 2026-08-27 next gates

1. Record the operator outcome for trial 20 and, if it can be reviewed
   reliably, trial 10. Do not infer either outcome from q alone.
2. Continue to eight reviewed, reasonably balanced trials. Standardize carton
   placement and the middle start; keep half-speed repeat 2 for head-data
   collection so the intervention window is usable.
3. First run one corrected intervention trial and verify from `control.jsonl`
   that the `c` latch preserves the ACT q command. Then collect replacement
   direction labels; do not train on pre-fix intervention directions.
4. Add the smallest offline trainer: four-step feature windows, grouped split
   by trial, direction cross-entropy on reviewed direct direction/hold labels,
   and stability binary loss from reviewed lift/retention outcomes. Do not use
   a random frame split.
5. Report trial-group direction confusion, stability/slip performance, and
   calibration. Do not actuate from an offline result.
6. Load a candidate only through `--grip-residual-shadow-model` first. Preserve
   the ACT body and release sequence; keep the head log-only until shadow
   evidence supports a bounded enabled test.
7. Continue reporting approach, closure, lift, retention, and release as
   separate gates. Base ACT lift variability remains a confound that the
   gripper head cannot fix.

## Grip residual corrected trials and offline result on 2026-08-28

The `c` latch fix was physically confirmed in trial 25: q command stayed at
`24.18` before, during, and after the latch while q readback was `26.23`.
Trials 25-32 were then reviewed as follows:

| Trial | Corrected intervention | Reviewed result | Head use |
| ---: | --- | --- | --- |
| 25 | Hold | One-side slight slip; normal release | Borderline stability negative |
| 26 | None | No grasp/lift | Tighten-needed and stability negative |
| 27 | Loosen `+1.0` | Full stable lift; no place | Loosen event; stable post phase |
| 28 | Loosen `+0.4` | Reduced an initially tight grasp; normal release | Loosen event; stable post phase |
| 29 | Tighten `-4.4` | Changed no-lift to partial lift | Tighten event; stability negative |
| 30 | Rejected | Operator stopped the trial | Excluded |
| 31 | Loosen `+0.2` | Full lift, no slip, normal place/release | Loosen event; stable post phase |
| 32 | None | Acceptable stable grasp and complete loop | Hold and stability positive |

The minimal trainer is
`lerobot_teleoperator_so101_webcam/train_grip_residual_head.py`. It uses only
four-step numeric telemetry histories, a trial-grouped split, direction
cross-entropy, and stability binary loss. Reviewed pre-fix no-grasp/partial-
lift/slip outcomes from trials 21-24 remain valid `tighten_needed` labels; the
old key directions remain excluded. Stable post-intervention phases are hold
labels. Samples are restricted to q command and readback below 32 so the
opening/release tail is not mislabeled as hold.

The first 2k run overfit: train direction/stability reached 100%, while held-out
direction was 64.7% and stability was 41.7%. It is a NO-GO checkpoint at
`training/phase_c_grip_residual/grip_residual_head_20260828.pt`; do not load it
for shadow selection or motor control. After adding the previously omitted
stable-post hold labels and excluding release frames, a 50/100/250/500-step
sweep also failed to beat trivial held-out baselines. The least-overfit 50-step
candidate had direction 52.0% versus 64.0% for always-hold, and stability
45.8% versus 66.7% for always-stable. Its direction confusion nevertheless
showed that the retained old loose failures are useful: held-out trial 26 was
8/8 `tighten`, and the single held-out corrected loosen event was 1/1
`loosen`. The failure is the hold-versus-loosen boundary: only 4/16 held-out
hold samples were classified as hold.

Trial 32 explains the ambiguity. One operator-approved hold sample had
`q_cmd=24.81`, `q_read=26.16`, current `4`, load `96`, and lag `1.36`, nearly
the same numeric state as the training loosen mean (`23.77`, `25.21`, `5.29`,
`103.43`, `1.44`). With these inputs alone, more optimizer steps cannot teach
the missing distinction between acceptable force and visually excessive
carton deformation.

## Superseded initial 2026-08-28 next gates

1. Do not collect more generic loose/no-lift examples; the tighten-needed class
   already transferred to its held-out trial.
2. Do not actuate either saved or temporary head candidate. The ACT checkpoint,
   prefix 14, repeat 2, 300-step body/lift/release sequence remains unchanged.
3. To pursue automatic loosening, add information that can distinguish the
   conflicting labels: either a visual deformation feature or paired corrected
   loosen-versus-hold trials with a stricter operator definition. Do not claim
   current/load/lag alone measure permanent carton deformation.
4. Any next model must again use a trial-grouped held-out split and beat the
   majority baselines for both direction and stability before log-only shadow
   evaluation.

## Augmented grip-head data and shadow candidate on 2026-08-28

Trials 33-36 exposed and corrected two additional recorder effects before the
new labels were used:

| Trial | Reviewed result | Use |
| ---: | --- | --- |
| 33 | Acceptable hold before `c`; slipped when pause replaced the body target with lagging readback | Hold/stable before `c` only |
| 34 | Stable lift with no q change; remained stable through `c` | Hold/stable positive; physical pause-continuity check |
| 35 | First `c` jumped from executed q `26.34` to new ACT q `27.25`; nine tighten inputs followed | Excluded because latch introduced `+0.91 q` |
| 36 | Lift required net `-3.8 q`; lift was not fully stable; normal place | Tighten positive, stability negative; one reversed loosen input excluded |

Paused intervention cycles now reuse the fixed last policy body target rather
than replacing it with changing readback. A new latch now starts from the
previous policy q that was actually executed, rather than a newly selected ACT
q on the keypress cycle. Trial 36 confirmed exact q continuity at all four
pause entries and equality of predicted and sent body targets. Focused software
verification still passes 11 tests.

The trainer now includes trials 33/34/36, restricts trial 36 to its reviewed
tighten events, uses square-root inverse-frequency direction weights, and uses
a positive-class weight for the 56 stable versus 80 unstable training samples.
The selected 50-step trial-grouped result is:

- held-out direction accuracy `68.0%` versus `64.0%` always-hold;
- held-out direction confusion: tighten `8/8`, hold `8/16`, loosen `1/1`;
- held-out stability accuracy at 0.5 `54.2%`, with Brier `0.173` versus about
  `0.222` for the constant-prevalence baseline.

The checkpoint is
`training/phase_c_grip_residual/grip_residual_head_augmented_20260828_50step.pt`.
It is a log-only shadow candidate, not an actuation candidate. The remaining
safety failure is seven false-loosen predictions among sixteen held-out hold
samples.

## Current next gates after augmented 2026-08-28 training

1. Keep the base ACT body and gripper commands unchanged. If this candidate is
   run, use only `--grip-residual-shadow-model` and inspect its logged outputs.
2. Collect corrected paired loosen-versus-hold trials; generic tighten-needed
   data is no longer the limiting class.
3. Do not enable motor residuals until false-loosen errors on held-out hold
   phases are materially reduced and stability decisions beat their majority
   baseline, not only the Brier baseline.

## Targeted shadow collection and retraining on 2026-08-28

The augmented 50-step checkpoint was run only through
`--grip-residual-shadow-model`; it never changed a motor command. The requested
quota of one clean hold and two clean loosen trials was completed, with extra
negative and hold evidence retained:

| Trial | Reviewed result | Quota/use |
| ---: | --- | --- |
| 37 | No adjustment; carton slowly slipped | Stability negative; shadow false-stable |
| 38 | Operator requested restart | Excluded |
| 39 | Tighten `-0.4`; lift and place succeeded; no release | Tighten/retention positive |
| 40 | Good stable lift; no adjustment | Quota clean hold; new held-out hold |
| 41 | Loosen `+0.4`; remained stable; no release | Quota clean loosen; training loosen |
| 42 | No adjustment; carton slipped | Stability negative |
| 43 | Tighten `-0.8`; only marginal lift | Tighten positive; stability negative/borderline |
| 44 | Good stable lift; no adjustment | Extra clean hold |
| 45 | Loosen `+1.0`; remained stable | Quota clean loosen; new held-out loosen |

The shadow head failed in complementary ways. During trial 37's reviewed slow
slip, its stability probability rose to about `0.8` and most predictions were
hold. At the two trial-41 loosen events it predicted hold, and it also predicted
hold at all five trial-45 loosen events. On clean hold trials 40 and 44 it made
only two and three false-loosen predictions respectively, but its stability
score was not consistently calibrated.

Retraining included trials 37/39/41/42/43/44 while holding out new clean trials
40 and 45 in addition to 26/31/32. The best 50-step result had direction
accuracy `66.7%`, exactly the always-hold baseline, and classified `0/6`
held-out loosen events correctly. Stability accuracy was `66.7%`, below the
`77.8%` always-stable baseline; Brier was `0.170`, only slightly better than
the constant-prevalence baseline `0.173`. A swapped clean-loosen fold identified
only `1/3` loosen events. The 100/250-step runs did not rescue direction
generalization. These runs remain temporary NO-GO candidates and did not
replace the saved shadow checkpoint.

The next model change should not be more steps on the same three-way numeric
classifier. Use action-conditioned candidate scoring for `delta_q` choices and
add an observation of object slip/deformation from the synchronized side/front
videos or later pressure/tactile sensing. Preserve a reject/hold default and
forbid learned loosening before a confirmed lift.

## Action-conditioned video candidate smoke on 2026-08-28

The minimal replacement scorer now evaluates `delta_q = {-0.2, 0, +0.2}` as
an input rather than directly classifying tighten/hold/loosen. Its four-step
context combines the existing q/current/load/lag telemetry with synchronized
front/side optical-flow features: frame motion, relative upper/lower motion,
and horizontal/vertical strain as coarse slip/deformation observations. Each
candidate event is scored from the telemetry frame before the adjusted command,
so the chosen delta is not leaked through q command.

The action labels are still insufficient for control. Training support is:

- `-0.2`: 8 unstable and 2 stable samples;
- `0`: 96 unstable and 96 stable samples;
- `+0.2`: 0 unstable and 9 stable samples.

At 50 steps, trial-grouped held-out accuracy was `86.1%` and Brier `0.122`
versus constant-prevalence Brier `0.247`. However, a numeric-only ablation was
better on the same held-out rows (`94.4%`, Brier `0.109`), and all candidates
were generally ranked more stable as delta increased. The 100-step result had
Brier `0.120`; 200 steps degraded to `0.160` while train Brier fell to `0.089`,
showing overfit. Thus the model separates many stable versus unstable trial
contexts, but has not learned the causal effect of a one-step adjustment.

This is a software/offline NO-GO, not a deployment checkpoint. Only temporary
checkpoints were written under `/tmp`; the saved numeric shadow checkpoint was
not replaced and no motor command was sent. The selector defaults to hold when
the score or advantage is uncertain, masks actions without both stable and
unstable training support, and always masks `+0.2` before a confirmed stable
lift. Because no valid failed-loosen example exists, automatic loosen remains
unsupported even after lift. The missing evidence is a post-lift `+0.2` step
that produces reviewed slip/instability, plus per-step post-action outcomes;
more optimizer steps cannot supply those counterfactual labels.

## Matched loosen-boundary trials and decisive retrain on 2026-08-31

Three valid post-lift boundary trials were collected with ACT unchanged and
manual `+0.2 q` steps. Absolute q was deliberately not treated as a reusable
threshold:

| Trial | Last stable q | First unstable q | Reviewed result |
| ---: | ---: | ---: | --- |
| boundary 01 | 26.27 | 26.47 | Both sides had cleared; one side returned to the table |
| boundary 03 | 24.95 | 25.15 | One side slipped down |
| boundary 04 | 25.45 | 25.65 | Carton slipped |

Boundary trial 02 never established an initial grasp/lift and is excluded from
loosen training. Each valid boundary is labeled per adjustment: earlier
`+0.2` steps that remained lifted are positive, while the first slipping step
is negative. The final pre-step context is also paired with `delta_q=0` as a
stable hold label. This directly supplies the contrast that was absent from
the earlier trial-level labels.

The single predeclared 50-step retrain used boundary trials 01 and 03 for
training and held out boundary trial 04 in full. Training candidate/outcome
counts were `-0.2: 8 unstable / 2 stable`, `0: 96 / 98`, and `+0.2: 2 / 17`.
Overall held-out accuracy was `82.9%` with Brier `0.143` versus constant Brier
`0.250`, but the decisive matched boundary test failed: at boundary 04's
first-slip context the model assigned stability `0.457` to hold and `0.494` to
`+0.2`. It still ranked the known failing loosen action above hold.

Therefore the action-conditioned video/numeric head remains NO-GO for candidate
selection. The only checkpoint is temporary at
`/tmp/grip_candidate_boundary_20260831_50step.pt`; no saved shadow checkpoint
was replaced and no head controlled a motor. Do not add optimizer steps to
rescue this result. The matched data are sufficient to reject the current head
test, though not to prove that all possible learned controllers are impossible.
The next practical route is a conservative post-lift boundary-search controller
that reverts to the last stable command, or an added direct force/tactile
measurement.

Two transient motor-bus failures also occurred: first motor ID 5 returned no
status packet, then a later firmware handshake returned no status packet.
After each, `so101_diag.py ids` showed IDs 1-6 at `0/10` misses and `health`
showed all six `Torque_Enable=0`; wrist-roll temperature was 52 then 56 C. The
final middle reset succeeded. If the error recurs, stop repeated retries and
inspect/reseat the wrist-roll daisy-chain connectors before further arm runs.

## Stability-constrained effort head smoke on 2026-08-31

The existing action-conditioned trainer was minimally extended with a second
output for mean absolute `Present_Load` during the `0.8 s` after each observed
candidate. Candidate choice is stability first, then minimum predicted load;
the two training boundary pairs also receive an explicit hold-over-failing-
loosen ranking loss. ACT and the pre-lift mask are unchanged.

One fixed 50-step run used 223 training windows / 26 trials and kept trials
`29,40,42,45` plus boundary 04 held out. Held-out stability was `80.5%`, Brier
`0.149`; load MAE was `26.1` versus `46.4` for the constant train-mean
baseline. At boundary 04's first-slip context, hold finally ranked above the
failing loosen (`0.464` versus `0.437`). Under the predeclared `0.65` stability
threshold, however, trial 45's four valid loosen events selected loosen only
twice, hold once, and an incorrect tighten once. This is partial offline
improvement but remains NO-GO for actuation. The temporary checkpoint is
`/tmp/grip_candidate_stability_effort_20260831_50step.pt`; no deployed or saved
shadow checkpoint was replaced.

## Post-lift-only selection and historical load gate on 2026-08-31

A deployment-matched ablation now trains only post-lift `hold/+0.2` outcomes;
`-0.2` is reserved as deterministic rollback. Three no-intervention post-lift
slips (trials 18, 22, 25) were retained as stability negatives. The 50-step
run used 117 training windows and 29 held-out windows. Held-out stability was
`79.3%`, Brier `0.162` versus constant Brier `0.214`; load MAE was `30.1`
versus `50.3` for the constant train-mean baseline.

Historical replay selected a strict pre-action `abs(Present_Load) > 60` veto
for loosening: it preserved 6 safe `+0.2` steps and blocked all 3 known first-
slip steps across boundaries 01/03/04. With that veto, trial 45 selected all
4 reviewed safe loosens, boundary 04 selected hold, and the slipping hold trial
42 selected no loosens. This is promising developmental evidence, not an
independent gate: the value 60 was selected after viewing all three boundary
trials, so fresh sealed physical validation is required. Trial 40 would have
attempted loosen in 3/8 reviewed hold windows, whose counterfactual outcomes
are unknown. No runtime actuation path was enabled. The temporary checkpoint
is `/tmp/grip_candidate_post_lift_slips_loadgate60_20260831_50step.pt`.

## Three-action grip candidate retrain on 2026-08-31

Two fresh no-Load-gate physical trials started from operator-confirmed stable
lifts. In both, the fixed head applied exactly one `+0.2 q` and the carton then
slipped; both are retained as `loosen_unstable`. The checkpoint was not updated
online during either trial.

A fixed 50-step action-conditioned retrain now scores `delta_q={-0.2,0,+0.2}`.
Head-only trial 01 is training evidence and head-only trial 02 is held out in
full. Training support is `-0.2: 8 unstable / 2 stable`, `0: 96 / 99`, and
`+0.2: 3 / 17`. Held-out stability is `81.4%`, Brier `0.153` versus constant
Brier `0.250`; held-out Load MAE is `25.1` versus `44.8` for the train-mean
baseline.

The held-out head-only slip selects hold, and known slip trial 42 selects no
loosens. Safe-loosen trial 45 retains 3/4 loosen events. However clean hold
trial 40 still selects loosen in 3/8 windows, boundary 04 retains none of its
three safe loosen steps, and the learned selector does not yet choose `-0.2`
on slip trial 42. Therefore this is an offline NO-GO for actuation. The
temporary checkpoint is
`/tmp/grip_candidate_three_action_headonly_20260831_50step.pt`.

## Conservative post-lift retrain with fresh slip negatives on 2026-08-31

The simpler `{hold,+0.2}` route was retrained for a fixed 50 steps with no
Load gate. Head-only trial 01 is a training `+0.2 -> slip` example and
head-only trial 02 remains fully held out. Training support is `hold: 32
unstable / 67 stable` and `+0.2: 3 / 17`. Held-out stability is `74.2%`,
Brier `0.177` versus constant Brier `0.219`; held-out Load MAE is `29.0`
versus `47.6` for the train-mean baseline.

At the developmental `0.65` threshold, the held-out fresh slip selects hold,
known slip trial 42 selects no loosens, boundary 04's failed loosen selects
hold, and all 4 reviewed safe trial-45 loosens remain. Clean hold trial 40,
however, still selects loosen in 4/8 windows. A post-hoc threshold sweep shows
that `0.78` removes those known false loosens but retains only 1/4 safe trial-45
loosens; those trials have already been viewed and cannot validate that
threshold. This remains NO-GO for actuation. The temporary checkpoint is
`/tmp/grip_candidate_post_lift_headonly_negatives_20260831_50step.pt`.

## Targeted post-lift A/B collection and hardware stop on 2026-08-31

This is the current handoff state and supersedes the earlier plan to continue
hardware trials immediately. The active ACT checkpoint and body contract were
not changed: fixed middle start
`[1.32,-38.42,42.68,86.20,0.92,99.34]`, prefix 14, action repeat 2, 300 ACT
steps, and `delta_q=0` before any manual post-lift intervention. The grip head
did not control the motor and no checkpoint learned online. Collection used
the corrected manual intervention recorder: after an operator-confirmed lift,
`c` latched the actually executed q, `]` applied exactly `+0.2`, and the second
`c` resumed the buffered ACT body trajectory.

The predeclared eight-slot schedule, generated with seed 42, was
`hold, loosen, loosen, loosen, hold, loosen, hold, hold`. A slot counts only if
ACT first establishes a lift and the assigned action is actually executed.
No-lift attempts, forgotten key inputs, and adaptive manual recovery are
retained but do not replace a scheduled A/B slot.

Four of eight randomized slots were completed:

| Slot | Evidence | Executed action | q command | Reviewed outcome | Use |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `grip_targeted_ab_trial01_hold_20260831` | Hold | 25.388 | Lifted, then gradually slipped | Valid A/B; `hold_unstable` |
| 2 | `grip_targeted_ab_trial02_loosen_retry03_20260831` | Loosen `+0.2` | 25.353 -> 25.553 | Gradual slip | Valid A/B; `loosen_unstable` |
| 3 | `grip_targeted_ab_trial03_loosen_20260831` | Loosen `+0.2` | 25.439 -> 25.639 | Stable | Valid A/B; `loosen_stable` |
| 4 | `grip_targeted_ab_trial04_loosen_retry01_20260831` | Loosen `+0.2` | 23.983 -> 24.183 | Stable | Valid A/B; `loosen_stable` |

Slot 1 was also replayed offline through the conservative head. Across 163
active contexts, the `0.65` selector would have chosen loosen 42 times and its
maximum loosen stability probability was `0.813`. The physical hold itself
slipped, so this trial is a hold-stability negative, not a stable-hold hard
negative and not evidence that loosening would have been safe. The contrast
between slots 2-4 also confirms that absolute q is not a reusable decision
threshold: `+0.2` slipped from one initial state but remained stable from two
others, including a numerically very similar final q.

Two additional trials are trainable but are not randomized A/B evidence:

| Evidence | Actual sequence | Reviewed outcome | Permitted use |
| --- | --- | --- | --- |
| `grip_targeted_ab_trial02_loosen_retry02_20260831` | Assigned loosen, but executed hold at q 24.010 | Stable, no slip | Stable-hold training only; exclude from A/B |
| `grip_targeted_ab_trial05_hold_20260831` | Initial grip too loose; tighten `-0.2` 33 times, then loosen `+0.2` | Tightening enabled lift; final loosen 25.434 -> 25.634 caused slight slip | Segment carefully: tighten recovery for a future three-action model and final post-lift loosen negative; exclude from A/B |

The following attempts must not be silently counted:

- `grip_targeted_ab_trial02_loosen_20260831`: no `c` or `]` was recorded and
  the physical outcome was unknown; raw evidence only.
- `grip_targeted_ab_trial02_loosen_retry01_20260831`: ACT did not lift.
- `grip_targeted_ab_trial04_loosen_20260831`: ACT did not lift.
- `grip_targeted_ab_trial05_hold_retry01_20260831`: ACT did not lift.
- `grip_targeted_ab_trial05_hold_retry02_20260831`: ACT did not lift after a
  successful reconnect; this run itself completed normally.
- `grip_targeted_ab_trial05_hold_retry03_20260831` never started and has no
  evidence directory because the motor handshake failed before arm motion.

All reviewed outcomes are stored beside their raw `control.jsonl` and videos
as `outcome.json`. The randomized collection therefore remains incomplete at
4/8. The unfilled schedule is slot 5 hold, slot 6 loosen, slot 7 hold, and slot
8 hold. Do not relabel the two non-random trials to make the count appear
complete. No retraining has been performed with these targeted A/B trials.
The current conservative temporary checkpoint remains
`/tmp/grip_candidate_post_lift_headonly_negatives_20260831_50step.pt`, and it
remains a deployment NO-GO. The load gate is not part of the current plan, and
the head must not automatically loosen or tighten a real grasp.

Hardware collection was stopped after a repeated motor ID 5 failure:

1. The first connection attempt for slot-5 retry 2 failed while disabling
   torque with `Failed to write 'Torque_Enable' on id_=5 ... There is no
   status packet`. No policy step ran.
2. A read-only check confirmed the stable serial path still resolved to
   `/dev/ttyACM0` and no other process held the port. This ruled out a missing
   USB device or competing deploy process but did not prove motor-bus health.
3. After the operator checked the wrist-roll wiring and explicitly approved a
   retry, one full run and the following middle reset completed. That run was
   the no-lift `grip_targeted_ab_trial05_hold_retry02_20260831` above.
4. The next connection failed during the firmware handshake. IDs
   `1,2,3,4,6` were found with expected model 777, while ID 5 was completely
   missing. The failure occurred before configuration or autonomous motion.

This recurrence supersedes the earlier note that the ID-5 failures were merely
transient. Current hardware status is **STOP**. The last successful commanded
pose was the standard middle reset, but current power and torque state must not
be assumed. Do not retry autonomous deployment or issue another reset until
the arm is powered off and the ID-5 wrist-roll motor, both connectors, and its
cable segment are inspected. If reseating does not resolve it, replace the
cable segment; persistent single-ID loss after that implicates the motor or
its electronics. Before any arm-enabled run, a motor scan/handshake must find
all IDs 1-6 reliably. Resume the remaining A/B schedule only if the operator
later chooses to continue; do not restart it automatically.
