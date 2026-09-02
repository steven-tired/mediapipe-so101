# Datasets

No dataset is tracked in Git. Recorded episodes live under `local/datasets/`,
which is ignored. This file is the contract: what a dataset must contain, and
which ones this project actually used.

## Layout

Datasets are LeRobot format, written by
`lerobot_teleoperator_so101_webcam.programs.record_so101_ee` (or, in
PressureVision mode, the composition entry point). The recorder resolves its
output root from `paths.dataset_root()`:

```
SO101_DATASET_ROOT   explicit dataset root
SO101_LOCAL_DIR      root of the ignored local tree (default <repo>/local)
```

Default: `<repo>/local/datasets/`.

## Recording contract

- 10 Hz. Not 30: `record_loop` reads the motor bus every frame, and under
  gripping load this Feetech bus drops reads and the loop degrades to ~9 Hz. 10 Hz
  runs steady and is adequate for the policies trained here.
- The workspace camera records `observation.images.front`. In dual-view sessions a
  side camera adds a second stream.
- An interrupted session is backed up to `<dataset>.bak` before a resume writes to
  it, so a failed append cannot destroy the previous episodes.

## Datasets migrated into `local/datasets/`

All are `hand_tracking_pv_carton_*`: the PressureVision carton line. See
`docs/LOCAL_ARTIFACT_MIGRATION_MANIFEST.tsv` for the exact file and byte counts of
each.

`hand_tracking_pv_carton_dual_view` is the primary recording target. Its many
`.session-<uuid>.bak` and `.aborted-session-*` siblings are **not** training data:
they are per-session safety copies and abandoned sessions, retained for provenance
only. Do not merge them into a training set.

`hand_tracking_pv_carton_phase_b`, `..._phase_b_train24`, and
`..._phase_b_train24_onset_qreplace` are derived training splits, built by
`integrations/pressurevision/tools/build_phase_b_training_dataset.py` from the
dual-view recordings plus an episode map.

## On the Hub

Nothing here is in Git, so the Hub is where these actually live. All are private
except where noted.

| Dataset | Episodes | What it is |
| --- | --- | --- |
| [`hand_tracking_pv_carton_dual_view`](https://huggingface.co/datasets/stevenzenith/hand_tracking_pv_carton_dual_view) | 54 | **the raw recordings** every carton dataset derives from |
| [`hand_tracking_pv_carton_middle_standard`](https://huggingface.co/datasets/stevenzenith/hand_tracking_pv_carton_middle_standard) | 13 | middle-pose episodes, used for the middle-labels fine-tune |
| [`hand_tracking_pv_carton_phase_b`](https://huggingface.co/datasets/stevenzenith/hand_tracking_pv_carton_phase_b) | 30 | reviewed training split, derived from dual_view |
| [`hand_tracking_pick_place`](https://huggingface.co/datasets/stevenzenith/hand_tracking_pick_place) | 100 | the earlier pick-place line |

The `_train24` and `_train24_onset_qreplace` splits are **not** uploaded on
purpose: `build_phase_b_training_dataset.py` rebuilds them from dual_view plus
an episode map, so publishing them would duplicate derivable data.

Policies trained on these:

| Model | Trained on |
| --- | --- |
| [`act_pickplace`](https://huggingface.co/stevenzenith/act_pickplace), [`dp_pickplace`](https://huggingface.co/stevenzenith/dp_pickplace) (public) | pick_place |
| [`act_carton_phase_c_80k`](https://huggingface.co/stevenzenith/act_carton_phase_c_80k), [`diffusion_carton_phase_c_90k`](https://huggingface.co/stevenzenith/diffusion_carton_phase_c_90k), [`smolvla_carton_phase_c_80k`](https://huggingface.co/stevenzenith/smolvla_carton_phase_c_80k) | the carton line |
| [`act_carton_phase_c_50k`](https://huggingface.co/stevenzenith/act_carton_phase_c_50k) | the base the fine-tune below started from |
| [`act_carton_middle_labels_5k`](https://huggingface.co/stevenzenith/act_carton_middle_labels_5k) | fine-tuned on middle_standard + phase_b; its recipe is `training/train_middle_labeled_act.py`, not a `train_config.json` |

**Correction, 2026-09-02:** an earlier version of this file said
`hand_tracking_pick_place` "no longer exists". It is not on this machine any
more, but it is on the Hub, and the link above is where it went.

## Not here

IR and Lepton datasets belong to the private project and are not in this
repository. They are also **not on the Hub**, and the Lepton hardware has been
dead since 2026-07-17, so those captures cannot be retaken — they exist in one
place only.
