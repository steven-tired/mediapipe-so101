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

## Not here

The earlier `hand_tracking_pick_place` dataset referenced by older code no longer
exists; it was reorganised away before this migration. Code that named it has been
changed to resolve `dataset_root()` instead. IR and Lepton datasets belong to the
private project and are not in this repository.
