# Source Map

Provenance for every file that came from the previous `hand-teleop` meta-workspace.

**All source paths in this file are relative to the old workspace root.** They are
never written as `/home/<user>/hand-teleop/...` — an absolute path would both fail
the release gate in `RELEASE_AUDIT.md` and publish a developer's home directory.

Disposition values:

- `verbatim` — copied unchanged.
- `path-rewrite` — only absolute paths, sibling-checkout assumptions, or import
  namespaces changed; behaviour untouched.
- `rewritten` — reconstructed against a new interface; behaviour reviewed.
- `new` — written for this repository.

## Scaffold

| Destination | Source | Disposition | Notes |
| --- | --- | --- | --- |
| `README.md` | `README.md`, `CLAUDE.md` | rewritten | old workspace README described a meta-workspace of several repos and was stale (it named `dp_models/` and `eval/`, both since removed) |
| `LICENSE` | — | new | Apache-2.0 |
| `NOTICE` | — | new | attributions for LeRobot, LeFranX (Apache-2.0), PressureVision, vr-dex-retargeting (MIT) |
| `.gitignore` | — | new | |
| `pyproject.toml` | — | new | pytest configuration only; the root is not an installable package |
| `docs/CLAIMS_AND_GATES.md` | `docs/PV_HANDTRACK_PRIVILEGED_COMPARISON.md`, `docs/GENTLE_GRASP_OBJECTIVE_PLAN.md`, `docs/SO101_ORIENTATION_*` | rewritten | study status read from the source protocols at migration time: W0 frozen 2026-08-06, W1 complete, W3 pilot not run, v1.1 amendment not adopted |

## packages/webcam_input

Source: `webcam-input/` on branch `so101-webcam-diffusion` at 4e1f2fb.

| Destination (under `packages/webcam_input/`) | Source (under `webcam-input/`) | Disposition |
| --- | --- | --- |
| `src/webcam_input/__init__.py` | `webcam_input/__init__.py` | verbatim |
| `src/webcam_input/depth.py` | `webcam_input/depth.py` | verbatim |
| `src/webcam_input/detector.py` | `webcam_input/detector.py` | path-rewrite |
| `src/webcam_input/gestures.py` | `webcam_input/gestures.py` | verbatim |
| `src/webcam_input/oak_camera.py` | `webcam_input/oak_camera.py` | verbatim |
| `src/webcam_input/types.py` | `webcam_input/types.py` | verbatim |
| `src/webcam_input/webcam_source.py` | `webcam_input/webcam_source.py` | verbatim |
| `src/webcam_input/webcam_source_manager.py` | `webcam_input/webcam_source_manager.py` | verbatim |
| `src/webcam_input/wrist_estimator.py` | `webcam_input/wrist_estimator.py` | verbatim |
| `tests/test_depth.py` | `webcam_input/tests/test_depth.py` | verbatim |
| `tests/test_gestures.py` | `webcam_input/tests/test_gestures.py` | verbatim |
| `tests/test_process_hands.py` | `webcam_input/tests/test_process_hands.py` | verbatim |
| `tests/test_source_manager.py` | `webcam_input/tests/test_source_manager.py` | verbatim |
| `tests/test_wrist_estimator.py` | `webcam_input/tests/test_wrist_estimator.py` | verbatim |
| `tests/test_public_dependency_boundary.py` | — | new |
| `pyproject.toml` | — | new |

**`detector.py` path rewrite.** The original resolved `SingleHandDetector` from
`VR_DEX_RETARGETING_DIR`, falling back to a hardcoded sibling checkout at
`../LeFranX/vr-dex-retargeting/example/vector_retargeting`. The sibling fallback is
removed; resolution is now the env var, else an importable `single_hand_detector`
module, else an `ImportError` naming both options. `detector_dir()` now returns
`Path | None` instead of always returning a path — the only signature change.

**Not copied.** `realsense_camera.py` and `pinch_geometry.py` are absent from this
branch. `__pycache__/` was excluded.

**Tests:** 19 collected in the source, 21 here (19 migrated + 2 new boundary tests),
all passing under `.venv-webcam` with `VR_DEX_RETARGETING_DIR` set.

## packages/so101_teleop

Source: `webcam-input/lerobot_teleoperator_so101_webcam/` at 4e1f2fb. The nine
package modules come from the **inner** `lerobot_teleoperator_so101_webcam/`
directory; the ten programs come from the outer one.

| Destination | Source | Disposition |
| --- | --- | --- |
| `src/lerobot_teleoperator_so101_webcam/{__init__,config_so101_webcam,config_so101_webcam_ee,control,ee_control,ee_controller,servo_pid,so101_webcam,so101_webcam_ee}.py` | inner package, same names | verbatim |
| `src/lerobot_teleoperator_so101_webcam/programs/*.py` | outer dir: `deploy_so101_ee`, `diagnose_deploy`, `dry_run`, `dry_run_viz`, `record_so101_ee`, `servo_current`, `teleop_viz`, `teleop_viz_ee`, `teleop_viz_oak`, `tune_servo_pid` | path-rewrite |
| `src/lerobot_teleoperator_so101_webcam/paths.py` | — | new |
| `tests/{test_discovery,test_ee_control,test_ee_teleoperator,test_filters,test_retarget,test_teleoperator}.py` | outer `tests/` | verbatim |
| `tests/test_record_so101_ee_terminal_messages.py` | outer `tests/`, **untracked in the source** | path-rewrite |
| `tests/test_public_boundary.py` | — | new |

**Programs became a subpackage.** They were loose scripts run by path; here they
are `lerobot_teleoperator_so101_webcam.programs.*`, invoked with `python -m`. The
wrappers therefore do not depend on a checkout location.

**Path rewrites.** `paths.py` resolves the repository root from the installed
package and reads `SO101_URDF`/`SO_ARM100_DIR`, `SO101_LOCAL_DIR`,
`SO101_DATASET_ROOT`, and `SO101_EVIDENCE_DIR`. Four rewritten defaults were not
merely cosmetic — they pointed at paths that no longer exist in the source
workspace:

| File | Old default | Status in source |
| --- | --- | --- |
| `record_so101_ee.py` `DATASET_ROOT` | `datasets/hand_tracking_pick_place` | removed; datasets were reorganised to `hand_tracking_pv_carton_*` |
| `diagnose_deploy.py` `--csv` | `eval/csv/diag_log.csv` | `eval/` removed |
| `diagnose_deploy.py` `--frame-dir` | `eval/frames/diag_frames` | `eval/` removed |
| `record_so101_ee.py`, `teleop_viz_ee.py` `URDF_PATH` | absolute SO-ARM100 path | SO-ARM100 is an optional external dependency here |

**IR exclusion.** 21 IR programs and 26 IR tests were left behind for
`ir-camera-force`. They are named `analyze_ir_*`, `record_ir_*`, `compare_ir_*`,
`extract_ir_*`, `organize_ir_*`, `characterize_ir_*`, `report_ir_*`,
`verify_ir_*`, `view_ir_*`, and `prepare_gpt_pro_rep08_review.py` — **none starts
with `ir_`**, so a prefix rule would have leaked all of them.

**Tests:** 31 in the source, 34 here (31 migrated + 3 new boundary tests), all
passing under `.venv-lerobot`.

### Deferred: recorder reconstruction

`record_so101_ee.py` here is the base-checkout version. The
`ir-hand-pressure-so101-teleop` worktree carries a restructured recorder (663 lines
vs 481) with `build_dataset_features`, `_close_and_dispose_recording_session`, and
an `ExitStack`-based `_run_recording`. That structure is wanted, but its IR sidecar
plumbing threads through the teleoperator constructor, so it is reconstructed
against the gripper contract rather than copied. See the grip-contract commit.

`ee_controller.py` is the base version (211 lines, zero IR/PV references). The
worktree version is 1052 lines with 257 IR/PV references and is deliberately not
used.

## Source repository state at migration

Recorded so a later reader can reproduce exactly what was copied.

| Source repo / worktree | Branch | HEAD |
| --- | --- | --- |
| `webcam-input` | so101-webcam-diffusion | 4e1f2fb |
| `webcam-input/.worktrees/ir-hand-pressure-so101-teleop` | ir-hand-pressure-so101-teleop | af97c9f |
| `hand-pressure` | (see backup manifest) | — |
| `LeFranX` | — | — |

Full pre-migration state, including every dirty and untracked path, is in the
backup manifests: `manifests/status-before.txt`, `manifests/worktrees-before.txt`,
and `manifests/rsync-copy.log`.

## Uncommitted sources

Files copied from a working tree rather than a commit. These exist in no Git
history and were preserved only by the migration backup's `working-state/` tree.

| Destination | Source | Notes |
| --- | --- | --- |
| _(filled in as packages are migrated)_ | | |
