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
