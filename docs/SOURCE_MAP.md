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
