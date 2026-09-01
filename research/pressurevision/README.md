# PressureVision research scripts

**Status: exploratory. Not runtime code, not a validated result.**

These are the training and evaluation scripts from the PressureVision grip-intent
line. They are kept for reproducibility of what was tried, not because any of
them produced a conclusion this project stands behind.

What the runtime actually uses is in `integrations/pressurevision/`:
`pv_pressure.py` (the UDP packet contract), `pv_relative_mapping.py` and
`pv_object_profile.py` (the position span), and `adapter.py` (the gripper
controller). Nothing here is imported by the runtime.

The controlled comparison these scripts belong to is **mid-flight**: the W0
protocol was frozen 2026-08-06 and W1 (fixed-position trials) is complete, but
the W3 pilot has not run and the v1.1 protocol amendment has not been adopted.
No claim about PressureVision's benefit over MediaPipe-only grip control follows
from anything in this directory. See `docs/CLAIMS_AND_GATES.md`.

Absolute paths in these files were replaced with `<workspace>` during migration;
they will not run unmodified.
