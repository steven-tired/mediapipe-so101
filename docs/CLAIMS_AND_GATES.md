# Claims and Gates

This repository distinguishes three kinds of evidence and never lets one stand in
for another:

- **software** — tests, imports, and static checks. Says nothing about hardware.
- **locked inference** — offline evaluation on recorded data with a frozen policy.
  Predicts nothing about closed-loop success on its own.
- **physical robot** — observed behaviour on a real SO-101 in closed loop.

## Verified on hardware

- Webcam → SO-101 end-effector teleoperation runs live: the arm tracks the operator's
  wrist and the gripper opens and closes on pinch.
- LeRobot-format episode recording from the teleoperation path.
- Autonomous deployment of trained policies. Diffusion-policy wrappers run DDIM at
  10 steps (~9 Hz) rather than 100-step DDPM (~3.6 Hz); the earlier rate was too slow
  for the task.

## Verified in software only

- Everything covered by `pytest` in `packages/` and `integrations/`.
- Import boundaries: no public runtime file imports IR, FLIR, Lepton, or thermal code.

## Open gates

- **Post-migration hardware re-verification.** This repository was reconstructed from
  a previous workspace, and the run wrappers, virtualenv resolution, and dependency
  paths were all rewritten. The physical smoke test recorded in `docs/RELEASE_AUDIT.md`
  is what re-establishes the hardware claims above *for this tree*. Until that audit
  shows a pass, treat the hardware section as inherited, not re-confirmed.
- **PressureVision comparison study.** The controlled comparison between MediaPipe-only
  and PressureVision-assisted grip control is mid-flight: the W0 protocol was frozen
  2026-08-06 and W1 (fixed-position trials) is complete, but the W3 pilot has not run
  and the v1.1 protocol amendment has not been adopted. No conclusion about PV's
  benefit is claimed here.
- **Gentle-grasp objective.** Planned, not executed. No deformation-aware objective
  has been trained or evaluated.

## Deliberately not claimed

- No orientation-complete end-effector control. The SO-101's wrist yaw is coupled to
  the arm's azimuth, so the reachable orientation set is rank-deficient outside the
  downward-pointing pose. Tasks are designed around this, not in spite of it.
- No result from the private IR/thermal line appears here. That work lives in a
  separate repository and this one does not depend on it.
