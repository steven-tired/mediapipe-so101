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
- **Gripper step resolution, 2026-09-02.** On the carton, loosening resolves a `0.5`
  command and delivers 90-95% of it; tightening resolves only `2.0` and delivers
  about 38%. Measured with equal-travel staircases in both directions, scoring a step
  size as resolved only when every tread moved past the noise floor. The tighten limit
  is compliance, not deadband.
- **The lift-to-slip window, 2026-09-02.** Across paired trials the readback distance
  from the lift boundary to the first slip onset had a median of about `1.3` — narrower
  than the smallest tighten step, which is why the boundary is measured by loosening.
- **A visible carton dent at a readback near `21.3`.** One operator observation, not an
  instrument reading, and it moved every boundary in that trial. It bounds the objective
  from the tight side and makes boundaries comparable only within one floor setting.
- **First contact is detectable at the servo, 2026-09-03.** Swept from free space, both
  `Present_Current` and `Present_Load` separate the carton from an empty jaw by about ten
  sigma at depth; two independent empty-jaw sweeps agree to within their own spread. First
  contact on this carton is a readback of about `50`, which puts the lift boundary at `24`
  units of compression and the dent at `29`. A fast close still detects contact but places
  it six units too open, so `x0` must be taken by a slow probe before the grasp.

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
- **Gentle-grasp objective.** The objective is stated and its measurement channel is
  calibrated, but no deformation-aware head has been trained or evaluated. The
  paired-boundary collection that would supply its labels is at n=11 trials, and only
  n=3 at the current floor setting; the carton's position is the only variable that has
  been changed, so a head trained on this data would have nothing to condition on.
- **A repeatable deformation criterion.** First contact is now measured (see the
  hardware list above), so compression `x0 - x` is computable -- but its upper bound
  still rests on one operator's eye on one trial. Until deformation is scored the same
  way twice, a compression bound has no ceiling to enforce.
- **Portability of compression across carton states.** Whether `x0 - x` at the lift
  boundary survives the carton being flattened has not been tested. If it does, the
  eleven paired trials pool into one dataset; if it does not, they stay three groups.
- **Carton deformation has never been scored.** It is visible in the dual-view video and
  the one damage measurement on record is an operator's eye. Any claim of "less pressure"
  must name whether it rests on jaw depth, on motor effort, or on nothing.

## Deliberately not claimed

- No orientation-complete end-effector control. The SO-101's wrist yaw is coupled to
  the arm's azimuth, so the reachable orientation set is rank-deficient outside the
  downward-pointing pose. Tasks are designed around this, not in spite of it.
- No result from the private IR/thermal line appears here. That work lives in a
  separate repository and this one does not depend on it.
