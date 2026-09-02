#!/usr/bin/env bash
# Collect one paired lift/slip boundary trial (gate 3 of the 2026-09-01 plan).
#
# ACT runs normally. If it stalls, the tighten ramp breaks the stall and that
# depth is the lift boundary. Once the carton is stably lifted -- press 'l' --
# the loosen ramp opens the jaw until it drops; press 'd' the moment it lets go
# and that depth is the slip boundary. Both branches continue into the loosen
# ramp, which is what makes the two boundaries comparable within one grasp.
#
# Steps come from the 2026-09-02 deadband calibration: 2.0 to tighten, 0.5 to
# loosen. Do not lower them without re-running that calibration -- below the
# measured deadband a ramp advances in stick-slip bursts and the depth it
# reports is not the depth the jaw is at.
#
# Usage: ./scripts/run_paired_boundaries.sh [extra deploy args]
#        TRIAL=07 ./scripts/run_paired_boundaries.sh    # pin the number
# With TRIAL unset the next free number for today is used, so a run that failed
# before it wrote anything simply reuses its number instead of leaving a hole.
# The arm moves on its own. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export PYTHONPATH="$REPO/integrations/pressurevision/tools:$PYTHONPATH"

: "${SO101_GRIP_POLICY:?set SO101_GRIP_POLICY to the pretrained_model directory of the checkpoint}"

EVIDENCE_ROOT="${SO101_EVIDENCE_DIR:-$REPO/local/evidence}"
GROUP="$EVIDENCE_ROOT/phase_c_recovery_minimal"
STAMP="$(date +%Y%m%d)"

if [ -z "${TRIAL:-}" ]; then
  # First number with no directory. Scanning beats "highest + 1" because a
  # connection that dropped before the run wrote anything leaves no directory,
  # and its number should be reused rather than skipped.
  n=1
  while [ -e "$GROUP/paired_boundary_trial$(printf '%02d' "$n")_$STAMP" ]; do
    n=$((n + 1))
    if [ "$n" -gt 99 ]; then
      echo "run_paired_boundaries: 99 trials already recorded for $STAMP" >&2
      exit 1
    fi
  done
  TRIAL="$(printf '%02d' "$n")"
fi

OUT="$GROUP/paired_boundary_trial${TRIAL}_$STAMP"
echo "[paired] trial $TRIAL -> $OUT"

# No hide_gpu: this runs a policy.
# The fixed-middle physical contract (TRAINING_HANDOFF, "Fixed-middle labeled
# ACT recovery"). attempt03 on 2026-09-02 ran without it -- executed prefix 1
# instead of 14, full speed instead of half -- and the policy never came near a
# grasp. These are not tuning knobs.
exec "$PYTHON" "$REPO/integrations/pressurevision/tools/deploy_so101_grip_ee.py" \
  --arm-enabled \
  --evidence-dir "$OUT" \
  --policy "$SO101_GRIP_POLICY" \
  --gripper-telemetry-hz 5 \
  --start-joints 1.32,-38.42,42.68,86.2,0.92,99.34 \
  --act-action-steps 14 --action-step-repeat 2 --max-steps 300 \
  --paired-boundaries \
  --stall-tighten-step 2 --stall-tighten-interval-s 1.0 --stall-tighten-floor 22 \
  --loosen-step 0.5 --loosen-interval-s 1.0 --loosen-ceiling 60 \
  "$@"
