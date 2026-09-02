#!/usr/bin/env bash
# Autonomous rollout with opt-in PV gripper takeovers. Press `c` to start/end a
# correction window; each window is saved as one episode in the correction dataset.
# THE ARM MOVES ON ITS OWN. Keep the e-stop within reach.
#
# The PressureVision sender needs its own environment: point SO101_PV_PYTHON at it.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The tools live outside src/ because they are programs, not importable modules.
export PYTHONPATH="$REPO/integrations/pressurevision/tools:$PYTHONPATH"

DEPLOY="$REPO/integrations/pressurevision/tools/deploy_so101_grip_ee.py"
SENDER="$REPO/integrations/pressurevision/tools/serve_pad_pressure.py"
PV_PYTHON="${SO101_PV_PYTHON:-$PYTHON}"

LEVELS="${PV_LEVELS:-}"
LIGHT_POS="${GRIP_LIGHT_POS:-}"
HARD_POS="${GRIP_HARD_POS:-}"
CONTEXT="${PV_GRIP_CONTEXT:-unknown}"
CONTROL="${PV_GRIP_CONTROL:-direct}"
MAX_LEVEL_AGE_MINUTES="${PV_MAX_LEVEL_AGE_MINUTES:-180}"
PV_CAMERA="${PV_CAMERA:-2}"
PV_CROP="${PV_CROP:-40,0,980,720}"
CORRECTION_ROOT="${PV_CORRECTION_ROOT:-$REPO/local/datasets/hand_tracking_pv_pick_place}"
CORRECTION_REPO="${PV_CORRECTION_REPO:-local/hand_tracking_pv_pick_place}"
EVIDENCE_ROOT="${PV_CORRECTION_EVIDENCE_ROOT:-$REPO/local/evidence/hand_tracking_pv_corrections}"

if [[ -z "${LEVELS}" ]]; then
    echo "PV_LEVELS must point to a fresh fitted levels.json" >&2
    exit 2
fi
if [[ -z "${LIGHT_POS}" || -z "${HARD_POS}" ]]; then
    echo "GRIP_LIGHT_POS and GRIP_HARD_POS must come from gripper calibration" >&2
    exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
SESSION="$(mktemp -d "${EVIDENCE_ROOT}/session-XXXXXX")"
SENDER_LOG="${SESSION}/pv_sender.csv"
SENDER_VIDEO="${SESSION}/pv_sender.avi"

sender_pid=0
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    if [[ "${sender_pid}" -gt 0 ]] && kill -0 "${sender_pid}" 2>/dev/null; then
        kill "${sender_pid}" 2>/dev/null || true
    fi
    if [[ "${sender_pid}" -gt 0 ]]; then
        wait "${sender_pid}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap cleanup EXIT INT TERM

"${PV_PYTHON}" "${SENDER}" \
    --levels "${LEVELS}" \
    --camera "${PV_CAMERA}" \
    --crop "${PV_CROP}" \
    --mjpg \
    --require-scene-match \
    --max-level-age-minutes "${MAX_LEVEL_AGE_MINUTES}" \
    --log "${SENDER_LOG}" \
    --video-out "${SENDER_VIDEO}" &
sender_pid=$!

# No hide_gpu: this loads a policy, and the sender runs the PV network.
"$PYTHON" "${DEPLOY}" \
    --scheduler ddim \
    --inference-steps 10 \
    --grip-context "${CONTEXT}" \
    --grip-control "${CONTROL}" \
    --grip-light-pos "${LIGHT_POS}" \
    --grip-hard-pos "${HARD_POS}" \
    --correction-dataset-root "${CORRECTION_ROOT}" \
    --correction-repo-id "${CORRECTION_REPO}" \
    "$@"
