#!/usr/bin/env bash
# Fixed-pose carton grip trials, with both workspace cameras recording.
# THE ARM MOVES to a stored pickup pose. Keep the e-stop within reach.
#
# First run needs the pose captured by hand:
#   position the empty open gripper at the pickup pose, then
#   ./scripts/run_carton_fixed_grip_trials.sh --capture-current-pose
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: no policy is loaded

TRIAL_SCRIPT="$REPO/integrations/pressurevision/tools/carton_fixed_grip_trials.py"
ARM_PORT="${SO101_ARM_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00}"
CREATIVE_CAMERA="${SO101_WORKSPACE_CAM:-/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0}"
ETRON_CAMERA="${SO101_SIDE_CAM:-/dev/v4l/by-id/usb-Etron_Technology__Inc._USB2.0_Camera-video-index0}"
EVIDENCE_ROOT="${CARTON_EVIDENCE_ROOT:-$REPO/local/evidence/carton_fixed_grip_trials}"
POSE_FILE="${CARTON_PICK_POSE:-${EVIDENCE_ROOT}/fixed_pick_pose.json}"

if [[ "${1:-}" == "--help" ]]; then
    exec "$PYTHON" "${TRIAL_SCRIPT}" --pose-file "${POSE_FILE}" --help
fi

if [[ "${1:-}" == "--capture-current-pose" ]]; then
    mkdir -p "${EVIDENCE_ROOT}"
    exec "$PYTHON" "${TRIAL_SCRIPT}" \
        --port "${ARM_PORT}" --pose-file "${POSE_FILE}" "$@"
fi

if [[ ! -e "${ARM_PORT}" ]]; then
    echo "SO-101 arm is unavailable: ${ARM_PORT}" >&2
    exit 2
fi
if [[ ! -e "${CREATIVE_CAMERA}" ]]; then
    echo "required Creative side camera is unavailable: ${CREATIVE_CAMERA}" >&2
    exit 2
fi
if [[ ! -f "${POSE_FILE}" ]]; then
    echo "fixed pickup pose is missing: ${POSE_FILE}" >&2
    echo "Position the empty open gripper at the pickup pose, then run:" >&2
    echo "  $0 --capture-current-pose" >&2
    exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
SESSION="$(mktemp -d "${EVIDENCE_ROOT}/session-XXXXXX")"
CREATIVE_VIDEO="${SESSION}/creative_side.ts"
ETRON_VIDEO="${SESSION}/etron_overview.ts"

creative_pid=0
etron_pid=0
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    for pid in "${creative_pid}" "${etron_pid}"; do
        if [[ "${pid}" -gt 0 ]] && kill -0 "${pid}" 2>/dev/null; then
            if kill "${pid}" 2>/dev/null; then :; fi
        fi
    done
    for pid in "${creative_pid}" "${etron_pid}"; do
        if [[ "${pid}" -gt 0 ]]; then
            if wait "${pid}" 2>/dev/null; then :; fi
        fi
    done
    echo "evidence: ${SESSION}"
    exit "${rc}"
}
trap cleanup EXIT INT TERM

ffmpeg -nostdin -hide_banner -loglevel error \
    -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 \
    -i "${CREATIVE_CAMERA}" -an -c:v libx264 -preset ultrafast -tune zerolatency \
    -crf 28 -g 15 -x264-params repeat-headers=1 -f mpegts "${CREATIVE_VIDEO}" \
    >"${SESSION}/creative_ffmpeg.log" 2>&1 &
creative_pid=$!

if [[ -e "${ETRON_CAMERA}" ]] && \
   [[ "$(readlink -f "${CREATIVE_CAMERA}")" != "$(readlink -f "${ETRON_CAMERA}")" ]]; then
    ffmpeg -nostdin -hide_banner -loglevel error \
        -f v4l2 -framerate 30 -video_size 640x480 \
        -i "${ETRON_CAMERA}" -an -c:v libx264 -preset ultrafast -tune zerolatency \
        -crf 28 -g 15 -x264-params repeat-headers=1 -f mpegts "${ETRON_VIDEO}" \
        >"${SESSION}/etron_ffmpeg.log" 2>&1 &
    etron_pid=$!
else
    echo "optional Etron recording skipped: ${ETRON_CAMERA}" >&2
fi

for _ in $(seq 1 100); do
    if ! kill -0 "${creative_pid}" 2>/dev/null; then
        echo "Creative recorder exited during startup; inspect ${SESSION}/creative_ffmpeg.log" >&2
        exit 1
    fi
    if [[ -s "${CREATIVE_VIDEO}" ]]; then
        break
    fi
    sleep 0.1
done
if [[ ! -s "${CREATIVE_VIDEO}" ]]; then
    echo "Creative recording did not start; inspect ${SESSION}/creative_ffmpeg.log" >&2
    exit 1
fi

"$PYTHON" "${TRIAL_SCRIPT}" \
    --port "${ARM_PORT}" \
    --pose-file "${POSE_FILE}" \
    --evidence-dir "${SESSION}" \
    "$@"
