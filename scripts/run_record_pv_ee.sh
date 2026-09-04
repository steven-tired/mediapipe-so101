#!/usr/bin/env bash
# Record a PV-supervised episode: the PressureVision sender and the recorder run
# as two processes, and every invocation gets a fresh evidence session. Neither
# the dataset nor an earlier session is ever removed or reused implicitly.
#
# The sender needs its own environment (torch + segmentation-models-pytorch);
# point SO101_PV_PYTHON at that interpreter.
#
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The tools live outside src/ because they are programs, not importable modules.
export PYTHONPATH="$REPO/integrations/pressurevision/tools:$PYTHONPATH"

RECORDER="$REPO/integrations/pressurevision/tools/record_so101_pv_ee.py"
SENDER="$REPO/integrations/pressurevision/tools/serve_pad_pressure.py"
PV_PYTHON="${SO101_PV_PYTHON:-$PYTHON}"

DATASET_ROOT="${PV_DATASET_ROOT:-}"
EVIDENCE_ROOT="${PV_EVIDENCE_ROOT:-$REPO/local/evidence/hand_tracking_pv_carton_dual_view}"
LEVELS="${PV_LEVELS:-}"
MAPPING="${PV_MAPPING:-carton_span}"
GRIP_CONTEXT="${PV_GRIP_CONTEXT:-auto}"
MAX_LEVEL_AGE_MINUTES="${PV_MAX_LEVEL_AGE_MINUTES:-180}"
# The scene gate stays on by default: the framing a calibration was fit in is what
# its pressure bands mean, and the same press reads a different band once the crop
# moves. PV_REQUIRE_SCENE_MATCH=0 downgrades it to the sender's own warning, for
# footage where the reading is illustrative and no number is being claimed.
REQUIRE_SCENE_MATCH="${PV_REQUIRE_SCENE_MATCH:-1}"
SCENE_MATCH_FLAG=(--require-scene-match)
if [[ "${REQUIRE_SCENE_MATCH}" == "0" ]]; then
    SCENE_MATCH_FLAG=()
    echo "PV_REQUIRE_SCENE_MATCH=0: the sender will warn on scene drift instead of refusing;"\
         " readings from this session describe its own framing only" >&2
fi
PROFILE="${PV_OBJECT_PROFILE:-}"
PV_CAMERA="${PV_CAMERA:-2}"
PV_CROP="${PV_CROP:-40,0,980,720}"
DEFAULT_FRONT_CAMERA="/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0"
FRONT_CAMERA="${PV_FRONT_CAMERA:-${DEFAULT_FRONT_CAMERA}}"
DEFAULT_SIDE_CAMERA="/dev/v4l/by-id/usb-Etron_Technology__Inc._USB2.0_Camera-video-index0"
SIDE_CAMERA="${PV_SIDE_CAMERA:-${DEFAULT_SIDE_CAMERA}}"
EPISODES="${PV_EPISODES:-1}"
EPISODE_SECONDS="${PV_EPISODE_SECONDS:-120}"
MAX_LOAD="${PV_MAX_LOAD:-}"
MAX_CURRENT="${PV_MAX_CURRENT:-}"
MAX_POSITION_LAG="${PV_MAX_POSITION_LAG:-}"

if [[ -z "${LEVELS}" ]]; then
    echo "PV_LEVELS must point to a fresh fitted levels.json" >&2
    exit 2
fi
if [[ -n "${MAX_LOAD}" || -n "${MAX_CURRENT}" || -n "${MAX_POSITION_LAG}" ]]; then
    if [[ -z "${MAX_LOAD}" || -z "${MAX_CURRENT}" || -z "${MAX_POSITION_LAG}" ]]; then
        echo "PV_MAX_LOAD, PV_MAX_CURRENT, and PV_MAX_POSITION_LAG must be provided together" >&2
        exit 2
    fi
fi
if [[ -z "${FRONT_CAMERA}" ]]; then
    echo "PV_FRONT_CAMERA must name a separate workspace camera; PV_CAMERA=${PV_CAMERA} is reserved for PressureVision" >&2
    exit 2
fi
pv_camera_path="$(readlink -f "/dev/video${PV_CAMERA}")"
front_camera_path="$(readlink -f "${FRONT_CAMERA}")"
side_camera_path="$(readlink -f "${SIDE_CAMERA}")"
if [[ -z "${front_camera_path}" || ! -e "${front_camera_path}" ]]; then
    echo "PV_FRONT_CAMERA does not resolve to a camera: ${FRONT_CAMERA}" >&2
    exit 2
fi
if [[ "${front_camera_path}" == "${pv_camera_path}" ]]; then
    echo "PV_FRONT_CAMERA and PV_CAMERA resolve to the same device: ${front_camera_path}" >&2
    exit 2
fi
if [[ -z "${side_camera_path}" || ! -e "${side_camera_path}" ]]; then
    echo "PV_SIDE_CAMERA does not resolve to a camera: ${SIDE_CAMERA}" >&2
    exit 2
fi
if [[ "${side_camera_path}" == "${pv_camera_path}" || "${side_camera_path}" == "${front_camera_path}" ]]; then
    echo "PV_SIDE_CAMERA must resolve to a distinct Etron view: ${side_camera_path}" >&2
    exit 2
fi
if [[ "${MAPPING}" == "hard_profile" && -z "${PROFILE}" ]]; then
    echo "PV_OBJECT_PROFILE is required for hard_profile" >&2
    exit 2
fi
if [[ "${MAPPING}" != "hard_profile" && -n "${PROFILE}" ]]; then
    echo "PV_OBJECT_PROFILE is not valid for ${MAPPING}" >&2
    exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
SESSION="$(mktemp -d "${EVIDENCE_ROOT}/session-XXXXXX")"
RECORDER_EVIDENCE="${SESSION}/recorder"
PREVIEW_SHARE="${SESSION}/pv_preview.mmap"
SENDER_LOG="${SESSION}/pv_sender.csv"
SENDER_VIDEO="${SESSION}/pv_sender.avi"

REC_ARGS=(
    --levels "${LEVELS}"
    --pv-mapping "${MAPPING}"
    --grip-context "${GRIP_CONTEXT}"
    --max-level-age-minutes "${MAX_LEVEL_AGE_MINUTES}"
    --episodes "${EPISODES}"
    --episode-seconds "${EPISODE_SECONDS}"
    --front-camera "${FRONT_CAMERA}"
    --side-camera "${SIDE_CAMERA}"
    --evidence-dir "${RECORDER_EVIDENCE}"
    --pv-preview-share "${PREVIEW_SHARE}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
    REC_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${MAX_LOAD}" ]]; then
    REC_ARGS+=(
        --max-load "${MAX_LOAD}"
        --max-current "${MAX_CURRENT}"
        --max-position-lag "${MAX_POSITION_LAG}"
    )
fi
if [[ -n "${PROFILE}" ]]; then
    REC_ARGS+=(--object-profile "${PROFILE}")
fi
# Extra recorder flags (for example --discard-session or --no-oak) are appended last.
REC_ARGS+=("$@")

# The recorder is CPU-only (MediaPipe + placo IK), but the SENDER runs the
# PressureVision network and needs the GPU. So the hiding is per-invocation --
# exporting it for the whole script is what put a previous smoke run on CPU.
run_recorder() {
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "${RECORDER}" "$@"
}

# Configuration is checked before either process can open a camera, serial bus, UDP socket or mmap.
run_recorder "${REC_ARGS[@]}" --check-config >"${SESSION}/config-check.txt"

sender_pid=0
recorder_pid=0
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    if [[ "${sender_pid}" -gt 0 ]] && kill -0 "${sender_pid}" 2>/dev/null; then
        if kill "${sender_pid}" 2>/dev/null; then :; fi
    fi
    if [[ "${recorder_pid}" -gt 0 ]] && kill -0 "${recorder_pid}" 2>/dev/null; then
        if kill "${recorder_pid}" 2>/dev/null; then :; fi
    fi
    if [[ "${sender_pid}" -gt 0 ]]; then
        set +e
        wait "${sender_pid}" 2>/dev/null
        wait_rc=$?
        set -e
        if [[ "${rc}" -eq 0 && "${wait_rc}" -ne 0 ]]; then rc="${wait_rc}"; fi
    fi
    if [[ "${recorder_pid}" -gt 0 ]]; then
        set +e
        wait "${recorder_pid}" 2>/dev/null
        wait_rc=$?
        set -e
        if [[ "${rc}" -eq 0 && "${wait_rc}" -ne 0 ]]; then rc="${wait_rc}"; fi
    fi
    exit "${rc}"
}
trap cleanup EXIT INT TERM

# No --preview here: the sender still publishes the shared-memory PV panel and records its AVI.
"${PV_PYTHON}" "${SENDER}" \
    --levels "${LEVELS}" \
    --camera "${PV_CAMERA}" \
    --crop "${PV_CROP}" \
    --mjpg \
    "${SCENE_MATCH_FLAG[@]}" \
    --max-level-age-minutes "${MAX_LEVEL_AGE_MINUTES}" \
    --log "${SENDER_LOG}" \
    --video-out "${SENDER_VIDEO}" \
    --preview-share "${PREVIEW_SHARE}" \
    2> >(sed -u '/^Corrupt JPEG data:/d' >&2) &
sender_pid=$!

run_recorder "${REC_ARGS[@]}" &
recorder_pid=$!

# Whichever side exits first owns the session outcome; the trap closes the other side.
set +e
wait -n "${sender_pid}" "${recorder_pid}"
rc=$?
set -e
exit "${rc}"
