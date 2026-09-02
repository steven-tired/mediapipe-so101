#!/usr/bin/env bash
# PressureVision pad rig: aim, calibrate, fit, serve. One short command each,
# because the full invocations are long enough that pasting them line-wraps.
#
#   ./scripts/run_pv_pad.sh aim                  # frame the pad, writes the crop
#   ./scripts/run_pv_pad.sh rematch 07           # put the camera back where session 07 was calibrated
#   ./scripts/run_pv_pad.sh capture 07           # labelled light/hard session -> pv_labelled_07
#   ./scripts/run_pv_pad.sh fit 07               # fit the level boundaries -> pv_levels_07.json
#   ./scripts/run_pv_pad.sh serve 07 [--preview] # stream to the teleop on udp:8090
#   ./scripts/run_pv_pad.sh probe 07             # serve + log packets and frames, for diagnosis
#
# NO ARM MOTION: this is the sensor side only.
#
# The crop is remembered in CROP_FILE between steps, so it cannot drift apart
# from the session it was aimed for. Everything runs MJPG: the C270 caps
# uncompressed 1280x720 at 7.5 fps against 30 for MJPG, and calibration and
# streaming have to share a format.
#
# These run the PressureVision network, so they need its environment
# (torch + segmentation-models-pytorch): point SO101_PV_PYTHON at it.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TOOLS="$REPO/integrations/pressurevision/tools"
PV_PYTHON="${SO101_PV_PYTHON:-$PYTHON}"
SESSIONS="${PV_SESSION_DIR:-$REPO/local/pv_sessions}"
CROP_FILE="${PV_CROP_FILE:-$SESSIONS/pv_crop.json}"
mkdir -p "$SESSIONS"

usage() { sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 1; }
[ $# -ge 1 ] || usage
CMD=$1; shift

read_crop() {
  [ -f "$CROP_FILE" ] || { echo "no crop yet -- run '$0 aim' first" >&2; exit 1; }
  "$PV_PYTHON" -c "import json;c=json.load(open('$CROP_FILE'))['crop'];print(','.join(map(str,c)))"
}

need_session() {
  [ $# -ge 1 ] || { echo "usage: $0 $CMD <session-number> [extra args]" >&2; exit 1; }
}

case "$CMD" in
  aim)
    "$PV_PYTHON" "$TOOLS/aim_pad_camera.py" --camera 2 --out "$CROP_FILE" "$@"
    ;;
  rematch)
    need_session "$@"; N=$1; shift
    "$PV_PYTHON" "$TOOLS/aim_pad_camera.py" --camera 2 --out "$CROP_FILE" \
      --match "$SESSIONS/pv_levels_$N.json" "$@"
    ;;
  capture)
    need_session "$@"; N=$1; shift
    "$PV_PYTHON" "$TOOLS/capture_labelled_press.py" \
      --session-dir "$SESSIONS/pv_labelled_$N" --crop "$(read_crop)" --mjpg \
      --intent-labels none,light,hard --surface "white paper on table, no scale" "$@"
    ;;
  fit)
    need_session "$@"; N=$1; shift
    "$PV_PYTHON" "$TOOLS/serve_pad_pressure.py" \
      --session-dir "$SESSIONS/pv_labelled_$N" --levels-out "$SESSIONS/pv_levels_$N.json" "$@"
    ;;
  serve)
    need_session "$@"; N=$1; shift
    "$PV_PYTHON" "$TOOLS/serve_pad_pressure.py" \
      --levels "$SESSIONS/pv_levels_$N.json" --crop "$(read_crop)" --mjpg "$@"
    ;;
  probe)
    need_session "$@"; N=$1; shift
    LOGDIR="$(mktemp -d "$SESSIONS/probe-XXXXXX")"
    echo "logging to $LOGDIR/pv.csv and $LOGDIR/frames/"
    mkdir -p "$LOGDIR/frames"
    "$PV_PYTHON" "$TOOLS/serve_pad_pressure.py" \
      --levels "$SESSIONS/pv_levels_$N.json" --crop "$(read_crop)" --mjpg \
      --log "$LOGDIR/pv.csv" --log-frames "$LOGDIR/frames" "$@"
    ;;
  *) usage ;;
esac
