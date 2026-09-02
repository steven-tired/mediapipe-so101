# Hardware configuration for the physical smoke test on this machine.
# Source this, then run the wrappers. It sets only hardware/location config;
# the wrappers resolve the Python path themselves.
export SO101_PYTHON=/home/zhuokai/hand-teleop/.venv-lerobot/bin/python
export VR_DEX_RETARGETING_DIR=/home/zhuokai/hand-teleop/LeFranX/vr-dex-retargeting/example/vector_retargeting
export SO_ARM100_DIR=/home/zhuokai/hand-teleop/SO-ARM100
# The PressureVision sender runs the network and needs its own interpreter.
# Without this the PV wrappers fall back to python3 -- the defect-#1 mistake.
# The released PressureVision checkout (config/ + data/model/paper_59.pt). It is
# an external upstream clone, not vendored, so this repo names it here rather
# than hardcoding a /home path in the tools.
export SO101_PV_REPO=/home/zhuokai/hand-teleop/pressurevision
export SO101_PV_PYTHON=/home/zhuokai/hand-teleop/.venv-pressurevision/bin/python
export SO101_ARM_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00
export SO101_WORKSPACE_CAM=/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SO101_LOCAL_DIR="$_REPO/local"
export SO101_DATASET_ROOT="$_REPO/local/datasets/smoke"
export SO101_EVIDENCE_DIR="$_REPO/local/evidence/smoke"
# Migrated calibration for the overhead C270 pad rig (fit 2026-08-26, session 26).
# mtime is deliberately preserved, so the sender's freshness gate fires honestly:
# a run older than PV_MAX_LEVEL_AGE_MINUTES has to raise it on purpose, not by
# having been copied. The scene fingerprint is the guard that does the real work.
export PV_LEVELS="$_REPO/local/pv_sessions/pv_levels_26.json"
export PV_SESSION_DIR="$_REPO/local/pv_sessions"
echo "smoke env ready: repo=$_REPO  python=$SO101_PYTHON"
