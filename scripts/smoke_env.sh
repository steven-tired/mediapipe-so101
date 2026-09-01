# Hardware configuration for the physical smoke test on this machine.
# Source this, then run the wrappers. It sets only hardware/location config;
# the wrappers resolve the Python path themselves.
export SO101_PYTHON=/home/zhuokai/hand-teleop/.venv-lerobot/bin/python
export VR_DEX_RETARGETING_DIR=/home/zhuokai/hand-teleop/LeFranX/vr-dex-retargeting/example/vector_retargeting
export SO_ARM100_DIR=/home/zhuokai/hand-teleop/SO-ARM100
export SO101_ARM_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00
export SO101_WORKSPACE_CAM=/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SO101_LOCAL_DIR="$_REPO/local"
export SO101_DATASET_ROOT="$_REPO/local/datasets/smoke"
export SO101_EVIDENCE_DIR="$_REPO/local/evidence/smoke"
echo "smoke env ready: repo=$_REPO  python=$SO101_PYTHON"
