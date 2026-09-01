# Shared by every wrapper. Resolves the repository from this script's location, so
# the wrappers work from any cwd and from any checkout path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This repo's sources go FIRST on the path. An older build of
# lerobot_teleoperator_so101_webcam may be installed in the interpreter's
# site-packages; without this, `python -m` would silently run that one instead.
export PYTHONPATH="$REPO/packages/so101_teleop/src:$REPO/packages/webcam_input/src:$REPO/integrations/pressurevision/src${PYTHONPATH:+:$PYTHONPATH}"

# Teleop and recording are CPU-only (MediaPipe + placo IK). Keeping torch off the
# GPU avoids waking a flaky dGPU on import. Training and deployment override this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

PYTHON="${SO101_PYTHON:-python3}"

run_module() {
  local module="$1"; shift
  exec "$PYTHON" -m "lerobot_teleoperator_so101_webcam.programs.$module" "$@"
}
