# Shared by every wrapper. Resolves the repository from this script's location, so
# the wrappers work from any cwd and from any checkout path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This repo's sources go FIRST on the path. An older build of
# lerobot_teleoperator_so101_webcam may be installed in the interpreter's
# site-packages; without this, `python -m` would silently run that one instead.
export PYTHONPATH="$REPO/packages/so101_teleop/src:$REPO/packages/webcam_input/src:$REPO/integrations/pressurevision/src${PYTHONPATH:+:$PYTHONPATH}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Teleop and recording are CPU-only (MediaPipe + placo IK), and hiding the GPU
# stops torch waking a flaky dGPU on import. Deployment and diagnostics NEED the
# GPU, so a wrapper opts in by calling this BEFORE run_module. Note the empty
# assignment hides every GPU -- exporting it unconditionally is what broke
# deployment with "No CUDA GPUs are available".
hide_gpu() {
  export CUDA_VISIBLE_DEVICES=""
}

PYTHON="${SO101_PYTHON:-python3}"

run_module() {
  local module="$1"; shift
  exec "$PYTHON" -m "lerobot_teleoperator_so101_webcam.programs.$module" "$@"
}
