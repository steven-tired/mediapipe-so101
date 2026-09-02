#!/usr/bin/env bash
# Autonomous deployment with grip supervision: the residual head, the grip
# context, and the keyboard intervention path the correction datasets are
# recorded through.
#
# The plain policy path is scripts/run_deploy_ee.sh, which stays free of any
# PressureVision dependency. Only the --correction mode here opens a PV sender;
# start it separately (scripts/run_record_pv_ee.sh shows the invocation).
#
# The arm moves on its own. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# The tools live outside src/ because they are programs, not importable modules.
export PYTHONPATH="$REPO/integrations/pressurevision/tools:$PYTHONPATH"

# No hide_gpu: this runs a policy, and hiding the GPU made deployment fail with
# "No CUDA GPUs are available".
exec "$PYTHON" "$REPO/integrations/pressurevision/tools/deploy_so101_grip_ee.py" "$@"
