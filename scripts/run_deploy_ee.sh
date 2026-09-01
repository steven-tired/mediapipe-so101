#!/usr/bin/env bash
# Run a trained policy autonomously (DDIM @ 10 steps, ~9 Hz).
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_module deploy_so101_ee "$@"
