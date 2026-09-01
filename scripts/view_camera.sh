#!/usr/bin/env bash
# Camera preview / workspace alignment, no arm motion.
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_module teleop_viz "$@"
