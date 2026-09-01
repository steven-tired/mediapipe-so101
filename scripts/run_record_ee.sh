#!/usr/bin/env bash
# Record a LeRobot dataset from the teleop path (appends).
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_module record_so101_ee "$@"
