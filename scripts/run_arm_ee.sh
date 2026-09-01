#!/usr/bin/env bash
# Webcam end-effector teleop (drives the arm via IK).
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_module teleop_viz_ee "$@"
