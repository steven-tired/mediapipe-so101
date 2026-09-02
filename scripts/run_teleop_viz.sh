#!/usr/bin/env bash
# Teleop the arm AND show the diagnostic overlay in one process.
# THE ARM MOVES. Right hand drives it, left fist freezes it.
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: MediaPipe + placo IK
run_module teleop_viz "$@"
