#!/usr/bin/env bash
# Gripper deadband calibration on a held carton (gate 1 of the 2026-09-01 plan).
# THE GRIPPER MOVES with --arm-enabled: it walks a staircase against whatever is
# in the jaw. Without that flag nothing is sent and it only reads telemetry.
# Close the jaw on the carton before starting; keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: servo commands and register reads
run_module gripper_deadband "$@"
