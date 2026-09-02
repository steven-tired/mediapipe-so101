#!/usr/bin/env bash
# SO-101 servo fault diagnostics. Run the steps in order; `--help` lists them.
# THE ARM MOVES during the `lift` step, and `relax` drops torque so it will fall
# under gravity. Support the arm and keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: register reads
run_module so101_diag "$@"
