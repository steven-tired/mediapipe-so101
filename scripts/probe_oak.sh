#!/usr/bin/env bash
# Report what an attached OAK device actually is. NO ARM MOTION.
# The first check for the open OAK gate: depth needs a CAM_B/CAM_C mono pair.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: reads device descriptors
run_module oak_probe "$@"
