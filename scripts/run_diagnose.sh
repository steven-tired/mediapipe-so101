#!/usr/bin/env bash
# Telemetry harness for debugging a misbehaving deploy.
# Stop other camera apps first. Keep the e-stop within reach.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_module diagnose_deploy "$@"
