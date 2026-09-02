#!/usr/bin/env bash
# Live camera preview for aligning the policy camera. NO ARM MOTION.
#
# --profile picks which trained layout to align to: dp100 (default, matches
# DP100/ACT100) or dp50. Getting this wrong is a visual domain shift the policy
# sees and nothing reports, so run it before any deployment.
#
# For the arm-driving diagnostic preview, use run_teleop_viz.sh instead.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hide_gpu           # CPU-only: OpenCV preview
run_module view_camera "$@"
