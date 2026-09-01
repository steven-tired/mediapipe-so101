"""Webcam + MediaPipe input for LeFranX-style teleoperation.

Standalone package. Reuses the seniors' MediaPipe→MANO detector
(vr-dex-retargeting SingleHandDetector); adds only the genuinely new pieces:
metric wrist position + depth, fist/clutch, VR-frame mapping, source/manager glue.
"""
