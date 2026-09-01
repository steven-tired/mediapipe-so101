"""Print live SO-101 joint targets from the webcam, no robot. Ctrl+C to stop.

Run:
  env -u PYTHONPATH python -m lerobot_teleoperator_so101_webcam.programs.dry_run
"""

import time

from lerobot_teleoperator_so101_webcam.config_so101_webcam import SO101WebcamConfig
from lerobot_teleoperator_so101_webcam.so101_webcam import SO101Webcam


def main():
    teleop = SO101Webcam(SO101WebcamConfig(camera_index=0))
    teleop.connect(calibrate=False)
    print("Right hand controls; close LEFT fist to pause. Ctrl+C to stop.")
    try:
        while True:
            a = teleop.get_action()
            print(" ".join(f"{k.split('.')[0]:>13}={a[k]:+7.1f}" for k in a))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
