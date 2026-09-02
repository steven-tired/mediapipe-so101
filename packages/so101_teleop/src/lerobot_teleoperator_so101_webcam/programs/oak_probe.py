"""Probe a connected OAK device: confirm it is a STEREO OAK-D, not an OAK-1.

Stereo depth needs a left+right mono pair (sockets CAM_B / CAM_C). If only
CAM_A (RGB) is present it is an OAK-1, there is no depth, and the OAK depth
plan is impossible.

This does not move the arm. It is the first thing to run against the open OAK
gate: `oak_camera.py` targets the depthai v3 API while the environment has
depthai 2.32, so this probe is also where that mismatch shows up first.

Run:  ./scripts/probe_oak.sh
"""


def main():
    # Imported here, not at module scope: depthai is an optional dependency and
    # every program in this package must import with nothing plugged in.
    import depthai as dai

    avail = dai.Device.getAllAvailableDevices()
    print(f"Available OAK devices: {[d.name for d in avail] or 'NONE (plug it in / check USB3 cable)'}")
    if not avail:
        return

    with dai.Device() as device:
        try:
            print("Device name:", device.getDeviceName())
        except Exception:
            pass
        cams = device.getConnectedCameraFeatures()
        print(f"\nConnected cameras ({len(cams)}):")
        sockets = []
        for c in cams:
            sockets.append(str(c.socket))
            print(f"  socket={c.socket}  sensor={c.sensorName}  "
                  f"res={c.width}x{c.height}  types={[str(t) for t in c.supportedTypes]}")
        mono_pair = sum(s.endswith(('CAM_B', 'CAM_C')) for s in sockets)
        print("\n=== VERDICT ===")
        if mono_pair >= 2:
            print("STEREO present (CAM_B + CAM_C) -> this is an OAK-D. Depth is possible. PROCEED.")
        else:
            print("NO stereo pair -> looks like an OAK-1 (RGB only). Depth NOT possible. STOP.")


if __name__ == "__main__":
    main()
