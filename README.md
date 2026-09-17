# StreetL RTSP human detector

StreetL reads a Hikvision DVR/NVR RTSP stream, detects people with a YOLOv8n
ONNX model, and controls a Raspberry Pi GPIO output. The deployed configuration
uses camera 1's main stream (`/Streaming/Channels/101`) and BCM GPIO 11.

## Detection behavior

- A dedicated capture thread continuously drains RTSP and retains only the
  newest frame, preventing latency from an accumulated video backlog.
- Inference runs at 4 FPS by default.
- Confidence `>= 0.50` confirms a person immediately.
- Confidence from `0.25` through `0.49` requires a hit in 2 of the latest 3
  processed frames.
- The output turns off only after 3 continuous seconds without a confirmed
  detection. A new confirmed detection restarts that delay.
- A health line is printed every second with stream state, GPIO state, measured
  FPS, confidence, confirmation state, frame age, and skipped-frame count.
- Failed RTSP reads use a two-second timeout and reconnect automatically.

## Files required on the Raspberry Pi

Place a YOLOv8n ONNX model at:

```text
models/yolov8n.onnx
```

Create the local configuration from the safe example:

```bash
cp config.example.json config.json
```

Edit `config.json` with the real DVR address and credentials. This file is
Git-ignored and must not be committed.

```json
{
  "ip": "192.168.1.50",
  "port": 554,
  "username": "admin",
  "password": "change-me",
  "channel": 1,
  "stream": "main",
  "gpio": 11,
  "fps": 4,
  "threshold": 0.25,
  "strong_threshold": 0.5,
  "off_delay": 3
}
```

`gpio` is a BCM number. BCM 11 is physical header pin 23. Use a suitable
transistor or 3.3 V-compatible relay module for a lamp or other load; do not
power a load directly from the GPIO pin.

## Run

Install the Python dependencies in the project's virtual environment and start
the detector:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -u main.py
```

For a development machine without GPIO hardware:

```bash
.venv/bin/python main.py --no-gpio --boot-blink 0 --startup-blink 0
```

## systemd service

The recovered service definition expects the same deployed path used by the
working device:

```text
/home/admin/Desktop/StreetHumanDetector
```

Install and start it with:

```bash
sudo cp streetl.service /etc/systemd/system/streetl.service
sudo systemctl daemon-reload
sudo systemctl enable --now streetl
journalctl -u streetl -f -o cat
```

The service uses the `lgpio` GPIO backend, forces RTSP over TCP, and restarts
automatically after failures.
