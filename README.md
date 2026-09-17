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

`main.py` and `test.py` both download the official YOLOv8n ONNX export
automatically on first run if it isn't already present at:

```text
models/yolov8n.onnx
```

To place it manually instead (e.g. offline), download it from
[ultralytics/assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx)
into `models/`.

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
  "off_delay": 3,
  "weights": "models/yolov8n_320.onnx",
  "input_size": 320,
  "threads": 4
}
```

`gpio` is a BCM number. BCM 11 is physical header pin 23. Use a suitable
transistor or 3.3 V-compatible relay module for a lamp or other load; do not
power a load directly from the GPIO pin.

`weights`, `input_size`, and `threads` are optional and only needed to
override the default 640px model — see "Performance on weaker hardware"
below. `threads: 0` keeps OpenCV's own default thread count.

## Performance on weaker hardware

On a Raspberry Pi 3B+ (1GB RAM), the default 640px YOLOv8n model measured
**~0.5 FPS** — well below a 1.5-3 FPS target — using `test.py`. The deployed
config uses a **320px re-export at 4 threads, ~2.1 FPS sustained**, chosen as
the best speed/accuracy balance found for this hardware (see below for what
was tried and why).

1. **Re-export the model at a smaller input size.** A 640px input is far more
   resolution than needed to detect a person at typical camera distances.
   Re-exporting at 320px cut inference time roughly 4x with the same
   `yolov8n.pt` weights:

   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx imgsz=320 simplify=True opset=12
   ```

   Copy the resulting `yolov8n.onnx` to the Pi (e.g. as `models/yolov8n_320.onnx`)
   — already in place in this deployment's `models/` directory.

   Smaller input size trades away some accuracy on small/distant people;
   320px was chosen over an even faster 256px export (~2.6-2.75 FPS) as the
   better accuracy/speed balance. Validate against real footage once the
   camera is connected, and drop to 256px if 320px still misses too much.

2. **Tune OpenCV's thread count — the optimum isn't always "all cores".**
   The best thread count differs by model size: 320px was fastest at 4
   threads (2.13 FPS vs. 1.87 FPS at 3 threads), while 256px was fastest at 3
   threads (2.75 FPS vs. 2.70 FPS at 4) — leaving one core free can reduce
   scheduling contention, but it depends on the workload, so it's worth
   re-testing with `test.py --threads N` for whatever model size you deploy.

3. **Int8 quantization was tried and found ineffective on this hardware.**
   The 640px model was statically quantized to int8 (detection head kept in
   fp32 to preserve output accuracy — a naive full-model quantization
   collapsed all confidences to 0) and benchmarked via `onnxruntime`
   (`test.py --backend onnxruntime`). Accuracy held up (confidences within
   ~0.02 of the fp32 model on a real test image), but speed **did not
   improve** — still ~0.5 FPS. The Pi 3B+'s Cortex-A53 CPU is ARMv8.0-A,
   which predates the ARM dot-product instructions (`SDOT`/`UDOT`, added in
   ARMv8.2-A) that make int8 inference fast in practice. This is a hardware
   ceiling: full 640px accuracy at the target FPS would need different
   hardware (e.g. a Pi 4/5, whose Cortex-A72/A76 cores do have dot-product
   support) rather than a software fix on this board.

Also worth checking before assuming a resolution problem: boot the Pi
headless (`sudo systemctl set-default multi-user.target`) if it's running a
desktop image, since the desktop environment alone can consume ~300-400MB of
a 1GB Pi's RAM. And check `vcgencmd get_throttled` and `dmesg | grep -i volt`
— a marginal power supply or cable can cause under-voltage resets under the
CPU load spikes this workload produces, which looks like instability but is
a power problem, not a software one.

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
