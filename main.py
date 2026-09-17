"""Detect people in a Hikvision RTSP stream and drive a GPIO output."""

from __future__ import annotations

import argparse
from collections import deque
import json
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import cv2
import numpy as np


PERSON_CLASS_ID = 0
YOLO_INPUT_SIZE = 640
MODEL_FILENAME = "yolov8n.onnx"
MODEL_DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
)


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read configuration {path}: {exc}") from exc


def build_rtsp_url(config: dict) -> str:
    """Build the Hikvision channel URL, e.g. channel 1 main stream is 101."""
    missing = [key for key in ("ip", "username", "password") if key not in config]
    if missing:
        raise RuntimeError(f"config.json is missing key(s): {', '.join(missing)}")
    ip = str(config["ip"]).strip()
    if not ip:
        raise RuntimeError("Camera IP/hostname cannot be empty")
    user = quote(str(config["username"]), safe="")
    password = quote(str(config["password"]), safe="")
    try:
        port = int(config.get("port", 554))
        channel = int(config.get("channel", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Camera port and channel must be integers") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("Camera port must be between 1 and 65535")
    if channel < 1:
        raise RuntimeError("Camera channel must be 1 or greater")
    stream = str(config.get("stream", "main")).lower()
    stream_code = {"main": "01", "sub": "02"}.get(stream)
    if stream_code is None:
        raise RuntimeError(f"Camera stream must be 'main' or 'sub', got {stream!r}")
    return f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}{stream_code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.json",
        help="JSON file containing camera, GPIO, and detection settings",
    )
    parser.add_argument("--gpio", type=int, default=None, help="Override the BCM GPIO number")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Weak detection confidence (config default: 0.25)",
    )
    parser.add_argument(
        "--strong-threshold", type=float, default=None,
        help="Immediate-ON confidence (config default: 0.50)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Target inference rate (config default: 4; 0 disables the cap)",
    )
    parser.add_argument(
        "--off-delay", type=float, default=None,
        help="Seconds without a confirmed detection before turning off",
    )
    parser.add_argument(
        "--boot-blink", type=float, default=3.0,
        help="Rapid output-test duration before loading the model (0 disables it)",
    )
    parser.add_argument(
        "--startup-blink", type=float, default=0.5,
        help="Output pulse after the RTSP stream starts (0 disables it)",
    )
    parser.add_argument("--preview", action="store_true", help="Show annotated video")
    parser.add_argument("--active-low", action="store_true", help="Use LOW as the active output")
    parser.add_argument("--no-gpio", action="store_true", help="Run without GPIO hardware")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Use this ONNX file directly instead of the auto-downloaded default "
             "(config default: models/yolov8n.onnx)",
    )
    parser.add_argument(
        "--input-size", type=int, default=None,
        help="Network input size in pixels; must match --weights (config default: 640)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="cv2 thread count override (config default: OpenCV's own default)",
    )
    return parser.parse_args()


class Camera:
    """Continuously drain RTSP and expose only the newest decoded frame."""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.capture = None
        self.latest_frame = None
        self.latest_frame_at = None
        self.sequence = 0
        self.last_error = None
        self.thread = threading.Thread(
            target=self._capture_loop, name="rtsp-capture", daemon=True
        )
        self.thread.start()
        try:
            self.read(after_sequence=0, timeout=10.0)
        except Exception:
            self.close()
            raise

    def _open_capture(self):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ]
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG, params)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open Hikvision RTSP stream")
        return capture

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            capture = None
            try:
                print("RTSP capture: connecting", flush=True)
                capture = self._open_capture()
                with self.condition:
                    self.capture = capture
                    self.last_error = None
                print("RTSP capture: connected", flush=True)
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError("camera stopped returning frames")
                    captured_at = time.monotonic()
                    with self.condition:
                        self.latest_frame = frame
                        self.latest_frame_at = captured_at
                        self.sequence += 1
                        self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.last_error = str(exc)
                    self.condition.notify_all()
                if not self.stop_event.is_set():
                    print(f"RTSP capture: {exc}; reconnecting in 1s", flush=True)
            finally:
                with self.condition:
                    if self.capture is capture:
                        self.capture = None
                if capture is not None:
                    capture.release()
            self.stop_event.wait(1.0)

    def read(self, after_sequence: int, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after_sequence and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self.last_error}" if self.last_error else ""
                    raise TimeoutError(f"No fresh RTSP frame within {timeout:g}s{detail}")
                self.condition.wait(remaining)
            if self.latest_frame is None:
                raise RuntimeError("RTSP capture closed before receiving a frame")
            skipped = max(0, self.sequence - after_sequence - 1)
            return self.latest_frame, self.sequence, self.latest_frame_at, skipped

    def close(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        # Only the capture thread releases VideoCapture. Cross-thread release can
        # race FFmpeg and segfault during a systemd shutdown.
        self.thread.join(timeout=4.0)
        if self.thread.is_alive():
            print("RTSP capture: reader did not stop within 4s", flush=True)


class Output:
    def __init__(self, gpio: int, active_low: bool, disabled: bool):
        self.device = None
        self.state = False
        if not disabled:
            try:
                from gpiozero import DigitalOutputDevice
            except ImportError as exc:
                raise RuntimeError("gpiozero is not installed; use --no-gpio for testing") from exc
            self.device = DigitalOutputDevice(
                gpio, active_high=not active_low, initial_value=False
            )

    def set(self, active: bool) -> None:
        if active == self.state:
            return
        self.state = active
        if self.device:
            self.device.on() if active else self.device.off()
        print(f"GPIO output: {'ON' if active else 'OFF'}", flush=True)

    def blink(self, duration: float) -> None:
        if duration <= 0:
            return
        print("Startup output test", flush=True)
        self.set(True)
        time.sleep(duration)
        self.set(False)

    def flash(self, duration: float, on_time: float = 0.15, off_time: float = 0.15) -> None:
        if duration <= 0:
            return
        print("Boot flash test", flush=True)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.set(True)
            time.sleep(on_time)
            self.set(False)
            time.sleep(off_time)

    def close(self) -> None:
        self.set(False)
        if self.device:
            self.device.close()


class StatusMonitor:
    """Print stream, decision, and GPIO health once per second."""

    def __init__(self, output: Output, gpio: int):
        self.output = output
        self.gpio = gpio
        self.last_frame_at = None
        self.frames = 0
        self.people = 0
        self.confidence = 0.0
        self.confirmed = False
        self.weak_hits = 0
        self.frame_age_ms = 0.0
        self.skipped = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="status-monitor", daemon=True
        )
        self.thread.start()

    def frame_received(
        self, people: int, confidence: float, confirmed: bool, weak_hits: int,
        frame_age_ms: float, skipped: int,
    ) -> None:
        with self.lock:
            self.last_frame_at = time.monotonic()
            self.frames += 1
            self.people = people
            self.confidence = confidence
            self.confirmed = confirmed
            self.weak_hits = weak_hits
            self.frame_age_ms = frame_age_ms
            self.skipped += skipped

    def _run(self) -> None:
        last_report = time.monotonic()
        next_report = last_report + 1.0
        while not self.stop_event.wait(max(0.0, next_report - time.monotonic())):
            now = time.monotonic()
            with self.lock:
                last_frame_at = self.last_frame_at
                frames = self.frames
                people = self.people
                confidence = self.confidence
                confirmed = self.confirmed
                weak_hits = self.weak_hits
                frame_age_ms = self.frame_age_ms
                skipped = self.skipped
                self.frames = 0
                self.skipped = 0
            if last_frame_at is None:
                stream_status = "STARTING"
            elif now - last_frame_at >= 2.0:
                stream_status = "STALLED"
            else:
                stream_status = "OK"
            elapsed = now - last_report
            measured_fps = frames / elapsed if elapsed > 0 else 0.0
            print(
                f"Status: stream={stream_status} gpio=BCM{self.gpio}:"
                f"{'ON' if self.output.state else 'OFF'} fps={measured_fps:.1f} "
                f"people={people} conf={confidence:.2f} "
                f"confirmed={'YES' if confirmed else 'NO'} weak_hits={weak_hits}/3 "
                f"age_ms={frame_age_ms:.0f} skipped={skipped}",
                flush=True,
            )
            last_report = now
            next_report += 1.0
            if next_report <= now:
                next_report = now + 1.0

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.5)


def preprocess(frame):
    """Letterbox frame to a square canvas and build the YOLO input blob."""
    height, width = frame.shape[:2]
    length = max(height, width)
    square = np.zeros((length, length, 3), dtype=np.uint8)
    square[0:height, 0:width] = frame
    scale = length / YOLO_INPUT_SIZE
    blob = cv2.dnn.blobFromImage(
        square, scalefactor=1 / 255, size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True
    )
    return blob, scale, width, height


def postprocess(raw_output, scale: float, width: int, height: int, threshold: float):
    """Turn a raw (84, N) YOLOv8 output into person boxes as (x1, y1, x2, y2, confidence)."""
    outputs = cv2.transpose(raw_output)
    boxes = []
    scores = []
    for row in outputs:
        confidence = float(row[4 + PERSON_CLASS_ID])
        if confidence < threshold:
            continue
        cx, cy, box_width, box_height = row[0:4]
        boxes.append([
            cx - 0.5 * box_width, cy - 0.5 * box_height, box_width, box_height
        ])
        scores.append(confidence)
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.45, 0.5)
    people = []
    for index in np.array(indices).flatten():
        x, y, box_width, box_height = boxes[index]
        x1 = max(0, int(x * scale))
        y1 = max(0, int(y * scale))
        x2 = min(width - 1, int((x + box_width) * scale))
        y2 = min(height - 1, int((y + box_height) * scale))
        people.append((x1, y1, x2, y2, scores[index]))
    return people


def detect_people(net, frame, threshold: float):
    """Run YOLOv8n and return person boxes as (x1, y1, x2, y2, confidence)."""
    blob, scale, width, height = preprocess(frame)
    net.setInput(blob)
    raw_output = net.forward()[0]
    return postprocess(raw_output, scale, width, height, threshold)


def confirm_detection(
    max_confidence: float,
    weak_history: deque[bool],
    weak_threshold: float,
    strong_threshold: float,
) -> tuple[bool, int]:
    """Confirm strong detections immediately and weak detections in 2 of 3 frames."""
    weak_detected = max_confidence >= weak_threshold
    weak_history.append(weak_detected)
    weak_hits = sum(weak_history)
    strong_detected = max_confidence >= strong_threshold
    confirmed = strong_detected or (weak_detected and weak_hits >= 2)
    return confirmed, weak_hits


def ensure_weights(model_dir: Path) -> Path:
    """Return the ONNX weights path, downloading the official export if missing."""
    weights = model_dir / MODEL_FILENAME
    if weights.is_file():
        return weights
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model not found at {weights}; downloading from {MODEL_DOWNLOAD_URL}", flush=True)
    tmp_path = weights.with_suffix(".onnx.part")
    try:
        with urlopen(MODEL_DOWNLOAD_URL, timeout=120) as response:
            tmp_path.write_bytes(response.read())
    except (OSError, URLError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {MODEL_FILENAME}: {exc}") from exc
    tmp_path.replace(weights)
    print(f"Saved {weights} ({weights.stat().st_size / 1_000_000:.1f} MB)", flush=True)
    return weights


def load_network(weights: Path):
    return cv2.dnn.readNetFromONNX(str(weights))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    args.threshold = (
        args.threshold if args.threshold is not None else float(config.get("threshold", 0.25))
    )
    args.strong_threshold = (
        args.strong_threshold if args.strong_threshold is not None
        else float(config.get("strong_threshold", 0.50))
    )
    args.off_delay = (
        args.off_delay if args.off_delay is not None else float(config.get("off_delay", 3.0))
    )
    args.fps = args.fps if args.fps is not None else float(config.get("fps", 4.0))
    gpio = args.gpio if args.gpio is not None else int(config.get("gpio", 11))
    weights_override = args.weights if args.weights is not None else config.get("weights")
    input_size = (
        args.input_size if args.input_size is not None else config.get("input_size")
    )
    threads = args.threads if args.threads is not None else config.get("threads")
    if not 0.0 <= args.threshold <= args.strong_threshold <= 1.0:
        raise RuntimeError("Thresholds must satisfy 0 <= threshold <= strong_threshold <= 1")
    if args.off_delay < 0 or args.fps < 0:
        raise RuntimeError("off_delay and fps cannot be negative")
    if input_size:
        global YOLO_INPUT_SIZE
        YOLO_INPUT_SIZE = int(input_size)
    if threads:
        cv2.setNumThreads(int(threads))
    print(
        f"Settings: fps={args.fps:g} weak_threshold={args.threshold:g} "
        f"strong_threshold={args.strong_threshold:g} "
        f"off_delay={args.off_delay:g}s gpio=BCM{gpio} "
        f"input_size={YOLO_INPUT_SIZE} threads={cv2.getNumThreads()}", flush=True,
    )
    output = Output(gpio, args.active_low, args.no_gpio)
    output.flash(args.boot_blink)
    camera = None
    status_monitor = None
    try:
        if weights_override:
            weights = Path(weights_override)
            if not weights.is_file():
                print(f"Missing model file: {weights}", file=sys.stderr)
                return 2
        else:
            try:
                weights = ensure_weights(args.model_dir)
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 2
        net = load_network(weights)
        camera = Camera(build_rtsp_url(config))
        output.blink(args.startup_blink)
        running = True

        def stop(_signum=None, _frame=None):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        last_seen = float("-inf")
        last_sequence = 0
        weak_history: deque[bool] = deque(maxlen=3)
        frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
        status_monitor = StatusMonitor(output, gpio)
        while running:
            loop_start = time.monotonic()
            try:
                frame, sequence, captured_at, skipped = camera.read(
                    after_sequence=last_sequence, timeout=1.0
                )
            except TimeoutError:
                output.set(time.monotonic() - last_seen <= args.off_delay)
                continue
            last_sequence = sequence
            people = detect_people(net, frame, args.threshold)
            max_confidence = max((person[4] for person in people), default=0.0)
            confirmed, weak_hits = confirm_detection(
                max_confidence, weak_history, args.threshold, args.strong_threshold
            )
            now = time.monotonic()
            if confirmed:
                last_seen = now
            output.set(now - last_seen <= args.off_delay)
            status_monitor.frame_received(
                len(people), max_confidence, confirmed, weak_hits,
                (now - captured_at) * 1000.0, skipped,
            )
            if args.preview:
                for x1, y1, x2, y2, confidence in people:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"person {confidence:.0%}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                cv2.imshow("Human detector - q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_interval:
                remaining = frame_interval - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if status_monitor is not None:
            status_monitor.close()
        if camera is not None:
            camera.close()
        output.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
