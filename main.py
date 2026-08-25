"""Detect people with MobileNet-SSD and drive a Raspberry Pi GPIO output."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2


PERSON_CLASS_ID = 15  # Pascal VOC class index used by MobileNet-SSD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("auto", "picamera", "usb"), default="auto")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index")
    parser.add_argument("--gpio", type=int, default=17, help="BCM GPIO number (physical pin 11 is BCM17)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection confidence")
    parser.add_argument("--off-delay", type=float, default=1.0, help="Seconds to hold output after last detection")
    parser.add_argument("--startup-blink", type=float, default=0.5,
                        help="Startup output pulse duration in seconds (0 disables it)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--preview", action="store_true", help="Show annotated camera window")
    parser.add_argument("--active-low", action="store_true", help="Use LOW as the active output level")
    parser.add_argument("--no-gpio", action="store_true", help="Test without Raspberry Pi GPIO hardware")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    return parser.parse_args()


class Camera:
    def __init__(self, kind: str, index: int, size: tuple[int, int]):
        self.picamera = None
        self.capture = None

        if kind in ("auto", "picamera"):
            try:
                from picamera2 import Picamera2

                self.picamera = Picamera2()
                config = self.picamera.create_video_configuration(
                    main={"size": size, "format": "RGB888"}
                )
                self.picamera.configure(config)
                self.picamera.start()
                time.sleep(0.5)
            except (ImportError, RuntimeError) as exc:
                self.picamera = None
                if kind == "picamera":
                    raise RuntimeError(f"Could not start Picamera2: {exc}") from exc

        if self.picamera is None:
            self.capture = cv2.VideoCapture(index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open USB camera index {index}")

    def read(self):
        if self.picamera is not None:
            # RGB888 from Picamera2 is arranged correctly for OpenCV display/DNN use.
            return self.picamera.capture_array()
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera stopped returning frames")
        return frame

    def close(self) -> None:
        if self.picamera is not None:
            self.picamera.stop()
            self.picamera.close()
        if self.capture is not None:
            self.capture.release()


class Output:
    def __init__(self, gpio: int, active_low: bool, disabled: bool):
        self.device = None
        self.state = False
        if not disabled:
            try:
                from gpiozero import DigitalOutputDevice
            except ImportError as exc:
                raise RuntimeError("gpiozero is not installed; install python3-gpiozero or use --no-gpio") from exc
            self.device = DigitalOutputDevice(gpio, active_high=not active_low, initial_value=False)

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

    def close(self) -> None:
        self.set(False)
        if self.device:
            self.device.close()


def detect_people(net, frame, threshold: float):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    people = []
    
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if class_id != PERSON_CLASS_ID or confidence < threshold:
            continue
        box = detections[0, 0, i, 3:7] * [width, height, width, height]
        x1, y1, x2, y2 = box.astype("int")
        people.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2), confidence))
    return people


def load_network(prototxt: Path, weights: Path):
    loader = getattr(cv2.dnn, "readNetFromCaffe", None)
    if loader is None:
        version = getattr(cv2, "__version__", "unknown")
        raise RuntimeError(
            f"OpenCV {version} has no Caffe model loader. This application currently "
            "requires OpenCV 4.x. In the active virtual environment run: "
            "pip uninstall -y opencv-python opencv-python-headless && "
            "pip install 'opencv-python>=4.8,<5'"
        )
    return loader(str(prototxt), str(weights))


def main() -> int:
    args = parse_args()
    prototxt = args.model_dir / "MobileNetSSD_deploy.prototxt"
    weights = args.model_dir / "MobileNetSSD_deploy.caffemodel"
    missing = [str(path) for path in (prototxt, weights) if not path.is_file()]
    if missing:
        print("Missing model file(s): " + ", ".join(missing), file=sys.stderr)
        print("Run: python download_models.py", file=sys.stderr)
        return 2

    net = load_network(prototxt, weights)
    camera = Camera(args.camera, args.camera_index, (args.width, args.height))
    output = Output(args.gpio, args.active_low, args.no_gpio)
    output.blink(args.startup_blink)
    running = True

    def stop(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    last_seen = float("-inf")

    try:
        while running:
            frame = camera.read()
            people = detect_people(net, frame, args.threshold)
            now = time.monotonic()
            if people:
                last_seen = now
            output.set(now - last_seen <= args.off_delay)

            if args.preview:
                for x1, y1, x2, y2, confidence in people:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"person {confidence:.0%}", (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Human detector - q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        output.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
