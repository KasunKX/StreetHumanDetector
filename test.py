"""Benchmark YOLOv8n ONNX inference throughput to gauge device feasibility.

Runs the same detect_people() pipeline main.py uses per frame, against
synthetic frames at the configured resolution, and reports the average
FPS the device can sustain compared to the 1.5-3 fps target.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import main as main_module
from main import detect_people, ensure_weights, load_network, postprocess, preprocess

TARGET_FPS_LOW = 1.5
TARGET_FPS_HIGH = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Use this ONNX file directly instead of the auto-downloaded default",
    )
    parser.add_argument(
        "--backend", choices=("opencv", "onnxruntime"), default="opencv",
        help="Inference backend to benchmark",
    )
    parser.add_argument(
        "--input-size", type=int, default=None,
        help="Network input size in pixels; must match --weights (default: 640)",
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Synthetic frame width (Hikvision main-stream resolution)",
    )
    parser.add_argument("--height", type=int, default=1080, help="Synthetic frame height")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10, help="Frames run before timing starts")
    parser.add_argument("--frames", type=int, default=100, help="Timed frames to measure")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="cv2 thread count override (0 keeps the OpenCV default)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic frame RNG seed")
    return parser.parse_args()


def make_frame(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def main() -> int:
    args = parse_args()
    if args.weights:
        weights = args.weights
        if not weights.is_file():
            print(f"Missing model file: {weights}")
            return 2
    else:
        try:
            weights = ensure_weights(args.model_dir)
        except RuntimeError as exc:
            print(exc)
            return 2
    if args.input_size:
        main_module.YOLO_INPUT_SIZE = args.input_size
    if args.frames <= 0:
        print("--frames must be positive")
        return 2

    if args.threads > 0:
        cv2.setNumThreads(args.threads)

    print(f"OpenCV {cv2.__version__}, cv2 threads={cv2.getNumThreads()}", flush=True)
    print(f"Loading model: {weights} (backend={args.backend})", flush=True)

    if args.backend == "onnxruntime":
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        if args.threads > 0:
            session_options.intra_op_num_threads = args.threads
        session = ort.InferenceSession(
            str(weights), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        def run_inference(frame):
            blob, scale, width, height = preprocess(frame)
            raw = session.run(None, {input_name: blob})[0][0]
            return postprocess(raw, scale, width, height, args.threshold)
    else:
        net = load_network(weights)

        def run_inference(frame):
            return detect_people(net, frame, args.threshold)

    frame = make_frame(args.width, args.height, args.seed)

    print(f"Warm-up: {args.warmup} frame(s) (excluded from stats)", flush=True)
    for _ in range(args.warmup):
        run_inference(frame)

    print(
        f"Benchmark: {args.frames} frame(s) at {args.width}x{args.height} synthetic input, "
        f"model input={main_module.YOLO_INPUT_SIZE}, threshold={args.threshold:g}",
        flush=True,
    )
    latencies_ms = []
    start = time.monotonic()
    for i in range(args.frames):
        frame_start = time.monotonic()
        run_inference(frame)
        latencies_ms.append((time.monotonic() - frame_start) * 1000.0)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.frames} frames...", flush=True)
    elapsed = time.monotonic() - start

    avg_fps = args.frames / elapsed if elapsed > 0 else 0.0
    sorted_latencies = sorted(latencies_ms)
    print("", flush=True)
    print(
        f"Latency ms: min={sorted_latencies[0]:.1f} p50={percentile(sorted_latencies, 0.50):.1f} "
        f"p95={percentile(sorted_latencies, 0.95):.1f} max={sorted_latencies[-1]:.1f} "
        f"mean={statistics.fmean(latencies_ms):.1f}",
        flush=True,
    )
    print(f"Average FPS: {avg_fps:.2f} over {elapsed:.1f}s ({args.frames} frames)", flush=True)
    if avg_fps >= TARGET_FPS_HIGH:
        verdict = f"MEETS TARGET ({TARGET_FPS_LOW:g}-{TARGET_FPS_HIGH:g} fps), with headroom"
    elif avg_fps >= TARGET_FPS_LOW:
        verdict = f"MEETS TARGET ({TARGET_FPS_LOW:g}-{TARGET_FPS_HIGH:g} fps), no headroom"
    else:
        verdict = f"BELOW TARGET ({TARGET_FPS_LOW:g}-{TARGET_FPS_HIGH:g} fps)"
    print(f"Verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
