"""Offline tests for configuration and temporal detection behavior."""

from collections import deque
import importlib
import sys
import types
import unittest


try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.SimpleNamespace()

try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.SimpleNamespace()

app = importlib.import_module("main")


class HikvisionUrlTests(unittest.TestCase):
    def config(self, **overrides):
        config = {
            "ip": "192.168.1.50",
            "port": 554,
            "username": "camera user",
            "password": "p@ss/word",
            "channel": 1,
            "stream": "main",
        }
        config.update(overrides)
        return config

    def test_main_stream_and_encoded_credentials(self):
        self.assertEqual(
            app.build_rtsp_url(self.config()),
            "rtsp://camera%20user:p%40ss%2Fword@192.168.1.50:554/Streaming/Channels/101",
        )

    def test_substream_channel_number(self):
        url = app.build_rtsp_url(self.config(channel=3, stream="sub"))
        self.assertTrue(url.endswith("/Streaming/Channels/302"))

    def test_invalid_port_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "port"):
            app.build_rtsp_url(self.config(port=70000))


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.history = deque(maxlen=3)

    def confirm(self, confidence):
        return app.confirm_detection(confidence, self.history, 0.25, 0.50)

    def test_strong_detection_confirms_immediately(self):
        confirmed, hits = self.confirm(0.50)
        self.assertTrue(confirmed)
        self.assertEqual(hits, 1)

    def test_single_weak_detection_is_not_confirmed(self):
        confirmed, hits = self.confirm(0.30)
        self.assertFalse(confirmed)
        self.assertEqual(hits, 1)

    def test_two_weak_detections_in_three_frames_confirm(self):
        self.confirm(0.30)
        self.confirm(0.0)
        confirmed, hits = self.confirm(0.35)
        self.assertTrue(confirmed)
        self.assertEqual(hits, 2)

    def test_empty_current_frame_does_not_confirm_old_hits(self):
        self.confirm(0.30)
        self.confirm(0.35)
        confirmed, hits = self.confirm(0.0)
        self.assertFalse(confirmed)
        self.assertEqual(hits, 2)


if __name__ == "__main__":
    unittest.main()
