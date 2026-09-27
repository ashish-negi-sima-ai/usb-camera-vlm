"""Continuous USB capture shared by browser viewers and vision questions."""

from datetime import datetime, timezone
import threading
import time

import main as app


class LiveCamera:
    def __init__(self, args, stopping):
        self.args, self.stopping = args, stopping
        self.condition = threading.Condition()
        self.frame = self.jpeg = None
        self.sequence = 0
        self.received = 0
        self.captured_at = self.device = None
        self.error = None
        self.fps = 0
        self.thread = threading.Thread(target=self.run, name='usb-preview', daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        with self.condition:
            self.condition.notify_all()
        self.thread.join(3)

    def state(self):
        with self.condition:
            fresh = self.frame is not None and time.monotonic() - self.received < 2
            return dict(status='streaming' if fresh else 'error' if self.error else 'starting',
                        sequence=self.sequence, captured_at=self.captured_at,
                        fps=self.fps if fresh else 0,
                        error=self.error or (None if fresh else 'Waiting for camera frames…'))

    def publish(self, frame, jpeg, device, captured_at, fps):
        with self.condition:
            self.frame, self.jpeg, self.device = frame, jpeg, device
            self.captured_at, self.received = captured_at, time.monotonic()
            self.sequence += 1
            self.error, self.fps = None, fps
            self.condition.notify_all()

    def failed(self, error):
        with self.condition:
            self.frame = self.jpeg = None
            self.captured_at = None
            self.error = str(error)
            self.condition.notify_all()

    def snapshot(self, timeout=2):
        """Wait for a frame published after Ask, then give the worker its own copy."""
        deadline = time.monotonic() + timeout
        with self.condition:
            previous = self.sequence
            while not self.stopping.is_set():
                if self.error:
                    raise RuntimeError(f'USB camera unavailable: {self.error}')
                if self.sequence > previous and self.frame is not None:
                    return self.frame.copy(), self.device, self.captured_at
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('No fresh camera frame. Check the USB camera and retry.')
                self.condition.wait(min(remaining, .2))
        raise RuntimeError('Camera is stopping.')

    def preview(self, previous, timeout=1):
        """Each viewer takes the newest JPEG; slow viewers never queue old frames."""
        with self.condition:
            self.condition.wait_for(lambda: self.stopping.is_set() or
                                    (self.sequence > previous and self.jpeg is not None), timeout)
            if self.jpeg is None or self.sequence <= previous or self.stopping.is_set():
                return None
            if time.monotonic() - self.received >= 2:
                return None
            return self.sequence, self.jpeg

    def run(self):
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception as exc:
            self.failed(exc)
            return
        while not self.stopping.is_set():
            cap = None
            try:
                cap, device = app.open_camera(self.args, cv2)
                warmup = self.args.warmup_frames
                next_frame = 0
                count, measured_fps = 0, 0
                window = time.monotonic()
                while not self.stopping.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None or not frame.size:
                        raise RuntimeError('No image from USB camera; reconnect it to resume.')
                    if frame.shape[:2] != (self.args.height, self.args.width):
                        raise RuntimeError('Camera returned an unsupported image size. Check --width/--height.')
                    warmup -= 1
                    now = time.monotonic()
                    if warmup > 0 or now < next_frame:
                        continue
                    captured_at = datetime.now(timezone.utc).isoformat()
                    # Read continuously to drain V4L2; publish at most 15 JPEGs/s.
                    next_frame = now + 1 / 15
                    width = min(960, frame.shape[1])
                    height = round(frame.shape[0] * width / frame.shape[1])
                    preview = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError('Could not encode USB video frame.')
                    count += 1
                    if now - window >= 1:
                        measured_fps = round(count / (now - window), 1)
                        count, window = 0, now
                    self.publish(frame, encoded.tobytes(), device, captured_at, measured_fps)
            except Exception as exc:
                self.failed(exc)
            finally:
                if cap is not None:
                    cap.release()
            self.stopping.wait(1)
