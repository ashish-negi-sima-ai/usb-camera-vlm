"""Lifecycle and input validation checks; no model or camera required."""

import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main as app
from webapp import Application, ModelWorker
from camera import LiveCamera


def slow_native_worker(sender):
    import ctypes
    sender.send({'png': b'snapshot', 'status': 'loading'})
    # PyDLL deliberately keeps the GIL, like the installed model constructor.
    ctypes.PyDLL(None).sleep(1)
    sender.send({'status': 'ready'})
    sender.close()


class ValidationTests(unittest.TestCase):
    def test_missing_and_incomplete_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                app.validate_model(root)
            (root / 'devkit').mkdir()
            (root / 'devkit/vlm_config.json').write_text(json.dumps({
                'model_type': 'vlm-gemma4', 'vision_model_name': 'vision',
                'language_model_name': 'language', 'vm_cfg': {'image_size': [480, 480]},
            }))
            with self.assertRaisesRegex(ValueError, 'ELF'):
                app.validate_model(root)
            (root / 'elf_files').mkdir()
            for name in ('vision', 'language'):
                (root / f'elf_files/{name}.elf').write_bytes(b'artifact')
            app.validate_model(root)
            (root / 'elf_files/vision.elf').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'empty'):
                app.validate_model(root)

    def test_ambiguous_camera_is_not_silently_selected(self):
        with patch.object(app, 'camera_paths', return_value=[Path('/dev/video1'), Path('/dev/video2')]):
            with self.assertRaisesRegex(ValueError, 'Found 2'):
                app.resolve_camera('auto')

    def test_regular_file_is_not_camera(self):
        with tempfile.NamedTemporaryFile() as file:
            with self.assertRaisesRegex(ValueError, 'V4L2'):
                app.resolve_camera(file.name)

    def test_failed_capture_releases_camera(self):
        cv2 = Mock()
        camera = cv2.VideoCapture.return_value
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        args = SimpleNamespace(camera='auto', width=1280, height=720, warmup_frames=2)
        with patch.object(app, 'resolve_camera', return_value='/dev/video1'):
            with self.assertRaisesRegex(RuntimeError, 'No image'):
                app.capture(args, cv2)
        camera.release.assert_called_once()

    def test_capture_uses_last_warmup_frame_and_releases(self):
        cv2 = Mock()
        camera = cv2.VideoCapture.return_value
        frames = [SimpleNamespace(size=12, shape=(720, 1280, 3)) for _ in range(3)]
        camera.read.side_effect = [(True, frame) for frame in frames]
        args = SimpleNamespace(camera='auto', width=1280, height=720, warmup_frames=3)
        with patch.object(app, 'resolve_camera', return_value='/dev/video1'):
            bgr, device, timestamp = app.capture(args, cv2)
        self.assertIs(bgr, frames[-1])
        self.assertEqual(device, '/dev/video1')
        self.assertTrue(timestamp.endswith('+00:00'))
        camera.release.assert_called_once()


class GenerationTests(unittest.TestCase):
    def test_empty_answer_is_an_error(self):
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter([]))
        model = SimpleNamespace(stream=lambda request: stream)
        with self.assertRaisesRegex(RuntimeError, 'empty answer'):
            app.generate(model, object(), 1, lambda text: None)
        stream.cancel.assert_called_once()

    def test_timeout_cancels_a_blocked_stream(self):
        class Stream:
            def __init__(self):
                self.cancelled = threading.Event()

            def cancel(self):
                self.cancelled.set()

            def __iter__(self):
                if not self.cancelled.wait(2):
                    raise AssertionError('Generation was never cancelled')
                return iter([])

        stream = Stream()
        model = SimpleNamespace(stream=lambda request: stream)
        with self.assertRaises(TimeoutError):
            app.generate(model, object(), .02, lambda text: None)
        self.assertTrue(stream.cancelled.is_set())


class BrowserStateTests(unittest.TestCase):
    def setUp(self):
        self.app = Application(SimpleNamespace(prompt='Describe the image', max_tokens=160))
        self.app.state.update(model_ready=True, status='ready', snapshot_id=3)
        self.app.png = b'snapshot'
        self.app.camera.snapshot = Mock(return_value=('frame', '/dev/video1', 'now'))

    def tearDown(self):
        self.app.jobs.close()
        self.app.jobs.join_thread()
        self.app.responses.close()
        self.app.sender.close()

    def test_vision_question_reserves_a_fresh_capture(self):
        self.app.submit('ask', {'prompt': 'What is visible?', 'vision': True})
        self.assertEqual(self.app.state['status'], 'capturing')
        self.assertIsNone(self.app.png)
        self.assertIsNone(self.app.state['captured_at'])
        self.app.camera.snapshot.assert_called_once()
        payload = self.app.jobs.get(timeout=1)[1]
        self.assertEqual(payload['snapshot'], ('frame', '/dev/video1', 'now'))

    def test_text_question_needs_no_snapshot_and_blocks_other_operations(self):
        self.app.png = None
        self.app.submit('ask', {'prompt': 'Hello', 'vision': False})
        self.assertEqual(self.app.state['status'], 'answering')
        with self.assertRaisesRegex(RuntimeError, 'Wait'):
            self.app.submit('clear', {})
        with self.assertRaisesRegex(RuntimeError, 'Wait'):
            self.app.submit('ask', {'prompt': 'Again', 'vision': False})
        action, payload = self.app.jobs.get(timeout=1)
        self.assertEqual(action, 'ask')
        self.assertEqual(payload['prompt'], 'Hello')
        self.assertFalse(payload['vision'])
        self.assertIsNone(payload['snapshot'])
        self.app.camera.snapshot.assert_not_called()

    def test_camera_failure_does_not_reserve_model_worker(self):
        self.app.camera.snapshot.side_effect = RuntimeError('No fresh frame')
        with self.assertRaisesRegex(RuntimeError, 'No fresh'):
            self.app.submit('ask', {'prompt': 'What is visible?', 'vision': True})
        self.assertEqual(self.app.state['status'], 'ready')

    def test_vision_flag_must_be_an_explicit_boolean(self):
        for value in (None, 'false', 0, []):
            with self.assertRaisesRegex(ValueError, 'vision'):
                self.app.submit('ask', {'prompt': 'Hello', 'vision': value})
        self.assertEqual(self.app.state['status'], 'ready')

    def test_capture_action_was_removed(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.app.submit('capture', {})

    def test_blank_question_does_not_reserve_worker(self):
        with self.assertRaises(ValueError):
            self.app.submit('ask', {'prompt': ' ', 'vision': True})
        self.assertEqual(self.app.state['status'], 'ready')


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(max_tokens=160)
        self.neat = SimpleNamespace(genai=SimpleNamespace(
            ChatMessage=lambda: SimpleNamespace(images=[], use_cached_images=False),
            GenerationRequest=lambda: SimpleNamespace(prompt=None, images=[], use_cached_images=False)))

    def test_only_current_message_has_image_and_roles_alternate(self):
        history = [{'prompt': 'My name is Ada', 'answer': 'Hello Ada', 'vision': True}]
        image = object()
        request, context = app.chat_request(self.neat, self.args, 'What is here?', history, image)
        self.assertEqual([m.role for m in request.messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual([m.content for m in request.messages[1:]], ['My name is Ada', 'Hello Ada', 'What is here?'])
        self.assertEqual([m.images for m in request.messages], [[], [], [], [image]])
        self.assertIsNone(request.prompt)
        self.assertEqual(request.images, [])
        request, _ = app.chat_request(self.neat, self.args, 'What is my name?', context)
        self.assertTrue(all(not m.images and not m.use_cached_images for m in request.messages))
        self.assertFalse(request.use_cached_images)

    def test_history_keeps_recent_complete_pairs_with_byte_budget(self):
        turns = [dict(prompt=str(i), answer='ok') for i in range(5)]
        self.assertEqual(app.short_history(turns), turns[-3:])
        self.assertEqual(app.short_history(turns, 6), turns[-2:])
        self.assertEqual(app.short_history([dict(prompt='😀', answer='😀')], 7), [])

    def test_large_question_reserves_room_for_answer(self):
        self.args.max_tokens = 2048
        with self.assertRaisesRegex(ValueError, 'too long'):
            app.chat_request(self.neat, self.args, '😀' * 2000, [])
        turns = [dict(prompt='x' * 1000, answer='y' * 300) for _ in range(3)]
        _, context = app.chat_request(self.neat, self.args, 'z' * 2000, turns)
        self.assertEqual(len(context), 2)

    def test_text_only_ask_does_not_touch_camera_or_save_images(self):
        with tempfile.TemporaryDirectory() as folder:
            self.args.output_dir = Path(folder)
            self.args.model = Path('/model')
            self.args.timeout = 1
            cv = Mock()
            report = dict(answer='Hello', finish_reason='stop', generation_s=.1)
            with patch.object(app, 'capture') as capture, patch.object(app, 'generate', return_value=report):
                result = app.ask(SimpleNamespace(model_id=lambda: 'test-gemma'), self.neat,
                                 self.args, 'Hello', cv, Mock(), vision=False)
            capture.assert_not_called()
            self.assertEqual(cv.mock_calls, [])
            self.assertEqual([p.name for p in Path(result['result_path']).parent.iterdir()], ['result.json'])
            self.assertIsNone(result['image'])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = ModelWorker(SimpleNamespace(), Mock(), Mock(), threading.Event(), Mock(), Mock(), Mock())
        self.model = Mock()

    def test_each_vision_turn_captures_and_text_turn_skips_camera(self):
        with patch.object(self.worker, 'capture') as capture, patch.object(app, 'ask', return_value={'answer': 'ok'}) as ask:
            for vision in (True, True, False, False):
                self.worker.handle(self.model, 'ask', {'prompt': str(vision), 'vision': vision})
            self.assertEqual(capture.call_count, 2)
            self.assertFalse(ask.call_args.kwargs['vision'])
            self.assertIsNone(ask.call_args.kwargs['snapshot'])
            self.assertEqual(len(self.worker.history), 3)
            self.assertEqual(len(ask.call_args.kwargs['history']), 3)

    def test_capture_failure_does_not_infer_or_change_history(self):
        self.worker.history = [{'prompt': 'Old', 'answer': 'Okay'}]
        with patch.object(self.worker, 'capture', side_effect=RuntimeError('Camera missing')), patch.object(app, 'ask') as ask:
            with self.assertRaisesRegex(RuntimeError, 'Camera missing'):
                self.worker.handle(self.model, 'ask', {'prompt': 'New', 'vision': True})
            ask.assert_not_called()
        self.assertEqual(self.worker.history, [{'prompt': 'Old', 'answer': 'Okay'}])

    def test_generation_failure_is_not_added_to_history(self):
        with patch.object(app, 'ask', side_effect=TimeoutError('Timed out')):
            with self.assertRaises(TimeoutError):
                self.worker.handle(self.model, 'ask', {'prompt': 'Hello', 'vision': False})
        self.assertEqual(self.worker.history, [])

    def test_clear_removes_history_and_runtime_cache(self):
        self.worker.history = [{'prompt': 'Secret', 'answer': 'Okay'}]
        self.worker.handle(self.model, 'clear', None)
        self.model.clear_kv_caches.assert_called_once()
        self.assertEqual(self.worker.history, [])
        self.assertIsNone(self.worker.snapshot)
        self.assertEqual(self.worker.responses.send.call_args.args[0]['history'], [])


class ProcessTests(unittest.TestCase):
    def test_snapshot_arrives_before_native_loading_finishes(self):
        context = multiprocessing.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=slow_native_worker, args=(sender,))
        process.start()
        sender.close()
        try:
            self.assertTrue(receiver.poll(20), 'No startup snapshot')
            self.assertEqual(receiver.recv()['png'], b'snapshot')
            self.assertFalse(receiver.poll(.05), 'Native load should still be running')
            self.assertTrue(receiver.poll(3))
            self.assertEqual(receiver.recv()['status'], 'ready')
        finally:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)
            receiver.close()
            process.close()


class LiveCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = LiveCamera(SimpleNamespace(), threading.Event())

    def test_snapshot_waits_for_new_frame_and_copies_it(self):
        old = Mock()
        fresh = Mock()
        self.camera.publish(old, b'old', '/dev/video1', 'before', 15)
        waiting = threading.Event()
        result = []
        original_wait = self.camera.condition.wait

        def wait(timeout=None):
            waiting.set()
            return original_wait(timeout)

        with patch.object(self.camera.condition, 'wait', side_effect=wait):
            thread = threading.Thread(target=lambda: result.append(self.camera.snapshot()))
            thread.start()
            try:
                self.assertTrue(waiting.wait(1))
                self.camera.publish(fresh, b'new', '/dev/video1', 'after', 15)
            finally:
                thread.join(3)
        self.assertFalse(thread.is_alive())
        old.copy.assert_not_called()
        fresh.copy.assert_called_once()
        self.assertEqual(result, [(fresh.copy.return_value, '/dev/video1', 'after')])

    def test_viewers_independently_get_latest_frame(self):
        self.camera.publish(Mock(), b'old', '/dev/video1', 'before', 15)
        self.camera.publish(Mock(), b'new', '/dev/video1', 'after', 15)
        self.assertEqual(self.camera.preview(0), (2, b'new'))
        self.assertEqual(self.camera.preview(0), (2, b'new'))
        self.assertIsNone(self.camera.preview(2, timeout=.001))

    def test_failed_camera_does_not_reuse_stale_frame(self):
        self.camera.publish(Mock(), b'old', '/dev/video1', 'before', 15)
        self.camera.failed('Disconnected')
        with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
            self.camera.snapshot()
        self.assertIsNone(self.camera.preview(0, timeout=.001))
        self.assertEqual(self.camera.state()['status'], 'error')

    def test_snapshot_times_out_if_capture_stalls(self):
        self.camera.publish(Mock(), b'old', '/dev/video1', 'before', 15)
        with self.assertRaisesRegex(RuntimeError, 'No fresh'):
            self.camera.snapshot(timeout=.001)

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


@unittest.skipIf(cv2 is None, 'OpenCV and NumPy required (available on Modalix)')
class ImageTests(unittest.TestCase):
    def test_letterbox_preserves_entire_wide_view(self):
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        bgr[:, :640] = [255, 0, 0]
        bgr[:, 640:] = [0, 0, 255]
        square = app.model_image(bgr, cv2, np)
        self.assertEqual(square.shape, (480, 480, 3))
        self.assertEqual(square[0, 0].tolist(), [32, 32, 32])
        self.assertEqual(square[240, 0].tolist(), [255, 0, 0])
        self.assertEqual(square[240, -1].tolist(), [0, 0, 255])


if __name__ == '__main__':
    unittest.main()
