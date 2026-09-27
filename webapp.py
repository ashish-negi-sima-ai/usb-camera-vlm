"""Browser UI with continuous USB capture and a separate Gemma worker."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing
from pathlib import Path
import queue
import threading
import time
from urllib.parse import parse_qs, urlsplit

import main as app
from camera import LiveCamera


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        context = multiprocessing.get_context('spawn')
        self.stopping = context.Event()
        self.camera = LiveCamera(args, self.stopping)
        self.jobs = context.Queue(maxsize=1)
        # Synchronous delivery also works when subsequent native calls hold the GIL.
        self.responses, self.sender = context.Pipe(duplex=False)
        self.png = None
        self.state = dict(status='loading', model_ready=False, error=None,
                          snapshot_id=0, captured_at=None, prompt='', vision=True,
                          answer='', result=None, load_s=None, history=[],
                          history_limit=app.HISTORY_TURNS)
        self.worker = context.Process(target=model_worker, args=(args, self.jobs, self.sender, self.stopping),
                                      name='camera-gemma', daemon=True)
        self.receiver = threading.Thread(target=self.receive, daemon=True)

    def start(self):
        self.camera.start()
        self.worker.start()
        self.sender.close()
        self.receiver.start()

    def receive(self):
        while not self.stopping.is_set():
            try:
                if not self.responses.poll(.2):
                    continue
                message = self.responses.recv()
            except EOFError:
                if not self.worker.is_alive():
                    with self.lock:
                        self.state.update(status='error', model_ready=False,
                                          error=self.state['error'] or
                                          f'Gemma worker exited ({self.worker.exitcode}). Restart the app.')
                    return
                continue
            with self.lock:
                if 'png' in message:
                    self.png = message.pop('png')
                self.state.update(message)

    def state_snapshot(self):
        with self.lock:
            return dict(self.state, camera=self.camera.state())

    def submit(self, action, body):
        with self.lock:
            if not self.state['model_ready'] or self.state['status'] in ('capturing', 'answering', 'clearing'):
                raise RuntimeError('Wait for the current operation to finish.')
            if action == 'ask':
                prompt = app.validate_prompt(body.get('prompt'), self.args.max_tokens)
                vision = body.get('vision')
                if not isinstance(vision, bool):
                    raise ValueError('vision must be true or false.')
                started = time.monotonic()
                snapshot = self.camera.snapshot() if vision else None
                self.jobs.put_nowait(('ask', dict(prompt=prompt, vision=vision,
                                                snapshot=snapshot, started=started)))
                self.png = None
                self.state.update(status='capturing' if vision else 'answering',
                                  prompt=prompt, vision=vision, answer='', result=None,
                                  captured_at=None, error=None)
            elif action == 'clear':
                self.jobs.put_nowait(('clear', None))
                self.state.update(status='clearing', error=None)
            else:
                raise ValueError('Unknown operation.')

    def close(self):
        self.stopping.set()
        self.camera.close()
        self.worker.join(30)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(3)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(2)
        self.receiver.join(1)
        self.jobs.cancel_join_thread()
        self.jobs.close()
        self.responses.close()
        self.worker.close()


class ModelWorker:
    def __init__(self, args, jobs, responses, stopping, neat, cv2, np):
        self.args, self.jobs, self.responses, self.stopping = args, jobs, responses, stopping
        self.neat, self.cv2, self.np = neat, cv2, np
        self.snapshot = None
        self.snapshot_id = 0
        self.answer = ''
        self.history = []

    def update(self, **fields):
        if not self.stopping.is_set():
            self.responses.send(fields)

    def capture(self, snapshot):
        if snapshot is None:
            raise RuntimeError('A fresh video frame is required for a vision question.')
        square = app.model_image(snapshot[0], self.cv2, self.np)
        ok, encoded = self.cv2.imencode('.png', square)
        if not ok:
            raise RuntimeError('Could not encode the camera snapshot.')
        self.snapshot = snapshot
        self.snapshot_id += 1
        self.update(status='answering', snapshot_id=self.snapshot_id, png=encoded.tobytes(),
                    captured_at=snapshot[2], answer='', result=None, error=None)

    def append(self, text):
        self.answer += text
        self.update(answer=self.answer)

    def handle(self, model, action, payload):
        if action == 'clear':
            model.clear_kv_caches()
            self.history = []
            self.snapshot = None
            self.update(status='ready', history=[], prompt='', answer='', result=None,
                        captured_at=None, png=None, error=None)
            return
        started = payload.get('started', time.monotonic())
        self.snapshot = None
        self.answer = ''
        try:
            if payload['vision']:
                # The web process freezes a fresh frame while preview keeps running.
                self.capture(payload.get('snapshot'))
            report = app.ask(model, self.neat, self.args, payload['prompt'], self.cv2, self.np,
                             vision=payload['vision'], history=self.history, snapshot=self.snapshot,
                             emit=self.append, stopping=self.stopping, started=started)
            turn = dict(prompt=payload['prompt'], answer=report['answer'], vision=payload['vision'])
            self.history = app.short_history([*self.history, turn])
            self.update(status='ready', answer=report['answer'], result=report, history=self.history)
        finally:
            self.snapshot = None

    def run(self):
        model = None
        try:
            started = time.monotonic()
            print(f'Loading Gemma from {self.args.model} ...', flush=True)
            model = self.neat.genai.GenAIModel(str(self.args.model))
            if not model.accepts_image():
                raise ValueError('The selected model does not accept images.')
            elapsed = round(time.monotonic() - started, 2)
            self.update(model_ready=True, status='ready', load_s=elapsed, error=None)
            print(f'Gemma ready in {elapsed}s.', flush=True)
            while not self.stopping.is_set():
                try:
                    action, payload = self.jobs.get(timeout=.2)
                except queue.Empty:
                    continue
                try:
                    self.handle(model, action, payload)
                except Exception as exc:
                    self.update(status='error', error=str(exc))
        except Exception as exc:
            self.update(status='error', model_ready=False, error=str(exc))
            print(f'Gemma startup failed: {exc}', flush=True)
        finally:
            del model


def model_worker(args, jobs, responses, stopping):
    # Native model loading can hold the GIL. A separate process keeps HTTP responsive.
    import cv2
    import numpy as np
    import pyneat as neat
    cv2.setNumThreads(1)
    ModelWorker(args, jobs, responses, stopping, neat, cv2, np).run()


def serve(args):
    application = Application(args)
    static = Path(__file__).with_name('web')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, content_type='application/json'):
            if isinstance(payload, dict):
                payload = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path in assets:
                name, mime = assets[path]
                self.send(200, (static / name).read_bytes(), mime)
            elif path == '/api/state':
                self.send(200, application.state_snapshot())
            elif path == '/api/video.mjpg':
                self.stream_video()
            elif path == '/api/snapshot.png':
                with application.lock:
                    png = application.png
                    requested = parse_qs(urlsplit(self.path).query).get('id', [''])[0]
                    if requested and requested != str(application.state['snapshot_id']):
                        png = None
                if png:
                    self.send(200, png, 'image/png')
                else:
                    self.send(404, {'error': 'No image for the current question.'})
            else:
                self.send(404, {'error': 'Not found'})

        def stream_video(self):
            self.connection.settimeout(5)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            sequence = -1
            try:
                while not application.stopping.is_set():
                    preview = application.camera.preview(sequence)
                    if preview is None:
                        continue
                    sequence, jpeg = preview
                    header = (f'--frame\r\nContent-Type: image/jpeg\r\n'
                              f'Content-Length: {len(jpeg)}\r\nX-Frame-Id: {sequence}\r\n\r\n').encode()
                    self.wfile.write(header + jpeg + b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ('/api/clear', '/api/ask'):
                self.send(404, {'error': 'Not found'})
                return
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.send(403, {'error': 'Use the application on this server.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request size.')
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected application/json.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object.')
                application.submit('ask' if path == '/api/ask' else 'clear', body)
                self.send(202, application.state_snapshot())
            except (ValueError, TypeError) as exc:
                self.send(400, {'error': str(exc)})
            except (RuntimeError, queue.Full) as exc:
                self.send(409, {'error': str(exc)})

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    application.start()
    print(f'USB camera + Gemma browser: http://{args.bind}:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        application.stopping.set()
        server.server_close()
        application.close()
