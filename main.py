#!/usr/bin/env python3
"""Ask Gemma about a fresh snapshot from a USB camera attached to Modalix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time


DEFAULT_MODEL = Path('/workspace/llima/models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy')
DEFAULT_PROMPT = 'Describe what you see in this image in two short sentences.'
HISTORY_TURNS = 3
HISTORY_BYTES = 4000
SYSTEM_PROMPT = (
    'You are Jarvic, a helpful, concise assistant. Previous exchanges contain text only. '
    'If the current question includes an image, use that fresh image for visual details. '
    'Otherwise answer from the text and conversation; do not claim to see a new camera image.'
)


def validate_prompt(prompt, max_tokens):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError('Enter a question between 1 and 2000 characters.')
    # Conservative byte budget for the 8k context; reserve image/template tokens
    # and the requested answer. UTF-8 bytes bound byte-fallback text tokens.
    if len(prompt.encode('utf-8')) > 8192 - 1536 - max_tokens:
        raise ValueError('This question is too long for the model context. Please shorten it.')
    return prompt.strip()


def short_history(turns, byte_budget=HISTORY_BYTES):
    kept = list(turns[-HISTORY_TURNS:])
    while kept and sum(len((t['prompt'] + t['answer']).encode('utf-8')) for t in kept) > byte_budget:
        kept.pop(0)
    return kept


def chat_request(neat, args, prompt, history, image=None):
    prompt = validate_prompt(prompt, args.max_tokens)
    budget = min(HISTORY_BYTES, 8192 - 1536 - args.max_tokens - len(prompt.encode('utf-8')))
    context = short_history(history, budget)

    def message(role, content):
        item = neat.genai.ChatMessage()
        item.role, item.content = role, content
        return item

    messages = [message('system', SYSTEM_PROMPT)]
    for turn in context:
        messages.extend([message('user', turn['prompt']), message('assistant', turn['answer'])])
    current = message('user', prompt)
    if image is not None:
        current.images = [image]
    messages.append(current)
    request = neat.genai.GenerationRequest()
    request.messages = messages
    request.max_new_tokens = args.max_tokens
    request.enable_thinking = False
    return request, context


def camera_paths():
    return sorted(Path('/dev/v4l/by-id').glob('*-video-index0'))


def resolve_camera(value):
    if value == 'auto':
        cameras = camera_paths()
        if len(cameras) != 1:
            raise ValueError(f'Found {len(cameras)} cameras. Use --list-cameras and --camera PATH.')
        path = cameras[0]
    else:
        path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f'Camera does not exist: {path}')
    if not path.is_char_device():
        raise ValueError(f'Expected a V4L2 camera device: {path}')
    return str(path.resolve())


def validate_model(path):
    config_path = path / 'devkit/vlm_config.json'
    if not config_path.is_file():
        raise ValueError(f'Missing deployed VLM configuration: {config_path}')
    config = json.loads(config_path.read_text())
    if config.get('model_type') != 'vlm-gemma4':
        raise ValueError('This application expects a deployed Gemma 4 VLM.')
    for key in ('vision_model_name', 'language_model_name'):
        name = config.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f'Model configuration is missing {key}')
        artifacts = list((path / 'elf_files').glob(name + '*.elf'))
        if not artifacts or any(p.stat().st_size == 0 for p in artifacts):
            raise ValueError(f'Missing or empty {key} ELF artifacts in {path / "elf_files"}')
    if config.get('vm_cfg', {}).get('image_size') != [480, 480]:
        raise ValueError('This application expects the supplied 480 x 480 Gemma vision model.')


def open_camera(args, cv2):
    """Open the board's USB camera in the requested MJPEG mode."""
    device = resolve_camera(args.camera)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {device}; check permissions and other camera applications.')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap, device
    except BaseException:
        cap.release()
        raise


def capture(args, cv2):
    """Terminal mode: warm up a camera, copy a frame, and close it."""
    cap, device = open_camera(args, cv2)
    try:
        for _ in range(args.warmup_frames):
            ok, bgr = cap.read()
            if not ok or bgr is None or not bgr.size:
                raise RuntimeError(f'No image from {device}; reconnect the USB camera and retry.')
        if bgr.shape[:2] != (args.height, args.width):
            raise RuntimeError(f'Camera returned {bgr.shape[1]} x {bgr.shape[0]}; '
                               f'requested {args.width} x {args.height}. Choose a supported mode.')
        return bgr, device, datetime.now(timezone.utc).isoformat()
    finally:
        cap.release()


def model_image(bgr, cv2, np):
    """Fit the complete view inside 480 x 480 without stretching or cropping."""
    height, width = bgr.shape[:2]
    scale = min(480 / width, 480 / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    square = np.full((480, 480, 3), 32, dtype=np.uint8)
    x, y = (480 - w) // 2, (480 - h) // 2
    square[y:y + h, x:x + w] = resized
    return square


def generate(model, request, timeout, emit, stopping=None):
    """Use public stream cancellation for a bounded, single-turn request."""
    started = time.monotonic()
    stream = model.stream(request)
    expired = threading.Event()

    finished = threading.Event()

    def watch():
        while not finished.wait(.02):
            if stopping is not None and stopping.is_set():
                stream.cancel()
                return
            if time.monotonic() - started >= timeout:
                expired.set()
                stream.cancel()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    answer, final = [], None
    try:
        watcher.start()
        for token in stream:
            if stopping is not None and stopping.is_set():
                raise InterruptedError('Generation stopped.')
            if expired.is_set():
                raise TimeoutError(f'Gemma exceeded {timeout:g} seconds.')
            if token.text:
                answer.append(token.text)
                emit(token.text)
            if token.is_final:
                final = token
        if stopping is not None and stopping.is_set():
            raise InterruptedError('Generation stopped.')
        if expired.is_set() or time.monotonic() - started >= timeout:
            raise TimeoutError(f'Gemma exceeded {timeout:g} seconds.')
        text = ''.join(answer).strip()
        if not text:
            raise RuntimeError('Gemma returned an empty answer.')
        if final is None or final.finish_reason not in ('stop', 'length'):
            raise RuntimeError(f'Generation did not complete: {getattr(final, "finish_reason", "no final token")}')
        metrics = final.metrics
        return dict(answer=text, finish_reason=final.finish_reason,
                    generation_s=round(time.monotonic() - started, 3),
                    generated_tokens=metrics.generated_tokens,
                    time_to_first_token_s=metrics.time_to_first_token_s,
                    tokens_per_second=metrics.tokens_per_second)
    finally:
        finished.set()
        watcher.join()
        stream.cancel()


def ask(model, neat, args, prompt, cv2, np, *, vision=True, history=(),
        snapshot=None, emit=None, stopping=None, started=None):
    started = time.monotonic() if started is None else started
    prompt = validate_prompt(prompt, args.max_tokens)
    if vision and snapshot is None:
        print('\nCapturing a fresh USB snapshot...', flush=True)
        snapshot = capture(args, cv2)
    folder = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True)
    image = None
    report = dict(model=str(args.model), model_id=model.model_id(), prompt=prompt,
                  vision=vision, camera=None, captured_at=None, image=None, camera_image=None)
    if vision:
        bgr, device, captured_at = snapshot
        square = model_image(bgr, cv2, np)
        for filename, pixels in (('camera.jpg', bgr), ('model-input.png', square)):
            if not cv2.imwrite(str(folder / filename), pixels):
                raise RuntimeError(f'Could not save {folder / filename}')
        # PNG contains the exact image pixels sent to the current user message.
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)
        report.update(camera=device, captured_at=captured_at, image='model-input.png',
                      camera_image='camera.jpg', width=args.width, height=args.height)
    request, context = chat_request(neat, args, prompt, history, image)
    report['history'] = context
    print(f'Vision: {vision}; history: {len(context)} exchanges\nQuestion: {prompt}\nJarvic: ', end='', flush=True)
    try:
        report.update(generate(model, request, args.timeout,
                               emit or (lambda text: print(text, end='', flush=True)), stopping))
        report['status'] = 'ok'
    except BaseException as exc:
        report.update(status='error', error=str(exc) or type(exc).__name__)
        raise
    finally:
        report['request_s'] = round(time.monotonic() - started, 3)
        (folder / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'\nSaved: {folder / "result.json"}', flush=True)
    print(f'Generation: {report["generation_s"]:.2f}s; '
          f'request: {report["request_s"]:.2f}s', flush=True)
    if report['finish_reason'] == 'length':
        print('Answer reached --max-tokens; increase it for a longer answer.', flush=True)
    return dict(report, result_path=str(folder / 'result.json'))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--camera', default='auto', help='Stable /dev/v4l/by-id path, /dev/videoN, or auto')
    parser.add_argument('--list-cameras', action='store_true', help='List stable camera identities and exit')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Question to ask about the snapshot')
    parser.add_argument('--interactive', action='store_true', help='Keep Gemma loaded; capture anew for each question')
    parser.add_argument('--web', action='store_true', help='Serve the snapshot/question browser interface')
    parser.add_argument('--bind', default='0.0.0.0', help='Browser server bind address')
    parser.add_argument('--port', type=int, default=8023, help='Browser server port')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--warmup-frames', type=int, default=15, help='Frames to read for exposure settling')
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--timeout', type=float, default=60, help='Generation deadline in seconds; cooperative cancellation')
    parser.add_argument('--output-dir', type=Path, default=Path.home() / '.cache/usb-camera-vlm/captures')
    args = parser.parse_args(argv)
    if args.web and args.interactive:
        parser.error('--web and --interactive are separate modes')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    for name in ('width', 'height', 'warmup_frames', 'max_tokens'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.max_tokens > 2048:
        parser.error('--max-tokens must be at most 2048')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if not args.prompt.strip():
        parser.error('--prompt must contain a question')
    args.prompt = args.prompt.strip()
    args.model = args.model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = arguments(argv)
    if args.list_cameras:
        for path in camera_paths():
            print(f'{path} -> {path.resolve()}')
        return 0
    validate_model(args.model)
    def interrupt(*_):
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    if args.web:
        from webapp import serve
        serve(args)
        return 0
    resolve_camera(args.camera)
    import cv2
    import numpy as np
    import pyneat as neat

    cv2.setNumThreads(1)
    print(f'Loading Gemma from {args.model} ...', flush=True)
    started = time.monotonic()
    model = neat.genai.GenAIModel(str(args.model))
    try:
        if not model.accepts_image():
            raise ValueError('The selected model does not accept images.')
        print(f'Gemma ready in {time.monotonic() - started:.2f}s.', flush=True)
        if not args.interactive:
            ask(model, neat, args, args.prompt, cv2, np)
        else:
            print('Type a question, press Enter for the default description, or type /quit.')
            while True:
                try:
                    prompt = input('\nQuestion> ').strip()
                except EOFError:
                    break
                if prompt.lower() in ('/quit', '/exit'):
                    break
                ask(model, neat, args, prompt or args.prompt, cv2, np)
    finally:
        # GenAIModel has no close() method; dropping the handle invokes its destructor.
        del model
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'USB camera VLM error: {exc}', file=sys.stderr)
        raise SystemExit(1)
