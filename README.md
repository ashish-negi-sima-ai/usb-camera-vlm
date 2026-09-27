# Jarvic: USB camera + Gemma on Modalix

A standalone browser application: watch live USB video and chat with Gemma,
optionally including a fresh frame with your question. It uses OpenCV/V4L2 capture and the installed
public Neat `GenAIModel` API. All inference runs on the board.

The supplied source checkpoint (and default `--model` argument) is:

```text
/workspace/llima/models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy
```

On this board, `/workspace` is an NFS mount. The working deployment is a copy of
that checkpoint on local NVMe:

```text
/media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy
```

Use the NVMe copy in the launch command below. The updated LLiMa runtime can
offload embedding tables from verified local NVMe storage. Loading the source
directly over NFS keeps those tables in DRAM and still fails to allocate memory
for this model's vision component on this board.

## Run

On the Modalix board:

```bash
ssh sima@10.42.0.147
cd /workspace/GitHub/usb-camera-vlm
./run.sh --web \
  --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy
```

Open **http://10.42.0.147:8023** in your browser. Type a question and click
**Ask Jarvic**. Gemma stays loaded, and the answer streams into the conversation.
The USB video stays live during model loading, inference, and text-only chat.

- **Enable vision checked:** each question copies a fresh frame from the live
  feed, then sends that image, the question, and recent conversation text to Gemma.
  The preview continues moving. Expand **Image sent to Jarvic** to inspect the
  exact frozen model input. There is no separate capture action.
- **Enable vision unchecked:** only the question and recent conversation text
  are sent. Video keeps playing, but no image is sent to Gemma.
- **Short history:** retains up to three completed question/answer exchanges.
  Older complete exchanges are dropped to fit the context budget. Failed requests
  do not enter history. You can switch vision on and off within one conversation.
- **Clear conversation:** removes the in-memory history, current snapshot, and
  reusable model cache. It does not delete previously saved results.

The app opens the camera at startup and retries if capture fails. Text-only chat
also works without a USB camera connected; the preview reports the camera error.
One question runs at a time. Conversation state is shared by
all browser tabs using this app, survives a page refresh, and resets when the app
restarts. The browser uses the board's USB camera.

The server binds to `0.0.0.0:8023` by default. Use `--bind` or `--port` to change
that. This is a local-network HTTP demo without authentication. Ctrl+C stops the
server, continuous camera capture, and its model worker.

Terminal modes are also available:

```bash
./run.sh --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy \
  --prompt "Describe what you see in this image."
./run.sh --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy \
  --interactive
```

Each terminal question captures a new image. Enter a blank line for the default question;
use `/quit`, Ctrl+D, or Ctrl+C to exit. Questions are independent, without chat
history. A capture or inference error ends the application with a nonzero code;
correct the error and rerun. The browser reports operation errors and allows
another question; a model startup failure requires restarting the app.

The launcher uses `~/pyneat/bin/python`. Set `PYTHON` or `PYNEAT_ENV` if needed.
Dependencies are the board's installed Neat/LLiMa runtime, OpenCV, and NumPy.
No model download, compilation, or pip installation is required on the target.
Run this app while the camera and accelerator are available to it.

## Camera and options

```bash
./run.sh --list-cameras
./run.sh --web --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy \
  --camera /dev/v4l/by-id/usb-YOUR_CAMERA-video-index0
./run.sh --web --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy \
  --width 640 --height 480 --max-tokens 96
./run.sh --web --model /media/nvme/usb-camera-vlm-models/Gemma-4-E4B-it-GPTQ-a16w4-8k-deploy \
  --output-dir /home/sima/usb-vlm-captures
./run.sh --help
```

`auto` requires exactly one `/dev/v4l/by-id/*-video-index0` identity. A stable
identity is preferable to `/dev/videoN`, whose number can change. The eMeet Nova
on this board supports the default MJPEG 1280×720 mode. Capture requests 30 FPS
and reads 15 frames for exposure settling after each camera open.
`--warmup-frames` changes that count. Browser mode keeps this one capture device
open and drains frames continuously. Preview JPEGs are at most 960 pixels wide,
published at up to 15 FPS, and shared by all viewers; slow viewers skip old
frames. The page reports the measured publication FPS. Terminal mode still
opens and closes the camera for each question.

A vision question waits up to two seconds for a new frame after Ask, copies the
original 1280×720 BGR frame, and sends the copy to the model process. Camera
failure or a stalled stream cannot silently reuse an old frame. Preview frames
are held in memory, not recorded to disk.

The complete snapshot is fitted inside a 480×480 image with padding, converted
from BGR to RGB, and supplied as a Neat image tensor. Browser requests use native
`GenerationRequest.messages` with alternating user/assistant roles. Only the
current user message receives an image when vision is enabled; old turns retain
text only. Thinking is disabled and answer tokens stream to the UI or terminal.
The model remains loaded in browser and interactive modes.
Inference operates on individual frozen frames; the live preview is a separate
MJPEG HTTP stream. The app has no dependency on the adjacent demo repo.

History retains at most 4,000 UTF-8 bytes across three complete exchanges, and
the request may include fewer exchanges for a long question. The 8k context
budget reserves space for image/template tokens and `--max-tokens`; an oversized
question is rejected before inference. Questions are limited to 2,000 characters.

`--timeout` defaults to 60 seconds and uses Neat's cooperative stream cancellation.
It covers generation, not model loading or a blocking camera-driver call.
SIGINT, SIGTERM, and SIGHUP trigger camera/stream cleanup and model release once
native calls return control. The browser runs Neat in a spawned worker process
because native model loading can hold Python's interpreter lock. The HTTP server
and camera thread run in the parent process, keeping video responsive during
model loading and inference. Shutdown waits thirty seconds for the worker before terminating it
if a native call is still blocked. The generation token limit defaults to 160.

## Saved results

Each question creates a timestamped directory under
`~/.cache/usb-camera-vlm/captures/` containing:

- `camera.jpg` (vision only): original camera snapshot.
- `model-input.png` (vision only): the exact 480×480 image pixels passed to Gemma, saved losslessly.
- `result.json`: prompt, answer or generation error, model and camera identity,
  vision flag, text history actually used, UTC capture time when applicable,
  finish reason, and timing measurements. Text-only requests create no images.

The console prints the result directory. Generation time is measured separately
from total request time (including capture when enabled); model-loading time is printed at startup. Saved
images and results remain on disk until you remove them.

## Implementation

`main.py` contains device discovery, capture, image preparation, streaming
generation, and the terminal interface. `camera.py` owns continuous capture and
shares the latest JPEG with all viewers. `webapp.py` owns the HTTP server and
model worker; `web/` contains the browser interface. `run.sh` selects the board's Python.
The API shape follows Neat's VLM/streaming tutorials and the board-camera flow
in the Neat GenAI Studio example, checked against the target's installed public
headers and bindings.

HTTP endpoints:

- `GET /api/state`: readiness, current answer, snapshot identity, recent history,
  camera status, frame sequence and measured preview FPS.
- `GET /api/video.mjpg`: continuous multipart JPEG video; each part includes
  `Content-Length` and `X-Frame-Id` headers.
- `POST /api/ask`: JSON `{"prompt": "What is here?", "vision": true}` (or `false`).
- `POST /api/clear`: JSON `{}` to start a fresh conversation.
- `GET /api/snapshot.png?id=N`: current vision question's image; unavailable for
  a text-only question. `/api/capture` has been removed.

```bash
~/pyneat/bin/python -m unittest discover -s . -p test_main.py -v
node --check web/app.js  # On a development host with Node installed.
```

## Target validation — 2026-09-27

On `sima@10.42.0.147`, all 27 unit/process tests passed. The eMeet Nova captured
1280×720 BGR frames successfully, and the 480×480 padded input was inspected.
The browser loaded on desktop and at 390-pixel mobile width without JavaScript
errors or horizontal overflow. Missing model/camera paths and invalid requests
returned errors as expected.

**Real VLM inference passed after the library update and NVMe deployment.**
The source checkpoint was copied to the mounted `nvme0n1p1` filesystem; all 235
files were checked for matching size and the deployment configuration matched.
The application used the default bulk loader with automatic embedding offload.

- The live feed displayed **960×540 video while Gemma was still loading**.
  During three subsequent inference requests it published roughly **12–13 FPS**.
  A second MJPEG viewer received **62, 42, and 35 frames** across those requests,
  and browser pixel comparisons confirmed that the displayed video kept moving.
- Two fresh vision requests and a text-only history follow-up passed while video
  streamed continuously. Vision generation took **3.945 / 2.357 seconds**;
  full request times were **4.243 / 2.574 seconds**. Text-only generation took
  **2.834 seconds**, with no image supplied. Preview stayed visible with vision
  unchecked, after clearing the conversation, and after reloading the page.
- Before adding the live feed, six conversation requests passed: text-only chat, recall of a previously supplied
  code, a vision question that also recalled that code, a second fresh vision
  capture, a text-only follow-up using previous scene descriptions, and a question
  after clearing history. The cleared conversation no longer recalled the code.
- The history stayed bounded to three exchanges and survived page refresh.
  Clear conversation, strict boolean validation, removal of the capture endpoint,
  desktop/mobile layout, and JavaScript error checks passed.
- Saved text-only requests contained only `result.json`; vision requests also
  contained the camera JPEG and exact model-input PNG. Unit tests verified that
  text inference requests make no snapshot/image calls, fresh snapshots wait for
  a new frame after Ask, stale frames are rejected, and failed requests preserve history.

Installed packages during this successful run were Neat internals
`0.4.0+develop.deeda30ef75b`, Neat public API build `721fab189e2e`, and LLiMa
runtime build `a23266247d30` (the latter two from the reusable-prompt-contexts
feature branch).

The earlier failure remains reproducible when loading directly from the NFS
source: `..._vision_stage1_mla.elf` fails with `MLA_LOAD_FAILED` and DMS0
512 MiB / 256 MiB allocation errors. The local NVMe placement enables the
runtime's embedding offload and resolves that failure for this application.

The browser launched during validation uses port 8023. Its log and recorded PID
are at `~/.cache/usb-camera-vlm/web.log` and `web.pid`. To stop that instance,
check the recorded PID's command before sending SIGTERM:

```bash
app_pid=$(cat ~/.cache/usb-camera-vlm/web.pid)
ps -p "$app_pid" -o pid,args
# If this is usb-camera-vlm/main.py:
kill -TERM "$app_pid"
```
