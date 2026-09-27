'use strict';
const el = id => document.getElementById(id);
let state = null, imageId = null, displayedId = null, submitting = false;
let historyKey = '', requestError = '';
let streamRetry = null;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function renderHistory(history) {
  const key = JSON.stringify(history);
  if (key === historyKey) return;
  historyKey = key;
  el('history').replaceChildren(...history.map(turn => {
    const article = document.createElement('article');
    article.className = 'turn';
    for (const [className, text] of [
      ['speaker', `You · ${turn.vision ? 'with vision' : 'text only'}`], ['message', turn.prompt],
      ['speaker', 'Jarvic'], ['message', turn.answer]
    ]) {
      const p = document.createElement('p');
      p.className = className;
      p.textContent = text;
      article.append(p);
    }
    return article;
  }));
  el('history').scrollTop = el('history').scrollHeight;
}

function render(next) {
  state = next;
  const busy = submitting || ['loading', 'capturing', 'answering', 'clearing'].includes(state.status);
  const labels = {loading: 'Loading Gemma…', capturing: 'Capturing…', answering: 'Jarvic is answering…',
    clearing: 'Clearing…', ready: 'Ready', error: 'Needs attention'};
  el('status').textContent = labels[state.status] || state.status;
  el('status').className = busy ? 'busy' : '';
  el('ask').disabled = busy || !state.model_ready;
  el('clear').disabled = busy || !state.model_ready || (!state.history.length && !state.prompt);
  el('question').disabled = el('vision').disabled = busy;
  if (state.captured_at && imageId !== state.snapshot_id) {
    imageId = state.snapshot_id;
    displayedId = null;
    el('snapshot').src = `/api/snapshot.png?id=${imageId}`;
  }
  if (!state.captured_at) {
    imageId = displayedId = null;
    el('snapshot').removeAttribute('src');
  }
  const vision = el('vision').checked;
  el('snapshot').hidden = !state.captured_at || displayedId !== state.snapshot_id;
  el('question-image').hidden = !state.captured_at;
  el('captured').textContent = state.captured_at ?
    `Image sent to Jarvic · ${new Date(state.captured_at).toLocaleTimeString()}` : 'Image sent to Jarvic';
  const camera = state.camera;
  const live = camera?.status === 'streaming';
  el('video').hidden = !live || !el('video').naturalWidth;
  el('placeholder').hidden = !el('video').hidden;
  el('placeholder').textContent = camera?.error || 'Connecting to the USB camera…';
  el('camera-status').textContent = live ? `LIVE · ${camera.fps.toFixed(1)} FPS` : 'RECONNECTING';
  el('camera-hint').textContent = vision ? 'The video stays live. Ask Jarvic sends one fresh frame with your question.' :
    'The video stays live. Vision is off, so Jarvic receives only text and conversation history.';
  renderHistory(state.history);
  const last = state.history.at(-1);
  el('current-turn').hidden = !state.prompt || !!(state.result && last?.prompt === state.prompt && last?.answer === state.answer);
  el('current-label').textContent = `You · ${state.vision ? 'with vision' : 'text only'}`;
  el('current-prompt').textContent = state.prompt;
  el('answer').textContent = state.answer || (state.status === 'capturing' ? 'Capturing a fresh image…' :
    state.status === 'answering' ? 'Thinking…' : 'No answer completed. Try asking again.');
  el('empty-chat').hidden = !!(state.history.length || state.prompt);
  el('error').textContent = requestError || state.error || '';
  el('error').hidden = !(requestError || state.error);
  el('timing').textContent = state.result ? `${state.result.generation_s.toFixed(2)} s generation · ${state.result.request_s.toFixed(2)} s total${state.result.finish_reason === 'length' ? ' · token limit reached' : ''}` : '';
  el('saved').hidden = !state.result;
  el('saved').textContent = state.result?.vision ? 'Snapshot and answer saved on Modalix.' : 'Answer saved on Modalix.';
}

el('snapshot').onload = () => {
  displayedId = Number(new URL(el('snapshot').src).searchParams.get('id'));
  if (state) render(state);
};
el('snapshot').onerror = () => { displayedId = imageId = null; };
el('video').onload = () => { if (state) render(state); };
el('video').onerror = () => {
  if (streamRetry) return;
  el('video').hidden = true;
  streamRetry = setTimeout(() => {
    streamRetry = null;
    el('video').src = `/api/video.mjpg?retry=${Date.now()}`;
  }, 1500);
};
el('vision').onchange = () => { if (state) render(state); };

async function action(path, body) {
  submitting = true;
  requestError = '';
  if (state) render(state);
  try { render(await api(path, body)); }
  catch (error) { requestError = error.message; }
  finally {
    submitting = false;
    if (state) render(state);
  }
}
el('clear').onclick = () => action('/api/clear', {});
el('question-form').onsubmit = event => {
  event.preventDefault();
  const prompt = el('question').value.trim();
  if (prompt && !el('ask').disabled) action('/api/ask', {prompt, vision: el('vision').checked});
};
async function poll() {
  try { render(await api('/api/state')); }
  catch (error) {
    el('status').textContent = 'Connection lost · retrying…';
    el('ask').disabled = el('clear').disabled = el('vision').disabled = true;
  }
  setTimeout(poll, 350);
}
poll();
