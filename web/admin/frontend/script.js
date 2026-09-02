// Camera object detection: captures webcam frames in-browser and posts them
// to the Python backend's /api/detect (YOLO), then draws the returned boxes
// on a canvas overlaid on the video.
// Chat: posts one message at a time to the Python backend's /api/chat
// (a from-scratch seq2seq model, see chats.py), which replies in kind.

// When this page is served by the FastAPI backend itself (python app.py),
// relative paths reach it directly. When served from Firebase Hosting (a
// different origin than the Render-hosted backend), point this at the
// backend's own URL instead — fill in after deploying to Render.
const BACKEND_URL = ''; // e.g. 'https://ai-module-backend.onrender.com'

const HEALTH_URL = `${BACKEND_URL}/api/health`;
const DETECT_URL = `${BACKEND_URL}/api/detect`;
const CHAT_URL = `${BACKEND_URL}/api/chat`;
const DETECT_INTERVAL_MS = 500;
const CAPTURE_WIDTH = 640; // frames are downscaled before upload to keep requests fast

const ICONS = {
  camera:
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8a2 2 0 0 1 2-2h1.5l1-1.5h7l1 1.5H18a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/><circle cx="12" cy="13" r="3.3"/></svg>',
  cameraOff:
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l18 18"/><path d="M9.5 4.5H15l1 1.5H18a2 2 0 0 1 2 2v9c0 .4-.09.77-.25 1.1"/><path d="M4 8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h11c.5 0 .95-.15 1.33-.4"/><circle cx="12" cy="13" r="3.3"/></svg>',
  chat:
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v11H8l-4 4Z"/></svg>',
  send:
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4Z"/></svg>',
};

function buildUI(root) {
  root.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark">${ICONS.camera}</div>
          <div class="brand-text">
            <h1>AI Modules</h1>
            <span>本機視覺與對話模型 Playground</span>
          </div>
        </div>
        <div class="status-pill" id="statusPill" data-state="checking" title="點一下重新檢查連線">
          <span class="status-dot"></span>
          <span id="statusText">連線中…</span>
        </div>
      </header>

      <div class="grid">
        <section class="card">
          <div class="card-head">
            ${ICONS.camera}
            <div>
              <h2>物件偵測</h2>
              <p>即時攝影機影像 + YOLO 偵測</p>
            </div>
          </div>
          <div class="card-body">
            <div id="stage">
              <video id="video" autoplay playsinline muted></video>
              <canvas id="overlay"></canvas>
              <div class="stage-empty" id="stageEmpty">
                ${ICONS.cameraOff}
                <span>相機尚未開啟</span>
              </div>
            </div>
            <div id="controls">
              <button id="toggleBtn" class="btn" type="button">開啟相機</button>
              <label class="slider-field">
                信心值
                <input id="confSlider" type="range" min="0.1" max="0.9" step="0.05" value="0.35">
                <span class="val" id="confValue">0.35</span>
              </label>
              <span id="latency"></span>
            </div>
            <div id="results"></div>
          </div>
        </section>

        <section class="card">
          <div class="card-head">
            ${ICONS.chat}
            <div>
              <h2>對話</h2>
              <p>自建 seq2seq 模型（無外部 AI API）</p>
            </div>
          </div>
          <div class="card-body">
            <div id="chatLog">
              <div class="chat-empty" id="chatEmpty">說點什麼開始對話吧</div>
            </div>
            <form id="chatForm">
              <input id="chatInput" type="text" placeholder="輸入訊息…" autocomplete="off">
              <button id="chatSend" class="btn" type="submit">${ICONS.send}</button>
            </form>
          </div>
        </section>
      </div>

      <footer>&copy; Author: Roy Zeng</footer>
    </div>
  `;
}

// 極簡 Markdown → HTML 轉換，只涵蓋模型回覆會用到的語法（標題、粗體、斜體、
// 行內程式碼、程式碼區塊、清單），不是完整 CommonMark 實作。目的是讓聊天欄顯示
// 排版後的「預覽」畫面，而不是原始的 "**粗體**"、"# 標題" 這些符號本身——跟
// 桌面 GUI（lib/components/markdown_view.py）、CLI（rich.markdown.Markdown）
// 是同一個訴求，這裡是網頁版的對應實作。escapeHtml() 先跑過一輪，避免回覆內容
// 剛好含 "<script>" 之類字串被當成真的 HTML 標籤解析、注入頁面。
function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function renderInlineMarkdown(line) {
  let html = escapeHtml(line);
  html = html.replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_, a, b) => `<strong>${a ?? b}</strong>`);
  html = html.replace(/`([^`]+)`/g, (_, code) => `<code class="md-code-inline">${code}</code>`);
  html = html.replace(
    /(?<!\*)\*([^*\n]+?)\*(?!\*)|(?<!_)_([^_\n]+?)_(?!_)/g,
    (_, a, b) => `<em>${a ?? b}</em>`
  );
  return html;
}

function markdownToHtml(text) {
  const lines = text.split('\n');
  let html = '';
  let listOpen = false;
  const closeList = () => {
    if (listOpen) {
      html += '</ul>';
      listOpen = false;
    }
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      closeList();
      i++;
      const codeLines = [];
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // 跳過收尾的 ```（若原文沒收尾，這時 i 已經等於 lines.length）
      html += `<pre class="md-code-block"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
      continue;
    }

    const header = line.match(/^(#{1,3})\s+(.*)/);
    if (header) {
      closeList();
      const level = header[1].length + 2; // 偏移到 h3~h5，避免蓋過頁面自己的 h1/h2
      html += `<h${level} class="md-heading">${renderInlineMarkdown(header[2])}</h${level}>`;
      i++;
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)/);
    if (bullet) {
      if (!listOpen) {
        html += '<ul class="md-list">';
        listOpen = true;
      }
      html += `<li>${renderInlineMarkdown(bullet[1])}</li>`;
      i++;
      continue;
    }

    closeList();
    if (line.trim() === '') {
      html += '<br>';
    } else {
      html += `<p class="md-p">${renderInlineMarkdown(line)}</p>`;
    }
    i++;
  }
  closeList();
  return html;
}

// fetch()'s res.json() throws a cryptic "Unexpected end of JSON input" (or
// "Unexpected token ... is not valid JSON") whenever the body isn't valid
// JSON — e.g. an empty 405/500 from the wrong server, or a plain-text crash
// page. Reading as text first lets us surface a message that actually says
// what went wrong instead of a raw JSON.parse SyntaxError.
async function safeJson(res) {
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (err) {
      throw new Error(
        `伺服器回傳了非預期的內容（HTTP ${res.status}），可能是後端沒有啟動，或這個頁面不是從後端伺服器開啟的。`
      );
    }
  }
  if (!res.ok) {
    const detail = data && (data.detail || data.error);
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${res.status}`);
  }
  if (data === null) {
    throw new Error(`伺服器沒有回傳任何內容（HTTP ${res.status}）。`);
  }
  return data;
}

async function checkHealth() {
  const pill = document.getElementById('statusPill');
  const text = document.getElementById('statusText');
  pill.dataset.state = 'checking';
  text.textContent = '連線中…';
  pill.title = '點一下重新檢查連線';
  try {
    await safeJson(await fetch(HEALTH_URL));
    pill.dataset.state = 'ok';
    text.textContent = '後端已連線';
  } catch (err) {
    pill.dataset.state = 'down';
    text.textContent = '無法連線後端';
    pill.title = `${err.message} 請確認已執行「python app.py」，並從 http://localhost:8000 開啟本頁（而非直接用檔案總管開啟 index.html）。點一下重試。`;
  }
}

function setupChat() {
  const chatLog = document.getElementById('chatLog');
  const chatEmpty = document.getElementById('chatEmpty');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const chatSend = document.getElementById('chatSend');

  function appendMessage(role, content) {
    chatEmpty.remove();
    const div = document.createElement('div');
    div.className = `chat-message chat-${role}`;
    // 只有 AI 回覆（assistant）需要 markdown 預覽渲染；使用者自己打的訊息維持
    // textContent 純文字顯示，不需要也不該被當成 markdown 解析。
    if (role === 'assistant') {
      div.innerHTML = markdownToHtml(content);
    } else {
      div.textContent = content;
    }
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
  }

  // The backend model is a from-scratch seq2seq net (see chats.py) that
  // answers one prompt at a time — no conversation-history conditioning yet.
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    chatSend.disabled = true;
    appendMessage('user', text);
    const pending = appendMessage('assistant', '…');
    pending.classList.add('is-pending');

    try {
      const res = await fetch(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await safeJson(res);
      pending.innerHTML = markdownToHtml(data.reply);
      pending.classList.remove('is-pending');
    } catch (err) {
      pending.textContent = `錯誤：${err.message}`;
    } finally {
      chatSend.disabled = false;
      chatInput.focus();
    }
  });
}

function main() {
  const root = document.getElementById('root');
  buildUI(root);
  setupChat();
  checkHealth();
  document.getElementById('statusPill').addEventListener('click', checkHealth);

  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const stageEmpty = document.getElementById('stageEmpty');
  const ctx = overlay.getContext('2d');
  const toggleBtn = document.getElementById('toggleBtn');
  const confSlider = document.getElementById('confSlider');
  const confValue = document.getElementById('confValue');
  const latencyEl = document.getElementById('latency');
  const resultsEl = document.getElementById('results');

  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  let stream = null;
  let loopHandle = null;
  let inFlight = false;

  confSlider.addEventListener('input', () => {
    confValue.textContent = confSlider.value;
  });

  toggleBtn.addEventListener('click', () => {
    if (stream) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      resultsEl.textContent = `相機錯誤：${err.message}`;
      return;
    }
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    stageEmpty.style.display = 'none';
    toggleBtn.textContent = '關閉相機';
    toggleBtn.classList.add('is-active');
    resultsEl.textContent = '偵測中…';
    loopHandle = setInterval(captureAndDetect, DETECT_INTERVAL_MS);
  }

  function stopCamera() {
    clearInterval(loopHandle);
    loopHandle = null;
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    stageEmpty.style.display = 'flex';
    toggleBtn.textContent = '開啟相機';
    toggleBtn.classList.remove('is-active');
    resultsEl.textContent = '';
    latencyEl.textContent = '';
  }

  async function captureAndDetect() {
    if (!video.videoWidth || inFlight) return;
    inFlight = true;
    try {
      const scale = CAPTURE_WIDTH / video.videoWidth;
      captureCanvas.width = CAPTURE_WIDTH;
      captureCanvas.height = Math.round(video.videoHeight * scale);
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      const blob = await new Promise((resolve) =>
        captureCanvas.toBlob(resolve, 'image/jpeg', 0.7)
      );
      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');

      const t0 = performance.now();
      const res = await fetch(`${DETECT_URL}?conf=${confSlider.value}`, {
        method: 'POST',
        body: form,
      });
      const data = await safeJson(res);
      latencyEl.textContent = `${Math.round(performance.now() - t0)} ms`;
      drawDetections(data);
    } catch (err) {
      resultsEl.textContent = `偵測錯誤：${err.message}`;
    } finally {
      inFlight = false;
    }
  }

  function drawDetections(data) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!data.width || !data.height) return;
    const sx = overlay.width / data.width;
    const sy = overlay.height / data.height;

    ctx.lineWidth = 2;
    ctx.font = '14px system-ui, sans-serif';
    ctx.textBaseline = 'top';

    const counts = {};
    for (const det of data.detections) {
      counts[det.label] = (counts[det.label] || 0) + 1;
      const [x1, y1, x2, y2] = det.box;
      const x = x1 * sx;
      const y = y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;

      ctx.strokeStyle = '#5b8cff';
      ctx.strokeRect(x, y, w, h);

      const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = '#5b8cff';
      ctx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);
      ctx.fillStyle = '#0b0d12';
      ctx.fillText(label, x + 4, Math.max(0, y - 18) + 2);
    }

    resultsEl.innerHTML = '';
    const labels = Object.entries(counts);
    if (labels.length) {
      for (const [label, n] of labels) {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.innerHTML = `<span class="dot"></span>${label} × ${n}`;
        resultsEl.appendChild(chip);
      }
    } else {
      resultsEl.textContent = '沒有偵測到物件';
    }
  }
}

document.addEventListener('DOMContentLoaded', main);
