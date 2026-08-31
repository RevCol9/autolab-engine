"""本地联调：浏览器上传图片、选 model_key、查看推理 JSON。"""

from fastapi.responses import HTMLResponse

TEST_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>autolab-engine 推理测试</title>
  <style>
    :root { font-family: system-ui, sans-serif; color: #1a1a1a; background: #f5f5f7; }
    body { max-width: 1100px; margin: 0 auto; padding: 1.25rem; }
    h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
    .muted { color: #666; font-size: 0.875rem; margin-bottom: 1rem; }
    .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
    .panel { background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    .panel form { display: grid; gap: 0.75rem; min-width: 280px; }
    label { display: grid; gap: 0.25rem; font-size: 0.875rem; font-weight: 500; }
    input, select, button { font: inherit; padding: 0.4rem 0.5rem; }
    button { cursor: pointer; background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 0.55rem 1rem; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.secondary { background: #64748b; }
    #preview-wrap { position: relative; display: block; max-width: 100%; line-height: 0; }
    #preview-canvas { max-width: 100%; height: auto; display: block; border-radius: 4px;
      background: #e2e8f0; min-height: 120px; }
    #preview-canvas.empty::after { content: '上传图片并运行推理'; }
    img#preview-src { display: none; }
    pre { margin: 0; padding: 0.75rem; background: #0f172a; color: #e2e8f0; border-radius: 6px;
         font-size: 12px; overflow: auto; max-height: 480px; white-space: pre-wrap; word-break: break-all; }
    .err { color: #b91c1c; font-size: 0.875rem; }
    .meta { font-size: 0.8rem; color: #475569; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <h1>autolab-engine 推理测试</h1>
  <p class="muted">选 config.yaml 中的 model_key，上传图片，调用 <code>POST /api/predict</code> 并展示返回 JSON。正式对接见 <code>/docs</code>。</p>

  <div class="row">
    <div class="panel">
      <form id="form">
        <label>model_key
          <select id="model_key" name="model_key" required></select>
        </label>
        <label>box_format
          <select id="box_format" name="box_format">
            <option value="xyxy">xyxy（像素框）</option>
            <option value="cxcywh_pct" selected>cxcywh_pct（平台百分比）</option>
          </select>
        </label>
        <label>conf（可选）
          <input type="number" id="conf" name="conf" step="0.01" min="0" max="1" placeholder="默认用 config" />
        </label>
        <label>图片
          <input type="file" id="image" name="image" accept="image/*" required />
        </label>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
          <button type="submit" id="btn">运行推理</button>
          <button type="button" class="secondary" id="btn_load">仅加载模型</button>
        </div>
        <p id="status" class="meta"></p>
        <p id="error" class="err"></p>
      </form>
    </div>

    <div class="panel" style="flex:1; min-width: 320px;">
      <p class="meta">预览（绿框为 bbox，标签为 class + score）</p>
      <div id="preview-wrap">
        <canvas id="preview-canvas" class="empty"></canvas>
        <img id="preview-src" alt="" />
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:1rem;">
    <p class="meta">响应 JSON</p>
    <pre id="out">（尚未请求）</pre>
  </div>

<script>
const sel = document.getElementById('model_key');
const out = document.getElementById('out');
const errEl = document.getElementById('error');
const statusEl = document.getElementById('status');
const imgEl = document.getElementById('preview-src');
const canvas = document.getElementById('preview-canvas');
const ctx = canvas.getContext('2d');
let lastPredict = null;

const COLORS = ['#22c55e', '#3b82f6', '#f97316', '#a855f7', '#ef4444', '#14b8a6', '#eab308'];

function boxToPixel(b, data, nw, nh) {
  const fmt = data.box_format || (b.unit === 'percent' ? 'cxcywh_pct' : 'xyxy');
  if (fmt === 'cxcywh_pct' || b.unit === 'percent') {
    const cx = (b.cx / 100) * nw;
    const cy = (b.cy / 100) * nh;
    const w = (b.w / 100) * nw;
    const h = (b.h / 100) * nh;
    return { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 };
  }
  let x1 = Number(b.x1), y1 = Number(b.y1), x2 = Number(b.x2), y2 = Number(b.y2);
  return { x1, y1, x2, y2 };
}

function drawLabel(text, x, y, color) {
  ctx.font = 'bold 13px system-ui, sans-serif';
  const pad = 3;
  const tw = ctx.measureText(text).width;
  const th = 16;
  const ly = Math.max(th + 2, y);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
  ctx.fillRect(x, ly - th, tw + pad * 2, th + pad);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, x + pad, ly - 4);
  ctx.strokeStyle = color;
}

function renderPreview(data) {
  if (!imgEl.naturalWidth || !data) {
    canvas.classList.add('empty');
    return;
  }
  canvas.classList.remove('empty');
  const nw = data.image_width || imgEl.naturalWidth;
  const nh = data.image_height || imgEl.naturalHeight;
  const maxDisplay = Math.min(900, (document.getElementById('preview-wrap').clientWidth || 900));
  const scale = Math.min(1, maxDisplay / nw);
  const dw = Math.round(nw * scale);
  const dh = Math.round(nh * scale);
  canvas.width = dw;
  canvas.height = dh;
  ctx.clearRect(0, 0, dw, dh);
  ctx.drawImage(imgEl, 0, 0, dw, dh);

  const sx = dw / nw;
  const sy = dh / nh;
  const boxes = data.boxes || [];
  boxes.forEach((b, i) => {
    const { x1, y1, x2, y2 } = boxToPixel(b, data, nw, nh);
    const rx = x1 * sx;
    const ry = y1 * sy;
    const rw = Math.max(1, (x2 - x1) * sx);
    const rh = Math.max(1, (y2 - y1) * sy);
    const color = COLORS[i % COLORS.length];
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const bl = parseInt(color.slice(5, 7), 16);
    ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + bl + ',0.22)';
    ctx.fillRect(rx, ry, rw, rh);
    ctx.lineWidth = Math.max(2, 2 * scale);
    ctx.strokeStyle = color;
    ctx.strokeRect(rx, ry, rw, rh);
    const label = (b.label != null ? b.label : 'cls' + b.class_id) +
      (b.score != null ? ' ' + Number(b.score).toFixed(2) : '');
    drawLabel(label, rx, ry, color);
  });

  if (boxes.length === 0) {
    ctx.font = '14px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(100,116,139,0.9)';
    ctx.fillText('无检测框 (boxes=0)', 12, 24);
  }
}

function scheduleRender() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => renderPreview(lastPredict));
  });
}

window.addEventListener('resize', () => { if (lastPredict) scheduleRender(); });

async function loadModels() {
  const r = await fetch('/api/models');
  const data = await r.json();
  sel.innerHTML = '';
  (data.models || []).forEach(m => {
    const o = document.createElement('option');
    o.value = m.key;
    o.textContent = m.key + (m.name ? ' · ' + m.name : '') + (m.task ? ' [' + m.task + ']' : '');
    if (m.key === data.default_model) o.selected = true;
    sel.appendChild(o);
  });
  statusEl.textContent = '默认模型: ' + (data.default_model || '-');
}

function drawBoxes(data) {
  lastPredict = data;
  scheduleRender();
}

document.getElementById('image').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  imgEl.onload = () => {
    lastPredict = null;
    canvas.classList.remove('empty');
    const nw = imgEl.naturalWidth;
    const nh = imgEl.naturalHeight;
    const maxDisplay = Math.min(900, (document.getElementById('preview-wrap').clientWidth || 900));
    const scale = Math.min(1, maxDisplay / nw);
    canvas.width = Math.round(nw * scale);
    canvas.height = Math.round(nh * scale);
    ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
  };
  imgEl.src = url;
});

document.getElementById('btn_load').addEventListener('click', async () => {
  errEl.textContent = '';
  const key = sel.value;
  statusEl.textContent = '加载中…';
  try {
    const r = await fetch('/api/models/' + encodeURIComponent(key) + '/load', { method: 'POST' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    out.textContent = JSON.stringify(data, null, 2);
    statusEl.textContent = '已加载: ' + key;
  } catch (e) {
    errEl.textContent = String(e.message || e);
    statusEl.textContent = '';
  }
});

document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  errEl.textContent = '';
  const fd = new FormData();
  const file = document.getElementById('image').files[0];
  if (!file) { errEl.textContent = '请选择图片'; return; }
  fd.append('image', file);
  fd.append('model_key', sel.value);
  fd.append('box_format', document.getElementById('box_format').value);
  const conf = document.getElementById('conf').value;
  if (conf !== '') fd.append('conf', conf);

  const btn = document.getElementById('btn');
  btn.disabled = true;
  statusEl.textContent = '推理中…';
  try {
    const r = await fetch('/api/predict', { method: 'POST', body: fd });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    out.textContent = JSON.stringify(data, null, 2);
    if (!r.ok) throw new Error(data.detail || r.statusText || ('HTTP ' + r.status));
    statusEl.textContent = '完成 · boxes=' + (data.boxes?.length ?? 0) +
      ' · ' + ((data.timings?.total ?? data.timings?.infer) != null ? Number(data.timings.total ?? data.timings.infer).toFixed(3) + 's' : '');
    drawBoxes(data);
    if (!imgEl.complete || !imgEl.naturalWidth) {
      imgEl.onload = () => drawBoxes(data);
    }
  } catch (e) {
    errEl.textContent = String(e.message || e);
    statusEl.textContent = '';
  } finally {
    btn.disabled = false;
  }
});

loadModels().catch(e => { errEl.textContent = '无法加载 /api/models: ' + e; });
</script>
</body>
</html>
"""


def test_page() -> HTMLResponse:
    return HTMLResponse(TEST_PAGE_HTML)
