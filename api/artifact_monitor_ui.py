"""Browser console for the model artifact inotify monitor."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["model-artifact-monitor-ui"])

MONITOR_PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NIII 模型文件安全监控</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --ink: #10233f;
      --muted: #62728a;
      --line: #dfe7f1;
      --panel: #ffffff;
      --canvas: #f3f6fa;
      --navy: #07192f;
      --navy-2: #102b4c;
      --blue: #1565d8;
      --cyan: #19a9c5;
      --green: #087f5b;
      --green-bg: #e9f8f2;
      --amber: #a85d00;
      --amber-bg: #fff5dc;
      --red: #ba2d31;
      --red-bg: #fff0f0;
      --shadow: 0 12px 36px rgba(16, 35, 63, .08);
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--canvas); min-width: 320px; }
    button, input { font: inherit; }
    button { cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: .46; }
    code, pre, .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
    .topbar {
      min-height: 72px; padding: 0 32px; color: #fff; background: var(--navy);
      display: flex; align-items: center; justify-content: space-between; gap: 24px;
      border-bottom: 1px solid rgba(255,255,255,.1);
    }
    .brand { display: flex; align-items: center; gap: 13px; min-width: 0; }
    .brand-mark {
      width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
      background: linear-gradient(145deg, #36d4d4, #1671e8); font-weight: 800;
      box-shadow: 0 8px 22px rgba(25,169,197,.28);
    }
    .brand-name { font-weight: 720; letter-spacing: .01em; }
    .brand-sub { color: #9fb1c8; font-size: 12px; margin-top: 2px; }
    .top-actions { display: flex; align-items: center; gap: 18px; }
    .top-actions a { color: #c8d6e8; text-decoration: none; font-size: 13px; }
    .connection { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: #95a4b8; }
    .dot.ok { background: #39d98a; box-shadow: 0 0 0 5px rgba(57,217,138,.13); }
    .dot.bad { background: #ff6b6f; box-shadow: 0 0 0 5px rgba(255,107,111,.13); }
    .hero {
      color: #fff; padding: 30px 32px 58px;
      background: radial-gradient(circle at 78% -40%, #245b94 0, transparent 43%),
                  linear-gradient(135deg, var(--navy), var(--navy-2));
    }
    .hero-inner { max-width: 1540px; margin: 0 auto; }
    .eyebrow { color: #66dce5; text-transform: uppercase; letter-spacing: .15em; font-size: 11px; font-weight: 800; }
    h1 { margin: 8px 0 6px; font-size: clamp(25px, 3vw, 38px); line-height: 1.15; letter-spacing: -.025em; }
    .hero p { max-width: 720px; margin: 0; color: #b8c7da; font-size: 14px; line-height: 1.6; }
    .page { max-width: 1604px; margin: -31px auto 0; padding: 0 32px 40px; }
    .workspace { display: grid; grid-template-columns: 330px minmax(0, 1fr); gap: 20px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 15px; box-shadow: var(--shadow); }
    .control { position: sticky; top: 16px; overflow: hidden; }
    .panel-head { padding: 18px 20px; border-bottom: 1px solid var(--line); }
    .panel-head h2 { margin: 0; font-size: 15px; }
    .panel-head p { color: var(--muted); font-size: 12px; margin: 5px 0 0; line-height: 1.45; }
    .control form { padding: 18px 20px; display: grid; gap: 14px; }
    label { display: grid; gap: 7px; color: #374c68; font-size: 12px; font-weight: 700; }
    input[type="text"] {
      width: 100%; border: 1px solid #cbd7e6; border-radius: 9px; padding: 10px 11px;
      color: var(--ink); outline: none; background: #fbfcfe;
    }
    input[type="text"]:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(21,101,216,.1); background: #fff; }
    .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-top: 3px; }
    .btn { border: 0; border-radius: 9px; padding: 10px 13px; font-size: 13px; font-weight: 750; }
    .btn-primary { background: var(--blue); color: #fff; box-shadow: 0 7px 17px rgba(21,101,216,.2); }
    .btn-danger { color: var(--red); background: var(--red-bg); border: 1px solid #f3c8c9; }
    .auto-row { padding: 14px 20px; background: #f8fafc; border-top: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--muted); }
    .switch { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; }
    .switch input { accent-color: var(--blue); }
    .dashboard { display: grid; gap: 16px; min-width: 0; }
    .alert { border-radius: 12px; padding: 13px 16px; font-size: 13px; font-weight: 680; border: 1px solid; }
    .alert-danger { color: var(--red); border-color: #efc1c3; background: var(--red-bg); }
    .alert-warning { color: var(--amber); border-color: #f0d39a; background: var(--amber-bg); }
    [hidden] { display: none !important; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .metric { background: #fff; border: 1px solid var(--line); border-radius: 13px; padding: 15px 17px; min-height: 104px; box-shadow: 0 8px 28px rgba(16,35,63,.05); }
    .metric-label { color: var(--muted); font-size: 11px; font-weight: 750; letter-spacing: .045em; text-transform: uppercase; }
    .metric-value { margin-top: 13px; font-size: 25px; font-weight: 760; line-height: 1; letter-spacing: -.02em; }
    .metric-foot { margin-top: 9px; color: #8290a4; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .status-badge { display: inline-flex; align-items: center; gap: 7px; padding: 6px 9px; border-radius: 99px; font-size: 13px; font-weight: 760; }
    .status-idle { color: #55657a; background: #eef2f7; }
    .status-running, .status-passed { color: var(--green); background: var(--green-bg); }
    .status-starting { color: var(--amber); background: var(--amber-bg); }
    .status-failed, .status-error { color: var(--red); background: var(--red-bg); }
    .section-head { padding: 16px 18px; display: flex; align-items: center; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--line); }
    .section-title { display: flex; align-items: baseline; gap: 9px; min-width: 0; }
    .section-title h2 { margin: 0; font-size: 14px; }
    .count { color: var(--muted); font-size: 11px; }
    .tiny-button { border: 1px solid #d1dbe8; color: #40536e; background: #fff; border-radius: 7px; padding: 6px 9px; font-size: 11px; }
    .table-wrap { overflow: auto; max-height: 430px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    thead { position: sticky; top: 0; z-index: 1; background: #f7f9fc; }
    th { color: #66778f; text-align: left; font-size: 10px; letter-spacing: .055em; text-transform: uppercase; }
    th, td { padding: 10px 13px; border-bottom: 1px solid #edf1f6; vertical-align: top; }
    td.path { max-width: 480px; word-break: break-all; color: #30435f; }
    tbody tr:hover { background: #f8fbff; }
    tbody tr.forbidden { background: #fff4f4; }
    .event-kind { display: inline-block; color: #34506f; background: #edf3fa; border-radius: 5px; padding: 3px 6px; font-weight: 700; }
    .event-kind.danger { color: var(--red); background: #ffe1e2; }
    .empty { padding: 36px 18px; text-align: center; color: #8190a4; font-size: 12px; }
    .lower-grid { display: grid; grid-template-columns: minmax(280px, .85fr) minmax(360px, 1.15fr); gap: 16px; }
    .detail-body { padding: 15px 18px; }
    .detail-list { margin: 0; display: grid; grid-template-columns: 104px minmax(0, 1fr); gap: 10px 12px; font-size: 12px; }
    .detail-list dt { color: var(--muted); }
    .detail-list dd { margin: 0; color: #2f4664; word-break: break-all; }
    pre {
      margin: 0; color: #cdd9e7; background: #071421; padding: 15px 17px; min-height: 210px;
      max-height: 330px; overflow: auto; font-size: 11px; line-height: 1.58; white-space: pre-wrap;
      word-break: break-word; border-radius: 0 0 14px 14px;
    }
    .summary { margin-top: 16px; }
    .footer-note { color: #75849a; font-size: 11px; line-height: 1.55; padding: 2px 3px; }
    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
      .control { position: static; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .lower-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
      .topbar, .hero { padding-left: 18px; padding-right: 18px; }
      .page { padding-left: 12px; padding-right: 12px; }
      .brand-sub, .top-actions a { display: none; }
      .metrics { grid-template-columns: 1fr 1fr; }
      .metric { min-height: 94px; padding: 13px; }
      .metric-value { font-size: 21px; }
      th, td { padding: 9px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">N</div>
      <div>
        <div class="brand-name">NIII Model Guard</div>
        <div class="brand-sub">模型文件安全证据控制台</div>
      </div>
    </div>
    <div class="top-actions">
      <a href="/docs" target="_blank" rel="noreferrer">API 文档</a>
      <span class="connection"><span id="connection-dot" class="dot"></span><span id="connection-text">正在连接</span></span>
    </div>
  </header>

  <section class="hero">
    <div class="hero-inner">
      <div class="eyebrow">Linux inotify evidence</div>
      <h1>模型文件事件监控</h1>
      <p>独立采集训练目录与专属临时目录的文件事件。页面只负责控制与展示，JSONL 和最终 summary 才是发布门禁证据。</p>
    </div>
  </section>

  <main class="page">
    <div class="workspace">
      <aside class="panel control">
        <div class="panel-head">
          <h2>监控会话</h2>
          <p>选择已有训练批次，确认监控进入运行状态后再启动训练。</p>
        </div>
        <form id="monitor-form">
          <label>项目 ID
            <input id="project-id" type="text" value="algorithms" autocomplete="off" required>
          </label>
          <label>任务 ID
            <input id="task-id" type="text" placeholder="例如 Helmet" autocomplete="off" required>
          </label>
          <label>训练批次
            <input id="train-num" type="text" value="train1" autocomplete="off" required>
          </label>
          <div class="button-row">
            <button id="start-button" class="btn btn-primary" type="submit">启动监控</button>
            <button id="stop-button" class="btn btn-danger" type="button" disabled>停止监控</button>
          </div>
        </form>
        <div class="auto-row">
          <span>刷新间隔 1 秒</span>
          <label class="switch"><input id="auto-refresh" type="checkbox" checked> 自动刷新</label>
        </div>
      </aside>

      <section class="dashboard">
        <div id="danger-alert" class="alert alert-danger" role="alert" hidden></div>
        <div id="warning-alert" class="alert alert-warning" role="status" hidden></div>

        <div class="metrics">
          <article class="metric">
            <div class="metric-label">监控状态</div>
            <div class="metric-value"><span id="status-badge" class="status-badge status-idle">未启动</span></div>
            <div id="status-foot" class="metric-foot">等待创建会话</div>
          </article>
          <article class="metric">
            <div class="metric-label">文件事件</div>
            <div id="event-count" class="metric-value">0</div>
            <div class="metric-foot">已持久化 filesystem events</div>
          </article>
          <article class="metric">
            <div class="metric-label">禁止工件事件</div>
            <div id="forbidden-count" class="metric-value">0</div>
            <div class="metric-foot">.pt / .pth / .ckpt / ONNX</div>
          </article>
          <article class="metric">
            <div class="metric-label">证据完整性异常</div>
            <div id="integrity-count" class="metric-value">0</div>
            <div class="metric-foot">溢出、根目录丢失或读取错误</div>
          </article>
        </div>

        <section class="panel">
          <div class="section-head">
            <div class="section-title"><h2>实时文件事件</h2><span id="visible-event-count" class="count">0 条</span></div>
            <button id="clear-events" class="tiny-button" type="button">清空页面</button>
          </div>
          <div class="table-wrap">
            <table aria-label="文件事件日志">
              <thead><tr><th>序号</th><th>时间</th><th>类型</th><th>事件</th><th>路径</th></tr></thead>
              <tbody id="event-body"></tbody>
            </table>
            <div id="event-empty" class="empty">监控启动后，文件系统事件将在这里实时出现。</div>
          </div>
        </section>

        <div class="lower-grid">
          <section class="panel">
            <div class="section-head"><div class="section-title"><h2>会话与证据</h2></div><button id="copy-evidence" class="tiny-button" type="button" disabled>复制证据目录</button></div>
            <div class="detail-body">
              <dl class="detail-list">
                <dt>Session</dt><dd id="session-id" class="mono">—</dd>
                <dt>PID</dt><dd id="monitor-pid" class="mono">—</dd>
                <dt>训练目录</dt><dd id="run-root" class="mono">—</dd>
                <dt>临时目录</dt><dd id="temp-root" class="mono">—</dd>
                <dt>证据目录</dt><dd id="evidence-root" class="mono">—</dd>
                <dt>退出码</dt><dd id="exit-code" class="mono">—</dd>
              </dl>
            </div>
          </section>

          <section class="panel">
            <div class="section-head"><div class="section-title"><h2>监控进程日志</h2><span id="process-log-count" class="count">0 行</span></div></div>
            <pre id="process-log" aria-live="polite">尚无进程输出。</pre>
          </section>
        </div>

        <section class="panel summary">
          <div class="section-head"><div class="section-title"><h2>最终证据摘要</h2><span class="count">停止后生成</span></div></div>
          <pre id="summary-json">尚未生成 summary.json。</pre>
        </section>
        <div class="footer-note">判定通过必须同时满足 status=passed、forbiddenEventCount=0、integrityFailures=[]。Web 页面状态不能替代原始证据文件。</div>
      </section>
    </div>
  </main>

  <script>
    const apiBase = '/api/model-artifact-monitor';
    const byId = id => document.getElementById(id);
    const state = { sessionId: null, lastSequence: 0, events: [], polling: false };
    const statusLabels = {
      idle: '未启动', starting: '启动中', running: '监控中',
      passed: '已通过', failed: '未通过', error: '异常'
    };

    async function requestJson(url, options = {}) {
      const response = await fetch(url, { cache: 'no-store', ...options });
      const text = await response.text();
      let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      return data;
    }

    function setConnection(ok, text) {
      byId('connection-dot').className = `dot ${ok ? 'ok' : 'bad'}`;
      byId('connection-text').textContent = text;
    }

    function setAlert(id, message) {
      const element = byId(id);
      element.textContent = message || '';
      element.hidden = !message;
    }

    function resetEventView() {
      state.lastSequence = 0;
      state.events = [];
      renderEvents();
    }

    function renderStatus(data) {
      if (data.sessionId && data.sessionId !== state.sessionId) {
        state.sessionId = data.sessionId;
        resetEventView();
      }
      if (!data.sessionId) state.sessionId = null;
      const status = data.status || 'error';
      const summary = data.summary || {};
      const isRunning = status === 'running' || status === 'starting';
      const liveFilesystemEvents = state.events.filter(
        item => item.kind === 'filesystem_event'
      ).length;
      const liveForbiddenEvents = state.events.filter(
        item => item.kind === 'forbidden_artifact'
      ).length;
      const liveIntegrityFailures = state.events.filter(
        item => item.kind === 'monitor_integrity_failure' || item.kind === 'monitor_error'
      ).length;
      const badge = byId('status-badge');
      badge.className = `status-badge status-${status}`;
      badge.textContent = statusLabels[status] || status;
      byId('status-foot').textContent = data.startedAt || '等待创建会话';
      byId('event-count').textContent = summary.filesystemEventCount ?? liveFilesystemEvents;
      byId('forbidden-count').textContent = summary.forbiddenEventCount ?? liveForbiddenEvents;
      byId('integrity-count').textContent = summary.integrityFailures
        ? summary.integrityFailures.length : liveIntegrityFailures;
      byId('session-id').textContent = data.sessionId || '—';
      byId('monitor-pid').textContent = data.pid ?? '—';
      byId('run-root').textContent = (data.roots || [])[0] || '—';
      byId('temp-root').textContent = data.tempDir || '—';
      byId('evidence-root').textContent = data.evidenceDir || '—';
      byId('exit-code').textContent = data.monitorExitCode ?? '—';
      byId('start-button').disabled = isRunning;
      byId('stop-button').disabled = !isRunning;
      byId('copy-evidence').disabled = !data.evidenceDir;
      byId('summary-json').textContent = data.summary ? JSON.stringify(data.summary, null, 2) : '尚未生成 summary.json。';

      const forbidden = Number(summary.forbiddenEventCount ?? liveForbiddenEvents);
      const integrityCount = summary.integrityFailures
        ? summary.integrityFailures.length : liveIntegrityFailures;
      if (status === 'failed' || forbidden > 0 || integrityCount > 0) {
        setAlert('danger-alert', `证据判定失败：禁止工件事件 ${forbidden}，完整性异常 ${integrityCount}。请检查事件和 summary。`);
      } else {
        setAlert('danger-alert', '');
      }
    }

    function kindLabel(record) {
      if (record.kind === 'forbidden_artifact') return '禁止工件';
      if (record.kind === 'monitor_integrity_failure') return '完整性异常';
      if (record.kind === 'filesystem_event') return '文件事件';
      if (record.kind === 'monitor_started') return '监控启动';
      if (record.kind === 'monitor_stopped') return '监控停止';
      if (record.kind === 'monitor_error') return '监控错误';
      return record.kind || '未知';
    }

    function renderEvents() {
      const body = byId('event-body');
      body.replaceChildren();
      const records = [...state.events].reverse();
      for (const record of records) {
        const row = document.createElement('tr');
        const dangerous = record.forbidden || record.kind === 'forbidden_artifact' ||
          record.kind === 'monitor_integrity_failure' || record.kind === 'monitor_error';
        if (dangerous) row.className = 'forbidden';
        const values = [
          record.sequence ?? '—',
          record.observedAt || '—',
          kindLabel(record),
          Array.isArray(record.events) ? record.events.join(' · ') : (record.event || '—'),
          record.path || record.message || record.error || '—'
        ];
        values.forEach((value, index) => {
          const cell = document.createElement('td');
          if (index === 2) {
            const badge = document.createElement('span');
            badge.className = `event-kind${dangerous ? ' danger' : ''}`;
            badge.textContent = String(value);
            cell.appendChild(badge);
          } else {
            cell.textContent = String(value);
          }
          if (index === 4) cell.className = 'path mono';
          row.appendChild(cell);
        });
        body.appendChild(row);
      }
      byId('event-empty').hidden = records.length > 0;
      byId('visible-event-count').textContent = `${records.length} 条`;
    }

    async function loadEvents() {
      if (!state.sessionId) return;
      const data = await requestJson(`${apiBase}/events?afterSequence=${state.lastSequence}&limit=500`);
      if (data.sessionId !== state.sessionId) return;
      for (const record of data.events || []) {
        state.events.push(record);
        state.lastSequence = Math.max(state.lastSequence, Number(record.sequence || 0));
      }
      if (state.events.length > 500) state.events = state.events.slice(-500);
      renderEvents();
    }

    async function loadProcessLog() {
      if (!state.sessionId) return;
      const data = await requestJson(`${apiBase}/process-log?tail=300`);
      if (data.sessionId !== state.sessionId) return;
      const lines = data.lines || [];
      byId('process-log').textContent = lines.length ? lines.join('\n') : '监控进程当前没有标准输出。';
      byId('process-log-count').textContent = `${lines.length} 行`;
    }

    async function poll() {
      if (state.polling) return;
      state.polling = true;
      try {
        const status = await requestJson(`${apiBase}/status`);
        renderStatus(status);
        if (status.sessionId) {
          await Promise.all([loadEvents(), loadProcessLog()]);
          renderStatus(status);
        }
        setConnection(true, '服务已连接');
        setAlert('warning-alert', '');
      } catch (error) {
        setConnection(false, '连接异常');
        setAlert('warning-alert', `刷新失败：${error.message || error}`);
      } finally {
        state.polling = false;
      }
    }

    byId('monitor-form').addEventListener('submit', async event => {
      event.preventDefault();
      setAlert('warning-alert', '');
      byId('start-button').disabled = true;
      try {
        const data = await requestJson(`${apiBase}/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            projectId: byId('project-id').value.trim(),
            taskId: byId('task-id').value.trim(),
            trainNum: byId('train-num').value.trim()
          })
        });
        state.sessionId = data.sessionId || null;
        resetEventView();
        renderStatus(data);
        await poll();
      } catch (error) {
        setAlert('warning-alert', `启动失败：${error.message || error}`);
        byId('start-button').disabled = false;
      }
    });

    byId('stop-button').addEventListener('click', async () => {
      byId('stop-button').disabled = true;
      setAlert('warning-alert', '正在停止监控并刷新证据…');
      try {
        const data = await requestJson(`${apiBase}/stop`, { method: 'POST' });
        renderStatus(data);
        await Promise.all([loadEvents(), loadProcessLog()]);
        setAlert('warning-alert', data.status === 'passed' ? '监控已停止，证据判定通过。' : '监控已停止，请处理未通过项。');
      } catch (error) {
        setAlert('warning-alert', `停止失败：${error.message || error}`);
      } finally {
        await poll();
      }
    });

    byId('clear-events').addEventListener('click', () => {
      state.events = [];
      renderEvents();
    });
    byId('copy-evidence').addEventListener('click', async () => {
      const value = byId('evidence-root').textContent;
      if (!value || value === '—') return;
      try {
        await navigator.clipboard.writeText(value);
        setAlert('warning-alert', '证据目录已复制。');
      } catch {
        setAlert('warning-alert', `无法访问剪贴板，请手动复制：${value}`);
      }
    });

    setInterval(() => {
      if (byId('auto-refresh').checked && !document.hidden) poll();
    }, 1000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
    poll();
  </script>
</body>
</html>
"""


@router.get("/monitor/model-artifacts", response_class=HTMLResponse, include_in_schema=False)
def artifact_monitor_page() -> HTMLResponse:
    return HTMLResponse(
        MONITOR_PAGE_HTML,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = ["artifact_monitor_page", "router"]
