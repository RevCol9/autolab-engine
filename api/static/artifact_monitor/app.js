'use strict';

const apiBase = '/api/model-artifact-monitor';
const maxVisibleEvents = 500;
const state = {
  sessionId: null,
  cursor: 0,
  events: [],
  filesystemEvents: 0,
  forbiddenEvents: 0,
  integrityFailures: 0,
  summary: null,
  polling: false
};

const byId = id => document.getElementById(id);

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function setText(id, value) {
  byId(id).textContent = value ?? '—';
}

function setAlert(message = '') {
  const element = byId('alert');
  element.hidden = !message;
  element.textContent = message;
}

function resetSession(sessionId) {
  state.sessionId = sessionId;
  state.cursor = 0;
  state.events = [];
  state.filesystemEvents = 0;
  state.forbiddenEvents = 0;
  state.integrityFailures = 0;
  state.summary = null;
  renderEvents();
}

function applyEvent(record) {
  if (record.kind === 'filesystem_event') state.filesystemEvents += 1;
  if (record.kind === 'forbidden_artifact') state.forbiddenEvents += 1;
  if (record.kind === 'monitor_integrity_failure' || record.kind === 'monitor_error') {
    state.integrityFailures += 1;
  }
  state.events.push(record);
}

function renderCounters(summary = state.summary) {
  const filesystem = summary?.filesystemEventCount ?? state.filesystemEvents;
  const forbidden = summary?.forbiddenEventCount ?? state.forbiddenEvents;
  const integrity = summary?.integrityFailureCount ?? state.integrityFailures;
  setText('event-count', filesystem);
  setText('forbidden-count', forbidden);
  setText('integrity-count', integrity);
  byId('forbidden-count').classList.toggle('active', forbidden > 0);
  byId('integrity-count').classList.toggle('active', integrity > 0);
}

function renderStatus(data) {
  if (data.sessionId && data.sessionId !== state.sessionId) resetSession(data.sessionId);
  if (data.summary) state.summary = data.summary;
  const status = String(data.status || 'idle').toUpperCase();
  const active = status === 'RUNNING' || status === 'STARTING';
  setText('status', status);
  setText('session-id', data.sessionId);
  setText('pid', data.pid);
  setText('roots', data.roots?.join('\n'));
  setText('evidence-root', data.evidenceDir);
  renderCounters();
  byId('start-button').disabled = active;
  byId('stop-button').disabled = !active;
}

function renderEvents() {
  const body = byId('events');
  if (!state.events.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">暂无事件</td></tr>';
    return;
  }
  body.replaceChildren(...state.events.map(record => {
    const row = document.createElement('tr');
    const violation = record.forbidden || record.kind === 'forbidden_artifact' ||
      record.kind.includes('error') || record.kind.includes('failure');
    if (violation) row.className = 'violation';
    const values = [
      record.sequence,
      record.observedAt,
      record.kind,
      Array.isArray(record.events) ? record.events.join('|') : (record.event || ''),
      record.path || record.message || record.error || ''
    ];
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = value ?? '';
      row.appendChild(cell);
    }
    return row;
  }));
}

async function loadEvents() {
  for (let page = 0; page < 5; page += 1) {
    const data = await requestJson(`${apiBase}/events?cursor=${state.cursor}&limit=500`);
    if (data.sessionId !== state.sessionId) return;
    for (const record of data.events) applyEvent(record);
    state.cursor = data.nextCursor;
    if (data.count < 500) break;
  }
  state.events = state.events.slice(-maxVisibleEvents);
  renderCounters();
  renderEvents();
}

async function loadProcessLog() {
  const data = await requestJson(`${apiBase}/process-log?tail=200`);
  if (data.sessionId === state.sessionId) {
    byId('process-log').textContent = data.lines.join('\n') || '暂无日志';
  }
}

async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const status = await requestJson(`${apiBase}/status`);
    renderStatus(status);
    if (status.sessionId) await Promise.all([loadEvents(), loadProcessLog()]);
    const connection = byId('connection');
    connection.textContent = '服务已连接';
    connection.className = 'badge ok';
  } catch (error) {
    const connection = byId('connection');
    connection.textContent = '连接异常';
    connection.className = 'badge bad';
    setAlert(`刷新失败：${error.message || error}`);
  } finally {
    state.polling = false;
  }
}

byId('monitor-form').addEventListener('submit', async event => {
  event.preventDefault();
  setAlert();
  byId('start-button').disabled = true;
  try {
    const data = await requestJson(`${apiBase}/start`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        projectId: byId('project-id').value.trim(),
        taskId: byId('task-id').value.trim(),
        trainNum: byId('train-num').value.trim()
      })
    });
    resetSession(data.sessionId);
    renderStatus(data);
    await poll();
  } catch (error) {
    setAlert(`启动失败：${error.message || error}`);
    byId('start-button').disabled = false;
  }
});

byId('stop-button').addEventListener('click', async () => {
  byId('stop-button').disabled = true;
  setAlert('正在停止监控并生成汇总…');
  try {
    const data = await requestJson(`${apiBase}/stop`, {method: 'POST'});
    renderStatus(data);
    await Promise.all([loadEvents(), loadProcessLog()]);
    setAlert(data.status === 'passed' ? '监控已停止，证据判定通过。' : '监控已停止，请处理未通过项。');
  } catch (error) {
    setAlert(`停止失败：${error.message || error}`);
  }
});

byId('clear-events').addEventListener('click', () => {
  state.events = [];
  renderEvents();
});

setInterval(() => {
  if (byId('auto-refresh').checked && !document.hidden) poll();
}, 1000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
poll();
