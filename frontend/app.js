// === Configuration ===
const API_BASE_URL = (() => {
  // If served from the backend (same origin), use relative paths
  if (window.location.port === '8000' || window.location.origin.includes('localhost:8000')) {
    return '';
  }
  return 'http://localhost:8000';
})();

// === State ===
let sampleInputs = [];
let sessionHistory = '';
let lastTelemetry = null;
let lastExplanation = '';
let isAnalyzing = false;

// === DOM References ===
const statusDot = document.querySelector('.status-dot');
const statusLabel = document.querySelector('.status-label');
const sensorGrid = document.getElementById('sensor-grid');
const sampleButtonsContainer = document.getElementById('sample-buttons');
const telemetryInput = document.getElementById('telemetry-input');
const jsonValidation = document.getElementById('json-validation');
const analyzeBtn = document.getElementById('analyze-btn');
const outputSection = document.getElementById('output-section');
const anomalySummary = document.getElementById('anomaly-summary');
const anomalySeverity = document.getElementById('anomaly-severity');
const anomalyDirection = document.getElementById('anomaly-direction');
const anomalyDeviation = document.getElementById('anomaly-deviation');
const anomalyHeadline = document.getElementById('anomaly-headline');
const thinkingIndicator = document.getElementById('thinking-indicator');
const explanationContent = document.getElementById('explanation-content');
const followupArea = document.getElementById('followup-area');
const followupButtons = document.getElementById('followup-buttons');
const emergencyBanner = document.getElementById('emergency-banner');
const emergencyText = document.getElementById('emergency-text');
const errorDisplay = document.getElementById('error-display');

// === Initialization ===
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadSchema();
  setupEventListeners();
});

function setupEventListeners() {
  analyzeBtn.addEventListener('click', handleAnalyze);
  telemetryInput.addEventListener('input', validateJson);

  // Sample buttons
  sampleButtonsContainer.addEventListener('click', (e) => {
    const btn = e.target.closest('.sample-btn');
    if (!btn) return;
    const index = parseInt(btn.dataset.index, 10);
    loadSample(index);

    // Update active state
    document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
}

// === Health Check ===
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();

    statusDot.className = 'status-dot status-ok';
    statusLabel.textContent = data.status === 'healthy' ? 'System Online' : data.status;
  } catch (err) {
    statusDot.className = 'status-dot status-error';
    statusLabel.textContent = 'Offline';
  }
}

// === Load Schema / Samples ===
async function loadSchema() {
  try {
    const res = await fetch(`${API_BASE_URL}/schema`);
    if (!res.ok) throw new Error('Schema fetch failed');
    const data = await res.json();
    sampleInputs = data.sample_inputs || [];
  } catch (err) {
    console.warn('Could not load schema:', err.message);
  }
}

function loadSample(index) {
  if (index < 0 || index >= sampleInputs.length) return;
  telemetryInput.value = JSON.stringify(sampleInputs[index], null, 2);
  validateJson();
}

// === JSON Validation ===
function validateJson() {
  const raw = telemetryInput.value.trim();
  if (!raw) {
    jsonValidation.textContent = '';
    jsonValidation.className = 'json-validation';
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    jsonValidation.textContent = 'Valid JSON';
    jsonValidation.className = 'json-validation valid';
    return parsed;
  } catch (e) {
    jsonValidation.textContent = 'Invalid JSON';
    jsonValidation.className = 'json-validation invalid';
    return null;
  }
}

// === Analyze ===
async function handleAnalyze() {
  if (isAnalyzing) return;

  const telemetry = validateJson();
  if (!telemetry) {
    showError('Please enter valid JSON telemetry data.');
    return;
  }

  await submitAnalysis(telemetry, '', sessionHistory);
}

async function submitAnalysis(telemetry, query, history) {
  if (isAnalyzing) return;
  isAnalyzing = true;
  lastTelemetry = telemetry;

  // Reset UI
  hideError();
  dismissEmergency();
  outputSection.style.display = '';
  outputSection.classList.add('fade-in');
  anomalySummary.classList.add('hidden');
  followupArea.classList.add('hidden');
  followupButtons.innerHTML = '';
  explanationContent.textContent = '';
  lastExplanation = '';
  thinkingIndicator.classList.remove('hidden');
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="btn-icon">&#9203;</span> Analyzing...';

  const body = {
    telemetry: telemetry,
    query: query || '',
    history: history || ''
  };

  try {
    const res = await fetch(`${API_BASE_URL}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      if (res.status === 422 && errData) {
        const detail = errData.detail;
        const msg = Array.isArray(detail)
          ? detail.map(d => d.msg || JSON.stringify(d)).join('; ')
          : (typeof detail === 'string' ? detail : JSON.stringify(detail));
        throw new Error(`Validation error: ${msg}`);
      }
      throw new Error(`Server error (${res.status})`);
    }

    await readSSEStream(res);
  } catch (err) {
    showError(err.message || 'An unexpected error occurred. Check that the backend is running.');
    thinkingIndicator.classList.add('hidden');
  } finally {
    isAnalyzing = false;
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">&#9654;</span> Analyze';
  }
}

// === SSE Stream Reader ===
async function readSSEStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data:')) continue;

      const jsonStr = trimmed.slice(5).trim();
      if (!jsonStr || jsonStr === '[DONE]') continue;

      try {
        const event = JSON.parse(jsonStr);
        handleSSEEvent(event);
      } catch (e) {
        // Skip malformed events
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim().startsWith('data:')) {
    const jsonStr = buffer.trim().slice(5).trim();
    if (jsonStr && jsonStr !== '[DONE]') {
      try {
        handleSSEEvent(JSON.parse(jsonStr));
      } catch (e) { /* skip */ }
    }
  }
}

function handleSSEEvent(event) {
  switch (event.type) {
    case 'metadata':
      handleMetadata(event);
      break;
    case 'token':
      handleToken(event);
      break;
    case 'done':
      handleDone();
      break;
  }
}

// === Event Handlers ===
function handleMetadata(event) {
  thinkingIndicator.classList.add('hidden');

  const anomaly = event.anomaly;
  if (anomaly) {
    updateAnomalySummary(anomaly);
    updateSensorCard(lastTelemetry, anomaly);
    checkEmergency(lastTelemetry, anomaly);
  }

  if (event.follow_ups && event.follow_ups.length > 0) {
    renderFollowUps(event.follow_ups);
  }
}

function handleToken(event) {
  thinkingIndicator.classList.add('hidden');
  if (event.content) {
    lastExplanation += event.content;
    explanationContent.textContent = lastExplanation;

    // Auto-scroll
    const panel = document.getElementById('explanation-panel');
    panel.scrollTop = panel.scrollHeight;
  }
}

function handleDone() {
  sessionHistory = lastExplanation;
}

// === Anomaly Summary ===
function updateAnomalySummary(anomaly) {
  anomalySummary.classList.remove('hidden');

  const sev = (anomaly.severity || 'Unknown').toLowerCase();
  anomalySeverity.textContent = anomaly.severity || 'Unknown';
  anomalySeverity.className = 'anomaly-severity';
  if (sev === 'low') anomalySeverity.classList.add('sev-low');
  else if (sev === 'medium') anomalySeverity.classList.add('sev-medium');
  else if (sev === 'high') anomalySeverity.classList.add('sev-high');
  else if (sev === 'critical') anomalySeverity.classList.add('sev-critical');

  const dir = anomaly.direction || '';
  anomalyDirection.innerHTML = dir ? (dir === 'above' ? '&#9650; Above Normal' : '&#9660; Below Normal') : '';

  const dev = anomaly.deviation_value;
  const pct = anomaly.deviation_percent;
  if (dev != null && pct != null) {
    anomalyDeviation.textContent = `Deviation: ${dev} (${pct}%)`;
  } else {
    anomalyDeviation.textContent = '';
  }

  anomalyHeadline.textContent = anomaly.summary || '';
}

// === Sensor Card Updates ===
function updateSensorCard(telemetry, anomaly) {
  if (!telemetry) return;

  const sensorType = (telemetry.sensor_type || '').toUpperCase();
  const cards = document.querySelectorAll('.sensor-card');

  cards.forEach(card => {
    const cardType = card.dataset.sensor;
    if (cardType !== sensorType) return;

    card.classList.remove('active', 'severity-low', 'severity-medium', 'severity-high', 'severity-critical');
    card.classList.add('active');

    const badge = card.querySelector('.sensor-badge');
    if (anomaly.is_anomaly) {
      const sev = (anomaly.severity || '').toLowerCase();
      card.classList.add(`severity-${sev}`);

      if (sev === 'critical') {
        badge.className = 'sensor-badge badge-critical';
        badge.textContent = 'CRITICAL';
      } else if (sev === 'high') {
        badge.className = 'sensor-badge badge-high';
        badge.textContent = 'HIGH';
      } else if (sev === 'medium') {
        badge.className = 'sensor-badge badge-medium';
        badge.textContent = 'MEDIUM';
      } else {
        badge.className = 'sensor-badge badge-anomaly';
        badge.textContent = 'LOW';
      }
    } else {
      badge.className = 'sensor-badge badge-normal';
      badge.textContent = 'Normal';
    }
  });
}

// === Emergency Alert (GAS or SMOKE) ===
function checkEmergency(telemetry, anomaly) {
  if (!telemetry || !anomaly) return;
  const type = (telemetry.sensor_type || '').toUpperCase();
  const isCritical = (anomaly.severity || '').toLowerCase() === 'critical';

  if ((type === 'GAS' || type === 'SMOKE') && isCritical && anomaly.is_anomaly) {
    if (type === 'GAS') {
      emergencyText.textContent = 'CRITICAL GAS LEAK DETECTED \u2014 EVACUATE IMMEDIATELY';
    } else {
      emergencyText.textContent = 'CRITICAL SMOKE / FIRE DETECTED \u2014 TRIGGER FIRE ALARM';
    }
    emergencyBanner.classList.remove('hidden');
    document.body.style.paddingTop = '60px';
  }
}

function dismissEmergency() {
  emergencyBanner.classList.add('hidden');
  document.body.style.paddingTop = '0';
}
window.dismissEmergency = dismissEmergency;

// === Follow-ups ===
function renderFollowUps(suggestions) {
  followupArea.classList.remove('hidden');
  followupButtons.innerHTML = '';

  const items = suggestions.slice(0, 3);
  items.forEach(text => {
    const btn = document.createElement('button');
    btn.className = 'followup-btn';
    btn.textContent = text;
    btn.addEventListener('click', () => {
      submitAnalysis(lastTelemetry, text, sessionHistory);
    });
    followupButtons.appendChild(btn);
  });
}

// === Copy & Download ===
function copyExplanation() {
  if (!lastExplanation) return;
  navigator.clipboard.writeText(lastExplanation).then(() => {
    const btn = document.getElementById('copy-btn');
    const orig = btn.innerHTML;
    btn.innerHTML = '&#10003; Copied!';
    setTimeout(() => { btn.innerHTML = orig; }, 2000);
  }).catch(() => {
    showError('Failed to copy to clipboard.');
  });
}
window.copyExplanation = copyExplanation;

function downloadReport() {
  if (!lastExplanation) return;

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const telemetryStr = lastTelemetry ? JSON.stringify(lastTelemetry, null, 2) : 'N/A';
  const content = [
    '=== SENSOR ANOMALY REPORT ===',
    `Generated: ${new Date().toISOString()}`,
    '',
    '--- TELEMETRY DATA ---',
    telemetryStr,
    '',
    '--- AI ANALYSIS ---',
    lastExplanation,
    '',
    '=== END OF REPORT ==='
  ].join('\n');

  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sensor-report-${timestamp}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
window.downloadReport = downloadReport;

// === Error Handling ===
function showError(msg) {
  errorDisplay.textContent = msg;
  errorDisplay.classList.remove('hidden');
  setTimeout(() => hideError(), 10000);
}

function hideError() {
  errorDisplay.classList.add('hidden');
}
