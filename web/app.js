// Vessel Dashboard - Phase C: Full API with envelope, SSE, all endpoints
const API = '';

// =========================================================================
// Helpers
// =========================================================================
function stars(n) {
    const full = Math.round(n / 2);
    return '★'.repeat(Math.min(full, 5)) + '☆'.repeat(5 - Math.min(full, 5));
}

function formatBytes(b) {
    if (b >= 1e12) return (b / 1e12).toFixed(1) + ' TB';
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB';
    return b + ' B';
}

function setStatus(id, text, type) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.className = 'status ' + (type || ''); }
}

// =========================================================================
// Hardware (with SSE for live updates)
// =========================================================================
let hwEventSource = null;

function startHardwareStream() {
    if (hwEventSource) hwEventSource.close();
    hwEventSource = new EventSource(`${API}/api/hardware/live`);
    hwEventSource.addEventListener('hardware', (e) => {
        const hw = JSON.parse(e.data);
        updateGauges(hw);
    });
    hwEventSource.onerror = () => {
        // Fall back to polling
        setTimeout(loadHardware, 2000);
    };
}

function updateGauges(hw) {
    const vramTotal = hw.vram_free_bytes + (hw.vram_used_bytes || 0);
    const vramUsed = hw.vram_used_bytes || 0;
    const vramPct = vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0;

    const vramBar = document.getElementById('vram-bar');
    if (vramBar) {
        vramBar.style.width = vramPct + '%';
        vramBar.className = 'gauge-fill' + (vramPct > 90 ? ' danger' : vramPct > 70 ? ' warn' : '');
    }
    const vramText = document.getElementById('vram-text');
    if (vramText) vramText.textContent = formatBytes(vramUsed) + ' / ' + formatBytes(vramTotal);

    const tempText = document.getElementById('temp-text');
    if (tempText) tempText.textContent = hw.gpu_temp_celsius > 0 ? hw.gpu_temp_celsius + '°C' : 'N/A';

    const bwText = document.getElementById('bw-text');
    if (bwText) bwText.textContent = (hw.gpu_bandwidth_gbs || 0) > 0 ? hw.gpu_bandwidth_gbs.toFixed(0) + ' GB/s' : 'N/A';

    // RAM
    const ramTotal = hw.ram_free_bytes + (hw.ram_used_bytes || 0);
    const ramUsed = hw.ram_used_bytes || 0;
    const ramPct = ramTotal > 0 ? (ramUsed / ramTotal * 100) : 0;
    const ramBar = document.getElementById('ram-bar');
    if (ramBar) {
        ramBar.style.width = ramPct + '%';
        ramBar.className = 'gauge-fill' + (ramPct > 90 ? ' danger' : ramPct > 70 ? ' warn' : '');
    }
    const ramText = document.getElementById('ram-text');
    if (ramText) ramText.textContent = formatBytes(ramUsed) + ' / ' + formatBytes(ramTotal);
}

async function loadHardware() {
    try {
        const res = await fetch(`${API}/api/hardware`);
        const json = await res.json();
        if (json.success) {
            updateGauges(json.data);
            document.getElementById('platform-badge').textContent = json.data.gpu_name || 'CPU';
        }
    } catch (e) { console.error('Hardware fetch failed:', e); }
}

// =========================================================================
// Model Analysis (POST /api/predict)
// =========================================================================
let currentStrategies = [];

async function analyzeModel() {
    const url = document.getElementById('model-url').value.trim();
    if (!url) return;

    setStatus('analyze-status', 'Fetching metadata...', 'loading');
    document.getElementById('btn-analyze').disabled = true;

    try {
        const res = await fetch(`${API}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_url: url, priority: 'speed' })
        });
        const json = await res.json();

        if (!json.success) {
            throw new Error(json.error?.message || 'Request failed');
        }

        const data = json.data;
        currentStrategies = data.strategies || [];

        document.getElementById('model-info').innerHTML =
            `<strong>${data.model.name}</strong> | ` +
            `${data.model.params} params | ${data.model.layers} layers | ` +
            `${data.model.architecture} | ${data.model.is_moe ? 'MoE' : 'Dense'}`;

        renderStrategies(data.strategies);
        document.getElementById('strategy-panel').classList.remove('hidden');
        setStatus('analyze-status', `Found ${data.strategies.length} strategies in ${data.time_ms}ms`);
    } catch (e) {
        setStatus('analyze-status', `Error: ${e.message}`, 'error');
    } finally {
        document.getElementById('btn-analyze').disabled = false;
    }
}

function renderStrategies(strategies) {
    const tbody = document.getElementById('strategy-body');
    tbody.innerHTML = '';
    const names = { 'FULL_GPU':'Full GPU', 'GPU_CPU_SPLIT':'Split', 'CPU_ONLY':'CPU', 'HOT_COLD_SPLIT':'Hot/Cold', 'LAYER_STREAM':'Stream' };

    strategies.forEach((s, i) => {
        const statusClass = s.status === 'VIABLE' ? 'status-viable' : s.status === 'NO_FIT' ? 'status-no-fit' : s.status === 'TIGHT' ? 'status-tight' : 'status-stream';
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${names[s.placement] || s.placement}</td>
            <td>${s.gpu_layers}</td>
            <td>${s.context_length >= 1024 ? (s.context_length/1024)+'K' : s.context_length}</td>
            <td>${s.kv_quant_bits === 16 ? 'FP16' : 'Q8'}</td>
            <td>${s.vram_bytes > 0 ? formatBytes(s.vram_bytes) : '--'}</td>
            <td>${s.ram_bytes > 0 ? formatBytes(s.ram_bytes) : '--'}</td>
            <td>${s.tokens_per_sec > 0 ? '~'+s.tokens_per_sec.toFixed(1) : '--'}</td>
            <td>${s.ttft_ms > 0 ? '~'+s.ttft_ms.toFixed(0)+'ms' : '--'}</td>
            <td class="${statusClass}">${s.status}</td>
            <td>${s.viable ? `<button class="btn-sm" onclick="selectStrategy(${i})">Use</button>` : ''}</td>
        `;
        tbody.appendChild(row);
    });
}

function selectStrategy(i) {
    const s = currentStrategies[i];
    document.getElementById('exec-panel').classList.remove('hidden');
    document.getElementById('exec-log').textContent =
        `Strategy #${i+1}: ${s.placement}, ${s.gpu_layers} layers, ${s.context_length/1024}K context\n` +
        `Run in CLI: vessel --model <url> --execute\n`;
}

// =========================================================================
// Recommendations (GET /api/recommend)
// =========================================================================
async function loadRecommendations() {
    const priority = document.getElementById('rec-priority').value;
    const useCase = document.getElementById('rec-usecase').value;

    try {
        const res = await fetch(`${API}/api/recommend?priority=${priority}&use_case=${useCase}&top=8`);
        const json = await res.json();
        if (!json.success) return;

        const tbody = document.getElementById('rec-body');
        tbody.innerHTML = '';
        (json.data.recommendations || []).forEach((r, i) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${i + 1}</td>
                <td>${r.label || ''}</td>
                <td>${r.model}</td>
                <td>${r.quant}</td>
                <td>${r.strategy}</td>
                <td>${r.vram_bytes > 0 ? formatBytes(r.vram_bytes) : '--'}</td>
                <td>${r.tokens_per_sec > 0 ? '~'+r.tokens_per_sec.toFixed(1) : '--'}</td>
                <td class="stars">${r.quality_stars}</td>
                <td>${r.download_gb > 0 ? r.download_gb.toFixed(1)+' GB' : '--'}</td>
                <td>${r.hf_url ? `<button class="btn-sm" onclick="document.getElementById('model-url').value='${r.hf_url}';analyzeModel()">Use</button>` : ''}</td>
            `;
            tbody.appendChild(row);
        });
    } catch (e) { console.error('Recommendations failed:', e); }
}

// =========================================================================
// Local Models (GET /api/models/local)
// =========================================================================
async function loadLocalModels() {
    try {
        const res = await fetch(`${API}/api/models/local`);
        const json = await res.json();
        if (json.success && json.data.models.length > 0) {
            const el = document.getElementById('local-models');
            if (el) el.innerHTML = `<strong>${json.data.count}</strong> local models found`;
        }
    } catch (e) { console.error('Local models fetch failed:', e); }
}

// =========================================================================
// Calibration (GET /api/calibration)
// =========================================================================
async function loadCalibration() {
    try {
        const res = await fetch(`${API}/api/calibration`);
        const json = await res.json();
        const info = document.getElementById('cal-info');
        if (!json.success || !info) return;
        const d = json.data;
        if (d.records === 0) {
            info.textContent = 'No calibration data yet. Run a model execution to generate data.';
        } else {
            info.innerHTML = `<strong>${d.matching_records}</strong> records for this hardware | ` +
                `GPU overhead: ${d.gpu_overhead_mb || 512} MB | ` +
                `GPU eff: ${(d.gpu_decode_efficiency || 0.27).toFixed(3)}`;
        }
    } catch (e) { console.error('Calibration fetch failed:', e); }
}

// =========================================================================
// Health Check
// =========================================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const json = await res.json();
        const el = document.getElementById('footer-status');
        if (el) el.textContent = json.success ? 'Connected' : 'Error';
    } catch (e) {
        const el = document.getElementById('footer-status');
        if (el) el.textContent = 'Disconnected';
    }
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    loadHardware();
    loadRecommendations();
    loadCalibration();
    checkHealth();
    startHardwareStream();
});
