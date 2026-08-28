// Vessel Dashboard - app.js
const API = '';

// =========================================================================
// Hardware
// =========================================================================
async function loadHardware() {
    try {
        const res = await fetch(`${API}/api/hardware`);
        const hw = await res.json();

        // GPU
        const vramPct = hw.vram_total > 0 ? (hw.vram_used / hw.vram_total * 100) : 0;
        const vramBar = document.getElementById('vram-bar');
        vramBar.style.width = vramPct + '%';
        vramBar.className = 'gauge-fill' + (vramPct > 90 ? ' danger' : vramPct > 70 ? ' warn' : '');
        document.getElementById('vram-text').textContent =
            `${(hw.vram_used/1e9).toFixed(1)} / ${(hw.vram_total/1e9).toFixed(1)} GB`;

        // RAM
        const ramPct = hw.ram_total > 0 ? (hw.ram_used / hw.ram_total * 100) : 0;
        const ramBar = document.getElementById('ram-bar');
        ramBar.style.width = ramPct + '%';
        ramBar.className = 'gauge-fill' + (ramPct > 90 ? ' danger' : ramPct > 70 ? ' warn' : '');
        document.getElementById('ram-text').textContent =
            `${(hw.ram_used/1e9).toFixed(1)} / ${(hw.ram_total/1e9).toFixed(1)} GB`;

        // Temp
        document.getElementById('temp-text').textContent =
            hw.temp > 0 ? `${hw.temp}°C` : 'N/A';

        // Bandwidth
        document.getElementById('bw-text').textContent =
            hw.gpu_bw > 0 ? `${hw.gpu_bw.toFixed(0)} GB/s` : 'N/A';

        // Platform badge
        document.getElementById('platform-badge').textContent = hw.platform || 'Unknown';
    } catch (e) {
        console.error('Hardware fetch failed:', e);
    }
}

// =========================================================================
// Model Analysis
// =========================================================================
let currentStrategies = [];

async function analyzeModel() {
    const url = document.getElementById('model-url').value.trim();
    if (!url) return;

    const status = document.getElementById('analyze-status');
    const btn = document.getElementById('btn-analyze');

    status.textContent = 'Fetching metadata...';
    status.className = 'status loading';
    btn.disabled = true;

    try {
        const res = await fetch(`${API}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_url: url })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || `HTTP ${res.status}`);
        }

        const data = await res.json();
        currentStrategies = data.strategies || [];

        // Show model info
        document.getElementById('model-info').innerHTML =
            `<strong>${data.model.name || url.split('/').pop()}</strong> | ` +
            `${data.model.params} params | ${data.model.layers} layers | ` +
            `${data.model.arch} | ${data.model.is_moe ? 'MoE' : 'Dense'}`;

        // Render table
        renderStrategies(data.strategies, data.hw);

        document.getElementById('strategy-panel').classList.remove('hidden');
        status.textContent = `Found ${data.strategies.length} strategies in ${data.time_ms}ms`;
        status.className = 'status';
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
        status.className = 'status error';
    } finally {
        btn.disabled = false;
    }
}

function renderStrategies(strategies, hw) {
    const tbody = document.getElementById('strategy-body');
    tbody.innerHTML = '';

    strategies.forEach((s, i) => {
        const viable = s.status === 'VIABLE' || s.status === 'TIGHT';
        const nofit = s.status === 'NO_FIT';
        const stream = s.status === 'STREAM';

        let statusClass = 'status-viable';
        let statusText = 'VIABLE';
        if (nofit) { statusClass = 'status-no-fit'; statusText = 'NO FIT'; }
        else if (stream) { statusClass = 'status-stream'; statusText = 'STREAM'; }
        else if (s.status === 'TIGHT') { statusClass = 'status-tight'; statusText = 'TIGHT'; }

        const placementNames = {
            'FULL_GPU': 'Full GPU',
            'GPU_CPU_SPLIT': 'Split',
            'CPU_ONLY': 'CPU Only',
            'HOT_COLD_SPLIT': 'Hot/Cold',
            'LAYER_STREAM': 'Layer Stream',
            'MOE_FULL_VRAM': 'MoE Full',
            'MOE_EXPERT_OFFLOAD': 'MoE Offload',
            'MOE_CPU_ONLY': 'MoE CPU'
        };

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${placementNames[s.placement] || s.placement}</td>
            <td>${s.gpu_layers}</td>
            <td>${s.ctx >= 1024 ? (s.ctx/1024)+'K' : s.ctx}</td>
            <td>${s.kv === 16 ? 'FP16' : 'Q8'}</td>
            <td>${s.vram > 0 ? (s.vram/1e9).toFixed(1)+' GB' : '--'}</td>
            <td>${s.ram > 0 ? (s.ram/1e9).toFixed(1)+' GB' : '--'}</td>
            <td>${s.tps > 0 ? '~'+s.tps.toFixed(1) : '--'}</td>
            <td>${s.ttft > 0 ? '~'+s.ttft.toFixed(0)+'ms' : '--'}</td>
            <td class="${statusClass}">${statusText}</td>
            <td>${viable ? `<button class="btn-sm" onclick="selectStrategy(${i})">Select</button>` : ''}</td>
        `;
        tbody.appendChild(row);
    });
}

function selectStrategy(index) {
    const s = currentStrategies[index];
    document.getElementById('exec-panel').classList.remove('hidden');
    const log = document.getElementById('exec-log');
    log.textContent = `Strategy #${index+1} selected: ${s.placement}, ${s.gpu_layers} layers, ${s.ctx/1024}K context\n`;
    log.textContent += `Execution requires the CLI: vessel --model <url> --execute\n`;
    // Future: POST /api/execute with SSE streaming
}

// =========================================================================
// Recommendations
// =========================================================================
async function loadRecommendations() {
    const priority = document.getElementById('rec-priority').value;
    const useCase = document.getElementById('rec-usecase').value;

    try {
        const res = await fetch(`${API}/api/recommend?priority=${priority}&use_case=${useCase}&top=8`);
        const data = await res.json();

        const tbody = document.getElementById('rec-body');
        tbody.innerHTML = '';

        (data.recommendations || []).forEach((r, i) => {
            const stars = '★'.repeat(Math.round(r.quality)) + '☆'.repeat(5 - Math.round(r.quality));
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${i + 1}</td>
                <td>${r.label || ''}</td>
                <td>${r.model}</td>
                <td>${r.quant}</td>
                <td>${r.strategy}</td>
                <td>${r.vram > 0 ? (r.vram/1e9).toFixed(1)+' GB' : '--'}</td>
                <td>${r.tps > 0 ? '~'+r.tps.toFixed(1) : '--'}</td>
                <td class="stars">${stars}</td>
                <td>${r.download_gb > 0 ? r.download_gb.toFixed(1)+' GB' : '--'}</td>
                <td>${r.url ? `<button class="btn-sm" onclick="document.getElementById('model-url').value='${r.url}';analyzeModel()">Use</button>` : ''}</td>
            `;
            tbody.appendChild(row);
        });

        document.getElementById('rec-summary').innerHTML =
            data.top_pick ? `<strong>Top pick:</strong> ${data.top_pick}` : '';
    } catch (e) {
        console.error('Recommendations failed:', e);
    }
}

// =========================================================================
// Calibration
// =========================================================================
async function loadCalibration() {
    try {
        const res = await fetch(`${API}/api/calibration`);
        const data = await res.json();

        const info = document.getElementById('cal-info');
        if (data.records === 0) {
            info.textContent = 'No calibration data yet. Run a model execution to generate data.';
        } else {
            info.innerHTML = `
                <strong>${data.records}</strong> calibration records | 
                Fingerprint: <code>${data.fingerprint || 'N/A'}</code><br>
                GPU overhead: ${data.gpu_overhead_mb || 512} MB |
                GPU decode eff: ${(data.gpu_decode_eff || 0.27).toFixed(3)} |
                CPU decode eff: ${(data.cpu_decode_eff || 0.80).toFixed(3)}
            `;
        }
    } catch (e) {
        console.error('Calibration fetch failed:', e);
    }
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    loadHardware();
    loadRecommendations();
    loadCalibration();
    // Refresh hardware every 5 seconds
    setInterval(loadHardware, 5000);
});
