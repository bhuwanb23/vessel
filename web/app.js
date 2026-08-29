// Vessel Dashboard - Phase D: Full SPA
const API = '';

// =========================================================================
// State
// =========================================================================
const state = {
    hardware: null,
    currentView: 'home',
    predictStrategies: [],
    hwSSE: null,
};

// =========================================================================
// API Client
// =========================================================================
const api = {
    async get(url) {
        const r = await fetch(API + url);
        const j = await r.json();
        if (!j.success) throw new Error(j.error?.message || 'Request failed');
        return j.data;
    },
    async post(url, body) {
        const r = await fetch(API + url, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        const j = await r.json();
        if (!j.success) throw new Error(j.error?.message || 'Request failed');
        return j.data;
    },
    async del(url) {
        const r = await fetch(API + url, { method: 'DELETE' });
        const j = await r.json();
        if (!j.success) throw new Error(j.error?.message || 'Request failed');
        return j.data;
    },
    async getHardware() { return this.get('/api/hardware'); },
    async predict(url, priority) { return this.post('/api/predict', { model_url: url, priority }); },
    async recommend(priority, useCase) { return this.get(`/api/recommend?priority=${priority}&use_case=${useCase}&top=8`); },
    async getCatalog() { return this.get('/api/catalog'); },
    async getLocalModels() { return this.get('/api/models/local'); },
    async getCalibration() { return this.get('/api/calibration'); },
    async getCalHistory() { return this.get('/api/calibration/history'); },
    async startDownload(url) { return this.post('/api/download', { model_url: url }); },
    async startExecute(path, strategy, prompt, maxTokens) {
        return this.post('/api/execute', { model_path: path, strategy, prompt, max_tokens: maxTokens });
    },
    async abortExecute(id) { return this.post(`/api/execute/${id}/abort`, {}); },
    async resetCalibration() { return this.del('/api/calibration'); },
};

// =========================================================================
// SSE Client
// =========================================================================
function subscribeSSE(url, handlers) {
    const source = new EventSource(API + url);
    for (const [event, handler] of Object.entries(handlers)) {
        source.addEventListener(event, e => handler(JSON.parse(e.data)));
    }
    source.onerror = () => {};
    return source;
}

// =========================================================================
// Helpers
// =========================================================================
function fmt(b) {
    if (b >= 1e12) return (b/1e12).toFixed(1)+' TB';
    if (b >= 1e9) return (b/1e9).toFixed(1)+' GB';
    if (b >= 1e6) return (b/1e6).toFixed(0)+' MB';
    return b+' B';
}

function stars(n) {
    const f = Math.round(n/2);
    return '★'.repeat(Math.min(f,5))+'☆'.repeat(5-Math.min(f,5));
}

function barHTML(pct, cls, label) {
    return `<div class="bar-bg"><div class="bar-fill ${cls}" style="width:${Math.min(pct,100)}%"></div><span class="bar-text">${label}</span></div>`;
}

function setStatus(id, text, cls) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.className = 'status '+(cls||''); }
}

// =========================================================================
// Navigation
// =========================================================================
function navigate(viewName) {
    state.currentView = viewName;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const view = document.getElementById('view-'+viewName);
    if (view) view.classList.add('active');
    const tab = document.querySelector(`.tab[data-view="${viewName}"]`);
    if (tab) tab.classList.add('active');

    // Load data for the view
    switch(viewName) {
        case 'home': loadHardware(); break;
        case 'recommend': loadRecommendations(); break;
        case 'models': loadLocalModels(); break;
        case 'calibration': loadCalibration(); break;
    }
}

// Tab clicks
document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => navigate(t.dataset.view));
});

// =========================================================================
// View: Home
// =========================================================================
function updateHomeGauges(hw) {
    const gpuName = hw.gpu_name || 'CPU Only';
    document.getElementById('hw-gpu-name').textContent = gpuName;
    document.getElementById('hw-summary').textContent =
        `${fmt(hw.vram_total_bytes)} VRAM | ${fmt(hw.ram_total_bytes)} RAM | ${hw.platform || 'Unknown'}`;
    document.getElementById('platform-badge').textContent = hw.platform || 'CPU';

    const vramUsed = hw.vram_total_bytes - hw.vram_free_bytes;
    const vramPct = hw.vram_total_bytes > 0 ? (vramUsed / hw.vram_total_bytes * 100) : 0;
    document.getElementById('vram-bar').style.width = vramPct+'%';
    document.getElementById('vram-bar').className = 'gauge-fill'+(vramPct>90?' danger':vramPct>70?' warn':'');
    document.getElementById('vram-text').textContent = fmt(vramUsed)+' / '+fmt(hw.vram_total_bytes);

    const ramUsed = hw.ram_total_bytes - hw.ram_free_bytes;
    const ramPct = hw.ram_total_bytes > 0 ? (ramUsed / hw.ram_total_bytes * 100) : 0;
    document.getElementById('ram-bar').style.width = ramPct+'%';
    document.getElementById('ram-bar').className = 'gauge-fill'+(ramPct>90?' danger':ramPct>70?' warn':'');
    document.getElementById('ram-text').textContent = fmt(ramUsed)+' / '+fmt(hw.ram_total_bytes);

    document.getElementById('temp-text').textContent = hw.gpu_temp_celsius > 0 ? hw.gpu_temp_celsius+'°C' : 'N/A';
    document.getElementById('bw-text').textContent = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs.toFixed(0)+' GB/s' : 'N/A';
}

async function loadHardware() {
    try {
        state.hardware = await api.getHardware();
        updateHomeGauges(state.hardware);
    } catch(e) { console.error(e); }
}

function startHWSSE() {
    if (state.hwSSE) state.hwSSE.close();
    state.hwSSE = subscribeSSE('/api/hardware/live', {
        hardware: (hw) => {
            // Only update home view gauges if on home
            if (state.currentView === 'home') {
                // Merge with existing state
                if (state.hardware) {
                    state.hardware.vram_free_bytes = hw.vram_free_bytes;
                    state.hardware.ram_free_bytes = hw.ram_free_bytes;
                    state.hardware.gpu_temp_celsius = hw.gpu_temp_celsius;
                    updateHomeGauges(state.hardware);
                }
            }
        }
    });
}

// =========================================================================
// View: Predict
// =========================================================================
async function runPredict() {
    const url = document.getElementById('predict-url').value.trim();
    if (!url) return;
    const priority = document.querySelector('input[name="priority"]:checked').value;

    setStatus('predict-status', 'Fetching metadata and predicting...', 'loading');
    document.getElementById('btn-predict').disabled = true;

    try {
        const data = await api.predict(url, priority);
        state.predictStrategies = data.strategies || [];

        // Model info
        const mi = document.getElementById('predict-model-info');
        mi.classList.remove('hidden');
        mi.innerHTML = `<div class="model-info"><strong>${data.model.name}</strong> | ${data.model.params} params | ${data.model.layers} layers | ${data.model.architecture} | ${data.model.is_moe ? 'MoE' : 'Dense'} | <em>Generated in ${data.time_ms}ms</em></div>`;

        // Strategy table
        renderPredictTable(data.strategies);
        document.getElementById('predict-results').classList.remove('hidden');
        setStatus('predict-status', `${data.strategies.length} strategies found`);
    } catch(e) {
        setStatus('predict-status', 'Error: '+e.message, 'error');
    } finally {
        document.getElementById('btn-predict').disabled = false;
    }
}

function renderPredictTable(strategies) {
    const tbody = document.getElementById('predict-body');
    tbody.innerHTML = '';
    const names = {'FULL_GPU':'Full GPU','GPU_CPU_SPLIT':'Split','CPU_ONLY':'CPU Only','HOT_COLD_SPLIT':'Hot/Cold','LAYER_STREAM':'Stream'};
    const maxTPS = Math.max(...strategies.map(s => s.tokens_per_sec || 0), 1);
    const maxVRAM = Math.max(...strategies.map(s => s.vram_bytes || 0), 1);

    strategies.forEach((s, i) => {
        const cls = s.status==='VIABLE'?'status-viable':s.status==='NO_FIT'?'status-no-fit':s.status==='TIGHT'?'status-tight':'status-viable';
        const tpsPct = (s.tokens_per_sec||0)/maxTPS*100;
        const vramPct = (s.vram_bytes||0)/maxVRAM*100;
        const tpsClass = s.placement==='FULL_GPU'?'gpu':s.placement==='GPU_CPU_SPLIT'?'split':'cpu';
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i+1}</td>
            <td><strong>${names[s.placement]||s.placement}</strong></td>
            <td class="bar-cell">${s.tokens_per_sec>0?barHTML(tpsPct,tpsClass,'~'+s.tokens_per_sec.toFixed(0)):barHTML(0,'cpu','--')}</td>
            <td class="bar-cell">${s.vram_bytes>0?barHTML(vramPct,'gpu',fmt(s.vram_bytes)):barHTML(0,'cpu','--')}</td>
            <td>${s.context_length>=1024?(s.context_length/1024)+'K':s.context_length}</td>
            <td>${s.kv_quant_bits===16?'FP16':'Q8'}</td>
            <td>${s.ttft_ms>0&&s.ttft_ms<60000?'~'+s.ttft_ms.toFixed(0)+'ms':'--'}</td>
            <td class="${cls}">${s.status}</td>
            <td>${s.viable?`<button class="btn-sm btn-primary" onclick="selectStrategy(${i})">Use</button>`:''}</td>
        `;
        tbody.appendChild(row);
    });
}

function selectStrategy(i) {
    const s = state.predictStrategies[i];
    openExecModal({
        strategy: s,
        placement: s.placement,
        gpu_layers: s.gpu_layers,
        context: s.context_length,
        kv: s.kv_quant_bits
    });
}

// =========================================================================
// View: Recommend
// =========================================================================
async function loadRecommendations() {
    const priority = document.getElementById('rec-priority').value;
    const useCase = document.getElementById('rec-usecase').value;
    const container = document.getElementById('rec-cards');
    container.innerHTML = '<div class="dim">Loading recommendations...</div>';

    try {
        const data = await api.recommend(priority, useCase);
        container.innerHTML = '';
        (data.recommendations||[]).forEach((r, i) => {
            const card = document.createElement('div');
            card.className = 'rec-card';
            card.innerHTML = `
                ${r.label?`<span class="label-badge">${r.label}</span>`:''}
                <h3>${r.model}</h3>
                <div class="meta">${r.quant} | ${r.download_gb.toFixed(1)} GB download | <span class="stars">${r.quality_stars}</span></div>
                <div class="perf">
                    <span>Strategy: <strong>${r.strategy}</strong></span>
                    <span>~${r.tokens_per_sec.toFixed(0)} tok/s</span>
                    <span>${r.vram_bytes>0?fmt(r.vram_bytes)+' VRAM':'CPU'}</span>
                </div>
                <button class="btn btn-primary btn-sm" onclick="document.getElementById('predict-url').value='${r.hf_url}';navigate('predict')">Use This →</button>
            `;
            container.appendChild(card);
        });
    } catch(e) {
        container.innerHTML = `<div class="dim">Error: ${e.message}</div>`;
    }
}

// =========================================================================
// View: Models
// =========================================================================
async function loadLocalModels() {
    try {
        const data = await api.getLocalModels();
        document.getElementById('models-summary').textContent = `${data.count} local model(s) found`;
        const list = document.getElementById('models-list');
        list.innerHTML = '';
        (data.models||[]).forEach(m => {
            const item = document.createElement('div');
            item.className = 'model-item';
            item.innerHTML = `
                <div>
                    <div class="model-name">${m.filename}</div>
                    <div class="model-meta">${fmt(m.size_bytes)} | ${m.path}</div>
                </div>
                <button class="btn btn-primary btn-sm" onclick="document.getElementById('predict-url').value='${m.path}';navigate('predict')">Analyze</button>
            `;
            list.appendChild(item);
        });
    } catch(e) {
        document.getElementById('models-summary').textContent = 'Error: '+e.message;
    }
}

// =========================================================================
// View: Calibration
// =========================================================================
async function loadCalibration() {
    try {
        const data = await api.getCalibration();
        const summary = document.getElementById('cal-summary');
        if (data.records === 0) {
            summary.innerHTML = 'No calibration data yet. Run a model execution to generate data.';
        } else {
            summary.innerHTML = `<strong>${data.matching_records}</strong> records for this hardware | Fingerprint: <code>${data.fingerprint}</code>`;
        }

        const hist = await api.getCalHistory();
        const tbody = document.getElementById('cal-body');
        tbody.innerHTML = '';
        if (hist.entries && hist.entries.length > 0) {
            document.getElementById('cal-table').classList.remove('hidden');
            hist.entries.slice(-20).reverse().forEach(e => {
                const delta = e.actual_tps > 0 ? ((e.predicted_tps - e.actual_tps) / e.actual_tps * 100).toFixed(1) : '--';
                const deltaClass = Math.abs(parseFloat(delta)) < 15 ? 'status-viable' : Math.abs(parseFloat(delta)) < 30 ? 'status-tight' : 'status-no-fit';
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${e.model_id||'--'}</td>
                    <td>${e.placement||'--'}</td>
                    <td>${e.predicted_tps>0?'~'+e.predicted_tps.toFixed(1):'--'}</td>
                    <td>${e.actual_tps>0?e.actual_tps.toFixed(1):'--'}</td>
                    <td class="${deltaClass}">${delta!=='--'?delta+'%':'--'}</td>
                    <td>${e.timestamp||'--'}</td>
                `;
                tbody.appendChild(row);
            });
        }
    } catch(e) { console.error(e); }
}

async function resetCalibration() {
    if (!confirm('Reset all calibration data? This cannot be undone.')) return;
    try {
        await api.resetCalibration();
        loadCalibration();
    } catch(e) { alert('Error: '+e.message); }
}

// =========================================================================
// Execution Modal
// =========================================================================
let execSSE = null;

async function openExecModal(config) {
    document.getElementById('exec-modal').classList.remove('hidden');
    document.getElementById('exec-output').textContent = 'Starting execution...\n';
    document.getElementById('exec-tps').textContent = '--';
    document.getElementById('exec-vram').textContent = '--';
    document.getElementById('exec-temp').textContent = '--';
    document.getElementById('exec-tokens').textContent = '0';
    document.getElementById('exec-progress').style.width = '0%';
    document.getElementById('btn-abort').classList.remove('hidden');
    document.getElementById('btn-close-exec').classList.add('hidden');

    const output = document.getElementById('exec-output');
    const totalTokens = config.max_tokens || 200;
    let tokensGenerated = 0;

    try {
        const result = await api.post('/api/execute', {
            model_path: config.model_path,
            prompt: config.prompt || 'Hello, how are you?',
            max_tokens: totalTokens
        });
        const streamUrl = result.stream_url;
        if (!streamUrl) throw new Error('No stream URL returned');

        output.textContent += 'Connected to execution stream...\n';
        execSSE = new EventSource(streamUrl);

        execSSE.addEventListener('loading', e => {
            const d = JSON.parse(e.data);
            output.textContent += d.message + '\n';
        });
        execSSE.addEventListener('loaded', e => {
            const d = JSON.parse(e.data);
            const vramGB = (d.vram_used_bytes / 1e9).toFixed(1);
            document.getElementById('exec-vram').textContent = vramGB + ' GB';
            output.textContent += `Model loaded (${(d.load_time_ms/1000).toFixed(1)}s)\n`;
        });
        execSSE.addEventListener('prefill', e => {
            const d = JSON.parse(e.data);
            output.textContent += `Prefill: ${d.ttft_ms}ms for ${d.prompt_tokens} tokens\n`;
        });
        execSSE.addEventListener('token', e => {
            const d = JSON.parse(e.data);
            output.textContent += d.token;
            tokensGenerated = d.tokens_generated;
            document.getElementById('exec-tokens').textContent = `${tokensGenerated}/${totalTokens}`;
            document.getElementById('exec-tps').textContent = d.current_tok_per_sec.toFixed(0);
            document.getElementById('exec-progress').style.width = `${Math.min(100, (tokensGenerated/totalTokens)*100)}%`;
            output.scrollTop = output.scrollHeight;
        });
        execSSE.addEventListener('hardware_sample', e => {
            const d = JSON.parse(e.data);
            document.getElementById('exec-vram').textContent = (d.vram_used_bytes / 1e9).toFixed(1) + ' GB';
            if (d.gpu_temp_celsius > 0) document.getElementById('exec-temp').textContent = d.gpu_temp_celsius + '\u00b0C';
        });
        execSSE.addEventListener('status', e => {
            const d = JSON.parse(e.data);
            output.textContent += '\n' + d.message + '\n';
            if (d.cli_command) output.textContent += `\nRun: ${d.cli_command}\n`;
        });
        execSSE.addEventListener('complete', e => {
            const d = JSON.parse(e.data);
            if (d.actual_tokens_per_sec) output.textContent += `\n\nAvg: ${d.actual_tokens_per_sec.toFixed(1)} tok/s`;
            if (d.actual_ttft_ms) output.textContent += ` | TTFT: ${d.actual_ttft_ms}ms`;
            if (d.throttled) output.textContent += ' | THROTTLED';
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
            execSSE.close(); execSSE = null;
        });
        execSSE.addEventListener('error', e => {
            output.textContent += '\n[Connection lost]\n';
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
            if (execSSE) { execSSE.close(); execSSE = null; }
        });
    } catch(err) {
        output.textContent += `\nError: ${err.message}\n`;
        document.getElementById('btn-abort').classList.add('hidden');
        document.getElementById('btn-close-exec').classList.remove('hidden');
    }
}

function closeExecModal() {
    document.getElementById('exec-modal').classList.add('hidden');
    if (execSSE) { execSSE.close(); execSSE = null; }
}

async function abortExecution() {
    closeExecModal();
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    loadHardware();
    startHWSSE();
});
