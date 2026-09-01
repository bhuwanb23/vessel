// =========================================================================
// Vessel Dashboard — Application Layer
// =========================================================================

// ╔════════════════════════════════════════════════════════════════════════╗
// ║  SINGLE TOGGLE: Set to `false` to connect to live backend API       ║
// ╚════════════════════════════════════════════════════════════════════════╝
let USE_MOCK_DATA = false;

const API = '';

// =========================================================================
// State
// =========================================================================
const state = {
    hardware: null,
    currentView: 'dashboard',
    predictStrategies: [],
    hwSSE: null,
    recommendData: [],
};

// Chart instances (so we can destroy/recreate)
const charts = {
    vramHistory: null,
    tempHistory: null,
    modelCompare: null,
    strategySpeed: null,
    strategyVram: null,
    recScatter: null,
    calScatter: null,
    calAccuracy: null,
};

// Time-series buffers for live charts
const timeSeries = {
    vram: [],
    temp: [],
    labels: [],
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
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
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
    async recommend(priority, useCase) {
        return this.get(`/api/recommend?priority=${priority}&use_case=${useCase}&top=8`);
    },
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
function fmtBytes(b) {
    if (b >= 1e12) return (b / 1e12).toFixed(1) + ' TB';
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB';
    return b + ' B';
}

function chartColors() {
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return {
        isDark,
        text: isDark ? '#94a3b8' : '#475569',
        grid: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
        surface: isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.7)',
        accent: isDark ? '#63b3ff' : '#2563eb',
        green: isDark ? '#4ade80' : '#16a34a',
        yellow: isDark ? '#fbbf24' : '#d97706',
        red: isDark ? '#f87171' : '#dc2626',
        purple: isDark ? '#a78bfa' : '#7c3aed',
        cyan: isDark ? '#22d3ee' : '#0891b2',
    };
}

function chartDefaults() {
    const c = chartColors();
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: c.isDark ? '#1a2236' : '#ffffff',
                titleColor: c.isDark ? '#f0f4f8' : '#0f172a',
                bodyColor: c.text,
                borderColor: c.grid,
                borderWidth: 1,
                padding: 10,
                titleFont: { family: 'Plus Jakarta Sans', weight: '600' },
                bodyFont: { family: 'JetBrains Mono', size: 12 },
                cornerRadius: 8,
            },
        },
        scales: {
            x: { grid: { color: c.grid, drawBorder: false }, ticks: { color: c.text, font: { family: 'Plus Jakarta Sans', size: 11 } } },
            y: { grid: { color: c.grid, drawBorder: false }, ticks: { color: c.text, font: { family: 'Plus Jakarta Sans', size: 11 } } },
        },
    };
}

// =========================================================================
// Navigation
// =========================================================================
function navigate(viewName) {
    state.currentView = viewName;
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const view = document.getElementById('view-' + viewName);
    if (view) view.classList.add('active');
    const nav = document.querySelector(`.nav-item[data-view="${viewName}"]`);
    if (nav) nav.classList.add('active');

    const titles = {
        dashboard: 'Dashboard',
        predict: 'Predict',
        recommend: 'Recommend',
        models: 'Local Models',
        calibration: 'Calibration',
        settings: 'Settings',
    };
    document.getElementById('page-title').textContent = titles[viewName] || viewName;

    switch (viewName) {
        case 'dashboard': loadHardware(); break;
        case 'recommend': loadRecommendations(); break;
        case 'models': loadLocalModels(); break;
        case 'calibration': loadCalibration(); break;
    }
    document.getElementById('sidebar').classList.remove('open');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// Priority pills
document.addEventListener('click', e => {
    const pill = e.target.closest('.pill');
    if (pill) {
        pill.closest('.priority-pills').querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
    }
});

// =========================================================================
// MOCK DATA — Rich & Realistic
// =========================================================================
const MockData = {
    hardware() {
        return {
            platform: 'nvidia',
            gpu_name: 'NVIDIA GeForce RTX 4090',
            vram_total_bytes: 24576000000,
            vram_free_bytes: 18432000000,
            vram_used_pct: 25.0,
            ram_total_bytes: 34359738368,
            ram_free_bytes: 21474836480,
            ram_used_pct: 37.5,
            gpu_bandwidth_gbs: 1008,
            gpu_tflops_fp16: 82.6,
            ram_bandwidth_gbs: 96,
            nvme_sequential_mbs: 7000,
            nvme_random_4k_mbs: 1200,
            gpu_temp_celsius: 45,
            gpu_clock_mhz: 2520,
            hardware_fingerprint: 'a1b2c3d4e5f6',
            is_unified_memory: false,
        };
    },

    hardwareLive() {
        const base = this.hardware();
        const vramFree = base.vram_free_bytes + (Math.random() - 0.5) * 1e9;
        const ramFree = base.ram_free_bytes + (Math.random() - 0.5) * 2e9;
        return {
            vram_free_bytes: Math.max(0, Math.min(base.vram_total_bytes, vramFree)),
            vram_used_bytes: base.vram_total_bytes - Math.max(0, Math.min(base.vram_total_bytes, vramFree)),
            ram_free_bytes: Math.max(0, Math.min(base.ram_total_bytes, ramFree)),
            gpu_temp_celsius: base.gpu_temp_celsius + Math.round((Math.random() - 0.5) * 4),
            gpu_clock_mhz: base.gpu_clock_mhz + Math.round((Math.random() - 0.5) * 30),
            gpu_utilization: Math.round(60 + Math.random() * 30),
        };
    },

    predict() {
        return {
            model: {
                name: 'Qwen2.5-14B-Instruct',
                params: '14B',
                layers: 81,
                architecture: 'Qwen2ForCausalLM',
                is_moe: false,
                quant: 'Q4_K_M',
                context: 32768,
            },
            hardware: this.hardware(),
            strategies: [
                { rank: 1, placement: 'FULL_GPU', gpu_layers: 81, context_length: 32768, kv_quant_bits: 8, tokens_per_sec: 142, vram_bytes: 12.8e9, ram_bytes: 0, ttft_ms: 12, viable: true, confidence: 'HIGH', status: 'VIABLE', warnings: [] },
                { rank: 2, placement: 'FULL_GPU', gpu_layers: 81, context_length: 65536, kv_quant_bits: 8, tokens_per_sec: 128, vram_bytes: 18.2e9, ram_bytes: 0, ttft_ms: 18, viable: true, confidence: 'HIGH', status: 'VIABLE', warnings: [] },
                { rank: 3, placement: 'GPU_CPU_SPLIT', gpu_layers: 45, context_length: 32768, kv_quant_bits: 8, tokens_per_sec: 67, vram_bytes: 8.1e9, ram_bytes: 4.2e9, ttft_ms: 35, viable: true, confidence: 'HIGH', status: 'VIABLE', warnings: [] },
                { rank: 4, placement: 'FULL_GPU', gpu_layers: 81, context_length: 131072, kv_quant_bits: 8, tokens_per_sec: 89, vram_bytes: 22.1e9, ram_bytes: 0, ttft_ms: 28, viable: true, confidence: 'MEDIUM', status: 'TIGHT', warnings: ['High VRAM usage'] },
                { rank: 5, placement: 'GPU_CPU_SPLIT', gpu_layers: 30, context_length: 65536, kv_quant_bits: 8, tokens_per_sec: 45, vram_bytes: 5.8e9, ram_bytes: 6.8e9, ttft_ms: 52, viable: true, confidence: 'MEDIUM', status: 'VIABLE', warnings: [] },
                { rank: 6, placement: 'CPU_ONLY', gpu_layers: 0, context_length: 32768, kv_quant_bits: 8, tokens_per_sec: 8, vram_bytes: 0, ram_bytes: 12.8e9, ttft_ms: 450, viable: true, confidence: 'HIGH', status: 'VIABLE', warnings: ['Slow'] },
                { rank: 7, placement: 'LAYER_STREAM', gpu_layers: 81, context_length: 32768, kv_quant_bits: 8, tokens_per_sec: 22, vram_bytes: 2.1e9, ram_bytes: 10.2e9, ttft_ms: 180, viable: true, confidence: 'LOW', status: 'VIABLE', warnings: ['Experimental'] },
                { rank: 8, placement: 'HOT_COLD_SPLIT', gpu_layers: 81, context_length: 32768, kv_quant_bits: 8, tokens_per_sec: 95, vram_bytes: 10.5e9, ram_bytes: 3.8e9, ttft_ms: 22, viable: true, confidence: 'MEDIUM', status: 'VIABLE', warnings: [] },
            ],
            time_ms: 342,
        };
    },

    recommendations() {
        return {
            recommendations: [
                { model: 'Qwen2.5-14B-Instruct', quant: 'Q4_K_M', download_gb: 8.9, quality_stars: '★★★★☆', strategy: 'Full GPU', tokens_per_sec: 142, vram_bytes: 12.8e9, hf_url: 'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf', label: 'Best Overall', quality_score: 8.2 },
                { model: 'Mistral-7B-Instruct-v0.3', quant: 'Q4_K_M', download_gb: 4.4, quality_stars: '★★★★☆', strategy: 'Full GPU', tokens_per_sec: 198, vram_bytes: 6.2e9, hf_url: 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf', label: 'Fastest', quality_score: 7.5 },
                { model: 'Llama-3.1-8B-Instruct', quant: 'Q4_K_M', download_gb: 4.9, quality_stars: '★★★★☆', strategy: 'Full GPU', tokens_per_sec: 185, vram_bytes: 6.8e9, hf_url: 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', label: null, quality_score: 7.8 },
                { model: 'DeepSeek-Coder-V2-Lite', quant: 'Q4_K_M', download_gb: 9.0, quality_stars: '★★★★★', strategy: 'GPU+CPU Split', tokens_per_sec: 88, vram_bytes: 9.2e9, hf_url: 'https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf', label: 'Best for Code', quality_score: 9.1 },
                { model: 'Phi-3-medium-14b', quant: 'Q4_K_M', download_gb: 8.0, quality_stars: '★★★★☆', strategy: 'Full GPU', tokens_per_sec: 135, vram_bytes: 11.2e9, hf_url: 'https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF/resolve/main/Phi-3-medium-128k-instruct-Q4_K_M.gguf', label: null, quality_score: 7.9 },
                { model: 'Gemma-2-27b-it', quant: 'Q4_K_M', download_gb: 16.2, quality_stars: '★★★★★', strategy: 'GPU+CPU Split', tokens_per_sec: 72, vram_bytes: 14.8e9, hf_url: 'https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/resolve/main/gemma-2-27b-it-Q4_K_M.gguf', label: 'Highest Quality', quality_score: 9.4 },
                { model: 'Yi-1.5-9B-Chat', quant: 'Q4_K_M', download_gb: 5.4, quality_stars: '★★★☆☆', strategy: 'Full GPU', tokens_per_sec: 175, vram_bytes: 7.2e9, hf_url: 'https://huggingface.co/01-ai/Yi-1.5-9B-Chat-GGUF/resolve/main/yi-1.5-9b-chat-q4_k_m.gguf', label: null, quality_score: 6.8 },
                { model: 'Command-R-35B', quant: 'Q4_K_M', download_gb: 19.4, quality_stars: '★★★★☆', strategy: 'GPU+CPU Split', tokens_per_sec: 58, vram_bytes: 18.2e9, hf_url: 'https://huggingface.co/bartowski/Command-R-35B-GGUF/resolve/main/Command-R-35B-Q4_K_M.gguf', label: null, quality_score: 8.5 },
            ],
        };
    },

    localModels() {
        return {
            count: 3,
            models: [
                { filename: 'qwen2.5-14b-instruct-q4_k_m.gguf', size_bytes: 8.9e9, path: 'C:\\Users\\user\\.vessel\\models\\qwen2.5-14b-instruct-q4_k_m.gguf' },
                { filename: 'mistral-7b-instruct-v0.3.Q4_K_M.gguf', size_bytes: 4.4e9, path: 'C:\\Users\\user\\.vessel\\models\\mistral-7b-instruct-v0.3.Q4_K_M.gguf' },
                { filename: 'llama-3.1-8b-instruct-q4_k_m.gguf', size_bytes: 4.9e9, path: 'C:\\Users\\user\\.vessel\\models\\llama-3.1-8b-instruct-q4_k_m.gguf' },
            ],
        };
    },

    calibration() {
        return { records: 12, matching_records: 12, fingerprint: 'a1b2c3d4e5f6' };
    },

    calibrationHistory() {
        return {
            entries: [
                { model_id: 'Qwen2.5-14B', placement: 'FULL_GPU', predicted_tps: 142, actual_tps: 138, timestamp: '2026-08-30 14:22' },
                { model_id: 'Mistral-7B', placement: 'FULL_GPU', predicted_tps: 198, actual_tps: 205, timestamp: '2026-08-30 13:45' },
                { model_id: 'Llama-3.1-8B', placement: 'FULL_GPU', predicted_tps: 185, actual_tps: 178, timestamp: '2026-08-29 16:10' },
                { model_id: 'DeepSeek-Coder-V2', placement: 'GPU_CPU_SPLIT', predicted_tps: 88, actual_tps: 82, timestamp: '2026-08-29 11:30' },
                { model_id: 'Qwen2.5-14B', placement: 'FULL_GPU', predicted_tps: 128, actual_tps: 131, timestamp: '2026-08-28 20:15' },
                { model_id: 'Phi-3-medium', placement: 'FULL_GPU', predicted_tps: 135, actual_tps: 128, timestamp: '2026-08-28 15:00' },
                { model_id: 'Gemma-2-27b', placement: 'GPU_CPU_SPLIT', predicted_tps: 72, actual_tps: 68, timestamp: '2026-08-27 19:22' },
                { model_id: 'Mistral-7B', placement: 'FULL_GPU', predicted_tps: 198, actual_tps: 201, timestamp: '2026-08-27 10:45' },
                { model_id: 'Llama-3.1-8B', placement: 'FULL_GPU', predicted_tps: 185, actual_tps: 180, timestamp: '2026-08-26 14:30' },
                { model_id: 'Qwen2.5-14B', placement: 'GPU_CPU_SPLIT', predicted_tps: 67, actual_tps: 64, timestamp: '2026-08-26 09:15' },
                { model_id: 'Yi-1.5-9B', placement: 'FULL_GPU', predicted_tps: 175, actual_tps: 168, timestamp: '2026-08-25 18:00' },
                { model_id: 'Command-R-35B', placement: 'GPU_CPU_SPLIT', predicted_tps: 58, actual_tps: 55, timestamp: '2026-08-25 11:20' },
            ],
        };
    },
};

// =========================================================================
// Dashboard View
// =========================================================================
function updateDashboard(hw) {
    const gpuName = hw.gpu_name || 'CPU Only';
    document.getElementById('hero-gpu-name').textContent = gpuName;
    document.getElementById('hero-gpu-detail').textContent = hw.gpu_tflops_fp16 > 0 ? `${hw.gpu_tflops_fp16} TFLOPS` : 'Integrated';

    // VRAM gauge
    const vramUsed = hw.vram_total_bytes - hw.vram_free_bytes;
    const vramPct = hw.vram_total_bytes > 0 ? (vramUsed / hw.vram_total_bytes * 100) : 0;
    const vramFill = document.getElementById('hero-vram-fill');
    vramFill.style.width = vramPct + '%';
    vramFill.className = 'stat-gauge-fill' + (vramPct > 90 ? ' danger' : vramPct > 70 ? ' warn' : '');
    document.getElementById('hero-vram-text').textContent = `${fmtBytes(vramUsed)} / ${fmtBytes(hw.vram_total_bytes)}`;

    // RAM gauge
    const ramUsed = hw.ram_total_bytes - hw.ram_free_bytes;
    const ramPct = hw.ram_total_bytes > 0 ? (ramUsed / hw.ram_total_bytes * 100) : 0;
    const ramFill = document.getElementById('hero-ram-fill');
    ramFill.style.width = ramPct + '%';
    ramFill.className = 'stat-gauge-fill' + (ramPct > 90 ? ' danger' : ramPct > 70 ? ' warn' : '');
    document.getElementById('hero-ram-text').textContent = `${fmtBytes(ramUsed)} / ${fmtBytes(hw.ram_total_bytes)}`;

    // Temperature
    const temp = hw.gpu_temp_celsius;
    document.getElementById('hero-temp').textContent = temp > 0 ? temp + '°' : '—';
    const tempStatus = document.getElementById('hero-temp-status');
    if (temp > 80) { tempStatus.textContent = 'Critical'; tempStatus.style.color = 'var(--red)'; }
    else if (temp > 65) { tempStatus.textContent = 'Warm'; tempStatus.style.color = 'var(--yellow)'; }
    else { tempStatus.textContent = 'Normal'; tempStatus.style.color = 'var(--green)'; }

    // Platform badge
    document.getElementById('platform-badge').textContent = hw.platform?.toUpperCase() || 'CPU';
    document.getElementById('platform-text').textContent = hw.platform?.toUpperCase() || 'CPU';
    document.getElementById('hw-platform-badge').textContent = hw.platform?.toUpperCase() || 'CPU';

    // Hardware specs
    document.getElementById('hw-bw').textContent = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs + ' GB/s' : 'N/A';
    document.getElementById('hw-tflops').textContent = hw.gpu_tflops_fp16 > 0 ? hw.gpu_tflops_fp16.toFixed(1) : 'N/A';
    document.getElementById('hw-ram-bw').textContent = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs + ' GB/s' : 'N/A';
    document.getElementById('hw-nvme-seq').textContent = hw.nvme_sequential_mbs > 0 ? hw.nvme_sequential_mbs + ' MB/s' : 'N/A';
    document.getElementById('hw-nvme-rnd').textContent = hw.nvme_random_4k_mbs > 0 ? hw.nvme_random_4k_mbs + ' MB/s' : 'N/A';
    document.getElementById('hw-unified').textContent = hw.is_unified_memory ? 'Yes (Apple Silicon)' : 'No';

    // Sidebar status
    document.getElementById('hw-status-mini').querySelector('.hw-status-text').textContent = hw.gpu_name || 'CPU Only';

    // Push to time-series
    pushTimeSeries(vramUsed, temp);
}

async function loadHardware() {
    try {
        if (USE_MOCK_DATA) {
            state.hardware = MockData.hardware();
        } else {
            state.hardware = await api.getHardware();
        }
        updateDashboard(state.hardware);
    } catch (e) {
        console.error('Failed to load hardware:', e);
    }
}

function startHWSSE() {
    if (USE_MOCK_DATA) return;
    if (state.hwSSE) state.hwSSE.close();
    state.hwSSE = subscribeSSE('/api/hardware/live', {
        hardware: (hw) => {
            if (state.currentView === 'dashboard' && state.hardware) {
                state.hardware.vram_free_bytes = hw.vram_free_bytes;
                state.hardware.ram_free_bytes = hw.ram_free_bytes;
                state.hardware.gpu_temp_celsius = hw.gpu_temp_celsius;
                updateDashboard(state.hardware);
            }
        },
    });
}

function startMockSSE() {
    if (!USE_MOCK_DATA) return;
    setInterval(() => {
        if (state.currentView === 'dashboard' && state.hardware) {
            const live = MockData.hardwareLive();
            state.hardware.vram_free_bytes = live.vram_free_bytes;
            state.hardware.ram_free_bytes = live.ram_free_bytes;
            state.hardware.gpu_temp_celsius = live.gpu_temp_celsius;
            updateDashboard(state.hardware);
        }
    }, 2000);
}

// =========================================================================
// Charts — Dashboard
// =========================================================================
function pushTimeSeries(vramUsed, temp) {
    const now = new Date();
    const label = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    timeSeries.labels.push(label);
    timeSeries.vram.push(parseFloat((vramUsed / 1e9).toFixed(1)));
    timeSeries.temp.push(temp);
    if (timeSeries.labels.length > 30) {
        timeSeries.labels.shift();
        timeSeries.vram.shift();
        timeSeries.temp.shift();
    }
    renderDashboardCharts();
}

function renderDashboardCharts() {
    const c = chartColors();
    const defaults = chartDefaults();

    // VRAM History
    const vramEl = document.getElementById('chart-vram-history');
    if (vramEl) {
        if (charts.vramHistory) charts.vramHistory.destroy();
        charts.vramHistory = new Chart(vramEl, {
            type: 'line',
            data: {
                labels: timeSeries.labels,
                datasets: [{
                    label: 'VRAM Used (GB)',
                    data: timeSeries.vram,
                    borderColor: c.green,
                    backgroundColor: c.isDark ? 'rgba(74,222,128,0.08)' : 'rgba(22,163,74,0.06)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }],
            },
            options: {
                ...defaults,
                scales: {
                    ...defaults.scales,
                    y: { ...defaults.scales.y, min: 0, max: 24, ticks: { ...defaults.scales.y.ticks, callback: v => v + ' GB' } },
                    x: { ...defaults.scales.x, ticks: { ...defaults.scales.x.ticks, maxTicksLimit: 6 } },
                },
                plugins: {
                    ...defaults.plugins,
                    tooltip: { ...defaults.plugins.tooltip, callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + ' GB' } },
                },
            },
        });
    }

    // Temperature History
    const tempEl = document.getElementById('chart-temp-history');
    if (tempEl) {
        if (charts.tempHistory) charts.tempHistory.destroy();
        const tempColor = timeSeries.temp.some(t => t > 75) ? c.red : c.yellow;
        charts.tempHistory = new Chart(tempEl, {
            type: 'line',
            data: {
                labels: timeSeries.labels,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: timeSeries.temp,
                    borderColor: tempColor,
                    backgroundColor: c.isDark ? 'rgba(251,191,36,0.08)' : 'rgba(217,119,6,0.06)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                }],
            },
            options: {
                ...defaults,
                scales: {
                    ...defaults.scales,
                    y: { ...defaults.scales.y, min: 30, max: 90, ticks: { ...defaults.scales.y.ticks, callback: v => v + '°' } },
                    x: { ...defaults.scales.x, ticks: { ...defaults.scales.x.ticks, maxTicksLimit: 6 } },
                },
                plugins: {
                    ...defaults.plugins,
                    tooltip: { ...defaults.plugins.tooltip, callbacks: { label: ctx => ctx.parsed.y + '°C' } },
                },
            },
        });
    }

    // Model Speed Comparison (bar chart)
    if (!charts.modelCompare && document.getElementById('chart-model-compare')) {
        const modelNames = ['Qwen2.5-14B', 'Mistral-7B', 'Llama-3.1-8B', 'DeepSeek-Coder', 'Phi-3-14B', 'Gemma-2-27B', 'Yi-1.5-9B', 'Command-R-35B'];
        const speeds = [142, 198, 185, 88, 135, 72, 175, 58];
        const barColors = speeds.map(s => s > 150 ? c.green : s > 100 ? c.accent : s > 70 ? c.yellow : c.red);
        charts.modelCompare = new Chart(document.getElementById('chart-model-compare'), {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'tok/s',
                    data: speeds,
                    backgroundColor: barColors.map(col => col + '40'),
                    borderColor: barColors,
                    borderWidth: 1,
                    borderRadius: 6,
                    borderSkipped: false,
                }],
            },
            options: {
                ...defaults,
                indexAxis: 'y',
                scales: {
                    x: { ...defaults.scales.x, ticks: { ...defaults.scales.x.ticks, callback: v => v + ' tok/s' } },
                    y: { ...defaults.scales.y, ticks: { ...defaults.scales.y.ticks, font: { family: 'Plus Jakarta Sans', size: 11, weight: '500' } } },
                },
                plugins: {
                    ...defaults.plugins,
                    tooltip: { ...defaults.plugins.tooltip, callbacks: { label: ctx => ctx.parsed.x + ' tok/s' } },
                },
            },
        });
    }
}

// =========================================================================
// Predict View
// =========================================================================
async function runPredict() {
    const url = document.getElementById('predict-url').value.trim();
    if (!url) return;
    const priority = document.querySelector('input[name="priority"]:checked').value;

    const statusEl = document.getElementById('predict-status');
    statusEl.textContent = 'Fetching metadata and predicting strategies...';
    statusEl.className = 'status-bar visible loading';
    document.getElementById('btn-predict').disabled = true;

    try {
        let data;
        if (USE_MOCK_DATA) {
            await new Promise(r => setTimeout(r, 800));
            data = MockData.predict();
        } else {
            data = await api.predict(url, priority);
        }

        state.predictStrategies = data.strategies || [];

        // Model info
        const mi = document.getElementById('predict-model-info');
        mi.classList.remove('hidden');
        mi.innerHTML = `
            <div class="model-info">
                <span class="model-info-tag"><strong>Model</strong> ${data.model.name}</span>
                <span class="model-info-tag"><strong>Params</strong> ${data.model.params}</span>
                <span class="model-info-tag"><strong>Layers</strong> ${data.model.layers}</span>
                <span class="model-info-tag"><strong>Arch</strong> ${data.model.architecture}</span>
                ${data.model.is_moe ? '<span class="model-info-tag"><strong>MoE</strong></span>' : ''}
                <span class="model-info-tag"><strong>Time</strong> ${data.time_ms}ms</span>
            </div>
        `;

        renderPredictTable(data.strategies);
        renderPredictCharts(data.strategies);
        document.getElementById('predict-results').classList.remove('hidden');
        document.getElementById('strategy-count').textContent = data.strategies.length + ' found';
        statusEl.textContent = `${data.strategies.length} strategies found in ${data.time_ms}ms`;
        statusEl.className = 'status-bar visible success';
    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        statusEl.className = 'status-bar visible error';
    } finally {
        document.getElementById('btn-predict').disabled = false;
    }
}

function renderPredictTable(strategies) {
    const tbody = document.getElementById('predict-body');
    tbody.innerHTML = '';
    const names = {
        FULL_GPU: 'Full GPU', GPU_CPU_SPLIT: 'GPU + CPU Split', CPU_ONLY: 'CPU Only',
        HOT_COLD_SPLIT: 'Hot/Cold Split', LAYER_STREAM: 'Layer Stream',
    };
    const maxTPS = Math.max(...strategies.map(s => s.tokens_per_sec || 0), 1);
    const maxVRAM = Math.max(...strategies.map(s => s.vram_bytes || 0), 1);

    strategies.forEach((s, i) => {
        const cls = s.status === 'VIABLE' ? 'status-viable' : s.status === 'NO_FIT' ? 'status-no-fit' : s.status === 'TIGHT' ? 'status-tight' : 'status-low-conf';
        const tpsPct = (s.tokens_per_sec || 0) / maxTPS * 100;
        const vramPct = (s.vram_bytes || 0) / maxVRAM * 100;
        const tpsClass = s.placement === 'FULL_GPU' ? 'gpu' : s.placement === 'GPU_CPU_SPLIT' ? 'split' : 'cpu';
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight:600;color:var(--text-tertiary)">${i + 1}</td>
            <td><strong>${names[s.placement] || s.placement}</strong></td>
            <td class="bar-cell">${s.tokens_per_sec > 0 ? `<div class="bar-bg"><div class="bar-fill ${tpsClass}" style="width:${tpsPct}%"></div><span class="bar-text">~${s.tokens_per_sec.toFixed(0)}</span></div>` : '<span style="color:var(--text-muted)">—</span>'}</td>
            <td class="bar-cell">${s.vram_bytes > 0 ? `<div class="bar-bg"><div class="bar-fill gpu" style="width:${vramPct}%"></div><span class="bar-text">${fmtBytes(s.vram_bytes)}</span></div>` : '<span style="color:var(--text-muted)">—</span>'}</td>
            <td style="font-family:var(--font-mono)">${s.context_length >= 1024 ? (s.context_length / 1024) + 'K' : s.context_length}</td>
            <td style="font-family:var(--font-mono)">${s.kv_quant_bits === 16 ? 'FP16' : 'Q' + s.kv_quant_bits}</td>
            <td style="font-family:var(--font-mono)">${s.ttft_ms > 0 && s.ttft_ms < 60000 ? '~' + s.ttft_ms + 'ms' : '—'}</td>
            <td class="${cls}">${s.status}</td>
            <td>${s.viable ? `<button class="btn btn-primary btn-sm" onclick="selectStrategy(${i})">Use</button>` : ''}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderPredictCharts(strategies) {
    const c = chartColors();
    const defaults = chartDefaults();
    const names = strategies.map(s => {
        const n = { FULL_GPU: 'GPU', GPU_CPU_SPLIT: 'Split', CPU_ONLY: 'CPU', HOT_COLD_SPLIT: 'Hot/Cold', LAYER_STREAM: 'Stream' };
        return n[s.placement] || s.placement;
    });

    // Speed chart
    const speedEl = document.getElementById('chart-strategy-speed');
    if (speedEl) {
        if (charts.strategySpeed) charts.strategySpeed.destroy();
        const speedColors = strategies.map(s => s.placement === 'FULL_GPU' ? c.green : s.placement === 'GPU_CPU_SPLIT' ? c.yellow : s.placement === 'HOT_COLD_SPLIT' ? c.purple : s.placement === 'LAYER_STREAM' ? c.cyan : c.red);
        charts.strategySpeed = new Chart(speedEl, {
            type: 'bar',
            data: {
                labels: names,
                datasets: [{ label: 'tok/s', data: strategies.map(s => s.tokens_per_sec), backgroundColor: speedColors.map(col => col + '50'), borderColor: speedColors, borderWidth: 1, borderRadius: 4, borderSkipped: false }],
            },
            options: {
                ...defaults,
                plugins: { ...defaults.plugins, title: { display: true, text: 'Speed (tok/s)', color: c.text, font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' } } },
                scales: { ...defaults.scales, y: { ...defaults.scales.y, beginAtZero: true } },
            },
        });
    }

    // VRAM chart
    const vramEl = document.getElementById('chart-strategy-vram');
    if (vramEl) {
        if (charts.strategyVram) charts.strategyVram.destroy();
        const vramGB = strategies.map(s => parseFloat((s.vram_bytes / 1e9).toFixed(1)));
        const vramColors = strategies.map(s => s.vram_bytes / 1e9 > 20 ? c.red : s.vram_bytes / 1e9 > 15 ? c.yellow : c.green);
        charts.strategyVram = new Chart(vramEl, {
            type: 'bar',
            data: {
                labels: names,
                datasets: [{ label: 'VRAM (GB)', data: vramGB, backgroundColor: vramColors.map(col => col + '50'), borderColor: vramColors, borderWidth: 1, borderRadius: 4, borderSkipped: false }],
            },
            options: {
                ...defaults,
                plugins: { ...defaults.plugins, title: { display: true, text: 'VRAM Usage (GB)', color: c.text, font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' } } },
                scales: { ...defaults.scales, y: { ...defaults.scales.y, beginAtZero: true, max: 24 } },
            },
        });
    }
}

function selectStrategy(i) {
    const s = state.predictStrategies[i];
    openExecModal({ strategy: s, placement: s.placement, gpu_layers: s.gpu_layers, context: s.context_length, kv: s.kv_quant_bits });
}

// =========================================================================
// Recommend View
// =========================================================================
async function loadRecommendations() {
    const priority = document.getElementById('rec-priority').value;
    const useCase = document.getElementById('rec-usecase').value;
    const container = document.getElementById('rec-cards');
    container.innerHTML = '<div class="empty-state"><p>Loading recommendations...</p></div>';

    try {
        let data;
        if (USE_MOCK_DATA) {
            await new Promise(r => setTimeout(r, 400));
            data = MockData.recommendations();
        } else {
            data = await api.recommend(priority, useCase);
        }

        state.recommendData = data.recommendations || [];
        container.innerHTML = '';

        state.recommendData.forEach((r) => {
            const card = document.createElement('div');
            card.className = 'rec-card';
            card.innerHTML = `
                ${r.label ? `<span class="label-badge">${r.label}</span>` : ''}
                <h3>${r.model}</h3>
                <div class="meta">${r.quant} · ${r.download_gb.toFixed(1)} GB download · <span class="stars">${r.quality_stars}</span></div>
                <div class="perf">
                    <span>Strategy: <strong>${r.strategy}</strong></span>
                    <span>~${r.tokens_per_sec.toFixed(0)} tok/s</span>
                    <span>${r.vram_bytes > 0 ? fmtBytes(r.vram_bytes) + ' VRAM' : 'CPU'}</span>
                </div>
                <div class="action-row">
                    <button class="btn btn-primary btn-sm" onclick="downloadAndRun('${r.hf_url}','${r.model}')">⬇ Download & Run</button>
                    <button class="btn btn-ghost btn-sm" onclick="document.getElementById('predict-url').value='${r.hf_url}';navigate('predict')">Details</button>
                </div>
            `;
            container.appendChild(card);
        });

        renderRecommendChart(state.recommendData);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><p>Error: ${e.message}</p></div>`;
    }
}

function renderRecommendChart(recommendations) {
    const c = chartColors();
    const defaults = chartDefaults();
    const el = document.getElementById('chart-rec-scatter');
    if (!el) return;

    if (charts.recScatter) charts.recScatter.destroy();

    charts.recScatter = new Chart(el, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Models',
                data: recommendations.map(r => ({
                    x: r.tokens_per_sec,
                    y: r.quality_score || 7,
                    label: r.model,
                })),
                backgroundColor: recommendations.map((_, i) => [c.accent, c.green, c.purple, c.yellow, c.cyan, c.red, '#fb923c', '#e879f9'][i % 8] + '90'),
                borderColor: recommendations.map((_, i) => [c.accent, c.green, c.purple, c.yellow, c.cyan, c.red, '#fb923c', '#e879f9'][i % 8]),
                borderWidth: 2,
                pointRadius: 8,
                pointHoverRadius: 11,
            }],
        },
        options: {
            ...defaults,
            plugins: {
                ...defaults.plugins,
                legend: { display: false },
                tooltip: {
                    ...defaults.plugins.tooltip,
                    callbacks: {
                        title: () => '',
                        label: ctx => {
                            const r = recommendations[ctx.dataIndex];
                            return [r.model, `Speed: ${r.tokens_per_sec} tok/s`, `Quality: ${r.quality_score}/10`];
                        },
                    },
                },
            },
            scales: {
                x: { ...defaults.scales.x, title: { display: true, text: 'Speed (tok/s)', color: c.text, font: { family: 'Plus Jakarta Sans' } }, min: 0 },
                y: { ...defaults.scales.y, title: { display: true, text: 'Quality Score', color: c.text, font: { family: 'Plus Jakarta Sans' } }, min: 5, max: 10 },
            },
        },
    });
}

// =========================================================================
// Download & Run Flow
// =========================================================================
downloadAndRun = async function (url, modelName) {
    const output = document.getElementById('exec-output');
    document.getElementById('exec-modal').classList.remove('hidden');
    output.textContent = `Starting download: ${modelName}\n`;
    document.getElementById('exec-tps').textContent = '--';
    document.getElementById('exec-vram').textContent = '--';
    document.getElementById('exec-temp').textContent = '--';
    document.getElementById('exec-tokens').textContent = '0';
    document.getElementById('exec-progress').style.width = '0%';
    document.getElementById('btn-abort').classList.remove('hidden');
    document.getElementById('btn-close-exec').classList.add('hidden');

    try {
        const result = await api.startDownload(url);
        const streamUrl = result.stream_url;
        output.textContent += `Download started (task: ${result.task_id})\n`;
        const dlSSE = new EventSource(API + streamUrl);
        dlSSE.addEventListener('progress', e => {
            const d = JSON.parse(e.data);
            const pct = d.bytes_total > 0 ? (d.bytes_downloaded / d.bytes_total * 100).toFixed(1) : 0;
            const dlGB = (d.bytes_downloaded / 1e9).toFixed(2);
            const totalGB = (d.bytes_total / 1e9).toFixed(2);
            const speed = d.speed_mbs ? d.speed_mbs.toFixed(1) : '0';
            const eta = d.eta_seconds > 60 ? Math.ceil(d.eta_seconds / 60) + 'm' : d.eta_seconds + 's';
            document.getElementById('exec-progress').style.width = pct + '%';
            document.getElementById('exec-tokens').textContent = `${dlGB}/${totalGB} GB`;
            document.getElementById('exec-tps').textContent = speed + ' MB/s';
            document.getElementById('exec-temp').textContent = eta;
        });
        dlSSE.addEventListener('complete', e => {
            dlSSE.close();
            const d = JSON.parse(e.data);
            output.textContent += `\nDownload complete: ${d.local_path}\nStarting execution...\n`;
            startExecution(d.local_path, output);
        });
        dlSSE.addEventListener('error', e => {
            dlSSE.close();
            const d = e.data ? JSON.parse(e.data) : {};
            output.textContent += `\nDownload failed: ${d.message || 'Unknown error'}\n`;
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
        });
    } catch (err) {
        output.textContent += `\nError: ${err.message}\n`;
        document.getElementById('btn-abort').classList.add('hidden');
        document.getElementById('btn-close-exec').classList.remove('hidden');
    }
};

async function startExecution(localPath, outputEl) {
    const output = outputEl || document.getElementById('exec-output');
    try {
        const exResult = await api.startExecute(localPath, null, 'Hello, how are you?', 200);
        const exSSE = new EventSource(API + exResult.stream_url);
        exSSE.addEventListener('token', e => {
            const td = JSON.parse(e.data);
            // Backend sends {tokens_generated, current_tok_per_sec} — no individual token text
            document.getElementById('exec-tokens').textContent = td.tokens_generated + '/200';
            document.getElementById('exec-tps').textContent = td.current_tok_per_sec ? td.current_tok_per_sec.toFixed(0) : '--';
            document.getElementById('exec-progress').style.width = Math.min(100, (td.tokens_generated / 200) * 100) + '%';
        });
        exSSE.addEventListener('complete', e => {
            exSSE.close();
            const td = JSON.parse(e.data);
            if (td.generated_text) output.textContent += td.generated_text;
            if (td.actual_tokens_per_sec) output.textContent += `\n\nAvg: ${td.actual_tokens_per_sec.toFixed(1)} tok/s`;
            if (td.actual_ttft_ms) output.textContent += ` | TTFT: ${td.actual_ttft_ms}ms`;
            if (td.peak_vram_bytes) document.getElementById('exec-vram').textContent = (td.peak_vram_bytes / 1e9).toFixed(1);
            output.scrollTop = output.scrollHeight;
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
        });
        exSSE.addEventListener('error', e => {
            exSSE.close();
            const d = e.data ? JSON.parse(e.data) : {};
            if (d.message) output.textContent += `\nError: ${d.message}\n`;
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
        });
    } catch (err) {
        output.textContent += `\nError: ${err.message}\n`;
        document.getElementById('btn-abort').classList.add('hidden');
        document.getElementById('btn-close-exec').classList.remove('hidden');
    }
}

// =========================================================================
// Models View
// =========================================================================
async function loadLocalModels() {
    try {
        let data;
        if (USE_MOCK_DATA) { await new Promise(r => setTimeout(r, 300)); data = MockData.localModels(); } else { data = await api.getLocalModels(); }
        document.getElementById('models-count').textContent = data.count;
        const list = document.getElementById('models-list');
        list.innerHTML = '';
        if (data.models.length === 0) { list.innerHTML = '<div class="empty-state"><p>No local models found.</p></div>'; return; }
        data.models.forEach(m => {
            const item = document.createElement('div');
            item.className = 'model-item';
            item.innerHTML = `<div><div class="model-name">${m.filename}</div><div class="model-meta">${fmtBytes(m.size_bytes)} · ${m.path}</div></div><button class="btn btn-primary btn-sm" onclick="document.getElementById('predict-url').value='${m.path}';navigate('predict')">Analyze</button>`;
            list.appendChild(item);
        });
    } catch (e) { document.getElementById('models-count').textContent = '0'; }
}

// =========================================================================
// Calibration View
// =========================================================================
async function loadCalibration() {
    try {
        let calData, histData;
        if (USE_MOCK_DATA) { await new Promise(r => setTimeout(r, 300)); calData = MockData.calibration(); histData = MockData.calibrationHistory(); } else { calData = await api.getCalibration(); histData = await api.getCalHistory(); }
        const summary = document.getElementById('cal-summary');
        summary.innerHTML = calData.records === 0 ? 'No calibration data yet.' : `<strong>${calData.matching_records}</strong> records · Fingerprint: <code>${calData.fingerprint}</code>`;

        const tbody = document.getElementById('cal-body');
        tbody.innerHTML = '';
        const entries = histData.entries || [];

        if (entries.length > 0) {
            // Table
            document.getElementById('cal-table-container').classList.remove('hidden');
            entries.slice(-20).reverse().forEach(e => {
                const delta = e.actual_tps > 0 ? ((e.predicted_tps - e.actual_tps) / e.actual_tps * 100).toFixed(1) : '--';
                const absDelta = Math.abs(parseFloat(delta));
                const deltaClass = absDelta < 10 ? 'status-viable' : absDelta < 20 ? 'status-tight' : 'status-no-fit';
                const row = document.createElement('tr');
                row.innerHTML = `<td style="font-weight:600">${e.model_id}</td><td>${e.placement}</td><td style="font-family:var(--font-mono)">~${e.predicted_tps.toFixed(1)}</td><td style="font-family:var(--font-mono)">${e.actual_tps.toFixed(1)}</td><td class="${deltaClass}" style="font-family:var(--font-mono)">${delta}%</td><td style="color:var(--text-tertiary)">${e.timestamp}</td>`;
                tbody.appendChild(row);
            });

            // Scatter chart
            const valid = entries.filter(e => e.predicted_tps > 0 && e.actual_tps > 0);
            if (valid.length > 0) {
                document.getElementById('cal-chart-container').classList.remove('hidden');
                renderCalScatter(valid);
            }

            // Accuracy over time
            document.getElementById('cal-accuracy-container').classList.remove('hidden');
            renderCalAccuracy(entries);
        }
    } catch (e) { console.error(e); }
}

function renderCalScatter(valid) {
    const c = chartColors();
    const defaults = chartDefaults();
    const el = document.getElementById('cal-chart');
    if (!el) return;
    if (charts.calScatter) charts.calScatter.destroy();
    const maxVal = Math.max(...valid.map(e => Math.max(e.predicted_tps, e.actual_tps))) * 1.2;
    charts.calScatter = new Chart(el, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Predicted vs Actual', data: valid.map(e => ({ x: e.predicted_tps, y: e.actual_tps })), backgroundColor: c.accent + '90', borderColor: c.accent, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 },
                { label: 'Perfect', data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }], type: 'line', borderColor: c.text + '40', borderDash: [5, 5], pointRadius: 0, borderWidth: 1, fill: false },
            ],
        },
        options: {
            ...defaults,
            plugins: { ...defaults.plugins, legend: { display: true, labels: { color: c.text, font: { family: 'Plus Jakarta Sans' } } } },
            scales: {
                x: { ...defaults.scales.x, title: { display: true, text: 'Predicted tok/s', color: c.text, font: { family: 'Plus Jakarta Sans' } } },
                y: { ...defaults.scales.y, title: { display: true, text: 'Actual tok/s', color: c.text, font: { family: 'Plus Jakarta Sans' } } },
            },
        },
    });
}

function renderCalAccuracy(entries) {
    const c = chartColors();
    const defaults = chartDefaults();
    const el = document.getElementById('cal-accuracy-chart');
    if (!el) return;
    if (charts.calAccuracy) charts.calAccuracy.destroy();
    const deltas = entries.map(e => e.actual_tps > 0 ? Math.abs((e.predicted_tps - e.actual_tps) / e.actual_tps * 100) : 0);
    const labels = entries.map(e => e.model_id);
    const colors = deltas.map(d => d < 5 ? c.green : d < 15 ? c.yellow : c.red);
    charts.calAccuracy = new Chart(el, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ label: 'Error %', data: deltas.map(d => parseFloat(d.toFixed(1))), backgroundColor: colors.map(col => col + '50'), borderColor: colors, borderWidth: 1, borderRadius: 4, borderSkipped: false }],
        },
        options: {
            ...defaults,
            scales: { ...defaults.scales, y: { ...defaults.scales.y, beginAtZero: true, ticks: { ...defaults.scales.y.ticks, callback: v => v + '%' } } },
            plugins: { ...defaults.plugins, tooltip: { ...defaults.plugins.tooltip, callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + '% error' } } },
        },
    });
}

async function resetCalibration() {
    if (!confirm('Reset all calibration data?')) return;
    try { if (!USE_MOCK_DATA) await api.resetCalibration(); loadCalibration(); } catch (e) { alert('Error: ' + e.message); }
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
    try {
        const result = await api.post('/api/execute', { model_path: config.model_path, prompt: config.prompt || 'Hello, how are you?', max_tokens: 200 });
        const streamUrl = result.stream_url;
        if (!streamUrl) throw new Error('No stream URL returned');
        output.textContent += 'Connected to execution stream...\n';
        execSSE = new EventSource(API + streamUrl);
        execSSE.addEventListener('loading', e => {
            const d = JSON.parse(e.data);
            output.textContent += d.message + '\n';
        });
        execSSE.addEventListener('token', e => {
            const d = JSON.parse(e.data);
            // Backend sends {tokens_generated, current_tok_per_sec} — no individual token text
            document.getElementById('exec-tokens').textContent = d.tokens_generated + '/200';
            document.getElementById('exec-tps').textContent = d.current_tok_per_sec ? d.current_tok_per_sec.toFixed(0) : '--';
            document.getElementById('exec-progress').style.width = Math.min(100, (d.tokens_generated / 200) * 100) + '%';
        });
        execSSE.addEventListener('complete', e => {
            const d = JSON.parse(e.data);
            if (d.generated_text) output.textContent += d.generated_text;
            if (d.actual_tokens_per_sec) output.textContent += `\n\nAvg: ${d.actual_tokens_per_sec.toFixed(1)} tok/s`;
            if (d.actual_ttft_ms) output.textContent += ` | TTFT: ${d.actual_ttft_ms}ms`;
            if (d.throttled) output.textContent += ' | THROTTLED';
            if (d.peak_vram_bytes) document.getElementById('exec-vram').textContent = (d.peak_vram_bytes / 1e9).toFixed(1);
            output.scrollTop = output.scrollHeight;
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
            execSSE.close(); execSSE = null;
        });
        execSSE.addEventListener('error', e => {
            const d = e.data ? JSON.parse(e.data) : {};
            if (d.message) output.textContent += `\nError: ${d.message}\n`;
            else output.textContent += '\n[Connection lost]\n';
            document.getElementById('btn-abort').classList.add('hidden');
            document.getElementById('btn-close-exec').classList.remove('hidden');
            if (execSSE) { execSSE.close(); execSSE = null; }
        });
    } catch (err) {
        output.textContent += `\nError: ${err.message}\n`;
        document.getElementById('btn-abort').classList.add('hidden');
        document.getElementById('btn-close-exec').classList.remove('hidden');
    }
}

function closeExecModal() { document.getElementById('exec-modal').classList.add('hidden'); if (execSSE) { execSSE.close(); execSSE = null; } }
async function abortExecution() { closeExecModal(); }

// =========================================================================
// Settings
// =========================================================================
function toggleMockMode(enabled) {
    USE_MOCK_DATA = enabled;
    const label = document.getElementById('data-source-label');
    const badge = document.getElementById('data-source-badge');
    if (enabled) { label.textContent = 'Mock Data (Development)'; badge.textContent = 'DEV'; badge.className = 'setting-badge dev'; startMockSSE(); }
    else { label.textContent = 'Live API (Production)'; badge.textContent = 'LIVE'; badge.className = 'setting-badge prod'; startHWSSE(); }
    navigate(state.currentView);
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('mock-toggle');
    if (toggle) toggle.checked = USE_MOCK_DATA;
    toggleMockMode(USE_MOCK_DATA);
    loadHardware();
    if (USE_MOCK_DATA) startMockSSE(); else startHWSSE();
});
