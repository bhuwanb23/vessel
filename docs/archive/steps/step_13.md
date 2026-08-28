# Step 13 — Web Dashboard (GUI): Full Detailed Plan

---

## Goal of Step 13
Add a local web dashboard to the tool that lets users interact with every capability from Steps 1–12 through a browser instead of the terminal. The user runs `llm-planner --serve-ui`, opens `http://localhost:8080` in any browser, and sees live hardware gauges, a model catalog, a prediction table with visual bars, one-click download and execution, and calibration history charts. The dashboard is served by the same C++ binary — no separate frontend server, no Node.js, no Python. Everything stays as a single-binary distribution.

---

## Why This Step Exists

The CLI is powerful but has three fundamental limitations:

1. **It excludes non-technical users.** Anyone uncomfortable with terminals will never use the tool, no matter how good it is.
2. **It can't show live data well.** Watching VRAM usage tick up during a long generation requires refreshing manually or writing a bash loop.
3. **It doesn't visualize comparisons.** A prediction table with 12 strategies is hard to scan as text. Bar charts make tradeoffs instant.

A web dashboard fixes all three without abandoning the CLI (which stays for power users and scripts).

**The product shift:** Before Step 13, the tool is for developers and enthusiasts. After Step 13, it's usable by anyone who can open a browser — including the millions of Ollama and LM Studio users who want a better tool but won't touch a terminal.

---

## What You Need Before Starting

### From Steps 1–12 (must be solid)
- The full pipeline works end-to-end via CLI
- All modules (profiler, fetcher, predictor, matrix, ranker, download manager, executor, calibration log, recommendation engine) are functional
- Platform expansion (Step 11) complete — the dashboard should work on Windows, Linux, macOS

### New Dependencies
- **cpp-httplib** — a single-header C++ HTTP server library. No compilation, no external dependency, just drop the header into your project.
  - Repository: `https://github.com/yhirose/cpp-httplib`
  - License: MIT
  - Size: ~10,000 lines, one header file
  - Features: HTTP/HTTPS server, request routing, JSON support (via nlohmann::json which you already use), Server-Sent Events for streaming, file serving
- **Frontend files (HTML/CSS/JS)** — plain vanilla, no React/Vue/framework. Total: ~2,000–3,000 lines of code.

### What You Are NOT Doing
- **No frontend framework.** React, Vue, Svelte all add complexity and build tooling. Vanilla JS is sufficient for a dashboard with 5–6 views.
- **No separate frontend server.** The C++ binary serves the HTML/CSS/JS files directly.
- **No user accounts or authentication.** This is a local dashboard on localhost. Authentication is unnecessary and adds friction.
- **No database.** All state lives in the existing calibration log (JSONL file) and in-memory during the session.
- **No mobile app.** Responsive web design covers mobile browsers if anyone wants to use it from a phone on the same network.

---

## Phase A — Architecture Overview

### A1. The Component Diagram

```
┌─────────────────────────────────────────────────┐
│  Browser (Chrome/Firefox/Safari/Edge)          │
│  http://localhost:8080                          │
│  ┌──────────────────────────────────────────┐  │
│  │  index.html + app.js + styles.css        │  │
│  │  (Vanilla HTML/JS, ~2000 lines total)    │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────┘
                      │ HTTP / JSON / SSE
                      ▼
┌─────────────────────────────────────────────────┐
│  llm-planner binary (C++)                       │
│  ┌──────────────────────────────────────────┐  │
│  │  cpp-httplib server (port 8080)          │  │
│  │  ┌────────────────────────────────────┐  │  │
│  │  │  REST API layer                    │  │  │
│  │  │  /api/hardware, /api/predict, etc. │  │  │
│  │  └───────────────┬────────────────────┘  │  │
│  └──────────────────┼──────────────────────┘  │
│                     ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Existing modules (Steps 1-12):          │  │
│  │  profiler, fetcher, predictor, matrix,  │  │
│  │  ranker, download_manager, executor,    │  │
│  │  calibration_log, recommendation_engine │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### A2. The Data Flow

**Read operations (hardware, predictions, calibration):**
```
Browser → HTTP GET → C++ handler → existing module → JSON response → Browser renders
```

**Write operations (download, execute):**
```
Browser → HTTP POST → C++ handler → existing module (may take minutes) 
       ↓
Browser opens SSE stream → C++ streams progress updates → Browser updates UI live
```

**Live data (hardware gauges during execution):**
```
Browser polls /api/hardware every 1 second → C++ profiler → JSON → Browser updates gauges
OR
Browser opens SSE stream → C++ pushes updates every 500ms → Browser updates gauges
```

### A3. Threading Model

- **Main thread:** cpp-httplib event loop, handles incoming HTTP requests
- **Worker threads:** cpp-httplib spawns a thread per request automatically
- **Long-running tasks (download, execute):** Run on the request thread but stream progress via SSE. The user can navigate away and come back — the task continues.
- **Background sampler:** During execution, the sampler thread from Step 6 pushes hardware samples to a shared queue. The SSE endpoint drains this queue and streams to the browser.

**Concurrency concerns:**
- Multiple simultaneous browsers can view the dashboard (fine, all read-only)
- Only one download or execution should run at a time (enforce with a mutex — display "Task in progress" if a second request arrives)
- The calibration log is append-only, so concurrent reads are safe

---

## Phase B — cpp-httplib Integration

### B1. Adding cpp-httplib

Download `httplib.h` from the GitHub repo and drop it into your project:

```
llm-planner/
├── external/
│   ├── llama.cpp/          (from Step 6)
│   └── httplib/
│       └── httplib.h        (single file, ~10K lines)
├── src/
│   ├── main.cpp
│   ├── profiler.cpp
│   ├── ... (existing modules)
│   └── web_server.cpp       (NEW)
├── web/                      (NEW)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── assets/
│       ├── favicon.ico
│       └── logo.svg
```

### B2. CMake Configuration

```cmake
# Add cpp-httplib include path
target_include_directories(llm-planner PRIVATE external/httplib)

# Windows: link Winsock (cpp-httplib depends on it)
if(WIN32)
    target_link_libraries(llm-planner PRIVATE ws2_32)
endif()

# Embed the web/ directory into the binary
# Option A: Compile-time embedding (recommended)
include(cmake/embed_files.cmake)  # Custom function that converts files to C++ string constants

# Option B: Ship the web/ directory alongside the binary
# The server serves files from ./web/ relative to the binary location
```

### B3. Embedding Web Files in the Binary

**Why embed:** Users get a single binary. No "make sure web/ is in the right place" bugs.

**How:** Convert HTML/CSS/JS files to C++ string constants at build time. A CMake function reads each file and generates a header:

```cpp
// Auto-generated: embedded_web.h
const char* WEB_INDEX_HTML = R"HTML(
<!DOCTYPE html>
<html>
...
</html>
)HTML";

const char* WEB_APP_JS = R"JS(
async function loadHardware() { ... }
...
)JS";

const char* WEB_STYLES_CSS = R"CSS(
body { font-family: system-ui; }
...
)CSS";
```

At runtime, the server responds with these strings for the corresponding URLs:

```cpp
server.Get("/", [](const auto& req, auto& res) {
    res.set_content(WEB_INDEX_HTML, "text/html");
});
server.Get("/app.js", [](const auto& req, auto& res) {
    res.set_content(WEB_APP_JS, "application/javascript");
});
```

### B4. Starting the Server

Add a new CLI flag `--serve-ui`:

```cpp
if (args.serve_ui) {
    httplib::Server server;
    
    // Register API routes
    setupApiRoutes(server);
    
    // Register static file routes
    setupStaticRoutes(server);
    
    // Print startup message
    std::cout << "🌐 Dashboard running at http://localhost:8080" << std::endl;
    std::cout << "   Press Ctrl+C to stop" << std::endl;
    
    // Optionally open browser automatically
    if (!args.no_browser) {
        openBrowser("http://localhost:8080");
    }
    
    // Start server (blocks until Ctrl+C)
    server.listen("localhost", 8080);
}
```

### B5. The `openBrowser` Helper

Opens the default browser cross-platform:

```cpp
void openBrowser(const std::string& url) {
    #ifdef _WIN32
        ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
    #elif __APPLE__
        std::system(("open " + url).c_str());
    #else  // Linux
        std::system(("xdg-open " + url + " &").c_str());
    #endif
}
```

---

## Phase C — The REST API Design

### C1. API Endpoint List

| Method | Endpoint | Purpose | Response Type |
|---|---|---|---|
| GET | `/api/hardware` | Current hardware profile | JSON |
| GET | `/api/hardware/live` | Server-Sent Events stream of hardware samples | SSE |
| POST | `/api/predict` | Given a model URL, return the strategy table | JSON |
| GET | `/api/recommend` | Model recommendations for this hardware | JSON |
| GET | `/api/catalog` | Full model catalog | JSON |
| GET | `/api/models/local` | List of models already downloaded | JSON |
| POST | `/api/download` | Start a model download | JSON (task ID) |
| GET | `/api/download/:id/progress` | SSE stream of download progress | SSE |
| POST | `/api/execute` | Start model execution | JSON (task ID) |
| GET | `/api/execute/:id/stream` | SSE stream of execution output | SSE |
| POST | `/api/execute/:id/abort` | Abort a running execution | JSON |
| GET | `/api/calibration` | Calibration log statistics | JSON |
| GET | `/api/calibration/history` | All calibration entries for charts | JSON |
| DELETE | `/api/calibration` | Reset calibration log | JSON |
| GET | `/api/health` | Health check (server is up) | JSON |

### C2. Response Format Convention

All JSON responses follow this envelope:

```json
{
    "success": true,
    "data": { ... },
    "error": null,
    "timestamp": "2026-08-15T14:32:07Z"
}
```

Or on error:

```json
{
    "success": false,
    "data": null,
    "error": {
        "code": "MODEL_NOT_FOUND",
        "message": "The specified model URL returned 404",
        "details": "..."
    },
    "timestamp": "2026-08-15T14:32:07Z"
}
```

### C3. Detailed Endpoint Specs

#### GET /api/hardware
Returns the current hardware profile.

**Response:**
```json
{
    "success": true,
    "data": {
        "platform": "nvidia",
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "vram_total_bytes": 10737418240,
        "vram_free_bytes": 9328496640,
        "vram_used_pct": 13.1,
        "ram_total_bytes": 34359738368,
        "ram_free_bytes": 26105491456,
        "ram_used_pct": 24.0,
        "gpu_bandwidth_gbs": 760.1,
        "ram_bandwidth_gbs": 41.2,
        "nvme_sequential_mbs": 4217,
        "nvme_random_4k_mbs": 127,
        "gpu_temp_celsius": 42,
        "gpu_clock_mhz": 1935,
        "hardware_fingerprint": "i7-12700K|NVIDIA GeForce RTX 3080|32GB|Samsung 980 PRO",
        "is_unified_memory": false,
        "profiled_at": "2026-08-15T14:32:07Z"
    }
}
```

#### GET /api/hardware/live (SSE)
Streams hardware samples every 500ms. Used for live gauges.

**Response stream:**
```
event: hardware
data: {"vram_free_bytes": 9328496640, "gpu_temp_celsius": 42, "timestamp": "..."}

event: hardware
data: {"vram_free_bytes": 9327104000, "gpu_temp_celsius": 43, "timestamp": "..."}

...
```

The client closes the connection when navigating away.

#### POST /api/predict
Input: model URL. Output: full strategy table.

**Request:**
```json
{
    "model_url": "https://huggingface.co/.../Llama-3.2-3B-Q4_K_M.gguf",
    "priority": "speed"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "model": {
            "name": "Llama 3.2 3B Instruct",
            "params": 3212749824,
            "architecture": "llama",
            "quant": "Q4_K_M",
            "context": 131072,
            "is_moe": false
        },
        "hardware": { ... },
        "strategies": [
            {
                "rank": 1,
                "placement": "FULL_GPU",
                "gpu_layers": 28,
                "context": 4096,
                "kv_quant": "FP16",
                "predicted": {
                    "tokens_per_sec": 385,
                    "ttft_ms": 45,
                    "vram_bytes": 2576980378,
                    "ram_bytes": 536870912,
                    "confidence": "HIGH"
                },
                "viable": true,
                "label": "🏆 Fastest",
                "warnings": []
            },
            ...
        ],
        "recommendation": "Strategy #1 for interactive chat. Strategy #3 for long documents."
    }
}
```

#### POST /api/download (returns task ID for SSE tracking)
**Request:**
```json
{
    "model_url": "https://huggingface.co/...",
    "target_dir": "C:\\dev\\models\\"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "task_id": "dl_a1b2c3d4",
        "stream_url": "/api/download/dl_a1b2c3d4/progress"
    }
}
```

#### GET /api/download/:id/progress (SSE)
```
event: progress
data: {"bytes_downloaded": 524288000, "bytes_total": 2013265920, "speed_mbs": 45.2, "eta_seconds": 34}

event: progress
data: {"bytes_downloaded": 1073741824, "bytes_total": 2013265920, "speed_mbs": 47.1, "eta_seconds": 20}

event: verifying
data: {"message": "Verifying SHA256..."}

event: complete
data: {"local_path": "C:\\dev\\models\\Llama-3.2-3B-Q4_K_M.gguf", "verified": true}
```

#### POST /api/execute
**Request:**
```json
{
    "model_path": "C:\\dev\\models\\Llama-3.2-3B-Q4_K_M.gguf",
    "strategy": {
        "placement": "FULL_GPU",
        "gpu_layers": 28,
        "context": 4096,
        "kv_quant": "FP16"
    },
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 200
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "task_id": "ex_e5f6g7h8",
        "stream_url": "/api/execute/ex_e5f6g7h8/stream"
    }
}
```

#### GET /api/execute/:id/stream (SSE)
```
event: loading
data: {"message": "Loading model..."}

event: loaded
data: {"vram_used_bytes": 2576980378, "load_time_ms": 3200}

event: prefill
data: {"ttft_ms": 52, "prompt_tokens": 12}

event: token
data: {"token": "Quantum", "tokens_generated": 1, "current_tok_per_sec": 371}

event: token
data: {"token": " computing", "tokens_generated": 2, "current_tok_per_sec": 373}

...

event: hardware_sample
data: {"vram_used_bytes": 2791728742, "gpu_temp_celsius": 68, "gpu_clock_mhz": 1935}

event: complete
data: {
    "tokens_generated": 200,
    "duration_sec": 0.54,
    "actual_tokens_per_sec": 370.4,
    "actual_ttft_ms": 52,
    "peak_vram_bytes": 2791728742,
    "throttled": false,
    "predicted_vs_actual": {...}
}
```

---

## Phase D — The Frontend (HTML/CSS/JS)

### D1. The Page Structure

Single-page application with tab-based navigation. No routing library needed — just show/hide sections.

```
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LLM Deployment Planner        [Settings] [Help]    │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌───┬────────────────────────────────────────────────┐   │
│  │ 🏠 │  [Home] [Predict] [Recommend] [Models] [Cal]  │   │
│  └───┴────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                       │   │
│  │           <Active view content>                       │   │
│  │                                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Hardware: RTX 3080 (10GB) | 32GB RAM | NVMe 4.2GB/s│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### D2. The Five Views

#### View 1: Home (Hardware Overview)

**Purpose:** Landing page. Shows what hardware the tool detected and links to the main actions.

**Elements:**
- Big hardware summary card (GPU model, VRAM, RAM, platform icon)
- Live gauges (VRAM used, RAM used, GPU temp, GPU clock)
- Three action buttons: "Predict a Model", "Get Recommendations", "Browse My Models"

**Live gauges:** Use SSE from `/api/hardware/live` to update every 500ms. Render as circular progress indicators or horizontal bars.

**Example layout:**
```
┌────────────────────────────────────────────────┐
│  🖥️  NVIDIA GeForce RTX 3080                    │
│      10 GB VRAM | 32 GB RAM | Windows          │
├────────────────────────────────────────────────┤
│  VRAM: [████░░░░░░] 42% (4.2/10 GB)            │
│  RAM:  [██░░░░░░░░] 24% (7.7/32 GB)            │
│  Temp: 42°C  Clock: 1935 MHz  NVMe: 4.2 GB/s   │
├────────────────────────────────────────────────┤
│  [🔍 Predict a Model]  [💡 Recommend]  [📦 My Models] │
└────────────────────────────────────────────────┘
```

#### View 2: Predict (Strategy Comparison)

**Purpose:** The main use case. Paste a URL, see the strategies.

**Elements:**
- URL input field with "Predict" button
- Priority selector (radio buttons: Speed / Quality / Safety)
- Loading spinner while predicting
- Results table with visual bars for tok/s, VRAM, TTFT
- Recommendation callout at the bottom
- Action buttons per row: "Execute" (if model downloaded) or "Download & Execute"

**The results table (visual style):**
```
 #  Placement       tok/s              VRAM             Context  Status
 1  Full GPU 4K    ████████████ 385   ██░░ 2.4 GB     4K       ✅
 2  Full GPU 32K   ██████████ 320     █████ 5.1 GB    32K      ✅
 3  Split 24/28    ████ 95            ███ 3.9 GB      128K     ✅
 4  CPU Only       █ 21               — 0 GB          4K       ✅
 5  Full GPU 128K  — no fit           ██████████ 22GB 128K     ❌
```

Bars are colored by strategy type (green = GPU, yellow = split, gray = CPU). Non-viable strategies are grayed out.

#### View 3: Recommend (Model Suggestions)

**Purpose:** "What should I run?" for users who don't know model names.

**Elements:**
- Priority selector (Balanced / Speed / Quality)
- Use case filter (All / Chat / Coding / Reasoning)
- Max download size slider
- Results as cards, one per model, sorted by rank

**Each recommendation card:**
```
┌────────────────────────────────────────────────┐
│  🏆 Best Overall                                 │
│  Qwen 2.5 7B Instruct                            │
│  ★★★★☆ Quality | Q4_K_M | 4.5 GB download       │
├────────────────────────────────────────────────┤
│  Best strategy: Full GPU, 4K context           │
│  Expected: ~45 tok/s | 5.2 GB VRAM              │
├────────────────────────────────────────────────┤
│  [📥 Download & Run]  [ℹ️ Details]              │
└────────────────────────────────────────────────┘
```

#### View 4: My Models (Local Model Manager)

**Purpose:** See what's downloaded, delete old models to free disk space.

**Elements:**
- List of local GGUF files with metadata (size, quant, family, last used)
- Actions per model: "Run", "Show Strategies", "Delete"
- Total disk usage summary at the top
- Sort by name, size, or last used

#### View 5: Calibration (History & Accuracy)

**Purpose:** Show that the tool is learning. Prove the predictor is trustworthy.

**Elements:**
- Summary card: total runs, average prediction accuracy, hardware fingerprint
- Chart 1: Prediction accuracy over time (line chart, delta% on Y axis, run number on X axis)
- Chart 2: Predicted vs actual scatter plot for tok/s
- Chart 3: Predicted vs actual scatter plot for memory
- Table of recent runs with delta columns
- Reset button (with confirmation modal)

**Charts:** Use vanilla JS with a simple charting library. **Recommendation:** Chart.js (single file, ~200KB, no dependencies) or uPlot (smaller, faster).

### D3. Real-Time Execution View

When a user clicks "Execute" from any view, open a modal or dedicated page with:

- Live streaming text output (tokens appear as generated)
- Live tok/s counter (updates every second)
- Live VRAM gauge (updates every 500ms)
- Live GPU temperature
- Progress bar (tokens generated / max tokens)
- Abort button

At the end, show the predicted-vs-actual comparison with color-coded deltas (green = close, yellow = off, red = wrong).

### D4. JavaScript Architecture

Keep it simple. No frameworks. Structure:

```javascript
// app.js — main application entry point

// State
const state = {
    hardware: null,
    currentView: 'home',
    activeTask: null,
};

// API client (thin wrapper around fetch)
const api = {
    async getHardware() { ... },
    async predict(url, priority) { ... },
    async recommend(priority, useCase, maxSize) { ... },
    async startDownload(url) { ... },
    async startExecute(path, strategy, prompt) { ... },
    // ... etc
};

// SSE client (wrapper around EventSource)
function subscribeSSE(url, handlers) {
    const source = new EventSource(url);
    for (const [event, handler] of Object.entries(handlers)) {
        source.addEventListener(event, e => handler(JSON.parse(e.data)));
    }
    return source;
}

// View controllers (one per view)
const views = {
    home: {
        render() { ... },
        subscribeLive() { ... },
    },
    predict: {
        render() { ... },
        onSubmit() { ... },
    },
    recommend: { ... },
    models: { ... },
    calibration: { ... },
};

// Router
function navigate(viewName) {
    state.currentView = viewName;
    views[viewName].render();
}

// Init
async function init() {
    state.hardware = await api.getHardware();
    navigate('home');
}

init();
```

Total JS: ~1,000–1,500 lines. Manageable without a framework.

### D5. CSS Approach

Use CSS custom properties for theming (light/dark mode). No CSS framework. Modern CSS (Grid, Flexbox) handles all layout needs.

```css
:root {
    --bg: #ffffff;
    --text: #1a1a1a;
    --primary: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --border: #e5e7eb;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1a1a1a;
        --text: #f5f5f5;
        --border: #374151;
    }
}

body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
}
```

Total CSS: ~500–800 lines.

---

## Phase E — Server-Sent Events (SSE) for Live Data

### E1. Why SSE Instead of WebSockets

| Aspect | SSE | WebSockets |
|---|---|---|
| Complexity | Simple (built into HTTP) | More complex (upgrade handshake) |
| Direction | Server → Client only | Bidirectional |
| Reconnection | Automatic in browsers | Manual |
| Firewall friendly | Yes (looks like HTTP) | Sometimes blocked |
| Use case | Progress updates, live gauges | Chat, real-time collaboration |

**For this dashboard, SSE is perfect** — you only need server-to-client streaming (progress, tokens, hardware samples). No need for bidirectional communication.

### E2. cpp-httplib SSE Implementation

```cpp
server.Get("/api/hardware/live", [&](const auto& req, auto& res) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    
    res.set_chunked_content_provider("text/event-stream",
        [&](size_t offset, httplib::DataSink& sink) {
            while (sink.is_writable()) {
                auto hw = profileHardwareLive();
                std::string json = serialize(hw);
                std::string event = "event: hardware\ndata: " + json + "\n\n";
                sink.write(event.c_str(), event.length());
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            sink.done();
            return true;
        });
});
```

### E3. Browser-Side SSE

```javascript
const source = new EventSource('/api/hardware/live');

source.addEventListener('hardware', (event) => {
    const data = JSON.parse(event.data);
    updateGauges(data);
});

source.addEventListener('error', (event) => {
    console.error('SSE error:', event);
    source.close();
});

// When leaving the page:
source.close();
```

### E4. Handling Long-Running Tasks

Downloads and executions are long-running. The pattern:

1. Client POSTs to `/api/download` → server returns a task ID
2. Client opens SSE to `/api/download/:id/progress`
3. Server pushes progress events as the download proceeds
4. Server sends `complete` event when done
5. Client closes SSE

If the client navigates away, the task continues on the server. When the client comes back, they can reconnect to the SSE endpoint with the same task ID and pick up where they left off.

**Task management:**
```cpp
struct Task {
    std::string id;
    std::string type;  // "download", "execute"
    std::atomic<bool> running;
    std::atomic<bool> aborted;
    std::mutex event_queue_mutex;
    std::queue<std::string> event_queue;  // JSON events waiting to be sent
    std::condition_variable event_available;
};

std::unordered_map<std::string, std::shared_ptr<Task>> active_tasks;
std::mutex tasks_mutex;

std::string startDownloadTask(const std::string& url) {
    auto task = std::make_shared<Task>();
    task->id = generateTaskId();
    task->type = "download";
    task->running = true;
    
    {
        std::lock_guard<std::mutex> lock(tasks_mutex);
        active_tasks[task->id] = task;
    }
    
    // Launch download in background thread
    std::thread([task, url]() {
        downloadWithProgressCallback(url, [task](auto progress) {
            std::string event = "event: progress\ndata: " + serialize(progress) + "\n\n";
            {
                std::lock_guard<std::mutex> lock(task->event_queue_mutex);
                task->event_queue.push(event);
            }
            task->event_available.notify_all();
        });
        task->running = false;
    }).detach();
    
    return task->id;
}
```

---

## Phase F — Security Considerations

### F1. Localhost-Only Binding

The server should **only** listen on `127.0.0.1` (localhost), not `0.0.0.0` (all interfaces). This prevents other machines on the network from accessing the dashboard.

```cpp
server.listen("127.0.0.1", 8080);  // NOT "0.0.0.0"
```

**Exception:** If the user explicitly wants remote access (for headless servers), add a flag:
```
llm-planner --serve-ui --bind 0.0.0.0 --port 8080
```
But default to localhost. Print a warning when binding to non-localhost.

### F2. No Authentication (For Localhost)

Since the dashboard only accepts localhost connections, anyone who can reach `localhost:8080` is already logged into the machine. Authentication would add friction without security benefit.

**If** you add remote access, then add optional basic auth:
```
llm-planner --serve-ui --bind 0.0.0.0 --auth admin:secretpassword
```

### F3. CSRF and CORS

- **CORS:** Set `Access-Control-Allow-Origin: http://localhost:8080` to prevent other sites from making requests to your API.
- **CSRF:** Not a concern for localhost-only. If adding remote access, add CSRF tokens.

### F4. Rate Limiting

The predictor is cheap (microseconds), so rate limiting is unnecessary for MVP. If someone starts spamming download requests, that's a DoS on themselves.

### F5. Path Traversal (For Model Path Handling)

When the user selects a local model file, validate the path:
- Must be an absolute path
- Must exist and be readable
- Must end in `.gguf`
- Must not contain `..` or symlinks that escape the models directory (optional)

---

## Phase G — Cross-Platform Considerations

### G1. Port Selection

Port 8080 is common and often taken by other services. If 8080 is in use, try 8081, 8082, etc., up to 8090. If all are taken, error out with a clear message.

```cpp
int port = 8080;
while (port < 8090 && !server.bind_to_port("127.0.0.1", port)) {
    port++;
}
if (port >= 8090) {
    std::cerr << "❌ No available port in 8080-8089 range" << std::endl;
    return 1;
}
std::cout << "🌐 Dashboard: http://localhost:" << port << std::endl;
```

### G2. Firewall Prompts

- **Windows:** First run may trigger a Windows Firewall prompt. Since you're binding to localhost only, click "Allow" (or "Cancel" — localhost works either way).
- **macOS:** May prompt for network access permission. Same as Windows.
- **Linux:** No prompt, but if `ufw` or similar is active, may need `sudo ufw allow 8080/tcp` (only for non-localhost binding).

Document this in the README.

### G3. Browser Auto-Launch

Auto-launching the browser is a nice UX touch. Suppress with `--no-browser` flag for headless/scripted use.

### G4. High-DPI Displays

Use CSS `vw`/`vh`/`rem` units and `viewport` meta tag for proper scaling on 4K monitors, Retina displays, and mobile browsers.

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

---

## Phase H — Testing

### H1. Test Scenarios

**Test 1: Server Starts and Stops Cleanly**
- Run `llm-planner --serve-ui`
- Verify: Server starts, prints URL, browser opens (if not `--no-browser`)
- Press Ctrl+C
- Verify: Server shuts down cleanly, no zombie processes

**Test 2: Hardware Gauges Update Live**
- Open dashboard
- Verify: VRAM/RAM/temp gauges show current values
- Run a CUDA workload in another terminal (`nvidia-smi -q` won't work, use something like a benchmark)
- Verify: VRAM gauge in browser updates within 1 second

**Test 3: Predict from URL**
- Paste a Hugging Face GGUF URL
- Click Predict
- Verify: Loading spinner appears
- Verify: Strategy table appears within 5 seconds
- Verify: Bars are visually proportional to values
- Verify: Non-viable strategies are grayed out

**Test 4: Priority Changes Reorder Table**
- On predict view, change priority from Speed to Safety
- Verify: Table reorders without a new API call (client-side re-sort)
- Or: Verify: New API call fetches re-ranked results

**Test 5: Download with Progress**
- Click "Download" on a model
- Verify: Progress bar appears
- Verify: Speed, ETA, percentage update in real time
- Verify: On completion, model appears in "My Models"
- Try aborting: click Cancel mid-download
- Verify: Download stops, `.partial` file preserved on disk

**Test 6: Execute with Live Tokens**
- Click "Execute" on a downloaded model
- Verify: Modal or view opens
- Verify: Tokens stream in as generated
- Verify: tok/s counter updates
- Verify: VRAM gauge updates during execution
- Verify: On completion, predicted-vs-actual report appears

**Test 7: Calibration History Chart**
- Run several executions to accumulate calibration data
- Navigate to Calibration view
- Verify: Charts render with real data
- Verify: Delta trend shows improvement over time (or at least stable accuracy)

**Test 8: Recommendation View**
- Navigate to Recommend
- Verify: Cards appear for viable models
- Change priority to Quality
- Verify: Order changes, quality-focused labels appear
- Click a card's "Download & Run"
- Verify: Flow proceeds to download, then execute

**Test 9: Multi-Tab Behavior**
- Open dashboard in two browser tabs
- Verify: Both tabs show current hardware
- Start a download in tab 1
- Verify: Tab 2 doesn't crash (it doesn't know about the download, which is fine)

**Test 10: Browser Compatibility**
- Test in Chrome, Firefox, Safari, Edge
- Verify: All views render correctly
- Verify: SSE works in all browsers
- Verify: No console errors

**Test 11: Dark Mode**
- Change OS dark mode setting
- Verify: Dashboard switches themes automatically (via `prefers-color-scheme`)

**Test 12: Server Robustness**
- Send malformed JSON to `/api/predict`
- Verify: Server responds with 400 Bad Request, doesn't crash
- Send request with missing `model_url` field
- Verify: Clear error message, doesn't crash

---

## Step 13 — Done Checklist

- [ ] cpp-httplib integrated into CMake build
- [ ] Web files embedded in binary (or served from `web/` directory)
- [ ] `--serve-ui` flag starts the server
- [ ] Server binds to `127.0.0.1` by default
- [ ] `--bind` flag allows non-localhost binding with warning
- [ ] `--port` flag allows custom port
- [ ] `--no-browser` flag prevents auto-launch
- [ ] Browser auto-launches on `--serve-ui` unless suppressed
- [ ] Port fallback works if 8080 is taken
- [ ] All 15 REST endpoints implemented
- [ ] JSON envelope format consistent across endpoints
- [ ] SSE streams work for hardware/live, download/progress, execute/stream
- [ ] Task manager handles concurrent tasks safely
- [ ] Ctrl+C shuts down server cleanly
- [ ] Home view shows hardware summary with live gauges
- [ ] Predict view accepts URL, shows strategy table with bars
- [ ] Recommend view shows model cards ranked by priority
- [ ] Models view lists local GGUF files
- [ ] Calibration view shows history charts
- [ ] Execution modal streams tokens in real time
- [ ] Predicted-vs-actual report appears after execution
- [ ] Dark mode works via `prefers-color-scheme`
- [ ] Responsive layout works on mobile browser
- [ ] Tested in Chrome, Firefox, Safari, Edge
- [ ] Server rejects malformed requests without crashing
- [ ] Path traversal protection on model path inputs
- [ ] No JavaScript framework used (vanilla JS only)
- [ ] Total frontend size < 500KB (including Chart.js)

---

## Common Failure Points at Step 13

| Problem | Likely Cause | Fix |
|---|---|---|
| Port 8080 already in use | Another service running | Try 8081-8089 automatically |
| SSE stream disconnects after 30s | Reverse proxy or firewall closing idle connections | Send a keepalive comment every 15s: `:keepalive\n\n` |
| Live gauges don't update | Client not receiving SSE events | Check browser console for CORS or connection errors |
| Chart.js not loading | External CDN blocked or offline | Embed Chart.js in the binary alongside other web files |
| Server crashes on malformed JSON | No exception handling in JSON parsing | Wrap `nlohmann::json::parse()` in try/catch, return 400 on failure |
| High CPU during idle | SSE loop polling too fast | Add `std::this_thread::sleep_for()` in the loop |
| Browser doesn't auto-launch | Wrong platform detection or blocked | Fall back to printing URL and instructing user to open manually |
| Concurrent downloads corrupt each other | Shared file handle | Mutex around active tasks, one download at a time |
| Model paths with spaces break execution | URL encoding not handled | Decode URL parameters before using as file paths |
| Dark mode flashes on load | CSS loaded after HTML | Inline critical CSS or preload the stylesheet |
| Mobile layout broken | No viewport meta tag or fixed-width elements | Use responsive units (rem, %), test on real phone |
| Windows Defender blocks server | First-run heuristic | Sign the binary, or document the false positive |

---

## Time Estimate for Step 13

| Phase | Work | Time |
|---|---|---|
| A | Architecture design + directory structure | 0.5 day |
| B | cpp-httplib integration + CMake + embedding | 1 day |
| C | REST API endpoints (15 endpoints) | 2–3 days |
| D | Frontend HTML/CSS/JS (5 views + execution modal) | 4–6 days |
| E | SSE for live data + task management | 1–2 days |
| F | Security (localhost binding, CORS, validation) | 0.5 day |
| G | Cross-platform testing + browser launch | 1 day |
| H | Testing all scenarios | 1–2 days |

**Total: 1–2 weeks, as estimated. The frontend (Phase D) is the largest time sink — writing 5 different views with proper UX takes significant effort. The API layer (Phase C) is straightforward because you're just exposing existing functions over HTTP. SSE (Phase E) has some tricky edge cases with task management and reconnection but is well-understood territory.**

---

## What "Done" Looks Like

A non-technical user:
1. Downloads `llm-planner.exe`
2. Double-clicks it
3. Browser opens automatically to `http://localhost:8080`
4. Sees their hardware detected with live gauges
5. Clicks "Recommend Models"
6. Sees a ranked list of models that will work on their machine
7. Clicks "Download & Run" on the top pick
8. Watches the download progress
9. Watches tokens generate in real time
10. Never touches the terminal

That is the dashboard's job. Everything else is bonus.