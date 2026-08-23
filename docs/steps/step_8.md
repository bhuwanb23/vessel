# Step 8 — Model Download Manager: Full Detailed Plan

---

## Goal of Step 8
Given a model selected from the ranked strategy table, download the actual GGUF weight file(s) to disk — resumable, integrity-checked, disk-space-safe — and hand off a verified local file path to the Executor. This eliminates the manual download step that currently breaks the user experience between "I know which strategy I want" and "the model is actually running." After this step, the full pipeline is: URL in → predictions out → user picks → tool downloads → tool executes → tool calibrates. Zero manual steps.

---

## What You Need Before Starting

### From Steps 1–7 (already done and working)
- **Step 1's Hardware Profiler:** You need live free disk space readings. The profiler already queries storage — you may need to extend it to report free space on a specific drive, not just read speed.
- **Step 2's Metadata Fetcher:** Already uses libcurl for HTTP range requests to Hugging Face. The download manager reuses the same libcurl setup, CA certificate configuration, and redirect-following logic.
- **Step 4's Pipeline:** The method matrix already knows the model URL and the predicted file size (derivable from parameter count × bits-per-weight / 8, or from the `Content-Length` header during the range request).
- **Step 5's Ranker:** The user has selected a strategy. The download manager activates after this selection.
- **Step 6's Executor:** Currently expects a local file path. After Step 8, the executor receives the path from the download manager instead of requiring the user to provide it manually.
- **libcurl:** Already linked and working with HTTPS, redirect following, and range request support.

### New Dependencies
- **SHA256 hashing library:** You need to hash the downloaded file and compare it against Hugging Face's published hash. Options:
  - **Windows CNG API** (`bcrypt.h`, `BCryptHashData`) — built into Windows, no external dependency. Recommended for MVP.
  - **OpenSSL** — if you already have it via libcurl's dependency chain, you can reuse it.
  - **A single-header library** like `sha256.h` — simplest integration but adds a third-party file.
- **No other new dependencies.** Everything else uses libcurl and Windows APIs you already have.

---

## Phase A — Pre-Download Safety Check

### What It Does
Before transferring a single byte, the download manager verifies that the target drive has enough space to complete the download safely. If not, it refuses cleanly and suggests an alternative.

### The Size Estimation Problem
You need to know the file size before downloading it. Three sources, in priority order:

**Source 1: HTTP HEAD request (most reliable)**
Before starting the download, send an HTTP HEAD request to the GGUF URL. Hugging Face's CDN responds with a `Content-Length` header containing the exact file size in bytes.

```
HEAD https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
→ Content-Length: 2013265920
```

This is a single round-trip, takes <1 second, and gives you the exact byte count. Use libcurl with `CURLOPT_NOBODY` set to `1L` to send a HEAD request instead of GET.

**Source 2: Content-Range from the initial range request**
If you already did a range request in Step 2's metadata fetch, the response header `Content-Range: bytes 0-65535/2013265920` contains the total file size as the last number. You may already have this value cached from Step 2 — check before making a redundant HEAD request.

**Source 3: Estimate from metadata (fallback)**
If both HTTP methods fail (unlikely with Hugging Face, but possible with other sources), estimate from the model metadata:
```
estimated_bytes = param_count × bits_per_weight / 8 × 1.05
```
The 1.05 multiplier accounts for GGUF header overhead, tensor metadata, and alignment padding. This is a rough estimate — flag it as such in the UI.

### The Disk Space Check

**Windows API:** `GetDiskFreeSpaceExW()`

```
ULARGE_INTEGER freeBytesAvailable, totalBytes, totalFreeBytes;
GetDiskFreeSpaceExW(
    L"C:\\dev\\models\\",   // target directory
    &freeBytesAvailable,     // free space available to the calling user
    &totalBytes,             // total disk size
    &totalFreeBytes          // total free space (may differ from above due to quotas)
);
```

Use `freeBytesAvailable`, not `totalFreeBytes` — the former respects per-user disk quotas on enterprise systems.

**The 1.15× Safety Margin:**
```
required_space = file_size × 1.15
```

Why 1.15× and not 1.0×:
- The `.partial` file and the final file briefly coexist during the rename window (Phase C) — actually no, rename is atomic on the same volume, so this isn't the real reason
- The real reason: filesystem overhead, fragmentation, and the possibility that the user starts another download or application between the check and the completion
- 15% is conservative enough to prevent surprises without being so aggressive that it blocks legitimate downloads on nearly-full drives

**The Refusal Message:**
If `freeBytesAvailable < required_space`:
```
❌ Insufficient disk space.
   Model size:    4.2 GB
   Required:      4.8 GB (with 15% safety margin)
   Available:     3.1 GB on C:\
   Shortfall:     1.7 GB

   Suggestions:
   • Free up disk space and try again
   • Try a smaller quantization: Q3_K_M (~3.1 GB) would fit
   • Download to a different drive: --download-dir D:\models\
```

The "smaller quantization" suggestion is the key UX detail. Don't just say "not enough space" — look at the method matrix from Step 4, find a strategy with a smaller model file that IS viable, and suggest it by name. This requires:
1. Knowing the file sizes of alternative quants (estimate from param_count × bpw / 8 for each quant in your lookup table)
2. Checking which of those would fit in the available space
3. Suggesting the largest one that fits (best quality within the space constraint)

### The Target Directory
Default to `C:\dev\models\` (or wherever the user's models folder is from Step 0). Allow override via `--download-dir <path>`.

If the directory doesn't exist, create it with `std::filesystem::create_directories()`. Don't fail just because the folder isn't there yet.

---

## Phase B — Resumable Download

### The Core Mechanism
HTTP Range requests let you ask the server for a specific byte range of a file. If a previous download was interrupted at byte 1,048,576, you can resume by requesting `Range: bytes=1048576-` and the server sends everything from that point onward.

### The `.partial` File Strategy
Never write directly to the final filename. This prevents the Executor from accidentally loading a half-downloaded file.

**File naming:**
- During download: `Llama-3.2-3B-Instruct-Q4_K_M.gguf.partial`
- After verification: `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (atomic rename)

**Resume detection:**
Before starting a download, check if a `.partial` file already exists for the target model:
1. If no `.partial` file exists → fresh download, start from byte 0
2. If a `.partial` file exists → check its size with `std::filesystem::file_size()`
3. If the `.partial` file size is 0 → delete it and start fresh (corrupt empty file)
4. If the `.partial` file size > 0 → resume from that byte offset

### The libcurl Download Sequence

**Step B1: Configure the curl handle**
```
curl_easy_setopt(curl, CURLOPT_URL, model_url);
curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // HF CDN redirects
curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);          // no total timeout for large files
curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L); // abort if <1KB/s for 60s
curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
```

**Step B2: Set the resume offset (if applicable)**
```
if (resume_offset > 0) {
    curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)resume_offset);
    // Open the .partial file in append mode
    file = fopen(partial_path, "ab");
} else {
    // Fresh download — open in write mode (truncates any existing partial)
    file = fopen(partial_path, "wb");
}
```

**Step B3: Set the write callback**
Same pattern as Step 2's metadata fetcher, but writing to a file instead of a memory buffer:
```
size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    FILE* file = (FILE*)userdata;
    return fwrite(ptr, size, nmemb, file);
}
```

**Step B4: Set the progress callback**
libcurl's `CURLOPT_XFERINFOFUNCTION` gives you download progress without polling:
```
int progress_callback(void* clientp, 
                      curl_off_t dltotal,    // total bytes to download
                      curl_off_t dlnow,      // bytes downloaded so far
                      curl_off_t ultotal,    // total bytes to upload (0 for downloads)
                      curl_off_t ulnow) {    // bytes uploaded so far (0)
    
    // Calculate percentage, speed, ETA
    double percent = (dltotal > 0) ? (100.0 * dlnow / dltotal) : 0;
    // Speed and ETA require tracking elapsed time since download start
    
    // Print progress bar to console
    print_progress_bar(percent, dlnow, dltotal, speed, eta);
    
    // Check for abort signal (Ctrl+C)
    if (abort_requested.load()) return 1;  // returning non-zero aborts the transfer
    
    return 0;  // continue
}
```

**Step B5: Execute the download**
```
CURLcode res = curl_easy_perform(curl);
```

**Step B6: Handle the result**
```
if (res == CURLE_OK) {
    // Download complete — proceed to Phase C (integrity check)
} else if (res == CURLE_ABORTED_BY_CALLBACK) {
    // User pressed Ctrl+C — .partial file is preserved for resume
    print("Download paused. Run again to resume.");
} else if (res == CURLE_OPERATION_TIMEDOUT) {
    // Network too slow — .partial file preserved
    print("Download timed out (too slow). Run again to resume.");
} else {
    // Other error — .partial file preserved
    print("Download error: %s. Run again to resume.", curl_easy_strerror(res));
}
```

### The Progress Display
A clean console progress bar that updates in-place (using `\r` to overwrite the line):

```
Downloading Llama-3.2-3B-Instruct-Q4_K_M.gguf
[████████████████████░░░░░░░░░░] 67.3% | 1.35 GB / 2.01 GB | 45.2 MB/s | ETA: 15s
```

**Update frequency:** Don't update on every callback (libcurl calls it very frequently). Throttle to once per 500ms or once per 1% progress, whichever is less frequent. This prevents console flickering.

**Speed calculation:** Track bytes downloaded over the last 5 seconds (sliding window) for a stable speed reading. Don't use instantaneous speed — it fluctuates wildly.

**ETA calculation:** `remaining_bytes / current_speed`. Cap the display at "ETA: >1h" for very slow downloads rather than showing "ETA: 47h 23m 12s."

### Handling Interruptions Gracefully

| Interruption Type | What Happens | Recovery |
|---|---|---|
| Ctrl+C | `abort_requested` flag set, progress callback returns 1, libcurl aborts | `.partial` file preserved at last written byte. Re-run resumes. |
| Network drop | libcurl returns `CURLE_RECV_ERROR` or similar | `.partial` file preserved. Re-run resumes. |
| Disk full mid-download | `fwrite()` returns 0, write callback returns 0, libcurl aborts | `.partial` file preserved at last successful write. User frees space, re-run resumes. |
| Server disconnect | libcurl returns `CURLE_PARTIAL_FILE` | `.partial` file preserved. Re-run resumes. |
| Process killed (Task Manager) | No cleanup possible | `.partial` file exists on disk. Re-run detects it and resumes. |

**Key invariant:** The `.partial` file is NEVER deleted on failure. It is only deleted after a successful download + failed integrity check (Phase C), or explicitly by the user.

---

## Phase C — Integrity Verification

### Why This Matters
A corrupted model file produces garbage output or crashes the Executor. The user won't know why — they'll blame your tool. SHA256 verification catches silent corruption from network errors, disk errors, or CDN issues before the file ever reaches llama.cpp.

### Getting the Expected Hash

**Hugging Face's hash API:**
Every file in a Hugging Face repository has a SHA256 hash accessible via the API:

```
GET https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF/revision/main
```

This returns a JSON object containing a `siblings` array, where each entry has:
```json
{
  "rfilename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  "size": 2013265920,
  "lfs": {
    "sha256": "a1b2c3d4e5f6...",
    "size": 2013265920,
    "pointer_size": 134
  }
}
```

The `lfs.sha256` field is the hash you need. Note: this is the SHA256 of the **actual file content**, not the Git LFS pointer file.

**When to fetch the hash:** Fetch it during the pre-download phase (Phase A), alongside the HEAD request for file size. Cache it for the verification step. This avoids an extra network call after the download completes.

**Fallback if hash is unavailable:** Some older repositories or non-HuggingFace sources may not provide SHA256 hashes. In that case:
- Print a warning: `"⚠️ No SHA256 hash available for verification. File integrity not confirmed."`
- Proceed with the rename anyway — don't block the user
- Flag the download as "unverified" in any log output

### Computing the Local File Hash

**Using Windows CNG API (recommended, no external dependency):**

The sequence is:
1. `BCryptOpenAlgorithmProvider()` — open the SHA256 algorithm provider
2. `BCryptCreateHash()` — create a hash object
3. Open the `.partial` file and read it in chunks (e.g., 4MB at a time — large enough for throughput, small enough to fit in RAM)
4. `BCryptHashData()` — feed each chunk to the hash object
5. `BCryptFinishHash()` — get the final 32-byte hash
6. Convert to hex string and compare against the expected hash
7. `BCryptDestroyHash()` and `BCryptCloseAlgorithmProvider()` — cleanup

**Performance note:** Hashing a 4GB file takes ~5-10 seconds on a modern CPU. This is acceptable — it's a one-time cost per download. Display a progress indicator during hashing so the user doesn't think the tool froze:

```
Verifying file integrity... 2.1 GB / 4.2 GB (50%)
```

**Memory note:** Read the file in chunks, not all at once. A 4GB file won't fit in a single buffer on most systems. 4MB chunks are a good balance.

### The Rename

After the hash matches:
```
std::filesystem::rename(partial_path, final_path);
```

On Windows, `std::filesystem::rename()` is atomic when source and destination are on the same volume. This means there is no window where the file exists under the final name but is incomplete.

**If the rename fails** (cross-volume move, permissions issue): fall back to copy + delete, but print a warning. This should be rare if the download directory and models directory are the same.

### Hash Mismatch Handling
If the computed hash doesn't match the expected hash:
```
❌ File integrity check failed.
   Expected SHA256: a1b2c3d4e5f6...
   Actual SHA256:   7f8e9d0c1b2a...
   
   The downloaded file is corrupted. This can happen due to:
   • Network errors during transfer
   • Disk errors on the target drive
   • CDN serving a stale/corrupt copy
   
   The corrupted file has been deleted. Run the command again to re-download.
```

Then delete the `.partial` file. Don't keep a corrupted file around — it will confuse the resume logic on the next run.

---

## Phase D — Multi-Shard Model Handling

### The Problem
Large models (13B+, especially 70B+) are often split across multiple GGUF files because Hugging Face has a per-file size limit (typically 50GB for LFS). The files follow a naming pattern:

```
Llama-3.1-70B-Instruct-Q4_K_M-00001-of-00005.gguf
Llama-3.1-70B-Instruct-Q4_K_M-00002-of-00005.gguf
Llama-3.1-70B-Instruct-Q4_K_M-00003-of-00005.gguf
Llama-3.1-70B-Instruct-Q4_K_M-00004-of-00005.gguf
Llama-3.1-70B-Instruct-Q4_K_M-00005-of-00005.gguf
```

### Detection
When the user provides a URL or model name, check the Hugging Face repository's file listing for shard patterns:

**Method 1: Parse the filename**
If the provided filename matches the pattern `*-NNNNN-of-MMMMM.gguf`, it's a shard. Extract `N` (this shard) and `M` (total shards).

**Method 2: Query the repository API**
```
GET https://huggingface.co/api/models/<repo>/tree/main
```
This returns a JSON array of all files in the repository. Filter for files matching the shard pattern with the same base name. If multiple matches exist, it's a multi-shard model.

### Download Sequence for Multi-Shard Models

1. **Identify all shards** from the repository listing
2. **Calculate total size** (sum of all shard sizes from the API)
3. **Run the disk space check** (Phase A) against the total size × 1.15
4. **Download each shard sequentially** (not in parallel — parallel downloads from the same CDN can trigger rate limiting and actually slow things down)
5. **Verify each shard independently** (each shard has its own SHA256 in the API response)
6. **Only report "ready" when ALL shards are downloaded and verified**

### Progress Display for Multi-Shard
```
Downloading Llama-3.1-70B-Instruct-Q4_K_M (5 shards, 38.2 GB total)

Shard 1/5: [██████████████████████████████] 100% | 8.1 GB | Done ✅
Shard 2/5: [████████████████░░░░░░░░░░░░░░]  53% | 4.3 GB / 8.1 GB | 42 MB/s | ETA: 1m32s
Shard 3/5: Waiting...
Shard 4/5: Waiting...
Shard 5/5: Waiting...

Total: 12.4 GB / 38.2 GB (32.5%) | Overall ETA: 10m15s
```

### Executor Integration for Multi-Shard
llama.cpp handles multi-shard GGUF files natively — you pass the path to the **first shard** (`-00001-of-00005.gguf`) and it automatically discovers and loads the remaining shards from the same directory. The download manager just needs to ensure all shards are present and verified before handing the first shard's path to the Executor.

**Validation check:** Before handing off to the Executor, verify that all `M` shard files exist in the target directory. If any are missing, the download is incomplete — don't proceed.

---

## Phase E — Integration with the Existing Pipeline

### Where the Download Manager Sits

```
Before Step 8:
    Profile → Fetch → Matrix → Predict → Rank → User Selects → 
    Executor (expects file already on disk) → Calibrate

After Step 8:
    Profile → Fetch → Matrix → Predict → Rank → User Selects → 
    DOWNLOAD MANAGER (checks space, downloads, verifies) → 
    Executor (receives verified file path) → Calibrate
```

### The New Pipeline Function

```
struct DownloadResult {
    bool success;
    std::string local_path;        // verified file path (first shard for multi-shard)
    std::string error_message;     // empty if successful
    uint64_t bytes_downloaded;     // for logging
    uint64_t bytes_total;          // for logging
    bool was_resumed;              // true if this was a resume, not a fresh download
    bool integrity_verified;       // true if SHA256 matched (false if no hash available)
};

DownloadResult downloadModel(
    const std::string& model_url,
    const std::string& target_directory,
    const HardwareSpec& hardware    // for disk space check
);
```

### The User Interaction Flow

**Scenario 1: Model already on disk**
```
User selects Strategy #1.
Checking for model file... found: C:\dev\models\Llama-3.2-3B-Q4_K_M.gguf
Verifying integrity... ✅ SHA256 matches
Launching executor...
```
No download needed. The download manager detects the existing file, verifies it (if hash is cached), and passes the path to the Executor. This is the fast path.

**Scenario 2: Model not on disk, fresh download**
```
User selects Strategy #1.
Checking for model file... not found locally.
Checking disk space... 24.3 GB available, 2.3 GB required ✅
Downloading Llama-3.2-3B-Q4_K_M.gguf...
[██████████████████████████████] 100% | 2.01 GB | 45 MB/s | Done
Verifying integrity... ✅ SHA256 matches
Launching executor...
```

**Scenario 3: Model partially downloaded**
```
User selects Strategy #1.
Checking for model file... found partial download (1.35 GB / 2.01 GB).
Resuming download...
[██████████████████████████████] 100% | 0.66 GB remaining | 42 MB/s | Done
Verifying integrity... ✅ SHA256 matches
Launching executor...
```

**Scenario 4: Insufficient disk space**
```
User selects Strategy #1.
Checking disk space... 1.2 GB available, 4.8 GB required ❌
Insufficient disk space. Shortfall: 3.6 GB.
Suggestion: Q3_K_M quantization (~3.1 GB) would fit. Re-run with:
  llm-planner --model <url_q3km> --execute
```

### The `--download-dir` Flag
Add a new CLI flag to control where models are saved:
```
llm-planner --model <url> --execute --download-dir D:\models\
```
Default to the models directory from Step 0 (`C:\dev\models\` or equivalent).

### The `--skip-verify` Flag (Escape Hatch)
For advanced users who want to skip SHA256 verification (e.g., downloading from a non-HF source without hashes):
```
llm-planner --model <url> --execute --skip-verify
```
Print a warning when this flag is used. Don't make it the default.

---

## Phase F — Edge Cases and Robustness

### Edge Case 1: Hugging Face Rate Limiting
Hugging Face's CDN may rate-limit aggressive downloads, especially for unauthenticated users. Symptoms: HTTP 429 (Too Many Requests) or sudden speed drops to zero.

**Handling:**
- If you get HTTP 429, back off for 30 seconds and retry
- libcurl's `CURLOPT_LOW_SPEED_LIMIT` will catch sustained zero-speed situations
- Consider adding an optional `--hf-token <token>` flag for authenticated downloads (higher rate limits)

### Edge Case 2: Redirect Chains
Hugging Face URLs redirect through multiple CDN nodes. The initial URL redirects to `cdn-lfs.huggingface.co`, which may redirect again to a regional edge server. libcurl handles this with `CURLOPT_FOLLOWLOCATION`, but:

- Set `CURLOPT_MAXREDIRS` to 10 to prevent infinite redirect loops
- The `Content-Length` header may only be available on the final redirect target, not the initial URL. The HEAD request in Phase A should follow redirects to get the real size.

### Edge Case 3: Filename Conflicts
Two different models might have the same filename (e.g., two different repos both have a file called `model-Q4_K_M.gguf`).

**Handling:** Use a subdirectory structure based on the repository name:
```
C:\dev\models\bartowski\Llama-3.2-3B-Instruct-GGUF\Llama-3.2-3B-Q4_K_M.gguf
C:\dev\models\MaziyarPanahi\Qwen2.5-7B-GGUF\Qwen2.5-7B-Q4_K_M.gguf
```
Extract the repo owner and name from the Hugging Face URL to construct the subdirectory path.

### Edge Case 4: Disk Space Changes During Download
The free space check in Phase A is a point-in-time snapshot. A long download (30+ minutes for large models) can be interrupted by the user's OS writing updates, browser caches growing, or other applications consuming space.

**Handling:**
- Re-check disk space every 500MB of downloaded data (in the progress callback)
- If free space drops below the remaining download size + 100MB buffer, pause the download and warn the user
- Don't abort — the `.partial` file is preserved and the user can resume after freeing space

### Edge Case 5: Antivirus Interference
Windows Defender or third-party antivirus may scan the `.partial` file during download, causing speed drops or even quarantining the file mid-transfer.

**Handling:**
- This is why Step 0 recommended excluding the models folder from Defender
- If the download speed drops to zero for >60 seconds (triggering the low-speed timeout), print a suggestion: `"⚠️ Download stalled. Check if your antivirus is scanning the models folder."`

---

## Phase G — Testing the Download Manager

### Test Scenarios

**Test 1: Fresh Download (Small Model)**
- Delete any existing model files
- Run the pipeline with `--execute` for a ~2GB model
- Verify: Disk space check passes, download completes, SHA256 matches, file renamed from `.partial`, executor launches successfully

**Test 2: Resume After Interruption**
- Start a download for a ~4GB model
- Press Ctrl+C at ~50% progress
- Verify: `.partial` file exists with ~2GB of data
- Re-run the same command
- Verify: Download resumes from ~50%, not from 0%
- Verify: Final file passes SHA256 verification

**Test 3: Resume After Network Drop**
- Start a download
- Disconnect the network cable / disable WiFi at ~30%
- Verify: Download fails gracefully, `.partial` file preserved
- Reconnect network, re-run
- Verify: Resumes from ~30%

**Test 4: Disk Space Insufficient**
- Fill the target drive until <2GB free (or use a small USB drive)
- Run the pipeline for a 4GB model
- Verify: Tool refuses with clear error message and shortfall amount
- Verify: Suggestion for smaller quant is printed

**Test 5: SHA256 Mismatch**
- Download a model successfully
- Manually corrupt the `.partial` file before the rename (flip a byte with a hex editor)
- Re-run the verification step
- Verify: Hash mismatch detected, corrupted file deleted, clear error message

**Test 6: Multi-Shard Download**
- Run the pipeline for a multi-shard model (if you have the disk space)
- Verify: All shards downloaded, each verified independently
- Verify: Executor receives the first shard path and loads successfully

**Test 7: Model Already on Disk**
- Run the pipeline for a model that's already downloaded and verified
- Verify: No re-download occurs, integrity check passes quickly, executor launches immediately

**Test 8: Progress Display**
- Run a download and observe the progress bar
- Verify: Percentage, speed, and ETA are reasonable and update smoothly
- Verify: Progress bar doesn't flicker or corrupt the console output

**Test 9: Hugging Face Redirect**
- Use a URL that goes through multiple redirects
- Verify: libcurl follows all redirects, download completes successfully
- Verify: File size from HEAD request matches actual download size

**Test 10: `--skip-verify` Flag**
- Run with `--skip-verify`
- Verify: Download completes without hashing step
- Verify: Warning message is printed about unverified integrity

---

## Step 8 — Done Checklist

Before moving to Step 9, confirm every item:

- [ ] HEAD request retrieves exact file size from Hugging Face CDN
- [ ] `GetDiskFreeSpaceExW()` reports accurate free space on target drive
- [ ] 1.15× safety margin correctly blocks downloads when space is insufficient
- [ ] Refusal message includes shortfall amount and smaller-quant suggestion
- [ ] Fresh download writes to `.partial` filename
- [ ] Progress bar displays percentage, speed, and ETA
- [ ] Progress updates are throttled (no console flickering)
- [ ] Ctrl+C pauses download and preserves `.partial` file
- [ ] Re-running after Ctrl+C resumes from the correct byte offset
- [ ] Network drop preserves `.partial` file for later resume
- [ ] SHA256 hash fetched from Hugging Face API before download
- [ ] Local file hashed using Windows CNG API (BCrypt)
- [ ] Hash comparison is case-insensitive hex string match
- [ ] `.partial` renamed to final filename only after hash match
- [ ] Hash mismatch deletes the corrupted file and prints clear error
- [ ] Missing hash (non-HF source) prints warning but proceeds
- [ ] Multi-shard detection works via filename pattern and/or repo API
- [ ] All shards downloaded and verified before reporting "ready"
- [ ] Executor receives correct file path (first shard for multi-shard)
- [ ] Already-downloaded model is detected and not re-downloaded
- [ ] `--download-dir` flag overrides default target directory
- [ ] `--skip-verify` flag skips SHA256 check with warning
- [ ] Filename conflicts handled via repo-based subdirectories
- [ ] Disk space re-checked periodically during long downloads
- [ ] Tested with at least one fresh download, one resume, and one integrity failure

---

## Common Failure Points at Step 8

| Problem | Likely Cause | Fix |
|---|---|---|
| Resume downloads a corrupted file | Resume offset is off by one byte | Use `CURLOPT_RESUME_FROM_LARGE` with the exact `.partial` file size. Don't subtract 1. The HTTP Range header is inclusive, but libcurl handles this correctly when you use the resume option. |
| `Content-Length` header missing | Server doesn't support HEAD requests, or redirect chain strips the header | Fall back to the Content-Range from the initial range request, or estimate from metadata. Flag the size as estimated. |
| SHA256 always mismatches | Hashing the wrong data (e.g., hashing the Git LFS pointer instead of the actual file) | Confirm you're hashing the full file content, not the first 134 bytes (which is the LFS pointer size). Also confirm you're comparing against `lfs.sha256`, not the Git object hash. |
| Download speed is 0 but no error | Antivirus scanning the file, or CDN rate limiting | Check Defender exclusions. Add retry logic with backoff for HTTP 429. |
| `.partial` file grows but final file is smaller | Server sent compressed content (gzip) but libcurl decompressed it transparently | Set `CURLOPT_ACCEPT_ENCODING` to `""` (empty string) to let libcurl negotiate, or explicitly disable compression with `"identity"` to ensure byte-for-byte correspondence with the server's Content-Length. |
| Multi-shard model fails to load in Executor | Shards downloaded to different directories, or first shard path wrong | Ensure all shards are in the same directory. Pass the `-00001-of-NNNNN.gguf` path to the Executor. |
| `std::filesystem::rename()` fails | Source and destination on different volumes (e.g., download to C:, models on D:) | Fall back to copy + delete. Or: download directly to the target volume. |
| Progress bar shows >100% | `dltotal` from libcurl is 0 or incorrect (some CDN configurations don't report total size) | Guard against `dltotal == 0`. If total is unknown, show bytes downloaded and speed without percentage. |
| Disk space check passes but download fails mid-way | Another application consumed space between check and download | Re-check space periodically during download (Phase F, Edge Case 4). |
| Download works but Executor can't find the file | Path encoding issue (Unicode characters in repo name) | Use `std::filesystem::path` with wide strings (`std::wstring`) on Windows. Ensure the path passed to llama.cpp is UTF-8 encoded (llama.cpp expects UTF-8, not Windows wide chars). |

---

## Time Estimate for Step 8

| Phase | Task | Time |
|---|---|---|
| A | Pre-download safety check (HEAD request, disk space, refusal logic) | 2–3 hours |
| B | Resumable download (libcurl config, `.partial` files, progress bar, interrupt handling) | 1–2 days |
| C | Integrity verification (HF hash API, Windows CNG SHA256, rename logic) | 3–4 hours |
| D | Multi-shard handling (detection, sequential download, per-shard verification) | 3–4 hours |
| E | Pipeline integration (DownloadResult struct, user interaction flow, CLI flags) | 2–3 hours |
| F | Edge cases (rate limiting, redirects, filename conflicts, mid-download space check) | 2–3 hours |
| G | Testing (all 10 test scenarios) | 3–4 hours |

**Total: 2–3 days, as estimated. The resumable download (Phase B) is the bulk of the work — getting the progress reporting, interrupt handling, and resume offset correct takes careful testing. The rest is straightforward API integration.**