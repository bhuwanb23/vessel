# Step 2 — Metadata Fetcher: Full Detailed Plan

---

## Goal of Step 2
Build a standalone C++ binary that, given a Hugging Face GGUF URL, downloads **only the header** (a few kilobytes) via an HTTP range request, parses the GGUF binary format, and prints the model's self-described metadata: architecture, parameter count, layer count, quantization type, context length, and attention dimensions. No model weights are downloaded. No prediction math is performed. This step proves you can read a model's description of itself without pulling the full multi-gigabyte file.

---

## What You Need Before Starting

### From Steps 0–1 (already done)
- Working MSVC + CMake toolchain
- Your project skeleton builds cleanly
- Your hardware profiler binary works (Step 1 is complete but not used in this step — these two steps are independent)

### New Dependencies to Acquire During This Step
- **libcurl** — the HTTP library for making range requests. You need the Windows build.
- **Understanding of the GGUF binary format** — this is a custom binary format, not JSON, not protobuf. You will be parsing raw bytes.

---

## Phase A — Understand the GGUF File Format

Before writing any code, you must understand what you are parsing. The GGUF format is documented in the llama.cpp repository at `docs/gguf.md`. Read that document fully before proceeding. What follows is the subset you need for this step.

### The GGUF Header Structure (First Bytes of the File)

The file begins with a fixed header, followed by a variable-length metadata section, followed by tensor info, followed by the actual weight data (which you will never download in this step).

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 bytes | Magic | Always `0x46554747` (ASCII "GGUF" in little-endian). If this doesn't match, it's not a GGUF file. |
| 4 | 4 bytes | Version | GGUF format version. Currently `3`. If you see `2`, it's an old file. If you see `1`, it's ancient and likely broken. |
| 8 | 8 bytes | n_tensors | Number of tensors in the file (uint64, little-endian). You don't need this for metadata extraction but you need to skip past it. |
| 16 | 8 bytes | n_kv | Number of key-value metadata pairs (uint64, little-endian). **This tells you how many metadata entries to parse.** |
| 24 | variable | KV pairs | The actual metadata, stored as a sequence of key-value pairs. |

### The Key-Value Pair Format

Each KV pair is stored as:
1. **Key:** a GGUF string (length-prefixed: 8-byte uint64 length, then that many UTF-8 bytes, no null terminator)
2. **Value type:** a 4-byte uint32 enum indicating the data type
3. **Value:** the actual data, format depends on the type

### The Value Type Enum (Critical to Get Right)

| Type ID | Name | Size | Notes |
|---|---|---|---|
| 0 | UINT8 | 1 byte | |
| 1 | INT8 | 1 byte | |
| 2 | UINT16 | 2 bytes | Little-endian |
| 3 | INT16 | 2 bytes | Little-endian |
| 4 | UINT32 | 4 bytes | Little-endian |
| 5 | INT32 | 4 bytes | Little-endian |
| 6 | FLOAT32 | 4 bytes | IEEE 754 |
| 7 | BOOL | 1 byte | 0 = false, nonzero = true |
| 8 | STRING | variable | Length-prefixed (8-byte uint64 length + bytes) |
| 9 | ARRAY | variable | Complex — contains type + count + elements. You need this for some metadata. |
| 10 | UINT64 | 8 bytes | Little-endian |
| 11 | INT64 | 8 bytes | Little-endian |
| 12 | FLOAT64 | 8 bytes | IEEE 754 |

### The Metadata Keys You Care About

GGUF files store metadata as string-keyed pairs. The key names follow a convention: `general.*` for model-wide info, and `{architecture}.*` for architecture-specific info. The architecture name is itself stored in a metadata key, so you must read `general.architecture` first, then use its value to construct the other key names.

**Example:** If `general.architecture` = `"llama"`, then the context length key is `llama.context_length`. If it's `"qwen2"`, the key is `qwen2.context_length`.

| What You Want | Metadata Key | Value Type | Example Value |
|---|---|---|---|
| Architecture family | `general.architecture` | STRING | `"llama"`, `"qwen2"`, `"mistral"`, `"phi3"` |
| Model name | `general.name` | STRING | `"Llama 3.2 3B Instruct"` |
| Parameter count | `general.parameter_count` | UINT64 | `3212749824` (≈3.2B) |
| Context length | `{arch}.context_length` | UINT32 | `131072` (128K) |
| Layer count | `{arch}.block_count` | UINT32 | `28` |
| Embedding dimension | `{arch}.embedding_length` | UINT32 | `3072` |
| Attention heads | `{arch}.attention.head_count` | UINT32 | `24` |
| KV heads | `{arch}.attention.head_count_kv` | UINT32 | `8` (GQA) |
| Feed-forward dim | `{arch}.feed_forward_length` | UINT32 | `8192` |
| Quantization type | `general.file_type` | UINT32 | `15` (maps to Q4_K_M — see lookup table below) |
| RMS norm epsilon | `{arch}.attention.layer_norm_rms_epsilon` | FLOAT32 | `0.00001` |

### The Quantization Type Lookup Table

The `general.file_type` value is an integer enum. You need a lookup table to translate it to a human-readable name. The most common values:

| file_type | Name | Description |
|---|---|---|
| 0 | F32 | Full precision float32 |
| 1 | F16 | Half precision float16 |
| 2 | Q4_0 | 4-bit, oldest quant |
| 3 | Q4_1 | 4-bit with min |
| 6 | Q5_0 | 5-bit |
| 7 | Q5_1 | 5-bit with min |
| 8 | Q8_0 | 8-bit |
| 10 | Q2_K | 2-bit k-quant |
| 11 | Q3_K_S | 3-bit k-quant small |
| 12 | Q3_K_M | 3-bit k-quant medium |
| 13 | Q3_K_L | 3-bit k-quant large |
| 14 | Q4_K_S | 4-bit k-quant small |
| 15 | Q4_K_M | 4-bit k-quant medium (most popular) |
| 16 | Q4_K_L | 4-bit k-quant large |
| 17 | Q5_K_S | 5-bit k-quant small |
| 18 | Q5_K_M | 5-bit k-quant medium |
| 19 | Q6_K | 6-bit k-quant |
| 20 | IQ2_XXS | 2-bit imatrix |
| 21 | IQ2_XS | 2-bit imatrix |
| 22 | IQ3_XXS | 3-bit imatrix |
| 23 | IQ1_S | 1-bit imatrix |
| 24 | IQ4_NL | 4-bit imatrix |
| 28 | Q4_0_4_4 | 4-bit repacked |
| 29 | Q4_0_4_8 | 4-bit repacked |
| 30 | Q4_0_8_8 | 4-bit repacked |

**For MVP:** Hardcode the 15-20 most common entries. If you encounter an unknown `file_type`, print the raw integer and "unknown quant type" — don't crash.

---

## Phase B — Understand HTTP Range Requests

### What a Range Request Is
A standard HTTP GET downloads the entire file. A range request tells the server "I only want bytes X through Y." This is how you download a 64KB header from a 4GB file.

### The HTTP Header You Need
```
Range: bytes=0-65535
```
This asks for the first 64KB of the file. For virtually all GGUF files, the entire metadata section fits within the first 64KB. The metadata is at the very beginning of the file, before any tensor data.

**Edge case:** Some extremely large models (70B+ with many tensors) might have metadata sections that exceed 64KB. For MVP, 64KB is sufficient. If you hit this edge case later, you can increase to 256KB or 1MB — still a tiny fraction of the full file.

### The Server Response
- **Success:** HTTP 206 Partial Content. The response body contains exactly the bytes you requested. The response header includes `Content-Range: bytes 0-65535/4294967296` (the last number is the total file size — useful to know but not critical for this step).
- **Failure:** HTTP 200 OK with the full file body. This means the server doesn't support range requests. Hugging Face **does** support range requests, so this shouldn't happen in normal operation, but your code should detect it (if you get a 200 instead of a 206, you're about to download the entire file — abort immediately).

### How Hugging Face URLs Work
A typical Hugging Face GGUF URL looks like:
```
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Key parts:
- `bartowski/Llama-3.2-3B-Instruct-GGUF` — the repository
- `resolve/main/` — the branch (usually `main`)
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` — the specific file

Your fetcher takes this full URL as input. No authentication is needed for public models.

---

## Phase C — Set Up libcurl on Windows

### Downloading libcurl for Windows
You need a pre-built Windows binary of libcurl. The cleanest source is the official curl website.

**Download from:** [curl.se/windows](https://curl.se/windows/)

**What you get:** A zip file containing:
- `bin/curl.exe` (the command-line tool — useful for testing, not needed in your code)
- `include/curl/` (header files)
- `lib/` (import libraries: `libcurl.lib` or `libcurl_a.lib` for static linking)

**Recommendation:** Use the **static** build (`libcurl_a.lib`) to avoid DLL dependency issues. This means libcurl is compiled into your binary and you don't need to ship a separate `libcurl.dll`.

**Alternative:** Use vcpkg (`vcpkg install curl`) if you already have vcpkg set up. This handles include paths and linking automatically. Either approach works.

### CMake Configuration
You need to tell CMake:
1. Where the curl headers are (`target_include_directories`)
2. Where the curl library is (`target_link_libraries`)
3. On Windows, libcurl depends on a few system libraries: `ws2_32` (Winsock), `crypt32` (for HTTPS certificate validation), `wldap32` (LDAP, sometimes needed), `normaliz` (IDN support). Link these too or you will get unresolved symbol errors.

### Quick Smoke Test Before Proceeding
Before writing the GGUF parser, confirm libcurl works at all. Write a 20-line program that fetches `https://huggingface.co` and prints the HTTP status code. If you get 200, libcurl is working. If you get SSL errors, you may need to configure the CA certificate bundle (curl ships one — point libcurl to it via `CURLOPT_CAINFO`).

---

## Phase D — Build the Fetcher Logic

### The Fetch Sequence (Step by Step)

1. **Initialize libcurl** — call `curl_global_init(CURL_GLOBAL_DEFAULT)` once at program start
2. **Create a curl handle** — `curl_easy_init()`
3. **Set the URL** — `curl_easy_setopt(curl, CURLOPT_URL, url)`
4. **Set the range** — `curl_easy_setopt(curl, CURLOPT_RANGE, "0-65535")`
5. **Set a write callback** — this is a function that libcurl calls every time it receives a chunk of data. Your callback appends the received bytes to a `std::vector<uint8_t>` buffer. This is how you capture the response body in memory.
6. **Set a header callback** (optional but recommended) — capture the HTTP status code to verify you got a 206, not a 200
7. **Set timeouts** — `CURLOPT_TIMEOUT` to 30 seconds, `CURLOPT_CONNECTTIMEOUT` to 10 seconds. Don't hang forever if the server is slow.
8. **Follow redirects** — `curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)`. Hugging Face sometimes redirects to a CDN.
9. **Execute** — `curl_easy_perform(curl)`
10. **Check the result** — verify `CURLE_OK` return code and HTTP 206 status
11. **Clean up** — `curl_easy_cleanup(curl)` and `curl_global_cleanup()`

### The Write Callback (Critical Detail)
libcurl doesn't store response data for you. You must provide a callback function with this signature:

```
size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata)
```

- `ptr` points to the received data chunk
- `size × nmemb` is the number of bytes in this chunk
- `userdata` is a pointer you set via `CURLOPT_WRITEDATA` — pass your `std::vector<uint8_t>*` here
- Your callback appends the chunk to the vector and returns `size * nmemb`
- **If you return a different number, libcurl treats it as an error and aborts**

### Safety Check: Abort If Full Download Starts
In your write callback, check the total accumulated size. If it exceeds your range request (64KB) by a large margin, something is wrong (the server ignored the range header and is sending the full file). Abort the transfer by returning 0 from the callback. This prevents accidentally downloading a 4GB file into RAM.

---

## Phase E — Build the GGUF Parser

### The Parser Logic (Step by Step)

Once you have the raw bytes in your `std::vector<uint8_t>`, parse them sequentially:

1. **Read magic** (bytes 0–3): Cast to `uint32_t`, check it equals `0x46554747`. If not, print "Not a valid GGUF file" and exit.

2. **Read version** (bytes 4–7): Cast to `uint32_t`. Confirm it's `2` or `3`. Version 1 is deprecated and structurally different — don't try to support it.

3. **Read n_tensors** (bytes 8–15): Cast to `uint64_t`. Store it but you won't use it in this step.

4. **Read n_kv** (bytes 16–23): Cast to `uint64_t`. This is how many key-value pairs to parse.

5. **Parse KV pairs** (starting at byte 24): Loop `n_kv` times. For each iteration:
   - Read the key string (8-byte length + that many bytes)
   - Read the value type (4-byte uint32)
   - Read the value based on its type (using the type enum from Phase A)
   - Store the key-value pair in a `std::map<std::string, Variant>` or similar structure

### The Variant Problem
GGUF values can be strings, integers, floats, booleans, or arrays. C++ doesn't have a built-in variant type that covers all of these cleanly (well, `std::variant` exists in C++17, which you are using). You have a few options:

| Approach | Pros | Cons |
|---|---|---|
| `std::variant<uint64_t, int64_t, double, std::string, bool>` | Type-safe, modern C++ | Slightly verbose to extract values |
| Custom union + type tag | Simple, fast | Manual memory management for strings |
| `std::map<std::string, std::string>` storing everything as strings | Dead simple | Lose numeric precision, have to re-parse later |

**Recommendation for MVP:** Use `std::variant`. It's clean, type-safe, and you're already on C++17. The verbosity is manageable since you only extract ~10 specific keys.

### The Two-Pass Parsing Strategy
Remember that the architecture-specific keys depend on the value of `general.architecture`, which is itself a metadata key. You have two options:

**Option A: Two passes.** First pass: parse all KV pairs into a map. Second pass: look up `general.architecture`, construct the architecture-specific key names, and extract the values you care about.

**Option B: One pass, store everything.** Parse all KV pairs into a map in one pass. Then query the map for the keys you need, constructing the architecture-specific names after reading `general.architecture`.

**These are the same thing.** Option B is the natural approach — parse everything into a map, then query. Don't overthink this.

### Array Values (The Tricky Type)
Some metadata values are arrays (type 9). The array format is:
1. Element type (4-byte uint32)
2. Element count (8-byte uint64)
3. That many elements of the given type

For MVP, you likely don't need to parse arrays deeply. The most important metadata (architecture, layers, context, quant) are all scalar values. If you encounter an array type, you can skip it by reading the element type and count, then advancing the read pointer by `count × element_size`. Just don't crash on it.

---

## Phase F — The config.json Fallback Path

### When This Path Activates
From §3 item 1: "If no GGUF exists, there's nothing to read a header from." Some models on Hugging Face are only available as safetensors (the original format), not as pre-quantized GGUF files. In that case, you can't do a GGUF header fetch.

### What config.json Contains
Every Hugging Face model repository has a `config.json` file in the root. It's a plain JSON file (not binary) that describes the model architecture. You can fetch it with a simple HTTP GET (no range request needed — it's typically <5KB).

**Example URL:**
```
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json
```

**Key fields in config.json:**
| Field | Maps To | Example |
|---|---|---|
| `model_type` | Architecture | `"llama"` |
| `num_hidden_layers` | Layer count | `28` |
| `hidden_size` | Embedding dimension | `3072` |
| `num_attention_heads` | Attention heads | `24` |
| `num_key_value_heads` | KV heads | `8` |
| `max_position_embeddings` | Context length | `131072` |
| `intermediate_size` | Feed-forward dim | `8192` |

### What config.json Does NOT Contain
- **Parameter count** — you have to calculate it from the architecture dimensions, or read it from the model card (not machine-readable)
- **Quantization type** — safetensors models are typically F16 or BF16, not quantized. The quant type only exists after someone converts to GGUF.
- **File size** — you'd need to sum up all the safetensors shard files

### Implementation Priority
**For MVP, the config.json fallback is lower priority than the GGUF path.** Implement the GGUF path first, test it thoroughly, then add config.json as a secondary code path. The spec doc flags config.json predictions as "low confidence" anyway.

**How to structure it:** Your fetcher takes a URL. If the URL ends in `.gguf`, use the GGUF header path. If it points to a repository root (no `.gguf` extension), try fetching `config.json` from that repository. Parse with nlohmann::json (which you already have in your tech stack).

---

## Phase G — Output Format

### What to Print
Design a clean report similar to Step 1's hardware profile:

```
=== LLM Deployment Planner — Model Metadata ===
Source: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
Fetch: 64KB via HTTP Range (206 Partial Content)

--- Model Identity ---
Name:            Llama 3.2 3B Instruct
Architecture:    llama
Quantization:    Q4_K_M (file_type=15)

--- Dimensions ---
Parameters:      3,212,749,824 (~3.2B)
Layers:          28
Embedding Dim:   3072
Attention Heads: 24
KV Heads:        8 (GQA ratio 3:1)
FFN Dim:         8192
Context Length:  131,072 (128K)

--- Derived ---
Head Dim:        128  (embedding_dim / attention_heads)
KV Dim per Head: 128
=================================================
```

### Important Details
- Show the **fetch method** (range request vs full GET) and **HTTP status code** — this builds user trust
- Show the **raw file_type integer** alongside the human-readable name — helps with debugging unknown quant types
- Show **derived values** like head dimension — these are needed for the KV cache formula in Step 3
- The GQA ratio (attention_heads / kv_heads) is worth highlighting because it directly affects KV cache size

---

## Phase H — Testing Against Real Models

### The Test Matrix
Test against **3–4 models** that cover different architectures and sizes. This validates that your parser handles the structural variations between model families.

| # | Model | Architecture | Size | Why This Model |
|---|---|---|---|---|
| 1 | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | llama | ~3B | The reference architecture. Most common. |
| 2 | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | qwen2 | ~7B | Different architecture family. Tests that your `{arch}.*` key construction works. |
| 3 | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | llama (or mistral) | ~7B | Tests a model that may use the "mistral" architecture tag despite being structurally similar to Llama. |
| 4 | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | phi3 | ~3.8B | Tests a less common architecture with potentially different metadata key names. |

### Where to Find These
Search Hugging Face for the model name + "GGUF". Popular quantizers who publish GGUF versions include `bartowski`, `MaziyarPanahi`, and `TheBloke` (older models). Look for repositories with "GGUF" in the name.

### What to Check for Each Model
1. Does the fetch complete successfully (HTTP 206)?
2. Does the magic number validate?
3. Does the architecture name match what the model card says?
4. Does the parameter count match the model name (e.g., "3B" model should show ~3 billion params)?
5. Does the layer count match published specs?
6. Does the context length match the model card?
7. Does the quant type match the filename (e.g., "Q4_K_M" in filename should show file_type=15)?

---

## Phase I — Validation Against Ground Truth

### How to Validate
For each test model, compare your fetcher's output against two independent sources:

**Source 1: The Hugging Face model card.** The repository's README usually lists architecture, parameter count, context length, and layer count.

**Source 2: llama.cpp itself.** From Step 0, you know that when you run `llama-cli` with a model, it prints the model's metadata during loading. Run each test model through `llama-cli` (you don't need to generate tokens — just let it load and print the header info, then Ctrl+C). Compare the numbers.

### Validation Table

| Field | Your Fetcher | Model Card | llama.cpp Output | Match? |
|---|---|---|---|---|
| Architecture | | | | |
| Parameters | | | | |
| Layers | | | | |
| Context Length | | | | |
| Attention Heads | | | | |
| KV Heads | | | | |
| Embedding Dim | | | | |
| Quant Type | | | | |

Fill this out for all 3-4 test models. Every field should match across all three sources.

---

## Step 2 — Done Checklist

Before moving to Step 3, confirm every item:

- [ ] libcurl is linked into your project and the binary builds cleanly
- [ ] HTTPS requests work (SSL certificates are configured correctly)
- [ ] Range request to a Hugging Face GGUF URL returns HTTP 206
- [ ] Safety abort triggers if the server returns HTTP 200 (full file) instead of 206
- [ ] GGUF magic number is validated correctly
- [ ] GGUF version is checked (accept 2 and 3, reject 1)
- [ ] All 10 target metadata keys are extracted correctly for at least 3 different models
- [ ] Architecture-specific key construction works (e.g., `llama.context_length` vs `qwen2.context_length`)
- [ ] Quantization type is translated from integer to human-readable name
- [ ] Derived values (head dimension, GQA ratio) are calculated and displayed
- [ ] Output matches Hugging Face model card for all test models
- [ ] Output matches llama.cpp's own reported metadata for all test models
- [ ] Non-GGUF URLs are handled gracefully (error message, not crash)
- [ ] Network timeout works (doesn't hang forever on unreachable URLs)
- [ ] config.json fallback path is implemented (even if basic) and flagged as low-confidence

---

## Common Failure Points at Step 2

| Problem | Likely Cause | Fix |
|---|---|---|
| `curl_easy_perform()` returns `CURLE_SSL_CACERT` | No CA certificate bundle configured | Download `cacert.pem` from [curl.se/ca](https://curl.se/ca/cacert.html) and set `CURLOPT_CAINFO` to its path |
| HTTP 200 instead of 206 | Server doesn't support range requests, or URL is wrong | Verify URL points to a specific file (not a directory listing). Hugging Face supports ranges for file URLs with `/resolve/` in the path. |
| Magic number doesn't match | Byte order issue, or you downloaded an HTML error page instead of the file | Print the first 20 bytes of the response as hex. If you see `<html>` or `<!DOCTYPE`, you got an error page, not a GGUF file. |
| Metadata values are garbage | Reading past the end of the buffer, or misaligned reads | Add bounds checking before every read. The header might be shorter than 64KB for small models. |
| String values contain garbage characters | Not handling the length-prefixed string format correctly | GGUF strings are NOT null-terminated. Read exactly `length` bytes and construct a `std::string` from them. Don't use `strlen()` or `strcpy()`. |
| `general.parameter_count` is missing | Not all GGUF files include this key | Some older GGUF files omit `parameter_count`. In that case, you'll need to estimate it from dimensions (layers × embedding × ffn) or mark it as "unknown." |
| Architecture key mismatch | Model uses a non-standard architecture name | Print all metadata keys when debugging. Some models use unexpected names like `"llama"` vs `"llama3"` vs `"llama-3"`. |
| libcurl links but crashes at runtime | Mixing debug/release builds, or wrong CRT (C Runtime) | Ensure libcurl was built with the same MSVC configuration (Debug/Release, MD/MT) as your project. |
| Very slow fetch (>10 seconds for 64KB) | Hugging Face CDN is slow, or you're not following redirects efficiently | Confirm `CURLOPT_FOLLOWLOCATION` is set. Hugging Face redirects to `cdn-lfs.huggingface.co` which is faster. |

---

## Time Estimate for Step 2
- Phase A (GGUF format understanding): **1–2 hours** (reading the spec and understanding the binary layout)
- Phase B (HTTP range request understanding): **30 minutes**
- Phase C (libcurl setup on Windows): **2–3 hours** (downloading, CMake config, SSL certificate debugging)
- Phase D (Fetcher logic): **2–3 hours**
- Phase E (GGUF parser): **4–6 hours** (this is the core complexity — binary parsing with variable-length types)
- Phase F (config.json fallback): **1–2 hours**
- Phase G (Output formatting): **1 hour**
- Phase H (Testing against 3-4 models): **2–3 hours** (downloading test models + running comparisons)
- Phase I (Validation): **1–2 hours**

**Total: 1–2 days, as originally estimated. The GGUF binary parser (Phase E) is where most of the time goes — getting the byte-level reads right for all value types takes careful debugging.**