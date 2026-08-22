# Step 2 — Metadata Fetcher Test Results

## Test Matrix Summary

| # | Model | Architecture | Size | Fetch | HTTP | Success |
|---|-------|-------------|------|-------|------|---------|
| 1 | Llama-3.2-3B-Instruct-Q4_K_M.gguf | llama | ~3B | 64KB Range | 206 | ✅ |
| 2 | Qwen2.5-7B-Instruct-Q4_K_M.gguf | qwen2 | ~7B | 64KB Range | 206 | ✅ |
| 3 | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | llama | ~7B | 64KB Range | 206 | ✅ |
| 4 | Phi-3.5-mini-instruct-Q4_K_M.gguf | phi3 | ~3.8B | 64KB Range | 206 | ✅ |

## Detailed Results

### Model 1: Llama-3.2-3B-Instruct
```
Architecture:    llama
Quantization:    Q4_K_M (file_type=15)
Layers:          28
Embedding Dim:   3072
Attention Heads: 24
KV Heads:        8 (GQA ratio 3:1)
FFN Dim:         8192
Context Length:  131,072 (128K)
Head Dim:        128
```
**Validation:** ✅ All values match published specs

### Model 2: Qwen2.5-7B-Instruct
```
Architecture:    qwen2
Quantization:    Q4_K_M (file_type=15)
Layers:          28
Embedding Dim:   3584
Attention Heads: 28
KV Heads:        4 (GQA ratio 7:1)
FFN Dim:         18944
Context Length:  32,768 (32K)
Head Dim:        128
```
**Validation:** ✅ Architecture correctly detected as "qwen2" (different from llama)
- GQA ratio 7:1 is correct for Qwen2.5-7B
- Context 32K matches model card

### Model 3: Mistral-7B-Instruct-v0.3
```
Architecture:    llama  (NOT "mistral"!)
Quantization:    Q4_K_M (file_type=15)
Layers:          32
Embedding Dim:   4096
Attention Heads: 32
KV Heads:        8 (GQA ratio 4:1)
FFN Dim:         14336
Context Length:  32,768 (32K)
Head Dim:        128
```
**Validation:** ✅ Architecture tag is "llama" (expected — Mistral uses llama architecture internally)
- 32 layers matches published specs
- GQA ratio 4:1 is correct

### Model 4: Phi-3.5-mini-instruct
```
Architecture:    phi3
Quantization:    Q4_K_M (file_type=15)
Layers:          32
Embedding Dim:   3072
Attention Heads: 32
KV Heads:        32 (MHA — no GQA)
FFN Dim:         8192
Context Length:  131,072 (128K)
Head Dim:        96
```
**Validation:** ✅ Architecture correctly detected as "phi3"
- MHA (no GQA) correctly detected — KV Heads == Attention Heads
- Head Dim 96 is unusual (not 128) — correct for Phi-3
- Context 128K matches model card

## Key Observations

1. **Architecture detection works:** llama, qwen2, phi3 all correctly identified
2. **GQA detection works:** Correctly identifies GQA ratios (3:1, 7:1, 4:1) and MHA (1:1)
3. **Quantization detection works:** All models correctly show Q4_K_M (file_type=15)
4. **64KB range request is sufficient:** All metadata fits within 64KB
5. **Mistral uses "llama" architecture tag:** This is expected and documented
6. **Head Dim varies:** Most models use 128, but Phi-3 uses 96 — our derived calculation handles this correctly

## Checklist

- [x] Fetch completes successfully (HTTP 206)
- [x] Magic number validates
- [x] Architecture name matches model card
- [x] Parameter count matches model name (~3B, ~7B, ~3.8B)
- [x] Layer count matches published specs
- [x] Context length matches model card
- [x] Quant type matches filename (Q4_K_M = file_type=15)
