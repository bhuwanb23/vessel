# Step 10 — Design Decisions

## Overview

This document records the key design decisions for Step 10 (Hot/Cold CPU-GPU Offload for Dense Models).

## Decision 1: Integration Approach

**Options:**
- (a) Merge PowerInfer's fork patches into our llama.cpp build
- (b) Implement sparse FFN from scratch against ggml

**Decision: (a) Study and adapt PowerInfer's fork**

**Rationale:**
- PowerInfer authors spent months tuning sparse operators
- Reimplementing from the paper alone will take longer and produce slower code
- PowerInfer is a fork of llama.cpp, so integration is natural
- We can cherry-pick specific optimizations without taking the entire fork

**Implementation Plan:**
1. Clone PowerInfer repository
2. Study `ggml/src/ggml-cuda.cu` for sparse CUDA kernels
3. Study `ggml/src/ggml.cpu.cpp` for sparse CPU kernels
4. Extract relevant sparse operators
5. Adapt to our llama.cpp build

## Decision 2: Profiling Prompt Set

**Options:**
- (a) Bundled general-purpose prompts (500-1000)
- (b) User-provided prompts

**Decision: (a) Bundled with user override option**

**Rationale:**
- Hot set is stable across prompts (paper confirms this)
- Bundled prompts cover diverse domains (Wikipedia, code, conversation, Q&A)
- User override for domain-specific optimization (e.g., medical, legal)

**Implementation Plan:**
1. Create `hotcold/prompts/default.txt` with 1000 diverse prompts
2. Include prompts from: Wikipedia, StackOverflow, books, conversations, Q&A
3. Allow `--hotcold-prompts <file>` flag for custom prompts
4. Default prompts are ~20 tokens each (50-100 words)

## Decision 3: Hot Set Storage

**Options:**
- (a) Per-model binary mask file (separate from GGUF)
- (b) Embedded in the GGUF

**Decision: (a) Separate `.hot_neurons.bin` file**

**Rationale:**
- Don't modify GGUF format (avoids compatibility issues)
- Hot set can be regenerated without re-downloading model
- Multiple hot sets can exist for same model (different VRAM budgets)
- Simpler implementation

**File Format:**
```
model_name.hot_neurons.bin
├── Header (64 bytes)
│   ├── magic: "HOTN" (4 bytes)
│   ├── version: uint32
│   ├── num_layers: uint32
│   ├── ffn_dim: uint32
│   ├── hidden_dim: uint32
│   └── reserved: 44 bytes
├── Layer 0 hot neuron indices (num_hot_0 × uint32)
├── Layer 1 hot neuron indices (num_hot_1 × uint32)
├── ...
└── Layer N hot neuron indices (num_hot_N × uint32)
```

## Decision 4: Granularity

**Options:**
- (a) Per-layer hot sets
- (b) Global hot set

**Decision: (a) Per-layer hot sets**

**Rationale:**
- Different layers have different activation patterns
- Layer 0 might have 15% hot neurons, Layer 20 might have 8%
- Global set wastes VRAM on layers where hot neurons are different
- PowerInfer uses per-layer profiling

**Implementation:**
- Each layer has its own hot neuron count: `n_hot[layer]`
- Hot set size is determined by VRAM budget per layer
- Total hot neurons across all layers = sum(n_hot[layer])

## Decision 5: Activation Function Handling

**Options:**
- (a) ReLU only (sparse by nature)
- (b) SiLU/Swish only (Llama models)
- (c) Handle both

**Decision: (c) Handle both**

**Rationale:**
- ReLU: exact zeros (neurons with output ≤ 0 are inactive)
- SiLU: never exactly zero, but near-zero for negative inputs
- Need threshold for SiLU: neurons with pre-activation < threshold are "cold"
- Most modern models use SiLU (Llama, Mistral, Qwen)

**Implementation:**
```cpp
// ReLU: exact sparsity
if (activation == RELU) {
    is_cold = (pre_activation <= 0);
}

// SiLU: threshold-based sparsity
if (activation == SILU) {
    is_cold = (pre_activation < SILU_THRESHOLD);  // threshold = -2.0
}
```

## Summary Table

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration | Adapt PowerInfer fork | Months of optimization already done |
| Profiling | Bundled prompts + user override | Hot set is stable, diverse prompts needed |
| Storage | Separate .hot_neurons.bin | Don't modify GGUF, flexible |
| Granularity | Per-layer | Different layers have different patterns |
| Activation | ReLU + SiLU | Cover both legacy and modern models |
