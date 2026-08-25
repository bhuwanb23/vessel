#pragma once

// =============================================================================
// Embedded Model Catalog (Step 12, Phase A)
// =============================================================================
// The default model catalog is embedded directly in the binary as a string
// constant. This ensures the tool works out of the box without requiring
// an external file.
//
// This file is auto-generated from data/models_catalog.json by CMake.
// Do not edit manually — edit the JSON file instead.
// =============================================================================

static const char* BUILTIN_CATALOG_JSON = R"CATALOG(
{
  "version": "2026.08.25",
  "models": [
    {
      "id": "llama-3.2-1b-instruct",
      "name": "Llama 3.2 1B Instruct",
      "family": "llama",
      "use_case": ["chat", "instruction", "lightweight"],
      "description": "Ultra-lightweight model. Runs on any hardware.",
      "params_billions": 1.2,
      "architecture": "llama",
      "max_context": 131072,
      "quality_score": 3.5,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 0.7,
          "bpw": 4.85,
          "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
          "hf_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 1.3,
          "bpw": 8.5,
          "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
          "hf_file": "Llama-3.2-1B-Instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 16,
        "embedding_dim": 2048,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 64,
        "ffn_dim": 5632
      },
      "is_moe": false
    },
    {
      "id": "llama-3.2-3b-instruct",
      "name": "Llama 3.2 3B Instruct",
      "family": "llama",
      "use_case": ["chat", "instruction", "coding"],
      "description": "Fast, capable small model. Best for real-time chat.",
      "params_billions": 3.2,
      "architecture": "llama",
      "max_context": 131072,
      "quality_score": 6.8,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 2.0,
          "bpw": 4.85,
          "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
          "hf_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 3.4,
          "bpw": 8.5,
          "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
          "hf_file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 28,
        "embedding_dim": 3072,
        "attention_heads": 24,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 8192
      },
      "is_moe": false
    },
    {
      "id": "llama-3.1-8b-instruct",
      "name": "Llama 3.1 8B Instruct",
      "family": "llama",
      "use_case": ["chat", "instruction", "coding", "reasoning"],
      "description": "Strong all-around model. Great balance of quality and speed.",
      "params_billions": 8.0,
      "architecture": "llama",
      "max_context": 131072,
      "quality_score": 7.8,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q3_K_M",
          "file_size_gb": 3.5,
          "bpw": 3.69,
          "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
          "hf_file": "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf"
        },
        {
          "quant": "Q4_K_M",
          "file_size_gb": 4.9,
          "bpw": 4.85,
          "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
          "hf_file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q5_K_M",
          "file_size_gb": 5.7,
          "bpw": 5.69,
          "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
          "hf_file": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 8.5,
          "bpw": 8.5,
          "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
          "hf_file": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 32,
        "embedding_dim": 4096,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 14336
      },
      "is_moe": false
    },
    {
      "id": "qwen2.5-3b-instruct",
      "name": "Qwen 2.5 3B Instruct",
      "family": "qwen",
      "use_case": ["chat", "instruction", "coding", "multilingual"],
      "description": "Excellent multilingual model. Strong coding ability.",
      "params_billions": 3.0,
      "architecture": "qwen2",
      "max_context": 32768,
      "quality_score": 6.5,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 1.9,
          "bpw": 4.85,
          "hf_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
          "hf_file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 3.2,
          "bpw": 8.5,
          "hf_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
          "hf_file": "Qwen2.5-3B-Instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 36,
        "embedding_dim": 2048,
        "attention_heads": 16,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 11008
      },
      "is_moe": false
    },
    {
      "id": "qwen2.5-7b-instruct",
      "name": "Qwen 2.5 7B Instruct",
      "family": "qwen",
      "use_case": ["chat", "instruction", "coding", "reasoning", "multilingual"],
      "description": "Best all-around 7B model. Excellent coding and reasoning.",
      "params_billions": 7.6,
      "architecture": "qwen2",
      "max_context": 131072,
      "quality_score": 8.2,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 4.5,
          "bpw": 4.85,
          "hf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
          "hf_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q5_K_M",
          "file_size_gb": 5.4,
          "bpw": 5.69,
          "hf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
          "hf_file": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 28,
        "embedding_dim": 3584,
        "attention_heads": 28,
        "kv_heads": 4,
        "head_dim": 128,
        "ffn_dim": 18944
      },
      "is_moe": false
    },
    {
      "id": "qwen2.5-14b-instruct",
      "name": "Qwen 2.5 14B Instruct",
      "family": "qwen",
      "use_case": ["chat", "instruction", "coding", "reasoning", "multilingual"],
      "description": "Large, high-quality model. Best for complex tasks.",
      "params_billions": 14.7,
      "architecture": "qwen2",
      "max_context": 131072,
      "quality_score": 8.8,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 8.9,
          "bpw": 4.85,
          "hf_repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
          "hf_file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q5_K_M",
          "file_size_gb": 10.5,
          "bpw": 5.69,
          "hf_repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
          "hf_file": "Qwen2.5-14B-Instruct-Q5_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q5_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 40,
        "embedding_dim": 5120,
        "attention_heads": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 13824
      },
      "is_moe": false
    },
    {
      "id": "mistral-7b-instruct-v0.3",
      "name": "Mistral 7B Instruct v0.3",
      "family": "mistral",
      "use_case": ["chat", "instruction", "reasoning"],
      "description": "Proven model with excellent instruction following.",
      "params_billions": 7.2,
      "architecture": "llama",
      "max_context": 32768,
      "quality_score": 7.5,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 4.4,
          "bpw": 4.85,
          "hf_repo": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
          "hf_file": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
        },
        {
          "quant": "Q5_K_M",
          "file_size_gb": 5.1,
          "bpw": 5.69,
          "hf_repo": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
          "hf_file": "mistral-7b-instruct-v0.3.Q5_K_M.gguf",
          "hf_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q5_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 32,
        "embedding_dim": 4096,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 14336
      },
      "is_moe": false
    },
    {
      "id": "phi-3.5-mini-instruct",
      "name": "Phi 3.5 Mini Instruct",
      "family": "phi",
      "use_case": ["chat", "instruction", "coding", "reasoning"],
      "description": "Microsoft's efficient small model. Surprisingly capable.",
      "params_billions": 3.8,
      "architecture": "phi3",
      "max_context": 131072,
      "quality_score": 7.0,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 2.4,
          "bpw": 4.85,
          "hf_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
          "hf_file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 4.0,
          "bpw": 8.5,
          "hf_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
          "hf_file": "Phi-3.5-mini-instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 32,
        "embedding_dim": 3072,
        "attention_heads": 32,
        "kv_heads": 32,
        "head_dim": 96,
        "ffn_dim": 8192
      },
      "is_moe": false
    },
    {
      "id": "gemma-2-2b-it",
      "name": "Gemma 2 2B Instruct",
      "family": "gemma",
      "use_case": ["chat", "instruction", "lightweight"],
      "description": "Google's efficient small model. Good for edge deployment.",
      "params_billions": 2.6,
      "architecture": "gemma2",
      "max_context": 8192,
      "quality_score": 5.8,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 1.6,
          "bpw": 4.85,
          "hf_repo": "bartowski/gemma-2-2b-it-GGUF",
          "hf_file": "gemma-2-2b-it-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 26,
        "embedding_dim": 2304,
        "attention_heads": 8,
        "kv_heads": 4,
        "head_dim": 256,
        "ffn_dim": 9216
      },
      "is_moe": false
    },
    {
      "id": "gemma-2-9b-it",
      "name": "Gemma 2 9B Instruct",
      "family": "gemma",
      "use_case": ["chat", "instruction", "coding", "reasoning"],
      "description": "Google's strong mid-size model. Excellent reasoning.",
      "params_billions": 9.2,
      "architecture": "gemma2",
      "max_context": 8192,
      "quality_score": 8.0,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 5.5,
          "bpw": 4.85,
          "hf_repo": "bartowski/gemma-2-9b-it-GGUF",
          "hf_file": "gemma-2-9b-it-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 42,
        "embedding_dim": 3584,
        "attention_heads": 16,
        "kv_heads": 8,
        "head_dim": 256,
        "ffn_dim": 14336
      },
      "is_moe": false
    },
    {
      "id": "deepseek-v2-lite",
      "name": "DeepSeek V2 Lite",
      "family": "deepseek",
      "use_case": ["chat", "instruction", "coding", "reasoning"],
      "description": "MoE model with strong coding ability. 16B total, 2.4B active.",
      "params_billions": 15.7,
      "architecture": "deepseek2",
      "max_context": 32768,
      "quality_score": 7.2,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q4_K_M",
          "file_size_gb": 9.0,
          "bpw": 4.85,
          "hf_repo": "bartowski/DeepSeek-V2-Lite-Chat-GGUF",
          "hf_file": "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/DeepSeek-V2-Lite-Chat-GGUF/resolve/main/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 27,
        "embedding_dim": 2048,
        "attention_heads": 16,
        "kv_heads": 16,
        "head_dim": 128,
        "ffn_dim": 1408
      },
      "is_moe": true,
      "expert_count": 64,
      "expert_used_count": 6,
      "expert_ffn_dim": 1408
    },
    {
      "id": "mixtral-8x7b-instruct",
      "name": "Mixtral 8x7B Instruct",
      "family": "mixtral",
      "use_case": ["chat", "instruction", "reasoning"],
      "description": "Popular MoE model. 47B total, 13B active per token.",
      "params_billions": 46.7,
      "architecture": "mixtral",
      "max_context": 32768,
      "quality_score": 8.0,
      "quality_source": "Open LLM Leaderboard v2",
      "gguf_variants": [
        {
          "quant": "Q3_K_M",
          "file_size_gb": 18.0,
          "bpw": 3.69,
          "hf_repo": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
          "hf_file": "Mixtral-8x7B-Instruct-v0.1-Q3_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-Q3_K_M.gguf"
        },
        {
          "quant": "Q4_K_M",
          "file_size_gb": 26.0,
          "bpw": 4.85,
          "hf_repo": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
          "hf_file": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf"
        }
      ],
      "dimensions": {
        "layers": 32,
        "embedding_dim": 4096,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 14336
      },
      "is_moe": true,
      "expert_count": 8,
      "expert_used_count": 2,
      "expert_ffn_dim": 14336
    }
  ]
}
)CATALOG";
