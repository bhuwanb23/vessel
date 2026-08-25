// Test layer-streaming prediction with simulated hardware specs
#include "types.h"
#include "hotcold/hotcold_predictor.h"
#include <cstdio>

int main() {
    printf("=== Layer-Streaming Prediction Test ===\n\n");

    // Simulate hardware specs (RTX 3060 12GB + 32GB RAM)
    HardwareSpec hw;
    hw.gpu_name = "NVIDIA GeForce RTX 3060";
    hw.vram_total_bytes = 12ULL * 1024 * 1024 * 1024;
    hw.vram_free_bytes = 10ULL * 1024 * 1024 * 1024;
    hw.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 28ULL * 1024 * 1024 * 1024;
    hw.gpu_bandwidth_gbs = 360.0;
    hw.gpu_tflops_fp16 = 12.7;
    hw.ram_bandwidth_gbs = 45.0;
    hw.nvme_sequential_mbs = 3500.0;
    hw.nvme_random_4k_mbs = 80.0;

    // Test 1: 3B model (fits on GPU)
    printf("Test 1: 3B model (fits on GPU)\n");
    {
        ModelSpec model;
        model.name = "Llama-3.2-3B";
        model.param_count = 3000000000;
        model.layers = 28;
        model.embedding_dim = 3072;
        model.attention_heads = 24;
        model.kv_heads = 8;
        model.head_dim = 128;
        model.ffn_dim = 8192;
        model.bits_per_weight = 4.85;
        model.context_length = 131072;

        auto pred = predict_layer_streaming(hw, model, 4096, 16);

        printf("  Viable: %s\n", pred.viable ? "yes" : "no");
        printf("  tok/s: %.4f\n", pred.tok_s);
        printf("  Worthwhile: %s\n", pred.is_worthwhile ? "yes" : "no");
        printf("  Reason: %s\n", pred.worthit_reason.c_str());
    }
    printf("\n");

    // Test 2: 7B model (might fit on GPU with Q4)
    printf("Test 2: 7B model (might fit on GPU with Q4)\n");
    {
        ModelSpec model;
        model.name = "Qwen2.5-7B";
        model.param_count = 7000000000;
        model.layers = 28;
        model.embedding_dim = 3584;
        model.attention_heads = 28;
        model.kv_heads = 4;
        model.head_dim = 128;
        model.ffn_dim = 18944;
        model.bits_per_weight = 4.85;
        model.context_length = 131072;

        auto pred = predict_layer_streaming(hw, model, 4096, 16);

        printf("  Viable: %s\n", pred.viable ? "yes" : "no");
        printf("  tok/s: %.4f\n", pred.tok_s);
        printf("  Seconds/token: %.1f\n", pred.seconds_per_token);
        printf("  Worthwhile: %s\n", pred.is_worthwhile ? "yes" : "no");
        printf("  Reason: %s\n", pred.worthit_reason.c_str());
        printf("  Disk read/token: %.1f GB\n", pred.disk_read_bytes_per_token / 1e9);
    }
    printf("\n");

    // Test 3: 13B model (likely needs layer-streaming)
    printf("Test 3: 13B model (needs layer-streaming)\n");
    {
        ModelSpec model;
        model.name = "Llama-3.1-13B";
        model.param_count = 13000000000;
        model.layers = 40;
        model.embedding_dim = 5120;
        model.attention_heads = 40;
        model.kv_heads = 8;
        model.head_dim = 128;
        model.ffn_dim = 13824;
        model.bits_per_weight = 4.85;
        model.context_length = 131072;

        auto pred = predict_layer_streaming(hw, model, 4096, 16);

        printf("  Viable: %s\n", pred.viable ? "yes" : "no");
        printf("  tok/s: %.4f\n", pred.tok_s);
        printf("  Seconds/token: %.1f\n", pred.seconds_per_token);
        printf("  Worthwhile: %s\n", pred.is_worthwhile ? "yes" : "no");
        printf("  Reason: %s\n", pred.worthit_reason.c_str());
        printf("  Time per layer: %.1f ms\n", pred.time_per_layer_ms);
        printf("  Total time/token: %.1f ms\n", pred.total_time_per_token_ms);
        printf("  Disk read/token: %.1f GB\n", pred.disk_read_bytes_per_token / 1e9);
    }
    printf("\n");

    // Test 4: 70B model (extreme case)
    printf("Test 4: 70B model (extreme case)\n");
    {
        ModelSpec model;
        model.name = "Llama-3.1-70B";
        model.param_count = 70000000000;
        model.layers = 80;
        model.embedding_dim = 8192;
        model.attention_heads = 64;
        model.kv_heads = 8;
        model.head_dim = 128;
        model.ffn_dim = 28672;
        model.bits_per_weight = 4.85;
        model.context_length = 131072;

        auto pred = predict_layer_streaming(hw, model, 4096, 16);

        printf("  Viable: %s\n", pred.viable ? "yes" : "no");
        printf("  tok/s: %.6f\n", pred.tok_s);
        printf("  Seconds/token: %.1f\n", pred.seconds_per_token);
        printf("  Minutes/token: %.1f\n", pred.seconds_per_token / 60.0);
        printf("  Worthwhile: %s\n", pred.is_worthwhile ? "yes" : "no");
        printf("  Reason: %s\n", pred.worthit_reason.c_str());
        printf("  Time per layer: %.1f ms\n", pred.time_per_layer_ms);
        printf("  Total time/token: %.1f ms\n", pred.total_time_per_token_ms);
        printf("  Disk read/token: %.1f GB\n", pred.disk_read_bytes_per_token / 1e9);
    }
    printf("\n");

    printf("=== All tests completed ===\n");
    return 0;
}
