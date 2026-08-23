#include "calibration_log.h"
#include "types.h"
#include "executor.h"
#include <cstdio>
#include <cassert>
#include <cmath>

// Test: Create a record, serialize to JSON, deserialize back, compare
static bool test_json_roundtrip() {
    printf("Test 1: JSON serialization round-trip... ");

    // Create a record with all fields populated
    CalibrationRecord original;
    original.hardware_fingerprint = "Ultra 7 265K|GeForce RTX 5060|32GB|XPG GAMMIX S70 BLADE";
    original.model_id = "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    original.backend = "llama.cpp";
    original.quant = "Q4_K_M";
    original.placement = "FULL_GPU";
    original.gpu_layers = 28;
    original.context = 4096;
    original.kv_quant_bits = 8;
    original.predicted_tokens_per_sec = 70.0;
    original.predicted_ttft_ms = 5721.0;
    original.predicted_vram_bytes = 2700000000ULL;
    original.predicted_ram_bytes = 536870912ULL;
    original.predicted_confidence = "MEDIUM";
    original.actual_tokens_per_sec = 152.0;
    original.actual_ttft_ms = 63.0;
    original.actual_peak_vram_bytes = 3800000000ULL;
    original.actual_peak_ram_bytes = 16000000000ULL;
    original.actual_throttled = false;
    original.actual_tokens_generated = 50;
    original.actual_duration_sec = 0.33;
    original.timestamp = "2026-08-23T12:00:00Z";
    original.tool_version = "0.1.0";

    // Serialize
    std::string json = record_to_json(original);
    if (json.empty()) {
        printf("FAIL (empty JSON)\n");
        return false;
    }

    // Deserialize
    CalibrationRecord restored;
    if (!json_to_record(json, restored)) {
        printf("FAIL (parse error)\n");
        return false;
    }

    // Compare
    bool ok = true;
    auto check_str = [&](const char* name, const std::string& a, const std::string& b) {
        if (a != b) { printf("\n  MISMATCH %s: '%s' vs '%s'", name, a.c_str(), b.c_str()); ok = false; }
    };
    auto check_uint = [&](const char* name, uint64_t a, uint64_t b) {
        if (a != b) { printf("\n  MISMATCH %s: %llu vs %llu", name, (unsigned long long)a, (unsigned long long)b); ok = false; }
    };
    auto check_int = [&](const char* name, int a, int b) {
        if (a != b) { printf("\n  MISMATCH %s: %d vs %d", name, a, b); ok = false; }
    };
    auto check_double = [&](const char* name, double a, double b) {
        if (fabs(a - b) > 0.001) { printf("\n  MISMATCH %s: %.3f vs %.3f", name, a, b); ok = false; }
    };
    auto check_bool = [&](const char* name, bool a, bool b) {
        if (a != b) { printf("\n  MISMATCH %s: %d vs %d", name, a, b); ok = false; }
    };

    check_str("hardware_fingerprint", original.hardware_fingerprint, restored.hardware_fingerprint);
    check_str("model_id", original.model_id, restored.model_id);
    check_str("backend", original.backend, restored.backend);
    check_str("quant", original.quant, restored.quant);
    check_str("placement", original.placement, restored.placement);
    check_uint("gpu_layers", original.gpu_layers, restored.gpu_layers);
    check_uint("context", original.context, restored.context);
    check_uint("kv_quant_bits", original.kv_quant_bits, restored.kv_quant_bits);
    check_double("predicted_tokens_per_sec", original.predicted_tokens_per_sec, restored.predicted_tokens_per_sec);
    check_double("predicted_ttft_ms", original.predicted_ttft_ms, restored.predicted_ttft_ms);
    check_uint("predicted_vram_bytes", original.predicted_vram_bytes, restored.predicted_vram_bytes);
    check_uint("predicted_ram_bytes", original.predicted_ram_bytes, restored.predicted_ram_bytes);
    check_str("predicted_confidence", original.predicted_confidence, restored.predicted_confidence);
    check_double("actual_tokens_per_sec", original.actual_tokens_per_sec, restored.actual_tokens_per_sec);
    check_double("actual_ttft_ms", original.actual_ttft_ms, restored.actual_ttft_ms);
    check_uint("actual_peak_vram_bytes", original.actual_peak_vram_bytes, restored.actual_peak_vram_bytes);
    check_uint("actual_peak_ram_bytes", original.actual_peak_ram_bytes, restored.actual_peak_ram_bytes);
    check_bool("actual_throttled", original.actual_throttled, restored.actual_throttled);
    check_int("actual_tokens_generated", original.actual_tokens_generated, restored.actual_tokens_generated);
    check_double("actual_duration_sec", original.actual_duration_sec, restored.actual_duration_sec);
    check_str("timestamp", original.timestamp, restored.timestamp);
    check_str("tool_version", original.tool_version, restored.tool_version);

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok;
}

// Test: File append and read
static bool test_file_io() {
    printf("Test 2: File append and read... ");

    const char* test_file = "test_calibration_log.jsonl";

    // Clean up any previous test file
    remove(test_file);

    // Create and write 3 records
    for (int i = 0; i < 3; i++) {
        CalibrationRecord r;
        r.hardware_fingerprint = "TestCPU|TestGPU|16GB|TestNVMe";
        r.model_id = "test/model-Q4_K_M.gguf";
        r.backend = "llama.cpp";
        r.quant = "Q4_K_M";
        r.placement = "FULL_GPU";
        r.gpu_layers = 28;
        r.context = 4096;
        r.kv_quant_bits = 16;
        r.predicted_tokens_per_sec = 100.0 + i;
        r.predicted_ttft_ms = 50.0;
        r.predicted_vram_bytes = 2000000000ULL;
        r.predicted_ram_bytes = 0;
        r.predicted_confidence = "HIGH";
        r.actual_tokens_per_sec = 90.0 + i;
        r.actual_ttft_ms = 55.0;
        r.actual_peak_vram_bytes = 2100000000ULL;
        r.actual_peak_ram_bytes = 0;
        r.actual_throttled = false;
        r.actual_tokens_generated = 100;
        r.actual_duration_sec = 1.1;
        r.timestamp = "2026-08-23T12:00:0" + std::to_string(i) + "Z";
        r.tool_version = "0.1.0";

        if (!append_record(r, test_file)) {
            printf("FAIL (write error on record %d)\n", i);
            remove(test_file);
            return false;
        }
    }

    // Read back
    auto records = read_all_records(test_file);
    if (records.size() != 3) {
        printf("FAIL (expected 3 records, got %zu)\n", records.size());
        remove(test_file);
        return false;
    }

    // Verify content
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        double expected_tps = 100.0 + i;
        if (fabs(records[i].predicted_tokens_per_sec - expected_tps) > 0.001) {
            printf("\n  MISMATCH record %d tps: %.1f vs %.1f", i, records[i].predicted_tokens_per_sec, expected_tps);
            ok = false;
        }
        if (records[i].gpu_layers != 28) {
            printf("\n  MISMATCH record %d gpu_layers: %u vs 28", i, records[i].gpu_layers);
            ok = false;
        }
    }

    // Count records for hardware
    int count = count_records_for_hardware("TestCPU|TestGPU|16GB|TestNVMe", test_file);
    if (count != 3) {
        printf("\n  MISMATCH count_for_hardware: %d vs 3", count);
        ok = false;
    }

    // Cleanup
    remove(test_file);

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok;
}

// Test: Edge cases — special characters, empty fields
static bool test_edge_cases() {
    printf("Test 3: Edge cases (special chars, empty fields)... ");

    CalibrationRecord r;
    r.hardware_fingerprint = "CPU|GPU with spaces|32GB|NVMe \"Quoted\"";
    r.model_id = "user/model-name.gguf";
    r.backend = "llama.cpp";
    r.quant = "Q4_K_M";
    r.placement = "GPU_CPU_SPLIT";
    r.gpu_layers = 14;
    r.context = 131072;
    r.kv_quant_bits = 8;
    r.predicted_tokens_per_sec = 45.5;
    r.predicted_ttft_ms = 1200.0;
    r.predicted_vram_bytes = 3500000000ULL;
    r.predicted_ram_bytes = 1500000000ULL;
    r.predicted_confidence = "HIGH";
    r.actual_tokens_per_sec = 42.0;
    r.actual_ttft_ms = 80.0;
    r.actual_peak_vram_bytes = 3600000000ULL;
    r.actual_peak_ram_bytes = 1600000000ULL;
    r.actual_throttled = true;
    r.actual_tokens_generated = 200;
    r.actual_duration_sec = 4.76;
    r.timestamp = "2026-08-23T12:00:00Z";
    r.tool_version = "0.1.0";

    // Round-trip
    std::string json = record_to_json(r);
    CalibrationRecord restored;
    if (!json_to_record(json, restored)) {
        printf("FAIL (parse error)\n");
        return false;
    }

    bool ok = true;
    if (restored.hardware_fingerprint != r.hardware_fingerprint) {
        printf("\n  MISMATCH fingerprint with special chars");
        ok = false;
    }
    if (restored.actual_throttled != true) {
        printf("\n  MISMATCH throttled: expected true, got false");
        ok = false;
    }
    if (fabs(restored.actual_duration_sec - 4.76) > 0.001) {
        printf("\n  MISMATCH duration_sec");
        ok = false;
    }

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok;
}

int main() {
    printf("=== Calibration Log Tests ===\n\n");

    bool all_pass = true;
    all_pass &= test_json_roundtrip();
    all_pass &= test_file_io();
    all_pass &= test_edge_cases();

    printf("\n%s\n", all_pass ? "All tests passed!" : "SOME TESTS FAILED!");
    return all_pass ? 0 : 1;
}
