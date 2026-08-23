#include "hotcold/weight_splitter.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// =============================================================================
// Weight matrix splitting
// =============================================================================

WeightSplit split_weight_matrix(
    const float* weights,       // Input weight matrix [rows, cols] row-major
    uint32_t rows,
    uint32_t cols,
    const std::vector<uint32_t>& hot_indices,
    const std::vector<uint32_t>& cold_indices)
{
    WeightSplit split;
    split.rows = rows;
    split.n_hot = static_cast<uint32_t>(hot_indices.size());
    split.n_cold = static_cast<uint32_t>(cold_indices.size());

    split.hot_weights.resize(static_cast<size_t>(rows) * split.n_hot);
    split.cold_weights.resize(static_cast<size_t>(rows) * split.n_cold);

    // Copy hot neuron columns (GPU-resident)
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t h = 0; h < split.n_hot; h++) {
            uint32_t col = hot_indices[h];
            split.hot_weights[static_cast<size_t>(r) * split.n_hot + h] =
                weights[static_cast<size_t>(r) * cols + col];
        }
    }

    // Copy cold neuron columns (CPU-resident)
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < split.n_cold; c++) {
            uint32_t col = cold_indices[c];
            split.cold_weights[static_cast<size_t>(r) * split.n_cold + c] =
                weights[static_cast<size_t>(r) * cols + col];
        }
    }

    return split;
}

// =============================================================================
// Layer-level weight splitting (up_proj, gate_proj, down_proj)
// =============================================================================

LayerWeightSplit split_layer_weights(
    const float* up_proj_weights,
    const float* gate_proj_weights,
    const float* down_proj_weights,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    const LayerHotSet& hot_set)
{
    LayerWeightSplit split;

    // up_proj: [hidden_dim, ffn_dim] — columns are neurons
    split.up_proj = split_weight_matrix(
        up_proj_weights, hidden_dim, ffn_dim,
        hot_set.hot_indices, hot_set.cold_indices);

    // gate_proj: [hidden_dim, ffn_dim] — columns are neurons
    split.gate_proj = split_weight_matrix(
        gate_proj_weights, hidden_dim, ffn_dim,
        hot_set.hot_indices, hot_set.cold_indices);

    // down_proj: [ffn_dim, hidden_dim] — rows are neurons (transposed view)
    // We split rows here, not columns
    // For down_proj, hot_indices refer to rows, not columns
    // We need to extract rows by index
    split.down_proj.rows = ffn_dim;
    split.down_proj.n_hot = static_cast<uint32_t>(hot_set.hot_indices.size());
    split.down_proj.n_cold = static_cast<uint32_t>(hot_set.cold_indices.size());

    split.down_proj.hot_weights.resize(static_cast<size_t>(split.down_proj.n_hot) * hidden_dim);
    split.down_proj.cold_weights.resize(static_cast<size_t>(split.down_proj.n_cold) * hidden_dim);

    // Copy hot neuron rows
    for (uint32_t h = 0; h < split.down_proj.n_hot; h++) {
        uint32_t row = hot_set.hot_indices[h];
        memcpy(&split.down_proj.hot_weights[static_cast<size_t>(h) * hidden_dim],
               &down_proj_weights[static_cast<size_t>(row) * hidden_dim],
               sizeof(float) * hidden_dim);
    }

    // Copy cold neuron rows
    for (uint32_t c = 0; c < split.down_proj.n_cold; c++) {
        uint32_t row = hot_set.cold_indices[c];
        memcpy(&split.down_proj.cold_weights[static_cast<size_t>(c) * hidden_dim],
               &down_proj_weights[static_cast<size_t>(row) * hidden_dim],
               sizeof(float) * hidden_dim);
    }

    return split;
}

// =============================================================================
// Save/Load split weights
// =============================================================================

static bool save_weight_split(FILE* f, const WeightSplit& ws) {
    fwrite(&ws.rows, sizeof(ws.rows), 1, f);
    fwrite(&ws.n_hot, sizeof(ws.n_hot), 1, f);
    fwrite(&ws.n_cold, sizeof(ws.n_cold), 1, f);

    if (ws.n_hot > 0) {
        fwrite(ws.hot_weights.data(), sizeof(float),
               static_cast<size_t>(ws.rows) * ws.n_hot, f);
    }
    if (ws.n_cold > 0) {
        fwrite(ws.cold_weights.data(), sizeof(float),
               static_cast<size_t>(ws.rows) * ws.n_cold, f);
    }
    return true;
}

static WeightSplit load_weight_split(FILE* f) {
    WeightSplit ws;
    fread(&ws.rows, sizeof(ws.rows), 1, f);
    fread(&ws.n_hot, sizeof(ws.n_hot), 1, f);
    fread(&ws.n_cold, sizeof(ws.n_cold), 1, f);

    ws.hot_weights.resize(static_cast<size_t>(ws.rows) * ws.n_hot);
    ws.cold_weights.resize(static_cast<size_t>(ws.rows) * ws.n_cold);

    if (ws.n_hot > 0) {
        fread(ws.hot_weights.data(), sizeof(float),
              static_cast<size_t>(ws.rows) * ws.n_hot, f);
    }
    if (ws.n_cold > 0) {
        fread(ws.cold_weights.data(), sizeof(float),
              static_cast<size_t>(ws.rows) * ws.n_cold, f);
    }
    return ws;
}

bool save_split_weights(const LayerWeightSplit& split, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not save split weights to %s\n", path.c_str());
        return false;
    }

    const char magic[] = "HCSW";
    uint32_t version = 1;
    fwrite(magic, 4, 1, f);
    fwrite(&version, sizeof(version), 1, f);

    save_weight_split(f, split.up_proj);
    save_weight_split(f, split.gate_proj);
    save_weight_split(f, split.down_proj);

    fclose(f);
    return true;
}

LayerWeightSplit load_split_weights(const std::string& path) {
    LayerWeightSplit split;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return split;
    }

    char magic[4];
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, "HCSW", 4) != 0) {
        fclose(f);
        return split;
    }

    uint32_t version;
    fread(&version, sizeof(version), 1, f);

    split.up_proj = load_weight_split(f);
    split.gate_proj = load_weight_split(f);
    split.down_proj = load_weight_split(f);

    fclose(f);
    return split;
}

// =============================================================================
// Memory estimation
// =============================================================================

SplitMemoryEstimate estimate_split_memory(
    const HotNeuronProfile& profile,
    uint64_t vram_budget_bytes,
    uint64_t ram_budget_bytes,
    double bytes_per_param)
{
    SplitMemoryEstimate est;

    if (profile.layers.empty() || profile.ffn_dim == 0 || profile.hidden_dim == 0) {
        return est;
    }

    // Calculate hot neurons per layer based on hot_ratio
    uint32_t hot_per_layer = static_cast<uint32_t>(profile.ffn_dim * profile.hot_ratio);
    if (hot_per_layer == 0) hot_per_layer = 1;

    // Each FFN layer has 3 weight matrices: up_proj, gate_proj, down_proj
    // up_proj: [hidden_dim, ffn_dim] — hot part: [hidden_dim, hot_per_layer]
    // gate_proj: [hidden_dim, ffn_dim] — hot part: [hidden_dim, hot_per_layer]
    // down_proj: [ffn_dim, hidden_dim] — hot part: [hot_per_layer, hidden_dim]
    uint64_t hot_params_per_layer =
        static_cast<uint64_t>(profile.hidden_dim) * hot_per_layer +  // up_proj hot
        static_cast<uint64_t>(profile.hidden_dim) * hot_per_layer +  // gate_proj hot
        static_cast<uint64_t>(hot_per_layer) * profile.hidden_dim;   // down_proj hot

    uint64_t cold_per_layer =
        static_cast<uint64_t>(profile.hidden_dim) * (profile.ffn_dim - hot_per_layer) +  // up_proj cold
        static_cast<uint64_t>(profile.hidden_dim) * (profile.ffn_dim - hot_per_layer) +  // gate_proj cold
        static_cast<uint64_t>(profile.ffn_dim - hot_per_layer) * profile.hidden_dim;     // down_proj cold

    est.hot_weights_bytes = hot_params_per_layer * profile.num_layers *
                            static_cast<uint64_t>(bytes_per_param);
    est.cold_weights_bytes = cold_per_layer * profile.num_layers *
                              static_cast<uint64_t>(bytes_per_param);

    // CUDA overhead: 512 MB base + KV cache will be added by caller
    uint64_t cuda_overhead = 512ULL * 1024 * 1024;
    est.vram_required_bytes = est.hot_weights_bytes + cuda_overhead;
    est.ram_required_bytes = est.cold_weights_bytes;

    est.fits_in_budget =
        (est.vram_required_bytes <= vram_budget_bytes) &&
        (est.ram_required_bytes <= ram_budget_bytes);

    return est;
}
