#include "recommend/catalog_loader.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>

// =============================================================================
// JSON Parser (Simple - no external dependency for MVP)
// =============================================================================
// For MVP, we use a simple hand-written JSON parser.
// This avoids adding nlohmann::json as a dependency just for the catalog.
// The JSON structure is simple enough for a recursive descent parser.
// =============================================================================

namespace {

// Simple JSON value types
enum class JsonType { NULL_VAL, BOOL, NUMBER, STRING, ARRAY, OBJECT };

struct JsonValue {
    JsonType type = JsonType::NULL_VAL;
    bool bool_val = false;
    double number_val = 0.0;
    std::string string_val;
    std::vector<JsonValue> array_val;
    std::vector<std::pair<std::string, JsonValue>> object_val;
    
    // Accessors
    const JsonValue* find(const std::string& key) const {
        if (type != JsonType::OBJECT) return nullptr;
        for (const auto& [k, v] : object_val) {
            if (k == key) return &v;
        }
        return nullptr;
    }
    
    std::string get_string(const std::string& key, const std::string& def = "") const {
        auto* val = find(key);
        if (val && val->type == JsonType::STRING) return val->string_val;
        return def;
    }
    
    double get_number(const std::string& key, double def = 0.0) const {
        auto* val = find(key);
        if (val && val->type == JsonType::NUMBER) return val->number_val;
        return def;
    }
    
    bool get_bool(const std::string& key, bool def = false) const {
        auto* val = find(key);
        if (val && val->type == JsonType::BOOL) return val->bool_val;
        return def;
    }
    
    uint32_t get_uint32(const std::string& key, uint32_t def = 0) const {
        auto* val = find(key);
        if (val && val->type == JsonType::NUMBER) return static_cast<uint32_t>(val->number_val);
        return def;
    }
    
    const JsonValue* get_array(const std::string& key) const {
        auto* val = find(key);
        if (val && val->type == JsonType::ARRAY) return val;
        return nullptr;
    }
    
    const JsonValue* get_object(const std::string& key) const {
        auto* val = find(key);
        if (val && val->type == JsonType::OBJECT) return val;
        return nullptr;
    }
};

// JSON Parser class
class JsonParser {
public:
    JsonParser(const std::string& input) : input_(input), pos_(0) {}
    
    JsonValue parse() {
        skip_whitespace();
        return parse_value();
    }
    
    bool has_error() const { return !error_.empty(); }
    const std::string& error() const { return error_; }
    
private:
    std::string input_;
    size_t pos_;
    std::string error_;
    
    char peek() const {
        if (pos_ >= input_.size()) return '\0';
        return input_[pos_];
    }
    
    char advance() {
        if (pos_ >= input_.size()) return '\0';
        return input_[pos_++];
    }
    
    void skip_whitespace() {
        while (pos_ < input_.size() && (input_[pos_] == ' ' || input_[pos_] == '\t' || 
               input_[pos_] == '\n' || input_[pos_] == '\r')) {
            pos_++;
        }
    }
    
    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        
        if (c == '"') return parse_string_value();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        
        error_ = "Unexpected character: " + std::string(1, c);
        return JsonValue();
    }
    
    JsonValue parse_string_value() {
        JsonValue val;
        val.type = JsonType::STRING;
        val.string_val = parse_string();
        return val;
    }
    
    std::string parse_string() {
        if (advance() != '"') {
            error_ = "Expected '\"'";
            return "";
        }
        
        std::string result;
        while (pos_ < input_.size()) {
            char c = advance();
            if (c == '"') return result;
            if (c == '\\') {
                char esc = advance();
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'u': {
                        // Parse 4 hex digits
                        std::string hex;
                        for (int i = 0; i < 4 && pos_ < input_.size(); i++) {
                            hex += advance();
                        }
                        uint32_t cp = std::stoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            result += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            result += static_cast<char>(0xC0 | (cp >> 6));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            result += static_cast<char>(0xE0 | (cp >> 12));
                            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: result += esc; break;
                }
            } else {
                result += c;
            }
        }
        
        error_ = "Unterminated string";
        return result;
    }
    
    JsonValue parse_number() {
        JsonValue val;
        val.type = JsonType::NUMBER;
        
        std::string num_str;
        if (peek() == '-') num_str += advance();
        while (pos_ < input_.size() && ((input_[pos_] >= '0' && input_[pos_] <= '9') || input_[pos_] == '.' || input_[pos_] == 'e' || input_[pos_] == 'E' || input_[pos_] == '+' || input_[pos_] == '-')) {
            if (num_str.size() > 0 && (input_[pos_] == '+' || input_[pos_] == '-') && num_str.back() != 'e' && num_str.back() != 'E') break;
            num_str += advance();
        }
        
        try {
            val.number_val = std::stod(num_str);
        } catch (...) {
            error_ = "Invalid number: " + num_str;
        }
        
        return val;
    }
    
    JsonValue parse_bool() {
        JsonValue val;
        val.type = JsonType::BOOL;
        
        if (input_.substr(pos_, 4) == "true") {
            val.bool_val = true;
            pos_ += 4;
        } else if (input_.substr(pos_, 5) == "false") {
            val.bool_val = false;
            pos_ += 5;
        } else {
            error_ = "Invalid boolean";
        }
        
        return val;
    }
    
    JsonValue parse_null() {
        JsonValue val;
        val.type = JsonType::NULL_VAL;
        
        if (input_.substr(pos_, 4) == "null") {
            pos_ += 4;
        } else {
            error_ = "Invalid null";
        }
        
        return val;
    }
    
    JsonValue parse_array() {
        JsonValue val;
        val.type = JsonType::ARRAY;
        
        if (advance() != '[') {
            error_ = "Expected '['";
            return val;
        }
        
        skip_whitespace();
        if (peek() == ']') {
            advance();
            return val;
        }
        
        while (pos_ < input_.size()) {
            val.array_val.push_back(parse_value());
            skip_whitespace();
            if (peek() == ']') {
                advance();
                return val;
            }
            if (peek() != ',') {
                error_ = "Expected ',' or ']' in array";
                return val;
            }
            advance();
        }
        
        error_ = "Unterminated array";
        return val;
    }
    
    JsonValue parse_object() {
        JsonValue val;
        val.type = JsonType::OBJECT;
        
        if (advance() != '{') {
            error_ = "Expected '{'";
            return val;
        }
        
        skip_whitespace();
        if (peek() == '}') {
            advance();
            return val;
        }
        
        while (pos_ < input_.size()) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            if (advance() != ':') {
                error_ = "Expected ':' after key";
                return val;
            }
            JsonValue value = parse_value();
            val.object_val.push_back({key, std::move(value)});
            
            skip_whitespace();
            if (peek() == '}') {
                advance();
                return val;
            }
            if (peek() != ',') {
                error_ = "Expected ',' or '}' in object";
                return val;
            }
            advance();
        }
        
        error_ = "Unterminated object";
        return val;
    }
};

// =============================================================================
// Parse catalog from JSON value
// =============================================================================

GgufVariant parse_variant(const JsonValue& val) {
    GgufVariant v;
    v.quant = val.get_string("quant");
    v.file_size_gb = val.get_number("file_size_gb");
    v.bpw = val.get_number("bpw");
    v.hf_repo = val.get_string("hf_repo");
    v.hf_file = val.get_string("hf_file");
    v.hf_url = val.get_string("hf_url");
    return v;
}

ModelDimensions parse_dimensions(const JsonValue& val) {
    ModelDimensions d;
    d.layers = val.get_uint32("layers");
    d.embedding_dim = val.get_uint32("embedding_dim");
    d.attention_heads = val.get_uint32("attention_heads");
    d.kv_heads = val.get_uint32("kv_heads");
    d.head_dim = val.get_uint32("head_dim");
    d.ffn_dim = val.get_uint32("ffn_dim");
    return d;
}

CatalogModel parse_model(const JsonValue& val) {
    CatalogModel m;
    m.id = val.get_string("id");
    m.name = val.get_string("name");
    m.family = val.get_string("family");
    m.description = val.get_string("description");
    m.params_billions = val.get_number("params_billions");
    m.architecture = val.get_string("architecture");
    m.max_context = val.get_uint32("max_context");
    m.quality_score = val.get_number("quality_score");
    m.quality_source = val.get_string("quality_source");
    m.is_moe = val.get_bool("is_moe");
    
    // MoE fields
    m.expert_count = val.get_uint32("expert_count");
    m.expert_used_count = val.get_uint32("expert_used_count");
    m.expert_ffn_dim = val.get_uint32("expert_ffn_dim");
    
    // Use cases
    auto* use_cases_arr = val.get_array("use_case");
    if (use_cases_arr) {
        for (const auto& uc : use_cases_arr->array_val) {
            if (uc.type == JsonType::STRING) {
                m.use_cases.push_back(uc.string_val);
            }
        }
    }
    
    // Variants
    auto* variants_arr = val.get_array("gguf_variants");
    if (variants_arr) {
        for (const auto& v : variants_arr->array_val) {
            m.variants.push_back(parse_variant(v));
        }
    }
    
    // Dimensions
    auto* dims = val.get_object("dimensions");
    if (dims) {
        m.dimensions = parse_dimensions(*dims);
    }
    
    return m;
}

ModelCatalog parse_catalog(const JsonValue& val) {
    ModelCatalog catalog;
    catalog.version = val.get_string("version");
    
    auto* models_arr = val.get_array("models");
    if (models_arr) {
        for (const auto& m : models_arr->array_val) {
            catalog.models.push_back(parse_model(m));
        }
    }
    
    return catalog;
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

ModelCatalog load_catalog_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        fprintf(stderr, "[CatalogLoader] Failed to open catalog file: %s\n", file_path.c_str());
        return ModelCatalog();
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    JsonParser parser(content);
    JsonValue root = parser.parse();
    
    if (parser.has_error()) {
        fprintf(stderr, "[CatalogLoader] JSON parse error: %s\n", parser.error().c_str());
        return ModelCatalog();
    }
    
    if (root.type != JsonType::OBJECT) {
        fprintf(stderr, "[CatalogLoader] Expected JSON object at root\n");
        return ModelCatalog();
    }
    
    return parse_catalog(root);
}

ModelCatalog load_builtin_catalog() {
    // The built-in catalog is defined in catalog_data.h
    // This is generated by CMake from the JSON file
    #include "recommend/catalog_data.h"
    
    JsonParser parser(BUILTIN_CATALOG_JSON);
    JsonValue root = parser.parse();
    
    if (parser.has_error()) {
        fprintf(stderr, "[CatalogLoader] Failed to parse built-in catalog: %s\n", 
                parser.error().c_str());
        return ModelCatalog();
    }
    
    return parse_catalog(root);
}

ModelCatalog load_catalog(const std::string& custom_path) {
    if (!custom_path.empty()) {
        return load_catalog_from_file(custom_path);
    }
    return load_builtin_catalog();
}

std::string get_builtin_catalog_path() {
    // Return empty string - the built-in catalog is embedded in the binary
    return "";
}

bool validate_catalog(const ModelCatalog& catalog) {
    if (catalog.models.empty()) {
        fprintf(stderr, "[CatalogLoader] Catalog is empty\n");
        return false;
    }
    
    int errors = 0;
    for (const auto& model : catalog.models) {
        if (model.id.empty()) {
            fprintf(stderr, "[CatalogLoader] Model missing 'id'\n");
            errors++;
        }
        if (model.name.empty()) {
            fprintf(stderr, "[CatalogLoader] Model '%s' missing 'name'\n", model.id.c_str());
            errors++;
        }
        if (model.variants.empty()) {
            fprintf(stderr, "[CatalogLoader] Model '%s' has no variants\n", model.id.c_str());
            errors++;
        }
        for (const auto& variant : model.variants) {
            if (variant.hf_url.empty()) {
                fprintf(stderr, "[CatalogLoader] Variant %s/%s missing 'hf_url'\n",
                        model.id.c_str(), variant.quant.c_str());
                errors++;
            }
        }
    }
    
    return errors == 0;
}

void print_catalog_stats(const ModelCatalog& catalog) {
    printf("Catalog version: %s\n", catalog.version.c_str());
    printf("Total models: %zu\n", catalog.models.size());
    
    int total_variants = 0;
    int moe_models = 0;
    int families = 0;
    std::vector<std::string> family_list;
    
    for (const auto& model : catalog.models) {
        total_variants += model.variants.size();
        if (model.is_moe) moe_models++;
        if (std::find(family_list.begin(), family_list.end(), model.family) == family_list.end()) {
            family_list.push_back(model.family);
            families++;
        }
    }
    
    printf("Total variants: %d\n", total_variants);
    printf("Model families: %d (%s)\n", families, 
           [&family_list]() {
               std::string result;
               for (size_t i = 0; i < family_list.size(); i++) {
                   if (i > 0) result += ", ";
                   result += family_list[i];
               }
               return result;
           }().c_str());
    printf("MoE models: %d\n", moe_models);
    
    // Size range
    double min_size = 1e9, max_size = 0;
    for (const auto& model : catalog.models) {
        for (const auto& v : model.variants) {
            if (v.file_size_gb < min_size) min_size = v.file_size_gb;
            if (v.file_size_gb > max_size) max_size = v.file_size_gb;
        }
    }
    printf("Size range: %.1f GB — %.1f GB\n", min_size, max_size);
}
