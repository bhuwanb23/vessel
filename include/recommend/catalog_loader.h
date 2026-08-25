#pragma once

#include "catalog_types.h"
#include <string>

// =============================================================================
// Model Catalog Loader (Step 12, Phase A/E)
// =============================================================================
// Loads and parses the model catalog from JSON
// =============================================================================

// Load catalog from a JSON file path
// Returns empty catalog on error (check catalog.models.empty())
ModelCatalog load_catalog_from_file(const std::string& file_path);

// Load the built-in default catalog (embedded in binary)
ModelCatalog load_builtin_catalog();

// Load catalog from custom path or fall back to built-in
ModelCatalog load_catalog(const std::string& custom_path = "");

// Get the path to the built-in catalog file (for --catalog flag)
std::string get_builtin_catalog_path();

// Validate a catalog (check required fields, URLs, etc.)
// Returns true if valid, prints errors to stderr if not
bool validate_catalog(const ModelCatalog& catalog);

// Get catalog statistics
void print_catalog_stats(const ModelCatalog& catalog);
