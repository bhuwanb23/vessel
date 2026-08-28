# Contributing to Vessel

Thank you for your interest in contributing to Vessel! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- CMake 3.20+
- C++17 compiler (MSVC 17+, GCC 11+, Clang 14+)
- CUDA Toolkit 12+ (for NVIDIA builds)
- Git

### Building

```bash
git clone https://github.com/bhuwanb23/vessel.git
cd vessel
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Debug
```

### Running Tests

```bash
cd build/bin/Debug
./e2e_test
./recommend_test
./catalog_test
./step11_test
./predictor_test
```

All tests must pass before submitting a PR.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Hardware specs (GPU, RAM, OS)
   - Vessel version and output

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the use case, not just the feature
3. Include expected output format if applicable

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests if applicable
5. Run the full test suite
6. Commit with a clear message
7. Push and open a PR

### Code Style

- Follow existing code patterns
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes and structs
- Add comments for non-obvious logic
- Keep functions focused (< 50 lines when possible)

### Commit Messages

```
<type>: <short description>

<optional body>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements

## Project Structure

```
vessel/
├── src/
│   ├── planner/          # Main pipeline
│   ├── profiler/         # Hardware profiling
│   ├── predictor/        # Performance prediction
│   ├── fetcher/          # Model metadata fetching
│   ├── platform/         # Platform-specific code
│   ├── recommend/        # Auto-recommendation
│   ├── hotcold/          # Hot/cold neuron split
│   └── moe/              # MoE expert-offload
├── include/              # Headers
├── data/                 # Model catalog
└── docs/                 # Documentation
```

## Areas for Contribution

### High Priority

- **New model support**: Add models to `data/models_catalog.json`
- **Bug fixes**: Check issues labeled `bug`
- **Documentation**: Improve guides, add examples
- **Tests**: Increase coverage for edge cases

### Medium Priority

- **Performance**: Optimize predictor formulas, reduce prediction time
- **UX**: Better error messages, progress indicators
- **Platform**: Improve AMD/Apple support
- **Calibration**: Better aggregation algorithms

### Advanced

- **New strategies**: Implement novel placement strategies
- **API server**: OpenAI-compatible API endpoint
- **GUI**: Web-based or desktop interface
- **Distributed**: Multi-node inference support

## Adding a New Model to the Catalog

1. Edit `data/models_catalog.json`
2. Add the model entry with:
   - `id`, `name`, `family`, `use_cases`
   - `params_billions`, `architecture`, `max_context`
   - `quality_score` (from benchmarks)
   - `gguf_variants` with download URLs
   - `dimensions` (layers, embedding_dim, etc.)
3. Update the embedded catalog in `include/recommend/catalog_data.h`
4. Run `catalog_test` to verify

## Testing Guidelines

- Write tests for new features
- Test on multiple hardware configurations if possible
- Include edge cases (empty inputs, extreme values)
- Verify predictions are reasonable (not just correct type)

## Questions?

Open a discussion in the GitHub Discussions tab.
