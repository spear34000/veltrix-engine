# Getting Started

## Prerequisites

- C11 compiler (MSVC, GCC, Clang)
- CMake ≥ 3.15
- OpenMP (for multi-threading)
- Python 3 (for model download scripts)
- Git

## Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Binaries are in build/
```

### Debug Build
```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --config Debug
```

## Run

```bash
# Basic inference (128 tokens)
./build/veltrix models/llama3.2-1b-q4_0.gguf -p "Hello" -n 128

# With custom thread count
./build/veltrix models/llama3.2-1b-q4_0.gguf -t 8 -p "The meaning of life is"

# Benchmark mode
./build/veltrix models/llama3.2-1b-q4_0.gguf --benchmark

# Disable meta-learning
./build/veltrix models/llama3.2-1b-q4_0.gguf --no-meta

# Set target skip rate (meta-learning)
./build/veltrix models/llama3.2-1b-q4_0.gguf -s 0.5
```

## Tests

```bash
# Unit tests (26 tests)
./build/veltrix_test

# End-to-end synthetic test
./build/veltrix_e2e

# Benchmarks
./build/veltrix_bench
```

## Download Models

```powershell
# Download Llama 3.2 1B Q4_0
.\download_model.ps1
```

Or place any GGUF / safetensors model in `models/` manually.

## Project Structure

```
veltrix/
├── include/         # All header files (public API)
├── src/             # Implementation files
│   └── simd/        # SIMD-optimized kernels
├── tests/           # Test suite
├── models/          # Downloaded model files (gitignored)
├── scripts/         # Utility scripts
├── docs/            # Documentation
├── CMakeLists.txt   # Build configuration
└── download_model.ps1  # Model download helper
```

## Debugging

### NaN Detection
Veltrix includes NaN guards in RMS norm and softmax operations. If you encounter NaN during inference:
1. Check `vx_get_synthetic_weight_scale()` (synthetic test only)
2. Verify tensor shapes match model expectations
3. Ensure RoPE theta is set correctly (default 10000.0)

### Common Build Errors

**"implicit declaration of function"** — missing `#include` for the header declaring that function.

**"undefined symbol" at link time** — `.c` file not added to `CMakeLists.txt` SOURCES.

**OpenMP not found** — install OpenMP development package or set `-DCMAKE_C_FLAGS=/openmp` (MSVC).

### .gitignore

```
build/
models/
*.gguf
*.bin
*.exe
*.obj
*.ilk
*.pdb
__pycache__/
```
