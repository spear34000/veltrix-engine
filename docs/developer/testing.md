# Testing Guide

## Test Suite

| File | Target | Tests |
|---|---|---|
| `tests/test_gguf.c` | Unit tests | 26 tests (tensor ops, quantization, metalearn, scheduler) |
| `tests/test_e2e.c` | End-to-end | Synthetic GGUF model → forward pass |
| `tests/benchmark.c` | Performance | Benchmarking various operations |

## Running Tests

```bash
# All unit tests
./build/veltrix_test

# End-to-end synthetic test
./build/veltrix_e2e

# Benchmarks
./build/veltrix_bench
```

## Unit Tests (test_gguf.c)

The 26 tests cover:

### Tensor Operations (7 tests)
- `add_f32`: Vector addition
- `vec_dot_f32`: Dot product
- `mat_mul_f32 (2,2,3)`: Matrix multiply
- `rms_norm`: RMS normalization (with NaN guard)
- `softmax`: Softmax (with divide-by-zero guard)
- `silu`: SiLU activation
- `norm_l2`: L2 norm

### Quantization (3 tests)
- `q4_0 quantize size`: Correct block count calculation
- `q4_0 roundtrip`: Quantize → dequantize preserves values approximately
- `q4_0 dot product`: Quantized dot product matches F32

### Meta-Predictor (5 tests)
- `meta_predictor create`: Allocation and initialization
- `meta_predictor learns`: Online update reduces error
- `meta_predictor confidence`: Confidence decreases with noise
- `meta_system create`: System allocates correctly
- `meta_decide_layer returns`: Decision function returns valid output

### Meta-Decision (3 tests)
- `meta_decide_layer confidence range`: Confidence ∈ [0,1]
- `meta_decide_exit returns`: Exit prediction works
- `meta_decide_exit confidence range`: Exit confidence ∈ [0,1]

### Scheduler (9 tests)
- `fe_state create`: State allocation
- `scheduler config default policy`: Default policy is MIN_FE
- `set_policy MIN_FE`: Policy switching
- `set_target_skip`: Skip rate configuration
- `decide skip at high confidence`: Skip when confidence high
- `decide exact at low confidence`: Exact when confidence low
- `scheduler_update increments`: Statistics tracking
- `compute_free_energy`: FE formula correctness
- `compute_prediction_error`: PE formula correctness

## E2E Test (test_e2e.c)

Generates a synthetic GGUF model with:
- 2 layers, n_embd=64, n_ff=128, n_vocab=100
- Q4_0 quantization
- Random weight data

Then:
1. Loads the model via `vx_model_create()`
2. Runs 10 forward passes
3. Verifies loading: layer count, dimension match
4. Verifies inference: non-zero logits, no crashes
5. Reports throughput (tok/s)

## Adding New Tests

### Adding to test_gguf.c

```c
// In the test registration section:
static void test_my_feature(void) {
    // Setup
    // Exercise
    // Assert (use PASS/FAIL macros)
}

// In main():
// test_my_feature();  // Add call
```

### Test Conventions

- Use `PASS("description")` and `FAIL("description")` macros
- Tests should be independent (no shared mutable state)
- Avoid hardcoded magic numbers in assertions (compute expected values)
- Test edge cases: zero, negative, NaN, overflow
- Synthetic GGUF generation functions are in the test file itself

## Continuous Integration

Before merging:
1. All 26 unit tests pass
2. E2E test passes
3. At least one real model loads and generates without NaN
4. No compilation warnings (MSVC, GCC, Clang)
5. Benchmarks show no regression vs. previous run

## Known Issues

- E2E test creates a GGUF file with Q4_0 quantization; if any quantized tensor dimensions don't align with block boundaries (32), the GGUF is invalid and load will fail.
- The `ggml_type` enum values in `gguf.h` must match the GGUF v3 specification exactly (Q2_K=10, Q3_K=11, …, Q6_K=14, Q8_K=15). Wrong values cause tensor type misidentification and crash on load.
