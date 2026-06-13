# SIMD Optimizations

`src/simd/` — Architecture-specific SIMD kernels for performance-critical operations.

## Dispatch

```c
// simd.h — compile-time dispatch
#ifdef __AVX2__
    // Use AVX2 path
#else
    // Use scalar fallback
#endif
```

Currently only AVX2 is implemented.

## AVX2 Kernels (avx2.c)

### GEMV Q4_0
`vx_gemv_q4_0_avx2()` — performs quantized matrix-vector multiply for Q4_0 blocks:

1. Loads 32 4-bit quantized values + fp16 scale from a Q4_0 block
2. Dequantizes to 32 floats in YMM registers
3. Loads corresponding 32 input floats
4. Fused multiply-add into accumulator
5. Horizontal sum across lanes → per-row result

### GEMV Q4_0 Sparse Rows
`vx_gemv_q4_0_sparse_rows_avx2()` — same as above but skips rows where gate value is below threshold:
1. Compare gate values against threshold
2. Create mask of active rows
3. Only process active rows

### GEMV Q4_0 Sparse Cols
`vx_gemv_q4_0_sparse_cols_add_avx2()` — for sparse input vectors: skip columns where input is near-zero.

## Fallback Path

When AVX2 is not available, all operations fall back to scalar C implementations in `tensor.c` and `quantize.c`.

## Adding New SIMD Targets

Create a new file `src/simd/<arch>.c` with optimized implementations and add to `simd.h` dispatch:

```c
#ifdef __ARM_NEON
    #include "neon.c"
#elif defined(__AVX512F__)
    #include "avx512.c"
#endif
```

### Candidates for Optimization

- `vx_gemv_f32()` — simple dot-product, bandwidth-bound → prefetch
- `vx_rms_norm()` — reduction + broadcast
- `vx_softmax()` — max + exp + sum + normalize (3 passes)
- `vx_rope()` — sin/cos generation + complex multiply
- `vx_attention_masked()` — batched score computation
