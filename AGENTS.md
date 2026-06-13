# Veltrix — Agent Guide

Compact reference for working on this C LLM inference engine targeting low-end devices (N100, phones).

## Build & Test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/veltrix_test          # 26 unit tests (tensor, quant, metalearn, scheduler)
./build/veltrix_e2e           # synthetic GGUF end-to-end
./build/veltrix_onnx          # ONNX load + forward
./build/veltrix_lora          # LoRA merge + forward
./build/veltrix_tokjson       # BPE tokenizer round-trip
./build/veltrix_tiktok        # TikToken tokenizer round-trip
```
**All 8 targets** build from same source set (minus main.c for tests). No `--bench` flag.

- Compiler: MSVC `/arch:AVX2 /openmp` or GCC `-mavx2 -mfma -march=native -fopenmp`
- C11, CMake ≥ 3.15, OpenMP required on all platforms
- Use `MinGW` on Windows (MSVC fragile with gomp linking)

## CLI
```bash
veltrix <model.gguf> [-p prompt] [-n tokens] [-t threads] [--benchmark] [--lora path] [--no-meta]
```
- `-p` is the *only* prompt input (no interactive mode yet)
- `--benchmark` runs timed loop with warmup, reports us/tok and tok/s via `QueryPerformanceCounter`

## Known Bugs

### Qwen2 tokenizer loads as BPE (should be TikToken)
`tokenizer.ggml.model: gpt2` + `tokenizer.ggml.pre: qwen2` in Qwen2 GGUF. Code sees `gpt2` → BPE. Qwen2 actually uses TikToken. Fix: detect `tokenizer.ggml.pre == "qwen2"` or `general.architecture == "qwen2"` and force TikToken.

### vx_type_size() returns 0 for K-quants
Only Q3_K and Q6_K have correct sizes. Q4_K, Q5_K, Q8_K return 0. `tensor.c:20-24`.

### K-quant dequant missing
Only Q2_K and Q6_K dequant functions exist. Q3_K, Q4_K, Q5_K, Q8_K declared but unimplemented in `quantize.c`.

### FE state reset wrong
`scheduler.c:30`: `memset(..., sizeof(float) * 1)` instead of `* n_layers`.

### GGUF type mapping incomplete
`gguf.c:11`: `gguf_to_vx_type()` missing Q4_1, Q5_0, Q5_1, Q8_1 (returns F32 fallback).

## Architecture

### Core dataflow
```
GGUF file → gguf_parse_header → gguf_parse_tensors → tensor data → vx_model_resolve_weights
→ memory consolidation (single alloc) → vx_model_forward
```

### Weight resolution
Format-agnostic name pattern matching in `format.c`. Supports both GGUF (`blk.%d.attn_q.weight`) and HF (`model.layers.%d.self_attn.q_proj.weight`) naming. Architecture auto-detected from `general.architecture` string.

### Tensor layout
- `ne[0]` = fastest-changing dimension (GGUF convention)
- ONNX/safetensors dims reversed on load
- Quantized blocks: Q4_0 = 18 B/32 elts, Q6_K = 210 B/256 elts

### Forward pass (per layer)
1. RMSNorm → QKV GEMV → RoPE → KV cache → Attention → Output GEMV → Residual
2. RMSNorm → FFN GEMV (gate/up/down, SwiGLU/GEGLU/Classic) → Residual
3. Final norm → Output GEMV → logits

### GEMV dispatch
```
VX_TYPE_F32 → vx_gemv_f32_avx2 (or scalar)
VX_TYPE_Q4_0 → vx_gemv_q4_0_avx2 (direct, no dequant)
default → dequant to reusable scratch buffer → f32 gemv
```
- `dequant_tmp` on `vx_model` avoids malloc per forward (was 256 MB for Q6_K output.weight)

### Performance bottleneck
TinyLlama 1.1B: **4.6 tok/s** vs target 40+. Bandwidth utilization ~10% (2.9 GB/s of 30 GB/s).
- Q6_K output.weight still hits dequant path (no direct Q6_K GEMV)
- Dequant is bandwidth-heavy: read 210 bytes → write 1024 bytes per Q6_K block
- Attention is O(n²) for KV cache reads

### Immediate perf targets
1. Direct Q6_K GEMV (eliminate dequant entirely for output.weight)
2. Multi-thread attention (OpenMP parallel over heads)
3. Q4_0 direct GEMV for non-AVX2 path
4. ARM NEON kernels for phone target

## Critical Conventions
- **Zero external dependencies**: JSON parser in `safetensors.c`, protobuf reader in `onnx.c` must stay self-contained
- **All F32 internals**: No mixed precision in forward path
- **New format**: implement `vx_loader` interface + register in `vx_format_init()`
- **New arch**: add to `arch_table` in `format.c` with MLP type, norm type, bias flags
- **Tokenizer auto-load**: GGUF metadata first, then HF `tokenizer.json` alongside safetensors. Non-fatal on failure.
- **LoRA merge**: destructive (modifies weights in-place, pre-merge not on-the-fly)
- **Test fixtures**: synthetic GGUF generated in `test_e2e.c`, ONNX in `gen_onnx.py`, LoRA in `gen_lora.py`
- **All tests independent** (no shared state). Use `PASS(name)`/`FAIL(name)` macros.
