# System Architecture

## High-Level Flow

```
model.gguf / model.safetensors
         │
         ▼
   vx_model_create(path)
         │
    ┌─────┴──────┐
    │            │
    ▼            ▼
vx_detect_format()   format detection (header magic + extension)
    │            │
    ▼            ▼
vx_loader_for_format()   lookup loader in registry
    │            │
    ▼            ▼
loader->load()    # gguf_loader_load() or st_loader_load()
    │
    ▼
vx_model_resolve_weights()   # wire up weight pointers (format-agnostic)
    │
    ▼
vx_model_auto_detect_dims()   # fill gaps in config from weight shapes
    │
    ▼
Ready for inference
```

## File Organization

```
include/             # Public API (headers)
├── veltrix.h        # Core types: vx_model, vx_tensor, vx_model_config, error codes
├── format.h         # Format abstraction: vx_loader, vx_format, vx_lora
├── gguf.h           # GGUF binary format: parser, tensor reader
├── tensor.h         # Tensor ops: alloc, GEMV, norm, attention, RoPE
├── quantize.h       # Quantization types: block_q4_0..block_q6_K, fp16 helpers
├── model.h          # Model lifecycle: create, destroy, forward (thin wrapper)
├── metalearn.h      # Meta-learning: predictors, decisions
├── scheduler.h      # Free-energy scheduler: skip/atten/exact decisions
├── speculative.h    # Speculative decoding: draft-target verification
└── tokenizer.h      # Tokenizer vtable: BPE, SentencePiece, TikToken

src/                 # Implementation
├── main.c           # CLI entry point (text prompt → tokenize → infer → decode)
├── model.c          # vx_model_create/dispatch, forward pipeline, tokenizer auto-load
├── format.c         # Format detection, loader registry, weight resolution
├── gguf.c           # GGUF parser + loader
├── safetensors.c    # Safetensors parser + loader
├── tokenizer.c      # BPE + SentencePiece tokenizer, GGUF KV tokenizer loader
├── tensor.c         # Tensor ops, dequant dispatch
├── quantize.c       # Quantize/dequantize/fp16 conversion
├── metalearn.c      # Meta-predictor implementation
├── scheduler.c      # Scheduler implementation
├── speculative.c    # Speculative decoding
└── simd/
    ├── simd.h       # SIMD dispatch header
    └── avx2.c       # AVX2-optimized kernels

tests/               # Test suite
├── test_gguf.c      # GGUF parser tests (synthetic GGUF generation)
├── test_e2e.c       # End-to-end inference test
└── benchmark.c      # Performance benchmarks
```

## Threading Model

- OpenMP `#pragma omp parallel for` within GEMV operations (row-parallel).
- Thread count set from `omp_get_max_threads()` at load time, configurable via `vx_set_n_threads()`.
- Each row of a matrix-vector multiply is independent → natural parallelization.
- Current limitation: no pipelining between layers (each layer is sequential).
