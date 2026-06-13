# Veltrix — Prediction-Error Driven Inference Engine

Veltrix is a lightweight, multi-format local LLM inference engine written in C. It loads and runs large language models on CPU with support for quantization, meta-learning, speculative decoding, and adaptive layer skipping.

## Quick Start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/veltrix models/llama3.2-1b-q4_0.gguf -p "Hello" -n 128
```

## Supported Formats

| Format | Status | Extension |
|--------|--------|-----------|
| GGUF | Done | `.gguf` |
| Safetensors | Done | `.safetensors` |
| ONNX | Planned | `.onnx` |
| LoRA | Planned | `.lora` |

## Supported Architectures

Qwen2, Llama, Mistral, Gemma, Phi, Phi-3, StarCoder2 — auto-detected from model metadata.

## Docs Structure

```
docs/
├── README.md                    # This file
├── architecture/
│   ├── overview.md              # High-level system design
│   ├── model.md                 # Model struct, weight resolution, config
│   ├── forward.md               # Forward pass pipeline
│   └── quantization.md          # Quantized types (Q4_0..Q6_K)
├── formats/
│   ├── format-abstraction.md    # vx_loader interface, format detection
│   ├── gguf.md                  # GGUF parser + loader
│   └── safetensors.md           # Safetensors parser + loader
├── optimization/
│   ├── metalearn.md             # Meta-learning predictor system
│   ├── scheduler.md             # Free-energy based layer skipping
│   ├── speculative.md           # Speculative decoding
│   └── simd.md                  # SIMD (AVX2) optimizations
├── developer/
│   ├── getting-started.md       # Build, run, debug
│   ├── adding-new-format.md     # Adding a new model format
│   ├── adding-new-architecture.md # Adding a new model architecture
│   └── testing.md               # Test suite guide
└── roadmap.md                   # Development roadmap
```

## Key Design Decisions

1. **Format-abstraction layer**: Single `vx_loader` interface (`format.h`) means any model format can be added with a `.c` file and a 3-line registration.
2. **All-F32 internals**: Quantized tensors are dequantized on-the-fly during GEMV; all intermediate values are F32. No mixed-precision complexity in the forward pass.
3. **Weight resolution is format-agnostic**: `vx_model_resolve_weights()` in `format.c` matches HF and GGUF naming conventions (both `model.layers.%d.` and `blk.%d.`).
4. **Per-architecture defaults**: `arch_table` in `format.c` encodes MLP type (SwiGLU/GEGLU/Classic), norm type (RMS/LayerNorm), bias flags — no per-model hardcoding.

## Tests

```bash
./build/veltrix_test    # 26 unit tests (tensor ops, quant, metalearn, scheduler)
./build/veltrix_e2e     # Synthetic GGUF end-to-end test
./build/veltrix_bench   # Performance benchmarks
```

All tests must pass before merging.
