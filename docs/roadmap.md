# Development Roadmap

## Legend

- ✅ Done
- 🔧 In Progress
- ⏳ Planned
- 💡 Future Idea

---

## Phase 1: Format Abstraction ✅

- [x] `include/format.h` — `vx_loader` interface, `vx_format` enum
- [x] `src/format.c` — Format detection, loader registry, weight resolution, arch auto-detect
- [x] GGUF loader refactored into `gguf_loader_load()`
- [x] `vx_model_create()` dispatches through format abstraction
- [x] `include/tokenizer.h` — Tokenizer vtable (interface only)

## Phase 2: Safetensors Support ✅

- [x] `src/safetensors.c` — Safetensors binary parser
- [x] Embedded JSON parser (no external dependencies)
- [x] HF `config.json` reader → `vx_model_config`
- [x] F16/BF16 → F32 conversion on load
- [x] Weight shape reversal (HF → Veltrix convention)
- [x] Architecture detection from `model_type`

## Phase 3: Tokenizer ✅

- [x] BPE tokenizer (GPT-2 style) — vocab + merge table
- [x] SentencePiece tokenizer (Unigram style) — greedy longest-prefix match
- [x] GGUF KV metadata → tokenizer creation (`tokenizer.ggml.*` KVs)
- [x] `gguf_scan_str_array()` + `gguf_scan_f32_array()` for array KV scanning
- [x] CLI integration: `-p "text"` → tokenize → infer → decode
- [x] Llama 3.2 / Qwen2.5 tokenizer loading verified
- [ ] TikToken tokenizer (Qwen2, Phi-3, DeepSeek)
- [ ] HF `tokenizer.json` parser (needs full JSON dict parser)

## Phase 4: ONNX Loader ⏳

- [ ] ONNX protobuf parsing (minimal embedded parser)
- [ ] ONNX → Veltrix weight mapping
- [ ] ONNX model config extraction
- [ ] Support for standard ONNX transformer export format

## Phase 5: LoRA Adapters ⏳

- [ ] LoRA weight loading (rank decomposition matrices)
- [ ] `vx_lora_merge()` — fuse LoRA into base weights
- [ ] `vx_lora_apply()` — runtime LoRA computation (no merge)
- [ ] Multiple LoRA adapter support

## Phase 6: Metalearn Integration 💡

- [ ] Wire meta-predictors into forward pass loop
- [ ] Actual layer skipping based on confidence threshold
- [ ] Attention head sparsity prediction (skip inattentive heads)
- [ ] Early exit prediction (stop at layer k, skip remaining)

## Phase 7: Performance 💡

- [ ] AVX-512 quantized GEMV kernels
- [ ] ARM NEON kernels
- [ ] NUMA-aware thread scheduling
- [ ] Memory-mapped tensor loading (mmap)
- [ ] FP16 intermediate storage for KV cache
- [ ] Flash attention (CPU-optimized)

## Phase 8: Features 💡

- [ ] LLM API server (HTTP/JSON)
- [ ] Grammar-constrained generation
- [ ] Structured output (JSON mode)
- [ ] Prefix caching
- [ ] Continuous batching
- [ ] Quantization-aware training (QAT) import
- [ ] FP8 support (HuggingFace FP8 format)

## Phase 9: Ecosystem 💡

- [ ] Python bindings
- [ ] Web UI
- [ ] Model hub integration (HuggingFace)
- [ ] ONNX export from Veltrix
- [ ] Fine-tuning adapter export
