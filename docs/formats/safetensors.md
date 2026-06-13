# Safetensors Format Loader

`src/safetensors.c` — Safetensors (HuggingFace) format parser and model loader.

## Binary Layout

```
┌──────────────────────────────────────────┐
│ Header Length (8 bytes, LE uint64)        │
├──────────────────────────────────────────┤
│ JSON Header (variable length)             │
│  {                                        │
│    "tensor_name": {                       │
│      "dtype": "F16",                      │
│      "shape": [out_features, in_features],│
│      "data_offsets": [start, end]         │
│    },                                     │
│    ...                                    │
│    "__metadata__": { ... }                │
│  }                                        │
├──────────────────────────────────────────┤
│ Tensor Data (raw bytes, concatenated)     │
│  Each tensor at its byte offset           │
│  No padding between tensors               │
└──────────────────────────────────────────┘
```

## Embedded JSON Parser

The safetensors loader includes a minimal, dependency-free JSON parser in `safetensors.c`:

```c
// Skip whitespace
static const char *js_skip_ws(const char *p);

// Skip a JSON string (escaped quotes handled)
static const char *js_skip_string(const char *p);

// Skip any JSON value (object, array, string, number)
static const char *js_skip_value(const char *p);

// Find value by key at top level of JSON object
static const char *js_find_key(const char *json, const char *key);

// Read string value ("...") → pointer + length
static const char *js_read_string(const char *p, int *out_len);

// Read number → double
static double js_read_number(const char *p, int *out_len);

// Read [n1, n2, ...] → int array
static int js_read_int_array(const char *p, int *vals, int max_vals);

// Read [n1, n2, ...] → uint64 array
static int js_read_uint64_array(const char *p, uint64_t *vals, int max_vals);
```

## Header Parsing

`st_parse_header()` extracts tensor metadata from the JSON:

```c
typedef struct {
    char name[256];        // Tensor name (e.g. "model.layers.0.self_attn.q_proj.weight")
    char dtype[16];        // "F16", "BF16", "F32", "I8", etc.
    uint32_t shape[4];     // Dimensions (HF order: [out, in])
    int n_dims;
    uint64_t data_start;   // Byte offset in data section
    uint64_t data_end;     // End offset (exclusive)
} st_tensor_entry;
```

## Config Parsing

`st_load_config()` reads `config.json` alongside the safetensors file:

| JSON Key | Config Field |
|---|---|
| `model_type` | → `vx_model_apply_arch()` |
| `hidden_size` | `n_embd` |
| `num_hidden_layers` / `n_layer` | `n_layers` |
| `num_attention_heads` / `n_head` | `n_heads` |
| `num_key_value_heads` / `n_kv_head` | `n_kv_heads` |
| `intermediate_size` / `n_inner` | `n_ff` |
| `vocab_size` | `n_vocab` |
| `rope_theta` / `rope.freq_base` | `rope_theta` |
| `max_position_embeddings` | `n_ctx` |
| `head_dim` | `n_head_dim` |

## DType Conversion

On load, all tensors are converted to F32:

| Safetensors dtype | Conversion |
|---|---|
| `F32` / `float32` | Direct copy |
| `F16` / `float16` | `vx_fp16_to_fp32()` per element |
| `BF16` / `bfloat16` | Left-shift 16 bits → F32 |

## Weight Shape Convention

HF stores weights in PyTorch convention: `shape = [out_features, in_features]`.
Veltrix uses GGUF convention: `ne[0]` = fastest-changing = input features.

```
HF shape  [d0, d1, d2, d3]
               ↓ reverse
Veltrix ne[0]=d3, ne[1]=d2, ne[2]=d1, ne[3]=d0
```

This ensures the dot product dimension (`ne[0]`) = `n_embd` for projection matrices.

## Architecture Detection

Maps `model_type` from `config.json` to Veltrix architecture:

| model_type | arch |
|---|---|
| `qwen2` / `qwen2.5` | `VX_ARCH_QWEN2` |
| `llama` | `VX_ARCH_LLAMA` |
| `mistral` | `VX_ARCH_MISTRAL` |
| `gemma` / `gemma2` | `VX_ARCH_GEMMA` |
| `phi` | `VX_ARCH_PHI` |
| `phi3` | `VX_ARCH_PHI3` |
| `starcoder2` | `VX_ARCH_STARCODER` |

## `st_loader_load()` Flow

1. `st_load_config()` — read `config.json`, fill `model->config`
2. If arch not set, apply default (llama)
3. `fopen` safetensors file
4. Read 8-byte header length
5. Read JSON header, parse with `st_parse_header()`
6. For each tensor:
   - Reverse dimensions (HF → Veltrix)
   - Read raw bytes from file at `data_start`
   - Convert F16/BF16 → F32
   - Allocate and store `vx_tensor`
7. Close file
