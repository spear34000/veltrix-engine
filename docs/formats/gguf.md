# GGUF Format Loader

`src/gguf.c` — GGUF (GPT-Generated Unified Format) v3 parser and model loader.

## Binary Layout

```
┌──────────────────────────────────────┐
│ Magic: "GGUF" (4 bytes, LE u32)      │  ← 0x46554747
│ Version: 3 (4 bytes, LE u32)         │
│ Tensor Count (8 bytes, LE u64)       │
│ KV Pair Count (8 bytes, LE u64)      │
├──────────────────────────────────────┤
│ KV Pairs (variable length)           │
│  Each: key (GGUF_STR) + value (typed)│
├──────────────────────────────────────┤
│ Tensor Info (n_tensors × fixed)      │
│  Each: name (GGUF_STR) + n_dims +    │
│        dims (uint32[]) + type +       │
│        offset (uint64)                │
├──────────────────────────────────────┤
│ Padding to 32-byte alignment          │
├──────────────────────────────────────┤
│ Tensor Data (raw bytes)              │
│  Each tensor at its stored offset     │
│  (relative to tensor data section)    │
└──────────────────────────────────────┘
```

## GGUF Value Types

| Enum | Name | Size |
|---|---|---|
| 0 | U8 | 1 B |
| 1 | I8 | 1 B |
| 2 | U16 | 2 B |
| 3 | I16 | 2 B |
| 4 | U32 | 4 B |
| 5 | I32 | 4 B |
| 6 | F32 | 4 B |
| 7 | Bool | 1 B |
| 8 | Str | 8+len B |
| 9 | Array | 4+ N×value |
| 10 | U64 | 8 B |
| 11 | I64 | 8 B |
| 12 | F64 | 8 B |

## Key Functions

### `gguf_parse_header(buf, size, hdr)`
- Validates magic, reads version, tensor/KV counts, returns `gguf_header`.

### `gguf_parse_tensors(buf, size, hdr, tensors, max)`
- Skips KV pairs (reads and discards), then reads tensor info array.
- Returns count of tensors parsed, or -1 on error.
- Sets `hdr->tensor_data_offset` (offset from start of file to tensor data section).

### `gguf_tensor_data(buf, hdr, ti)`
- Returns pointer to raw tensor bytes: `buf + hdr->tensor_data_offset + ti->offset`.
- `ti->offset` is relative to tensor data section start.

### `gguf_load_tensor(buf, hdr, ti, t)`
- Allocates a `vx_tensor`, copies raw data from file buffer.

### KV Scan Functions

```c
uint32_t gguf_scan_u32(buf, size, offset, n_kv, suffix, fallback);
float    gguf_scan_f32(buf, size, offset, n_kv, suffix, fallback);
int      gguf_scan_str(buf, size, offset, n_kv, suffix, out, out_len);
```

These iterate KV pairs in the file buffer, find the one whose key ends with `suffix`, and return its value. Used to read model config from GGUF metadata.

## GGUF → Config Mapping

| KV Key (suffix) | Config Field |
|---|---|
| `block_count` | `n_layers` |
| `attention.head_count` | `n_heads` |
| `attention.head_count_kv` | `n_kv_heads` |
| `embedding_length` | `n_embd` |
| `feed_forward_length` | `n_ff` |
| `tokenizer.ggml.model` | tokenizer type |
| `general.architecture` | arch string (→ `vx_model_apply_arch`) |
| `rope.freq_base` | `rope_theta` |
| `rope.dimension_count` | `rope_partial_dims` |

## `gguf_loader_load()` Flow

1. `fopen`/`fread` entire file into buffer
2. `gguf_parse_header` → validate GGUF magic
3. `gguf_parse_tensors` → parse KV + tensor info
4. Scan KV metadata → fill `model->config`
5. `vx_model_apply_arch()` → set MLP/norm/bias defaults
6. Set thread count
7. Allocate `model->tensors[]`
8. For each tensor info: `gguf_load_tensor()` → allocate + copy data
9. Free file buffer
