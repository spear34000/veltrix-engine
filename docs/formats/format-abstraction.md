# Format Abstraction Layer

`include/format.h` + `src/format.c` — the interface for adding new model formats.

## Key Types

```c
// Format identifier
typedef enum {
    VX_FORMAT_UNKNOWN = 0,
    VX_FORMAT_GGUF,
    VX_FORMAT_SAFETENSORS,
    VX_FORMAT_ONNX,
    VX_FORMAT_LORA,
} vx_format;

// Loader vtable — one per format
typedef struct vx_loader {
    const char *name;                           // "gguf", "safetensors", ...
    vx_format format;                           // format enum
    vx_error (*load)(const char *path, vx_model *model);  // main entry point
} vx_loader;
```

## Format Detection

`vx_detect_format()` checks:
1. **Magic bytes**: GGUF starts with `GGUF` (4 bytes)
2. **Safetensors header**: 8-byte LE uint64 length + `{` at offset 8
3. **File extension fallback**: `.gguf`, `.safetensors`, `.onnx`

## Loader Registry

```c
void vx_loader_register(const vx_loader *loader);          // add a loader
const vx_loader *vx_loader_for_format(vx_format fmt);      // lookup by format
void vx_format_init(void);                                 // register all built-in loaders
```

Built-in loaders are registered in `vx_format_init()`:
- `vx_loader_gguf` (declared extern in format.h, defined in gguf.c)
- `vx_loader_safetensors` (declared extern in format.h, defined in safetensors.c)
- `vx_loader_onnx` (stub, planned)

## Loader Contract

A loader's `load()` function must:
1. Open and parse the model file
2. Fill `model->config` with detected architecture parameters
3. Allocate `model->tensors[]` array and populate with `vx_tensor` structs
4. Return `VX_OK` on success, appropriate error code otherwise
5. NOT resolve weight pointers (done by `vx_model_resolve_weights()` post-load)
6. NOT auto-detect missing dims (done by `vx_model_auto_detect_dims()` post-load)

## Post-Load Processing

After `loader->load()` returns, `vx_model_create()` calls:

1. **`vx_model_resolve_weights(model)`**: Iterates `model->tensors[]`, matches names by HF/GGUF conventions, assigns dedicated pointer fields (`model->tok_embd`, `model->attn_q[i]`, etc.).
2. **`vx_model_auto_detect_dims(model)`**: Fills any zero-valued config fields by inspecting weight tensor shapes (e.g., `n_embd` from `tok_embd->ne[0]`, `n_ff` from `ffn_up->ne[1]`, etc.).

## Adding a New Format

See `developer/adding-new-format.md`.
