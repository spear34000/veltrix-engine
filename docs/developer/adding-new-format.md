# Adding a New Model Format

This guide covers adding support for a new model file format (e.g., ONNX, GGJT, SafeTensors-v2).

## Step 1: Add Format Enum

In `include/format.h`:
```c
typedef enum {
    VX_FORMAT_UNKNOWN = 0,
    VX_FORMAT_GGUF,
    VX_FORMAT_SAFETENSORS,
    VX_FORMAT_ONNX,
    // VX_FORMAT_MYFORMAT,  ← Add here
} vx_format;
```

Add name in `vx_format_name()` in `src/format.c`.

## Step 2: Add Format Detection

In `vx_detect_format()` in `src/format.c`:
```c
// Check magic bytes
if (header_size >= 4 && memcmp(header, "MYFM", 4) == 0)
    return VX_FORMAT_MYFORMAT;

// Extension fallback
if (strcasecmp(ext, ".myfmt") == 0) return VX_FORMAT_MYFORMAT;
```

## Step 3: Create Loader File

Create `src/myformat.c`:
```c
#include "format.h"
#include "tensor.h"
#include "quantize.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

vx_error myfmt_loader_load(const char *path, vx_model *model) {
    // 1. Open file, parse format-specific binary/JSON
    // 2. Fill model->config:
    //    model->config.n_layers = ...;
    //    model->config.n_embd = ...;
    //    model->config.n_heads = ...;
    //    (call vx_model_apply_arch() if architecture string available)
    // 3. Allocate model->tensors:
    //    model->n_tensors = count;
    //    model->tensors = calloc(count, sizeof(vx_tensor *));
    //    For each tensor:
    //      vx_tensor *t = calloc(1, sizeof(vx_tensor));
    //      strncpy(t->name, name, VX_MAX_NAME-1);
    //      t->ne[0] = ...; // fastest-changing dim
    //      t->ne[1] = ...;
    //      t->type = VX_TYPE_F32; // always F32 in core
    //      t->nbytes = ne0 * ne1 * 4;
    //      t->data = malloc(t->nbytes);
    //      t->owned = true;
    //      // read data from file
    //      model->tensors[i] = t;
    // 4. Return VX_OK
}
```

## Step 4: Declare Loader

At the bottom of `src/myformat.c`:
```c
const vx_loader vx_loader_myfmt = {
    .name = "myformat",
    .format = VX_FORMAT_MYFORMAT,
    .load = myfmt_loader_load,
};
```

In `include/format.h`, add extern declaration:
```c
extern const vx_loader vx_loader_myfmt;
```

## Step 5: Register Loader

In `vx_format_init()` in `src/format.c`:
```c
void vx_format_init(void) {
    vx_loader_register(&vx_loader_gguf);
    vx_loader_register(&vx_loader_safetensors);
    vx_loader_register(&vx_loader_myfmt);  // ← Add here
}
```

## Step 6: Add to CMakeLists

```cmake
set(SOURCES
    ...
    src/myformat.c     # ← Add here
)
```

## Step 7: Weight Naming (if different conventions)

If your format uses different weight naming than GGUF/HF, add patterns to `vx_try_match_name()` in `src/format.c`:

```c
// MyFormat naming
if (strstr(name, "myfmt.layer.")) {
    // extract layer index, match to model.layers.X. convention
}
```

## Loader Contract

Your `load()` function must NOT:
- Resolve weight pointers (`vx_model_resolve_weights()` handles this)
- Auto-detect dimensions (`vx_model_auto_detect_dims()` handles this)
- Set thread count or apply arch defaults (done by `vx_model_create()` after load)

Your `load()` function MUST:
- Populate `model->config` with detected parameters
- Populate `model->tensors[]` with all weight tensors
- Set tensor names following GGUF convention (or HF convention) for weight resolution to work

## Testing

```bash
# Add test file to tests/
# Build and run
cmake --build build --config Release
./build/veltrix_test
./build/veltrix_e2e

# Test with real model
./build/veltrix models/my-model.myfmt -p "Test" -n 32
```
