#ifndef VX_FORMAT_H
#define VX_FORMAT_H

#include "veltrix.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// === 1. Format Detection ===

typedef enum {
    VX_FORMAT_UNKNOWN = 0,
    VX_FORMAT_GGUF,          // .gguf - llama.cpp binary format
    VX_FORMAT_SAFETENSORS,   // .safetensors - HuggingFace format
    VX_FORMAT_ONNX,          // .onnx - ONNX protobuf format
    VX_FORMAT_LORA,          // LoRA adapter (safetensors with lora_* keys)
} vx_format;

// Detect format from file header bytes (first 16 bytes + filename extension)
vx_format vx_detect_format(const uint8_t *header, size_t header_size, const char *filename);

// Format name string
const char *vx_format_name(vx_format fmt);

// === 2. Format Loader Interface ===

// Each format implements this interface
typedef struct vx_loader {
    const char *name;          // "gguf", "safetensors", "onnx"
    vx_format format;          // format enum value

    // Full model load: reads file, parses config + allocates + populates tensors
    // Must fill: model->config, model->tensors[], model->n_tensors
    // Must allocate model->tensors[i] for each tensor with data
    // Must set tensor name, type, ne[4], nbytes
    vx_error (*load)(const char *path, vx_model *model);

} vx_loader;

// Register a loader
void vx_loader_register(const vx_loader *loader);

// Find registered loader by format
const vx_loader *vx_loader_for_format(vx_format fmt);

// Initialize loader registry (called from vx_model_create)
void vx_format_init(void);

// Apply architecture defaults from arch name string
void vx_model_apply_arch(vx_model_config *config, const char *arch_name);

// === 3. Weight Name Resolution (format-agnostic) ===

// Resolve weight tensor pointers in model from flat model->tensors[] array
// Uses naming format arrays (tok_embd_fmts, attn_q_fmts, etc.) to match by name
// This step is identical regardless of how tensors were loaded
vx_error vx_model_resolve_weights(vx_model *model);

// Auto-detect model dimensions from tensor shapes (fallback when config is incomplete)
void vx_model_auto_detect_dims(vx_model *model);

// === 5. Built-in Loaders ===

// GGUF loader (existing code, adapted)
extern const vx_loader vx_loader_gguf;

// Safetensors loader (to be implemented)
extern const vx_loader vx_loader_safetensors;

// ONNX loader (to be implemented)
extern const vx_loader vx_loader_onnx;

// LoRA adapter 
// Unlike full model loaders, LoRA merges into an existing model
typedef struct vx_lora {
    int rank;                  // LoRA rank
    float alpha;               // LoRA alpha scaling
    int n_adapters;            // number of LoRA weight pairs loaded
    // Per-layer LoRA weights: for each target module, store A and B matrices
    // Each entry: {layer_idx, weight_name, vx_tensor *lora_A, vx_tensor *lora_B}
    void *impl;
} vx_lora;

// Load LoRA weights from a safetensors file
vx_error vx_lora_load(const char *path, int base_n_layers, vx_lora *lora);

// Merge LoRA weights into a model (mutates model weights in place)
// W' = W + (alpha/rank) * A @ B for each matched layer
vx_error vx_lora_merge(vx_lora *lora, vx_model *model);

// Apply LoRA at runtime (no mutation, slower but reversible)
vx_error vx_lora_apply(vx_lora *lora, const float *input, float *output,
                       int layer, const char *weight_name, int n, int d);

// Free LoRA resources
void vx_lora_destroy(vx_lora *lora);

#endif
