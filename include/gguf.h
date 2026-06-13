#ifndef VELTRIX_GGUF_H
#define VELTRIX_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"

#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3

typedef enum {
    GGUF_TYPE_U8   = 0,
    GGUF_TYPE_I8   = 1,
    GGUF_TYPE_U16  = 2,
    GGUF_TYPE_I16  = 3,
    GGUF_TYPE_U32  = 4,
    GGUF_TYPE_I32  = 5,
    GGUF_TYPE_F32  = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STR  = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_U64  = 10,
    GGUF_TYPE_I64  = 11,
    GGUF_TYPE_F64  = 12,
} gguf_type;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
} ggml_type;

// GGUF -> Veltrix type mapping
vx_type gguf_to_vx_type(int ggml_t);

typedef struct {
    uint64_t n_tensors;
    uint64_t n_kv_pairs;
    uint64_t tensor_data_offset; // offset of first tensor data
    size_t file_size;
} gguf_header;

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint32_t dims[4];
    uint32_t ggml_t;
    uint64_t offset;
    size_t size; // bytes
} gguf_tensor_info;

// Read GGUF header from buffer
int gguf_parse_header(const void *buf, size_t size, gguf_header *hdr);

// Read all tensor infos (caller provides array of size hdr->n_tensors)
// Sets hdr->tensor_data_offset
int gguf_parse_tensors(const void *buf, size_t size,
                       gguf_header *hdr,
                       gguf_tensor_info *tensors, int max_tensors);

// Find tensor info by name
int gguf_find_tensor(const gguf_tensor_info *tensors, int n,
                     const char *name, gguf_tensor_info *out);

// Get pointer to tensor data in buffer
const void *gguf_tensor_data(const void *buf, const gguf_header *hdr,
                             const gguf_tensor_info *ti);

// Load a single tensor into a vx_tensor
vx_error gguf_load_tensor(const void *buf, size_t size, const gguf_header *hdr,
                          const gguf_tensor_info *ti,
                          vx_tensor *t);

// Read a string KV pair value
int gguf_read_string(const void *buf, size_t size, uint64_t *offset,
                     char *out, int max_len);

// Read a uint32 KV pair value (convenience)
int gguf_read_u32(const void *buf, size_t size, uint64_t *offset,
                  uint32_t *out);

// === Scan KV pairs by key suffix ===

// Scan GGUF KV metadata for a uint32 value by key suffix match
uint32_t gguf_scan_u32(const void *buf, size_t size, uint64_t *offset,
                       uint64_t n_kv, const char *suffix, uint32_t fallback);

// Scan GGUF KV metadata for a float32 value by key suffix match
float gguf_scan_f32(const void *buf, size_t size, uint64_t *offset,
                    uint64_t n_kv, const char *suffix, float fallback);

// Scan GGUF KV metadata for a string value by key suffix match
int gguf_scan_str(const void *buf, size_t size, uint64_t *offset,
                  uint64_t n_kv, const char *suffix, char *out, int out_len);

// === GGUF Model Loader (implements vx_loader interface) ===
// Reads GGUF file and populates pre-allocated vx_model with config + tensors
vx_error gguf_loader_load(const char *path, vx_model *model);

#endif
