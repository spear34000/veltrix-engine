# Quantization

## Supported Types

| Enum | GGML Type | Block Size | Block Struct | Size/Block |
|---|---|---|---|---|
| `VX_TYPE_Q4_0` | `GGML_TYPE_Q4_0` | 32 | `block_q4_0 { uint16_t d; uint8_t qs[16]; }` | 18 B |
| `VX_TYPE_Q4_1` | `GGML_TYPE_Q4_1` | 32 | `block_q4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; }` | 20 B |
| `VX_TYPE_Q5_0` | `GGML_TYPE_Q5_0` | 32 | `block_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; }` | 22 B |
| `VX_TYPE_Q5_1` | `GGML_TYPE_Q5_1` | 32 | `block_q5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; }` | 24 B |
| `VX_TYPE_Q8_0` | `GGML_TYPE_Q8_0` | 32 | `block_q8_0 { uint16_t d; int8_t qs[32]; }` | 34 B |
| `VX_TYPE_Q2_K` | `GGML_TYPE_Q2_K` | 256 | `block_q2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; }` | 84 B |
| `VX_TYPE_Q6_K` | `GGML_TYPE_Q6_K` | 256 | `block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; }` | 210 B |

## FP16 Representation

`d` (scale) and `m` (min) fields in block structs are stored as **IEEE 754 binary16**:

```c
uint16_t vx_fp32_to_fp16(float f);
float   vx_fp16_to_fp32(uint16_t h);
```

This matches the GGML binary format exactly.

## Dequantization

Each quantized type has a `vx_dequantize_row_*` function:
- Reads a block of N elements
- Computes `value = q * scale + min` (or variations)
- Writes N floats

Dispatch table in `vx_dequantize()` (`tensor.c`):
```c
void vx_dequantize(const void *src, float *dst, int n, vx_type type);
```

## Quantization (Q4_0 only)

- `vx_quantize_q4_0()`: converts F32 → Q4_0 block format
- Used only for synthetic test GGUF generation

## Block Format Requirements

IMPORTANT: Block structs must use `uint16_t` for fp16 fields (not `float` or `__fp16`). This ensures exact binary compatibility with GGUF files saved by llama.cpp, HF Transformers, etc.
