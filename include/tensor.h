#ifndef VX_TENSOR_H
#define VX_TENSOR_H

#include "veltrix.h"

vx_error vx_tensor_alloc(vx_tensor *t, int n_dims, const int *shape, vx_type type);
void vx_tensor_free(vx_tensor *t);
vx_error vx_tensor_copy(vx_tensor *dst, const vx_tensor *src);

size_t vx_type_size(vx_type type);
int vx_type_is_quantized(vx_type type);

void vx_mat_mul_f32(float *dst, const float *a, const float *b,
                    int m, int n, int k);
void vx_mat_mul_f32_f16(float *dst, const float *a, const void *b_f16,
                        int m, int n, int k);
void vx_vec_dot_f32(float *dst, const float *a, const float *b, int n);

void vx_gemv_f32(float *restrict y, const float *restrict A, const float *restrict x,
                 int rows, int cols);
void vx_gemv_f32_sparse_rows(float *restrict y, const float *restrict A,
                              const float *restrict x, int rows, int cols,
                              const float *restrict gate, float threshold);
void vx_gemv_f32_sparse_cols_add(float *restrict y, const float *restrict A,
                                  int rows, int cols,
                                  const float *restrict x, float threshold);
void vx_gemv_q4_0(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);

void vx_rms_norm(float *o, const float *x, const float *weight, int n);
void vx_layer_norm(float *o, const float *x, const float *weight, const float *bias, int n);
void vx_softmax(float *x, int n);
void vx_rope(float *q, float *k, int pos, int n_heads, int n_kv_heads, int head_dim, int rope_dims, float theta);
void vx_silu(float *x, int n);
void vx_add_f32(float *c, const float *a, const float *b, int n);
void vx_mul_f32(float *c, const float *a, const float *b, int n);
void vx_copy_f32(float *dst, const float *src, int n);
float vx_norm_l2(const float *x, int n);
float vx_dot_f32(const float *a, const float *b, int n);

void vx_quantize_q4_0(float *src, int n, void *dst);
void vx_dequantize_q4_0(const void *src, float *dst, int n);
void vx_dequantize(const void *src, float *dst, int n, vx_type type);

void vx_attention(float *out, const float *q, const float *k, const float *v,
                  int n_heads, int head_dim, int seq_len, int kv_len);
void vx_attention_masked(float *out, const float *q, const float *k, const float *v,
                         int n_heads, int head_dim, int seq_len, int kv_len);

void vx_gelu(float *x, int n);

#endif
