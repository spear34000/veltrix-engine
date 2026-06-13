#ifndef VX_SIMD_H
#define VX_SIMD_H

#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
float vx_dot_f32_avx2(const float *a, const float *b, int n);
void vx_mat_mul_f32_avx2(float *dst, const float *a, const float *b,
                          int m, int n, int k);
void vx_rms_norm_avx2(float *o, const float *x, const float *weight, int n);
void vx_softmax_avx2(float *x, int n);
void vx_silu_avx2(float *x, int n);
void vx_add_f32_avx2(float *c, const float *a, const float *b, int n);
void vx_gemv_f32_avx2(float *restrict y, const float *restrict A,
                      const float *restrict x, int rows, int cols);
void vx_gemv_f32_avx2_sparse_rows(float *restrict y, const float *restrict A,
                                   const float *restrict x, int rows, int cols,
                                   const float *restrict gate, float threshold);
void vx_gemv_f32_avx2_sparse_cols_add(float *restrict y, const float *restrict A,
                                       int rows, int cols,
                                       const float *restrict x, float threshold);
void vx_gemv_q4_0_avx2(const void *restrict A, const float *restrict x,
                       float *restrict y, int rows, int cols);
void vx_gemv_q4_0_avx2_sparse_rows(const void *restrict A, const float *restrict x,
                                    float *restrict y, int rows, int cols,
                                    const float *restrict gate, float threshold);
void vx_gemv_q4_0_avx2_sparse_cols_add(const void *restrict A, float *restrict y,
                                         int rows, int cols,
                                         const float *restrict x, float threshold);
#endif

#endif
