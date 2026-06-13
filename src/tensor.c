#include "tensor.h"
#include "quantize.h"
#include "simd/simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

size_t vx_type_size(vx_type type) {
    switch (type) {
        case VX_TYPE_F32:  return 4;
        case VX_TYPE_F16:  return 2;
        case VX_TYPE_BF16: return 2;
        case VX_TYPE_Q4_0: return sizeof(block_q4_0);
        case VX_TYPE_Q4_1: return sizeof(block_q4_1);
        case VX_TYPE_Q5_0: return sizeof(block_q5_0);
        case VX_TYPE_Q5_1: return sizeof(block_q5_1);
        case VX_TYPE_Q8_0: return sizeof(block_q8_0);
        case VX_TYPE_Q8_1: return sizeof(block_q8_1);
        case VX_TYPE_Q2_K: return sizeof(block_q2_K);
        case VX_TYPE_Q3_K: return sizeof(block_q3_K);
        case VX_TYPE_Q4_K: return sizeof(block_q4_K);
        case VX_TYPE_Q5_K: return sizeof(block_q5_K);
        case VX_TYPE_Q6_K: return sizeof(block_q6_K);
        case VX_TYPE_Q8_K: return sizeof(block_q8_K);
        default:           return 4;
    }
}

int vx_type_is_quantized(vx_type type) {
    return type >= VX_TYPE_Q4_0 && type <= VX_TYPE_Q8_K;
}

vx_error vx_tensor_alloc(vx_tensor *t, int n_dims, const int *shape, vx_type type) {
    memset(t, 0, sizeof(vx_tensor));
    t->type = type;
    int ne = 1;
    for (int i = 0; i < n_dims; i++) {
        t->ne[i] = shape[i];
        ne *= shape[i];
    }
    if (vx_type_is_quantized(type)) {
        t->nbytes = vx_quantized_size(ne, type);
    } else {
        t->nbytes = (size_t)ne * vx_type_size(type);
    }
    t->data = calloc(1, t->nbytes);
    if (!t->data) return VX_ERR_MEMORY;
    t->owned = true;
    return VX_OK;
}

void vx_tensor_free(vx_tensor *t) {
    if (t && t->data && t->owned) free(t->data);
    if (t) memset(t, 0, sizeof(vx_tensor));
}

void vx_mat_mul_f32(float *dst, const float *a, const float *b,
                    int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            dst[i * n + j] = sum;
        }
    }
}

void vx_gemv_f32_sparse_rows(float *restrict y, const float *restrict A,
                              const float *restrict x, int rows, int cols,
                              const float *restrict gate, float threshold) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (fabsf(gate[i]) < threshold) { y[i] = 0.0f; continue; }
        float sum = 0.0f;
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            sum += A[i * cols + j]     * x[j]     +
                   A[i * cols + j + 1] * x[j + 1] +
                   A[i * cols + j + 2] * x[j + 2] +
                   A[i * cols + j + 3] * x[j + 3];
        }
        for (; j < cols; j++) sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

void vx_gemv_f32_sparse_cols_add(float *restrict y, const float *restrict A,
                                  int rows, int cols,
                                  const float *restrict x, float threshold) {
    for (int j = 0; j < cols; j++) {
        if (fabsf(x[j]) < threshold) continue;
        float xj = x[j];
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            y[i] += A[i * cols + j] * xj;
        }
    }
}

void vx_gemv_f32(float *restrict y, const float *restrict A,
                 const float *restrict x, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            sum += A[i * cols + j]     * x[j]     +
                   A[i * cols + j + 1] * x[j + 1] +
                   A[i * cols + j + 2] * x[j + 2] +
                   A[i * cols + j + 3] * x[j + 3];
        }
        for (; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

void vx_vec_dot_f32(float *dst, const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    *dst = sum;
}

void vx_rms_norm(float *o, const float *x, const float *weight, int n) {
#ifdef __AVX2__
    vx_rms_norm_avx2(o, x, weight, n);
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + 1e-5f);
    if (rms < 1e-10f) {
        for (int i = 0; i < n; i++) o[i] = 0.0f;
        return;
    }
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) o[i] = weight[i] * (x[i] * inv_rms);
#endif
}

void vx_layer_norm(float *o, const float *x, const float *weight, const float *bias, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var = var / n + 1e-5f;
    float inv_std = 1.0f / sqrtf(var);
    for (int i = 0; i < n; i++) {
        o[i] = weight[i] * ((x[i] - mean) * inv_std) + (bias ? bias[i] : 0.0f);
    }
}

void vx_softmax(float *x, int n) {
#ifdef __AVX2__
    vx_softmax_avx2(x, n);
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    if (sum < 1e-10f) {
        for (int i = 0; i < n; i++) x[i] = 1.0f / n;
        return;
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
#endif
}

void vx_rope(float *q, float *k, int pos, int n_heads, int n_kv_heads, int head_dim, int rope_dims, float theta) {
    if (rope_dims <= 0 || rope_dims > head_dim) rope_dims = head_dim;
    int n_q_heads = n_heads;
    int n_k_heads = n_kv_heads > 0 ? n_kv_heads : n_heads;
    int max_heads = n_q_heads > n_k_heads ? n_q_heads : n_k_heads;

    // Precompute sin/cos for all pairs (avoid repeated powf)
    float fcr[4096], fci[4096];
    int npairs = rope_dims / 2;
    for (int i = 0; i < npairs; i++) {
        float freq = 1.0f / powf(theta, (float)(i * 2) / head_dim);
        float val = pos * freq;
        fcr[i] = cosf(val);
        fci[i] = sinf(val);
    }

#ifdef __AVX2__
    for (int h = 0; h < max_heads; h++) {
        if (h < n_q_heads) {
            float *q_h = q + h * head_dim;
            for (int i = 0; i + 4 <= rope_dims; i += 4) {
                __m256 vq = _mm256_loadu_ps(q_h + i);
                // Process in 2-wide: lo=q0,q1  hi=q2,q3
                __m128 lo = _mm256_castps256_ps128(vq);
                __m128 hi = _mm256_extractf128_ps(vq, 1);
                // lo0=q0, lo1=q1, lo2=q2, lo3=q3
                // hi0=q4, hi1=q5, hi2=q6, hi3=q7
                __m128 cr_lo = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(fcr + i/2));
                __m128 ci_lo = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(fci + i/2));
                __m128 cr_hi = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(fcr + i/2 + 2));
                __m128 ci_hi = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(fci + i/2 + 2));
                // Process lo half: pair0 and pair1
                float l0 = lo[0], l1 = lo[1], l2 = lo[2], l3 = lo[3];
                float cr0 = cr_lo[0], ci0 = ci_lo[0];
                float cr1 = cr_lo[1], ci1 = ci_lo[1];
                float r0 = l0*cr0 - l1*ci0, r1 = l0*ci0 + l1*cr0;
                float r2 = l2*cr1 - l3*ci1, r3 = l2*ci1 + l3*cr1;
                // Process hi half: pair2 and pair3
                float h0 = hi[0], h1 = hi[1], h2 = hi[2], h3 = hi[3];
                float cr2 = cr_hi[0], ci2 = ci_hi[0];
                float cr3 = cr_hi[1], ci3 = ci_hi[1];
                float r4 = h0*cr2 - h1*ci2, r5 = h0*ci2 + h1*cr2;
                float r6 = h2*cr3 - h3*ci3, r7 = h2*ci3 + h3*cr3;
                float res[8] = {r0,r1,r2,r3,r4,r5,r6,r7};
                _mm256_storeu_ps(q_h + i, _mm256_loadu_ps(res));
            }
            for (int i = rope_dims & ~3; i < rope_dims; i += 2) {
                int p = i/2;
                float q0 = q_h[i], q1 = q_h[i+1];
                q_h[i]   = q0 * fcr[p] - q1 * fci[p];
                q_h[i+1] = q0 * fci[p] + q1 * fcr[p];
            }
        }
        if (h < n_k_heads) {
            float *k_h = k + h * head_dim;
            for (int i = 0; i + 4 <= rope_dims; i += 4) {
                // same rotation for k
                float k0=k_h[i],k1=k_h[i+1],k2=k_h[i+2],k3=k_h[i+3];
                float k4=k_h[i+4],k5=k_h[i+5],k6=k_h[i+6],k7=k_h[i+7];
                float cr0=fcr[i/2],ci0=fci[i/2];
                float cr1=fcr[i/2+1],ci1=fci[i/2+1];
                float cr2=fcr[i/2+2],ci2=fci[i/2+2];
                float cr3=fcr[i/2+3],ci3=fci[i/2+3];
                k_h[i]=k0*cr0-k1*ci0; k_h[i+1]=k0*ci0+k1*cr0;
                k_h[i+2]=k2*cr1-k3*ci1; k_h[i+3]=k2*ci1+k3*cr1;
                k_h[i+4]=k4*cr2-k5*ci2; k_h[i+5]=k4*ci2+k5*cr2;
                k_h[i+6]=k6*cr3-k7*ci3; k_h[i+7]=k6*ci3+k7*cr3;
            }
            for (int i = rope_dims & ~3; i < rope_dims; i += 2) {
                int p = i/2;
                float k0 = k_h[i], k1 = k_h[i+1];
                k_h[i]   = k0 * fcr[p] - k1 * fci[p];
                k_h[i+1] = k0 * fci[p] + k1 * fcr[p];
            }
        }
    }
#else
    for (int h = 0; h < max_heads; h++) {
        if (h < n_q_heads) {
            float *q_h = q + h * head_dim;
            for (int i = 0; i < rope_dims; i += 2) {
                int p = i/2;
                float q0 = q_h[i], q1 = q_h[i+1];
                q_h[i]   = q0 * fcr[p] - q1 * fci[p];
                q_h[i+1] = q0 * fci[p] + q1 * fcr[p];
            }
        }
        if (h < n_k_heads) {
            float *k_h = k + h * head_dim;
            for (int i = 0; i < rope_dims; i += 2) {
                int p = i/2;
                float k0 = k_h[i], k1 = k_h[i+1];
                k_h[i]   = k0 * fcr[p] - k1 * fci[p];
                k_h[i+1] = k0 * fci[p] + k1 * fcr[p];
            }
        }
    }
#endif
}

void vx_silu(float *x, int n) {
#ifdef __AVX2__
    vx_silu_avx2(x, n);
#else
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
#endif
}

void vx_add_f32(float *c, const float *a, const float *b, int n) {
#ifdef __AVX2__
    vx_add_f32_avx2(c, a, b, n);
#else
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
#endif
}

void vx_mul_f32(float *c, const float *a, const float *b, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(c + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++) c[i] = a[i] * b[i];
#endif
}

void vx_copy_f32(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

float vx_norm_l2(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

float vx_dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void vx_attention(float *out, const float *q, const float *k, const float *v,
                  int n_heads, int head_dim, int seq_len, int kv_len) {
    int n_kv_heads = n_heads;
    int score_size = seq_len * kv_len * sizeof(float);
    float *scores = malloc(score_size);
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_len; j++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++) {
                    s += q[h * head_dim + i * n_heads * head_dim + d] *
                         k[h * head_dim + j * n_kv_heads * head_dim + d];
                }
                if (j > i) s = -1e9f;
                scores[i * kv_len + j] = s;
            }
            vx_softmax(&scores[i * kv_len], kv_len);
        }
    }

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0;
                for (int j = 0; j < kv_len; j++) {
                    sum += scores[i * kv_len + j] *
                           v[h * head_dim + j * n_kv_heads * head_dim + d];
                }
                out[h * head_dim + i * n_heads * head_dim + d] = sum;
            }
        }
    }
    free(scores);
}

void vx_attention_masked(float *out, const float *q, const float *k, const float *v,
                         int n_heads, int head_dim, int seq_len, int kv_len) {
    int n_kv_heads = n_heads;
    float *scores = malloc(seq_len * kv_len * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_len; j++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++) {
                    s += q[h * head_dim + i * n_heads * head_dim + d] *
                         k[h * head_dim + j * n_kv_heads * head_dim + d];
                }
                if (j > i) s = -1e9f;
                scores[i * kv_len + j] = s;
            }
            vx_softmax(&scores[i * kv_len], kv_len);
        }
    }

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0;
                for (int j = 0; j < kv_len; j++) {
                    sum += scores[i * kv_len + j] *
                           v[h * head_dim + j * n_kv_heads * head_dim + d];
                }
                out[h * head_dim + i * n_heads * head_dim + d] = sum;
            }
        }
    }
    free(scores);
}

void vx_gelu(float *x, int n) {
#ifdef __AVX2__
    __m256 vhalf = _mm256_set1_ps(0.5f);
    __m256 va = _mm256_set1_ps(0.7978845608f);
    __m256 vb = _mm256_set1_ps(0.044715f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vx3 = _mm256_mul_ps(vx, _mm256_mul_ps(vx, vx));
        __m256 vinner = _mm256_fmadd_ps(vb, vx3, vx);
        vinner = _mm256_mul_ps(va, vinner);
        // tanhf approximation: vinner anyways
        float tmp[8];
        _mm256_storeu_ps(tmp, vinner);
        for (int j = 0; j < 8; j++) tmp[j] = tanhf(tmp[j]);
        __m256 vtanh = _mm256_loadu_ps(tmp);
        __m256 vresult = _mm256_mul_ps(vhalf, _mm256_mul_ps(vx, _mm256_add_ps(_mm256_set1_ps(1.0f), vtanh)));
        _mm256_storeu_ps(x + i, vresult);
    }
    for (; i < n; i++) {
        float xi = x[i];
        x[i] = 0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + 0.044715f * xi * xi * xi)));
    }
#else
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        x[i] = 0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + 0.044715f * xi * xi * xi)));
    }
#endif
}

void vx_dequantize(const void *src, float *dst, int n, vx_type type) {
    switch (type) {
        case VX_TYPE_Q4_0: vx_dequantize_row_q4_0(src, dst, n); break;
        case VX_TYPE_Q4_1: vx_dequantize_row_q4_1(src, dst, n); break;
        case VX_TYPE_Q5_0: vx_dequantize_row_q5_0(src, dst, n); break;
        case VX_TYPE_Q5_1: vx_dequantize_row_q5_1(src, dst, n); break;
        case VX_TYPE_Q8_0: vx_dequantize_row_q8_0(src, dst, n); break;
        case VX_TYPE_Q8_1: vx_dequantize_row_q8_1(src, dst, n); break;
        case VX_TYPE_Q2_K: vx_dequantize_row_q2_K(src, dst, n); break;
        case VX_TYPE_Q3_K: vx_dequantize_row_q3_K(src, dst, n); break;
        case VX_TYPE_Q4_K: vx_dequantize_row_q4_K(src, dst, n); break;
        case VX_TYPE_Q5_K: vx_dequantize_row_q5_K(src, dst, n); break;
        case VX_TYPE_Q6_K: vx_dequantize_row_q6_K(src, dst, n); break;
        case VX_TYPE_Q8_K: vx_dequantize_row_q8_K(src, dst, n); break;
        default:
            memcpy(dst, src, n * sizeof(float));
            break;
    }
}
