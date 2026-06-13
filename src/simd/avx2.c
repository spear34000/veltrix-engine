#include "simd.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include "quantize.h"

static inline __m256 vx256_exp_ps(__m256 x) {
    float buf[8];
    _mm256_storeu_ps(buf, x);
    for (int j = 0; j < 8; j++) buf[j] = expf(buf[j]);
    return _mm256_loadu_ps(buf);
}

float vx_dot_f32_avx2(const float *a, const float *b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2,3,0,1)));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1,0,3,2)));
    float result = _mm_cvtss_f32(sum128);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

void vx_mat_mul_f32_avx2(float *dst, const float *a, const float *b,
                          int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __m256 sum = _mm256_setzero_ps();
            int l = 0;
            for (; l + 8 <= k; l += 8) {
                __m256 va = _mm256_loadu_ps(a + i * k + l);
                __m256 vb = _mm256_loadu_ps(b + l * n + j);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum),
                                       _mm256_extractf128_ps(sum, 1));
            sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2,3,0,1)));
            sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1,0,3,2)));
            float s = _mm_cvtss_f32(sum128);
            for (; l < k; l++) s += a[i * k + l] * b[l * n + j];
            dst[i * n + j] = s;
        }
    }
}

void vx_rms_norm_avx2(float *o, const float *x, const float *weight, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum = _mm256_fmadd_ps(vx, vx, sum);
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2,3,0,1)));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1,0,3,2)));
    float ss = _mm_cvtss_f32(sum128);
    for (; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + 1e-5f);
    float inv_rms = 1.0f / rms;
    __m256 vinv = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
        _mm256_storeu_ps(o + i, vo);
    }
    for (; i < n; i++) o[i] = weight[i] * (x[i] * inv_rms);
}

void vx_softmax_avx2(float *x, int n) {
    if (n <= 0) return;
    if (n < 8) {
        float max_val = x[0];
        for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < n; i++) x[i] *= inv_sum;
        return;
    }

    __m256 vmax = _mm256_loadu_ps(x);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, vx);
    }
    __m128 max128 = _mm_max_ps(_mm256_castps256_ps128(vmax),
                               _mm256_extractf128_ps(vmax, 1));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2,3,0,1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1,0,3,2)));
    float max_val = _mm_cvtss_f32(max128);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    __m256 vsum = _mm256_setzero_ps();
    __m256 vmaxv = _mm256_set1_ps(max_val);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 ve = vx256_exp_ps(_mm256_sub_ps(vx, vmaxv));
        _mm256_storeu_ps(x + i, ve);
        vsum = _mm256_add_ps(vsum, ve);
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(vsum),
                               _mm256_extractf128_ps(vsum, 1));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2,3,0,1)));
    sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1,0,3,2)));
    float sum = _mm_cvtss_f32(sum128);
    for (; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    float inv_sum = 1.0f / (sum + 1e-10f);
    __m256 vinv = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vinv));
    }
    for (; i < n; i++) x[i] *= inv_sum;
}

void vx_silu_avx2(float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vneg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        __m256 vexp = vx256_exp_ps(vneg);
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vsig = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vsig));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void vx_gemv_f32_avx2_sparse_rows(float *restrict y, const float *restrict A,
                                   const float *restrict x, int rows, int cols,
                                   const float *restrict gate, float threshold) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (fabsf(gate[i]) < threshold) { y[i] = 0.0f; continue; }
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_loadu_ps(A + i * cols + j);
            __m256 vx = _mm256_loadu_ps(x + j);
            sum = _mm256_fmadd_ps(va, vx, sum);
        }
        __m128 sum128 = _mm256_castps256_ps128(sum);
        __m128 sum128h = _mm256_extractf128_ps(sum, 1);
        sum128 = _mm_add_ps(sum128, sum128h);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        for (; j < cols; j++) s += A[i * cols + j] * x[j];
        y[i] = s;
    }
}

void vx_gemv_f32_avx2_sparse_cols_add(float *restrict y, const float *restrict A,
                                       int rows, int cols,
                                       const float *restrict x, float threshold) {
    for (int j = 0; j < cols; j++) {
        if (fabsf(x[j]) < threshold) continue;
        float xj = x[j];
        #pragma omp parallel for
        for (int i = 0; i < rows; i += 8) {
            __m256 va = _mm256_loadu_ps(A + i * cols + j);
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vxj = _mm256_set1_ps(xj);
            vy = _mm256_fmadd_ps(va, vxj, vy);
            _mm256_storeu_ps(y + i, vy);
        }
    }
}

void vx_gemv_f32_avx2(float *restrict y, const float *restrict A,
                      const float *restrict x, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 va = _mm256_loadu_ps(A + i * cols + j);
            __m256 vx = _mm256_loadu_ps(x + j);
            sum = _mm256_fmadd_ps(va, vx, sum);
        }
        __m128 sum128 = _mm256_castps256_ps128(sum);
        __m128 sum128h = _mm256_extractf128_ps(sum, 1);
        sum128 = _mm_add_ps(sum128, sum128h);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        for (; j < cols; j++) {
            s += A[i * cols + j] * x[j];
        }
        y[i] = s;
    }
}

void vx_add_f32_avx2(float *c, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(c + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void vx_gemv_q4_0_avx2_sparse_rows(const void *restrict A, const float *restrict x,
                                    float *restrict y, int rows, int cols,
                                    const float *restrict gate, float threshold) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int nb = (cols + 31) / 32;
    const __m128i m0F = _mm_set1_epi8(0x0F);
    const __m256 vd8 = _mm256_set1_ps(8.0f);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (fabsf(gate[i]) < threshold) { y[i] = 0.0f; continue; }
        __m256 acc = _mm256_setzero_ps();
        int col = 0;

        for (int b = 0; b < nb; b++) {
            float d = vx_fp16_to_fp32(w[i * nb + b].d);
            __m256 vd = _mm256_set1_ps(d);
            __m128i qs = _mm_loadu_si128((const __m128i*)(w[i * nb + b].qs));

            __m128i lo = _mm_and_si128(qs, m0F);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m0F);
            __m128i i0 = _mm_unpacklo_epi8(lo, hi);
            __m128i i1 = _mm_unpackhi_epi8(lo, hi);

            __m256 sum_qx = _mm256_setzero_ps();
            __m256 sum_xv = _mm256_setzero_ps();

            for (int g = 0; g < 4; g++) {
                const uint8_t *base = (const uint8_t*)((g < 2) ? &i0 : &i1);
                int off = (g & 1) * 8;
                __m128i eight = _mm_loadl_epi64(
                    (const __m128i*)(base + off));
                __m128i e_lo = eight;
                __m128i e_hi = _mm_srli_si128(eight, 4);
                __m256i q_lo = _mm256_cvtepi8_epi32(e_lo);
                __m256i q_hi = _mm256_cvtepi8_epi32(e_hi);
                __m256 qf_lo = _mm256_cvtepi32_ps(q_lo);
                __m256 qf_hi = _mm256_cvtepi32_ps(q_hi);
                __m256 vx_lo = _mm256_loadu_ps(&x[col + g * 8 + 0]);
                __m256 vx_hi = _mm256_loadu_ps(&x[col + g * 8 + 4]);
                sum_qx = _mm256_fmadd_ps(qf_lo, vx_lo, sum_qx);
                sum_qx = _mm256_fmadd_ps(qf_hi, vx_hi, sum_qx);
                sum_xv = _mm256_add_ps(sum_xv, vx_lo);
                sum_xv = _mm256_add_ps(sum_xv, vx_hi);
            }

            acc = _mm256_fmadd_ps(vd, sum_qx, acc);
            __m256 offset = _mm256_mul_ps(_mm256_mul_ps(vd, vd8), sum_xv);
            acc = _mm256_sub_ps(acc, offset);

            col += 32;
        }

        y[i] = hsum256(acc);
    }
}

void vx_gemv_q4_0_avx2_sparse_cols_add(const void *restrict A, float *restrict y,
                                         int rows, int cols,
                                         const float *restrict x, float threshold) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int nb = (cols + 31) / 32;

    for (int j = 0; j < cols; j++) {
        if (fabsf(x[j]) < threshold) continue;
        float xj = x[j];
        int blk_idx = j / 32;
        int off = j % 32;

        #pragma omp parallel for
        for (int i = 0; i < rows; i += 8) {
            float vals[8];
            for (int r = 0; r < 8 && i + r < rows; r++) {
                const block_q4_0 *b = &w[(i + r) * nb + blk_idx];
                uint8_t byte = b->qs[off / 2];
                int val = (int)(((byte >> ((off & 1) * 4)) & 0x0F)) - 8;
                vals[r] = val * vx_fp16_to_fp32(b->d);
            }
            __m256 vv = _mm256_loadu_ps(vals);
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vxj = _mm256_set1_ps(xj);
            vy = _mm256_fmadd_ps(vv, vxj, vy);
            _mm256_storeu_ps(y + i, vy);
        }
    }
}

void vx_gemv_q4_0_avx2(const void *restrict A, const float *restrict x,
                        float *restrict y, int rows, int cols) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int nb = (cols + 31) / 32;
    const __m128i m0F = _mm_set1_epi8(0x0F);
    const __m256 vd8 = _mm256_set1_ps(8.0f);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        __m256 acc = _mm256_setzero_ps();
        int col = 0;

        for (int b = 0; b < nb; b++) {
            float d = vx_fp16_to_fp32(w[i * nb + b].d);
            __m256 vd = _mm256_set1_ps(d);
            __m128i qs = _mm_loadu_si128((const __m128i*)(w[i * nb + b].qs));

            __m128i lo = _mm_and_si128(qs, m0F);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m0F);
            __m128i i0 = _mm_unpacklo_epi8(lo, hi);
            __m128i i1 = _mm_unpackhi_epi8(lo, hi);

            __m256 sum_qx = _mm256_setzero_ps();
            __m256 sum_xv = _mm256_setzero_ps();

            for (int g = 0; g < 4; g++) {
                const uint8_t *base = (const uint8_t*)((g < 2) ? &i0 : &i1);
                int off = (g & 1) * 8;
                __m128i eight = _mm_loadl_epi64(
                    (const __m128i*)(base + off));

                __m128i e_lo = eight;
                __m128i e_hi = _mm_srli_si128(eight, 4);

                __m256i q_lo = _mm256_cvtepi8_epi32(e_lo);
                __m256i q_hi = _mm256_cvtepi8_epi32(e_hi);

                __m256 qf_lo = _mm256_cvtepi32_ps(q_lo);
                __m256 qf_hi = _mm256_cvtepi32_ps(q_hi);

                __m256 vx_lo = _mm256_loadu_ps(&x[col + g * 8 + 0]);
                __m256 vx_hi = _mm256_loadu_ps(&x[col + g * 8 + 4]);

                sum_qx = _mm256_fmadd_ps(qf_lo, vx_lo, sum_qx);
                sum_qx = _mm256_fmadd_ps(qf_hi, vx_hi, sum_qx);

                sum_xv = _mm256_add_ps(sum_xv, vx_lo);
                sum_xv = _mm256_add_ps(sum_xv, vx_hi);
            }

            acc = _mm256_fmadd_ps(vd, sum_qx, acc);
            __m256 offset = _mm256_mul_ps(_mm256_mul_ps(vd, vd8), sum_xv);
            acc = _mm256_sub_ps(acc, offset);

            col += 32;
        }

        y[i] = hsum256(acc);
    }
}
