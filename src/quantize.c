#include "quantize.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static float g_synthetic_weight_scale = 1.0f;

void vx_set_synthetic_weight_scale(float scale) {
    g_synthetic_weight_scale = scale;
}

float vx_get_synthetic_weight_scale(void) {
    return g_synthetic_weight_scale;
}

uint16_t vx_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp  = (x >> 23) & 0xFF;
    uint32_t mant = (x)       & 0x7FFFFF;
    if (exp == 0) {
        return (uint16_t)(sign << 15);
    } else if (exp == 0xFF) {
        return (uint16_t)(sign << 15) | (0x1F << 10) | (uint16_t)(mant >> 13);
    }
    int32_t nexp = (int32_t)exp - 127 + 15;
    if (nexp <= 0) {
        return (uint16_t)(sign << 15);
    } else if (nexp >= 31) {
        return (uint16_t)(sign << 15) | (0x1F << 10);
    }
    return (uint16_t)(sign << 15) | ((uint16_t)nexp << 10) | (uint16_t)(mant >> 13);
}

float vx_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h)       & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++; mant &= 0x3FF;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

size_t vx_quantized_size(int n, vx_type type) {
    int block_size;
    switch (type) {
        case VX_TYPE_Q2_K: block_size = 256; break;
        case VX_TYPE_Q3_K:
        case VX_TYPE_Q4_K:
        case VX_TYPE_Q5_K:
        case VX_TYPE_Q6_K:
        case VX_TYPE_Q8_K: block_size = 256; break;
        default: block_size = 32;
    }
    int n_blocks = (n + block_size - 1) / block_size;
    switch (type) {
        case VX_TYPE_Q4_0: return n_blocks * sizeof(block_q4_0);
        case VX_TYPE_Q4_1: return n_blocks * sizeof(block_q4_1);
        case VX_TYPE_Q5_0: return n_blocks * sizeof(block_q5_0);
        case VX_TYPE_Q5_1: return n_blocks * sizeof(block_q5_1);
        case VX_TYPE_Q8_0: return n_blocks * sizeof(block_q8_0);
        case VX_TYPE_Q8_1: return n_blocks * sizeof(block_q8_1);
        case VX_TYPE_Q2_K: return n_blocks * sizeof(block_q2_K);
        case VX_TYPE_Q3_K: return n_blocks * sizeof(block_q3_K);
        case VX_TYPE_Q4_K: return n_blocks * sizeof(block_q4_K);
        case VX_TYPE_Q5_K: return n_blocks * sizeof(block_q5_K);
        case VX_TYPE_Q6_K: return n_blocks * sizeof(block_q6_K);
        case VX_TYPE_Q8_K: return n_blocks * sizeof(block_q8_K);
        default: return (size_t)n * 4;
    }
}

void vx_quantize_row_q4_0(const float *x, void *vy, int n) {
    block_q4_0 *y = (block_q4_0 *)vy;
    for (int i = 0; i < n; i += 32) {
        int rem = (i + 32 <= n) ? 32 : (n - i);
        float amax = 0.0f;
        for (int j = 0; j < rem; j++) {
            float ax = fabsf(x[i + j]);
            if (ax > amax) amax = ax;
        }
        float d = amax / 7.0f;
        if (d < 1e-10f) d = 1e-10f;
        y[i/32].d = vx_fp32_to_fp16(d);
        float id = 1.0f / d;
        for (int j = 0; j < rem; j += 2) {
            int xi0 = (int)(x[i + j] * id);
            int xi1 = (int)(x[i + j + 1] * id);
            if (xi0 < -8) xi0 = -8;
            if (xi0 > 7) xi0 = 7;
            if (xi1 < -8) xi1 = -8;
            if (xi1 > 7) xi1 = 7;
            y[i/32].qs[j/2] = (uint8_t)(xi0 + 8) | ((uint8_t)(xi1 + 8) << 4);
        }
        for (int j = rem; j < 32; j += 2) {
            y[i/32].qs[j/2] = 0;
        }
    }
}

void vx_dequantize_row_q2_K(const void *vx, float *y, int n) {
    const block_q2_K *x = (const block_q2_K *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 256) {
        float d = vx_fp16_to_fp32(x[i/256].d) * scale;
        float dmin = vx_fp16_to_fp32(x[i/256].dmin) * scale;
        const uint8_t *scales = x[i/256].scales;
        const uint8_t *qs = x[i/256].qs;
        for (int j = 0; j < 16; j++) {
            uint8_t sc = scales[j/2];
            uint8_t sc_val = (j & 1) ? (sc >> 4) : (sc & 0xF);
            float dl = d * (sc_val - 8);
            float ml = dmin * (sc_val - 8);
            for (int l = 0; l < 16; l++) {
                int v = (qs[l/4] >> (2 * (l%4))) & 3;
                y[i + j*16 + l] = dl * ((float)v - 0.5f) + ml;
            }
        }
    }
}

void vx_dequantize_row_q3_K(const void *vx, float *y, int n) {
    const block_q3_K *x = (const block_q3_K *)vx;
    const int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t * scales = (const int8_t *)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = vx_fp16_to_fp32(x[i].d);
        const uint8_t * q = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n0 = 0; n0 < QK_K; n0 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void vx_dequantize_row_q4_K(const void *vx, float *y, int n) {
    const block_q4_K *x = (const block_q4_K *)vx;
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = vx_fp16_to_fp32(x[i].d);
        const float min = vx_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void vx_dequantize_row_q5_K(const void *vx, float *y, int n) {
    const block_q5_K *x = (const block_q5_K *)vx;
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = vx_fp16_to_fp32(x[i].d);
        const float min = vx_fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void vx_dequantize_row_q8_K(const void *vx, float *y, int n) {
    const block_q8_K *x = (const block_q8_K *)vx;
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

static void vx_quantize_row_q8_K_ref(const float *x, block_q8_K *y, int n) {
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[i].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j*16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

static float vx_dot_q3_K_q8_K_one(const block_q3_K *x, const block_q8_K *y) {
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8] = {0};
    uint32_t auxs[4];
    const int8_t *scales = (const int8_t*)auxs;

    const uint8_t *q3 = x->qs;
    const uint8_t *hm = x->hmask;
    const int8_t *q8 = y->qs;
    int8_t *a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
        for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
        a += 32; m <<= 1;
        for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
        for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
        a += 32; m <<= 1;
        for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
        for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
        a += 32; m <<= 1;
        for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
        for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
        a += 32; m <<= 1;
        q3 += 32;
    }
    memcpy(auxs, x->scales, 12);
    uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    a = aux8;
    for (int j = 0; j < QK_K/16; ++j) {
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
        q8 += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
        q8 += 8; a += 8;
    }
    float sumf = 0.0f;
    const float d = vx_fp16_to_fp32(x->d) * y->d;
    for (int l = 0; l < 8; ++l) sumf += d * aux32[l];
    return sumf;
}

static float vx_dot_q2_K_q8_K_one(const block_q2_K *x, const block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;

    int summs = 0;
    for (int j = 0; j < 16; ++j) {
        summs += y->bsums[j] * (sc[j] >> 4);
    }

    const float dall = y->d * vx_fp16_to_fp32(x->d);
    const float dmin = y->d * vx_fp16_to_fp32(x->dmin);

    int isum = 0;
    int is = 0;
    for (int k = 0; k < QK_K/128; ++k) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            int d = sc[is++] & 0xF;
            int isuml = 0;
            for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
            isum += d * isuml;
            d = sc[is++] & 0xF;
            isuml = 0;
            for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
            isum += d * isuml;
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

static float vx_dot_q4_K_q8_K_one(const block_q4_K *x, const block_q8_K *y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8] = {0};

    const uint8_t *q4 = x->qs;
    const int8_t *q8 = y->qs;
    int8_t *a = aux8;
    memset(aux32, 0, sizeof(aux32));
    for (int j = 0; j < QK_K/64; ++j) {
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
        a += 32;
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
        a += 32;
        q4 += 32;
    }
    memcpy(utmp, x->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < QK_K/16; ++j) sumi += y->bsums[j] * mins[j/2];
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K/32; ++j) {
        int32_t scale = scales[is++];
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
    }
    float sumf = 0.0f;
    const float d = vx_fp16_to_fp32(x->d) * y->d;
    const float dmin = vx_fp16_to_fp32(x->dmin) * y->d;
    for (int l = 0; l < 8; ++l) sumf += d * aux32[l];
    sumf -= dmin * sumi;
    return sumf;
}

void vx_dequantize_row_q6_K(const void *vx, float *y, int n) {
    const block_q6_K *x = (const block_q6_K *)vx;
    for (int i = 0; i < n; i += 256) {
        float d = vx_fp16_to_fp32(x[i/256].d);
        const int8_t *sc = x[i/256].scales;
        const uint8_t *ql = x[i/256].ql;
        const uint8_t *qh = x[i/256].qh;
        for (int j = 0; j < 256; j++) {
            int low = (ql[j/2] >> ((j % 2) * 4)) & 0xF;
            int high = (qh[j/4] >> ((j % 4) * 2)) & 0x3;
            int val = (high << 4) | low;
            y[i + j] = d * (float)sc[j/16] * (float)(val - 32);
        }
    }
}

void vx_gemv_q2_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    const int nb = (cols + 255) / 256;
    const block_q2_K *w = (const block_q2_K *)A;
    float *xpad = calloc((size_t)nb * 256, sizeof(float));
    block_q8_K *qx = malloc((size_t)nb * sizeof(block_q8_K));
    if (!xpad || !qx) {
        free(xpad);
        free(qx);
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (int b = 0; b < nb; b++) {
                float d = vx_fp16_to_fp32(w[i * nb + b].d);
                float dmin = vx_fp16_to_fp32(w[i * nb + b].dmin);
                const uint8_t *scales = w[i * nb + b].scales;
                const uint8_t *qs = w[i * nb + b].qs;
                for (int j = 0; j < 16; j++) {
                    uint8_t sc = scales[j/2];
                    uint8_t sc_val = (j & 1) ? (sc >> 4) : (sc & 0xF);
                    float dl = d * (sc_val - 8);
                    float ml = dmin * (sc_val - 8);
                    for (int l = 0; l < 16; l++) {
                        int v = (qs[l/4] >> (2 * (l%4))) & 3;
                        sum += (dl * ((float)v - 0.5f) + ml) * x[b * 256 + j * 16 + l];
                    }
                }
            }
            y[i] = sum;
        }
        return;
    }
    memcpy(xpad, x, (size_t)cols * sizeof(float));
    vx_quantize_row_q8_K_ref(xpad, qx, nb * 256);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        const block_q2_K *row = w + (size_t)i * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            sum += vx_dot_q2_K_q8_K_one(&row[b], &qx[b]);
        }
        y[i] = sum;
    }
    free(xpad);
    free(qx);
}

static void gemv_k_blockwise(int rows, int cols, const void *A, const float *x,
                             float *y, size_t block_bytes,
                             void (*deq)(const void *, float *, int)) {
    const int nb = (cols + 255) / 256;
    const uint8_t *base = (const uint8_t *)A;
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        float tmp[256];
        const uint8_t *row = base + (size_t)i * nb * block_bytes;
        for (int b = 0; b < nb; b++) {
            deq(row + (size_t)b * block_bytes, tmp, 256);
            const float *xs = x + b * 256;
            int rem = cols - b * 256;
            if (rem > 256) rem = 256;
            for (int j = 0; j < rem; j++) {
                sum += tmp[j] * xs[j];
            }
        }
        y[i] = sum;
    }
}

void vx_gemv_q3_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    const int nb = (cols + 255) / 256;
    const block_q3_K *w = (const block_q3_K *)A;
    float *xpad = calloc((size_t)nb * 256, sizeof(float));
    block_q8_K *qx = malloc((size_t)nb * sizeof(block_q8_K));
    if (!xpad || !qx) {
        free(xpad);
        free(qx);
        gemv_k_blockwise(rows, cols, A, x, y, sizeof(block_q3_K), vx_dequantize_row_q3_K);
        return;
    }
    memcpy(xpad, x, (size_t)cols * sizeof(float));
    vx_quantize_row_q8_K_ref(xpad, qx, nb * 256);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        const block_q3_K *row = w + (size_t)i * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            sum += vx_dot_q3_K_q8_K_one(&row[b], &qx[b]);
        }
        y[i] = sum;
    }
    free(xpad);
    free(qx);
}

void vx_gemv_q4_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    const int nb = (cols + 255) / 256;
    const block_q4_K *w = (const block_q4_K *)A;
    float *xpad = calloc((size_t)nb * 256, sizeof(float));
    block_q8_K *qx = malloc((size_t)nb * sizeof(block_q8_K));
    if (!xpad || !qx) {
        free(xpad);
        free(qx);
        gemv_k_blockwise(rows, cols, A, x, y, sizeof(block_q4_K), vx_dequantize_row_q4_K);
        return;
    }
    memcpy(xpad, x, (size_t)cols * sizeof(float));
    vx_quantize_row_q8_K_ref(xpad, qx, nb * 256);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        const block_q4_K *row = w + (size_t)i * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            sum += vx_dot_q4_K_q8_K_one(&row[b], &qx[b]);
        }
        y[i] = sum;
    }
    free(xpad);
    free(qx);
}

void vx_gemv_q5_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    gemv_k_blockwise(rows, cols, A, x, y, sizeof(block_q5_K), vx_dequantize_row_q5_K);
}

void vx_gemv_q6_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    gemv_k_blockwise(rows, cols, A, x, y, sizeof(block_q6_K), vx_dequantize_row_q6_K);
}

void vx_gemv_q8_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    gemv_k_blockwise(rows, cols, A, x, y, sizeof(block_q8_K), vx_dequantize_row_q8_K);
}

void vx_dequantize_row_q4_0(const void *vx, float *y, int n) {
    const block_q4_0 *x = (const block_q4_0 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d) * scale;
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            int8_t xi0 = (int8_t)(byte & 0x0F) - 8;
            int8_t xi1 = (int8_t)(byte >> 4) - 8;
            y[i + j]     = xi0 * d;
            y[i + j + 1] = xi1 * d;
        }
    }
}

void vx_dequantize_row_q4_1(const void *vx, float *y, int n) {
    const block_q4_1 *x = (const block_q4_1 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d) * scale;
        float m = vx_fp16_to_fp32(x[i/32].m) * scale;
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            y[i + j]     = (byte & 0x0F) * d + m;
            y[i + j + 1] = (byte >> 4) * d + m;
        }
    }
}

void vx_dequantize_row_q5_0(const void *vx, float *y, int n) {
    const block_q5_0 *x = (const block_q5_0 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d) * scale;
        uint32_t h = 0;
        memcpy(&h, x[i/32].qh, 4);
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            int16_t xi0 = (int16_t)(byte & 0x0F) | (((h >> (j/2)) & 1) << 4);
            int16_t xi1 = (int16_t)(byte >> 4) | (((h >> (j/2 + 16)) & 1) << 4);
            y[i + j]     = (xi0 - 16) * d;
            y[i + j + 1] = (xi1 - 16) * d;
        }
    }
}

void vx_dequantize_row_q5_1(const void *vx, float *y, int n) {
    const block_q5_1 *x = (const block_q5_1 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d) * scale;
        float m = vx_fp16_to_fp32(x[i/32].m) * scale;
        uint32_t h = 0;
        memcpy(&h, x[i/32].qh, 4);
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            uint16_t xi0 = (uint16_t)(byte & 0x0F) | (((h >> j) & 1) << 4);
            uint16_t xi1 = (uint16_t)(byte >> 4) | (((h >> (j + 16)) & 1) << 4);
            y[i + j]     = xi0 * d + m;
            y[i + j + 1] = xi1 * d + m;
        }
    }
}

void vx_dequantize_row_q8_0(const void *vx, float *y, int n) {
    const block_q8_0 *x = (const block_q8_0 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d) * scale;
        for (int j = 0; j < 32 && (i + j) < n; j++) {
            y[i + j] = x[i/32].qs[j] * d;
        }
    }
}

void vx_dequantize_row_q8_1(const void *vx, float *y, int n) {
    const block_q8_1 *x = (const block_q8_1 *)vx;
    float scale = g_synthetic_weight_scale;
    for (int i = 0; i < n; i += 32) {
        float d = x[i/32].d * scale;
        float s = x[i/32].s * scale;
        for (int j = 0; j < 32 && (i + j) < n; j++) {
            y[i + j] = x[i/32].qs[j] * d + s;
        }
    }
}

void vx_dot_q4_0(const void *vx, const float *y, float *dst, int n) {
    const block_q4_0 *x = (const block_q4_0 *)vx;
    float sum = 0.0f;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d);
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            int8_t xi0 = (int8_t)(byte & 0x0F) - 8;
            int8_t xi1 = (int8_t)(byte >> 4) - 8;
            sum += (xi0 * d) * y[i + j] + (xi1 * d) * y[i + j + 1];
        }
    }
    *dst = sum;
}

void vx_gemv_q4_0_sparse_rows(const void *restrict A, const float *restrict x,
                               float *restrict y, int rows, int cols,
                               const float *restrict gate, float threshold) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int blk_size = 32;
    int n_blocks = (cols + blk_size - 1) / blk_size;
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (fabsf(gate[i]) < threshold) { y[i] = 0.0f; continue; }
        float sum = 0.0f;
        int j = 0;
        for (int b = 0; b < n_blocks; b++) {
            float d = vx_fp16_to_fp32(w[i * n_blocks + b].d);
            int lim = (j + blk_size <= cols) ? blk_size : (cols - j);
            for (int k = 0; k < lim; k += 2) {
                uint8_t byte = w[i * n_blocks + b].qs[k / 2];
                int xi0 = (int)(byte & 0x0F) - 8;
                int xi1 = (int)(byte >> 4) - 8;
                sum += (xi0 * d) * x[j + k] + (xi1 * d) * x[j + k + 1];
            }
            j += blk_size;
        }
        y[i] = sum;
    }
}

void vx_gemv_q4_0_sparse_cols_add(const void *restrict A, float *restrict y,
                                    int rows, int cols,
                                    const float *restrict x, float threshold) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int n_blocks = (cols + 31) / 32;
    for (int j = 0; j < cols; j++) {
        if (fabsf(x[j]) < threshold) continue;
        float xj = x[j];
        int blk_idx = j / 32;
        int off = j % 32;
        int n_shift = (off & 1) * 4;
        int half = off / 2;
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            const block_q4_0 *b = &w[i * n_blocks + blk_idx];
            uint8_t byte = b->qs[half];
            int val = (int)(((byte >> n_shift) & 0x0F)) - 8;
            y[i] += xj * (val * vx_fp16_to_fp32(b->d));
        }
    }
}

void vx_gemv_q4_0(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols) {
    const block_q4_0 *w = (const block_q4_0 *)A;
    int blk_size = 32;
    int n_blocks = (cols + blk_size - 1) / blk_size;
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        int j = 0;
        for (int b = 0; b < n_blocks; b++) {
            float d = vx_fp16_to_fp32(w[i * n_blocks + b].d);
            int lim = (j + blk_size <= cols) ? blk_size : (cols - j);
            for (int k = 0; k < lim; k += 2) {
                uint8_t byte = w[i * n_blocks + b].qs[k / 2];
                int xi0 = (int)(byte & 0x0F) - 8;
                int xi1 = (int)(byte >> 4) - 8;
                sum += (xi0 * d) * x[j + k] + (xi1 * d) * x[j + k + 1];
            }
            j += blk_size;
        }
        y[i] = sum;
    }
}

void vx_dot_q4_1(const void *vx, const float *y, float *dst, int n) {
    const block_q4_1 *x = (const block_q4_1 *)vx;
    float sum = 0.0f;
    for (int i = 0; i < n; i += 32) {
        float d = vx_fp16_to_fp32(x[i/32].d);
        float m = vx_fp16_to_fp32(x[i/32].m);
        for (int j = 0; j < 32 && (i + j) < n; j += 2) {
            uint8_t byte = x[i/32].qs[j/2];
            float x0 = (byte & 0x0F) * d + m;
            float x1 = (byte >> 4) * d + m;
            sum += x0 * y[i + j] + x1 * y[i + j + 1];
        }
    }
    *dst = sum;
}
