#include "model.h"
#include "tensor.h"
#include "quantize.h"
#include "metalearn.h"
#include "scheduler.h"
#include "format.h"
#include "gguf.h"
#include "tokenizer.h"
#include "platform.h"
#include "simd/simd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

vx_error vx_model_create(const char *path, vx_model **out) {
    vx_format_init();

    FILE *fp = fopen(path, "rb");
    if (!fp) return VX_ERR_FILE;
    uint8_t magic[16];
    size_t nread = fread(magic, 1, 16, fp);
    fclose(fp);

    vx_format fmt = vx_detect_format(magic, nread, path);
    if (fmt == VX_FORMAT_UNKNOWN) {
        fprintf(stderr, "Unknown model format: %s\n", path);
        return VX_ERR_UNSUPPORTED;
    }

    const vx_loader *loader = vx_loader_for_format(fmt);
    if (!loader) {
        fprintf(stderr, "No loader for format %s: %s\n", vx_format_name(fmt), path);
        return VX_ERR_UNSUPPORTED;
    }

    vx_model *model = calloc(1, sizeof(vx_model));
    if (!model) return VX_ERR_MEMORY;

    vx_error err = loader->load(path, model);
    if (err != VX_OK) {
        vx_model_destroy(model);
        return err;
    }

    err = vx_model_resolve_weights(model);
    if (err != VX_OK) {
        vx_model_destroy(model);
        return err;
    }

    // Consolidate all tensor data into a single contiguous allocation
    size_t total_data = 0;
    for (int i = 0; i < model->n_tensors; i++) {
        total_data += model->tensors[i]->nbytes;
    }
    uint8_t *big = malloc(total_data);
    if (!big) { vx_model_destroy(model); return VX_ERR_MEMORY; }
    size_t off = 0;
    for (int i = 0; i < model->n_tensors; i++) {
        vx_tensor *t = model->tensors[i];
        if (t->data) {
            memcpy(big + off, t->data, t->nbytes);
            free(t->data);
        }
        t->data = big + off;
        t->owned = false;
        off += t->nbytes;
    }
    model->tensor_data = big;

    // Allocate reusable dequant temp buffer (size of largest tensor's dequantized form)
    size_t max_tensor = 0;
    for (int i = 0; i < model->n_tensors; i++) {
        size_t nb = model->tensors[i]->nbytes;
        if (vx_type_is_quantized(model->tensors[i]->type)) {
            int ne = 1;
            for (int d = 0; d < 4 && model->tensors[i]->ne[d] > 0; d++)
                ne *= model->tensors[i]->ne[d];
            nb = (size_t)ne * sizeof(float);
        }
        if (nb > max_tensor) max_tensor = nb;
    }
    model->dequant_tmp = malloc(max_tensor);
    model->dequant_tmp_size = max_tensor;

    vx_model_auto_detect_dims(model);

    model->layer_predictors = NULL;
    model->attn_pattern_predictor = NULL;
    model->token_exit_predictor = NULL;
    model->k_cache = NULL;
    model->v_cache = NULL;
    model->cache_len = 0;
    model->tokenizer = NULL;

    // Load tokenizer from model file (best-effort)
    vx_tokenizer *tok = calloc(1, sizeof(vx_tokenizer));
    if (tok) {
        if (vx_tokenizer_auto_load(path, tok) == VX_OK) {
            model->tokenizer = tok;
        } else {
            free(tok);
        }
    }

    *out = model;
    return VX_OK;
}

void vx_model_destroy(vx_model *model) {
    if (!model) return;
    free(model->tensor_data);
    if (model->tensors) {
        for (int i = 0; i < model->n_tensors; i++) {
            if (model->tensors[i]) {
                // data is owned by tensor_data, don't free individually
                model->tensors[i]->data = NULL;
                free(model->tensors[i]);
            }
        }
        free(model->tensors);
    }
    free(model->attn_q);
    free(model->attn_k);
    free(model->attn_v);
    free(model->attn_o);
    free(model->attn_q_bias);
    free(model->attn_k_bias);
    free(model->attn_v_bias);
    free(model->ffn_gate);
    free(model->ffn_up);
    free(model->ffn_down);
    free(model->ffn_gate_bias);
    free(model->ffn_up_bias);
    free(model->ffn_down_bias);
    free(model->attn_norm);
    free(model->ffn_norm);
    free(model->attn_norm_bias);
    free(model->ffn_norm_bias);
    if (model->layer_predictors) {
        for (int i = 0; i < model->config.n_layers; i++) {
            if (model->layer_predictors[i])
                vx_meta_predictor_destroy(model->layer_predictors[i]);
        }
        free(model->layer_predictors);
    }
    if (model->attn_pattern_predictor)
        vx_meta_predictor_destroy(model->attn_pattern_predictor);
    if (model->token_exit_predictor)
        vx_meta_predictor_destroy(model->token_exit_predictor);
    free(model->k_cache);
    free(model->v_cache);
    free(model->scratch);
    model->scratch = NULL;
    model->scratch_size = 0;
    free(model->dequant_tmp);
    model->dequant_tmp = NULL;
    model->dequant_tmp_size = 0;
    if (model->tokenizer) {
        vx_tokenizer_free(model->tokenizer);
        free(model->tokenizer);
    }
    free(model);
}

static void apply_gemv_sparse_rows(vx_tensor *w, const float *x, float *y,
                                    int rows, int cols,
                                    const float *gate, float threshold,
                                    float *dqtmp) {
    if (!w || !w->data) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
    if (w->type == VX_TYPE_F32) {
#ifdef __AVX2__
        vx_gemv_f32_avx2_sparse_rows(y, (const float *)w->data, x, rows, cols, gate, threshold);
#else
        vx_gemv_f32_sparse_rows(y, (const float *)w->data, x, rows, cols, gate, threshold);
#endif
    } else if (w->type == VX_TYPE_Q4_0) {
#ifdef __AVX2__
        vx_gemv_q4_0_avx2_sparse_rows(w->data, x, y, rows, cols, gate, threshold);
#else
        float *tmp = dqtmp ? dqtmp : malloc((size_t)rows * cols * sizeof(float));
        if (!tmp) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
        vx_dequantize_row_q4_0(w->data, tmp, rows * cols);
        vx_gemv_f32_sparse_rows(y, tmp, x, rows, cols, gate, threshold);
        if (!dqtmp) free(tmp);
#endif
    } else {
        float *tmp = dqtmp ? dqtmp : malloc((size_t)rows * cols * sizeof(float));
        if (!tmp) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
        vx_dequantize(w->data, tmp, rows * cols, w->type);
#ifdef __AVX2__
        vx_gemv_f32_avx2_sparse_rows(y, tmp, x, rows, cols, gate, threshold);
#else
        vx_gemv_f32_sparse_rows(y, tmp, x, rows, cols, gate, threshold);
#endif
        if (!dqtmp) free(tmp);
    }
}

static void apply_gemv(vx_tensor *w, const float *x, float *y, int rows, int cols,
                       float *dqtmp) {
    if (!w || !w->data) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
    if (w->type == VX_TYPE_F32) {
#ifdef __AVX2__
        vx_gemv_f32_avx2(y, (const float *)w->data, x, rows, cols);
#else
        vx_gemv_f32(y, (const float *)w->data, x, rows, cols);
#endif
    } else if (w->type == VX_TYPE_Q4_0) {
#ifdef __AVX2__
        vx_gemv_q4_0_avx2(w->data, x, y, rows, cols);
#else
        float *tmp = dqtmp ? dqtmp : malloc((size_t)rows * cols * sizeof(float));
        if (!tmp) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
        vx_dequantize_row_q4_0(w->data, tmp, rows * cols);
        vx_gemv_f32(y, tmp, x, rows, cols);
        if (!dqtmp) free(tmp);
#endif
    } else if (w->type == VX_TYPE_Q2_K) {
        vx_gemv_q2_K(w->data, x, y, rows, cols);
    } else if (w->type == VX_TYPE_Q3_K) {
        vx_gemv_q3_K(w->data, x, y, rows, cols);
    } else if (w->type == VX_TYPE_Q4_K) {
        vx_gemv_q4_K(w->data, x, y, rows, cols);
    } else if (w->type == VX_TYPE_Q5_K) {
        vx_gemv_q5_K(w->data, x, y, rows, cols);
    } else if (w->type == VX_TYPE_Q6_K) {
        vx_gemv_q6_K(w->data, x, y, rows, cols);
    } else if (w->type == VX_TYPE_Q8_K) {
        vx_gemv_q8_K(w->data, x, y, rows, cols);
    } else {
        float *tmp = dqtmp ? dqtmp : malloc((size_t)rows * cols * sizeof(float));
        if (!tmp) { memset(y, 0, (size_t)rows * sizeof(float)); return; }
        vx_dequantize(w->data, tmp, rows * cols, w->type);
#ifdef __AVX2__
        vx_gemv_f32_avx2(y, tmp, x, rows, cols);
#else
        vx_gemv_f32(y, tmp, x, rows, cols);
#endif
        if (!dqtmp) free(tmp);
    }
}

static void apply_norm(float *x, float *tmp, vx_tensor *w, vx_tensor *b, int dim, vx_norm_type norm_type) {
    float nw[8192];
    float nb[8192];
    const float *weight = NULL;
    const float *bias = NULL;

    if (w && w->data && w->type == VX_TYPE_F32) {
        weight = (const float *)w->data;
    } else {
        for (int i = 0; i < dim && i < 8192; i++) nw[i] = 1.0f;
        weight = nw;
    }

    if (norm_type == VX_NORM_LAYER) {
        if (b && b->data && b->type == VX_TYPE_F32) {
            bias = (const float *)b->data;
        } else {
            memset(nb, 0, (size_t)dim * sizeof(float));
            bias = nb;
        }
        vx_layer_norm(tmp, x, weight, bias, dim);
    } else {
        vx_rms_norm(tmp, x, weight, dim);
    }

    memcpy(x, tmp, (size_t)dim * sizeof(float));
}

static void apply_attention(vx_model *model, float *x, float *q, float *k, float *v,
                            float *output, int n_heads, int n_kv_heads, int head_dim, int layer) {
    int n_embd = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int pos = model->cache_len;
    int max_ctx = model->config.n_ctx;
    if (max_ctx < 1) max_ctx = 4096;
    if (pos >= max_ctx) {
        pos = max_ctx - 1;
        model->cache_len = pos;
    }
    int cache_size = pos + 1;

    float *k_cache = model->k_cache;
    float *v_cache = model->v_cache;

    size_t layer_off = (size_t)layer * (size_t)max_ctx * kv_dim;

    float *new_k = k_cache + layer_off + (size_t)pos * kv_dim;
    float *new_v = v_cache + layer_off + (size_t)pos * kv_dim;
    memcpy(new_k, k, kv_dim * sizeof(float));
    memcpy(new_v, v, kv_dim * sizeof(float));

    memset(output, 0, (size_t)n_embd * sizeof(float));

    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        float *q_h = q + h * head_dim;
        int kv_h = n_kv_heads == 1 ? 0 : (h * n_kv_heads / n_heads);
        float local_scores[4096];
        int cs = cache_size > 4096 ? 4096 : cache_size;

        float scale = 1.0f / sqrtf((float)head_dim);
        for (int t = 0; t < cs; t++) {
            float *k_t = k_cache + layer_off + (size_t)t * kv_dim + kv_h * head_dim;
            float s = 0;
            int d = 0;
#ifdef __AVX2__
            __m256 vsum = _mm256_setzero_ps();
            for (; d + 8 <= head_dim; d += 8) {
                __m256 vq = _mm256_loadu_ps(q_h + d);
                __m256 vk = _mm256_loadu_ps(k_t + d);
                vsum = _mm256_fmadd_ps(vq, vk, vsum);
            }
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            s = _mm_cvtss_f32(sum128);
#endif
            for (; d < head_dim; d++) s += q_h[d] * k_t[d];
            local_scores[t] = (t <= pos) ? s * scale : -1e9f;
        }

        vx_softmax(local_scores, cs);

        for (int d = 0; d < head_dim; d++) {
            float sum = 0;
            for (int t = 0; t < cs; t++)
                sum += local_scores[t] * (v_cache[layer_off + (size_t)t * kv_dim + kv_h * head_dim + d]);
            output[h * head_dim + d] = sum;
        }
    }

    memcpy(x, output, (size_t)n_embd * sizeof(float));
}

typedef struct {
    double embed_us;
    double norm_us;
    double qkv_us;
    double attn_us;
    double ffn_us;
    double logits_us;
} vx_forward_timing;

static vx_forward_timing g_forward_timing = {0};
static int g_forward_timing_enabled = 0;
static vx_compute_level g_compute_level = VX_COMPUTE_EXACT;

void vx_set_forward_timing_enabled(int enabled) {
    g_forward_timing_enabled = enabled ? 1 : 0;
    memset(&g_forward_timing, 0, sizeof(g_forward_timing));
}

void vx_set_compute_level(vx_compute_level level) {
    g_compute_level = level;
}

void vx_print_forward_timing(void) {
    if (!g_forward_timing_enabled) return;
    fprintf(stderr,
            "[timing] embed=%.1fms norm=%.1fms qkv=%.1fms attn=%.1fms ffn=%.1fms logits=%.1fms\n",
            g_forward_timing.embed_us / 1000.0,
            g_forward_timing.norm_us / 1000.0,
            g_forward_timing.qkv_us / 1000.0,
            g_forward_timing.attn_us / 1000.0,
            g_forward_timing.ffn_us / 1000.0,
            g_forward_timing.logits_us / 1000.0);
}

void vx_set_n_threads(int n) {
    if (n < 1) n = 1;
    if (n > 128) n = 128;
    omp_set_dynamic(0);
    omp_set_num_threads(n);
}

void vx_model_kv_truncate(vx_model *model, int length) {
    if (!model) return;
    if (length < 0) length = 0;
    if (length <= model->cache_len)
        model->cache_len = length;
}

vx_error vx_model_forward(vx_model *model, const int *tokens, int n_tokens, float *logits) {
    if (!model || !tokens || !logits) return VX_ERR_PARAM;

    int n_embd = model->config.n_embd;
    int n_layers = model->config.n_layers;
    int n_heads = model->config.n_heads;
    int n_kv_heads = model->config.n_kv_heads;
    int head_dim = model->config.n_head_dim;
    int n_ff = model->config.n_ff;
    int n_vocab = model->config.n_vocab;

    if (n_embd <= 0 || n_embd > 8192) return VX_ERR_PARAM;

    if (model->k_cache == NULL) {
        int max_ctx = model->config.n_ctx;
        if (max_ctx <= 0) max_ctx = 4096;
        int n_layers_cached = n_layers > 0 ? n_layers : 1;
        model->k_cache = calloc((size_t)n_layers_cached * max_ctx * n_kv_heads * head_dim, sizeof(float));
        model->v_cache = calloc((size_t)n_layers_cached * max_ctx * n_kv_heads * head_dim, sizeof(float));
        model->cache_len = 0;
    }

    // Allocate permanent scratch on first use; grow if needed
    {
        int _kv_dim = n_kv_heads * head_dim;
        size_t need = ((size_t)n_embd * 4 + (size_t)_kv_dim * 2 + (size_t)n_ff * 2 + (size_t)n_embd + 64) * sizeof(float);
        if (need > model->scratch_size) {
            free(model->scratch);
            model->scratch = malloc(need);
            model->scratch_size = model->scratch ? need : 0;
        }
    }
    if (!model->scratch) return VX_ERR_MEMORY;

    float *x = model->scratch;
    float *x_orig = x + n_embd;
    float *q = x_orig + n_embd;
    int kv_dim = n_kv_heads * head_dim;
    float *k = q + n_embd;
    float *v = k + kv_dim + 8;
    float *attn_out = v + kv_dim + 8;
    float *ffn_h = n_ff > 0 ? attn_out + n_embd : NULL;
    float *ffn_h2 = ffn_h ? ffn_h + n_ff + 8 : NULL;
    // norm_tmp reuses ffn_h region (norm runs before ffn in each layer)
    float *norm_tmp = attn_out + n_embd;

    double t_embed_us = 0.0;
    double t_norm_us = 0.0;
    double t_qkv_us = 0.0;
    double t_attn_us = 0.0;
    double t_ffn_us = 0.0;
    double t_logits_us = 0.0;

    vx_tensor *tok_embd = model->tok_embd;
    vx_tensor *output_weight = model->output_w;
    vx_tensor *norm_final_w = model->norm_w;

    for (int pos = 0; pos < n_tokens; pos++) {
        int token = tokens[pos];
        int cache_pos = model->cache_len;

        // Embedding lookup
        double t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
        memset(x, 0, n_embd * sizeof(float));
        if (tok_embd && tok_embd->data && token >= 0) {
            int tok_stride = tok_embd->ne[0] > 0 ? (int)tok_embd->ne[0] : n_embd;
            if (tok_embd->type == VX_TYPE_F32) {
                float *table = (float *)tok_embd->data;
                int tok_off = token * tok_stride;
                if (tok_off >= 0 && tok_off + n_embd <= (int)(tok_embd->nbytes / sizeof(float)))
                    memcpy(x, table + tok_off, n_embd * sizeof(float));
            } else {
                int start_elem = token * tok_stride;
                int qblock_size = (tok_embd->type == VX_TYPE_Q2_K || tok_embd->type == VX_TYPE_Q3_K || tok_embd->type == VX_TYPE_Q4_K || tok_embd->type == VX_TYPE_Q5_K || tok_embd->type == VX_TYPE_Q6_K || tok_embd->type == VX_TYPE_Q8_K) ? 256 : 32;
                size_t qblock_bytes;
                switch (tok_embd->type) {
                    case VX_TYPE_Q4_0: qblock_bytes = sizeof(block_q4_0); break;
                    case VX_TYPE_Q4_1: qblock_bytes = sizeof(block_q4_1); break;
                    case VX_TYPE_Q5_0: qblock_bytes = sizeof(block_q5_0); break;
                    case VX_TYPE_Q5_1: qblock_bytes = sizeof(block_q5_1); break;
                    case VX_TYPE_Q8_0: qblock_bytes = sizeof(block_q8_0); break;
                    case VX_TYPE_Q8_1: qblock_bytes = sizeof(block_q8_1); break;
                    case VX_TYPE_Q2_K: qblock_bytes = sizeof(block_q2_K); break;
                    case VX_TYPE_Q3_K: qblock_bytes = sizeof(block_q3_K); break;
                    case VX_TYPE_Q4_K: qblock_bytes = sizeof(block_q4_K); break;
                    case VX_TYPE_Q5_K: qblock_bytes = sizeof(block_q5_K); break;
                    case VX_TYPE_Q6_K: qblock_bytes = sizeof(block_q6_K); break;
                    case VX_TYPE_Q8_K: qblock_bytes = sizeof(block_q8_K); break;
                    default: qblock_bytes = 4;
                }
                int offset_in_block = start_elem % qblock_size;
                size_t byte_off = (size_t)(start_elem / qblock_size) * qblock_bytes;
                int n_to_deq = tok_stride + offset_in_block;
                float *buf = model->dequant_tmp;
                size_t need = (size_t)n_to_deq * sizeof(float);
                if (buf && model->dequant_tmp_size >= need) {
                    vx_dequantize((const uint8_t*)tok_embd->data + byte_off, buf, n_to_deq, tok_embd->type);
                    memcpy(x, buf + offset_in_block, (size_t)tok_stride * sizeof(float));
                } else {
                    float *tmp = malloc(need);
                    if (tmp) {
                        vx_dequantize((const uint8_t*)tok_embd->data + byte_off, tmp, n_to_deq, tok_embd->type);
                        memcpy(x, tmp + offset_in_block, (size_t)tok_stride * sizeof(float));
                        free(tmp);
                    }
                }
            }
        }
        if (g_forward_timing_enabled) t_embed_us += vx_time_now_us() - t_stage;

        int layer_limit = n_layers;
        int skip_ffn = 0;
        if (g_compute_level == VX_COMPUTE_ATTN_ONLY) {
            layer_limit = n_layers < 4 ? n_layers : 4;
            skip_ffn = 1;
        } else if (g_compute_level == VX_COMPUTE_SKIP) {
            layer_limit = n_layers < 1 ? n_layers : 1;
            skip_ffn = 1;
        }

        // Forward through selected layers
        for (int l = 0; l < layer_limit; l++) {
            memcpy(x_orig, x, n_embd * sizeof(float));

            t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
            apply_norm(x, norm_tmp, model->attn_norm[l], model->attn_norm_bias[l], n_embd, model->config.norm_type);
            if (g_forward_timing_enabled) t_norm_us += vx_time_now_us() - t_stage;

            t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
            apply_gemv(model->attn_q[l], x, q, n_embd, n_embd, model->dequant_tmp);
            if (model->attn_q_bias[l] && model->attn_q_bias[l]->type == VX_TYPE_F32) {
                float *b = (float *)model->attn_q_bias[l]->data;
                for (int i = 0; i < n_embd; i++) q[i] += b[i];
            }
            apply_gemv(model->attn_k[l], x, k, kv_dim, n_embd, model->dequant_tmp);
            if (model->attn_k_bias[l] && model->attn_k_bias[l]->type == VX_TYPE_F32) {
                float *b = (float *)model->attn_k_bias[l]->data;
                for (int i = 0; i < kv_dim; i++) k[i] += b[i];
            }
            apply_gemv(model->attn_v[l], x, v, kv_dim, n_embd, model->dequant_tmp);
            if (model->attn_v_bias[l] && model->attn_v_bias[l]->type == VX_TYPE_F32) {
                float *b = (float *)model->attn_v_bias[l]->data;
                for (int i = 0; i < kv_dim; i++) v[i] += b[i];
            }
            if (g_forward_timing_enabled) t_qkv_us += vx_time_now_us() - t_stage;

            t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
            vx_rope(q, k, cache_pos, n_heads, n_kv_heads, head_dim, model->config.rope_partial_dims, model->config.rope_theta);
            apply_attention(model, attn_out, q, k, v, attn_out, n_heads, n_kv_heads, head_dim, l);

            if (model->attn_o[l])
                apply_gemv(model->attn_o[l], attn_out, attn_out, n_embd, n_embd, model->dequant_tmp);

            vx_add_f32(x, x_orig, attn_out, n_embd);
            memcpy(x_orig, x, n_embd * sizeof(float));
            if (g_forward_timing_enabled) t_attn_us += vx_time_now_us() - t_stage;

            t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
            apply_norm(x, norm_tmp, model->ffn_norm[l], model->ffn_norm_bias[l], n_embd, model->config.norm_type);

            if (skip_ffn) {
                memcpy(x, x_orig, n_embd * sizeof(float));
            } else if (ffn_h && n_ff > 0) {
                if (model->config.mlp_type == VX_MLP_CLASSIC && model->ffn_gate[l] == NULL) {
                    memset(ffn_h, 0, n_ff * sizeof(float));
                    apply_gemv(model->ffn_up[l], x, ffn_h, n_ff, n_embd, model->dequant_tmp);
                    if (model->ffn_up_bias[l] && model->ffn_up_bias[l]->type == VX_TYPE_F32) {
                        float *b = (float *)model->ffn_up_bias[l]->data;
                        for (int i = 0; i < n_ff; i++) ffn_h[i] += b[i];
                    }
                    vx_gelu(ffn_h, n_ff);
                    memset(attn_out, 0, n_embd * sizeof(float));
                    apply_gemv(model->ffn_down[l], ffn_h, attn_out, n_embd, n_ff, model->dequant_tmp);
                    if (model->ffn_down_bias[l] && model->ffn_down_bias[l]->type == VX_TYPE_F32) {
                        float *b = (float *)model->ffn_down_bias[l]->data;
                        for (int i = 0; i < n_embd; i++) attn_out[i] += b[i];
                    }
                } else if (model->config.mlp_type == VX_MLP_GEGLU) {
                    memset(ffn_h, 0, n_ff * sizeof(float));
                    memset(ffn_h2, 0, n_ff * sizeof(float));
                    apply_gemv(model->ffn_gate[l], x, ffn_h, n_ff, n_embd, model->dequant_tmp);
                    vx_gelu(ffn_h, n_ff);
                    apply_gemv_sparse_rows(model->ffn_up[l], x, ffn_h2, n_ff, n_embd, ffn_h, 0.01f, model->dequant_tmp);
                    vx_mul_f32(ffn_h, ffn_h, ffn_h2, n_ff);
                    memset(attn_out, 0, n_embd * sizeof(float));
                    apply_gemv(model->ffn_down[l], ffn_h, attn_out, n_embd, n_ff, model->dequant_tmp);
                } else {
                    memset(ffn_h, 0, n_ff * sizeof(float));
                    memset(ffn_h2, 0, n_ff * sizeof(float));
                    apply_gemv(model->ffn_gate[l], x, ffn_h, n_ff, n_embd, model->dequant_tmp);
                    vx_silu(ffn_h, n_ff);
                    apply_gemv_sparse_rows(model->ffn_up[l], x, ffn_h2, n_ff, n_embd, ffn_h, 0.01f, model->dequant_tmp);
                    vx_mul_f32(ffn_h, ffn_h, ffn_h2, n_ff);
                    memset(attn_out, 0, n_embd * sizeof(float));
                    apply_gemv(model->ffn_down[l], ffn_h, attn_out, n_embd, n_ff, model->dequant_tmp);
                }
                vx_add_f32(x, x_orig, attn_out, n_embd);
            } else {
                memcpy(x, x_orig, n_embd * sizeof(float));
            }
            if (g_forward_timing_enabled) t_ffn_us += vx_time_now_us() - t_stage;
        }

        model->cache_len++;

        if (pos == n_tokens - 1) {
            t_stage = g_forward_timing_enabled ? vx_time_now_us() : 0.0;
            if (norm_final_w) {
                apply_norm(x, norm_tmp, norm_final_w, model->norm_bias_w, n_embd, model->config.norm_type);
            }

            for (int i = 0; i < n_vocab; i++) logits[i] = -1e30f;
            if (output_weight && output_weight->data) {
                int out_rows = (int)output_weight->ne[1];
                int out_cols = output_weight->ne[0] > 0 ? (int)output_weight->ne[0] : n_embd;
                if (g_compute_level == VX_COMPUTE_ATTN_ONLY && out_rows > 8192) out_rows = 8192;
                if (g_compute_level == VX_COMPUTE_SKIP && out_rows > 1024) out_rows = 1024;
                if (out_rows > 0 && out_rows <= n_vocab * 2) {
                    apply_gemv(output_weight, x, logits, out_rows, out_cols, model->dequant_tmp);
                }
            }
            if (g_forward_timing_enabled) t_logits_us += vx_time_now_us() - t_stage;
        }
    }

    if (g_forward_timing_enabled) {
        g_forward_timing.embed_us += t_embed_us;
        g_forward_timing.norm_us += t_norm_us;
        g_forward_timing.qkv_us += t_qkv_us;
        g_forward_timing.attn_us += t_attn_us;
        g_forward_timing.ffn_us += t_ffn_us;
        g_forward_timing.logits_us += t_logits_us;
    }

    return VX_OK;
}
