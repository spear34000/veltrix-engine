#include "gguf.h"
#include "model.h"
#include "tensor.h"
#include "quantize.h"
#include "platform.h"
#include "runtime_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Calculate number of Q4_0 blocks needed for n elements
static int q4_blocks(int n) {
    return (n + 31) / 32;
}

// Calculate byte size for Q4_0 tensor with n elements
static size_t q4_size(int n) {
    return (size_t)q4_blocks(n) * sizeof(block_q4_0);
}

typedef struct {
    int cols, rows;
    const char *name;
} tensor_spec;

static uint8_t *alloc_gguf(int n_layers, int n_embd, int n_ff, int n_vocab,
                           int n_heads, int n_kv_heads, size_t *out_size) {
    int n_tensors = 7 * n_layers + 3; // 7 per layer + tok_embd + output_norm + output
    int n_kv = 10;
    int kv_dim = n_kv_heads * (n_embd / n_heads);

    // Build tensor specs and compute data offsets
    tensor_spec *specs = calloc((size_t)n_tensors, sizeof(tensor_spec));
    int idx = 0;
    for (int l = 0; l < n_layers; l++) {
        char *name;
        name = malloc(64); snprintf(name, 64, "blk.%d.attn_q.weight", l);
        specs[idx++] = (tensor_spec){n_embd, n_embd, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.attn_k.weight", l);
        specs[idx++] = (tensor_spec){n_embd, kv_dim, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.attn_v.weight", l);
        specs[idx++] = (tensor_spec){n_embd, kv_dim, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.attn_output.weight", l);
        specs[idx++] = (tensor_spec){n_embd, n_embd, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.ffn_gate.weight", l);
        specs[idx++] = (tensor_spec){n_embd, n_ff, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.ffn_up.weight", l);
        specs[idx++] = (tensor_spec){n_embd, n_ff, name};
        name = malloc(64); snprintf(name, 64, "blk.%d.ffn_down.weight", l);
        specs[idx++] = (tensor_spec){n_ff, n_embd, name};
    }
    specs[idx++] = (tensor_spec){n_embd, n_vocab, strdup("token_embd.weight")};
    specs[idx++] = (tensor_spec){1, n_embd, strdup("output_norm.weight")}; // 1D tensor
    specs[idx++] = (tensor_spec){n_embd, n_vocab, strdup("output.weight")};

    // Compute total header + KV + tensor info size
    size_t sz = 21; // header
    // KV space (estimate generously)
    sz += (size_t)n_kv * 200;
    // Tensor info space
    for (int i = 0; i < n_tensors; i++) {
        sz += 8 + strlen(specs[i].name) + 4 + 16 + 4 + 8; // name + ndims + dims + type + offset
    }
    sz += 1024; // padding

    // Compute tensor data offsets (aligned after tensor infos)
    uint64_t *offsets = calloc((size_t)n_tensors, sizeof(uint64_t));
    uint64_t data_off = (uint64_t)sz;
    data_off = (data_off + 31) & ~(uint64_t)31; // align to 32
    for (int i = 0; i < n_tensors; i++) {
        offsets[i] = data_off;
        int ne = specs[i].rows * specs[i].cols;
        // For 1D tensor (output_norm), ne = n_embd (specs[i].cols = 1, specs[i].rows = n_embd)
        if (specs[i].cols == 1) ne = specs[i].rows;
        data_off += q4_size(ne);
    }
    sz = (size_t)data_off;
    sz = (sz + 31) & ~(size_t)31;

    // Allocate buffer
    uint8_t *buf = calloc(1, sz);
    if (!buf) { for (int i=0; i<n_tensors; i++) free((void*)specs[i].name); free(specs); free(offsets); return NULL; }

    uint8_t *p = buf;
    #define WU32(v) do { uint32_t _v = (v); memcpy(p, &_v, 4); p += 4; } while(0)
    #define WU64(v) do { uint64_t _v = (v); memcpy(p, &_v, 8); p += 8; } while(0)
    #define WF32(v) do { float _v = (v); memcpy(p, &_v, 4); p += 4; } while(0)
    #define WTYPE(t) do { uint32_t _t = (t); memcpy(p, &_t, 4); p += 4; } while(0)
    #define WSTR(s) do { \
        uint64_t _len = strlen(s); \
        memcpy(p, &_len, 8); p += 8; \
        memcpy(p, s, _len); p += _len; \
    } while(0)

    // Header
    uint32_t magic = GGUF_MAGIC;
    uint32_t ver = GGUF_VERSION;
    memcpy(p, &magic, 4); p += 4;
    memcpy(p, &ver, 4); p += 4;
    WU64((uint64_t)n_tensors);
    WU64((uint64_t)n_kv);

    // KV metadata
    WSTR("general.architecture"); WTYPE(GGUF_TYPE_STR); WSTR("test");
    WSTR("llama.block_count"); WTYPE(GGUF_TYPE_U32); WU32(n_layers);
    WSTR("llama.attention.head_count"); WTYPE(GGUF_TYPE_U32); WU32(n_heads);
    WSTR("llama.attention.head_count_kv"); WTYPE(GGUF_TYPE_U32); WU32(n_kv_heads);
    WSTR("llama.embedding_length"); WTYPE(GGUF_TYPE_U32); WU32(n_embd);
    WSTR("llama.feedforward_length"); WTYPE(GGUF_TYPE_U32); WU32(n_ff);
    WSTR("llama.rope.freq_base"); WTYPE(GGUF_TYPE_F32); WF32(10000.0f);
    WSTR("llama.vocab_size"); WTYPE(GGUF_TYPE_U32); WU32(n_vocab);
    WSTR("test.vocab_size"); WTYPE(GGUF_TYPE_U32); WU32(n_vocab);
    WSTR("general.file_type"); WTYPE(GGUF_TYPE_U32); WU32(2); // Q4_0

    // Tensor infos
    for (int i = 0; i < n_tensors; i++) {
        WSTR(specs[i].name);
        if (specs[i].cols == 1 && specs[i].rows == n_embd) {
            WU32(1); WU64((uint64_t)specs[i].rows);
        } else {
            WU32(2);
            WU64((uint64_t)specs[i].cols);
            WU64((uint64_t)specs[i].rows);
        }
        WU32(GGML_TYPE_Q4_0);
        WU64(offsets[i]);
    }

    // Fill tensor data with deterministic pattern
    for (int i = 0; i < n_tensors; i++) {
        int ne = specs[i].rows * specs[i].cols;
        if (specs[i].cols == 1) ne = specs[i].rows;
        int nq = q4_blocks(ne);
        block_q4_0 *b = (block_q4_0 *)(buf + offsets[i]);
        for (int j = 0; j < nq; j++) {
            b[j].d = vx_fp32_to_fp16(0.01f);
            for (int k = 0; k < 16; k++) {
                b[j].qs[k] = (uint8_t)((j * 16 + k) % 16);
            }
        }
    }

    *out_size = sz;

    for (int i = 0; i < n_tensors; i++) free((void*)specs[i].name);
    free(specs);
    free(offsets);

    return buf;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Veltrix End-to-End Test (Synthetic GGUF)\n");
    printf("========================================\n\n");

    int n_layers = 2, n_embd = 64, n_ff = 128, n_vocab = 100;
    int n_heads = 4, n_kv_heads = 1;

    size_t fsize = 0;
    uint8_t *gguf_buf = alloc_gguf(n_layers, n_embd, n_ff, n_vocab,
                                   n_heads, n_kv_heads, &fsize);
    if (!gguf_buf) { printf("FAIL: alloc_gguf\n"); return 1; }

    const char *path = "test_synthetic.gguf";
    FILE *fp = fopen(path, "wb");
    if (!fp) { free(gguf_buf); printf("FAIL: cannot write %s\n", path); return 1; }
    fwrite(gguf_buf, 1, fsize, fp);
    fclose(fp);
    printf("Synthetic GGUF: %zu bytes, %d layers, %d embd, %d ff, %d vocab\n",
           fsize, n_layers, n_embd, n_ff, n_vocab);
    free(gguf_buf);

    vx_model *model = NULL;
    vx_error err = vx_model_create(path, &model);
    if (err != VX_OK) { printf("FAIL: vx_model_create: %d\n", err); remove(path); return 1; }
    fflush(stdout);
    printf("Model loaded: %d layers, %d embd, %d ff, %d vocab, %d heads, %d kv_heads\n",
           model->config.n_layers, model->config.n_embd, model->config.n_ff,
           model->config.n_vocab, model->config.n_heads, model->config.n_kv_heads);
    printf("Tensors loaded: %d\n", model->n_tensors);

    vx_runtime_goal goal = {
        .target_tok_s = 15.0f,
        .quality_floor = 0.85f,
        .ctx_limit = 256,
    };
    vx_device_profile device = vx_device_profile_auto();
    vx_runtime_profile profile = vx_choose_runtime_profile(&model->config, &device, &goal);
    vx_apply_runtime_profile(model, &profile);
    printf("Runtime profile: %s (%s)\n", profile.name, profile.allow_run ? "allowed" : "blocked");
    if (!profile.allow_run || !(model->config.n_ctx <= 256 && model->config.n_threads > 0)) {
        printf("FAIL: runtime profile application\n");
        vx_model_destroy(model);
        remove(path);
        return 1;
    }

    int token = 5;
    float *logits = malloc((size_t)n_vocab * sizeof(float));
    if (!logits) { vx_model_destroy(model); remove(path); return 1; }

    err = vx_model_forward(model, &token, 1, logits);
    if (err != VX_OK) { printf("FAIL: forward (warmup): %d\n", err); free(logits); vx_model_destroy(model); remove(path); return 1; }

    int n_iters = 10;
    double t0 = vx_time_now_us();
    for (int i = 0; i < n_iters; i++) {
        token = i % n_vocab;
        err = vx_model_forward(model, &token, 1, logits);
        if (err != VX_OK) { printf("FAIL: forward iter %d: %d\n", i, err); break; }
    }
    double total_us = vx_time_now_us() - t0;
    double avg_us = total_us / n_iters;
    double tok_s = 1e6 / avg_us;

    printf("\nForward pass benchmark (%d iters):\n", n_iters);
    printf("  Total: %.0f us\n", total_us);
    printf("  Avg:   %.1f us\n", avg_us);
    printf("  tok/s: %.2f\n", tok_s);
    fflush(stdout);

    float max_l = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (fabsf(logits[i]) > max_l) max_l = fabsf(logits[i]);
    }
    printf("  Max logit: %.4f (should be > 0)\n", max_l);
    fflush(stdout);

    free(logits);
    vx_model_destroy(model);
    remove(path);

    printf("\n%s\n", max_l > 0 ? "PASS: end-to-end" : "FAIL: logits zero");
    fflush(stdout);
    return max_l > 0 ? 0 : 1;
}
