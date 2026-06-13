#include "model.h"
#include "tensor.h"
#include "format.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Veltrix ONNX Loader Test\n");
    printf("=======================\n\n");

    // Synthetic ONNX model must exist from gen_onnx.py
    const char *path = "tests/test_synthetic.onnx";

    vx_model *model = NULL;
    vx_error err = vx_model_create(path, &model);
    if (err != VX_OK) {
        printf("FAIL: vx_model_create: %d\n", err);
        return 1;
    }

    printf("Model loaded:\n");
    printf("  arch:     %d\n", model->config.arch);
    printf("  layers:   %d (expected 2)\n", model->config.n_layers);
    printf("  embd:     %d (expected 64)\n", model->config.n_embd);
    printf("  ff:       %d (expected 128)\n", model->config.n_ff);
    printf("  vocab:    %d (expected 100)\n", model->config.n_vocab);
    printf("  heads:    %d (expected 4)\n", model->config.n_heads);
    printf("  kv_heads: %d (expected 1)\n", model->config.n_kv_heads);
    printf("  head_dim: %d (expected 16)\n", model->config.n_head_dim);
    printf("  tensors:  %d (expected 17)\n", model->n_tensors);

    int ok = 1;
    if (model->config.n_layers != 2) { printf("  FAIL: n_layers\n"); ok = 0; }
    if (model->config.n_embd != 64) { printf("  FAIL: n_embd\n"); ok = 0; }
    if (model->config.n_ff != 128) { printf("  FAIL: n_ff\n"); ok = 0; }
    if (model->config.n_vocab != 100) { printf("  FAIL: n_vocab\n"); ok = 0; }
    if (model->config.n_heads != 4) { printf("  FAIL: n_heads\n"); ok = 0; }
    if (model->config.n_kv_heads != 1) { printf("  FAIL: n_kv_heads\n"); ok = 0; }
    if (model->config.n_head_dim != 16) { printf("  FAIL: n_head_dim\n"); ok = 0; }
    if (model->n_tensors != 17) { printf("  FAIL: n_tensors\n"); ok = 0; }

    // Check that resolved weights are non-null
    if (!model->tok_embd) { printf("  FAIL: tok_embd is null\n"); ok = 0; }
    if (!model->output_w) { printf("  FAIL: output_w is null\n"); ok = 0; }
    if (!model->norm_w) { printf("  FAIL: norm_w is null\n"); ok = 0; }
    if (!model->attn_q[0]) { printf("  FAIL: attn_q[0] is null\n"); ok = 0; }
    if (!model->attn_k[0]) { printf("  FAIL: attn_k[0] is null\n"); ok = 0; }
    if (!model->attn_v[0]) { printf("  FAIL: attn_v[0] is null\n"); ok = 0; }
    if (!model->attn_o[0]) { printf("  FAIL: attn_o[0] is null\n"); ok = 0; }
    if (!model->ffn_gate[0]) { printf("  FAIL: ffn_gate[0] is null\n"); ok = 0; }
    if (!model->ffn_up[0]) { printf("  FAIL: ffn_up[0] is null\n"); ok = 0; }
    if (!model->ffn_down[0]) { printf("  FAIL: ffn_down[0] is null\n"); ok = 0; }
    // attn_norm and ffn_norm are optional (synthetic model omits them for simplicity)

    // Forward pass
    int token = 5;
    float *logits = malloc((size_t)model->config.n_vocab * sizeof(float));
    if (!logits) { printf("FAIL: malloc logits\n"); vx_model_destroy(model); return 1; }

    err = vx_model_forward(model, &token, 1, logits);
    if (err != VX_OK) {
        printf("FAIL: forward: %d\n", err);
        free(logits); vx_model_destroy(model);
        return 1;
    }

    float max_l = 0;
    for (int i = 0; i < model->config.n_vocab; i++) {
        if (fabsf(logits[i]) > max_l) max_l = fabsf(logits[i]);
    }
    printf("  Max logit: %.4f (should be > 0)\n", max_l);
    if (max_l <= 0) { printf("  FAIL: logits zero\n"); ok = 0; }
    else printf("  PASS: logits non-zero\n");

    // Benchmark
    int n_iters = 10;
    double t0 = vx_time_now_us();
    for (int i = 0; i < n_iters; i++) {
        token = i % model->config.n_vocab;
        err = vx_model_forward(model, &token, 1, logits);
        if (err != VX_OK) { printf("FAIL: forward iter %d: %d\n", i, err); break; }
    }
    double total_us = vx_time_now_us() - t0;
    double avg_us = total_us / n_iters;
    printf("  Forward avg: %.1f us, %.0f tok/s\n", avg_us, 1e6 / avg_us);

    free(logits);
    vx_model_destroy(model);

    printf("\n%s\n", ok ? "PASS: ONNX loader" : "FAIL: ONNX loader");
    return ok ? 0 : 1;
}
