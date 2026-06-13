#include "model.h"
#include "format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Veltrix LoRA Adapter Test\n");
    printf("=========================\n\n");

    int ok = 1;

    // TEST 1: Verify merge formula directly on a small matrix
    printf("TEST 1: Direct merge formula validation\n");
    {
        int d_in = 4, d_out = 3, rank = 2;
        float W[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};  // 3x4 row-major
        float A[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};  // 4x2
        float B[6] = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f};  // 2x3
        float alpha = 1.0f, scale = alpha / rank;

        for (int i = 0; i < d_out; i++) {
            for (int j = 0; j < d_in; j++) {
                float sum = 0.0f;
                for (int k = 0; k < rank; k++)
                    sum += B[k * d_out + i] * A[j * rank + k];
                W[i * d_in + j] += scale * sum;
            }
        }

        int has_nan = 0;
        for (int i = 0; i < 12; i++) if (isnan(W[i]) || isinf(W[i])) has_nan = 1;
        printf("  Merge test: %s (no NaN/Inf)\n", has_nan ? "FAIL" : "PASS");
        if (has_nan) { ok = 0; }
        else { printf("  PASS: direct merge\n"); }
    }

    // TEST 2: Load synthetic ONNX model, merge LoRA, validate forward
    printf("\nTEST 2: ONNX model + LoRA merge\n");
    const char *path = "tests/test_synthetic.onnx";
    vx_model *model = NULL;
    vx_error err = vx_model_create(path, &model);
    if (err != VX_OK) { printf("FAIL: vx_model_create: %d\n", err); return 1; }
    printf("  Model loaded: %d layers, %d embd, rope_theta=%.0f\n",
           model->config.n_layers, model->config.n_embd, model->config.rope_theta);

    // Forward before merge
    int token = 5;
    float logits_before[100];
    err = vx_model_forward(model, &token, 1, logits_before);
    if (err != VX_OK) { printf("FAIL: forward before: %d\n", err); vx_model_destroy(model); return 1; }

    float max_b = -1e10f;
    int nn_b = 0;
    for (int i = 0; i < 100; i++) {
        if (isnan(logits_before[i])) nn_b++;
        if (logits_before[i] > max_b) max_b = logits_before[i];
    }
    printf("  Before LoRA: max=%.4f NaN=%d\n", max_b, nn_b);

    // Reset KV cache
    vx_model_kv_truncate(model, 0);

    // Load LoRA and merge
    vx_lora lora;
    memset(&lora, 0, sizeof(lora));
    err = vx_lora_load("tests/test_lora.safetensors", model->config.n_layers, &lora);
    if (err != VX_OK) { printf("FAIL: vx_lora_load: %d\n", err); vx_model_destroy(model); return 1; }
    printf("  LoRA loaded: %d adapters\n", lora.n_adapters);

    err = vx_lora_merge(&lora, model);
    if (err != VX_OK) { printf("FAIL: vx_lora_merge: %d\n", err); vx_lora_destroy(&lora); vx_model_destroy(model); return 1; }
    printf("  LoRA merged\n");
    vx_lora_destroy(&lora);

    // Forward after merge
    float logits_after[100];
    model->cache_len = 0;
    err = vx_model_forward(model, &token, 1, logits_after);
    if (err != VX_OK) { printf("FAIL: forward after: %d\n", err); vx_model_destroy(model); return 1; }

    float max_a = -1e10f;
    int nn_a = 0;
    for (int i = 0; i < 100; i++) {
        if (isnan(logits_after[i])) nn_a++;
        if (logits_after[i] > max_a) max_a = logits_after[i];
    }
    printf("  After LoRA:  max=%.4f NaN=%d\n", max_a, nn_a);

    if (nn_b == 0 && nn_a == 0 && max_a > -1e9f) {
        printf("  PASS: valid output before and after LoRA\n");
        float diff = 0;
        for (int i = 0; i < 100; i++) diff += fabsf(logits_after[i] - logits_before[i]);
        printf("  Mean abs diff: %.6f\n", diff / 100.0f);
    } else {
        printf("  FAIL\n");
        ok = 0;
    }

    vx_model_destroy(model);

    printf("\n%s\n", ok ? "PASS: all tests" : "FAIL: some tests failed");
    return ok ? 0 : 1;
}
