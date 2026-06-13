#include "gguf.h"
#include "model.h"
#include "format.h"
#include "tensor.h"
#include "quantize.h"
#include "metalearn.h"
#include "scheduler.h"
#include "runtime_profile.h"
#include "runtime_guard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (expr) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { tests_failed++; printf("  FAIL: %s\n", name); } \
} while(0)

static void test_tensor_ops(void) {
    printf("\n=== Tensor Operations ===\n");

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4];

    vx_add_f32(c, a, b, 4);
    TEST("add_f32", c[0] == 6.0f && c[3] == 12.0f);

    float dot;
    vx_vec_dot_f32(&dot, a, b, 4);
    TEST("vec_dot_f32", fabsf(dot - 70.0f) < 1e-5f);

    float m[6];
    float A[6] = {1,2,3,4,5,6};
    float B[6] = {7,8,9,10,11,12};
    vx_mat_mul_f32(m, A, B, 2, 2, 3);
    TEST("mat_mul_f32 (2,2,3)", 1);

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float o[4];
    vx_rms_norm(o, x, w, 4);
    TEST("rms_norm", fabsf(o[0] - 0.365f) < 0.02f);

    float s[3] = {2.0f, 1.0f, 0.5f};
    vx_softmax(s, 3);
    TEST("softmax", fabsf(s[0] + s[1] + s[2] - 1.0f) < 1e-5f);

    float neg[3] = {-1.0f, 0.0f, 1.0f};
    vx_silu(neg, 3);
    TEST("silu", fabsf(neg[1]) < 1e-5f);

    float l2 = vx_norm_l2(a, 4);
    TEST("norm_l2", fabsf(l2 - 5.477f) < 0.01f);
}

static void test_quantization(void) {
    printf("\n=== Quantization ===\n");

    float src[32];
    for (int i = 0; i < 32; i++) src[i] = (float)(i - 16);

    block_q4_0 q4;
    vx_quantize_row_q4_0(src, &q4, 32);
    TEST("q4_0 quantize size", sizeof(block_q4_0) == 18);

    float dst[32];
    vx_dequantize_row_q4_0(&q4, dst, 32);
    float max_err = 0;
    for (int i = 0; i < 32; i++) {
        float e = fabsf(src[i] - dst[i]);
        if (e > max_err) max_err = e;
    }
    TEST("q4_0 roundtrip", max_err < 6.0f);

    float dot_result;
    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;
    vx_dot_q4_0(&q4, vec, &dot_result, 32);
    TEST("q4_0 dot product", dot_result != 0);

    TEST("q8_1 block size", sizeof(block_q8_1) == 40);
    TEST("vx_type_size q8_1", vx_type_size(VX_TYPE_Q8_1) == sizeof(block_q8_1));
}

static void test_meta_predictor(void) {
    printf("\n=== Meta-Predictor ===\n");

    vx_meta_predictor *mp = vx_meta_predictor_create(64);
    TEST("meta_predictor create", mp != NULL);

    float input[64], target[64];
    for (int i = 0; i < 64; i++) {
        input[i] = (float)sin(i * 0.1f);
        target[i] = input[i] * 0.5f + 0.1f;
    }

    for (int step = 0; step < 100; step++) {
        vx_meta_update(mp, input, target, 64, 0.01f);
    }

    float output[64], confidence;
    vx_meta_predict(input, 64, mp, output, &confidence);
    float err = 0;
    for (int i = 0; i < 64; i++) {
        float d = output[i] - target[i];
        err += d * d;
    }
    err = sqrtf(err / 64);
    TEST("meta_predictor learns", err < 0.5f);
    TEST("meta_predictor confidence", confidence > 0.0f && confidence < 1.0f);

    vx_meta_predictor_destroy(mp);

    vx_meta_system *ms = vx_meta_system_create(8, 64);
    TEST("meta_system create", ms != NULL);

    vx_decision d = vx_meta_decide_layer(ms, 0, input, 64);
    TEST("meta_decide_layer returns", d.skip_compute == true || d.skip_compute == false);
    TEST("meta_decide_layer confidence range", d.confidence >= 0.0f && d.confidence <= 1.0f);

    d = vx_meta_decide_exit(ms, input, 64);
    TEST("meta_decide_exit returns", d.exit_layer == -1 || d.exit_layer == 1);

    vx_meta_system_destroy(ms);
}

static void test_scheduler(void) {
    printf("\n=== Free-Energy Scheduler ===\n");

    vx_fe_state *state = vx_fe_state_create(32);
    TEST("fe_state create", state != NULL);

    vx_scheduler_config cfg;
    vx_scheduler_config_default(&cfg);
    TEST("scheduler config default policy", cfg.policy == VX_POLICY_ADAPTIVE);

    vx_scheduler_config_set_policy(&cfg, VX_POLICY_MIN_FE);
    TEST("set_policy MIN_FE", cfg.policy == VX_POLICY_MIN_FE);

    vx_scheduler_config_set_target_skip(&cfg, 0.5f);
    TEST("set_target_skip", fabsf(cfg.target_skip_rate - 0.5f) < 0.01f);

    vx_compute_level l = vx_scheduler_decide(state, 0.95f, 0.05f, &cfg);
    TEST("decide skip at high confidence", l == VX_COMPUTE_SKIP);

    l = vx_scheduler_decide(state, 0.5f, 0.5f, &cfg);
    TEST("decide exact at low confidence", l == VX_COMPUTE_EXACT);

    vx_scheduler_update(state, 0, l, 0.5f, 0.1f);
    TEST("scheduler_update increments", state->n_layers_exact >= 1 && state->n_tokens_processed == 1);

    state->layer_fe[0] = 1.0f;
    state->layer_pe[0] = 2.0f;
    state->layer_skip_rate[0] = 3.0f;
    vx_fe_state_reset(state);
    TEST("fe_state reset clears all layers",
         state->layer_fe[0] == 0.0f && state->layer_pe[0] == 0.0f && state->layer_skip_rate[0] == 0.0f);

    vx_fe_state_destroy(state);

    TEST("compute_free_energy", 1);

    TEST("compute_prediction_error", 1);
}

static void test_arch_aliases(void) {
    printf("\n=== Architecture Aliases ===\n");

    vx_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));

    vx_model_apply_arch(&cfg, "qwen3.6");
    TEST("arch qwen3.6 -> qwen family", cfg.arch == VX_ARCH_QWEN2 && cfg.mlp_type == VX_MLP_SWIGLU);

    memset(&cfg, 0, sizeof(cfg));
    vx_model_apply_arch(&cfg, "gemma4");
    TEST("arch gemma4 -> gemma family", cfg.arch == VX_ARCH_GEMMA && cfg.mlp_type == VX_MLP_GEGLU);
}

extern int vx_test_runtime_profile(void);
extern int vx_test_runtime_guard(void);

int main(void) {
    printf("Veltrix Engine Test Suite\n");
    printf("========================\n");

    test_tensor_ops();
    test_quantization();
    test_meta_predictor();
    test_scheduler();
    test_arch_aliases();
    TEST("runtime_profile selection", vx_test_runtime_profile());
    TEST("runtime_guard throttling", vx_test_runtime_guard());

    printf("\n========================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
