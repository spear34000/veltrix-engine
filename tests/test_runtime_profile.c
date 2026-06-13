#include "runtime_profile.h"
#include <stdio.h>
#include <string.h>

int vx_test_runtime_profile(void) {
    vx_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers = 24;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.n_embd = 2048;
    cfg.n_head_dim = 64;
    cfg.n_ff = 5632;
    cfg.n_vocab = 32000;
    cfg.n_ctx = 4096;

    vx_device_profile device;
    memset(&device, 0, sizeof(device));
    snprintf(device.name, sizeof(device.name), "%s", "n100");
    device.cpu_cores = 4;
    device.sustained_threads = 4;
    device.ram_mb = 4096;
    device.is_arm = false;
    device.thermal_budget = 70;

    vx_runtime_goal goal;
    goal.target_tok_s = 15.0f;
    goal.quality_floor = 0.85f;
    goal.ctx_limit = 256;

    vx_runtime_profile profile = vx_choose_runtime_profile(&cfg, &device, &goal);
    return profile.allow_run && strcmp(profile.name, "mobile-low") == 0 &&
           profile.ctx_cap <= 256 && profile.n_threads <= 4 &&
           profile.expected_tok_s >= 15.0f && profile.quality_floor >= 0.85f;
}
