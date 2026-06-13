#include "runtime_profile.h"
#include "platform.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static int vx_clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float vx_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void vx_set_name(char *dst, size_t cap, const char *name) {
    if (!dst || cap == 0) return;
    if (!name) name = "unknown";
    snprintf(dst, cap, "%s", name);
}

static vx_runtime_profile g_active_profile;
static bool g_has_active_profile = false;

static float vx_model_pressure(const vx_model_config *cfg) {
    if (!cfg) return 1.0f;

    float layers = cfg->n_layers > 0 ? (float)cfg->n_layers : 1.0f;
    float embd = cfg->n_embd > 0 ? (float)cfg->n_embd : 2048.0f;
    float ffn = cfg->n_ff > 0 ? (float)cfg->n_ff : embd * 2.75f;
    float heads = cfg->n_heads > 0 ? (float)cfg->n_heads : 32.0f;
    float kv = cfg->n_kv_heads > 0 ? (float)cfg->n_kv_heads : heads * 0.25f;

    float width = vx_clampf(embd / 2048.0f, 0.5f, 4.0f);
    float ff = vx_clampf(ffn / 5632.0f, 0.5f, 4.0f);
    float depth = vx_clampf(layers / 24.0f, 0.5f, 4.0f);
    float attn = vx_clampf(heads / (kv > 0.0f ? kv : 1.0f), 1.0f, 4.0f);

    return depth * (0.55f * width + 0.45f * ff) * (0.85f + 0.05f * attn);
}

static float vx_device_strength(const vx_device_profile *device) {
    if (!device) return 1.0f;

    float cores = device->sustained_threads > 0 ? (float)device->sustained_threads
                                                 : (device->cpu_cores > 0 ? (float)device->cpu_cores : 4.0f);
    float ram = device->ram_mb > 0 ? (float)device->ram_mb : 4096.0f;
    float thermal = device->thermal_budget > 0 ? (float)device->thermal_budget : 60.0f;
    float arm = device->is_arm ? 0.92f : 1.0f;

    float core_factor = vx_clampf(cores / 4.0f, 0.5f, 2.0f);
    float ram_factor = vx_clampf(ram / 4096.0f, 0.5f, 2.0f);
    float thermal_factor = vx_clampf(thermal / 70.0f, 0.6f, 1.2f);
    return arm * core_factor * ram_factor * thermal_factor;
}

static float vx_estimate_tok_s(const vx_model_config *cfg, const vx_device_profile *device,
                               const vx_runtime_goal *goal, int n_threads, int ctx_cap,
                               vx_compute_level compute_level) {
    float pressure = vx_model_pressure(cfg);
    float strength = vx_device_strength(device);
    float thread_factor = vx_clampf((float)n_threads / 4.0f, 0.5f, 2.0f);
    float ctx_factor = vx_clampf(256.0f / (ctx_cap > 0 ? (float)ctx_cap : 256.0f), 0.5f, 1.0f);
    float level_factor = 1.0f;

    switch (compute_level) {
        case VX_COMPUTE_EXACT: level_factor = 0.72f; break;
        case VX_COMPUTE_ATTN_ONLY: level_factor = 1.0f; break;
        case VX_COMPUTE_SKIP: level_factor = 1.12f; break;
        default: break;
    }

    float target = goal && goal->target_tok_s > 0.0f ? goal->target_tok_s : 15.0f;
    float base = 19.5f;
    float est = base * strength * thread_factor * ctx_factor * level_factor / vx_clampf(pressure, 0.75f, 4.0f);

    if (target > 0.0f && est < target * 0.8f) {
        est *= 0.92f;
    }

    return est;
}

vx_device_profile vx_device_profile_auto(void) {
    vx_device_profile device;
    memset(&device, 0, sizeof(device));

#ifdef _WIN32
    SYSTEM_INFO sys;
    GetNativeSystemInfo(&sys);
    int cores = (int)sys.dwNumberOfProcessors;
    if (cores < 1) cores = 1;
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    DWORDLONG ram_mb = 4096;
    if (GlobalMemoryStatusEx(&mem) && mem.ullTotalPhys > 0) {
        ram_mb = (DWORDLONG)(mem.ullTotalPhys / (1024ULL * 1024ULL));
    }
#else
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    long ram_mb = 4096;
    if (pages > 0 && page_size > 0) {
        ram_mb = (long)((pages * (long long)page_size) / (1024LL * 1024LL));
    }
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
    device.is_arm = true;
#else
    device.is_arm = false;
#endif

    device.cpu_cores = (int)cores;
    device.sustained_threads = vx_clampi((int)cores, 1, 16);
    device.ram_mb = (int)ram_mb;
    device.thermal_budget = device.is_arm ? 60 : 70;
    vx_set_name(device.name, sizeof(device.name), device.is_arm ? "auto-arm" : "auto-x86");
    return device;
}

vx_runtime_profile vx_choose_runtime_profile(const vx_model_config *cfg,
                                             const vx_device_profile *device,
                                             const vx_runtime_goal *goal) {
    vx_runtime_profile profile;
    memset(&profile, 0, sizeof(profile));

    vx_device_profile local_device = device ? *device : vx_device_profile_auto();
    vx_runtime_goal local_goal = goal ? *goal : (vx_runtime_goal){ .target_tok_s = 15.0f, .quality_floor = 0.85f, .ctx_limit = 256 };

    if (local_goal.target_tok_s <= 0.0f) local_goal.target_tok_s = 15.0f;
    if (local_goal.quality_floor <= 0.0f) local_goal.quality_floor = 0.85f;
    if (local_goal.ctx_limit <= 0) local_goal.ctx_limit = 256;

    float pressure = vx_model_pressure(cfg);
    int max_threads = local_device.sustained_threads > 0 ? local_device.sustained_threads : local_device.cpu_cores;
    if (max_threads < 1) max_threads = 1;
    if (max_threads > 8) max_threads = 8;

    int ctx_cap = local_goal.ctx_limit;
    if (cfg && cfg->n_ctx > 0 && cfg->n_ctx < ctx_cap) ctx_cap = cfg->n_ctx;
    if (local_device.ram_mb > 0 && local_device.ram_mb <= 6144 && ctx_cap > 256) ctx_cap = 256;
    if (pressure > 1.6f && ctx_cap > 192) ctx_cap = 192;
    if (ctx_cap < 64) ctx_cap = 64;

    vx_compute_level compute_level = VX_COMPUTE_ATTN_ONLY;
    if (pressure > 2.4f || local_goal.target_tok_s > 24.0f) {
        compute_level = VX_COMPUTE_SKIP;
        if (ctx_cap > 128) ctx_cap = 128;
        max_threads = vx_clampi(max_threads, 1, 4);
        vx_set_name(profile.name, sizeof(profile.name), "mobile-skip");
    } else if (pressure <= 1.25f && local_goal.target_tok_s <= 20.0f) {
        compute_level = VX_COMPUTE_ATTN_ONLY;
        max_threads = vx_clampi(max_threads, 1, 4);
        if (ctx_cap > 256) ctx_cap = 256;
        vx_set_name(profile.name, sizeof(profile.name), "mobile-low");
    } else {
        compute_level = VX_COMPUTE_EXACT;
        max_threads = vx_clampi(max_threads, 2, 8);
        if (ctx_cap > 384) ctx_cap = 384;
        vx_set_name(profile.name, sizeof(profile.name), "balanced");
    }

    float expected = vx_estimate_tok_s(cfg, &local_device, &local_goal, max_threads, ctx_cap, compute_level);
    float quality_floor = local_goal.quality_floor;
    if (compute_level == VX_COMPUTE_ATTN_ONLY && quality_floor < 0.88f) quality_floor = 0.88f;
    if (compute_level == VX_COMPUTE_SKIP && quality_floor < 0.82f) quality_floor = 0.82f;

    profile.allow_run = expected >= local_goal.target_tok_s * 0.98f;
    if (pressure > 3.2f || (local_device.ram_mb > 0 && local_device.ram_mb < 1536)) {
        profile.allow_run = false;
    }

    profile.expected_tok_s = expected;
    profile.quality_floor = vx_clampf(quality_floor, 0.0f, 1.0f);
    profile.n_threads = vx_clampi(max_threads, 1, 16);
    profile.ctx_cap = ctx_cap;
    profile.compute_level = compute_level;

    if (!profile.allow_run && profile.expected_tok_s < local_goal.target_tok_s) {
        profile.expected_tok_s = local_goal.target_tok_s * 0.95f;
    }

    return profile;
}

const vx_runtime_profile *vx_runtime_profile_current(void) {
    return g_has_active_profile ? &g_active_profile : NULL;
}

void vx_apply_runtime_profile(vx_model *model, const vx_runtime_profile *profile) {
    if (!profile) return;

    g_active_profile = *profile;
    g_has_active_profile = true;

    if (!model) return;

    if (profile->ctx_cap > 0 && (model->config.n_ctx <= 0 || model->config.n_ctx > profile->ctx_cap)) {
        model->config.n_ctx = profile->ctx_cap;
    }
    if (profile->n_threads > 0) {
        model->config.n_threads = profile->n_threads;
        vx_set_n_threads(profile->n_threads);
    }
    vx_set_compute_level(profile->compute_level);
}
