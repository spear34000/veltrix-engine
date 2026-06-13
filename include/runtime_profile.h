#ifndef VX_RUNTIME_PROFILE_H
#define VX_RUNTIME_PROFILE_H

#include "veltrix.h"

typedef struct {
    char name[32];
    int cpu_cores;
    int sustained_threads;
    int ram_mb;
    bool is_arm;
    int thermal_budget;
} vx_device_profile;

typedef struct {
    float target_tok_s;
    float quality_floor;
    int ctx_limit;
} vx_runtime_goal;

typedef struct {
    char name[32];
    bool allow_run;
    float expected_tok_s;
    float quality_floor;
    int n_threads;
    int ctx_cap;
    vx_compute_level compute_level;
} vx_runtime_profile;

vx_device_profile vx_device_profile_auto(void);
vx_runtime_profile vx_choose_runtime_profile(const vx_model_config *cfg,
                                             const vx_device_profile *device,
                                             const vx_runtime_goal *goal);
void vx_apply_runtime_profile(vx_model *model, const vx_runtime_profile *profile);
const vx_runtime_profile *vx_runtime_profile_current(void);

#endif
