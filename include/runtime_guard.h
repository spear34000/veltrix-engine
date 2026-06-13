#ifndef VX_RUNTIME_GUARD_H
#define VX_RUNTIME_GUARD_H

#include "runtime_profile.h"

typedef struct {
    float target_tok_s;
    float quality_floor;
    float recent_tok_s[8];
    int idx;
    int count;
    bool throughput_floor_breached;
    vx_compute_level recommended_compute_level;
} vx_runtime_guard;

vx_runtime_guard vx_runtime_guard_create(float target_tok_s, float quality_floor);
void vx_runtime_guard_tick(vx_runtime_guard *guard, float tok_s);

#endif
