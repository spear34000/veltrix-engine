#include "runtime_guard.h"
#include <string.h>

static float vx_guard_average(const vx_runtime_guard *guard) {
    if (!guard || guard->count <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < guard->count; i++) sum += guard->recent_tok_s[i];
    return sum / (float)guard->count;
}

vx_runtime_guard vx_runtime_guard_create(float target_tok_s, float quality_floor) {
    vx_runtime_guard guard;
    memset(&guard, 0, sizeof(guard));
    guard.target_tok_s = target_tok_s > 0.0f ? target_tok_s : 15.0f;
    guard.quality_floor = quality_floor > 0.0f ? quality_floor : 0.85f;
    guard.recommended_compute_level = VX_COMPUTE_EXACT;
    return guard;
}

void vx_runtime_guard_tick(vx_runtime_guard *guard, float tok_s) {
    if (!guard) return;

    guard->recent_tok_s[guard->idx] = tok_s;
    guard->idx = (guard->idx + 1) % 8;
    if (guard->count < 8) guard->count++;

    float avg = vx_guard_average(guard);
    if (avg <= 0.0f) avg = tok_s;

    if (tok_s < guard->target_tok_s || avg < guard->target_tok_s) {
        guard->throughput_floor_breached = true;
        guard->recommended_compute_level = (avg < guard->target_tok_s * 0.75f)
                                               ? VX_COMPUTE_SKIP
                                               : VX_COMPUTE_ATTN_ONLY;
    } else if (avg < guard->target_tok_s * 1.08f) {
        guard->recommended_compute_level = VX_COMPUTE_ATTN_ONLY;
    } else {
        guard->recommended_compute_level = VX_COMPUTE_EXACT;
    }

    if (guard->quality_floor < 0.85f && guard->recommended_compute_level == VX_COMPUTE_EXACT) {
        guard->recommended_compute_level = VX_COMPUTE_ATTN_ONLY;
    }
}
