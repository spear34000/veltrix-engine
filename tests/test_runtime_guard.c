#include "runtime_guard.h"

int vx_test_runtime_guard(void) {
    vx_runtime_guard guard = vx_runtime_guard_create(15.0f, 0.85f);
    vx_runtime_guard_tick(&guard, 18.2f);
    vx_runtime_guard_tick(&guard, 14.1f);
    vx_runtime_guard_tick(&guard, 13.8f);
    return guard.throughput_floor_breached && guard.recommended_compute_level != VX_COMPUTE_EXACT;
}
