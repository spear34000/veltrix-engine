# Target tok/s Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Veltrix a performance-contract runtime that can promise and enforce low-spec targets like 15 tok/s on weak devices by selecting an execution profile before generation starts and enforcing it while the model runs.

**Architecture:** Add a small runtime-profile layer that sits between CLI/model loading and the forward pass. It will convert model metadata plus device constraints into an explicit execution profile containing a context cap, thread count, compute level, and quality floor. The forward path then consumes that profile instead of relying on ad hoc `--fast` behavior, so low-spec behavior becomes predictable, testable, and easy to explain.

**Tech Stack:** C11, existing Veltrix runtime, OpenMP, CMake, unit tests in `tests/`, synthetic GGUF/ONNX fixtures already in the repo.

---

### Task 1: Add a runtime-profile API and device model

**Files:**
- Create: `C:/Users/User/Desktop/veltrix/include/runtime_profile.h`
- Create: `C:/Users/User/Desktop/veltrix/src/runtime_profile.c`
- Modify: `C:/Users/User/Desktop/veltrix/include/veltrix.h`
- Modify: `C:/Users/User/Desktop/veltrix/CMakeLists.txt`
- Test: `C:/Users/User/Desktop/veltrix/tests/test_runtime_profile.c`

- [ ] **Step 1: Write the failing test**

```c
#include "runtime_profile.h"
#include <assert.h>
#include <string.h>

int main(void) {
    vx_model_config cfg = {0};
    cfg.n_layers = 24;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.n_embd = 2048;
    cfg.n_head_dim = 64;
    cfg.n_ff = 5632;
    cfg.n_vocab = 32000;
    cfg.n_ctx = 4096;

    vx_device_profile device = {
        .name = "n100",
        .cpu_cores = 4,
        .sustained_threads = 4,
        .ram_mb = 4096,
        .is_arm = false,
        .thermal_budget = 70,
    };

    vx_runtime_goal goal = {
        .target_tok_s = 15.0f,
        .quality_floor = 0.85f,
        .ctx_limit = 256,
    };

    vx_runtime_profile profile = vx_choose_runtime_profile(&cfg, &device, &goal);
    assert(profile.allow_run);
    assert(strcmp(profile.name, "mobile-low") == 0);
    assert(profile.ctx_cap <= 256);
    assert(profile.n_threads <= 4);
    assert(profile.expected_tok_s >= 15.0f);
    assert(profile.quality_floor >= 0.85f);
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --config Release --target veltrix_test`

Expected: the new test target does not build because `runtime_profile.h`, `vx_device_profile`, and `vx_choose_runtime_profile()` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement these exact types and functions:

```c
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
```

The selection logic should prefer a bounded low-power profile when the target is 15 tok/s and the model is in the 1B to 3B range, and it should return `allow_run = false` when the requested goal is obviously unattainable under the supplied device profile.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --config Release --target veltrix_test && .\\build\\veltrix_test.exe`

Expected: the new runtime-profile test passes and the existing suite still reports 29/29 or better.

- [ ] **Step 5: Commit**

```bash
git add include/runtime_profile.h src/runtime_profile.c include/veltrix.h CMakeLists.txt tests/test_runtime_profile.c
git commit -m "feat: add runtime profile selection"
```

### Task 2: Wire the CLI to explicit target tok/s and quality floor options

**Files:**
- Modify: `C:/Users/User/Desktop/veltrix/src/main.c`
- Modify: `C:/Users/User/Desktop/veltrix/include/veltrix.h`
- Modify: `C:/Users/User/Desktop/veltrix/src/runtime_profile.c`
- Modify: `C:/Users/User/Desktop/veltrix/src/runtime_profile.c` or `C:/Users/User/Desktop/veltrix/src/runtime_profile.h` for CLI parsing helpers
- Test: `C:/Users/User/Desktop/veltrix/tests/test_runtime_profile.c`

- [ ] **Step 1: Write the failing test**

```c
#include "runtime_profile.h"
#include <assert.h>

int main(void) {
    const char *argv[] = {
        "veltrix",
        "model.gguf",
        "--target-toks", "15",
        "--quality-floor", "0.90",
        "--device", "auto",
        "--profile", "mobile-low"
    };

    vx_cli_options opt = vx_cli_options_parse(10, argv);
    assert(opt.target_tok_s == 15.0f);
    assert(opt.quality_floor == 0.90f);
    assert(opt.use_device_auto);
    assert(strcmp(opt.profile_name, "mobile-low") == 0);
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --config Release --target veltrix_test`

Expected: compile failure because `vx_cli_options` and `vx_cli_options_parse()` are not defined yet.

- [ ] **Step 3: Write the minimal implementation**

Add a CLI options struct that only carries the policy inputs needed by the runtime profile layer:

```c
typedef struct {
    float target_tok_s;
    float quality_floor;
    bool use_device_auto;
    char profile_name[32];
} vx_cli_options;

vx_cli_options vx_cli_options_parse(int argc, const char **argv);
```

Update `main.c` to map those options into `vx_runtime_goal`, call `vx_choose_runtime_profile()`, and print the chosen profile before generation begins.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --config Release --target veltrix_test && .\\build\\veltrix_test.exe`

Expected: the CLI parsing test passes, and `veltrix` still accepts the current prompt and benchmark flags.

- [ ] **Step 5: Commit**

```bash
git add src/main.c include/veltrix.h src/runtime_profile.c tests/test_runtime_profile.c
git commit -m "feat: parse target tok/s runtime options"
```

### Task 3: Enforce the profile inside the forward path

**Files:**
- Modify: `C:/Users/User/Desktop/veltrix/src/model.c`
- Modify: `C:/Users/User/Desktop/veltrix/include/veltrix.h`
- Modify: `C:/Users/User/Desktop/veltrix/src/runtime_profile.c`
- Test: `C:/Users/User/Desktop/veltrix/tests/test_e2e.c`

- [ ] **Step 1: Write the failing test**

Extend the synthetic end-to-end test with a profile application check:

```c
vx_runtime_goal goal = {
    .target_tok_s = 15.0f,
    .quality_floor = 0.85f,
    .ctx_limit = 256,
};
vx_device_profile device = vx_device_profile_auto();
vx_runtime_profile profile = vx_choose_runtime_profile(&model->config, &device, &goal);
vx_apply_runtime_profile(model, &profile);
assert(model->config.n_ctx <= 256);
assert(model->config.n_threads > 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --config Release --target veltrix_e2e`

Expected: the new API is missing and the test does not compile yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

```c
void vx_apply_runtime_profile(vx_model *model, const vx_runtime_profile *profile);
```

This function should:

- cap `model->config.n_ctx`
- set the active OpenMP thread count
- set the current compute level
- store the profile name and quality floor somewhere the forward path can read without re-parsing CLI flags

The forward pass should use the stored profile to decide whether to keep exact, attn-only, or skip mode when the target budget is tight.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --config Release --target veltrix_e2e && .\\build\\veltrix_e2e.exe`

Expected: the synthetic model still produces finite logits, and the profile application assertions pass.

- [ ] **Step 5: Commit**

```bash
git add src/model.c include/veltrix.h tests/test_e2e.c src/runtime_profile.c
git commit -m "feat: apply runtime profiles in forward pass"
```

### Task 4: Add sustained-performance guardrails and benchmark reporting

**Files:**
- Create: `C:/Users/User/Desktop/veltrix/include/runtime_guard.h`
- Create: `C:/Users/User/Desktop/veltrix/src/runtime_guard.c`
- Modify: `C:/Users/User/Desktop/veltrix/tests/benchmark.c`
- Modify: `C:/Users/User/Desktop/veltrix/src/main.c`
- Test: `C:/Users/User/Desktop/veltrix/tests/test_runtime_guard.c`

- [ ] **Step 1: Write the failing test**

```c
#include "runtime_guard.h"
#include <assert.h>

int main(void) {
    vx_runtime_guard guard = vx_runtime_guard_create(15.0f, 0.85f);
    vx_runtime_guard_tick(&guard, 18.2f);
    vx_runtime_guard_tick(&guard, 14.1f);
    vx_runtime_guard_tick(&guard, 13.8f);
    assert(guard.throughput_floor_breached);
    assert(guard.recommended_compute_level != VX_COMPUTE_EXACT);
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --config Release --target veltrix_test`

Expected: compile failure because `runtime_guard.h` and guard APIs do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement a small moving-average guard that watches recent tok/s and lowers the recommended compute level when throughput slips under the promised floor.

```c
typedef struct {
    float target_tok_s;
    float quality_floor;
    float recent_tok_s[8];
    int idx;
    bool throughput_floor_breached;
    vx_compute_level recommended_compute_level;
} vx_runtime_guard;
```

Update the benchmark output to print:

```text
Target tok/s: 15.0
Recommended profile: mobile-low
Expected tok/s: 16.4
Quality floor: 0.85
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --config Release --target veltrix_test veltrix_bench && .\\build\\veltrix_test.exe`

Expected: the new guard test passes and `veltrix_bench` still runs to completion.

- [ ] **Step 5: Commit**

```bash
git add include/runtime_guard.h src/runtime_guard.c tests/test_runtime_guard.c tests/benchmark.c src/main.c
git commit -m "feat: add sustained throughput guardrails"
```

### Task 5: Document the product position and operator workflow

**Files:**
- Modify: `C:/Users/User/Desktop/veltrix/README.md`
- Modify: `C:/Users/User/Desktop/veltrix/docs/roadmap.md`
- Create: `C:/Users/User/Desktop/veltrix/docs/optimization/runtime-profiles.md`

- [ ] **Step 1: Write the documentation changes**

Add a short operator section to README:

```md
## Performance contract

Veltrix can be launched with a target tok/s. When the target is 15 tok/s or lower on low-spec hardware, Veltrix selects a runtime profile that trades compute depth and context length for sustained throughput.

Example:

veltrix model.gguf --target-toks 15 --quality-floor 0.85 --device auto --profile mobile-low
```

Add a dedicated optimization note that explains why Veltrix is different from Ollama, llama.cpp, and LM Studio: it measures the device, chooses a bounded profile, and refuses to pretend a slow profile is acceptable if the contract cannot be met.

- [ ] **Step 2: Review for consistency**

Run: `Get-Content README.md`, `Get-Content docs/optimization/runtime-profiles.md`

Expected: no references to functions or flags that were not added in the implementation tasks.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/roadmap.md docs/optimization/runtime-profiles.md
git commit -m "docs: describe target tok/s runtime contract"
```

---

**Coverage check:**
- Target tok/s contract: Task 1, Task 2, Task 3, Task 4
- Device profiling and auto selection: Task 1, Task 2
- Quality floor and degrade ladder: Task 1, Task 3, Task 4
- Thermal/sustained throughput guard: Task 4
- Bench and operator visibility: Task 4, Task 5

**Primary risk:** If the runtime profile is kept as a CLI-only concept, the guarantee will be too weak to differentiate Veltrix. The implementation must make the profile visible to the model runner and to the benchmark output, not just to argument parsing.
