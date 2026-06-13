# Free-Energy Scheduler

`include/scheduler.h` + `src/scheduler.c`

An adaptive computation scheduler that decides, per-layer, whether to compute exactly, compute attention only, or skip entirely — based on free-energy principles from predictive coding theory.

## Concepts

- **Free Energy**: A measure of prediction error weighted by precision. Lower free energy = more predictable = layer can be skipped.
- **Prediction Error**: L2 distance between predicted and actual layer output.
- **Computation Cost**: Assigns cost to exact compute, attn-only, or skip decisions.
- **Policy**: Strategy for balancing computation vs. accuracy.

## vx_fe_state

```c
typedef struct {
    float *layer_fe;          // Per-layer free energy
    float *layer_pe;          // Per-layer prediction error
    float *layer_skip_rate;   // Per-layer skip rate (running average)
    float total_free_energy;
    int n_tokens_processed;
    int n_layers_skipped;
    int n_layers_exact;
    int n_attns_skipped;
} vx_fe_state;
```

## vx_scheduler_config

```c
typedef struct {
    vx_policy_type policy;         // MIN_FE | TARGET_SKIP | ADAPTIVE
    float target_skip_rate;        // Desired skip rate (0.0-0.95)
    float fe_temperature;          // Softmax temperature for probability
    float lr;                      // Learning rate for adaptation
    float precision_init;          // Initial precision value
    int window_size;               // Moving average window
    float *fe_history;             // FE history buffer
    int fe_history_idx;
    float avg_fe;                  // Running average FE
} vx_scheduler_config;
```

## Policies

### `VX_POLICY_MIN_FE`
Always choose the compute level with minimum free energy. Conservative — favors exact compute when uncertainty is high.

### `VX_POLICY_TARGET_SKIP`
Aim for a target skip rate (e.g., 40% of layers skipped). Uses a proportional controller to adjust the confidence threshold:
- If actual skip rate < target → lower threshold (skip more)
- If actual skip rate > target → raise threshold (skip less)

### `VX_POLICY_ADAPTIVE`
Combines free energy and target skip: adjusts threshold based on both prediction error and skip rate mismatch.

## Decision Logic

```c
vx_compute_level vx_scheduler_decide(state, confidence, free_energy, cfg)
```

1. Compute probabilities for each level (exact, attn, skip) using free energy + temperature
2. Sample or pick argmin based on policy
3. Return `VX_COMPUTE_EXACT`, `VX_COMPUTE_ATTN_ONLY`, or `VX_COMPUTE_SKIP`

## Update

```c
void vx_scheduler_update(state, layer_idx, level, free_energy, prediction_error)
```

Updates per-layer statistics and exponential moving averages after each forward pass.

## Free Energy Computation

```c
float vx_compute_free_energy(prediction, actual, dim, precision)
```

```
FE = 0.5 * precision * ||prediction - actual||² - 0.5 * log(precision)
```

Where `precision = 1/variance` represents confidence in the prediction.

## Prediction Error

```c
float vx_compute_prediction_error(pred, actual, dim)
```

```
PE = ||pred - actual||² / dim    (MSE)
```
