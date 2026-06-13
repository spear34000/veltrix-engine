# Meta-Learning Predictor System

`include/metalearn.h` + `src/metalearn.c`

A lightweight online meta-learning system that predicts layer outputs, attention sparsity, and early exit points — reducing computation while maintaining output quality.

## Architecture

```
For each layer i:
  input (n_embd)
     │
     ▼
  meta_predictor[i]           ← 2-layer MLP (n_embd → 128 → n_embd)
     │
     ├── predicted_output
     ├── confidence (0..1)
     │
     ├── [if confidence > threshold] → SKIP layer computation
     │                                  use predicted_output directly
     │
     └── [if confidence < threshold] → COMPUTE layer exactly
                                        then meta_update(input, actual_output)
```

## vx_meta_predictor

```c
typedef struct {
    float w1[VX_META_HDIM * VX_MAX_DIM];    // Input → hidden (8192 × 128)
    float b1[VX_META_HDIM];                  // Hidden bias (128)
    float w2[VX_MAX_DIM * VX_META_HDIM];    // Hidden → output (128 × 8192)
    float b2[VX_MAX_DIM];                    // Output bias (8192)
    float running_mean[VX_MAX_DIM];          // BN running mean
    float running_var[VX_MAX_DIM];           // BN running variance
    int update_count;
    float precision;                         // Learned precision parameter
} vx_meta_predictor;
```

The predictor is a 2-layer MLP with batch normalization. Input dimension = `n_embd`, hidden dimension = 128.

## Key Functions

### `vx_meta_predict(input, dim, mp, output, confidence)`
Forward pass through predictor. Confidence is computed from the L2 distance between predicted and expected output distribution, scaled by learned precision.

### `vx_meta_update(mp, input, target, dim, lr)`
Online gradient update (SGD) minimizing MSE between prediction and actual layer output.

### `vx_meta_estimate_confidence(mp, prediction, actual, dim)`
Statistical confidence based on normalized prediction error:
```c
confidence = exp(-0.5 * precision * ||prediction - actual||²)
```

## vx_meta_system

Orchestrates per-layer predictors, attention pattern predictor, and early-exit predictor:

```c
typedef struct {
    vx_meta_predictor **layer_preds;   // One per layer
    vx_meta_predictor *attn_pred;      // Predicts sparse attention heads
    vx_meta_predictor *exit_pred;      // Predicts early exit point
    int n_layers;
    int n_embd;
    float lr;
    float skip_threshold;              // Confidence threshold for skipping
    float exit_threshold;              // Confidence threshold for early exit
} vx_meta_system;
```

## Decision Functions

### `vx_meta_decide_layer(ms, layer_idx, input, dim)`
Returns `vx_decision` with:
- `skip_compute`: true if confidence > skip_threshold
- `confidence`: prediction confidence
- `predicted_norm`: L2 norm of predicted output (for scheduler)

### `vx_meta_decide_attention(ms, query, n_heads, head_dim, predicted_active)`
Predicts which attention heads can be skipped (sparse attention).

### `vx_meta_decide_exit(ms, hidden_state, dim)`
Predicts if early exit is possible at current layer.

## Usage in main.c

```c
// Per-layer predictors for skip decisions
model->layer_predictors[i] = vx_meta_predictor_create(n_embd);

// Attention sparsity predictor
model->attn_pattern_predictor = vx_meta_predictor_create(n_embd);

// Early exit predictor
model->token_exit_predictor = vx_meta_predictor_create(n_embd);
```

Note: Meta-predictors are allocated but not yet integrated into the forward pass loop. The integration (actually deciding to skip layers based on predictions) is planned for a future phase.
