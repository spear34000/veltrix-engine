# Model Struct & Configuration

## `vx_model` (veltrix.h)

Central struct holding everything needed for inference:

```c
typedef struct vx_model {
    vx_model_config config;        // Architecture parameters
    vx_tensor **tensors;           // All raw tensors (owned)
    int n_tensors;

    // Resolved weight pointers (point into tensors[])
    vx_tensor *tok_embd;           // Token embedding table [n_vocab, n_embd]
    vx_tensor *output_w;           // Output projection (or tied to tok_embd)
    vx_tensor *norm_w;             // Final norm weight
    vx_tensor *norm_bias_w;        // Final norm bias (optional)

    vx_tensor **attn_q, **attn_k, **attn_v, **attn_o;           // Per-layer
    vx_tensor **attn_q_bias, **attn_k_bias, **attn_v_bias;      // Attention biases
    vx_tensor **ffn_gate, **ffn_up, **ffn_down;                  // FFN weights
    vx_tensor **ffn_gate_bias, **ffn_up_bias, **ffn_down_bias;  // FFN biases
    vx_tensor **attn_norm, **ffn_norm;                           // Layer norms
    vx_tensor **attn_norm_bias, **ffn_norm_bias;                 // Norm biases (LayerNorm)

    // Meta-learning
    vx_meta_predictor **layer_predictors;
    vx_meta_predictor *attn_pattern_predictor;
    vx_meta_predictor *token_exit_predictor;

    // KV cache
    float *k_cache, *v_cache;
    int cache_len;
} vx_model;
```

## `vx_model_config`

```c
typedef struct {
    int n_layers;           // Number of transformer layers
    int n_heads;            // Number of attention heads (Q)
    int n_kv_heads;         // Number of KV heads (MQA/GQA)
    int n_embd;             // Embedding / hidden dimension
    int n_head_dim;         // Head dimension (usually n_embd / n_heads)
    int n_ff;               // Feed-forward intermediate dimension
    int n_vocab;            // Vocabulary size
    int n_ctx;              // Maximum context length
    float rope_theta;       // RoPE frequency base
    float rope_freq_base;   // Alternative rope base (some models)
    int n_threads;
    vx_arch arch;           // Architecture enum
    vx_mlp_type mlp_type;   // SwiGLU / GEGLU / Classic
    vx_norm_type norm_type; // RMS / LayerNorm
    int rope_partial_dims;  // Partial RoPE dimension (Gemma)
    bool has_q_bias;        // Attention Q has bias
    bool has_k_bias;
    bool has_v_bias;
    bool has_post_attn_norm;
    bool has_post_ffn_norm;
    bool has_ffn_bias;      // FFN layers have bias
} vx_model_config;
```

## Weight Resolution

`vx_model_resolve_weights()` in `format.c` iterates `model->tensors[]` and matches names against known patterns:

| Field | GGUF Pattern | HF Pattern |
|---|---|---|
| `tok_embd` | `token_embd.weight` | `model.embed_tokens.weight` |
| `output_w` | `output.weight` | `lm_head.weight` |
| `norm_w` | `output_norm.weight` | `model.norm.weight` |
| `attn_q[i]` | `blk.{i}.attn_q.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `attn_k[i]` | `blk.{i}.attn_k.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `attn_v[i]` | `blk.{i}.attn_v.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `attn_o[i]` | `blk.{i}.attn_output.weight` | `model.layers.{i}.self_attn.o_proj.weight` |
| `ffn_gate[i]` | `blk.{i}.ffn_gate.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `ffn_up[i]` | `blk.{i}.ffn_up.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `ffn_down[i]` | `blk.{i}.ffn_down.weight` | `model.layers.{i}.mlp.down_proj.weight` |
| `attn_norm[i]` | `blk.{i}.attn_norm.weight` | `model.layers.{i}.input_layernorm.weight` |
| `ffn_norm[i]` | `blk.{i}.ffn_norm.weight` | `model.layers.{i}.post_attention_layernorm.weight` |

Tied embeddings: if `output_w` not found, falls back to `tok_embd`.

## Architecture Auto-Detection

`vx_model_apply_arch()` maps the `general.architecture` (GGUF) or `model_type` (HF config.json) string to architecture-specific defaults:

| Arch | MLP Type | Norm Type | Biases | Notes |
|---|---|---|---|---|
| Qwen2 | SwiGLU | RMS | None | `rope_partial_dims=0` |
| Llama | SwiGLU | RMS | None | MQA/GQA |
| Mistral | SwiGLU | RMS | None | Sliding window |
| Gemma | GEGLU | RMS | Q/K/V bias | `rope_partial_dims=n_embd/2` |
| Phi | SwiGLU | Layer | Q bias | Bias in attn + ffn |
| Phi-3 | SwiGLU | RMS | None | |
| StarCoder | Classic | Layer | Q/K/V bias | GEGLU → Classic, LayerNorm |
