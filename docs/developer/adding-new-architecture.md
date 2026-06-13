# Adding a New Model Architecture

This guide covers adding support for a new transformer architecture (e.g., DBRX, Grok, DeepSeek-V2).

## Step 1: Add Architecture Enum

In `include/veltrix.h`:
```c
typedef enum {
    VX_ARCH_QWEN2 = 0,
    VX_ARCH_LLAMA,
    VX_ARCH_MISTRAL,
    VX_ARCH_GEMMA,
    VX_ARCH_PHI,
    VX_ARCH_PHI3,
    VX_ARCH_STARCODER,
    VX_ARCH_MYARCH,       // ← Add
    VX_ARCH_COUNT,
} vx_arch;
```

## Step 2: Add MLP/Norm Types (if needed)

If the architecture uses a novel activation or normalization:
```c
typedef enum {
    VX_MLP_SWIGLU = 0,
    VX_MLP_GEGLU,
    VX_MLP_CLASSIC,
    VX_MLP_MYACT,         // ← Add
} vx_mlp_type;

typedef enum {
    VX_NORM_RMS = 0,
    VX_NORM_LAYER,
    VX_NORM_MYNORM,       // ← Add
} vx_norm_type;
```

## Step 3: Add Architecture Table Entry

In `arch_table` in `src/format.c`:
```c
static const arch_defaults arch_table[] = {
    // ... existing entries ...
    {"myarch", "myarch", VX_ARCH_MYARCH, {
        .mlp_type = VX_MLP_MYACT,
        .norm_type = VX_NORM_MYNORM,
        .has_q_bias = false,
        .has_k_bias = false,
        .has_v_bias = true,
        .has_post_attn_norm = false,
        .has_ffn_bias = false,
        .rope_partial_dims = 0,
    }},
};
```

Fields you can set:
- `mlp_type`: SwiGLU (gate+up+down), GEGLU (gate+up+down with gelu), Classic (up+down with gelu)
- `norm_type`: RMS (rms_norm) or Layer (layer_norm with optional bias)
- `has_q/k/v_bias`: Attention projections with bias (Gemma, Phi, StarCoder)
- `has_post_attn_norm`: Extra norm after attention output
- `has_ffn_bias`: FFN projections with bias
- `rope_partial_dims`: Partial RoPE dimension count (Gemma uses half)

## Step 4: Add Architecture Name Mapping

### For GGUF
In `gguf_scan_str()` result matching in `src/gguf.c`:
```c
if (strcmp(arch_str, "myarch") == 0) {
    model->config.arch = VX_ARCH_MYARCH;
}
```

### For Safetensors
In `hf_archs[]` in `src/safetensors.c`:
```c
static const hf_arch_map hf_archs[] = {
    // ... existing ...
    {"myarch_model", VX_ARCH_MYARCH},
};
```

## Step 5: Add Weight Name Patterns (if different)

In `vx_try_match_name()` in `src/format.c`, if the architecture uses unique weight naming:
```c
// MyArch naming: "transformer.layer.{i}.myq_proj.weight"
if (strstr(name, "transformer.layer.")) {
    // Parse layer index, match weight suffix to internal patterns
    // e.g., "myq_proj" → "attn_q"
}
```

## Step 6: Add Forward Pass Logic (if novel)

If the architecture has a non-standard forward pass (e.g., multi-query attention variant, parallel attn+ffn, custom activation), modify `vx_model_forward()` in `src/model.c`:

```c
switch (model->config.arch) {
    case VX_ARCH_MYARCH:
        // Custom attention logic
        // Custom FFN logic
        break;
    default:
        // Standard path (Llama-style)
        break;
}
```

Common forward pass variants:
- **Parallel attn+ffn**: Compute both attention and FFN from same norm, then sum residuals (PaLM, some GPT-NeoX)
- **MLA (Multi-head Latent Attention)**: DeepSeek-V2 compressed KV projections
- **Multi-Query Attention**: Single K,V head shared across all Q heads (less common now)
- **Sliding Window Attention**: Mistral-style limited attention span
- **GQA with different head_dim**: Q and KV heads have different head dimensions

## Step 7: Test

```bash
# Unit tests should still pass (no regressions)
./build/veltrix_test

# Test with real model
./build/veltrix models/my-arch-model.gguf -p "Test" -n 64
```

Check that:
- All weight tensors are found (check `n_tensors` count)
- Forward pass produces non-zero logits (no NaN)
- Generated tokens are diverse (not all same token)
