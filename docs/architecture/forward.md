# Forward Pass Pipeline

`vx_model_forward()` in `src/model.c` — the core inference loop.

## Single-Token Path

```
Input: token ID
    │
    ▼
1. Embedding Lookup
   - F32: direct row copy from tok_embd[token]
   - Quantized: dequantize single row (byte-offset arithmetic)
    │
    ▼
2. For each layer i = 0 .. n_layers-1:
    │
    ├── a) Pre-Attention Norm
    │     RMS Norm or Layer Norm (with optional bias)
    │
    ├── b) QKV Projection
    │     Q = attn_q[i] · norm(x) + bias_q
    │     K = attn_k[i] · norm(x) + bias_k
    │     V = attn_v[i] · norm(x) + bias_v
    │
    ├── c) RoPE (Rotary Position Embedding)
    │     Apply sin/cos rotary to Q and K (first rope_dims dimensions)
    │
    ├── d) KV Cache Update
    │     Store K, V at position cache_len
    │     (Incremental: cache grows by 1 per token)
    │
    ├── e) Scaled Dot-Product Attention
    │     scores = Q · K^T / sqrt(head_dim)
    │     scores += mask (causal: upper triangle = -inf)
    │     weights = softmax(scores)
    │     out = weights · V
    │
    ├── f) Output Projection
    │     attn_out = attn_o[i] · attention_output
    │     residual = x + attn_out
    │
    ├── g) Post-Attention Norm (optional)
    │     Only if has_post_attn_norm (some Gemma/Phi variants)
    │
    ├── h) FFN (feed-forward)
    │     SwiGLU:  x = silu(gate(norm(x))) * up(norm(x)); out = down(x)
    │     GEGLU:   x = gelu(gate(norm(x))) * up(norm(x)); out = down(x)
    │     Classic: x = gelu(up(norm(x))); out = down(x)
    │     All include optional bias on gate/up/down
    │
    └── i) Residual
          x = residual + ffn_out
    │
    ▼
3. Final Norm
   norm_w (RMS or Layer, with optional bias)
    │
    ▼
4. Output Projection
   logits = output_w · norm(x)
    │
    ▼
Output: logits (n_vocab floats)
```

## Multi-Token Batched Path

When `n_tokens > 1`:
1. Process all tokens through embedding lookup → `(n_tokens, n_embd)` buffer
2. For each layer, GEMV is called once per row per token (sequential over tokens)
3. KV cache stores K/V for all tokens (each token appends to cache)
4. Attention uses full KV cache (causal mask applied)
5. Only the last token's logits are returned (decoder-only)

## KV Cache

- Stored as flat F32 arrays: `k_cache[n_layers * n_ctx * n_kv_heads * head_dim]`
- `v_cache` same layout
- `cache_len` tracks how many positions are filled
- `vx_model_kv_truncate(len)` resets cache_len (for speculative decoding rollback)

## Tensor Dimension Convention

Veltrix follows GGUF dimension order (`ne[0]` = fastest-changing):
- Weight matrix shape `ne = [in_features, out_features]`
- Row = in_features (dot product dimension)
- Column = out_features (number of independent dot products)

For safetensors, dimensions are reversed on load (HF `[out, in]` → Veltrix `[in, out]`).

## Dequantization Strategy

- GEMV for quantized types: dequantize one row at a time into a small F32 buffer, then dot with input vector.
- Embedding lookup for quantized tables: dequantize only the requested row (byte-offset = `token * strided_nbytes`).
- Full-tensor dequant is avoided — only the touched row is converted.
