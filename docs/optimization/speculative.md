# Speculative Decoding

`include/speculative.h` + `src/speculative.c`

Speculative decoding accelerates autoregressive generation by using a **draft model** to propose multiple tokens, which are then **verified** by a **target model** in parallel.

## Algorithm

```
1. Draft model generates γ candidate tokens (greedy/sampled)
2. Target model computes logits for each candidate
3. Rejection sampling:
   For each candidate token t_i:
     α = min(1, p_target(t_i) / p_draft(t_i))
     If random < α → ACCEPT
     Else → sample from residual distribution (p_target - p_draft)⁺  → BREAK
4. Output all accepted tokens
5. Roll back KV cache to last accepted position
6. Continue from accepted position
```

## Key Structures

```c
typedef struct vx_speculative_state {
    vx_model *draft_model;         // Smaller/faster model
    vx_model *target_model;        // Larger/more accurate model
    int max_draft_tokens;          // γ (typical: 5-10)
    float temperature;             // Sampling temperature
    float top_p;                   // Nucleus sampling threshold
    int *draft_tokens;             // Buffer for draft tokens
    float *draft_logits;           // Buffer for draft logprobs
    int *accepted_tokens;          // Buffer for verified tokens
    float *target_logits;          // Buffer for target logits
    int num_accepted;              // Number accepted in current round
} vx_speculative_state;
```

## Key Functions

### `vx_speculative_create(draft, target, max_draft_tokens)`
Allocates state with KV cache buffers. Both models must share the same vocabulary (or have compatible tokenizers).

### `vx_speculative_generate(state, prompt, prompt_len, output, max_tokens)`
Main generation loop. Returns number of tokens generated.

### `vx_speculative_generate_stream(state, token, output, max_tokens)`
Convenience wrapper for single-token streaming (auto-regressive loop).

### Draft Generation
- Runs draft model autoregressively for γ steps
- Stores both token IDs and log probabilities
- Uses top-p sampling (temperature configurable)

### Verification
For each draft token:
1. Looks up target probability and draft probability
2. Computes acceptance probability `α = min(1, target_prob / draft_prob)`
3. If accepted → store token, continue
4. If rejected → sample from `max(0, target - draft)` residual distribution

## Rollback Mechanism

When a token is rejected, KV cache must be truncated:
```c
vx_model_kv_truncate(model, position);
```
This resets `cache_len` to the last accepted position, allowing the model to generate the corrected token from the right context.

## Integration

Currently the speculative decoding API is standalone (not integrated into the main CLI loop). Typical usage:

```c
vx_speculative_state *state = vx_speculative_create(draft, target, 5);
vx_speculative_set_temp(state, 0.8f, 0.95f);
int n = vx_speculative_generate(state, prompt, prompt_len, output, max_tokens);
vx_speculative_destroy(state);
```

## Limitations

- Both models must share the same vocabulary
- Draft model must be significantly faster (3-10×) for net speedup
- No batching within verification step (sequential forward passes)
- KV cache rollback is O(1) (just reset `cache_len`)
