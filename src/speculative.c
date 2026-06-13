#include "speculative.h"
#include "model.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct vx_speculative_state {
    vx_model *draft_model;
    vx_model *target_model;
    int max_draft_tokens;
    float temperature;
    float top_p;
    int *draft_tokens;
    float *draft_logits;
    int *accepted_tokens;
    int num_accepted;
    float *target_logits;
};

static inline void softmax_inplace(float *logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / (sum + 1e-30f);
    for (int i = 0; i < n; i++)
        logits[i] *= inv_sum;
}

static inline int sample_top_p(float *probs, int vocab_size, float top_p, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (probs[i] > probs[best]) best = i;
        return best;
    }

    float *probs_copy = malloc(vocab_size * sizeof(float));
    float max_logit = probs[0];
    for (int i = 1; i < vocab_size; i++)
        if (probs[i] > max_logit) max_logit = probs[i];
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs_copy[i] = expf((probs[i] - max_logit) / temperature);
        sum += probs_copy[i];
    }
    float inv_sum = 1.0f / (sum + 1e-30f);
    for (int i = 0; i < vocab_size; i++) probs_copy[i] *= inv_sum;

    int *indices = malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) indices[i] = i;
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs_copy[indices[j]] > probs_copy[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    float cumsum = 0;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs_copy[indices[i]];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float r = (float)rand() / RAND_MAX * cumsum;
    float csum = 0;
    int token = indices[cutoff - 1];
    for (int i = 0; i < cutoff; i++) {
        csum += probs_copy[indices[i]];
        if (csum >= r) {
            token = indices[i];
            break;
        }
    }

    free(probs_copy);
    free(indices);
    return token;
}

static inline int sample_greedy(float *probs, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

static vx_error process_prompt(vx_model *model, int *tokens, int n_tokens, float *last_logits) {
    model->cache_len = 0;
    for (int i = 0; i < n_tokens; i++) {
        vx_error err = vx_model_forward(model, &tokens[i], 1, last_logits);
        if (err != VX_OK) return err;
    }
    return VX_OK;
}

vx_speculative_state *vx_speculative_create(vx_model *draft, vx_model *target, int max_draft_tokens) {
    vx_speculative_state *state = calloc(1, sizeof(vx_speculative_state));
    if (!state) return NULL;

    state->draft_model = draft;
    state->target_model = target;
    state->max_draft_tokens = max_draft_tokens;
    state->temperature = 0.8f;
    state->top_p = 0.95f;

    state->draft_tokens = malloc(max_draft_tokens * sizeof(int));
    state->draft_logits = malloc(max_draft_tokens * target->config.n_vocab * sizeof(float));
    state->accepted_tokens = malloc(max_draft_tokens * sizeof(int));
    state->target_logits = malloc(target->config.n_vocab * sizeof(float));

    if (!state->draft_tokens || !state->draft_logits || !state->accepted_tokens || !state->target_logits) {
        vx_speculative_destroy(state);
        return NULL;
    }

    return state;
}

void vx_speculative_destroy(vx_speculative_state *state) {
    if (!state) return;
    free(state->draft_tokens);
    free(state->draft_logits);
    free(state->accepted_tokens);
    free(state->target_logits);
    free(state);
}

void vx_speculative_set_temp(vx_speculative_state *state, float temp, float top_p) {
    if (state) {
        state->temperature = temp;
        state->top_p = top_p;
    }
}

static int draft_generate(vx_speculative_state *state, int *prefix, int prefix_len, int max_new) {
    vx_model *dm = state->draft_model;
    int n_vocab = dm->config.n_vocab;
    float *logits = state->draft_logits;

    dm->cache_len = 0;

    for (int i = 0; i < prefix_len; i++) {
        vx_error err = vx_model_forward(dm, &prefix[i], 1, logits);
        if (err != VX_OK) return 0;
    }

    memcpy(state->draft_tokens, prefix, prefix_len * sizeof(int));
    int total_len = prefix_len;

    for (int i = 0; i < max_new; i++) {
        int next_token = sample_top_p(logits, n_vocab, state->top_p, state->temperature);
        state->draft_tokens[total_len++] = next_token;

        memcpy(state->draft_logits + i * n_vocab, logits, n_vocab * sizeof(float));

        if (i + 1 < max_new) {
            vx_error err = vx_model_forward(dm, &next_token, 1, logits);
            if (err != VX_OK) return i + 1;
        }
    }

    return max_new;
}

static int verify_draft(vx_speculative_state *state, int *prefix, int prefix_len, int num_draft) {
    (void)prefix;
    (void)prefix_len;
    vx_model *tm = state->target_model;
    int n_vocab_t = tm->config.n_vocab;
    int n_vocab_d = state->draft_model->config.n_vocab;
    int accepted = 0;

    for (int i = 0; i < num_draft && accepted < state->max_draft_tokens; i++) {
        float *draft_probs = state->draft_logits + i * n_vocab_d;
        int draft_token = state->draft_tokens[prefix_len + i];
        float *target_l = state->target_logits;

        if (draft_token < 0 || draft_token >= n_vocab_t)
            target_l[draft_token] = -1e10f;

        float target_prob = target_l[draft_token];
        float draft_prob = 0;
        if (draft_token >= 0 && draft_token < n_vocab_d)
            draft_prob = draft_probs[draft_token];

        float alpha = fminf(1.0f, target_prob / fmaxf(draft_prob, 1e-10f));
        float r = (float)rand() / RAND_MAX;

        if (r < alpha) {
            state->accepted_tokens[accepted++] = draft_token;
            if (i + 1 < num_draft) {
                vx_error err = vx_model_forward(tm, &draft_token, 1, target_l);
                if (err != VX_OK) break;
            }
        } else {
            for (int v = 0; v < n_vocab_t; v++) {
                float p_v = (v < n_vocab_d) ? draft_probs[v] : 0;
                target_l[v] = fmaxf(0, target_l[v] - p_v);
            }
            float residual_sum = 0;
            for (int v = 0; v < n_vocab_t; v++)
                residual_sum += target_l[v];

            int corrected = 0;
            if (residual_sum > 1e-10f) {
                float r2 = (float)rand() / RAND_MAX * residual_sum;
                float csum = 0;
                for (int v = 0; v < n_vocab_t; v++) {
                    csum += target_l[v];
                    if (csum >= r2) { corrected = v; break; }
                }
            } else {
                corrected = sample_greedy(target_l, n_vocab_t);
            }
            state->accepted_tokens[accepted++] = corrected;
            vx_model_kv_truncate(tm, prefix_len + accepted - 1);
            vx_model_forward(tm, &corrected, 1, target_l);
            break;
        }
    }

    return accepted;
}

int vx_speculative_generate(vx_speculative_state *state, int *prompt, int prompt_len,
                            int *output, int max_tokens) {
    if (!state || !prompt || !output || prompt_len < 1 || max_tokens < 1)
        return 0;

    vx_model *tm = state->target_model;
    int max_ctx = tm->config.n_ctx;
    int *prefix = malloc((size_t)(prompt_len + state->max_draft_tokens) * sizeof(int));
    if (!prefix) return 0;
    memcpy(prefix, prompt, prompt_len * sizeof(int));

    vx_error err = process_prompt(tm, prompt, prompt_len, state->target_logits);
    if (err != VX_OK) { free(prefix); return 0; }

    softmax_inplace(state->target_logits, tm->config.n_vocab);

    int generated = 0;
    int total_ctx = prompt_len;

    while (generated < max_tokens && total_ctx < max_ctx) {
        int remaining = max_tokens - generated;
        int ctx_remaining = max_ctx - total_ctx;
        int draft_len = state->max_draft_tokens;
        if (draft_len > remaining) draft_len = remaining;
        if (draft_len > ctx_remaining) draft_len = ctx_remaining;
        if (draft_len < 1) break;

        int num_drafted = draft_generate(state, prefix, total_ctx, draft_len);
        if (num_drafted == 0) break;

        int accepted = verify_draft(state, prefix, total_ctx, num_drafted);
        if (accepted == 0) {
            int fallback = sample_greedy(state->target_logits, tm->config.n_vocab);
            output[generated++] = fallback;
            prefix[total_ctx++] = fallback;
            vx_model_kv_truncate(tm, total_ctx - 1);
            err = vx_model_forward(tm, &fallback, 1, state->target_logits);
            if (err != VX_OK) break;
            softmax_inplace(state->target_logits, tm->config.n_vocab);
            continue;
        }

        for (int i = 0; i < accepted; i++) {
            output[generated++] = state->accepted_tokens[i];
            prefix[total_ctx++] = state->accepted_tokens[i];
        }

        if (total_ctx >= max_ctx) break;

        if (accepted < num_drafted) {
            softmax_inplace(state->target_logits, tm->config.n_vocab);
        }
    }

    free(prefix);
    return generated;
}

int vx_speculative_generate_stream(vx_speculative_state *state, int token,
                                   int *output, int max_tokens) {
    int prompt[1] = {token};
    return vx_speculative_generate(state, prompt, 1, output, max_tokens);
}
