#include "scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

vx_fe_state *vx_fe_state_create(int n_layers) {
    vx_fe_state *state = calloc(1, sizeof(vx_fe_state));
    if (!state) return NULL;
    state->n_layers = n_layers;
    state->layer_fe = calloc(n_layers, sizeof(float));
    state->layer_pe = calloc(n_layers, sizeof(float));
    state->layer_skip_rate = calloc(n_layers, sizeof(float));
    if (!state->layer_fe || !state->layer_pe || !state->layer_skip_rate) {
        free(state->layer_fe); free(state->layer_pe);
        free(state->layer_skip_rate); free(state);
        return NULL;
    }
    return state;
}

void vx_fe_state_destroy(vx_fe_state *state) {
    if (!state) return;
    free(state->layer_fe);
    free(state->layer_pe);
    free(state->layer_skip_rate);
    free(state);
}

void vx_fe_state_reset(vx_fe_state *state) {
    if (!state) return;
    if (state->layer_fe && state->n_layers > 0)
        memset(state->layer_fe, 0, (size_t)state->n_layers * sizeof(float));
    if (state->layer_pe && state->n_layers > 0)
        memset(state->layer_pe, 0, (size_t)state->n_layers * sizeof(float));
    if (state->layer_skip_rate && state->n_layers > 0)
        memset(state->layer_skip_rate, 0, (size_t)state->n_layers * sizeof(float));
    state->total_free_energy = 0.0f;
    state->n_tokens_processed = 0;
    state->n_layers_skipped = 0;
    state->n_layers_exact = 0;
    state->n_attns_skipped = 0;
}

void vx_scheduler_config_default(vx_scheduler_config *cfg) {
    cfg->policy = VX_POLICY_ADAPTIVE;
    cfg->target_skip_rate = 0.40f;
    cfg->fe_temperature = 1.0f;
    cfg->lr = 0.01f;
    cfg->precision_init = 1.0f;
    cfg->window_size = 100;
    cfg->fe_history = NULL;
    cfg->fe_history_idx = 0;
    cfg->avg_fe = 1.0f;
}

void vx_scheduler_config_set_policy(vx_scheduler_config *cfg, vx_policy_type policy) {
    cfg->policy = policy;
}

void vx_scheduler_config_set_target_skip(vx_scheduler_config *cfg, float rate) {
    if (rate < 0.0f) rate = 0.0f;
    if (rate > 0.95f) rate = 0.95f;
    cfg->target_skip_rate = rate;
}

float vx_compute_free_energy(const float *prediction, const float *actual,
                             int dim, float precision) {
    float pe = vx_compute_prediction_error(prediction, actual, dim);
    float log_precision = logf(precision + 1e-10f);
    float model_complexity = 0.5f * log_precision;
    return pe * pe / (2.0f * precision) + model_complexity;
}

float vx_compute_prediction_error(const float *pred, const float *actual, int dim) {
    float err = 0.0f;
    float norm_actual = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = pred[i] - actual[i];
        err += d * d;
        norm_actual += actual[i] * actual[i];
    }
    norm_actual = sqrtf(norm_actual + 1e-10f);
    return sqrtf(err) / (norm_actual + 1e-10f);
}

vx_compute_level vx_scheduler_decide(vx_fe_state *state,
                                     float confidence,
                                     float free_energy,
                                     const vx_scheduler_config *cfg) {
    float fe_t = free_energy * cfg->fe_temperature;

    if (cfg->policy == VX_POLICY_MIN_FE) {
        if (confidence > 0.9f && fe_t < 0.1f) return VX_COMPUTE_SKIP;
        if (confidence > 0.7f && fe_t < 0.3f) return VX_COMPUTE_ATTN_ONLY;
        return VX_COMPUTE_EXACT;
    }

    if (cfg->policy == VX_POLICY_TARGET_SKIP) {
        float current_skip = 0;
        if (state->n_tokens_processed > 0) {
            current_skip = (float)state->n_layers_skipped /
                           (state->n_tokens_processed * (state->n_layers > 0 ? state->n_layers : 1) + 1);
        }
        if (current_skip < cfg->target_skip_rate) {
            if (confidence > 0.8f && fe_t < 0.2f) return VX_COMPUTE_SKIP;
            if (confidence > 0.6f && fe_t < 0.4f) return VX_COMPUTE_ATTN_ONLY;
        } else {
            if (confidence > 0.95f && fe_t < 0.05f) return VX_COMPUTE_SKIP;
        }
        return VX_COMPUTE_EXACT;
    }

    if (cfg->policy == VX_POLICY_ADAPTIVE) {
        float threshold = 0.85f - (cfg->avg_fe - 0.1f) * 0.5f;
        if (threshold < 0.6f) threshold = 0.6f;
        if (threshold > 0.95f) threshold = 0.95f;

        if (confidence > threshold + 0.1f && fe_t < 0.15f) return VX_COMPUTE_SKIP;
        if (confidence > threshold && fe_t < 0.3f) return VX_COMPUTE_ATTN_ONLY;
        return VX_COMPUTE_EXACT;
    }

    return VX_COMPUTE_EXACT;
}

void vx_scheduler_update(vx_fe_state *state, int layer_idx,
                         vx_compute_level level,
                         float free_energy, float prediction_error) {
    if (layer_idx == 0) state->n_tokens_processed++;
    state->total_free_energy += free_energy;
    if (level == VX_COMPUTE_SKIP) state->n_layers_skipped++;
    else if (level == VX_COMPUTE_ATTN_ONLY) state->n_attns_skipped++;
    else state->n_layers_exact++;

    if (state->layer_fe && layer_idx >= 0) {
        state->layer_fe[layer_idx] += free_energy;
        state->layer_pe[layer_idx] += prediction_error;
    }
}

void vx_scheduler_adapt(vx_scheduler_config *cfg, vx_fe_state *state) {
    if (state->n_tokens_processed % 50 != 0 || state->n_tokens_processed == 0) return;

    float total = state->n_layers_exact + state->n_layers_skipped + state->n_attns_skipped;
    if (total < 1) return;

    float skip_rate = (float)(state->n_layers_skipped + state->n_attns_skipped) / total;
    float fe_avg = state->total_free_energy / (total + 1);

    cfg->avg_fe = 0.9f * cfg->avg_fe + 0.1f * fe_avg;

    if (skip_rate < cfg->target_skip_rate - 0.1f) {
        cfg->fe_temperature *= 1.05f;
        if (cfg->fe_temperature > 2.0f) cfg->fe_temperature = 2.0f;
    } else if (skip_rate > cfg->target_skip_rate + 0.1f) {
        cfg->fe_temperature *= 0.95f;
        if (cfg->fe_temperature < 0.3f) cfg->fe_temperature = 0.3f;
    }
}

float vx_fe_expected_savings(const vx_scheduler_config *cfg, float confidence) {
    (void)cfg;
    if (confidence > 0.95f) return 1.0f;
    if (confidence > 0.85f) return 0.6f;
    if (confidence > 0.7f) return 0.3f;
    return 0.0f;
}
