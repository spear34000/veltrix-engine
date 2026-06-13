#include "metalearn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

vx_meta_predictor *vx_meta_predictor_create(int input_dim) {
    vx_meta_predictor *mp = calloc(1, sizeof(vx_meta_predictor));
    if (mp) vx_meta_predictor_init(mp, input_dim);
    return mp;
}

void vx_meta_predictor_destroy(vx_meta_predictor *mp) {
    free(mp);
}

void vx_meta_predictor_init(vx_meta_predictor *mp, int input_dim) {
    mp->precision = 1.0f;
    mp->update_count = 0;
    for (int i = 0; i < VX_META_HDIM; i++) {
        mp->b1[i] = 0.0f;
    }
    for (int i = 0; i < VX_MAX_DIM; i++) {
        mp->b2[i] = 0.0f;
        mp->running_mean[i] = 0.0f;
        mp->running_var[i] = 1.0f;
    }
    float scale = sqrtf(2.0f / input_dim);
    for (int i = 0; i < VX_META_HDIM && input_dim > 0; i++) {
        for (int j = 0; j < (input_dim < VX_MAX_DIM ? input_dim : VX_MAX_DIM); j++) {
            mp->w1[i * VX_MAX_DIM + j] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        }
    }
    scale = sqrtf(2.0f / VX_META_HDIM);
    for (int i = 0; i < (input_dim < VX_MAX_DIM ? input_dim : VX_MAX_DIM) && VX_META_HDIM > 0; i++) {
        for (int j = 0; j < VX_META_HDIM; j++) {
            mp->w2[i * VX_META_HDIM + j] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        }
    }
}

static float relu(float x) { return x > 0.0f ? x : 0.0f; }

void vx_meta_predict(float *input, int dim, vx_meta_predictor *mp,
                     float *output, float *confidence) {
    float hidden[VX_META_HDIM];
    int hdim = VX_META_HDIM < 128 ? VX_META_HDIM : 128;
    if (hdim > dim) hdim = dim / 4;
    if (hdim < 16) hdim = 16;

    for (int i = 0; i < hdim; i++) {
        float s = mp->b1[i];
        for (int j = 0; j < dim; j++) {
            s += mp->w1[i * VX_MAX_DIM + j] * input[j];
        }
        hidden[i] = relu(s);
    }

    float norm_pred = 0.0f;
    for (int i = 0; i < dim; i++) {
        float s = mp->b2[i];
        for (int j = 0; j < hdim; j++) {
            s += mp->w2[i * VX_META_HDIM + j] * hidden[j];
        }
        output[i] = s;
        norm_pred += s * s;
    }
    norm_pred = sqrtf(norm_pred + 1e-8f);

    if (mp->update_count > 10) {
        float var_sum = 0.0f;
        for (int i = 0; i < dim; i++) var_sum += mp->running_var[i];
        float pred_score = 0.0f;
        for (int i = 0; i < dim; i++) {
            float dev = (output[i] - mp->running_mean[i]);
            pred_score += dev * dev / (mp->running_var[i] + 1e-6f);
        }
        pred_score = sqrtf(pred_score / dim);
        float inv_precision = 1.0f / (mp->precision + 0.001f);
        *confidence = 1.0f / (1.0f + pred_score * inv_precision);
        if (*confidence > 0.99f) *confidence = 0.99f;
        if (*confidence < 0.01f) *confidence = 0.01f;
    } else {
        *confidence = 0.5f;
    }
}

void vx_meta_update(vx_meta_predictor *mp, float *input, float *target,
                    int dim, float lr) {
    float hidden[VX_META_HDIM];
    int hdim = VX_META_HDIM < 128 ? VX_META_HDIM : 128;
    if (hdim > dim) hdim = dim / 4;
    if (hdim < 16) hdim = 16;

    for (int i = 0; i < hdim; i++) {
        float s = mp->b1[i];
        for (int j = 0; j < dim; j++) {
            s += mp->w1[i * VX_MAX_DIM + j] * input[j];
        }
        hidden[i] = relu(s);
    }

    float d_output[VX_MAX_DIM];
    float error_sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float pred = 0;
        for (int j = 0; j < hdim; j++) {
            pred += mp->w2[i * VX_META_HDIM + j] * hidden[j];
        }
        pred += mp->b2[i];
        float err = target[i] - pred;
        d_output[i] = err;
        error_sum += err * err;

        for (int j = 0; j < hdim; j++) {
            mp->w2[i * VX_META_HDIM + j] += lr * err * hidden[j];
        }
        mp->b2[i] += lr * err * 0.1f;
    }
    float rmse = sqrtf(error_sum / dim + 1e-10f);

    float d_hidden[VX_META_HDIM] = {0};
    for (int j = 0; j < hdim; j++) {
        for (int i = 0; i < dim; i++) {
            d_hidden[j] += d_output[i] * mp->w2[i * VX_META_HDIM + j];
        }
        if (hidden[j] > 0) {
            for (int k = 0; k < dim; k++) {
                mp->w1[j * VX_MAX_DIM + k] += lr * d_hidden[j] * input[k];
            }
            mp->b1[j] += lr * d_hidden[j] * 0.1f;
        }
    }

    float decay = 0.99f;
    for (int i = 0; i < dim; i++) {
        mp->running_mean[i] = decay * mp->running_mean[i] + (1 - decay) * target[i];
        float dev = target[i] - mp->running_mean[i];
        mp->running_var[i] = decay * mp->running_var[i] + (1 - decay) * dev * dev;
        if (mp->running_var[i] < 0.001f) mp->running_var[i] = 0.001f;
    }
    mp->update_count++;

    mp->precision = 1.0f / (rmse + 0.01f);
    if (mp->precision > 100.0f) mp->precision = 100.0f;
    if (mp->precision < 0.01f) mp->precision = 0.01f;
}

float vx_meta_estimate_confidence(vx_meta_predictor *mp, float *prediction,
                                  float *actual, int dim) {
    (void)mp;
    float err = 0.0f, pred_norm = 0.0f, actual_norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = prediction[i] - actual[i];
        err += d * d;
        pred_norm += prediction[i] * prediction[i];
        actual_norm += actual[i] * actual[i];
    }
    pred_norm = sqrtf(pred_norm);
    actual_norm = sqrtf(actual_norm);
    if (actual_norm < 1e-6f) return 0.99f;
    float rel_err = sqrtf(err) / actual_norm;
    return 1.0f / (1.0f + rel_err * 10.0f);
}

vx_meta_system *vx_meta_system_create(int n_layers, int n_embd) {
    vx_meta_system *ms = calloc(1, sizeof(vx_meta_system));
    if (!ms) return NULL;
    ms->n_layers = n_layers;
    ms->n_embd = n_embd;
    ms->lr = 0.001f;
    ms->skip_threshold = 0.85f;
    ms->exit_threshold = 0.90f;

    ms->layer_preds = calloc(n_layers, sizeof(vx_meta_predictor *));
    if (!ms->layer_preds) { free(ms); return NULL; }
    for (int i = 0; i < n_layers; i++) {
        ms->layer_preds[i] = vx_meta_predictor_create(n_embd);
        if (!ms->layer_preds[i]) {
            for (int j = 0; j < i; j++) vx_meta_predictor_destroy(ms->layer_preds[j]);
            free(ms->layer_preds); free(ms);
            return NULL;
        }
    }
    ms->attn_pred = vx_meta_predictor_create(n_embd);
    ms->exit_pred = vx_meta_predictor_create(n_embd);
    return ms;
}

void vx_meta_system_destroy(vx_meta_system *ms) {
    if (!ms) return;
    if (ms->layer_preds) {
        for (int i = 0; i < ms->n_layers; i++)
            vx_meta_predictor_destroy(ms->layer_preds[i]);
        free(ms->layer_preds);
    }
    vx_meta_predictor_destroy(ms->attn_pred);
    vx_meta_predictor_destroy(ms->exit_pred);
    free(ms);
}

vx_decision vx_meta_decide_layer(vx_meta_system *ms, int layer_idx,
                                 float *input, int dim) {
    vx_decision d;
    d.exit_layer = -1;
    d.skip_compute = false;
    d.free_energy = 0.0f;

    if (layer_idx >= ms->n_layers || !ms->layer_preds[layer_idx]) {
        d.confidence = 0.0f;
        return d;
    }

    vx_meta_predictor *mp = ms->layer_preds[layer_idx];
    float predicted[VX_MAX_DIM];
    float norm_pred = 0.0f;
    for (int i = 0; i < dim; i++) {
        float s = 0;
        for (int j = 0; j < 16; j++) {
            s += mp->w1[j * VX_MAX_DIM + i] * input[i];
        }
        predicted[i] = s;
        norm_pred += s * s;
    }
    d.predicted_norm = sqrtf(norm_pred);

    float conf;
    vx_meta_predict(input, dim, mp, predicted, &conf);
    d.confidence = conf;

    float norm_input = 0;
    for (int i = 0; i < dim; i++) norm_input += input[i] * input[i];
    norm_input = sqrtf(norm_input);

    d.free_energy = d.predicted_norm / (norm_input + 1e-6f);
    d.free_energy = d.free_energy * (1.0f - conf);

    d.skip_compute = (conf >= ms->skip_threshold);

    return d;
}

vx_decision vx_meta_decide_attention(vx_meta_system *ms, float *query,
                                     int n_heads, int head_dim,
                                     float *predicted_active) {
    vx_decision d;
    d.exit_layer = -1;
    d.skip_compute = false;
    d.predicted_norm = 0.0f;

    float conf;
    float query_flat[VX_MAX_DIM];
    int flat_dim = n_heads * head_dim;
    if (flat_dim > VX_MAX_DIM) flat_dim = VX_MAX_DIM;
    memcpy(query_flat, query, flat_dim * sizeof(float));

    vx_meta_predict(query_flat, flat_dim, ms->attn_pred, query_flat, &conf);
    d.confidence = conf;

    if (predicted_active) {
        for (int h = 0; h < n_heads; h++) {
            predicted_active[h] = query_flat[h] > 0.0f ? 1.0f : 0.0f;
        }
    }

    d.skip_compute = (conf >= ms->skip_threshold);
    d.free_energy = (1.0f - conf) * n_heads / 32.0f;
    return d;
}

vx_decision vx_meta_decide_exit(vx_meta_system *ms, float *hidden_state,
                                int dim) {
    vx_decision d;
    d.skip_compute = false;
    d.predicted_norm = 0.0f;

    float exit_score[VX_MAX_DIM];
    float conf;
    vx_meta_predict(hidden_state, dim, ms->exit_pred, exit_score, &conf);
    d.confidence = conf;

    if (conf < 0.3f || conf > 0.95f) {
    } else if (conf < 0.5f) {
        conf = 0.5f;
    }

    d.exit_layer = (conf >= ms->exit_threshold) ? 1 : -1;
    d.free_energy = (1.0f - conf) * 10.0f;

    return d;
}

void vx_meta_observe_layer(vx_meta_system *ms, int layer_idx,
                           float *input, float *actual_output, int dim) {
    if (layer_idx >= ms->n_layers) return;
    vx_meta_predictor *mp = ms->layer_preds[layer_idx];
    vx_meta_update(mp, input, actual_output, dim, ms->lr);
}

void vx_meta_observe_attention(vx_meta_system *ms, float *query,
                               int *active_heads, int n_heads) {
    float targets[VX_MAX_DIM];
    for (int i = 0; i < n_heads && i < VX_MAX_DIM; i++) {
        targets[i] = active_heads[i] ? 1.0f : 0.0f;
    }
    vx_meta_update(ms->attn_pred, query, targets, n_heads, ms->lr);
}

void vx_meta_system_set_params(vx_meta_system *ms, float lr,
                               float skip_threshold, float exit_threshold) {
    ms->lr = lr;
    ms->skip_threshold = skip_threshold;
    ms->exit_threshold = exit_threshold;
}

void vx_meta_reset(vx_meta_system *ms) {
    for (int i = 0; i < ms->n_layers; i++) {
        vx_meta_predictor_init(ms->layer_preds[i], ms->n_embd);
    }
    vx_meta_predictor_init(ms->attn_pred, ms->n_embd);
    vx_meta_predictor_init(ms->exit_pred, ms->n_embd);
}
