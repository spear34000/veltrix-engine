#ifndef VX_METALEARN_H
#define VX_METALEARN_H

#include "veltrix.h"

vx_meta_predictor *vx_meta_predictor_create(int input_dim);
void vx_meta_predictor_destroy(vx_meta_predictor *mp);
void vx_meta_predictor_init(vx_meta_predictor *mp, int input_dim);

void vx_meta_predict(float *input, int dim, vx_meta_predictor *mp,
                     float *output, float *confidence);
void vx_meta_update(vx_meta_predictor *mp, float *input, float *target,
                    int dim, float lr);

float vx_meta_estimate_confidence(vx_meta_predictor *mp, float *prediction,
                                  float *actual, int dim);

void vx_meta_predict_attention_pattern(float *query, int n_heads, int head_dim,
                                       vx_meta_predictor *mp,
                                       float *predicted_active_heads,
                                       float *confidence);
void vx_meta_predict_exit(float *hidden_state, int dim,
                          vx_meta_predictor *mp,
                          float *exit_score, float *confidence);

typedef struct {
    vx_meta_predictor **layer_preds;   
    vx_meta_predictor *attn_pred;      
    vx_meta_predictor *exit_pred;      
    int n_layers;
    int n_embd;
    float lr;
    float skip_threshold;      
    float exit_threshold;      
} vx_meta_system;

vx_meta_system *vx_meta_system_create(int n_layers, int n_embd);
void vx_meta_system_destroy(vx_meta_system *ms);
void vx_meta_system_set_params(vx_meta_system *ms, float lr,
                               float skip_threshold, float exit_threshold);

vx_decision vx_meta_decide_layer(vx_meta_system *ms, int layer_idx,
                                 float *input, int dim);
vx_decision vx_meta_decide_attention(vx_meta_system *ms, float *query,
                                     int n_heads, int head_dim,
                                     float *predicted_active);
vx_decision vx_meta_decide_exit(vx_meta_system *ms, float *hidden_state,
                                int dim);

void vx_meta_observe_layer(vx_meta_system *ms, int layer_idx,
                           float *input, float *actual_output, int dim);
void vx_meta_observe_attention(vx_meta_system *ms, float *query,
                               int *active_heads, int n_heads);

void vx_meta_reset(vx_meta_system *ms);

#endif
