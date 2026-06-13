#ifndef VX_SCHEDULER_H
#define VX_SCHEDULER_H

#include "veltrix.h"

typedef struct {
    float free_energy;
    float prediction_error;
    float computation_cost;
    float precision_cost;
    float total_score;
} vx_fe_metrics;

typedef struct {
    int n_layers;
    float *layer_fe;       
    float *layer_pe;       
    float *layer_skip_rate;
    float total_free_energy;
    int n_tokens_processed;
    int n_layers_skipped;
    int n_layers_exact;
    int n_attns_skipped;
} vx_fe_state;

typedef enum {
    VX_POLICY_MIN_FE = 0,     
    VX_POLICY_TARGET_SKIP,    
    VX_POLICY_ADAPTIVE,       
} vx_policy_type;

typedef struct {
    vx_policy_type policy;
    float target_skip_rate;
    float fe_temperature;      
    float lr;                  
    float precision_init;      
    int window_size;           
    float *fe_history;         
    int fe_history_idx;
    float avg_fe;              
} vx_scheduler_config;

vx_fe_state *vx_fe_state_create(int n_layers);
void vx_fe_state_destroy(vx_fe_state *state);
void vx_fe_state_reset(vx_fe_state *state);

void vx_scheduler_config_default(vx_scheduler_config *cfg);
void vx_scheduler_config_set_policy(vx_scheduler_config *cfg, vx_policy_type policy);
void vx_scheduler_config_set_target_skip(vx_scheduler_config *cfg, float rate);

float vx_compute_free_energy(const float *prediction, const float *actual,
                             int dim, float precision);
float vx_compute_prediction_error(const float *pred, const float *actual, int dim);

vx_compute_level vx_scheduler_decide(vx_fe_state *state,
                                     float confidence,
                                     float free_energy,
                                     const vx_scheduler_config *cfg);

void vx_scheduler_update(vx_fe_state *state, int layer_idx,
                         vx_compute_level level,
                         float free_energy, float prediction_error);

void vx_scheduler_adapt(vx_scheduler_config *cfg, vx_fe_state *state);

float vx_fe_expected_savings(const vx_scheduler_config *cfg, float confidence);

#endif
