#ifndef VELTRIX_H
#define VELTRIX_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define VX_MAGIC "VX01"
#define VX_MAX_NAME 64
#define VX_MAX_LAYERS 256
#define VX_MAX_DIM 8192
#define VX_HEAD_DIM 128
#define VX_META_HDIM 128

typedef enum {
    VX_OK = 0,
    VX_ERR_FILE = -1,
    VX_ERR_FORMAT = -2,
    VX_ERR_MEMORY = -3,
    VX_ERR_PARAM = -4,
    VX_ERR_GGUF = -5,
    VX_ERR_NOMODEL = -6,
    VX_ERR_UNSUPPORTED = -7,
} vx_error;

typedef enum {
    VX_TYPE_F32 = 0,
    VX_TYPE_F16 = 1,
    VX_TYPE_Q4_0 = 2,
    VX_TYPE_Q4_1 = 3,
    VX_TYPE_Q5_0 = 6,
    VX_TYPE_Q5_1 = 7,
    VX_TYPE_Q8_0 = 8,
    VX_TYPE_Q8_1 = 9,
    VX_TYPE_Q2_K = 10,
    VX_TYPE_Q3_K = 11,
    VX_TYPE_Q4_K = 12,
    VX_TYPE_Q5_K = 13,
    VX_TYPE_Q6_K = 14,
    VX_TYPE_Q8_K = 15,
    VX_TYPE_IQ1_S = 20,
    VX_TYPE_IQ2_XXS = 21,
    VX_TYPE_IQ2_XS = 22,
    VX_TYPE_IQ3_XXS = 26,
    VX_TYPE_IQ1_M = 31,
    VX_TYPE_BF16 = 32,
} vx_type;

typedef struct {
    int32_t ne[4];
    size_t nbytes;
    vx_type type;
    uint8_t *data;
    bool owned;
    char name[VX_MAX_NAME];
} vx_tensor;

typedef enum {
    VX_ARCH_QWEN2 = 0,
    VX_ARCH_LLAMA,
    VX_ARCH_MISTRAL,
    VX_ARCH_GEMMA,
    VX_ARCH_PHI,
    VX_ARCH_PHI3,
    VX_ARCH_STARCODER,
    VX_ARCH_COUNT,
} vx_arch;

typedef enum {
    VX_MLP_SWIGLU = 0,
    VX_MLP_GEGLU,
    VX_MLP_CLASSIC,
} vx_mlp_type;

typedef enum {
    VX_NORM_RMS = 0,
    VX_NORM_LAYER,
} vx_norm_type;

typedef struct {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int n_embd;
    int n_head_dim;
    int n_ff;
    int n_vocab;
    int n_ctx;
    float rope_theta;
    float rope_freq_base;
    int n_threads;
    vx_arch arch;
    vx_mlp_type mlp_type;
    vx_norm_type norm_type;
    int rope_partial_dims;
    bool has_q_bias;
    bool has_k_bias;
    bool has_v_bias;
    bool has_post_attn_norm;
    bool has_post_ffn_norm;
    bool has_ffn_bias;
} vx_model_config;

typedef struct {
    float *data;
    int n;
    float mean;
    float std;
} vx_meta_pred_state;

typedef struct {
    float w1[VX_META_HDIM * VX_MAX_DIM]; 
    float b1[VX_META_HDIM];
    float w2[VX_MAX_DIM * VX_META_HDIM];
    float b2[VX_MAX_DIM];
    float running_mean[VX_MAX_DIM];
    float running_var[VX_MAX_DIM];
    int update_count;
    float precision;       
} vx_meta_predictor;

typedef struct {
    float confidence;
    float free_energy;
    int exit_layer;
    bool skip_compute;
    float predicted_norm;
} vx_decision;

typedef enum {
    VX_COMPUTE_EXACT = 0,    
    VX_COMPUTE_ATTN_ONLY,    
    VX_COMPUTE_SKIP,         
} vx_compute_level;

typedef struct vx_model {
    vx_model_config config;
    vx_tensor **tensors;
    int n_tensors;
    void *tensor_data;
    vx_tensor *tok_embd;
    vx_tensor **attn_q;
    vx_tensor **attn_k;
    vx_tensor **attn_v;
    vx_tensor **attn_o;
    vx_tensor **attn_q_bias;
    vx_tensor **attn_k_bias;
    vx_tensor **attn_v_bias;
    vx_tensor **ffn_gate;
    vx_tensor **ffn_up;
    vx_tensor **ffn_down;
    vx_tensor **ffn_gate_bias;
    vx_tensor **ffn_up_bias;
    vx_tensor **ffn_down_bias;
    vx_tensor **attn_norm;
    vx_tensor **ffn_norm;
    vx_tensor **attn_norm_bias;
    vx_tensor **ffn_norm_bias;
    vx_tensor *norm_w;
    vx_tensor *norm_bias_w;
    vx_tensor *output_w;
    vx_meta_predictor **layer_predictors;
    vx_meta_predictor *attn_pattern_predictor;
    vx_meta_predictor *token_exit_predictor;
    float *k_cache;
    float *v_cache;
    int cache_len;
    float *scratch;
    size_t scratch_size;
    float *dequant_tmp;
    size_t dequant_tmp_size;
    struct vx_tokenizer *tokenizer;
} vx_model;

vx_error vx_model_create(const char *path, vx_model **model);
void vx_model_destroy(vx_model *model);
vx_error vx_model_forward(vx_model *model, const int *tokens, int n_tokens, float *logits);
void vx_model_kv_truncate(vx_model *model, int length);
void vx_set_compute_level(vx_compute_level level);
void vx_enable_metalearn(bool enable);
void vx_set_n_threads(int n);
void vx_set_forward_timing_enabled(int enabled);
void vx_print_forward_timing(void);

#endif
