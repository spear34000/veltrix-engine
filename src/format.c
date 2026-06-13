#include "format.h"
#include "gguf.h"
#include "tensor.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

// ============================================================
// Loader Registry
// ============================================================

#define MAX_LOADERS 16
static const vx_loader *s_loaders[MAX_LOADERS];
static int s_n_loaders = 0;

void vx_loader_register(const vx_loader *loader) {
    if (s_n_loaders < MAX_LOADERS)
        s_loaders[s_n_loaders++] = loader;
}

const vx_loader *vx_loader_for_format(vx_format fmt) {
    for (int i = 0; i < s_n_loaders; i++)
        if (s_loaders[i]->format == fmt)
            return s_loaders[i];
    return NULL;
}

// ============================================================
// Format Detection
// ============================================================

vx_format vx_detect_format(const uint8_t *header, size_t header_size,
                           const char *filename) {
    // GGUF magic: "GGUF" (0x47 0x47 0x55 0x46)
    if (header_size >= 4 &&
        header[0] == 'G' && header[1] == 'G' &&
        header[2] == 'U' && header[3] == 'F')
        return VX_FORMAT_GGUF;

    // Safetensors: uint64 header_len + JSON object
    if (header_size >= 8) {
        uint64_t hlen = 0;
        memcpy(&hlen, header, 8);
        if (hlen > 0 && hlen < 1024 * 1024 && header_size > 8 + hlen) {
            if (header[8] == '{')
                return VX_FORMAT_SAFETENSORS;
        } else if (hlen == 0) {
            // Empty header unlikely; try extension
        }
    }

    // Fallback: use file extension
    if (filename) {
        const char *ext = strrchr(filename, '.');
        if (ext) {
            if (strcasecmp(ext, ".gguf") == 0) return VX_FORMAT_GGUF;
            if (strcasecmp(ext, ".safetensors") == 0) return VX_FORMAT_SAFETENSORS;
            if (strcasecmp(ext, ".onnx") == 0) return VX_FORMAT_ONNX;
        }
    }

    return VX_FORMAT_UNKNOWN;
}

const char *vx_format_name(vx_format fmt) {
    switch (fmt) {
        case VX_FORMAT_GGUF:        return "GGUF";
        case VX_FORMAT_SAFETENSORS: return "Safetensors";
        case VX_FORMAT_ONNX:        return "ONNX";
        case VX_FORMAT_LORA:        return "LoRA";
        default:                    return "Unknown";
    }
}

// ============================================================
// Weight Name Patterns (moved from model.c)
// ============================================================

static const char *tok_embd_fmts[] = {"token_embd.weight", "gpt_neox.embed_in.weight",
    "transformer.wte.weight", "model.embed_tokens.weight", NULL};
static const char *output_weight_fmts[] = {"output.weight", "lm_head.weight",
    "embed_out.weight", NULL};
static const char *output_norm_fmts[] = {"output_norm.weight", "norm.weight",
    "gpt_neox.final_layer_norm.weight", "model.norm.weight",
    "transformer.ln_f.weight", NULL};
static const char *attn_norm_fmts[] = {"blk.%d.attn_norm.weight",
    "blk.%d.rms_norm.weight", "model.layers.%d.input_layernorm.weight",
    "model.layers.%d.rms_norm.weight", "h.%d.ln_1.weight",
    "transformer.h.%d.ln_1.weight", NULL};
static const char *ffn_norm_fmts[] = {"blk.%d.ffn_norm.weight",
    "blk.%d.post_attention_norm.weight",
    "model.layers.%d.post_attention_layernorm.weight",
    "model.layers.%d.rms_norm.weight", "h.%d.ln_2.weight",
    "transformer.h.%d.ln_2.weight", NULL};
static const char *attn_q_fmts[] = {"blk.%d.attn_q.weight",
    "blk.%d.self_attn.q_proj.weight", "model.layers.%d.self_attn.q_proj.weight",
    "h.%d.attn.q_proj.weight", "transformer.h.%d.attn.q_proj.weight", NULL};
static const char *attn_k_fmts[] = {"blk.%d.attn_k.weight",
    "blk.%d.self_attn.k_proj.weight", "model.layers.%d.self_attn.k_proj.weight",
    "h.%d.attn.k_proj.weight", "transformer.h.%d.attn.k_proj.weight", NULL};
static const char *attn_v_fmts[] = {"blk.%d.attn_v.weight",
    "blk.%d.self_attn.v_proj.weight", "model.layers.%d.self_attn.v_proj.weight",
    "h.%d.attn.v_proj.weight", "transformer.h.%d.attn.v_proj.weight", NULL};
static const char *attn_q_bias_fmts[] = {"blk.%d.attn_q.bias",
    "blk.%d.self_attn.q_proj.bias", "model.layers.%d.self_attn.q_proj.bias",
    "h.%d.attn.q_proj.bias", NULL};
static const char *attn_k_bias_fmts[] = {"blk.%d.attn_k.bias",
    "blk.%d.self_attn.k_proj.bias", "model.layers.%d.self_attn.k_proj.bias",
    "h.%d.attn.k_proj.bias", NULL};
static const char *attn_v_bias_fmts[] = {"blk.%d.attn_v.bias",
    "blk.%d.self_attn.v_proj.bias", "model.layers.%d.self_attn.v_proj.bias",
    "h.%d.attn.v_proj.bias", NULL};
static const char *attn_o_fmts[] = {"blk.%d.attn_output.weight",
    "blk.%d.self_attn.o_proj.weight", "model.layers.%d.self_attn.o_proj.weight",
    "h.%d.attn.out_proj.weight", "transformer.h.%d.attn.out_proj.weight", NULL};
static const char *ffn_gate_fmts[] = {"blk.%d.ffn_gate.weight",
    "blk.%d.mlp.gate_proj.weight", "model.layers.%d.mlp.gate_proj.weight",
    "h.%d.mlp.gate_proj.weight", NULL};
static const char *ffn_up_fmts[] = {"blk.%d.ffn_up.weight",
    "blk.%d.mlp.up_proj.weight", "model.layers.%d.mlp.up_proj.weight",
    "h.%d.mlp.up_proj.weight", "model.layers.%d.mlp.fc1.weight", NULL};
static const char *ffn_down_fmts[] = {"blk.%d.ffn_down.weight",
    "blk.%d.mlp.down_proj.weight", "model.layers.%d.mlp.down_proj.weight",
    "h.%d.mlp.down_proj.weight", "model.layers.%d.mlp.fc2.weight", NULL};
static const char *attn_norm_bias_fmts[] = {"blk.%d.attn_norm.bias",
    "blk.%d.rms_norm.bias", "model.layers.%d.input_layernorm.bias",
    "model.layers.%d.rms_norm.bias", "h.%d.ln_1.bias",
    "transformer.h.%d.ln_1.bias", NULL};
static const char *ffn_norm_bias_fmts[] = {"blk.%d.ffn_norm.bias",
    "blk.%d.post_attention_norm.bias",
    "model.layers.%d.post_attention_layernorm.bias",
    "model.layers.%d.rms_norm.bias", "h.%d.ln_2.bias",
    "transformer.h.%d.ln_2.bias", NULL};
static const char *output_norm_bias_fmts[] = {"output_norm.bias", "norm.bias",
    "gpt_neox.final_layer_norm.bias", "model.norm.bias",
    "transformer.ln_f.bias", NULL};
static const char *ffn_gate_bias_fmts[] = {"blk.%d.ffn_gate.bias",
    "blk.%d.mlp.gate_proj.bias", "model.layers.%d.mlp.gate_proj.bias",
    "h.%d.mlp.gate_proj.bias", NULL};
static const char *ffn_up_bias_fmts[] = {"blk.%d.ffn_up.bias",
    "blk.%d.mlp.up_proj.bias", "model.layers.%d.mlp.up_proj.bias",
    "h.%d.mlp.up_proj.bias", "model.layers.%d.mlp.fc1.bias", NULL};
static const char *ffn_down_bias_fmts[] = {"blk.%d.ffn_down.bias",
    "blk.%d.mlp.down_proj.bias", "model.layers.%d.mlp.down_proj.bias",
    "h.%d.mlp.down_proj.bias", "model.layers.%d.mlp.fc2.bias", NULL};

typedef struct {
    const char *name;
    vx_arch arch;
    vx_mlp_type mlp_type;
    vx_norm_type norm_type;
    int rope_partial_dims;
    bool has_q_bias;
    bool has_k_bias;
    bool has_v_bias;
    bool has_ffn_bias;
} arch_entry;

static const arch_entry arch_table[] = {
    {"qwen2",    VX_ARCH_QWEN2,    VX_MLP_SWIGLU, VX_NORM_RMS,   0, false, false, false, false},
    {"qwen2.5",  VX_ARCH_QWEN2,    VX_MLP_SWIGLU, VX_NORM_RMS,   0, false, false, false, false},
    {"llama",    VX_ARCH_LLAMA,    VX_MLP_SWIGLU, VX_NORM_RMS,   0, false, false, false, false},
    {"mistral",  VX_ARCH_MISTRAL,  VX_MLP_SWIGLU, VX_NORM_RMS,   0, false, false, false, false},
    {"gemma",    VX_ARCH_GEMMA,    VX_MLP_GEGLU,  VX_NORM_RMS,   0, true,  true,  true,  false},
    {"gemma2",   VX_ARCH_GEMMA,    VX_MLP_GEGLU,  VX_NORM_RMS,   0, true,  true,  true,  false},
    {"phi2",     VX_ARCH_PHI,      VX_MLP_CLASSIC, VX_NORM_LAYER, 64, true,  true,  true,  true},
    {"phi3",     VX_ARCH_PHI3,     VX_MLP_SWIGLU,  VX_NORM_LAYER, 0, false, false, false, false},
    {"starcoder2",VX_ARCH_STARCODER,VX_MLP_CLASSIC, VX_NORM_RMS,  0, false, false, false, false},
};

void vx_model_apply_arch(vx_model_config *config, const char *arch_name) {
    for (size_t a = 0; a < sizeof(arch_table)/sizeof(arch_table[0]); a++) {
        if (strcmp(arch_name, arch_table[a].name) == 0) {
            config->arch = arch_table[a].arch;
            config->mlp_type = arch_table[a].mlp_type;
            config->norm_type = arch_table[a].norm_type;
            config->rope_partial_dims = arch_table[a].rope_partial_dims;
            config->has_q_bias = arch_table[a].has_q_bias;
            config->has_k_bias = arch_table[a].has_k_bias;
            config->has_v_bias = arch_table[a].has_v_bias;
            config->has_ffn_bias = arch_table[a].has_ffn_bias;
            return;
        }
    }
    // Default: Llama-like
    config->arch = VX_ARCH_LLAMA;
    config->mlp_type = VX_MLP_SWIGLU;
    config->norm_type = VX_NORM_RMS;
}

// ============================================================
// Find tensor by name with multiple format patterns
// ============================================================

static vx_tensor *find_tensor_multi(vx_model *model, int layer,
                                    const char **fmt_list) {
    char name[128];
    for (int f = 0; fmt_list[f] != NULL; f++) {
        if (layer >= 0)
            snprintf(name, sizeof(name), fmt_list[f], layer);
        else
            snprintf(name, sizeof(name), "%s", fmt_list[f]);
        for (int i = 0; i < model->n_tensors; i++) {
            if (model->tensors[i] &&
                strcmp(model->tensors[i]->name, name) == 0)
                return model->tensors[i];
        }
    }
    return NULL;
}

// ============================================================
// Weight Resolution
// ============================================================

vx_error vx_model_resolve_weights(vx_model *model) {
    if (!model || !model->tensors) return VX_ERR_PARAM;

    int n_layers = model->config.n_layers;
    if (n_layers <= 0) return VX_ERR_PARAM;

    model->tok_embd = find_tensor_multi(model, -1, tok_embd_fmts);
    model->output_w = find_tensor_multi(model, -1, output_weight_fmts);
    if (!model->output_w) model->output_w = model->tok_embd;
    model->norm_w = find_tensor_multi(model, -1, output_norm_fmts);

    model->attn_q = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_k = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_v = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_o = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_q_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_k_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_v_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_gate = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_up = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_down = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_norm = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_norm = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->attn_norm_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_norm_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_gate_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_up_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));
    model->ffn_down_bias = calloc((size_t)n_layers, sizeof(vx_tensor *));

    for (int l = 0; l < n_layers; l++) {
        model->attn_q[l] = find_tensor_multi(model, l, attn_q_fmts);
        model->attn_k[l] = find_tensor_multi(model, l, attn_k_fmts);
        model->attn_v[l] = find_tensor_multi(model, l, attn_v_fmts);
        model->attn_o[l] = find_tensor_multi(model, l, attn_o_fmts);
        model->attn_q_bias[l] = find_tensor_multi(model, l, attn_q_bias_fmts);
        model->attn_k_bias[l] = find_tensor_multi(model, l, attn_k_bias_fmts);
        model->attn_v_bias[l] = find_tensor_multi(model, l, attn_v_bias_fmts);
        model->ffn_gate[l] = find_tensor_multi(model, l, ffn_gate_fmts);
        model->ffn_up[l] = find_tensor_multi(model, l, ffn_up_fmts);
        model->ffn_down[l] = find_tensor_multi(model, l, ffn_down_fmts);
        model->attn_norm[l] = find_tensor_multi(model, l, attn_norm_fmts);
        model->ffn_norm[l] = find_tensor_multi(model, l, ffn_norm_fmts);
        model->attn_norm_bias[l] = find_tensor_multi(model, l, attn_norm_bias_fmts);
        model->ffn_norm_bias[l] = find_tensor_multi(model, l, ffn_norm_bias_fmts);
        model->ffn_gate_bias[l] = find_tensor_multi(model, l, ffn_gate_bias_fmts);
        model->ffn_up_bias[l] = find_tensor_multi(model, l, ffn_up_bias_fmts);
        model->ffn_down_bias[l] = find_tensor_multi(model, l, ffn_down_bias_fmts);
    }

    model->norm_bias_w = find_tensor_multi(model, -1, output_norm_bias_fmts);
    return VX_OK;
}

// ============================================================
// Auto-detect dimensions from tensor shapes
// ============================================================

void vx_model_auto_detect_dims(vx_model *model) {
    if (!model) return;

    // Detect n_ff from ffn_gate or ffn_down tensor
    if (model->config.n_ff == 0) {
        for (int i = 0; i < model->n_tensors; i++) {
            vx_tensor *t = model->tensors[i];
            if (!t) continue;
            if (strstr(t->name, "ffn_gate") || strstr(t->name, "gate_proj") ||
                strstr(t->name, ".fc1.")) {
                model->config.n_ff = (int)t->ne[1];
                break;
            }
            if (strstr(t->name, "ffn_down") || strstr(t->name, "down_proj") ||
                strstr(t->name, ".fc2.")) {
                model->config.n_ff = (int)t->ne[0];
                break;
            }
        }
    }

    // Detect vocab_size from token_embd
    if (model->config.n_vocab <= 32000) {
        for (int i = 0; i < model->n_tensors; i++) {
            vx_tensor *t = model->tensors[i];
            if (!t) continue;
            if (strstr(t->name, "token_embd") || strstr(t->name, "tok_embeddings") ||
                strstr(t->name, "embed_tokens")) {
                int vocab = (int)t->ne[1];
                if (vocab > model->config.n_vocab)
                    model->config.n_vocab = vocab;
                break;
            }
        }
    }
}

// ============================================================
// Built-in loader registrations
// ============================================================

// Declared in gguf.h
vx_error gguf_loader_load(const char *path, vx_model *model);

const vx_loader vx_loader_gguf = {
    .name = "gguf",
    .format = VX_FORMAT_GGUF,
    .load = gguf_loader_load,
};

// Safetensors loader (in safetensors.c)
vx_error st_loader_load(const char *path, vx_model *model);

const vx_loader vx_loader_safetensors = {
    .name = "safetensors",
    .format = VX_FORMAT_SAFETENSORS,
    .load = st_loader_load,
};

// ONNX loader (in onnx.c)
vx_error onnx_loader_load(const char *path, vx_model *model);

const vx_loader vx_loader_onnx = {
    .name = "onnx",
    .format = VX_FORMAT_ONNX,
    .load = onnx_loader_load,
};

// Called once from vx_model_create to init loaders
void vx_format_init(void) {
    static bool initialized = false;
    if (!initialized) {
        vx_loader_register(&vx_loader_gguf);
        vx_loader_register(&vx_loader_safetensors);
        vx_loader_register(&vx_loader_onnx);
        initialized = true;
    }
}
