#include "format.h"
#include "tensor.h"
#include "gguf.h"
#include "quantize.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// ============================================================
// Minimal Protobuf Wire Format Reader
// ============================================================

// Wire types
#define PB_WIRE_VARINT 0
#define PB_WIRE_I64    1
#define PB_WIRE_LEN    2
#define PB_WIRE_I32    5

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t offset;
} pb_reader;

static void pb_init(pb_reader *r, const void *buf, size_t size) {
    r->data = (const uint8_t *)buf;
    r->size = size;
    r->offset = 0;
}

static int pb_ok(pb_reader *r) {
    return r->offset < r->size;
}

static int pb_read_varint(pb_reader *r, uint64_t *out) {
    uint64_t val = 0;
    int shift = 0;
    while (pb_ok(r)) {
        uint8_t byte = r->data[r->offset++];
        val |= (uint64_t)(byte & 0x7F) << shift;
        shift += 7;
        if (!(byte & 0x80)) {
            *out = val;
            return 0;
        }
    }
    return -1;
}

static int pb_read_tag(pb_reader *r, int *field_num, int *wire_type) {
    uint64_t tag;
    if (pb_read_varint(r, &tag) != 0) return -1;
    *field_num = (int)(tag >> 3);
    *wire_type = (int)(tag & 7);
    return 0;
}

static int pb_skip_field(pb_reader *r, int wire_type) {
    switch (wire_type) {
        case PB_WIRE_VARINT: {
            uint64_t tmp;
            return pb_read_varint(r, &tmp);
        }
        case PB_WIRE_I64:
            if (r->offset + 8 > r->size) return -1;
            r->offset += 8;
            return 0;
        case PB_WIRE_LEN: {
            uint64_t len;
            if (pb_read_varint(r, &len) != 0) return -1;
            if (r->offset + len > r->size) return -1;
            r->offset += (size_t)len;
            return 0;
        }
        case PB_WIRE_I32:
            if (r->offset + 4 > r->size) return -1;
            r->offset += 4;
            return 0;
        default:
            return -1;
    }
}

static int pb_read_len(pb_reader *r, const uint8_t **data_out, size_t *len_out) {
    uint64_t len;
    if (pb_read_varint(r, &len) != 0) return -1;
    if (r->offset + len > r->size) return -1;
    *data_out = r->data + r->offset;
    *len_out = (size_t)len;
    r->offset += (size_t)len;
    return 0;
}

// ============================================================
// ONNX Type Constants
// ============================================================

#define ONNX_TENSOR_FLOAT    1
#define ONNX_TENSOR_FLOAT16  10
#define ONNX_TENSOR_BFLOAT16 16
#define ONNX_TENSOR_INT8     3
#define ONNX_TENSOR_INT16    5
#define ONNX_TENSOR_INT32    6
#define ONNX_TENSOR_INT64    7
#define ONNX_TENSOR_DOUBLE   11
#define ONNX_TENSOR_UINT8    2

// ============================================================
// ONNX Model Parser
// ============================================================

typedef struct {
    char name[256];
    int data_type;
    int64_t dims[8];
    int n_dims;
    const uint8_t *raw_data;
    size_t raw_size;
    // Packed data alternatives
    const float *float_data;
    int n_floats;
    const int64_t *int64_data;
    int n_int64s;
    const uint8_t *external_data;
} onnx_tensor;

// Parse TensorProto from a sub-reader
static int parse_tensor_proto(pb_reader *r, onnx_tensor *t) {
    memset(t, 0, sizeof(*t));
    const uint8_t *msg_start = r->data + r->offset;
    size_t msg_len;
    if (pb_read_len(r, &msg_start, &msg_len) != 0) return -1;

    pb_reader sub;
    pb_init(&sub, msg_start, msg_len);

    // Pre-scan: if there's a float_data or int64_data field, we need to read raw_data first
    // For simplicity: iterate twice — first pass for name and raw_data size,
    // second pass for actual reading. Actually, let's do it in one pass.

    // Read each field in the TensorProto
    while (pb_ok(&sub)) {
        int field_num, wire_type;
        if (pb_read_tag(&sub, &field_num, &wire_type) != 0) break;

        switch (field_num) {
            case 1: { // dims (repeated int64, packed varint)
                if (wire_type == PB_WIRE_LEN) {
                    const uint8_t *packed;
                    size_t plen;
                    if (pb_read_len(&sub, &packed, &plen) != 0) break;
                    pb_reader pr;
                    pb_init(&pr, packed, plen);
                    while (pb_ok(&pr)) {
                        uint64_t v;
                        if (pb_read_varint(&pr, &v) != 0) break;
                        if (t->n_dims < 8) t->dims[t->n_dims++] = (int64_t)v;
                    }
                } else if (wire_type == PB_WIRE_VARINT) {
                    uint64_t v;
                    if (pb_read_varint(&sub, &v) != 0) break;
                    if (t->n_dims < 8) t->dims[t->n_dims++] = (int64_t)v;
                }
                break;
            }
            case 2: { // data_type (int32, varint)
                uint64_t v;
                if (pb_read_varint(&sub, &v) != 0) break;
                t->data_type = (int)v;
                break;
            }
            case 4: { // float_data (repeated float, packed)
                const uint8_t *d;
                size_t dlen;
                if (pb_read_len(&sub, &d, &dlen) != 0) break;
                t->float_data = (const float *)d;
                t->n_floats = (int)(dlen / 4);
                break;
            }
            case 5: { // int32_data (repeated int32, packed)
                const uint8_t *d;
                size_t dlen;
                if (pb_read_len(&sub, &d, &dlen) != 0) break;
                break;
            }
            case 7: { // int64_data (repeated int64, packed)
                const uint8_t *d;
                size_t dlen;
                if (pb_read_len(&sub, &d, &dlen) != 0) break;
                t->int64_data = (const int64_t *)d;
                t->n_int64s = (int)(dlen / 8);
                break;
            }
            case 8: { // name (string, len)
                const uint8_t *s;
                size_t slen;
                if (pb_read_len(&sub, &s, &slen) != 0) break;
                int cpy = (int)slen;
                if (cpy > 255) cpy = 255;
                memcpy(t->name, s, (size_t)cpy);
                t->name[cpy] = 0;
                break;
            }
            case 9: { // raw_data (bytes)
                if (pb_read_len(&sub, &t->raw_data, &t->raw_size) != 0) break;
                break;
            }
            case 10: { // double_data (repeated double, packed)
                if (pb_skip_field(&sub, wire_type) != 0) break;
                break;
            }
            default:
                if (pb_skip_field(&sub, wire_type) != 0) break;
                break;
        }
    }
    return 0;
}

// Parse GraphProto from sub-reader, call callback for each initializer
// Returns number of initializers found
static int parse_graph_initializers(pb_reader *r, onnx_tensor *tensors, int max_tensors) {
    const uint8_t *graph_data;
    size_t graph_len;
    if (pb_read_len(r, &graph_data, &graph_len) != 0) return 0;

    pb_reader graph;
    pb_init(&graph, graph_data, graph_len);

    int n_init = 0;

    while (pb_ok(&graph)) {
        int field_num, wire_type;
        if (pb_read_tag(&graph, &field_num, &wire_type) != 0) break;

        if (field_num == 5 && wire_type == PB_WIRE_LEN) { // initializer[]
            if (n_init < max_tensors) {
                if (parse_tensor_proto(&graph, &tensors[n_init]) == 0) {
                    n_init++;
                }
            } else {
                if (pb_skip_field(&graph, wire_type) != 0) break;
            }
        } else {
            if (pb_skip_field(&graph, wire_type) != 0) break;
        }
    }

    return n_init;
}

// ============================================================
// ONNX Data Conversion
// ============================================================

static size_t onnx_type_size(int onnx_type) {
    switch (onnx_type) {
        case ONNX_TENSOR_FLOAT:    return 4;
        case ONNX_TENSOR_FLOAT16:  return 2;
        case ONNX_TENSOR_BFLOAT16: return 2;
        case ONNX_TENSOR_DOUBLE:   return 8;
        case ONNX_TENSOR_INT8:
        case ONNX_TENSOR_UINT8:    return 1;
        case ONNX_TENSOR_INT16:    return 2;
        case ONNX_TENSOR_INT32:    return 4;
        case ONNX_TENSOR_INT64:    return 8;
        default:                   return 4;
    }
}

static int read_onnx_tensor_data(const onnx_tensor *t, float *f32_out, int n_elems) {
    if (t->raw_data && t->raw_size > 0) {
        // raw_data is in the tensor's native type
        size_t elem_size = onnx_type_size(t->data_type);
        int n_raw = (int)(t->raw_size / elem_size);
        int n_copy = n_raw < n_elems ? n_raw : n_elems;

        switch (t->data_type) {
            case ONNX_TENSOR_FLOAT:
                memcpy(f32_out, t->raw_data, (size_t)n_copy * 4);
                break;
            case ONNX_TENSOR_FLOAT16: {
                const uint16_t *src = (const uint16_t *)t->raw_data;
                for (int i = 0; i < n_copy; i++)
                    f32_out[i] = vx_fp16_to_fp32(src[i]);
                break;
            }
            case ONNX_TENSOR_BFLOAT16: {
                const uint16_t *src = (const uint16_t *)t->raw_data;
                for (int i = 0; i < n_copy; i++) {
                    uint32_t bits = (uint32_t)src[i] << 16;
                    memcpy(&f32_out[i], &bits, 4);
                }
                break;
            }
            case ONNX_TENSOR_DOUBLE: {
                const double *src = (const double *)t->raw_data;
                for (int i = 0; i < n_copy; i++)
                    f32_out[i] = (float)src[i];
                break;
            }
            case ONNX_TENSOR_INT64: {
                const int64_t *src = (const int64_t *)t->raw_data;
                for (int i = 0; i < n_copy; i++)
                    f32_out[i] = (float)src[i];
                break;
            }
            case ONNX_TENSOR_INT32: {
                const int32_t *src = (const int32_t *)t->raw_data;
                for (int i = 0; i < n_copy; i++)
                    f32_out[i] = (float)src[i];
                break;
            }
            default:
                memset(f32_out, 0, (size_t)n_copy * 4);
                break;
        }

        // Pad remaining with zeros
        for (int i = n_copy; i < n_elems; i++) f32_out[i] = 0;
        return n_copy;
    }

    if (t->float_data && t->n_floats > 0) {
        int n_copy = t->n_floats < n_elems ? t->n_floats : n_elems;
        memcpy(f32_out, t->float_data, (size_t)n_copy * 4);
        for (int i = n_copy; i < n_elems; i++) f32_out[i] = 0;
        return n_copy;
    }

    if (t->int64_data && t->n_int64s > 0) {
        int n_copy = t->n_int64s < n_elems ? t->n_int64s : n_elems;
        for (int i = 0; i < n_copy; i++)
            f32_out[i] = (float)t->int64_data[i];
        for (int i = n_copy; i < n_elems; i++) f32_out[i] = 0;
        return n_copy;
    }

    memset(f32_out, 0, (size_t)n_elems * 4);
    return 0;
}

// ============================================================
// Minimal JSON Parser (for config.json)
// ============================================================

static const char *js_skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static const char *js_skip_string(const char *p) {
    if (*p != '"') return NULL;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') { p++; if (*p) p++; }
        else p++;
    }
    if (*p == '"') p++;
    return p;
}

static const char *js_skip_value(const char *p) {
    p = js_skip_ws(p);
    if (!p || !*p) return NULL;
    if (*p == '"') return js_skip_string(p);
    if (*p == '{' || *p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '"') { p = js_skip_string(p); if (!p) return NULL; }
            else if (*p == '{' || *p == '[') { depth++; p++; }
            else if (*p == '}' || *p == ']') { depth--; p++; }
            else p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']' && !(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        p++;
    return p;
}

static const char *js_find_key(const char *json, const char *key) {
    const char *p = js_skip_ws(json);
    if (*p != '{') return NULL;
    p++;
    int key_len = (int)strlen(key);
    while (*p && *p != '}') {
        p = js_skip_ws(p);
        if (*p != '"') break;
        const char *ks = p + 1;
        const char *ke = ks;
        while (*ke && *ke != '"') {
            if (*ke == '\\') ke++;
            if (*ke) ke++;
        }
        int klen = (int)(ke - ks);
        p = ke;
        if (*p == '"') p++;
        p = js_skip_ws(p);
        if (*p != ':') break;
        p++;
        p = js_skip_ws(p);
        if (klen == key_len && memcmp(ks, key, (size_t)klen) == 0)
            return p;
        p = js_skip_value(p);
        if (!p) return NULL;
        p = js_skip_ws(p);
        if (*p == ',') p++;
    }
    return NULL;
}

static const char *js_read_string(const char *p, int *out_len) {
    p = js_skip_ws(p);
    if (!p || *p != '"') return NULL;
    p++;
    const char *start = p;
    while (*p && *p != '"') {
        if (*p == '\\') p++;
        if (*p) p++;
    }
    *out_len = (int)(p - start);
    return start;
}

static double js_read_number(const char *p, int *out_len) {
    p = js_skip_ws(p);
    const char *start = p;
    if (*p == '-') p++;
    while (*p && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-'))
        p++;
    *out_len = (int)(p - start);
    char buf[64];
    int cpy = *out_len < 63 ? *out_len : 63;
    memcpy(buf, start, (size_t)cpy);
    buf[cpy] = 0;
    return atof(buf);
}

// ============================================================
// HF Config Reader (config.json → vx_model_config)
// ============================================================

typedef struct { const char *name; vx_arch arch; } onnx_hf_arch;

static const onnx_hf_arch onnx_hf_archs[] = {
    {"qwen2",       VX_ARCH_QWEN2},
    {"qwen2.5",     VX_ARCH_QWEN2},
    {"qwen3",       VX_ARCH_QWEN2},
    {"qwen3.5",     VX_ARCH_QWEN2},
    {"qwen3.6",     VX_ARCH_QWEN2},
    {"llama",       VX_ARCH_LLAMA},
    {"mistral",     VX_ARCH_MISTRAL},
    {"mistralnemo", VX_ARCH_MISTRAL},
    {"gemma",       VX_ARCH_GEMMA},
    {"gemma2",      VX_ARCH_GEMMA},
    {"gemma3",      VX_ARCH_GEMMA},
    {"gemma4",      VX_ARCH_GEMMA},
    {"phi",         VX_ARCH_PHI},
    {"phi3",        VX_ARCH_PHI3},
    {"starcoder2",  VX_ARCH_STARCODER},
    {"RefinedWebModel", VX_ARCH_LLAMA},
    {"falcon",      VX_ARCH_LLAMA},
    {"deepseek_v2", VX_ARCH_LLAMA},
    {"deepseek_v3", VX_ARCH_LLAMA},
};

static void onnx_get_dir(const char *path, char *dir, int dir_size) {
    const char *sep = strrchr(path, '/');
#ifdef _WIN32
    const char *sep2 = strrchr(path, '\\');
    if (!sep || (sep2 && sep2 > sep)) sep = sep2;
#else
    (void)0;
#endif
    if (sep) {
        int cpy = (int)(sep - path);
        if (cpy > dir_size - 1) cpy = dir_size - 1;
        memcpy(dir, path, (size_t)cpy);
        dir[cpy] = 0;
    } else {
        dir[0] = '.'; dir[1] = 0;
    }
}

static vx_error onnx_load_config(const char *model_path, vx_model_config *cfg) {
    char dir[512];
    onnx_get_dir(model_path, dir, sizeof(dir));

    char cfg_path[1024];
    snprintf(cfg_path, sizeof(cfg_path), "%s%cconfig.json", dir,
#ifdef _WIN32
        '\\'
#else
        '/'
#endif
    );

    FILE *fp = fopen(cfg_path, "rb");
    if (!fp) {
        fprintf(stderr, "Warning: config.json not found at %s (using defaults)\n", cfg_path);
        return VX_OK;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *json = malloc((size_t)fsize + 1);
    if (!json) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(json, 1, (size_t)fsize, fp) != (size_t)fsize) {
        free(json); fclose(fp); return VX_ERR_PARAM;
    }
    fclose(fp);
    json[fsize] = 0;

    const char *vp = js_find_key(json, "model_type");
    if (vp) {
        int slen = 0;
        const char *s = js_read_string(vp, &slen);
        if (s && slen > 0 && slen < 64) {
            char mtype[64];
            memcpy(mtype, s, (size_t)slen);
            mtype[slen] = 0;
            for (size_t i = 0; i < sizeof(onnx_hf_archs)/sizeof(onnx_hf_archs[0]); i++) {
                if (strcmp(mtype, onnx_hf_archs[i].name) == 0) {
                    cfg->arch = onnx_hf_archs[i].arch;
                    break;
                }
            }
            vx_model_apply_arch(cfg, mtype);
        }
    }

    vp = js_find_key(json, "hidden_size");
    if (vp) { int len; cfg->n_embd = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "num_hidden_layers");
    if (!vp) vp = js_find_key(json, "n_layer");
    if (vp) { int len; cfg->n_layers = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "num_attention_heads");
    if (!vp) vp = js_find_key(json, "n_head");
    if (vp) { int len; cfg->n_heads = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "num_key_value_heads");
    if (!vp) vp = js_find_key(json, "n_kv_head");
    if (vp) { int len; cfg->n_kv_heads = (int)js_read_number(vp, &len); }
    if (cfg->n_kv_heads == 0) cfg->n_kv_heads = cfg->n_heads;

    vp = js_find_key(json, "intermediate_size");
    if (!vp) vp = js_find_key(json, "n_inner");
    if (vp) { int len; cfg->n_ff = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "vocab_size");
    if (vp) { int len; cfg->n_vocab = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "rope_theta");
    if (!vp) vp = js_find_key(json, "rope.freq_base");
    if (vp) { int len; cfg->rope_theta = (float)js_read_number(vp, &len); }

    vp = js_find_key(json, "max_position_embeddings");
    if (vp) { int len; cfg->n_ctx = (int)js_read_number(vp, &len); }

    vp = js_find_key(json, "head_dim");
    if (vp) { int len; cfg->n_head_dim = (int)js_read_number(vp, &len); }

    if (cfg->n_head_dim == 0 && cfg->n_heads > 0)
        cfg->n_head_dim = cfg->n_embd / cfg->n_heads;
    if (cfg->n_ctx <= 0) cfg->n_ctx = 4096;
    if (cfg->rope_theta <= 0) cfg->rope_theta = 10000.0f;

    free(json);
    return VX_OK;
}

// ============================================================
// ONNX Model Loader (implements vx_loader interface)
// ============================================================

vx_error onnx_loader_load(const char *path, vx_model *model) {
    // Open and read the ONNX file
    FILE *fp = fopen(path, "rb");
    if (!fp) return VX_ERR_FILE;

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *buf = malloc(size);
    if (!buf) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(buf, 1, size, fp) != size) { free(buf); fclose(fp); return VX_ERR_PARAM; }
    fclose(fp);

    // Parse the protobuf top-level fields to find the GraphProto
    pb_reader main;
    pb_init(&main, buf, size);

    onnx_tensor tensors[4096];
    int n_tensors = 0;

    while (pb_ok(&main)) {
        int field_num, wire_type;
        if (pb_read_tag(&main, &field_num, &wire_type) != 0) break;

        if (field_num == 7 && wire_type == PB_WIRE_LEN) { // graph = 7
            n_tensors = parse_graph_initializers(&main, tensors, 4096);
            break;
        } else {
            if (pb_skip_field(&main, wire_type) != 0) break;
        }
    }

    if (n_tensors <= 0) {
        fprintf(stderr, "No initializer tensors found in ONNX model\n");
        free(buf);
        return VX_ERR_FORMAT;
    }

    // Load config.json
    vx_model_config *cfg = &model->config;
    memset(cfg, 0, sizeof(*cfg));
    onnx_load_config(path, cfg);

    // Apply arch if not set
    if (cfg->arch == 0 && cfg->mlp_type == 0) {
        vx_model_apply_arch(cfg, "llama");
    }

    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 128) n_threads = 128;
    omp_set_num_threads(n_threads);
    cfg->n_threads = n_threads;

    // Allocate tensors
    model->n_tensors = n_tensors;
    model->tensors = calloc((size_t)n_tensors, sizeof(vx_tensor *));
    if (!model->tensors) { free(buf); return VX_ERR_MEMORY; }

    for (int i = 0; i < n_tensors; i++) {
        onnx_tensor *ot = &tensors[i];
        vx_tensor *t = calloc(1, sizeof(vx_tensor));
        if (!t) continue;

        strncpy(t->name, ot->name, VX_MAX_NAME - 1);
        t->name[VX_MAX_NAME - 1] = 0;

        // Convert ONNX shape to Veltrix shape (reverse dim order)
        // ONNX dims: [d0, d1, ...] where d0 = outer, d_last = inner
        // Veltrix ne[0] = fastest-changing (inner), ne[last] = outer
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        int n_dims = ot->n_dims;
        if (n_dims > 4) n_dims = 4;
        for (int d = 0; d < n_dims; d++)
            t->ne[n_dims - 1 - d] = (int)ot->dims[d];

        int n_elems = 1;
        for (int d = 0; d < n_dims; d++) n_elems *= (int)ot->dims[d];
        if (n_elems <= 0) n_elems = 1;

        t->type = VX_TYPE_F32;
        t->nbytes = (size_t)n_elems * 4;
        t->data = malloc(t->nbytes);
        if (!t->data) { free(t); continue; }
        t->owned = true;

        read_onnx_tensor_data(ot, (float *)t->data, n_elems);

        model->tensors[i] = t;
    }

    free(buf);
    return VX_OK;
}
