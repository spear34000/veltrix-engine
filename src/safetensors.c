#include "format.h"
#include "gguf.h"
#include "tensor.h"
#include "quantize.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

// ============================================================
// Minimal JSON Parser (safetensors header only)
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
    // number, true, false, null
    while (*p && *p != ',' && *p != '}' && *p != ']' && !(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        p++;
    return p;
}

// Find value for a given key at the top level of a JSON object
static const char *js_find_key(const char *json, const char *key) {
    const char *p = js_skip_ws(json);
    if (*p != '{') return NULL;
    p++;
    int key_len = (int)strlen(key);
    while (*p && *p != '}') {
        p = js_skip_ws(p);
        if (*p != '"') break;
        // Read key
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

// Read a string value from JSON (expects '"..."')
// Returns pointer to start of string content, sets *out_len
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

// Read a number from JSON, return as double
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

// Read an array of numbers from JSON: [n1, n2, ...]
// Returns number of values read
static int js_read_int_array(const char *p, int *vals, int max_vals) {
    p = js_skip_ws(p);
    if (!p || *p != '[') return 0;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max_vals) {
        p = js_skip_ws(p);
        if (*p == ']') break;
        int num_len = 0;
        double d = js_read_number(p, &num_len);
        if (num_len == 0) break;
        vals[n++] = (int)d;
        p += num_len;
        p = js_skip_ws(p);
        if (*p == ',') p++;
    }
    return n;
}

static int js_read_uint64_array(const char *p, uint64_t *vals, int max_vals) {
    p = js_skip_ws(p);
    if (!p || *p != '[') return 0;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max_vals) {
        p = js_skip_ws(p);
        if (*p == ']') break;
        int num_len = 0;
        double d = js_read_number(p, &num_len);
        if (num_len == 0) break;
        vals[n++] = (uint64_t)d;
        p += num_len;
        p = js_skip_ws(p);
        if (*p == ',') p++;
    }
    return n;
}

// ============================================================
// Safetensors Header Parsing
// ============================================================

typedef struct {
    char name[256];
    char dtype[16];
    uint32_t shape[4];
    int n_dims;
    uint64_t data_start;
    uint64_t data_end;
} st_tensor_entry;

// Parse the full safetensors header JSON, fill entries array
// Returns number of tensor entries parsed
static int st_parse_header(const char *json, size_t json_len,
                           st_tensor_entry *entries, int max_entries) {
    (void)json_len;
    const char *p = js_skip_ws(json);
    if (!p || *p != '{') return -1;
    p++;
    int n = 0;

    while (*p && *p != '}' && n < max_entries) {
        p = js_skip_ws(p);
        if (*p != '"') break;

        // Read tensor name
        const char *name_start = p + 1;
        const char *name_end = name_start;
        while (*name_end && *name_end != '"') {
            if (*name_end == '\\') name_end++;
            if (*name_end) name_end++;
        }
        int name_len = (int)(name_end - name_start);
        if (name_len <= 0) break;
        p = name_end;
        if (*p == '"') p++;

        // Skip __metadata__
        if (name_len == 11 && memcmp(name_start, "__metadata__", 11) == 0) {
            p = js_skip_ws(p);
            if (*p == ':') p++;
            p = js_skip_value(p);
            if (!p) break;
            p = js_skip_ws(p);
            if (*p == ',') p++;
            continue;
        }

        p = js_skip_ws(p);
        if (*p != ':') break;
        p++;
        p = js_skip_ws(p);
        if (*p != '{') break;
        p++;

        // Parse tensor entry: { "dtype": "...", "shape": [...], "data_offsets": [...] }
        st_tensor_entry *e = &entries[n];
        memset(e, 0, sizeof(*e));

        int cpy = name_len < 255 ? name_len : 255;
        memcpy(e->name, name_start, (size_t)cpy);
        e->name[cpy] = 0;

        // Read fields inside the object
        while (*p && *p != '}') {
            p = js_skip_ws(p);
            if (*p != '"') break;
            const char *fstart = p + 1;
            const char *fend = fstart;
            while (*fend && *fend != '"') {
                if (*fend == '\\') fend++;
                if (*fend) fend++;
            }
            int flen = (int)(fend - fstart);
            p = fend;
            if (*p == '"') p++;
            p = js_skip_ws(p);
            if (*p != ':') break;
            p++;
            p = js_skip_ws(p);

            if (flen == 5 && memcmp(fstart, "dtype", 5) == 0) {
                int dlen = 0;
                const char *ds = js_read_string(p, &dlen);
                if (ds && dlen < 16) {
                    memcpy(e->dtype, ds, (size_t)dlen);
                    e->dtype[dlen] = 0;
                }
                p = js_skip_string(p);
            } else if (flen == 5 && memcmp(fstart, "shape", 5) == 0) {
                int vals[4];
                int vn = js_read_int_array(p, vals, 4);
                e->n_dims = vn;
                for (int i = 0; i < vn; i++) e->shape[i] = (uint32_t)vals[i];
                p = js_skip_value(p);
            } else if (flen == 12 && memcmp(fstart, "data_offsets", 12) == 0) {
                uint64_t vals[2];
                int vn = js_read_uint64_array(p, vals, 2);
                if (vn >= 2) {
                    e->data_start = vals[0];
                    e->data_end = vals[1];
                }
                p = js_skip_value(p);
            } else {
                p = js_skip_value(p);
            }
            if (!p) break;
            p = js_skip_ws(p);
            if (*p == ',') p++;
        }
        if (*p == '}') p++;
        n++;

        p = js_skip_ws(p);
        if (*p == ',') p++;
    }

    return n;
}

// ============================================================
// HF Config Parser (config.json → vx_model_config)
// ============================================================

typedef struct {
    const char *name;    // HF model_type string
    vx_arch arch;        // arch enum
} hf_arch_map;

static const hf_arch_map hf_archs[] = {
    {"qwen2",       VX_ARCH_QWEN2},
    {"qwen2.5",     VX_ARCH_QWEN2},
    {"qwen3",       VX_ARCH_QWEN2},
    {"qwen3moe",    VX_ARCH_QWEN2},
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
    {"RefinedWebModel", VX_ARCH_LLAMA}, // RWKV etc
    {"falcon",      VX_ARCH_LLAMA},
    {"deepseek_v2", VX_ARCH_LLAMA},
    {"deepseek_v3", VX_ARCH_LLAMA},
};

// Safetensors dtype → vx_type
static vx_type st_dtype_to_vx(const char *dtype) {
    if (strcmp(dtype, "F32") == 0 || strcmp(dtype, "float32") == 0)
        return VX_TYPE_F32;
    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "float16") == 0)
        return VX_TYPE_F16;
    if (strcmp(dtype, "BF16") == 0 || strcmp(dtype, "bfloat16") == 0)
        return VX_TYPE_BF16;
    return VX_TYPE_F32; // default fallback
}

// Get directory from file path
static void get_dir(const char *path, char *dir, int dir_size) {
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
        dir[0] = '.';
        dir[1] = 0;
    }
}

static vx_error st_load_config(const char *model_path, vx_model_config *cfg) {
    // Determine config.json path alongside model file
    char dir[512];
    get_dir(model_path, dir, sizeof(dir));

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
        return VX_OK; // continue with defaults, maybe GGUF path
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

    // Read model_type
    {
        const char *vp = js_find_key(json, "model_type");
        if (vp) {
            int slen = 0;
            const char *s = js_read_string(vp, &slen);
            if (s && slen > 0 && slen < 64) {
                char mtype[64];
                memcpy(mtype, s, (size_t)slen);
                mtype[slen] = 0;
                // Look up arch
                for (size_t i = 0; i < sizeof(hf_archs)/sizeof(hf_archs[0]); i++) {
                    if (strcmp(mtype, hf_archs[i].name) == 0) {
                        cfg->arch = hf_archs[i].arch;
                        break;
                    }
                }
                // Apply arch defaults (mlp_type, norm_type, bias flags)
                vx_model_apply_arch(cfg, mtype);
            }
        }
    }

    // Read config fields
    {
        const char *vp;

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
    }

    if (cfg->n_head_dim == 0 && cfg->n_heads > 0)
        cfg->n_head_dim = cfg->n_embd / cfg->n_heads;

    if (cfg->n_ctx <= 0) cfg->n_ctx = 4096;
    if (cfg->rope_theta <= 0) cfg->rope_theta = 10000.0f;

    free(json);
    return VX_OK;
}

// ============================================================
// Safetensors Model Loader
// ============================================================

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

vx_error st_loader_load(const char *path, vx_model *model) {
    // Load config first
    vx_error err = st_load_config(path, &model->config);
    if (err != VX_OK) return err;

    // Apply arch if not set
    if (model->config.arch == 0 && model->config.mlp_type == 0) {
        vx_model_apply_arch(&model->config, "llama");
    }

    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 128) n_threads = 128;
    omp_set_num_threads(n_threads);
    model->config.n_threads = n_threads;

    // Open safetensors file
    FILE *fp = fopen(path, "rb");
    if (!fp) return VX_ERR_FILE;

    // Read header length (8 bytes, little-endian uint64)
    uint64_t header_len = 0;
    if (fread(&header_len, 1, 8, fp) != 8) {
        fclose(fp); return VX_ERR_FORMAT;
    }

    // Read JSON header
    char *header_json = malloc((size_t)header_len + 1);
    if (!header_json) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(header_json, 1, (size_t)header_len, fp) != (size_t)header_len) {
        free(header_json); fclose(fp); return VX_ERR_FORMAT;
    }
    header_json[header_len] = 0;

    // Parse tensor entries from header
    st_tensor_entry entries[1024];
    int n_entries = st_parse_header(header_json, (size_t)header_len, entries, 1024);
    if (n_entries <= 0) {
        fprintf(stderr, "No tensors found in safetensors header\n");
        free(header_json); fclose(fp); return VX_ERR_FORMAT;
    }

    model->n_tensors = n_entries;
    model->tensors = calloc((size_t)n_entries, sizeof(vx_tensor *));
    if (!model->tensors) { free(header_json); fclose(fp); return VX_ERR_MEMORY; }

    // Load each tensor
    for (int i = 0; i < n_entries; i++) {
        st_tensor_entry *e = &entries[i];
        vx_tensor *t = calloc(1, sizeof(vx_tensor));
        if (!t) continue;

        strncpy(t->name, e->name, VX_MAX_NAME - 1);
        t->name[VX_MAX_NAME - 1] = 0;

        // Convert shape from HF convention to Veltrix convention
        // HF shape: [out_features, in_features] for 2D weights
        // Veltrix ne[0] = fastest-changing dimension
        int shape[4] = {1, 1, 1, 1};
        int n_dims = e->n_dims;
        if (n_dims > 4) n_dims = 4;
        for (int d = 0; d < n_dims; d++) {
            // Reverse order: HF [d0, d1, d2, d3] → Veltrix ne[0]=d3, ne[1]=d2, ...
            shape[n_dims - 1 - d] = (int)e->shape[d];
        }

        vx_type vx_dtype = st_dtype_to_vx(e->dtype);
        t->type = vx_dtype;

        for (int d = 0; d < 4; d++) t->ne[d] = shape[d];

        // Calculate element count
        int n_elems = 1;
        for (int d = 0; d < n_dims; d++) n_elems *= (int)e->shape[d];

        // Allocate F32 buffer always (convert F16/BF16 to F32 on load)
        t->type = VX_TYPE_F32;
        t->nbytes = (size_t)n_elems * 4;
        t->data = malloc(t->nbytes);
        if (!t->data) { free(t); continue; }
        t->owned = true;

        // Read raw data from file
        size_t raw_size = (size_t)(e->data_end - e->data_start);
        size_t data_pos = 8 + (size_t)header_len + (size_t)e->data_start;

        if (fseek(fp, (long)data_pos, SEEK_SET) != 0) { continue; }

        // Read and convert
        if (strcmp(e->dtype, "F32") == 0) {
            if (fread(t->data, 1, raw_size, fp) != raw_size) continue;
        } else if (strcmp(e->dtype, "F16") == 0 || strcmp(e->dtype, "float16") == 0) {
            uint16_t *f16_buf = malloc(raw_size);
            if (!f16_buf) continue;
            if (fread(f16_buf, 1, raw_size, fp) != raw_size) {
                free(f16_buf); continue;
            }
            int n_f16 = n_elems;
            float *f32 = (float *)t->data;
            for (int j = 0; j < n_f16; j++)
                f32[j] = vx_fp16_to_fp32(f16_buf[j]);
            free(f16_buf);
        } else if (strcmp(e->dtype, "BF16") == 0 || strcmp(e->dtype, "bfloat16") == 0) {
            uint16_t *bf16_buf = malloc(raw_size);
            if (!bf16_buf) continue;
            if (fread(bf16_buf, 1, raw_size, fp) != raw_size) {
                free(bf16_buf); continue;
            }
            int n_bf16 = n_elems;
            float *f32 = (float *)t->data;
            for (int j = 0; j < n_bf16; j++)
                f32[j] = bf16_to_f32(bf16_buf[j]);
            free(bf16_buf);
        } else {
            // Fallback: read raw (I8, I32, etc.)
            if (fread(t->data, 1, raw_size < t->nbytes ? raw_size : t->nbytes, fp) != (raw_size < t->nbytes ? raw_size : t->nbytes))
                continue;
        }

        model->tensors[i] = t;
    }

    free(header_json);
    fclose(fp);
    return VX_OK;
}
