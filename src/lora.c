#include "format.h"
#include "tensor.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// ============================================================
// Minimal JSON Parser (for adapter_config.json)
// ============================================================

static const char *lj_skip_ws(const char *p) {
    while (p && *p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static const char *lj_skip_string(const char *p) {
    if (!p || *p != '"') return NULL;
    p++;
    while (p && *p && *p != '"') {
        if (*p == '\\') { p++; if (p && *p) p++; }
        else { p++; }
    }
    if (p && *p == '"') p++;
    return p;
}

static const char *lj_skip_value(const char *p) {
    p = lj_skip_ws(p);
    if (!p || !*p) return NULL;
    if (*p == '"') return lj_skip_string(p);
    if (*p == '{' || *p == '[') {
        int depth = 1; p++;
        while (p && *p && depth > 0) {
            if (*p == '"') { p = lj_skip_string(p); if (!p) return NULL; }
            else if (*p == '{' || *p == '[') { depth++; p++; }
            else if (*p == '}' || *p == ']') { depth--; p++; }
            else { p++; }
        }
        return p;
    }
    while (p && *p && *p != ',' && *p != '}' && *p != ']' && *p != ' ') p++;
    return p;
}

static const char *lj_find_key(const char *json, const char *key) {
    if (!json) return NULL;
    const char *p = lj_skip_ws(json);
    if (*p != '{') return NULL;
    p++;
    int key_len = (int)strlen(key);
    while (p && *p && *p != '}') {
        p = lj_skip_ws(p);
        if (!p || *p != '"') break;
        const char *ks = p + 1;
        const char *ke = ks;
        while (ke && *ke && *ke != '"') {
            if (*ke == '\\') ke++;
            if (ke && *ke) ke++;
        }
        int klen = (int)(ke - ks);
        p = ke;
        if (p && *p == '"') p++;
        p = lj_skip_ws(p);
        if (!p || *p != ':') break;
        p++;
        p = lj_skip_ws(p);
        if (klen == key_len && memcmp(ks, key, (size_t)klen) == 0)
            return p;
        p = lj_skip_value(p);
        if (!p) return NULL;
        p = lj_skip_ws(p);
        if (p && *p == ',') p++;
    }
    return NULL;
}

static const char *lj_read_string(const char *p, int *out_len) {
    p = lj_skip_ws(p);
    if (!p || *p != '"') return NULL;
    p++;
    const char *start = p;
    while (p && *p && *p != '"') {
        if (*p == '\\') { p++; if (p && *p) p++; }
        else { p++; }
    }
    if (out_len) *out_len = (int)(p - start);
    return start;
}

static int lj_read_int(const char *p, int default_val) {
    p = lj_skip_ws(p);
    if (!p) return default_val;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    int val = 0;
    while (p && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return val * sign;
}

static float lj_read_float(const char *p, float default_val) {
    p = lj_skip_ws(p);
    if (!p) return default_val;
    return (float)atof(p);
}

// ============================================================
// Safetensors header parser for LoRA files
// ============================================================

typedef struct {
    char name[128];
    int64_t shape[4];
    int n_dims;
    int dtype;      // safetensors dtype code
    uint64_t offset;
    uint64_t len;
} lora_st_entry;

// Safetensors dtypes
#define ST_BOOL   0
#define ST_U8     1
#define ST_I8     2
#define ST_I16    3
#define ST_U16    4
#define ST_F16    5
#define ST_F32    6
#define ST_I32    7
#define ST_I64    8
#define ST_F64    9
#define ST_BF16   15

static int lora_dtype_size(int dtype) {
    switch (dtype) {
        case ST_BOOL: case ST_U8: case ST_I8: return 1;
        case ST_I16: case ST_U16: case ST_F16: case ST_BF16: return 2;
        case ST_F32: case ST_I32: return 4;
        case ST_I64: case ST_F64: return 8;
        default: return 4;
    }
}

static int lora_parse_header(const char *json, size_t json_len,
                              lora_st_entry *entries, int max_entries) {
    (void)json_len;
    if (!json) return 0;
    int n = 0;

    // Skip the opening {
    const char *p = lj_skip_ws(json);
    if (!p || *p != '{') return 0;
    p++;

    while (p && *p && *p != '}' && n < max_entries) {
        p = lj_skip_ws(p);
        if (!p || *p != '"') break;

        // Read tensor name
        const char *ks = p + 1;
        const char *ke = ks;
        while (ke && *ke && *ke != '"') {
            if (*ke == '\\') ke++;
            if (ke && *ke) ke++;
        }
        int name_len = (int)(ke - ks);
        if (name_len > 127) name_len = 127;
        p = ke;
        if (p && *p == '"') p++;
        p = lj_skip_ws(p);
        if (!p || *p != ':') break;
        p++;
        p = lj_skip_ws(p);
        if (!p || *p != '{') break;

        memcpy(entries[n].name, ks, (size_t)name_len);
        entries[n].name[name_len] = 0;
        entries[n].n_dims = 0;
        entries[n].offset = 0;
        entries[n].len = 0;

        // Parse inner dict: {"dtype": "...", "shape": [...], "data_offsets": [start, end]}
        p++;
        while (p && *p && *p != '}') {
            p = lj_skip_ws(p);
            if (!p || *p != '"') break;
            const char *fks = p + 1;
            const char *fke = fks;
            while (fke && *fke && *fke != '"') {
                if (*fke == '\\') fke++;
                if (fke && *fke) fke++;
            }
            int fklen = (int)(fke - fks);
            p = fke;
            if (p && *p == '"') p++;
            p = lj_skip_ws(p);
            if (!p || *p != ':') break;
            p++;
            p = lj_skip_ws(p);

            if (fklen == 5 && memcmp(fks, "dtype", 5) == 0) {
                // Read dtype string
                int dt_len = 0;
                const char *dt_str = lj_read_string(p, &dt_len);
                if (dt_str) {
                    if (dt_len == 3 && memcmp(dt_str, "F16", 3) == 0) entries[n].dtype = ST_F16;
                    else if (dt_len == 3 && memcmp(dt_str, "F32", 3) == 0) entries[n].dtype = ST_F32;
                    else if (dt_len == 4 && memcmp(dt_str, "BF16", 4) == 0) entries[n].dtype = ST_BF16;
                    else if (dt_len == 3 && memcmp(dt_str, "I32", 3) == 0) entries[n].dtype = ST_I32;
                    else if (dt_len == 3 && memcmp(dt_str, "I64", 3) == 0) entries[n].dtype = ST_I64;
                    else if (dt_len == 4 && memcmp(dt_str, "BOOL", 4) == 0) entries[n].dtype = ST_BOOL;
                    else if (dt_len == 4 && memcmp(dt_str, "F64", 3) == 0) entries[n].dtype = ST_F64;
                }
                p = lj_skip_string(p);
            } else if (fklen == 5 && memcmp(fks, "shape", 5) == 0) {
                // Read shape array
                if (p && *p == '[') {
                    p++;
                    entries[n].n_dims = 0;
                    while (p && *p && *p != ']' && entries[n].n_dims < 4) {
                        p = lj_skip_ws(p);
                        if (*p == ']') break;
                        entries[n].shape[entries[n].n_dims] = (int64_t)atof(p);
                        entries[n].n_dims++;
                        while (p && *p && *p != ',' && *p != ']') p++;
                        if (*p == ',') p++;
                    }
                    if (p && *p == ']') p++;
                }
            } else if (fklen == 12 && memcmp(fks, "data_offsets", 12) == 0) {
                if (p && *p == '[') {
                    p++;
                    p = lj_skip_ws(p);
                    entries[n].offset = (uint64_t)atof(p);
                    while (p && *p && *p != ',') p++;
                    if (*p == ',') p++;
                    p = lj_skip_ws(p);
                    uint64_t end = (uint64_t)atof(p);
                    entries[n].len = end - entries[n].offset;
                    while (p && *p && *p != ']') p++;
                    if (*p == ']') p++;
                }
            } else {
                p = lj_skip_value(p);
            }

            if (!p) break;
            p = lj_skip_ws(p);
            if (p && *p == ',') p++;
        }
        if (!p) break;
        if (*p == '}') p++;

        n++;
        p = lj_skip_ws(p);
        if (p && *p == ',') p++;
    }

    return n;
}

// ============================================================
// LoRA name parsing
// ============================================================

// Returns: 0 = not a LoRA weight, 1 = lora_A, 2 = lora_B
// Fills layer_idx, module_name (e.g., "q", "k", "v", "o", "gate", "up", "down")
static int lora_parse_name(const char *name, int *layer_idx, char *module, int module_cap) {
    if (!name) return 0;
    *layer_idx = -1;
    module[0] = 0;

    // Pattern: ...layers.{N}.self_attn.q_proj.lora_A.weight
    const char *layers_str = strstr(name, "layers.");
    if (!layers_str) return 0;

    layers_str += 7; // skip "layers."
    if (*layers_str < '0' || *layers_str > '9') return 0;
    *layer_idx = 0;
    while (*layers_str >= '0' && *layers_str <= '9') {
        *layer_idx = *layer_idx * 10 + (*layers_str - '0');
        layers_str++;
    }

    // Find module name: self_attn.q_proj or mlp.gate_proj etc
    const char *mod = NULL;
    if ((mod = strstr(layers_str, "self_attn."))) {
        mod += 10; // skip "self_attn."
        // Read up to ".lora_"
        const char *end = strstr(mod, ".lora_");
        if (!end) return 0;
        int len = (int)(end - mod);
        if (len >= module_cap) len = module_cap - 1;
        memcpy(module, mod, (size_t)len);
        module[len] = 0;

        // Map HF name to short name
        if (strcmp(module, "q_proj") == 0) memcpy(module, "q", 2);
        else if (strcmp(module, "k_proj") == 0) memcpy(module, "k", 2);
        else if (strcmp(module, "v_proj") == 0) memcpy(module, "v", 2);
        else if (strcmp(module, "o_proj") == 0) memcpy(module, "o", 2);
        else if (strcmp(module, "query") == 0) memcpy(module, "q", 2);
        else if (strcmp(module, "key") == 0) memcpy(module, "k", 2);
        else if (strcmp(module, "value") == 0) memcpy(module, "v", 2);
        else if (strcmp(module, "output") == 0) memcpy(module, "o", 2);

        // Determine A or B
        const char *lora_type = end + 1; // ".lora_" skipped
        if (memcmp(lora_type, "lora_A", 6) == 0) return 1;  // A matrix
        if (memcmp(lora_type, "lora_B", 6) == 0) return 2;  // B matrix
        if (memcmp(lora_type, "A", 1) == 0) return 1;
        if (memcmp(lora_type, "B", 1) == 0) return 2;
    } else if ((mod = strstr(layers_str, "mlp."))) {
        mod += 4;
        const char *end = strstr(mod, ".lora_");
        if (!end) return 0;
        int len = (int)(end - mod);
        if (len >= module_cap) len = module_cap - 1;
        memcpy(module, mod, (size_t)len);
        module[len] = 0;

        if (strcmp(module, "gate_proj") == 0) memcpy(module, "gate", 5);
        else if (strcmp(module, "up_proj") == 0) memcpy(module, "up", 3);
        else if (strcmp(module, "down_proj") == 0) memcpy(module, "down", 5);

        const char *lora_type = end + 1;
        if (memcmp(lora_type, "lora_A", 6) == 0) return 1;
        if (memcmp(lora_type, "lora_B", 6) == 0) return 2;
        if (memcmp(lora_type, "A", 1) == 0) return 1;
        if (memcmp(lora_type, "B", 1) == 0) return 2;
    }

    return 0;
}

// ============================================================
// LoRA adapter structure
// ============================================================

typedef struct {
    int layer;
    char module[16];    // "q", "k", "v", "o", "gate", "up", "down"
    float *lora_A;      // A matrix data (d_in × rank, row-major)
    float *lora_B;      // B matrix data (d_out × rank, row-major)
    int d_in;
    int d_out;
    int rank;
    int valid;
} lora_entry;

typedef struct {
    int rank;
    float alpha;
    int n_entries;
    int cap_entries;
    lora_entry *entries;
} lora_impl;

vx_error vx_lora_load(const char *path, int base_n_layers, vx_lora *lora) {
    if (!path || !lora) return VX_ERR_PARAM;
    
    // Build adapter_config.json path
    char config_path[1024];
    const char *sep = strrchr(path, '/');
#ifdef _WIN32
    const char *sep2 = strrchr(path, '\\');
    if (!sep || (sep2 && sep2 > sep)) sep = sep2;
#endif
    if (sep) {
        int cpy = (int)(sep - path);
        if (cpy > 1019) cpy = 1019;
        memcpy(config_path, path, (size_t)cpy);
        config_path[cpy] = 0;
    } else {
        config_path[0] = '.';
        config_path[1] = 0;
    }
    strcat(config_path, "/adapter_config.json");

    // Load config
    FILE *cfg_fp = fopen(config_path, "rb");
    if (!cfg_fp) {
        // Try alongside the model path
        // Use default rank=8, alpha=16
        lora->rank = 8;
        lora->alpha = 16.0f;
        lora->n_adapters = 0;
        lora->impl = NULL;
    } else {
        fseek(cfg_fp, 0, SEEK_END);
        long cfg_size = ftell(cfg_fp);
        fseek(cfg_fp, 0, SEEK_SET);
        char *cfg_json = malloc((size_t)cfg_size + 1);
        if (!cfg_json) { fclose(cfg_fp); return VX_ERR_MEMORY; }
        if (fread(cfg_json, 1, (size_t)cfg_size, cfg_fp) != (size_t)cfg_size) {
            free(cfg_json); fclose(cfg_fp); return VX_ERR_PARAM;
        }
        fclose(cfg_fp);
        cfg_json[cfg_size] = 0;

        // Parse config fields
        const char *rp = lj_find_key(cfg_json, "r");
        lora->rank = rp ? lj_read_int(rp, 8) : 8;

        const char *ap = lj_find_key(cfg_json, "lora_alpha");
        lora->alpha = ap ? lj_read_float(ap, 16.0f) : 16.0f;

        free(cfg_json);

        lora->n_adapters = 0;
        lora->impl = NULL;
    }

    // Open safetensors file
    FILE *fp = fopen(path, "rb");
    if (!fp) return VX_ERR_FILE;

    // Read header length (8 bytes, LE uint64)
    uint64_t header_len = 0;
    if (fread(&header_len, 1, 8, fp) != 8) { fclose(fp); return VX_ERR_FORMAT; }

    // Read JSON header
    char *header_json = malloc((size_t)header_len + 1);
    if (!header_json) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(header_json, 1, (size_t)header_len, fp) != (size_t)header_len) {
        free(header_json); fclose(fp); return VX_ERR_FORMAT;
    }
    header_json[header_len] = 0;

    // Parse tensor entries
    lora_st_entry entries[512];
    int n_entries = lora_parse_header(header_json, (size_t)header_len, entries, 512);
    if (n_entries <= 0) {
        free(header_json); fclose(fp); return VX_ERR_FORMAT;
    }

    // Count valid LoRA pairs
    int pair_count = 0;
    for (int i = 0; i < n_entries; i++) {
        int layer;
        char module[16];
        int type = lora_parse_name(entries[i].name, &layer, module, sizeof(module));
        // Count each pair as 0.5 (A and B make one pair); just count A entries
        if (type == 1 && layer >= 0 && layer < base_n_layers) pair_count++;
    }

    if (pair_count == 0) {
        free(header_json); fclose(fp);
        lora->n_adapters = 0;
        lora->impl = NULL;
        return VX_OK;
    }

    // Allocate impl
    lora_impl *li = calloc(1, sizeof(lora_impl));
    if (!li) { free(header_json); fclose(fp); return VX_ERR_MEMORY; }

    li->rank = lora->rank;
    li->alpha = lora->alpha;
    li->cap_entries = pair_count;
    li->entries = calloc((size_t)pair_count, sizeof(lora_entry));
    if (!li->entries) { free(li); free(header_json); fclose(fp); return VX_ERR_MEMORY; }

    // Temporary map: (layer, module) → entry index
    int *lora_map = calloc((size_t)(base_n_layers * 8), sizeof(int));
    if (!lora_map) { free(li->entries); free(li); free(header_json); fclose(fp); return VX_ERR_MEMORY; }
    for (int i = 0; i < base_n_layers * 8; i++) lora_map[i] = -1;

    // First pass: register A entries, create entries
    int ent_idx = 0;
    for (int i = 0; i < n_entries; i++) {
        int layer;
        char module[16];
        int type = lora_parse_name(entries[i].name, &layer, module, sizeof(module));
        if (type != 1 || layer < 0 || layer >= base_n_layers) continue;

        // Map module to index
        int mod_idx = -1;
        if (strcmp(module, "q") == 0) mod_idx = 0;
        else if (strcmp(module, "k") == 0) mod_idx = 1;
        else if (strcmp(module, "v") == 0) mod_idx = 2;
        else if (strcmp(module, "o") == 0) mod_idx = 3;
        else if (strcmp(module, "gate") == 0) mod_idx = 4;
        else if (strcmp(module, "up") == 0) mod_idx = 5;
        else if (strcmp(module, "down") == 0) mod_idx = 6;
        if (mod_idx < 0) continue;

        int map_key = layer * 8 + mod_idx;
        if (lora_map[map_key] >= 0) continue;

        lora_entry *e = &li->entries[ent_idx];
        e->layer = layer;
        memcpy(e->module, module, strlen(module) + 1);

        // Determine dimensions from A matrix shape
        int n_dims = entries[i].n_dims;
        // HF shape: [d_in, rank] → shape[0]=d_in, shape[1]=rank
        if (n_dims >= 2) {
            e->d_in = (int)entries[i].shape[0];
            e->rank = (int)entries[i].shape[1];
        } else {
            e->d_in = (int)entries[i].shape[0];
            e->rank = li->rank;
        }
        // d_out will be determined from B matrix
        e->d_out = 0;

        // Read A data
        size_t a_size = sizeof(float) * (size_t)e->d_in * (size_t)e->rank;
        e->lora_A = malloc(a_size);
        if (!e->lora_A) continue;
        memset(e->lora_A, 0, a_size);

        {
            uint64_t tensor_off = 8 + header_len + entries[i].offset;
#ifdef _WIN32
            _fseeki64(fp, (int64_t)tensor_off, SEEK_SET);
#else
            fseeko(fp, (off_t)tensor_off, SEEK_SET);
#endif
        }
        size_t raw_len = (size_t)entries[i].len;
        int dtype_size = lora_dtype_size(entries[i].dtype);
        size_t dtype_count = raw_len / (size_t)dtype_size;

        if (dtype_count == (size_t)e->d_in * (size_t)e->rank) {
            if (entries[i].dtype == ST_F32) {
                if (fread(e->lora_A, 1, a_size, fp) == a_size) {
                    // OK - A shape in safetensors is [d_in, rank]
                    // Store as flat row-major [d_in][rank]
                }
            } else if (entries[i].dtype == ST_F16) {
                uint16_t *f16_data = malloc(raw_len);
                if (f16_data && fread(f16_data, 1, raw_len, fp) == raw_len) {
                    size_t n = (size_t)e->d_in * (size_t)e->rank;
                    for (size_t j = 0; j < n; j++) {
                        // F16 to F32 conversion
                        uint16_t h = f16_data[j];
                        uint32_t sign = (h >> 15) & 1;
                        uint32_t exp = (h >> 10) & 0x1F;
                        uint32_t mant = h & 0x3FF;
                        uint32_t f;
                        if (exp == 0) {
                            f = (sign << 31) | (0x7F - 15 + 1) << 23 | (mant << 13);
                        } else if (exp == 0x1F) {
                            f = (sign << 31) | 0x7F800000 | (mant << 13);
                        } else {
                            f = (sign << 31) | (exp - 15 + 127) << 23 | (mant << 13);
                        }
                        memcpy(&e->lora_A[j], &f, 4);
                    }
                    free(f16_data);
                } else { free(f16_data); }
            } else if (entries[i].dtype == ST_BF16) {
                uint16_t *bf16_data = malloc(raw_len);
                if (bf16_data && fread(bf16_data, 1, raw_len, fp) == raw_len) {
                    size_t n = (size_t)e->d_in * (size_t)e->rank;
                    for (size_t j = 0; j < n; j++) {
                        uint32_t f = (uint32_t)bf16_data[j] << 16;
                        memcpy(&e->lora_A[j], &f, 4);
                    }
                    free(bf16_data);
                } else { free(bf16_data); }
            }
        }

        e->valid = 1;
        lora_map[map_key] = ent_idx;
        ent_idx++;
    }

    // Second pass: read B matrices into existing entries
    for (int i = 0; i < n_entries; i++) {
        int layer;
        char module[16];
        int type = lora_parse_name(entries[i].name, &layer, module, sizeof(module));
        if (type != 2 || layer < 0 || layer >= base_n_layers) continue;

        int mod_idx = -1;
        if (strcmp(module, "q") == 0) mod_idx = 0;
        else if (strcmp(module, "k") == 0) mod_idx = 1;
        else if (strcmp(module, "v") == 0) mod_idx = 2;
        else if (strcmp(module, "o") == 0) mod_idx = 3;
        else if (strcmp(module, "gate") == 0) mod_idx = 4;
        else if (strcmp(module, "up") == 0) mod_idx = 5;
        else if (strcmp(module, "down") == 0) mod_idx = 6;
        if (mod_idx < 0) continue;

        int map_key = layer * 8 + mod_idx;
        int entry_idx = lora_map[map_key];
        if (entry_idx < 0) continue;

        lora_entry *e = &li->entries[entry_idx];

        // Determine d_out from B matrix shape
        // HF shape: [rank, d_out] → shape[0]=rank, shape[1]=d_out
        int n_dims = entries[i].n_dims;
        int b_d_out = 0, b_rank = 0;
        if (n_dims >= 2) {
            b_rank = (int)entries[i].shape[0];
            b_d_out = (int)entries[i].shape[1];
        } else {
            b_d_out = (int)entries[i].shape[0];
            b_rank = 0;
        }

        if (b_rank != e->rank) e->rank = b_rank;
        e->d_out = b_d_out;

        size_t b_size = sizeof(float) * (size_t)e->d_out * (size_t)e->rank;
        e->lora_B = malloc(b_size);
        if (!e->lora_B) continue;
        memset(e->lora_B, 0, b_size);

        {
            uint64_t tensor_off = 8 + header_len + entries[i].offset;
#ifdef _WIN32
            _fseeki64(fp, (int64_t)tensor_off, SEEK_SET);
#else
            fseeko(fp, (off_t)tensor_off, SEEK_SET);
#endif
        }
        size_t raw_len = (size_t)entries[i].len;
        int dtype_size = lora_dtype_size(entries[i].dtype);
        size_t dtype_count = raw_len / (size_t)dtype_size;

        if (dtype_count == (size_t)e->d_out * (size_t)e->rank) {
            if (entries[i].dtype == ST_F32) {
                if (fread(e->lora_B, 1, b_size, fp) == b_size) {
                    // OK - store as flat row-major [rank][d_out]
                }
            } else if (entries[i].dtype == ST_F16) {
                uint16_t *f16_data = malloc(raw_len);
                if (f16_data && fread(f16_data, 1, raw_len, fp) == raw_len) {
                    size_t n = (size_t)e->d_out * (size_t)e->rank;
                    for (size_t j = 0; j < n; j++) {
                        uint16_t h = f16_data[j];
                        uint32_t sign = (h >> 15) & 1;
                        uint32_t exp = (h >> 10) & 0x1F;
                        uint32_t mant = h & 0x3FF;
                        uint32_t f;
                        if (exp == 0) {
                            f = (sign << 31) | (0x7F - 15 + 1) << 23 | (mant << 13);
                        } else if (exp == 0x1F) {
                            f = (sign << 31) | 0x7F800000 | (mant << 13);
                        } else {
                            f = (sign << 31) | (exp - 15 + 127) << 23 | (mant << 13);
                        }
                        memcpy(&e->lora_B[j], &f, 4);
                    }
                    free(f16_data);
                } else { free(f16_data); }
            } else if (entries[i].dtype == ST_BF16) {
                uint16_t *bf16_data = malloc(raw_len);
                if (bf16_data && fread(bf16_data, 1, raw_len, fp) == raw_len) {
                    size_t n = (size_t)e->d_out * (size_t)e->rank;
                    for (size_t j = 0; j < n; j++) {
                        uint32_t f = (uint32_t)bf16_data[j] << 16;
                        memcpy(&e->lora_B[j], &f, 4);
                    }
                    free(bf16_data);
                } else { free(bf16_data); }
            }
        }
    }

    free(lora_map);
    free(header_json);
    fclose(fp);

    li->n_entries = ent_idx;
    lora->n_adapters = ent_idx;
    lora->impl = li;

    return VX_OK;
}

vx_error vx_lora_merge(vx_lora *lora, vx_model *model) {
    if (!lora || !model || !lora->impl) return VX_ERR_PARAM;

    lora_impl *li = (lora_impl *)lora->impl;
    float scale = li->alpha / (float)li->rank;

    for (int i = 0; i < li->n_entries; i++) {
        lora_entry *e = &li->entries[i];
        if (!e->valid || !e->lora_A || !e->lora_B) continue;

        if (e->layer < 0 || e->layer >= model->config.n_layers) continue;

        // Find target tensor
        vx_tensor *t = NULL;
        if (strcmp(e->module, "q") == 0 && e->layer < model->config.n_layers) {
            t = model->attn_q[e->layer];
        } else if (strcmp(e->module, "k") == 0) {
            t = model->attn_k[e->layer];
        } else if (strcmp(e->module, "v") == 0) {
            t = model->attn_v[e->layer];
        } else if (strcmp(e->module, "o") == 0) {
            t = model->attn_o[e->layer];
        } else if (strcmp(e->module, "gate") == 0) {
            t = model->ffn_gate[e->layer];
        } else if (strcmp(e->module, "up") == 0) {
            t = model->ffn_up[e->layer];
        } else if (strcmp(e->module, "down") == 0) {
            t = model->ffn_down[e->layer];
        }

        if (!t || !t->data) continue;

        // Dimension check
        int w_d_in = t->ne[0];
        int w_d_out = t->ne[1];
        if (e->d_in != w_d_in || e->d_out != w_d_out) {
            continue;
        }

        // B is stored as row-major [rank][d_out] = rank rows, d_out cols
        // A is stored as row-major [d_in][rank] = d_in rows, rank cols
        // Merge: W += scale * A * B^T
        // W[i][j] += scale * sum_k A[j][k] * B[i][k]
        //   where i = output dim, j = input dim, k = rank
        // In memory (row-major):
        //   W[w_d_out][w_d_in]: W[i][j] = W[i * w_d_in + j]
        //   A[d_in][rank]: A[j * rank + k]
        //   B[rank][d_out]: B[k * d_out + i]
        // But we store B as [rank][d_out] in row-major flat array
        //   Actually, B is stored as B[i * rank + k] where i=d_out, k=rank
        // Wait, let me reconsider.

        // lora_A data layout: [d_in][rank] row-major = d_in rows, rank cols
        //   A[j * rank + k]
        // lora_B data layout: [d_out][rank] row-major = d_out rows, rank cols
        // Actually for the merge formula:
        //   W += (alpha/r) * B_mat @ A_mat
        // where B_mat is d_out × rank, A_mat is rank × d_in
        // But we store:
        //   lora_B: shape [rank, d_out] in HF = safetensors stores [rank][d_out]
        //     Row-major: B[k * d_out + i] for row k, col i
        //   lora_A: shape [d_in, rank] in HF = safetensors stores [d_in][rank]
        //     Row-major: A[j * rank + k] for row j, col k
        // So B_mat[i][k] = B[k * d_out + i] (need to transpose B)
        // W += scale * B_mat @ A_mat
        // W[i][j] += scale * sum_k B_mat[i][k] * A_mat[k][j]
        //   = scale * sum_k B[k * d_out + i] * A[j * rank + k]

        int r = e->rank;
        int d_in = e->d_in;
        int d_out = e->d_out;
        float *w_data = (float *)t->data;

        // Single-threaded merge (OpenMP can cause issues with small tensors)
        for (int i = 0; i < d_out; i++) {
            for (int j = 0; j < d_in; j++) {
                float sum = 0.0f;
                for (int k = 0; k < r; k++) {
                    sum += e->lora_B[k * (size_t)d_out + i] * e->lora_A[j * (size_t)r + k];
                }
                size_t idx = (size_t)i * (size_t)d_in + j;
                w_data[idx] += scale * sum;
            }
        }
    }

    return VX_OK;
}

vx_error vx_lora_apply(vx_lora *lora, const float *input, float *output,
                   int layer, const char *weight_name, int n, int d) {
    (void)input;
    (void)output;
    (void)layer;
    (void)weight_name;
    (void)n;
    (void)d;
    (void)lora;
    // Runtime application not yet implemented (use merge instead)
    return VX_ERR_UNSUPPORTED;
}

void vx_lora_destroy(vx_lora *lora) {
    if (!lora) return;
    lora_impl *li = (lora_impl *)lora->impl;
    if (!li) return;

    for (int i = 0; i < li->n_entries; i++) {
        free(li->entries[i].lora_A);
        free(li->entries[i].lora_B);
    }
    free(li->entries);
    free(li);
    lora->impl = NULL;
    lora->n_adapters = 0;
}
