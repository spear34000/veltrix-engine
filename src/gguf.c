#include "gguf.h"
#include "quantize.h"
#include "model.h"
#include "format.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

vx_type gguf_to_vx_type(int ggml_t) {
    switch (ggml_t) {
        case GGML_TYPE_F32:  return VX_TYPE_F32;
        case GGML_TYPE_F16:  return VX_TYPE_F16;
        case GGML_TYPE_Q4_0: return VX_TYPE_Q4_0;
        case GGML_TYPE_Q4_1: return VX_TYPE_Q4_1;
        case GGML_TYPE_Q5_0: return VX_TYPE_Q5_0;
        case GGML_TYPE_Q5_1: return VX_TYPE_Q5_1;
        case GGML_TYPE_Q8_0: return VX_TYPE_Q8_0;
        case GGML_TYPE_Q8_1: return VX_TYPE_Q8_1;
        case GGML_TYPE_Q2_K: return VX_TYPE_Q2_K;
        case GGML_TYPE_Q3_K: return VX_TYPE_Q3_K;
        case GGML_TYPE_Q4_K: return VX_TYPE_Q4_K;
        case GGML_TYPE_Q5_K: return VX_TYPE_Q5_K;
        case GGML_TYPE_Q6_K: return VX_TYPE_Q6_K;
        case GGML_TYPE_Q8_K: return VX_TYPE_Q8_K;
        default: return VX_TYPE_F32;
    }
}

int gguf_parse_header(const void *buf, size_t size, gguf_header *hdr) {
    if (!buf || !hdr || size < 21) return -1;
    const uint8_t *p = (const uint8_t*)buf;
    uint32_t magic;
    memcpy(&magic, p, 4); p += 4;
    if (magic != GGUF_MAGIC) { fprintf(stderr, "bad magic: 0x%X\n", magic); return -1; }
    uint32_t version;
    memcpy(&version, p, 4); p += 4;
    memcpy(&hdr->n_tensors, p, 8); p += 8;
    memcpy(&hdr->n_kv_pairs, p, 8); p += 8;
    hdr->tensor_data_offset = 0;
    hdr->file_size = size;
    return 0;
}

static int read_value(const uint8_t *buf, size_t size, uint64_t *off,
                      uint32_t type, void *out, int max_str) {
    if (*off + 1 > size) return -1;
    switch (type) {
        case GGUF_TYPE_U8: {
            if (*off + 1 > size) return -1;
            *(uint8_t*)out = buf[*off]; *off += 1;
            return 0;
        }
        case GGUF_TYPE_I8: {
            if (*off + 1 > size) return -1;
            *(int8_t*)out = (int8_t)buf[*off]; *off += 1;
            return 0;
        }
        case GGUF_TYPE_U16:
        case GGUF_TYPE_I16: {
            if (*off + 2 > size) return -1;
            memcpy(out, buf + *off, 2); *off += 2;
            return 0;
        }
        case GGUF_TYPE_U32:
        case GGUF_TYPE_I32:
        case GGUF_TYPE_F32: {
            if (*off + 4 > size) return -1;
            memcpy(out, buf + *off, 4); *off += 4;
            return 0;
        }
        case GGUF_TYPE_U64:
        case GGUF_TYPE_I64:
        case GGUF_TYPE_F64: {
            if (*off + 8 > size) return -1;
            memcpy(out, buf + *off, 8); *off += 8;
            return 0;
        }
        case GGUF_TYPE_BOOL: {
            if (*off + 1 > size) return -1;
            *(int*)out = buf[*off] ? 1 : 0; *off += 1;
            return 0;
        }
        case GGUF_TYPE_STR: {
            if (*off + 8 > size) return -1;
            uint64_t len;
            memcpy(&len, buf + *off, 8); *off += 8;
            if (*off + len > size) return -1;
            int copy = (int)len;
            if (copy > max_str - 1) copy = max_str - 1;
            memcpy(out, buf + *off, (size_t)copy);
            ((char*)out)[copy] = 0;
            *off += len;
            return 0;
        }
        default:
            return -1;
    }
}

int gguf_read_string(const void *buf, size_t size, uint64_t *offset,
                     char *out, int max_len) {
    return read_value((const uint8_t*)buf, size, offset, GGUF_TYPE_STR, out, max_len);
}

int gguf_read_u32(const void *buf, size_t size, uint64_t *offset,
                  uint32_t *out) {
    return read_value((const uint8_t*)buf, size, offset, GGUF_TYPE_U32, out, 0);
}

int gguf_parse_tensors(const void *buf, size_t size,
                       gguf_header *hdr,
                       gguf_tensor_info *tensors, int max_tensors) {
    if (!buf || !hdr || !tensors) return -1;
    const uint8_t *p = (const uint8_t*)buf;
    uint64_t off = 24;

    // Skip KV pairs
    for (uint64_t i = 0; i < hdr->n_kv_pairs; i++) {
        if (off + 8 > size) return -1;
        uint64_t key_len;
        memcpy(&key_len, p + off, 8); off += 8;
        if (off + key_len > size) return -1;
        off += key_len;

        if (off + 4 > size) return -1;
        uint32_t vtype;
        memcpy(&vtype, p + off, 4); off += 4;

        switch (vtype) {
            case GGUF_TYPE_U8: case GGUF_TYPE_I8: off += 1; break;
            case GGUF_TYPE_U16: case GGUF_TYPE_I16: off += 2; break;
            case GGUF_TYPE_U32: case GGUF_TYPE_I32: case GGUF_TYPE_F32: off += 4; break;
            case GGUF_TYPE_U64: case GGUF_TYPE_I64: case GGUF_TYPE_F64: off += 8; break;
            case GGUF_TYPE_BOOL: off += 1; break;
            case GGUF_TYPE_STR: {
                if (off + 8 > size) return -1;
                uint64_t slen;
                memcpy(&slen, p + off, 8); off += 8;
                if (off + slen > size) return -1;
                off += slen;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                if (off + 4 > size) return -1;
                uint32_t atype;
                memcpy(&atype, p + off, 4); off += 4;
                if (off + 8 > size) return -1;
                uint64_t alen;
                memcpy(&alen, p + off, 8); off += 8;
                for (uint64_t j = 0; j < alen; j++) {
                    switch (atype) {
                        case GGUF_TYPE_U8: case GGUF_TYPE_I8: off += 1; break;
                        case GGUF_TYPE_U16: case GGUF_TYPE_I16: off += 2; break;
                        case GGUF_TYPE_U32: case GGUF_TYPE_I32: case GGUF_TYPE_F32: off += 4; break;
                        case GGUF_TYPE_U64: case GGUF_TYPE_I64: case GGUF_TYPE_F64: off += 8; break;
                        case GGUF_TYPE_STR: {
                            if (off + 8 > size) return -1;
                            uint64_t slen;
                            memcpy(&slen, p + off, 8); off += 8;
                            if (off + slen > size) return -1;
                            off += slen;
                            break;
                        }
                        default: return -1;
                    }
                }
                break;
            }
            default: return -1;
        }
    }

    int count = (int)hdr->n_tensors;
    if (count > max_tensors) count = max_tensors;
    for (int i = 0; i < count; i++) {
        gguf_tensor_info *ti = &tensors[i];
        memset(ti, 0, sizeof(*ti));

        if (off + 8 > size) return -1;
        uint64_t nlen;
        memcpy(&nlen, p + off, 8); off += 8;
        if (off + nlen > size) return -1;
        int nc = (int)nlen;
        if (nc > 255) nc = 255;
        memcpy(ti->name, p + off, (size_t)nc);
        ti->name[nc] = 0;
        off += nlen;

        if (off + 4 > size) return -1;
        memcpy(&ti->n_dims, p + off, 4); off += 4;

        for (uint32_t d = 0; d < ti->n_dims && d < 4; d++) {
            if (off + 8 > size) return -1;
            uint64_t dim_val;
            memcpy(&dim_val, p + off, 8); off += 8;
            ti->dims[d] = (uint32_t)dim_val;
        }

        if (off + 4 > size) return -1;
        memcpy(&ti->ggml_t, p + off, 4); off += 4;

        if (off + 8 > size) return -1;
        memcpy(&ti->offset, p + off, 8); off += 8;

        int n_elems = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++)
            n_elems *= ti->dims[d];
        ti->size = vx_quantized_size(n_elems, gguf_to_vx_type(ti->ggml_t));
    }

    // Tensor data section starts after all tensor infos (32-byte aligned)
    uint64_t pad = (32 - (off % 32)) % 32;
    hdr->tensor_data_offset = off + pad;

    return count;
}

int gguf_find_tensor(const gguf_tensor_info *tensors, int n,
                     const char *name, gguf_tensor_info *out) {
    for (int i = 0; i < n; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            if (out) *out = tensors[i];
            return i;
        }
    }
    return -1;
}

const void *gguf_tensor_data(const void *buf, const gguf_header *hdr,
                             const gguf_tensor_info *ti) {
    return (const uint8_t*)buf + hdr->tensor_data_offset + ti->offset;
}

vx_error gguf_load_tensor(const void *buf, size_t size, const gguf_header *hdr,
                          const gguf_tensor_info *ti,
                          vx_tensor *t) {
    int shape[4] = {1,1,1,1};
    for (uint32_t d = 0; d < ti->n_dims && d < 4; d++)
        shape[d] = (int)ti->dims[d];
    vx_type type = gguf_to_vx_type(ti->ggml_t);

    vx_error err = vx_tensor_alloc(t, (int)ti->n_dims, shape, type);
    if (err != VX_OK) return err;

    if (ti->size > t->nbytes) {
        void *grown = realloc(t->data, ti->size);
        if (!grown) {
            vx_tensor_free(t);
            return VX_ERR_MEMORY;
        }
        memset((uint8_t *)grown + t->nbytes, 0, ti->size - t->nbytes);
        t->data = grown;
        t->nbytes = ti->size;
    }

    if (hdr->tensor_data_offset > size || ti->offset > size - hdr->tensor_data_offset) {
        fprintf(stderr,
                "GGUF tensor range invalid for %s: tensor_data_offset=%zu, tensor_off=%llu, file_size=%zu\n",
                ti->name,
                (size_t)hdr->tensor_data_offset,
                (unsigned long long)ti->offset,
                size);
        return VX_ERR_FORMAT;
    }
    size_t src_off = hdr->tensor_data_offset + (size_t)ti->offset;
    size_t copy = ti->size < t->nbytes ? ti->size : t->nbytes;
    if (copy > size - src_off) {
        fprintf(stderr,
                "GGUF tensor truncated for %s: src_off=%zu, copy=%zu, remaining=%zu, tensor_bytes=%zu, alloc_bytes=%zu, dims=%u\n",
                ti->name,
                src_off,
                copy,
                size - src_off,
                ti->size,
                t->nbytes,
                ti->n_dims);
        return VX_ERR_FORMAT;
    }

    const void *src = (const uint8_t *)buf + src_off;
    if (copy > 0) {
        memcpy(t->data, src, copy);
    }
    return VX_OK;
}

// ============================================================
// Scan KV pairs by key suffix (moved from model.c)
// ============================================================

uint32_t gguf_scan_u32(const void *buf, size_t size, uint64_t *off,
                       uint64_t n_kv, const char *suffix, uint32_t fallback) {
    const uint8_t *p = (const uint8_t*)buf;
    uint64_t o = *off;
    for (uint64_t i = 0; i < n_kv; i++) {
        if (o + 8 > size) return fallback;
        uint64_t klen;
        memcpy(&klen, p + o, 8); o += 8;
        if (o + klen > size) return fallback;
        char key[128];
        int nc = (int)klen;
        if (nc > 127) nc = 127;
        memcpy(key, p + o, (size_t)nc); key[nc] = 0;
        o += klen;
        if (o + 4 > size) return fallback;
        uint32_t vtype;
        memcpy(&vtype, p + o, 4); o += 4;
        uint32_t val = 0;
        int matched = 0;
        if (strstr(key, suffix) && (vtype == 4 || vtype == 5)) { // U32 or I32
            if (o + 4 > size) return fallback;
            memcpy(&val, p + o, 4); matched = 1;
        }
        switch (vtype) {
            case 0: case 1: o += 1; break;
            case 2: case 3: o += 2; break;
            case 4: case 5: case 6: o += 4; break;
            case 10: case 11: case 12: o += 8; break;
            case 7: o += 1; break;
            case 8: {
                if (o + 8 > size) return fallback;
                uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                if (o + slen > size) return fallback;
                o += slen; break;
            }
            case 9: {
                if (o + 4 > size) return fallback;
                uint32_t at; memcpy(&at, p + o, 4); o += 4;
                if (o + 8 > size) return fallback;
                uint64_t alen; memcpy(&alen, p + o, 8); o += 8;
                for (uint64_t j = 0; j < alen; j++) {
                    switch (at) {
                        case 0: case 1: o += 1; break;
                        case 2: case 3: o += 2; break;
                        case 4: case 5: case 6: o += 4; break;
                        case 10: case 11: case 12: o += 8; break;
                        case 8: {
                            if (o + 8 > size) return fallback;
                            uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                            if (o + slen > size) return fallback;
                            o += slen; break;
                        }
                        default: return fallback;
                    }
                }
                break;
            }
            default: return fallback;
        }
        if (matched) return val;
    }
    *off = o;
    return fallback;
}

float gguf_scan_f32(const void *buf, size_t size, uint64_t *off,
                    uint64_t n_kv, const char *suffix, float fallback) {
    const uint8_t *p = (const uint8_t*)buf;
    uint64_t o = *off;
    for (uint64_t i = 0; i < n_kv; i++) {
        if (o + 8 > size) return fallback;
        uint64_t klen;
        memcpy(&klen, p + o, 8); o += 8;
        if (o + klen > size) return fallback;
        char key[128];
        int nc = (int)klen;
        if (nc > 127) nc = 127;
        memcpy(key, p + o, (size_t)nc); key[nc] = 0;
        o += klen;
        if (o + 4 > size) return fallback;
        uint32_t vtype;
        memcpy(&vtype, p + o, 4); o += 4;
        float val = 0;
        int matched = 0;
        if (strstr(key, suffix) && vtype == 6) { // F32
            if (o + 4 > size) return fallback;
            memcpy(&val, p + o, 4); matched = 1;
        }
        switch (vtype) {
            case 0: case 1: o += 1; break;
            case 2: case 3: o += 2; break;
            case 4: case 5: case 6: o += 4; break;
            case 10: case 11: case 12: o += 8; break;
            case 7: o += 1; break;
            case 8: {
                if (o + 8 > size) return fallback;
                uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                if (o + slen > size) return fallback;
                o += slen; break;
            }
            case 9: {
                if (o + 4 > size) return fallback;
                uint32_t at; memcpy(&at, p + o, 4); o += 4;
                if (o + 8 > size) return fallback;
                uint64_t alen; memcpy(&alen, p + o, 8); o += 8;
                for (uint64_t j = 0; j < alen; j++) {
                    switch (at) {
                        case 0: case 1: o += 1; break;
                        case 2: case 3: o += 2; break;
                        case 4: case 5: case 6: o += 4; break;
                        case 10: case 11: case 12: o += 8; break;
                        case 8: {
                            if (o + 8 > size) return fallback;
                            uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                            if (o + slen > size) return fallback;
                            o += slen; break;
                        }
                        default: return fallback;
                    }
                }
                break;
            }
            default: return fallback;
        }
        if (matched) return val;
    }
    *off = o;
    return fallback;
}

int gguf_scan_str(const void *buf, size_t size, uint64_t *off,
                  uint64_t n_kv, const char *suffix, char *out, int out_len) {
    const uint8_t *p = (const uint8_t*)buf;
    uint64_t o = *off;
    for (uint64_t i = 0; i < n_kv; i++) {
        if (o + 8 > size) return -1;
        uint64_t klen;
        memcpy(&klen, p + o, 8); o += 8;
        if (o + klen > size) return -1;
        char key[128];
        int nc = (int)klen;
        if (nc > 127) nc = 127;
        memcpy(key, p + o, (size_t)nc); key[nc] = 0;
        o += klen;
        if (o + 4 > size) return -1;
        uint32_t vtype;
        memcpy(&vtype, p + o, 4); o += 4;
        if (strstr(key, suffix) && vtype == 8) { // STR
            if (o + 8 > size) return -1;
            uint64_t slen;
            memcpy(&slen, p + o, 8); o += 8;
            if (o + slen > size) return -1;
            int cpy = (int)slen;
            if (cpy > out_len - 1) cpy = out_len - 1;
            memcpy(out, p + o, (size_t)cpy); out[cpy] = 0;
            return 0;
        }
        switch (vtype) {
            case 0: case 1: o += 1; break;
            case 2: case 3: o += 2; break;
            case 4: case 5: case 6: o += 4; break;
            case 10: case 11: case 12: o += 8; break;
            case 7: o += 1; break;
            case 8: {
                if (o + 8 > size) return -1;
                uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                if (o + slen > size) return -1;
                o += slen; break;
            }
            case 9: {
                if (o + 4 > size) return -1;
                uint32_t at; memcpy(&at, p + o, 4); o += 4;
                if (o + 8 > size) return -1;
                uint64_t alen; memcpy(&alen, p + o, 8); o += 8;
                for (uint64_t j = 0; j < alen; j++) {
                    switch (at) {
                        case 0: case 1: o += 1; break;
                        case 2: case 3: o += 2; break;
                        case 4: case 5: case 6: o += 4; break;
                        case 10: case 11: case 12: o += 8; break;
                        case 8: {
                            if (o + 8 > size) return -1;
                            uint64_t slen; memcpy(&slen, p + o, 8); o += 8;
                            if (o + slen > size) return -1;
                            o += slen; break;
                        }
                        default: return -1;
                    }
                }
                break;
            }
            default: return -1;
        }
    }
    *off = o;
    return -1;
}

// ============================================================
// GGUF Model Loader (implements vx_loader interface)
// ============================================================

vx_error gguf_loader_load(const char *path, vx_model *model) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open: %s\n", path); return VX_ERR_FILE; }
    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *buf = malloc(size);
    if (!buf) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(buf, 1, size, fp) != size) { free(buf); fclose(fp); return VX_ERR_PARAM; }
    fclose(fp);

    gguf_header hdr;
    if (gguf_parse_header(buf, size, &hdr) != 0) { free(buf); return VX_ERR_FORMAT; }

    int n_tensors = (int)hdr.n_tensors;
    gguf_tensor_info *tensors = malloc((size_t)n_tensors * sizeof(gguf_tensor_info));
    if (!tensors) { free(buf); return VX_ERR_MEMORY; }
    int n_found = gguf_parse_tensors(buf, size, &hdr, tensors, n_tensors);
    if (n_found < 0) { free(tensors); free(buf); return VX_ERR_FORMAT; }

    for (int i = 0; i < n_found; i++) {
        size_t start = (size_t)(hdr.tensor_data_offset + tensors[i].offset);
        size_t end = (i + 1 < n_found)
            ? (size_t)(hdr.tensor_data_offset + tensors[i + 1].offset)
            : size;
        tensors[i].size = (end > start) ? (end - start) : 0;
    }

    // Scan KV metadata for model config
    uint64_t kv_off = 24;
    int n_layers = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "block_count", 32);
    int n_heads = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "head_count", 32);
    int n_kv_heads = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "head_count_kv", (uint32_t)n_heads);
    int n_embd = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "embedding_length", 4096);
    int n_ff = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "feedforward_length", 0);
    int n_vocab = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "vocab_size", 0);
    if (n_vocab == 0)
        n_vocab = (int)gguf_scan_u32(buf, size, &kv_off, hdr.n_kv_pairs, "vocabulary.vocab_size", 0);
    float rope_theta = gguf_scan_f32(buf, size, &kv_off, hdr.n_kv_pairs, "rope.freq_base", 10000.0f);

    char arch_str[64] = {0};
    gguf_scan_str(buf, size, &kv_off, hdr.n_kv_pairs, "general.architecture", arch_str, sizeof(arch_str));

    // Set config defaults, then apply architecture-specific overrides
    vx_model_config *cfg = &model->config;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = n_layers;
    cfg->n_heads = n_heads;
    cfg->n_kv_heads = n_kv_heads > 0 ? n_kv_heads : n_heads;
    cfg->n_embd = n_embd;
    cfg->n_head_dim = n_embd / (n_heads > 0 ? n_heads : 1);
    cfg->n_ff = n_ff;
    cfg->n_vocab = n_vocab;
    cfg->n_ctx = 4096;
    cfg->rope_theta = rope_theta;

    vx_model_apply_arch(cfg, arch_str);

    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 128) n_threads = 128;
    omp_set_num_threads(n_threads);
    cfg->n_threads = n_threads;

    model->n_tensors = n_found;
    model->tensors = calloc((size_t)n_found, sizeof(vx_tensor *));
    if (!model->tensors) { free(tensors); free(buf); return VX_ERR_MEMORY; }

    for (int i = 0; i < n_found; i++) {
        model->tensors[i] = calloc(1, sizeof(vx_tensor));
        if (!model->tensors[i]) continue;
        strncpy(model->tensors[i]->name, tensors[i].name, VX_MAX_NAME - 1);
        model->tensors[i]->name[VX_MAX_NAME - 1] = '\0';
        vx_error terr = gguf_load_tensor(buf, size, &hdr, &tensors[i], model->tensors[i]);
        if (terr != VX_OK) {
            fprintf(stderr, "GGUF tensor load failed for %s\n", tensors[i].name);
            free(tensors);
            free(buf);
            return terr;
        }
        strncpy(model->tensors[i]->name, tensors[i].name, VX_MAX_NAME - 1);
        model->tensors[i]->name[VX_MAX_NAME - 1] = '\0';
    }

    free(tensors);
    free(buf);
    return VX_OK;
}
