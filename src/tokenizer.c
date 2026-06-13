#include "tokenizer.h"
#include "gguf.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// ============================================================
// BPE Tokenizer (GPT-2 style)
// ============================================================

typedef struct {
    int id;
    int priority; // lower = apply first (0 = highest)
} merge_pair;

typedef struct {
    char **tokens;     // vocabulary: token_id → string
    int n_tokens;
    int cap_tokens;

    merge_pair *merges;        // merge table
    int *merge_left;           // left token ID (or -1 if none)
    int *merge_right;          // right token ID (or -1 if none)
    int n_merges;

    int bos_id;
    int eos_id;
    int pad_id;
    int unk_id;

    int byte_to_id[256];       // byte value → token ID mapping
} bpe_impl;

static void bpe_free(vx_tokenizer *tok) {
    bpe_impl *b = (bpe_impl *)tok->impl;
    if (!b) return;
    for (int i = 0; i < b->n_tokens; i++) free(b->tokens[i]);
    free(b->tokens);
    free(b->merges);
    free(b->merge_left);
    free(b->merge_right);
    free(b);
    tok->impl = NULL;
}

static int bpe_byte_to_id(bpe_impl *b, uint8_t byte) {
    return b->byte_to_id[byte];
}

static int bpe_encode_raw(bpe_impl *b, const char *text, int *output, int max_out) {
    if (!b || !text || !output) return -1;

    int ids[4096];
    int n_ids = 0;

    for (const char *p = text; *p && n_ids < 4096; p++) {
        int c = (unsigned char)*p;
        int id = bpe_byte_to_id(b, (uint8_t)c);
        if (id >= 0) ids[n_ids++] = id;
        else { ids[n_ids++] = b->unk_id; }
    }

    if (n_ids == 0) return 0;

    for (int iter = 0; iter < 65536 && n_ids > 1; iter++) {
        int best_rank = 999999999;
        int best_pos = -1;

        for (int i = 0; i < n_ids - 1; i++) {
            for (int m = 0; m < b->n_merges; m++) {
                if (b->merge_left[m] == ids[i] && b->merge_right[m] == ids[i + 1]) {
                    if (m < best_rank) {
                        best_rank = m;
                        best_pos = i;
                    }
                    break;
                }
            }
        }

        if (best_pos < 0) break;

        int new_id = b->merges[best_rank].id;
        ids[best_pos] = new_id;
        for (int j = best_pos + 1; j < n_ids - 1; j++)
            ids[j] = ids[j + 1];
        n_ids--;
    }

    int out_n = 0;
    for (int i = 0; i < n_ids && out_n < max_out; i++)
        output[out_n++] = ids[i];

    return out_n;
}

static int bpe_encode(vx_tokenizer *tok, const char *text, int *output, int max_out) {
    return bpe_encode_raw((bpe_impl *)tok->impl, text, output, max_out);
}

static char *bpe_decode_raw(bpe_impl *b, const int *ids, int n_ids) {
    if (!b) return NULL;

    // Estimate max output size
    size_t cap = 4096;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t pos = 0;

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= b->n_tokens) continue;
        if (id == b->bos_id || id == b->eos_id || id == b->pad_id) continue;

        const char *s = b->tokens[id];
        if (!s) continue;

        // Skip special markup tokens like <0xXX>
        if (s[0] == '<' && s[1] == '0' && s[2] == 'x' && strlen(s) == 6) {
            // Byte token: decode the hex value
            unsigned int byte_val;
            if (sscanf(s + 3, "%2x", &byte_val) == 1) {
                if (pos + 1 < cap) out[pos++] = (char)byte_val;
            }
            continue;
        }

        size_t slen = strlen(s);
        if (pos + slen + 1 > cap) {
            cap = pos + slen + 4096;
            char *tmp = realloc(out, cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }
        memcpy(out + pos, s, slen);
        pos += slen;
    }

    out[pos] = 0;
    return out;
}

static char *bpe_decode(vx_tokenizer *tok, const int *ids, int n_ids) {
    return bpe_decode_raw((bpe_impl *)tok->impl, ids, n_ids);
}

static const char *bpe_type_name(vx_tokenizer *tok) {
    (void)tok;
    return "BPE";
}

static const vx_tokenizer_vtable bpe_vtable = {
    .encode = bpe_encode,
    .decode = bpe_decode,
    .type_name = bpe_type_name,
    .free = bpe_free,
};

// ============================================================
// GPT-2 Pre-tokenizer (character-class state machine)
// ============================================================

// Character classes for GPT-2-style pre-tokenization
#define C_LETTER   1
#define C_DIGIT    2
#define C_WHITESPACE 4
#define C_PUNCT    8

static int char_class(int c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return C_WHITESPACE;
    if (c >= 'a' && c <= 'z') return C_LETTER;
    if (c >= 'A' && c <= 'Z') return C_LETTER;
    if (c >= '0' && c <= '9') return C_DIGIT;
    return C_PUNCT;
}

// GPT-2 style pre-tokenization: split text into chunks based on character classes.
// Returns number of chunks, fills starts[] and lens[] arrays.
// Each chunk is a substring in the original text.
static int gpt2_pre_tokenize(const char *text, int *starts, int *lens, int max_pieces) {
    if (!text) return 0;
    int n = 0;
    int len = (int)strlen(text);
    int i = 0;

    while (i < len && n < max_pieces) {
        // Skip leading whitespace
        int ws_start = i;
        while (i < len && char_class(text[i]) == C_WHITESPACE) i++;
        int ws_len = i - ws_start;

        // Determine current class at position i
        if (i >= len) {
            // Trailing whitespace
            if (ws_len > 0) {
                starts[n] = ws_start;
                lens[n] = ws_len;
                n++;
            }
            break;
        }

        // Check for contraction pattern: apostrophe + letters
        int is_contraction = 0;
        if (text[i] == '\'' && i + 1 < len) {
            int j = i + 1;
            while (j < len && char_class(text[j]) == C_LETTER) j++;
            if (j > i + 1) is_contraction = 1;
        }

        if (is_contraction) {
            // Emit optional whitespace prefix then apostrophe+letters
            // Actually, GPT-2 keeps the space with the PREVIOUS token
            // But for simplicity, attach space to this piece
            starts[n] = ws_start;
            int j = i + 1;
            while (j < len && char_class(text[j]) == C_LETTER) j++;
            lens[n] = j - ws_start;
            n++;
            i = j;
            continue;
        }

        int cls = char_class(text[i]);
        starts[n] = ws_start;
        int j = i;

        // For contractions: if current char is apostrophe and previous is letter,
        // we DON'T split - keep as one piece. This is handled by case analysis.
        // Simple approach: if current class is letter or digit or punct,
        // consume all chars of that class.

        if (cls == C_LETTER) {
            while (j < len && (char_class(text[j]) == C_LETTER ||
                   (text[j] == '\'' && j + 1 < len && char_class(text[j+1]) == C_LETTER))) {
                // For apostrophe inside a word, keep it
                if (text[j] == '\'') j++;
                else {
                    while (j < len && char_class(text[j]) == C_LETTER) j++;
                }
            }
        } else if (cls == C_DIGIT) {
            while (j < len && char_class(text[j]) == C_DIGIT) j++;
        } else {
            // Punctuation/symbol: consume one or more consecutive punct chars
            while (j < len && char_class(text[j]) == C_PUNCT) j++;
        }

        lens[n] = j - ws_start;
        n++;
        i = j;
    }

    return n;
}

// ============================================================
// TikToken Tokenizer (GPT-2 style BPE with pre-tokenization)
// ============================================================

typedef struct {
    bpe_impl *bpe;  // underlying BPE engine
} tiktoken_impl;

static void tiktoken_free(vx_tokenizer *tok) {
    tiktoken_impl *t = (tiktoken_impl *)tok->impl;
    if (!t) return;
    if (t->bpe) {
        bpe_impl *b = t->bpe;
        for (int i = 0; i < b->n_tokens; i++) free(b->tokens[i]);
        free(b->tokens);
        free(b->merges);
        free(b->merge_left);
        free(b->merge_right);
        free(b);
    }
    free(t);
    tok->impl = NULL;
}

static int tiktoken_encode(vx_tokenizer *tok, const char *text, int *output, int max_out) {
    tiktoken_impl *t = (tiktoken_impl *)tok->impl;
    if (!t || !t->bpe || !text || !output) return -1;

    int pieces[1024];
    int plens[1024];
    int n_pieces = gpt2_pre_tokenize(text, pieces, plens, 1024);

    int total = 0;
    char buf[4096];
    for (int p = 0; p < n_pieces && total < max_out; p++) {
        int cpy = plens[p] < 4095 ? plens[p] : 4095;
        memcpy(buf, text + pieces[p], (size_t)cpy);
        buf[cpy] = 0;

        int n = bpe_encode_raw(t->bpe, buf, output + total, max_out - total);
        if (n > 0) total += n;
    }
    return total;
}

static char *tiktoken_decode(vx_tokenizer *tok, const int *ids, int n_ids) {
    tiktoken_impl *t = (tiktoken_impl *)tok->impl;
    if (!t || !t->bpe) return NULL;
    return bpe_decode_raw(t->bpe, ids, n_ids);
}

static const char *tiktoken_type_name(vx_tokenizer *tok) {
    (void)tok;
    return "TikToken";
}

static const vx_tokenizer_vtable tiktoken_vtable = {
    .encode = tiktoken_encode,
    .decode = tiktoken_decode,
    .type_name = tiktoken_type_name,
    .free = tiktoken_free,
};

static int vx_tokenizer_create_tiktoken(bpe_impl *b, int n_tokens, int bos_id, int eos_id, int pad_id, int unk_id, vx_tokenizer *tok) {
    tiktoken_impl *t = calloc(1, sizeof(tiktoken_impl));
    if (!t) return VX_ERR_MEMORY;
    t->bpe = b;

    tok->type = VX_TOKENIZER_TIKTOKEN;
    tok->vocab_size = n_tokens;
    tok->special.bos_id = bos_id >= 0 ? bos_id : 1;
    tok->special.eos_id = eos_id >= 0 ? eos_id : 2;
    tok->special.pad_id = pad_id;
    tok->special.unk_id = unk_id >= 0 ? unk_id : 0;
    tok->vtable = &tiktoken_vtable;
    tok->impl = t;
    return VX_OK;
}

// ============================================================
// SentencePiece Tokenizer (Unigram style)
// ============================================================

typedef struct {
    char **tokens;
    float *scores;
    int n_tokens;
    int cap_tokens;
    int bos_id;
    int eos_id;
    int pad_id;
    int unk_id;
    int byte_fallback;
} sp_impl;

static void sp_free(vx_tokenizer *tok) {
    sp_impl *s = (sp_impl *)tok->impl;
    if (!s) return;
    for (int i = 0; i < s->n_tokens; i++) free(s->tokens[i]);
    free(s->tokens);
    free(s->scores);
    free(s);
    tok->impl = NULL;
}

static int sp_encode(vx_tokenizer *tok, const char *text, int *output, int max_out) {
    sp_impl *s = (sp_impl *)tok->impl;
    if (!s || !text || !output) return -1;

    int n = 0;
    // Simple greedy longest-prefix match (Viterbi would be better but this works)
    while (*text && n < max_out) {
        int best_id = s->unk_id >= 0 ? s->unk_id : 0;
        int best_len = 0;

        // Find longest matching token
        for (int i = 0; i < s->n_tokens; i++) {
            const char *token = s->tokens[i];
            if (!token || *token == 0) { if (best_len == 0 && i == 0) best_id = i; continue; }
            size_t tlen = strlen(token);
            if (tlen > 0 && strncmp(token, text, tlen) == 0) {
                if ((int)tlen > best_len) {
                    best_len = (int)tlen;
                    best_id = i;
                }
            }
        }

        if (best_len == 0) {
            // Byte fallback
            if (s->byte_fallback) {
                // Encode as <0xXX>
                unsigned char c = (unsigned char)*text;
                char byte_token[8];
                snprintf(byte_token, sizeof(byte_token), "<0x%02X>", c);
                for (int i = 0; i < s->n_tokens; i++) {
                    if (s->tokens[i] && strcmp(s->tokens[i], byte_token) == 0) {
                        best_id = i;
                        break;
                    }
                }
                output[n++] = best_id;
                text++;
                continue;
            }
            output[n++] = best_id;
            text++;
        } else {
            output[n++] = best_id;
            text += best_len;
        }
    }

    return n;
}

static char *sp_decode(vx_tokenizer *tok, const int *ids, int n_ids) {
    sp_impl *s = (sp_impl *)tok->impl;
    if (!s) return NULL;

    size_t cap = 4096;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t pos = 0;

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= s->n_tokens) continue;
        if (id == s->bos_id || id == s->eos_id || id == s->pad_id) continue;

        const char *token = s->tokens[id];
        if (!token) continue;

        // SentencePiece adds '▁' (U+2581) for spaces → convert to regular space
        size_t tlen = strlen(token);

        // Skip <0xXX> byte tokens in output (they'll be decoded as UTF-8)
        if (tlen == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x') {
            unsigned int byte_val;
            if (sscanf(token + 3, "%2x", &byte_val) == 1) {
                if (pos + 1 < cap) out[pos++] = (char)byte_val;
            }
            continue;
        }

        if (pos + tlen + 1 > cap) {
            cap = pos + tlen + 4096;
            char *tmp = realloc(out, cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }

        // Copy, replacing ▁ with space at word boundaries
        for (size_t j = 0; j < tlen; j++) {
            if ((unsigned char)token[j] == 0xE2 && j + 2 < tlen &&
                (unsigned char)token[j+1] == 0x96 && (unsigned char)token[j+2] == 0x81) {
                out[pos++] = ' ';
                j += 2;
            } else {
                out[pos++] = token[j];
            }
        }
    }

    // Trim leading space (SentencePiece adds ▁ before first word)
    while (pos > 0 && out[0] == ' ') {
        memmove(out, out + 1, --pos);
    }
    out[pos] = 0;
    return out;
}

static const char *sp_type_name(vx_tokenizer *tok) {
    (void)tok;
    return "SentencePiece";
}

static const vx_tokenizer_vtable sp_vtable = {
    .encode = sp_encode,
    .decode = sp_decode,
    .type_name = sp_type_name,
    .free = sp_free,
};

// ============================================================
// GGUF Tokenizer Loader
// ============================================================

// Internal: scan a GGUF STR array KV pair
static int gguf_scan_str_array(const void *buf, size_t size, uint64_t *off,
                                uint64_t n_kv, const char *suffix,
                                char **out, int max_items) {
    const uint8_t *p = (const uint8_t *)buf;
    uint64_t o = *off;
    int count = 0;

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

        int is_match = (strstr(key, suffix) != NULL);

        if (is_match && vtype == 9) { // ARRAY
            if (o + 4 > size) return -1;
            uint32_t atype;
            memcpy(&atype, p + o, 4); o += 4;
            if (o + 8 > size) return -1;
            uint64_t alen;
            memcpy(&alen, p + o, 8); o += 8;
            if (atype == 8) { // array of strings
                int n = (int)alen;
                int read_n = (n > max_items) ? max_items : n;
                for (int j = 0; j < n; j++) {
                    if (o + 8 > size) return -1;
                    uint64_t slen;
                    memcpy(&slen, p + o, 8); o += 8;
                    if (o + slen > size) return -1;
                    if (out && j < max_items) {
                        out[j] = malloc((size_t)slen + 1);
                        if (out[j]) {
                            memcpy(out[j], p + o, (size_t)slen);
                            out[j][slen] = 0;
                        }
                    }
                    o += slen;
                }
                count = read_n;
            } else {
                // skip array elements
                for (uint64_t j = 0; j < alen; j++) {
                    switch (atype) {
                        case 0: case 1: o += 1; break;
                        case 2: case 3: o += 2; break;
                        case 4: case 5: case 6: o += 4; break;
                        case 10: case 11: case 12: o += 8; break;
                        default: return -1;
                    }
                }
            }
            *off = o;
            return count;
        }

        // Skip non-matching KV
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

// Scan F32 array
static int gguf_scan_f32_array(const void *buf, size_t size, uint64_t *off,
                                 uint64_t n_kv, const char *suffix,
                                 float *out, int max_items) {
    const uint8_t *p = (const uint8_t *)buf;
    uint64_t o = *off;
    int count = 0;

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

        if (strstr(key, suffix) && vtype == 9) {
            if (o + 4 > size) return -1;
            uint32_t atype;
            memcpy(&atype, p + o, 4); o += 4;
            if (o + 8 > size) return -1;
            uint64_t alen;
            memcpy(&alen, p + o, 8); o += 8;
            if (atype == 6) { // array of f32
                int n = (int)alen;
                if (n > max_items) n = max_items;
                for (int j = 0; j < n; j++) {
                    if (o + 4 > size) return -1;
                    float val;
                    memcpy(&val, p + o, 4); o += 4;
                    if (out && j < max_items) out[j] = val;
                }
                count = n;
            } else {
                for (uint64_t j = 0; j < alen; j++) {
                    switch (atype) {
                        case 4: case 5: case 6: o += 4; break;
                        default: return -1;
                    }
                }
            }
            *off = o;
            return count;
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
                        case 4: case 5: case 6: o += 4; break;
                        default: o += 1; break;
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

vx_error vx_tokenizer_load_gguf(const uint8_t *buf, size_t size, vx_tokenizer *tok) {
    if (!buf || !tok) return VX_ERR_PARAM;

    // Parse GGUF header to get KV count
    gguf_header hdr;
    if (gguf_parse_header(buf, size, &hdr) != 0) return VX_ERR_FORMAT;

    uint64_t kv_off = 24;
    uint64_t n_kv = hdr.n_kv_pairs;

    // Read tokenizer type
    char model_type[64] = {0};
    gguf_scan_str(buf, size, &kv_off, n_kv, "tokenizer.ggml.model", model_type, sizeof(model_type));
    char tokenizer_pre[64] = {0};
    gguf_scan_str(buf, size, &kv_off, n_kv, "tokenizer.ggml.pre", tokenizer_pre, sizeof(tokenizer_pre));
    char arch_str[64] = {0};
    gguf_scan_str(buf, size, &kv_off, n_kv, "general.architecture", arch_str, sizeof(arch_str));

    if (model_type[0] == 0) {
        // No tokenizer metadata — try to infer from architecture
        char arch_str[64] = {0};
        gguf_scan_str(buf, size, &kv_off, n_kv, "general.architecture", arch_str, sizeof(arch_str));
        if (strstr(arch_str, "llama") || strstr(arch_str, "llama"))
            strcpy(model_type, "llama");
        else
            strcpy(model_type, "gpt2");
    }

    // Read special token IDs
    int bos_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "bos_id", 1);
    int eos_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "eos_id", 2);
    int pad_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "padding_id", -1);
    if (pad_id < 0) pad_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "padding_id", 0);
    int unk_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "unknown_token_id", -1);
    if (unk_id < 0) unk_id = (int)gguf_scan_u32(buf, size, &kv_off, n_kv, "unk_id", 0);
    if (unk_id < 0) unk_id = 0;

    // Read token strings — use max possible vocab size, actual count from array
    int max_vocab = 300000;
    char **tokens = calloc((size_t)max_vocab, sizeof(char *));
    if (!tokens) return VX_ERR_MEMORY;
    int n_tokens = gguf_scan_str_array(buf, size, &kv_off, n_kv, "tokenizer.ggml.tokens", tokens, max_vocab);
    if (n_tokens <= 0) {
        // Try alternative key
        kv_off = 24;
        n_tokens = gguf_scan_str_array(buf, size, &kv_off, n_kv, "tokens", tokens, max_vocab);
    }
    if (n_tokens <= 0) {
        free(tokens);
        return VX_ERR_FORMAT;
    }

    // Read scores (optional, for SentencePiece)
    float *scores = NULL;
    int n_scores = 0;
    if (strcmp(model_type, "llama") == 0 || strcmp(model_type, "sentencepiece") == 0) {
        scores = malloc((size_t)max_vocab * sizeof(float));
        if (scores) {
            kv_off = 24;
            n_scores = gguf_scan_f32_array(buf, size, &kv_off, n_kv, "tokenizer.ggml.scores", scores, max_vocab);
            if (n_scores <= 0) {
                free(scores);
                scores = NULL;
            }
        }
    }

    // Determine tokenizer type based on model_type
    int is_bpe = (strcmp(model_type, "gpt2") == 0 || strcmp(model_type, "neox") == 0);
    int is_sp = (strcmp(model_type, "llama") == 0 || strcmp(model_type, "sentencepiece") == 0 ||
                 strcmp(model_type, "unigram") == 0);
    bool use_tiktoken = (strstr(tokenizer_pre, "qwen2") != NULL) || (strstr(arch_str, "qwen2") != NULL);

    if (is_bpe || (!is_sp && n_scores <= 0)) {
        // BPE tokenizer
        bpe_impl *b = calloc(1, sizeof(bpe_impl));
        if (!b) { for (int i = 0; i < n_tokens; i++) free(tokens[i]); free(tokens); free(scores); return VX_ERR_MEMORY; }

        b->tokens = tokens;
        b->n_tokens = n_tokens;
        b->cap_tokens = max_vocab;
        b->bos_id = bos_id;
        b->eos_id = eos_id;
        b->pad_id = pad_id;
        b->unk_id = 0;

        // Build byte-to-ID mapping
        memset(b->byte_to_id, -1, sizeof(b->byte_to_id));
        for (int i = 0; i < n_tokens; i++) {
            if (b->tokens[i] && strlen(b->tokens[i]) == 1) {
                uint8_t byte = (uint8_t)b->tokens[i][0];
                b->byte_to_id[byte] = i;
            }
        }

        if (use_tiktoken) {
            vx_error err = vx_tokenizer_create_tiktoken(b, n_tokens, bos_id, eos_id, pad_id, unk_id, tok);
            if (err != VX_OK) {
                for (int i = 0; i < n_tokens; i++) free(tokens[i]);
                free(tokens); free(b);
                return err;
            }
            return VX_OK;
        }

        // Read BPE merges
        kv_off = 24;
        char **merge_strs = malloc(max_vocab * sizeof(char *));
        if (merge_strs) {
            int n_merge_strs = gguf_scan_str_array(buf, size, &kv_off, n_kv,
                                                     "tokenizer.ggml.merges", merge_strs, max_vocab);
            if (n_merge_strs > 0) {
                b->merges = malloc((size_t)n_merge_strs * sizeof(merge_pair));
                b->merge_left = calloc((size_t)n_merge_strs, sizeof(int));
                b->merge_right = calloc((size_t)n_merge_strs, sizeof(int));
                for (int j = 0; j < n_merge_strs; j++) {
                    if (merge_strs[j]) {
                        char *space = strchr(merge_strs[j], ' ');
                        if (space) {
                            *space = 0;
                            int left = atoi(merge_strs[j]);
                            int right = atoi(space + 1);
                            b->merges[b->n_merges].id = j + n_tokens;
                            b->merges[b->n_merges].priority = j;
                            b->merge_left[b->n_merges] = left;
                            b->merge_right[b->n_merges] = right;
                            b->n_merges++;
                        }
                        free(merge_strs[j]);
                    }
                }
            }
            free(merge_strs);
        }

        tok->type = VX_TOKENIZER_BPE;
        tok->vocab_size = n_tokens;
        tok->special.bos_id = bos_id;
        tok->special.eos_id = eos_id;
        tok->special.pad_id = pad_id;
        tok->special.unk_id = 0;
        tok->vtable = &bpe_vtable;
        tok->impl = b;
        return VX_OK;

    } else {
        // SentencePiece tokenizer
        sp_impl *s = calloc(1, sizeof(sp_impl));
        if (!s) { for (int i = 0; i < n_tokens; i++) free(tokens[i]); free(tokens); free(scores); return VX_ERR_MEMORY; }

        s->tokens = tokens;
        s->scores = scores; // may be NULL
        s->n_tokens = n_tokens;
        s->cap_tokens = max_vocab;
        s->bos_id = bos_id;
        s->eos_id = eos_id;
        s->pad_id = pad_id;
        s->unk_id = 0;
        s->byte_fallback = 1;

        tok->type = VX_TOKENIZER_SENTENCEPIECE;
        tok->vocab_size = n_tokens;
        tok->special.bos_id = bos_id;
        tok->special.eos_id = eos_id;
        tok->special.pad_id = pad_id;
        tok->special.unk_id = 0;
        tok->vtable = &sp_vtable;
        tok->impl = s;
        return VX_OK;
    }
}

// ============================================================
// Minimal JSON Parser (for HF tokenizer.json)
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

// Read a JSON number, return as double
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

// Read a JSON array of strings: ["a", "b", "c", ...]
// Calls callback(id, str, len, userdata) for each element
typedef void (*js_str_cb)(int id, const char *str, int len, void *user);

static int js_read_str_array(const char *p, js_str_cb cb, void *user) {
    p = js_skip_ws(p);
    if (!p || *p != '[') return 0;
    p++;
    int n = 0;
    while (*p && *p != ']') {
        p = js_skip_ws(p);
        if (*p == ']') break;
        int slen = 0;
        const char *s = js_read_string(p, &slen);
        if (!s) break;
        if (cb) cb(n, s, slen, user);
        n++;
        p++; // skip past closing quote
        p = js_skip_ws(p);
        if (*p == ',') p++;
    }
    return n;
}

// ============================================================
// HF tokenizer.json Parser
// ============================================================

typedef struct {
    char **strs;
    int n;
    int cap;
} str_array_builder;

static void merge_cb(int id, const char *str, int len, void *user) {
    (void)id;
    str_array_builder *sb = (str_array_builder *)user;
    if (sb->n >= sb->cap) return;
    char *s = malloc((size_t)len + 1);
    if (!s) return;
    memcpy(s, str, (size_t)len); s[len] = 0;
    sb->strs[sb->n++] = s;
}

vx_error vx_tokenizer_load_hf(const char *json_path, vx_tokenizer *tok) {
    FILE *fp = fopen(json_path, "rb");
    if (!fp) return VX_ERR_FILE;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *json = malloc((size_t)fsize + 1);
    if (!json) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(json, 1, (size_t)fsize, fp) != (size_t)fsize) { free(json); fclose(fp); return VX_ERR_PARAM; }
    fclose(fp);
    json[fsize] = 0;

    // Find "model" nested object
    const char *model_val = js_find_key(json, "model");
    if (!model_val) { free(json); return VX_ERR_FORMAT; }

    // Read type
    const char *type_val = js_find_key(model_val, "type");
    if (!type_val) { free(json); return VX_ERR_FORMAT; }
    int type_len = 0;
    const char *type_str = js_read_string(type_val, &type_len);
    bool is_bpe = (type_str && type_len == 3 && memcmp(type_str, "BPE", 3) == 0);
    bool is_unigram = (type_str && type_len == 7 && memcmp(type_str, "Unigram", 7) == 0);
    if (!is_bpe && !is_unigram) { free(json); return VX_ERR_UNSUPPORTED; }

    // Read vocab (JSON object of string→int mappings)
    int max_vocab = 131072;
    char **tokens = calloc((size_t)max_vocab, sizeof(char *));
    if (!tokens) { free(json); return VX_ERR_MEMORY; }

    {
        const char *vocab_val = js_find_key(model_val, "vocab");
        if (vocab_val) {
            const char *p = js_skip_ws(vocab_val);
            if (p && *p == '{') {
                p++;
                while (*p && *p != '}') {
                    p = js_skip_ws(p);
                    if (*p != '"') break;
                    int klen = 0;
                    const char *ks = js_read_string(p, &klen);
                    if (!ks) break;
                    p += klen + 2; // skip past key string + quotes

                    p = js_skip_ws(p);
                    if (*p != ':') break;
                    p++;
                    p = js_skip_ws(p);

                    // Read integer value (token ID)
                    int num_len = 0;
                    int token_id = 0;
                    {
                        const char *ns = p;
                        int neg = 0;
                        if (*ns == '-') { neg = 1; ns++; }
                        while (*ns && *ns >= '0' && *ns <= '9') {
                            token_id = token_id * 10 + (*ns - '0');
                            ns++;
                        }
                        num_len = (int)(ns - (neg ? p + 1 : p));
                        if (neg) token_id = -token_id;
                        p += num_len + (neg ? 1 : 0);
                    }

                    if (token_id >= 0 && token_id < max_vocab) {
                        if (!tokens[token_id]) {
                            char *s = malloc((size_t)klen + 1);
                            if (s) {
                                memcpy(s, ks, (size_t)klen);
                                s[klen] = 0;
                                tokens[token_id] = s;
                            }
                        }
                    }

                    p = js_skip_ws(p);
                    if (*p == ',') p++;
                }
            }
        }
    }

    // Count actual vocab entries
    int n_tokens = 0;
    for (int i = 0; i < max_vocab; i++) {
        if (tokens[i]) n_tokens = i + 1;
    }
    if (n_tokens <= 0) {
        for (int i = 0; i < max_vocab; i++) free(tokens[i]);
        free(tokens); free(json);
        return VX_ERR_FORMAT;
    }

    // Read special token IDs
    int bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1;
    {
        const char *av = js_find_key(json, "added_tokens");
        if (av) {
            const char *p = js_skip_ws(av);
            if (p && *p == '[') {
                p++;
                while (*p && *p != ']') {
                    p = js_skip_ws(p);
                    if (*p != '{') break;
                    p++;
                    int aid = -1;
                    const char *a_content = NULL;
                    int a_content_len = 0;
                    while (*p && *p != '}') {
                        p = js_skip_ws(p);
                        if (*p != '"') break;
                        int klen = 0;
                        const char *ks = js_read_string(p, &klen);
                        if (!ks) break;
                        p += klen + 2;
                        p = js_skip_ws(p);
                        if (*p != ':') break;
                        p++;
                        p = js_skip_ws(p);
                        if (klen == 2 && memcmp(ks, "id", 2) == 0) {
                            int nl = 0; aid = (int)js_read_number(p, &nl);
                            p += nl;
                        } else                         if (klen == 7 && memcmp(ks, "content", 7) == 0) {
                            a_content = js_read_string(p, &a_content_len);
                            if (a_content) p += a_content_len + 2;
                        } else {
                            p = js_skip_value(p);
                        }
                        if (!p) break;
                        p = js_skip_ws(p);
                        if (*p == ',') p++;
                    }
                    if (a_content && a_content_len > 0 && aid >= 0) {
                        if (a_content_len == 3 && memcmp(a_content, "bos", 3) == 0) bos_id = aid;
                        else if (a_content_len == 3 && memcmp(a_content, "eos", 3) == 0) eos_id = aid;
                        else if (a_content_len == 3 && memcmp(a_content, "pad", 3) == 0) pad_id = aid;
                        else if (a_content_len == 3 && memcmp(a_content, "unk", 3) == 0) unk_id = aid;
                    }
                    if (*p == '}') p++;
                    p = js_skip_ws(p);
                    if (*p == ',') p++;
                }
            }
        }
    }

    if (is_bpe) {
        // BPE: read merges
        bpe_impl *b = calloc(1, sizeof(bpe_impl));
        if (!b) { for (int i = 0; i < n_tokens; i++) free(tokens[i]); free(tokens); free(json); return VX_ERR_MEMORY; }

        b->tokens = tokens;
        b->n_tokens = n_tokens;
        b->cap_tokens = max_vocab;
        b->bos_id = bos_id;
        b->eos_id = eos_id;
        b->pad_id = pad_id;
        b->unk_id = unk_id >= 0 ? unk_id : 0;

        // Read merges from JSON (still have json data)
        {
            const char *mv = js_find_key(json, "model");
            if (mv) {
                const char *merges_val = js_find_key(mv, "merges");
                if (merges_val) {
                    str_array_builder sb = {0};
                    sb.cap = 131072;
                    sb.strs = calloc((size_t)sb.cap, sizeof(char *));
                    if (sb.strs) {
                        js_read_str_array(merges_val, merge_cb, &sb);

                        b->merges = calloc((size_t)sb.n, sizeof(merge_pair));
                        b->merge_left = calloc((size_t)sb.n, sizeof(int));
                        b->merge_right = calloc((size_t)sb.n, sizeof(int));
                        for (int j = 0; j < sb.n && j < sb.cap; j++) {
                            if (sb.strs[j]) {
                                char *space = strchr(sb.strs[j], ' ');
                                if (space) {
                                    *space = 0;
                                    int left = atoi(sb.strs[j]);
                                    int right = atoi(space + 1);
                                    // Find the result token ID by concatenating token strings
                                    int new_id = -1;
                                    if (left >= 0 && left < n_tokens && tokens[left] &&
                                        right >= 0 && right < n_tokens && tokens[right]) {
                                        size_t llen = strlen(tokens[left]);
                                        size_t rlen = strlen(tokens[right]);
                                        char *combined = malloc(llen + rlen + 1);
                                        if (combined) {
                                            memcpy(combined, tokens[left], llen);
                                            memcpy(combined + llen, tokens[right], rlen);
                                            combined[llen + rlen] = 0;
                                            for (int k = 0; k < n_tokens; k++) {
                                                if (tokens[k] && strcmp(tokens[k], combined) == 0) {
                                                    new_id = k;
                                                    break;
                                                }
                                            }
                                            free(combined);
                                        }
                                    }
                                    b->merges[b->n_merges].id = new_id >= 0 ? new_id : (n_tokens + j);
                                    b->merges[b->n_merges].priority = j;
                                    b->merge_left[b->n_merges] = left;
                                    b->merge_right[b->n_merges] = right;
                                    b->n_merges++;
                                }
                                free(sb.strs[j]);
                            }
                        }
                        free(sb.strs);
                    }
                }
            }
        }

        // Check for pre_tokenizer (TikToken-style)
        bool use_tiktoken = false;
        {
            const char *pt_val = js_find_key(json, "pre_tokenizer");
            if (pt_val) {
                const char *pt_type = js_find_key(pt_val, "type");
                if (pt_type) {
                    int pt_len = 0;
                    const char *pt_str = js_read_string(pt_type, &pt_len);
                    if (pt_str) {
                        if ((pt_len == 9 && memcmp(pt_str, "ByteLevel", 9) == 0) ||
                            (pt_len == 4 && memcmp(pt_str, "GPT2", 4) == 0) ||
                            (pt_len == 9 && memcmp(pt_str, "Metaspace", 9) == 0) ||
                            (pt_len == 10 && memcmp(pt_str, "Whitespace", 10) == 0)) {
                            use_tiktoken = true;
                        }
                    }
                }
            }
        }

        free(json);

        // Build byte_to_id mapping
        memset(b->byte_to_id, -1, sizeof(b->byte_to_id));
        for (int i = 0; i < n_tokens; i++) {
            if (b->tokens[i] && strlen(b->tokens[i]) == 1) {
                uint8_t byte = (uint8_t)b->tokens[i][0];
                b->byte_to_id[byte] = i;
            }
        }

        if (use_tiktoken) {
            vx_error err = vx_tokenizer_create_tiktoken(b, n_tokens, bos_id, eos_id, pad_id, unk_id, tok);
            if (err != VX_OK) {
                for (int i = 0; i < n_tokens; i++) free(tokens[i]);
                free(tokens); free(b);
                return err;
            }
            return VX_OK;
        }

        tok->type = VX_TOKENIZER_BPE;
        tok->vocab_size = n_tokens;
        tok->special.bos_id = bos_id >= 0 ? bos_id : 1;
        tok->special.eos_id = eos_id >= 0 ? eos_id : 2;
        tok->special.pad_id = pad_id;
        tok->special.unk_id = unk_id >= 0 ? unk_id : 0;
        tok->vtable = &bpe_vtable;
        tok->impl = b;
        return VX_OK;

    } else {
        // SentencePiece/Unigram
        free(json);
        sp_impl *s = calloc(1, sizeof(sp_impl));
        if (!s) { for (int i = 0; i < n_tokens; i++) free(tokens[i]); free(tokens); return VX_ERR_MEMORY; }

        s->tokens = tokens;
        s->scores = NULL;
        s->n_tokens = n_tokens;
        s->cap_tokens = max_vocab;
        s->bos_id = bos_id;
        s->eos_id = eos_id;
        s->pad_id = pad_id;
        s->unk_id = unk_id >= 0 ? unk_id : 0;
        s->byte_fallback = 1;

        tok->type = VX_TOKENIZER_SENTENCEPIECE;
        tok->vocab_size = n_tokens;
        tok->special.bos_id = bos_id >= 0 ? bos_id : 1;
        tok->special.eos_id = eos_id >= 0 ? eos_id : 2;
        tok->special.pad_id = pad_id;
        tok->special.unk_id = unk_id >= 0 ? unk_id : 0;
        tok->vtable = &sp_vtable;
        tok->impl = s;
        return VX_OK;
    }
}

vx_error vx_tokenizer_load_sentencepiece(const char *model_path, vx_tokenizer *tok) {
    // Try GGUF-based loading first
    FILE *fp = fopen(model_path, "rb");
    if (!fp) return VX_ERR_FILE;

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *buf = malloc(size);
    if (!buf) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(buf, 1, size, fp) != size) { free(buf); fclose(fp); return VX_ERR_PARAM; }
    fclose(fp);

    vx_error err = vx_tokenizer_load_gguf((const uint8_t *)buf, size, tok);
    free(buf);
    return err;
}

vx_error vx_tokenizer_auto_load(const char *model_path, vx_tokenizer *tok) {
    if (!model_path || !tok) return VX_ERR_PARAM;

    // Try loading as GGUF (safetensors path uses config.json alongside)
    FILE *fp = fopen(model_path, "rb");
    if (!fp) return VX_ERR_FILE;

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *buf = malloc(size);
    if (!buf) { fclose(fp); return VX_ERR_MEMORY; }
    if (fread(buf, 1, size, fp) != size) { free(buf); fclose(fp); return VX_ERR_PARAM; }
    fclose(fp);

    // Check if it's a GGUF file
    const uint8_t *bytes = (const uint8_t *)buf;
    if (size >= 4 && bytes[0] == 'G' && bytes[1] == 'G' && bytes[2] == 'U' && bytes[3] == 'F') {
        vx_error err = vx_tokenizer_load_gguf(bytes, size, tok);
        free(buf);
        return err;
    }

    free(buf);

    // Try HF tokenizer.json alongside safetensors file
    char dir[512], json_path[1024];
    const char *sep = strrchr(model_path, '/');
#ifdef _WIN32
    const char *sep2 = strrchr(model_path, '\\');
    if (!sep || (sep2 && sep2 > sep)) sep = sep2;
#endif
    if (sep) {
        int cpy = (int)(sep - model_path);
        if (cpy > 511) cpy = 511;
        memcpy(dir, model_path, (size_t)cpy); dir[cpy] = 0;
    } else {
        dir[0] = '.'; dir[1] = 0;
    }
    snprintf(json_path, sizeof(json_path), "%s/tokenizer.json", dir);

    vx_error err = vx_tokenizer_load_hf(json_path, tok);
    return err;
}
