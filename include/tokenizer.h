#ifndef VX_TOKENIZER_H
#define VX_TOKENIZER_H

#include "veltrix.h"
#include <stddef.h>

// === Tokenizer Abstraction ===

// Tokenizer types
typedef enum {
    VX_TOKENIZER_UNKNOWN = 0,
    VX_TOKENIZER_BPE,          // GPT-2 style BPE (used by GPT-2, StarCoder, Phi)
    VX_TOKENIZER_SENTENCEPIECE,// SentencePiece/Unigram (used by Llama, Mistral, Gemma)
    VX_TOKENIZER_TIKTOKEN,     // TikToken (used by Qwen2, Phi-3, DeepSeek)
    VX_TOKENIZER_WPM,          // WordPiece (used by BERT)
} vx_tokenizer_type;

// Special token IDs
typedef struct {
    int bos_id;       // <s> or <|begin_of_text|>
    int eos_id;       // </s> or <|end_of_text|>
    int pad_id;       // <pad>
    int unk_id;       // <unk>
    int sep_id;       // <sep> (SentencePiece) or <|extra_0|>
    int mask_id;      // <mask> (for MLM models)
} vx_special_tokens;

// Forward declaration
typedef struct vx_tokenizer vx_tokenizer;

// Tokenizer virtual table
typedef struct {
    // Encode text string into token IDs
    // Returns number of tokens written, or <0 on error
    int (*encode)(vx_tokenizer *tok, const char *text, int *output_ids, int max_ids);

    // Decode token IDs back to text string
    // Returns a malloc'd string (caller must free)
    char *(*decode)(vx_tokenizer *tok, const int *ids, int n_ids);

    // Get the tokenizer type name string
    const char *(*type_name)(vx_tokenizer *tok);

    // Free all tokenizer resources
    void (*free)(vx_tokenizer *tok);
} vx_tokenizer_vtable;

struct vx_tokenizer {
    const vx_tokenizer_vtable *vtable;
    vx_tokenizer_type type;
    vx_special_tokens special;
    int vocab_size;
    void *impl;  // format-specific data
};

// === Tokenizer Loading ===

// Load tokenizer from GGUF file buffer (KV metadata contains tokenizer data)
// Parses tokenizer.ggml.* KV pairs
vx_error vx_tokenizer_load_gguf(const uint8_t *gguf_buf, size_t size, vx_tokenizer *tok);

// Load tokenizer from HuggingFace tokenizer.json file
vx_error vx_tokenizer_load_hf(const char *tokenizer_json_path, vx_tokenizer *tok);

// Load tokenizer from SentencePiece .model file
vx_error vx_tokenizer_load_sentencepiece(const char *model_path, vx_tokenizer *tok);

// === Tokenizer Convenience ===

// Auto-detect and load tokenizer alongside a model file
// For GGUF: loads from GGUF metadata
// For HF: looks for tokenizer.json, then tokenizer_config.json, then .model file
vx_error vx_tokenizer_auto_load(const char *model_path, vx_tokenizer *tok);

// Encode (inline convenience)
static inline int vx_tokenizer_encode(vx_tokenizer *tok, const char *text,
                                       int *ids, int max_ids) {
    return tok && tok->vtable && tok->vtable->encode
        ? tok->vtable->encode(tok, text, ids, max_ids) : -1;
}

// Decode (inline convenience; returns malloc'd string)
static inline char *vx_tokenizer_decode(vx_tokenizer *tok, const int *ids, int n_ids) {
    return tok && tok->vtable && tok->vtable->decode
        ? tok->vtable->decode(tok, ids, n_ids) : NULL;
}

// Free
static inline void vx_tokenizer_free(vx_tokenizer *tok) {
    if (tok && tok->vtable && tok->vtable->free)
        tok->vtable->free(tok);
}

// Get type name string
static inline const char *vx_tokenizer_type_name(vx_tokenizer *tok) {
    if (tok && tok->vtable && tok->vtable->type_name)
        return tok->vtable->type_name(tok);
    return "none";
}

#endif
