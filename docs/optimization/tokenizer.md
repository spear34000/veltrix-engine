# Tokenizer System

`include/tokenizer.h` + `src/tokenizer.c` — text ↔ token ID conversion.

## Architecture

Tokenizers implement a **vtable interface**:

```c
typedef struct {
    int (*encode)(vx_tokenizer *tok, const char *text, int *output_ids, int max_ids);
    char *(*decode)(vx_tokenizer *tok, const int *ids, int n_ids);
    const char *(*type_name)(vx_tokenizer *tok);
    void (*free)(vx_tokenizer *tok);
} vx_tokenizer_vtable;
```

The `vx_tokenizer` struct wraps the vtable:

```c
struct vx_tokenizer {
    const vx_tokenizer_vtable *vtable;
    vx_tokenizer_type type;        // BPE, SENTENCEPIECE, TIKTOKEN
    vx_special_tokens special;     // bos_id, eos_id, pad_id, unk_id
    int vocab_size;
    void *impl;                    // format-specific data (bpe_impl / sp_impl)
};
```

## Supported Types

| Type | Enum | Models | Implementation |
|---|---|---|---|
| BPE | `VX_TOKENIZER_BPE` | GPT-2, StarCoder, Phi | Byte-level tokens + merge ranking |
| SentencePiece | `VX_TOKENIZER_SENTENCEPIECE` | Llama, Mistral, Gemma | Unigram vocab + greedy longest-prefix match |
| TikToken | `VX_TOKENIZER_TIKTOKEN` | Qwen2, Phi-3, DeepSeek | Not yet implemented |

## Tokenizer Loading

### From GGUF Metadata (primary path)

GGUF files embed tokenizer data in KV pairs:
- `tokenizer.ggml.model` — "gpt2" (BPE) or "llama" (SentencePiece)
- `tokenizer.ggml.tokens` — array of strings (vocabulary)
- `tokenizer.ggml.scores` — array of floats (SentencePiece scores)
- `tokenizer.ggml.merges` — array of strings (BPE merge rules)
- `tokenizer.ggml.bos_id`, `eos_id`, `pad_id` — special token IDs

`vx_tokenizer_load_gguf()` parses these KVs and creates the appropriate tokenizer.

### Auto-Load

`vx_tokenizer_auto_load()` detects GGUF vs safetensors format:
1. **GGUF**: loads tokenizer from embedded metadata
2. **Safetensors**: looks for `tokenizer.json` alongside the model file

## BPE Implementation

### Data Structures
- `tokens[]`: array of strings (vocabulary)
- `merge_rank[65536]`: (byte_pair → merge priority), indexed by `(left_byte << 8) | right_byte`
- `byte_pieces[]`: maps single-character tokens back to bytes

### Encoding
1. Split input text into individual bytes
2. Map each byte to its token ID via `byte_pieces`
3. Apply BPE merges greedily (find highest-priority adjacent pair, merge)
4. Return resulting token IDs

### Decoding
1. For each token ID, look up its string in the vocabulary
2. Skip special tokens (<s>, </s>, <pad>)
3. Decode `<0xXX>` byte tokens back to bytes
4. Concatenate all token strings → output text

## SentencePiece Implementation

### Data Structures
- `tokens[]`: array of strings (vocabulary)
- `scores[]`: float scores for each token (for unigram scoring)

### Encoding
Greedy longest-prefix matching:
1. At each position in input text, find the longest vocabulary token that matches
2. Emit its token ID
3. Advance by the matched length
4. Byte fallback: encode unknown bytes as `<0xXX>` tokens

### Decoding
1. For each token ID, look up its string
2. Replace `▁` (U+2581, SentencePiece space marker) with regular space
3. Trim leading space
4. Concatenate all token strings

## Special Tokens

```c
typedef struct {
    int bos_id;    // Beginning of sequence (typically 1)
    int eos_id;    // End of sequence (typically 2)
    int pad_id;    // Padding (typically 0)
    int unk_id;    // Unknown token
    int sep_id;    // Separator token
    int mask_id;   // Mask token (MLM)
} vx_special_tokens;
```

The generation loop stops when it encounters `eos_id`.

## CLI Integration

In `main.c`:
1. User provides `-p "prompt text"`
2. Text is encoded to token IDs via `vx_tokenizer_encode()`
3. Token IDs are fed to `vx_model_forward()`
4. Output token IDs are decoded to text via `vx_tokenizer_decode()`
5. Text is printed to stdout incrementally

## Future: TikToken

TikToken (used by Qwen2, Phi-3, DeepSeek) uses a different byte-level BPE with regex-based pre-tokenization. Implementation will require:
1. Regex pattern compilation for pre-tokenization
2. Byte-level BPE merge table
3. Special token handling (<|im_start|>, <|endoftext|>, etc.)
