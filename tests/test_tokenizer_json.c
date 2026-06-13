#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Veltrix tokenizer.json Test\n");
    printf("==========================\n\n");

    // Load HF tokenizer.json
    vx_tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    vx_error err = vx_tokenizer_load_hf("tests/test_tokenizer.json", &tok);
    if (err != VX_OK) {
        printf("FAIL: vx_tokenizer_load_hf: %d\n", err);
        return 1;
    }

    printf("Tokenizer loaded:\n");
    printf("  type:       %d (1=BPE)\n", tok.type);
    printf("  vocab_size: %d (expected 21)\n", tok.vocab_size);

    int ok = 1;
    if (tok.type != VX_TOKENIZER_BPE) { printf("  FAIL: type\n"); ok = 0; }
    if (tok.vocab_size != 21) { printf("  FAIL: vocab_size\n"); ok = 0; }

    // Test encoding
    const char *text = "hello world";
    int ids[64];
    int n = vx_tokenizer_encode(&tok, text, ids, 64);
    printf("  encode('%s') = %d tokens: ", text, n);
    for (int i = 0; i < n && i < 12; i++) printf("%d ", ids[i]);
    printf("\n");
    if (n <= 0) { printf("  FAIL: encode returned %d\n", n); ok = 0; }

    // Test decode (should round-trip)
    char *decoded = vx_tokenizer_decode(&tok, ids, n);
    printf("  decode = '%s'\n", decoded ? decoded : "(null)");
    if (!decoded) { printf("  FAIL: decode returned NULL\n"); ok = 0; }
    else if (strcmp(text, decoded) != 0) { printf("  FAIL: round-trip mismatch\n"); ok = 0; }
    free(decoded);

    // Test special tokens
    printf("  bos_id: %d, eos_id: %d, pad_id: %d, unk_id: %d\n",
           tok.special.bos_id, tok.special.eos_id, tok.special.pad_id, tok.special.unk_id);

    vx_tokenizer_free(&tok);

    printf("\n%s\n", ok ? "PASS: tokenizer.json" : "FAIL: tokenizer.json");
    return ok ? 0 : 1;
}
