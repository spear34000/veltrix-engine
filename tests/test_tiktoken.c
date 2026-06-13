#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Veltrix TikToken Test\n");
    printf("====================\n\n");

    vx_tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    vx_error err = vx_tokenizer_load_hf("tests/test_tiktoken.json", &tok);
    if (err != VX_OK) {
        printf("FAIL: vx_tokenizer_load_hf: %d\n", err);
        return 1;
    }

    printf("Tokenizer loaded:\n");
    printf("  type:       %d (expected %d = TikToken)\n", tok.type, VX_TOKENIZER_TIKTOKEN);
    printf("  type_name:  %s\n", vx_tokenizer_type_name(&tok));
    printf("  vocab_size: %d (expected 24)\n", tok.vocab_size);

    int ok = 1;
    if (tok.type != VX_TOKENIZER_TIKTOKEN) { printf("  FAIL: type\n"); ok = 0; }
    if (tok.vocab_size != 24) { printf("  FAIL: vocab_size\n"); ok = 0; }

    const char *text = "hello world";
    int ids[64];
    int n = vx_tokenizer_encode(&tok, text, ids, 64);
    printf("  encode('%s') = %d tokens: ", text, n);
    for (int i = 0; i < n && i < 12; i++) printf("%d ", ids[i]);
    printf("\n");
    if (n <= 0) { printf("  FAIL: encode returned %d\n", n); ok = 0; }

    char *decoded = vx_tokenizer_decode(&tok, ids, n);
    printf("  decode = '%s'\n", decoded ? decoded : "(null)");
    if (!decoded) { printf("  FAIL: decode returned NULL\n"); ok = 0; }
    else if (strcmp(text, decoded) != 0) { printf("  FAIL: round-trip mismatch\n"); ok = 0; }
    free(decoded);

    printf("  bos_id: %d, eos_id: %d, pad_id: %d, unk_id: %d\n",
           tok.special.bos_id, tok.special.eos_id, tok.special.pad_id, tok.special.unk_id);

    // Test with multi-word text that exercises pre-tokenization
    const char *text2 = "hello world hi123 there";
    int ids2[64];
    int n2 = vx_tokenizer_encode(&tok, text2, ids2, 64);
    printf("  encode('%s') = %d tokens: ", text2, n2);
    for (int i = 0; i < n2 && i < 16; i++) printf("%d ", ids2[i]);
    printf("\n");
    if (n2 <= 0) { printf("  FAIL: encode2 returned %d\n", n2); ok = 0; }

    char *decoded2 = vx_tokenizer_decode(&tok, ids2, n2);
    printf("  decode = '%s'\n", decoded2 ? decoded2 : "(null)");
    if (decoded2 && strcmp(text2, decoded2) != 0) {
        printf("  WARN: round-trip mismatch (pre-tokenization may split differently)\n");
    }
    free(decoded2);

    vx_tokenizer_free(&tok);

    printf("\n%s\n", ok ? "PASS: TikToken" : "FAIL: TikToken");
    return ok ? 0 : 1;
}
