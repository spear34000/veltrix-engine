#include "veltrix.h"
#include "model.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model>\n", argv[0]); return 1; }
    printf("Opening: %s\n", argv[1]);
    fflush(stdout);
    
    vx_model *model = NULL;
    vx_error err = vx_model_create(argv[1], &model);
    printf("err=%d model=%p\n", err, (void*)model);
    
    if (err == VX_OK && model) {
        printf("Config: %d layers, %d embd, %d heads, %d vocab\n",
               model->config.n_layers, model->config.n_embd,
               model->config.n_heads, model->config.n_vocab);
        printf("Arch: %d MLP:%d Norm:%d RopePartial:%d\n",
               model->config.arch, model->config.mlp_type,
               model->config.norm_type, model->config.rope_partial_dims);
        printf("tok_embd=%p output_w=%p norm_w=%p\n",
               (void*)model->tok_embd, (void*)model->output_w, (void*)model->norm_w);
        for (int l = 0; l < 2 && l < model->config.n_layers; l++) {
            printf("Layer %d: attn_q=%p attn_k=%p attn_v=%p attn_o=%p\n",
                   l, (void*)model->attn_q[l], (void*)model->attn_k[l],
                   (void*)model->attn_v[l], (void*)model->attn_o[l]);
            printf("  ffn_gate=%p ffn_up=%p ffn_down=%p\n",
                   (void*)model->ffn_gate[l], (void*)model->ffn_up[l],
                   (void*)model->ffn_down[l]);
            printf("  attn_norm=%p ffn_norm=%p\n",
                   (void*)model->attn_norm[l], (void*)model->ffn_norm[l]);
        }
        vx_model_destroy(model);
    }
    return 0;
}
