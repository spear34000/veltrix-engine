#ifndef VX_SPECULATIVE_H
#define VX_SPECULATIVE_H

#include "veltrix.h"

typedef struct vx_speculative_state vx_speculative_state;

vx_speculative_state *vx_speculative_create(vx_model *draft, vx_model *target, int max_draft_tokens);
void vx_speculative_destroy(vx_speculative_state *state);
void vx_speculative_set_temp(vx_speculative_state *state, float temp, float top_p);

int vx_speculative_generate(vx_speculative_state *state, int *prompt, int prompt_len,
                           int *output, int max_tokens);
int vx_speculative_generate_stream(vx_speculative_state *state, int token,
                                  int *output, int max_tokens);

#endif