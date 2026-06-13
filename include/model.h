#ifndef VX_MODEL_H
#define VX_MODEL_H

#include "veltrix.h"

vx_error vx_model_create(const char *path, vx_model **model);
void vx_model_destroy(vx_model *model);
vx_error vx_model_forward(vx_model *model, const int *tokens, int n_tokens, float *logits);

#endif
