#ifndef VX_QUANTIZE_H
#define VX_QUANTIZE_H

#include "veltrix.h"

#define QK4_0 32
typedef struct {
    uint16_t d;                
    uint8_t qs[QK4_0 / 2];    
} block_q4_0;

#define QK4_1 32
typedef struct {
    uint16_t d;                
    uint16_t m;                
    uint8_t qs[QK4_1 / 2];    
} block_q4_1;

#define QK5_0 32
typedef struct {
    uint16_t d;                
    uint8_t qh[4];             
    uint8_t qs[QK5_0 / 2];    
} block_q5_0;

#define QK5_1 32
typedef struct {
    uint16_t d;                
    uint16_t m;                
    uint8_t qh[4];             
    uint8_t qs[QK5_1 / 2];    
} block_q5_1;

#define QK8_0 32
typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];         
} block_q8_0;

#define QK8_1 32
typedef struct {
    float d;
    float s;
    int8_t qs[QK8_1];
} block_q8_1;

#define QK_K 256
typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;    // fp16
    uint16_t dmin; // fp16
} block_q2_K;      // 16+64+2+2 = 84 bytes

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    uint16_t d; // fp16
} block_q3_K;      // 32+64+12+2 = 110 bytes

typedef struct {
    uint8_t ql[QK_K/2];    // low 4 bits
    uint8_t qh[QK_K/4];    // high 2 bits
    int8_t scales[QK_K/16]; // scales
    uint16_t d;             // fp16 super-scale
} block_q6_K;              // 128+64+16+2 = 210 bytes

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

size_t vx_quantized_size(int n, vx_type type);

void vx_quantize_row_q4_0(const float *x, void *y, int n);
void vx_dequantize_row_q4_0(const void *x, float *y, int n);
void vx_dequantize_row_q4_1(const void *x, float *y, int n);
void vx_dequantize_row_q5_0(const void *x, float *y, int n);
void vx_dequantize_row_q5_1(const void *x, float *y, int n);
void vx_dequantize_row_q8_0(const void *x, float *y, int n);
void vx_dequantize_row_q8_1(const void *x, float *y, int n);

void vx_dequantize_row_q2_K(const void *x, float *y, int n);
void vx_dequantize_row_q3_K(const void *x, float *y, int n);
void vx_dequantize_row_q4_K(const void *x, float *y, int n);
void vx_dequantize_row_q5_K(const void *x, float *y, int n);
void vx_dequantize_row_q6_K(const void *x, float *y, int n);
void vx_dequantize_row_q8_K(const void *x, float *y, int n);
void vx_gemv_q2_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q3_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q4_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q5_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q6_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q8_K(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);

void vx_set_synthetic_weight_scale(float scale);
float vx_get_synthetic_weight_scale(void);

uint16_t vx_fp32_to_fp16(float f);
float vx_fp16_to_fp32(uint16_t h);

void vx_dot_q4_0(const void *x, const float *y, float *dst, int n);
void vx_dot_q4_1(const void *x, const float *y, float *dst, int n);

void vx_gemv_q4_0(const void *restrict A, const float *restrict x,
                  float *restrict y, int rows, int cols);
void vx_gemv_q4_0_sparse_rows(const void *restrict A, const float *restrict x,
                              float *restrict y, int rows, int cols,
                              const float *restrict gate, float threshold);
void vx_gemv_q4_0_sparse_cols_add(const void *restrict A, float *restrict y,
                                   int rows, int cols,
                                   const float *restrict x, float threshold);

#endif
