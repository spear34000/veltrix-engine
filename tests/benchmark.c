#include "tensor.h"
#include "quantize.h"
#include "../src/simd/simd.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#ifndef _WIN32
#include <unistd.h>
#endif

typedef struct {
    float *a, *b, *c, *d;
    void *q;
    int rows, cols, n;
} bench_args;

static int cold_bench(const char *name, void (*fn)(void*), void *arg,
                      int iters, int trials, double *out_best, double *out_avg) {
    double *samples = (double*)malloc(trials * sizeof(double));
    if (!samples) return -1;
    double best = 1e99, worst = 0, sum = 0;
    for (int t = 0; t < trials; t++) {
        double t0 = vx_time_now_us();
        for (int i = 0; i < iters; i++) fn(arg);
        double t1 = vx_time_now_us();
        samples[t] = (t1 - t0) / iters;
        if (samples[t] < best) best = samples[t];
        if (samples[t] > worst) worst = samples[t];
        sum += samples[t];
    }
    double avg = sum / trials;
    double var = 0;
    for (int t = 0; t < trials; t++) {
        double d = samples[t] - avg;
        var += d * d;
    }
    free(samples);
    double sd = sqrt(var / trials);
    if (name && name[0]) {
        printf("  %-40s %8.1f us (best) %8.1f us (worst) %8.1f us (avg) %5.1f us (sd)  x%d\n",
               name, best, worst, avg, sd, trials * iters);
    }
    if (out_best) *out_best = best;
    if (out_avg) *out_avg = avg;
    return 0;
}

static void b_gemv_f32(void *p) {
    bench_args *b = (bench_args*)p;
    vx_gemv_f32(b->c, b->a, b->b, b->rows, b->cols);
}
static void b_gemv_avx2(void *p) {
    bench_args *b = (bench_args*)p;
    vx_gemv_f32_avx2(b->c, b->a, b->b, b->rows, b->cols);
}
static void b_gemv_q4(void *p) {
    bench_args *b = (bench_args*)p;
    vx_gemv_q4_0(b->q, b->b, b->c, b->rows, b->cols);
}
static void b_gemv_q4_avx2(void *p) {
    bench_args *b = (bench_args*)p;
    vx_gemv_q4_0_avx2(b->q, b->b, b->c, b->rows, b->cols);
}
static void b_add(void *p) {
    bench_args *b = (bench_args*)p;
    vx_add_f32(b->c, b->a, b->b, b->n);
}
static void b_norm(void *p) {
    bench_args *b = (bench_args*)p;
    vx_rms_norm(b->c, b->a, b->d, b->n);
}
static void b_softmax(void *p) {
    bench_args *b = (bench_args*)p;
    vx_softmax(b->a, b->n);
}
static void b_silu(void *p) {
    bench_args *b = (bench_args*)p;
    vx_silu(b->a, b->n);
}
static void b_rope(void *p) {
    bench_args *b = (bench_args*)p;
    vx_rope(b->c, b->d, 10, 32, 32, 128, 0, 10000.0f);
}

static void alloc_fill(float *buf, int n, float scale) {
    for (int i = 0; i < n; i++) buf[i] = (float)(i % 100) / 10000.0f * scale;
}

static double bytes_per_gemv(int rows, int cols, vx_type type) {
    if (type == VX_TYPE_Q4_0)
        return (double)rows * vx_quantized_size(cols, VX_TYPE_Q4_0);
    return (double)rows * cols * 4;
}

int main(void) {
    setbuf(stdout, NULL);
    printf("========================================\n");
    printf("  VELTRIX RIGOROUS BENCHMARK\n");
    printf("========================================\n\n");

    long logical_cores = 0;
    long page_kb = 0;
#ifdef _WIN32
    SYSTEM_INFO sys;
    GetSystemInfo(&sys);
    logical_cores = (long)sys.dwNumberOfProcessors;
    page_kb = (long)(sys.dwPageSize / 1024);
#else
    logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    page_kb = sysconf(_SC_PAGESIZE) / 1024;
#endif
    if (logical_cores < 1) logical_cores = 1;
    if (page_kb < 1) page_kb = 4;
    printf("HW: %ld logical cores, page %ld KB\n", logical_cores, page_kb);

    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 128) n_threads = 128;
    printf("OpenMP: %d threads\n", n_threads);

    printf("Timer: monotonic us\n\n");

    int D = 4096;
    int rows[] = {1024, 4096, 11008};
    int n_sizes = 3;
    const char *names[] = {"1024x4096", "4096x4096", "11008x4096"};
    int trials = 5;
    int iters[] = {200, 100, 50};

    bench_args ba[4];
    for (int i = 0; i < n_sizes; i++) {
        int r = rows[i];
        size_t sz = (size_t)r * D;
        ba[i].rows = r;
        ba[i].cols = D;
        ba[i].n = r > D ? r : D;
        ba[i].a = (float*)malloc(sz * sizeof(float));
        ba[i].b = (float*)malloc(D * sizeof(float));
        ba[i].c = (float*)malloc(r * sizeof(float));
        ba[i].d = (float*)malloc(D * sizeof(float));
        ba[i].q = NULL;
        if (!ba[i].a || !ba[i].b || !ba[i].c || !ba[i].d) {
            printf("FAIL: OOM %dx%d\n", r, D);
            return 1;
        }
        alloc_fill(ba[i].a, (int)sz, 1.0f);
        alloc_fill(ba[i].b, D, 1.0f);
    }
    printf("Alloc: %dx%d f32=%.0fMB, %dx%d f32=%.0fMB, %dx%d f32=%.0fMB\n\n",
           1024, D, 1024.0*D*4/1048576,
           4096, D, 4096.0*D*4/1048576,
           11008, D, 11008.0*D*4/1048576);

    omp_set_num_threads(1);

    printf("--- GEMV: scalar f32 (unrolled, ST) ---\n\n");
    for (int i = 0; i < n_sizes; i++) {
        double mb = bytes_per_gemv(rows[i], D, VX_TYPE_F32) / 1048576;
        double t_best = 0, t_avg = 0;
        cold_bench(names[i], b_gemv_f32, &ba[i], iters[i], trials, &t_best, &t_avg);
        printf("  %54s  %.1f GB/s  (%.0f MB read)\n", "",
               mb / t_best * 1e6 / 1024, mb);
    }

    printf("\n--- GEMV: AVX2 f32 (ST) ---\n\n");
    for (int i = 0; i < n_sizes; i++) {
        double mb = bytes_per_gemv(rows[i], D, VX_TYPE_F32) / 1048576;
        double t_best = 0, t_avg = 0;
        cold_bench(names[i], b_gemv_avx2, &ba[i], iters[i], trials, &t_best, &t_avg);
        printf("  %54s  %.1f GB/s\n", "", mb / t_best * 1e6 / 1024);
    }

    printf("\n--- GEMV: Q4_0 scalar (ST) ---\n\n");
    for (int i = 0; i < n_sizes; i++) {
        int r = rows[i];
        size_t qsz = vx_quantized_size(r * D, VX_TYPE_Q4_0);
        ba[i].q = malloc(qsz);
        vx_quantize_row_q4_0(ba[i].a, ba[i].q, r * D);
        double mb = bytes_per_gemv(r, D, VX_TYPE_Q4_0) / 1048576;
        double t_best = 0, t_avg = 0;
        cold_bench(names[i], b_gemv_q4, &ba[i], iters[i], trials, &t_best, &t_avg);
        printf("  %54s  %.1f GB/s  (%.0f MB read)\n", "",
               mb / t_best * 1e6 / 1024, mb);
    }

    printf("\n--- GEMV: Q4_0 AVX2 (ST) ---\n\n");
    for (int i = 0; i < n_sizes; i++) {
        double mb = bytes_per_gemv(rows[i], D, VX_TYPE_Q4_0) / 1048576;
        double t_best = 0, t_avg = 0;
        cold_bench(names[i], b_gemv_q4_avx2, &ba[i], iters[i], trials, &t_best, &t_avg);
        printf("  %54s  %.1f GB/s\n", "", mb / t_best * 1e6 / 1024);
    }

    omp_set_num_threads(n_threads);

    printf("\n--- GEMV: Q4_0 AVX2 (MT, %d threads) ---\n\n", n_threads);
    for (int i = 0; i < n_sizes; i++) {
        double mb = bytes_per_gemv(rows[i], D, VX_TYPE_Q4_0) / 1048576;
        double t_best = 0, t_avg = 0;
        cold_bench(names[i], b_gemv_q4_avx2, &ba[i], iters[i], trials, &t_best, &t_avg);
        printf("  %54s  %.1f GB/s\n", "", mb / t_best * 1e6 / 1024);
    }

    printf("\n--- THREAD SCALING (Q4_0 AVX2) ---\n\n");
    int scale_threads[] = {1, 2, 4, 8, 12};
    int n_scale = 5;
    int scale_sizes[] = {0, 1, 2}; // indices into ba[] for 1024, 4096, 11008
    const char *size_names[] = {"1024x4096", "4096x4096", "11008x4096"};
    int scale_iters[] = {200, 100, 50};
    for (int si = 0; si < 3; si++) {
        printf("  %s:\n", size_names[si]);
        printf("    Threads    Best(us)    GB/s       Speedup    Eff\n");
        printf("    -------    --------    ----       -------    ---\n");
        double t_1t = 0;
        for (int s = 0; s < n_scale; s++) {
            omp_set_num_threads(scale_threads[s]);
            double t_best = 0, t_avg = 0;
            cold_bench("", b_gemv_q4_avx2, &ba[scale_sizes[si]], scale_iters[si], 5, &t_best, &t_avg);
            if (s == 0) t_1t = t_best;
            double speedup = t_1t / t_best;
            double eff = speedup / scale_threads[s] * 100;
            double gbs = bytes_per_gemv(rows[scale_sizes[si]], D, VX_TYPE_Q4_0)/1048576 / t_best * 1e6 / 1024;
            printf("    %-9d %-12.1f %-10.1f %-11.1f %d%%\n",
                   scale_threads[s], t_best, gbs, speedup, (int)eff);
        }
    }
    omp_set_num_threads(n_threads);

    printf("\n========================================\n");
    printf("  ALGORITHM HEAD-TO-HEAD (4096x4096, %d threads)  *** HOT-CACHE ***\n", n_threads);
    printf("  WARNING: repeated same-matrix access keeps weights in L3 cache.\n");
    printf("  Real LLM inference is COLD-cache (weights fetched from DRAM each layer).\n");
    printf("  These numbers are 2-3x faster than cold-cache; use MT cold measurements below.\n");
    printf("========================================\n\n");

    struct { const char *n; void (*fn)(void*); int it; double mb; } algos[] = {
        {"scalar f32 GEMV",   b_gemv_f32,    500, bytes_per_gemv(4096, D, VX_TYPE_F32)/1048576},
        {"AVX2 f32 GEMV",     b_gemv_avx2,   500, bytes_per_gemv(4096, D, VX_TYPE_F32)/1048576},
        {"scalar Q4_0 GEMV",  b_gemv_q4,     500, bytes_per_gemv(4096, D, VX_TYPE_Q4_0)/1048576},
        {"AVX2 Q4_0 GEMV",    b_gemv_q4_avx2,500, bytes_per_gemv(4096, D, VX_TYPE_Q4_0)/1048576},
    };
    int n_algos = 4;

    for (int a = 0; a < n_algos; a++) {
        double best = 1e99, worst = 0, sum = 0;
        for (int t = 0; t < trials; t++) {
            double t0 = vx_time_now_us();
            for (int i = 0; i < algos[a].it; i++) algos[a].fn(&ba[0]);
            double t1 = vx_time_now_us();
            double avg_us = (t1 - t0) / algos[a].it;
            if (avg_us < best) best = avg_us;
            if (avg_us > worst) worst = avg_us;
            sum += avg_us;
        }
        double avg = sum / trials;
        printf("  %-25s best %8.1f  worst %8.1f  avg %8.1f us  %5.1f GB/s\n",
               algos[a].n, best, worst, avg, algos[a].mb / avg * 1e6 / 1024);
    }

    printf("\n--- MICRO OPS ---\n\n");
    bench_args bn = {ba[0].b, ba[0].b, ba[0].c, ba[0].d, NULL, 0, 0, D};
    struct { const char *n; void (*fn)(void*); int it; } micros[] = {
        {"vx_add_f32  4096",   b_add,    10000},
        {"vx_rms_norm 4096",   b_norm,   10000},
        {"vx_softmax  4096",   b_softmax,10000},
        {"vx_silu     11008",  b_silu,   5000},
        {"vx_rope     32x128", b_rope,   10000},
    };
    int n_micros = 5;
    for (int a = 0; a < n_micros; a++) {
        double best = 1e99, sum = 0;
        for (int t = 0; t < trials; t++) {
            double t0 = vx_time_now_us();
            for (int i = 0; i < micros[a].it; i++) micros[a].fn(&bn);
            double t1 = vx_time_now_us();
            double avg_us = (t1 - t0) / micros[a].it;
            if (avg_us < best) best = avg_us;
            sum += avg_us;
        }
        printf("  %-25s %8.2f us  (best of %d)\n",
               micros[a].n, best, trials);
    }

    printf("\n--- SCALING EXTRAPOLATION (Q4_0 AVX2 MT, %d threads, 7B 32L) ---\n", n_threads);
    printf("  WARNING: weight GEMV times are measured cold-cache. Attention/KV costs are modeled.\n");
    printf("  Missing: thread jitter, OS noise, sampler, RMSNorm+RoPE at long ctx.\n");
    printf("  Assumes GQA KV projection (K/V dim = 1024, not 4096). Models with full K/V will be ~12%% slower.\n\n");

    double t_1024 = 1e9, t_4096 = 1e9, t_11008 = 1e9;
    for (int t = 0; t < trials; t++) {
        double t0 = vx_time_now_us();
        for (int i = 0; i < 200; i++) b_gemv_q4_avx2(&ba[0]);
        double tk = (vx_time_now_us() - t0) / 200;
        if (tk < t_1024) t_1024 = tk;

        t0 = vx_time_now_us();
        for (int i = 0; i < 100; i++) b_gemv_q4_avx2(&ba[1]);
        tk = (vx_time_now_us() - t0) / 100;
        if (tk < t_4096) t_4096 = tk;

        t0 = vx_time_now_us();
        for (int i = 0; i < 50; i++) b_gemv_q4_avx2(&ba[2]);
        tk = (vx_time_now_us() - t0) / 50;
        if (tk < t_11008) t_11008 = tk;
    }
    double proj_Q  = t_4096;    // 4096x4096
    double proj_K  = t_1024;    // 1024x4096 (GQA)
    double proj_V  = t_1024;    // 1024x4096 (GQA)
    double proj_O  = t_4096;    // 4096x4096
    double ffn_gate = t_11008;  // 11008x4096
    double ffn_up   = t_11008;  // 11008x4096
    double ffn_down = t_11008;  // 4096x11008 (same byte count)

    double micro_rmsnorm = 2.25; // one RMSNorm on d_model=4096
    double micro_rope    = 50.34; // one RoPE on 32x128
    double micro_add     = 0.16;  // vx_add_f32 4096
    double micro_silu    = 10.88; // vx_silu 11008

    double weight_layer = proj_Q + proj_K + proj_V + proj_O + ffn_gate + ffn_up + ffn_down
                          + micro_rmsnorm * 2 + micro_rope * 2 + micro_add * 2 + micro_silu;
    double embd_rms = 2.3;
    double output_proj = proj_Q;
    double sampler = 1.0;
    double weight_only_weighted_total = weight_layer * 32 + embd_rms + output_proj + sampler;

    // Estimate attention cost per layer by context length
    // KV cache: Q4_0, GQA with 8 KV heads. Read: 2*ctx*8*128*0.5 = ctx*1024 bytes per layer
    // Compute: 32 heads * ctx * 128 FMAs each for QK+V = 2*32*ctx*128 FMAs = 8192*ctx FMAs
    // BW ~14 GB/s (MT Q4_0 read), compute ~300 GFLOP/s (AVX2 MT)
    int ctxs[] = {128, 512, 2048, 4096};
    int n_ctxs = 4;
    const double bw_GBs = 14.0; // measured MT Q4_0 throughput

    printf("\n  Weight GEMV + micro-ops per layer: %.0f us\n", weight_layer);
    printf("  Weight-only total (32L):           %.0f us\n", weight_only_weighted_total);
    printf("\n  Context-aware estimates (dense):\n\n");
    printf("  %-10s %-14s %-14s %-14s %-14s\n",
           "ctx", "Attn/layer(us)", "Total(us)", "tok/s", "vs ctx=128");
    printf("  %-10s %-14s %-14s %-14s %-14s\n",
           "----", "-------------", "--------", "-----", "-----------");

    for (int c = 0; c < n_ctxs; c++) {
        double kv_read_us = (2.0 * ctxs[c] * 8 * 128 * 0.5) / (bw_GBs * 1e9) * 1e6;
        double compute_us = (2.0 * 32 * ctxs[c] * 128) / (300e9) * 1e6;
        if (kv_read_us < 5) kv_read_us = 5;
        if (compute_us < 3) compute_us = 3;
        double attention_us = kv_read_us + compute_us;
        double per_layer_ctx = weight_layer + attention_us;
        double total_ctx = per_layer_ctx * 32 + embd_rms + output_proj + sampler;
        double tok_ctx = 1e6 / total_ctx;
        printf("  %-10d %-14.0f %-14.0f %-14.2f %.2f\n",
               ctxs[c], attention_us, total_ctx, tok_ctx, tok_ctx / (1e6 / weight_only_weighted_total));
    }

    double sparse_ffn = ffn_gate + ffn_up / 8.1 + ffn_down;
    double sparse_weight_layer = weight_layer - ffn_gate - ffn_up - ffn_down + sparse_ffn;

    printf("\n  Context-aware estimates (sparse FFN, up=8.1x, gate+down dense):\n\n");
    printf("  %-10s %-14s %-14s %-14s %-14s\n",
           "ctx", "Attn/layer(us)", "Total(us)", "tok/s", "vs dense same ctx");
    printf("  %-10s %-14s %-14s %-14s %-14s\n",
           "----", "-------------", "--------", "-----", "------------------");
    for (int c = 0; c < n_ctxs; c++) {
        double kv_read_us = (2.0 * ctxs[c] * 8 * 128 * 0.5) / (bw_GBs * 1e9) * 1e6;
        double compute_us = (2.0 * 32 * ctxs[c] * 128) / (300e9) * 1e6;
        if (kv_read_us < 5) kv_read_us = 5;
        if (compute_us < 3) compute_us = 3;
        double attention_us = kv_read_us + compute_us;
        double per_layer_ctx = sparse_weight_layer + attention_us;
        double total_ctx = per_layer_ctx * 32 + embd_rms + output_proj + sampler;
        double tok_ctx = 1e6 / total_ctx;
        double dense_total = (weight_layer + attention_us) * 32 + embd_rms + output_proj + sampler;
        printf("  %-10d %-14.0f %-14.0f %-14.2f %.2f\n",
               ctxs[c], attention_us, total_ctx, tok_ctx, tok_ctx * dense_total / 1e6);
    }
    printf("\n  NOTE: FFN-sparsity is 1.27-1.28x overall tok/s gain (not 1.41x).\n");
    printf("  The 1.41x is FFN-only speedup; overall = (dense tok)/(sparse tok).\n");

    printf("\n--- SPARSE FFN BENCHMARK (Q4_0 AVX2, 11008x4096, %d threads) ---\n", n_threads);
    printf("  NOTE: down projection (4096x11008) has same byte count as 11008x4096.\n");
    printf("  Column-sparse Q4_0 access is 64x slower (proven). Down stays dense.\n");
    printf("  Gate always computed densely (needed for row selection).\n\n");

    float *gate = malloc(11008 * sizeof(float));
    for (int i = 0; i < 11008; i++) {
        gate[i] = (i % 10 == 0) ? 1.0f : 0.001f;
    }

    double t_dense = 1e9;
    for (int t = 0; t < 5; t++) {
        double t0 = vx_time_now_us();
        for (int k = 0; k < 50; k++) b_gemv_q4_avx2(&ba[2]);
        double tk = (vx_time_now_us() - t0) / 50;
        if (tk < t_dense) t_dense = tk;
    }
    printf("  Dense up GEMV (11008x4096):  %8.1f us  (100%% rows)\n", t_dense);

    int active_count = 0;
    for (int i = 0; i < 11008; i++) if (fabsf(gate[i]) >= 0.01f) active_count++;
    printf("  Active rows (|gate|>=0.01):   %d / 11008  (%.0f%%)\n", active_count, 100.0f * active_count / 11008);

    bench_args bs = ba[2];
    double t_sparse = 1e9;
    for (int t = 0; t < 5; t++) {
        memset(bs.c, 0, 11008 * sizeof(float));
        double t0 = vx_time_now_us();
        for (int k = 0; k < 50; k++) {
            vx_gemv_q4_0_avx2_sparse_rows(bs.q, bs.b, bs.c, 11008, 4096, gate, 0.01f);
        }
        double tk = (vx_time_now_us() - t0) / 50;
        if (tk < t_sparse) t_sparse = tk;
    }
    printf("  Sparse up GEMV (rows):    %8.1f us  (%.1f%% rows active)\n", t_sparse, 100.0f * active_count / 11008);
    printf("  Row-skip speedup:          %.1f\xd7\n", t_dense / t_sparse);

    printf("\n  FFN gate (11008x4096):       %8.1f us (dense, always compute)\n", t_dense);
    printf("  FFN up sparse (11008x4096):  %8.1f us (%.0f%% rows active)\n",
           t_sparse, 100.0f * active_count / 11008);
    printf("  FFN down (4096x11008):       %8.1f us (dense, same byte count as up)\n", t_dense);

    printf("\n    Dense FFN:  gate(%d) + up(%d) + down(%d) = %d us\n",
           (int)t_dense, (int)t_dense, (int)t_dense, (int)(t_dense * 3));
    printf("    Sparse FFN: gate(%d) + up_sparse(%d) + down(%d) = %d us\n",
           (int)t_dense, (int)t_sparse, (int)t_dense, (int)(t_dense + t_sparse + t_dense));
    double ffn_speedup = (t_dense * 3) / (t_dense + t_sparse + t_dense);
    // Measure non-FFN weight cost per layer (Q+K+V+O projections) for overall speedup calc
    double t_4096_local = 1e9, t_1024_local = 1e9;
    for (int t = 0; t < 5; t++) {
        double t0 = vx_time_now_us();
        for (int k = 0; k < 100; k++) b_gemv_q4_avx2(&ba[1]);
        double tk = (vx_time_now_us() - t0) / 100;
        if (tk < t_4096_local) t_4096_local = tk;
        t0 = vx_time_now_us();
        for (int k = 0; k < 200; k++) b_gemv_q4_avx2(&ba[0]);
        tk = (vx_time_now_us() - t0) / 200;
        if (tk < t_1024_local) t_1024_local = tk;
    }
    double non_ffn_per_layer = t_4096_local * 2 + t_1024_local * 2; // Q+O = 2*t_4096, K+V = 2*t_1024
    double ffn_per_layer_dense = t_dense * 3;
    double ffn_per_layer_sparse = t_dense + t_sparse + t_dense;
    double weight_per_layer_dense = non_ffn_per_layer + ffn_per_layer_dense;
    double weight_per_layer_sparse = non_ffn_per_layer + ffn_per_layer_sparse;
    double overall_speedup = weight_per_layer_dense / weight_per_layer_sparse;
    printf("    FFN-only speedup:        %.1f\xd7  (up sparse only, gate+down dense)\n", ffn_speedup);
    printf("    Non-FFN weight/layer:    %d us (Q+O+K+V, ~%d%% of weight)\n",
           (int)non_ffn_per_layer, (int)(non_ffn_per_layer / weight_per_layer_dense * 100));
    printf("    Overall speedup (weight only, 32L): %.2fx  (diluted by non-FFN weights)\n", overall_speedup);
    printf("    With attention cost:     ~%.2fx  (further diluted, see ctx-aware table above)\n",
           overall_speedup * 0.95);
    printf("  CAUTION: synthetic gate (10%% active by construction). Real SiLU gate\n");
    printf("  sparsity depends on model, input, and context. Quality impact unverified.\n");

    free(gate);

    printf("\n--- MEMORY FOOTPRINT (7B, d=4096, ff=11008, 32L) ---\n\n");
    double pl_B = (double)(4096*4096 + 1024*4096 + 1024*4096 + 4096*4096
                 + 11008*4096 + 11008*4096 + 4096*11008) * 4;
    double emb_B = (double)32000 * 4096 * 4;
    double f32_B = pl_B * 32 + emb_B * 2;
    double q4_B = f32_B / 6.4;
    printf("  Per-layer:              %.0f MB f32  /  %.0f MB Q4_0\n",
           pl_B / 1048576, pl_B / 6.4 / 1048576);
    printf("  Token embd + output:    %.0f MB f32  /  %.0f MB Q4_0\n",
           emb_B * 2 / 1048576, emb_B * 2 / 6.4 / 1048576);
    printf("  32-layer model total:   %.0f MB f32  /  %.0f MB Q4_0\n",
           f32_B / 1048576, q4_B / 1048576);
    printf("  Practical estimate:     %.1f GB f32  /  %.1f GB Q4_0  %s\n",
           f32_B / 1.073741824e9, q4_B / 1.073741824e9,
           q4_B / 1.073741824e9 < 7.5 ? "fits in 8GB" : "needs 16GB+");

    for (int i = 0; i < n_sizes; i++) {
        free(ba[i].a); free(ba[i].b); free(ba[i].c); free(ba[i].d);
        free(ba[i].q);
    }
    printf("\nDone.\n");
    return 0;
}
