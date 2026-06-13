#include "veltrix.h"
#include "model.h"
#include "gguf.h"
#include "tokenizer.h"
#include "metalearn.h"
#include "scheduler.h"
#include "format.h"
#include "platform.h"
#include "runtime_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_usage(const char *prog) {
    printf("Veltrix v0.1 - Prediction-Error Driven Inference Engine\n");
    printf("Usage: %s <model.gguf> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -p <prompt>     Input prompt (default: \"Hello\")\n");
    printf("  -n <tokens>     Number of tokens to generate (default: 128)\n");
    printf("  -t <threads>    Number of threads (default: 4)\n");
    printf("  -m              Enable meta-learning (default: on)\n");
    printf("  -s <rate>       Target skip rate 0.0-0.95 (default: 0.4)\n");
    printf("  --target-toks <n> Target throughput goal (default: 15)\n");
    printf("  --quality-floor <n> Minimum acceptable quality floor (default: 0.85)\n");
    printf("  --device <name>  Device profile (default: auto)\n");
    printf("  --profile <name> Preferred runtime profile (default: auto)\n");
    printf("  --no-meta       Disable meta-learning\n");
    printf("  --verbose       Show detailed stats\n");
    printf("  --benchmark     Run benchmark mode\n");
    printf("  --fast          Aggressive low-power mode\n");
    printf("  --lora <path>   Load LoRA adapter from safetensors\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = "Hello";
    int n_tokens = 128;
    int n_threads = 4;
    bool use_meta = true;
    float skip_rate = 0.40f;
    float target_tok_s = 15.0f;
    float quality_floor = 0.85f;
    char device_name[32] = "auto";
    char profile_name[32] = "auto";
    bool verbose = false;
    bool benchmark = false;
    bool fast_mode = false;
    const char *lora_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0) use_meta = true;
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) skip_rate = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--target-toks") == 0 && i + 1 < argc) target_tok_s = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--quality-floor") == 0 && i + 1 < argc) quality_floor = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) snprintf(device_name, sizeof(device_name), "%s", argv[++i]);
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) snprintf(profile_name, sizeof(profile_name), "%s", argv[++i]);
        else if (strcmp(argv[i], "--no-meta") == 0) use_meta = false;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else if (strcmp(argv[i], "--fast") == 0) fast_mode = true;
        else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) lora_path = argv[++i];
    }
    (void)n_threads;

    if (fast_mode) {
        vx_set_compute_level(VX_COMPUTE_ATTN_ONLY);
        use_meta = false;
        target_tok_s = 15.0f;
        quality_floor = 0.85f;
        snprintf(profile_name, sizeof(profile_name), "%s", "mobile-low");
    }

    printf("Veltrix Engine v0.1\n");
    printf("  Model: %s\n", model_path);
    printf("  Meta-learning: %s\n", use_meta ? "enabled" : "disabled");
    if (verbose) printf("  Verbose: enabled\n");
    if (fast_mode) printf("  Mode: fast low-power\n");
    printf("  Target tok/s: %.1f\n", target_tok_s);
    printf("  Quality floor: %.2f\n", quality_floor);
    printf("  Device hint: %s\n", device_name);
    printf("  Profile hint: %s\n", profile_name);
    if (use_meta) printf("  Target skip rate: %.0f%%\n", skip_rate * 100);
    printf("\n");

    vx_set_n_threads(n_threads);

    vx_model *model = NULL;
    vx_error err = vx_model_create(model_path, &model);
    if (err != VX_OK) {
        fprintf(stderr, "Error loading model: %d\n", err);
        return 1;
    }

    printf("Model config:\n");
    printf("  Layers: %d\n", model->config.n_layers);
    printf("  Heads: %d (KV: %d)\n", model->config.n_heads, model->config.n_kv_heads);
    printf("  Embedding: %d\n", model->config.n_embd);
    printf("  FF: %d\n", model->config.n_ff);
    printf("  Vocab: %d\n", model->config.n_vocab);

    if (model->tokenizer) {
        printf("  Tokenizer: %s (%d tokens, bos=%d eos=%d)\n",
               vx_tokenizer_type_name(model->tokenizer),
               model->tokenizer->vocab_size,
               model->tokenizer->special.bos_id,
               model->tokenizer->special.eos_id);
    } else {
        printf("  Tokenizer: none\n");
    }
    printf("\n");

    vx_device_profile device = vx_device_profile_auto();
    if (strcmp(device_name, "auto") != 0) {
        snprintf(device.name, sizeof(device.name), "%s", device_name);
    }

    vx_runtime_goal goal = {
        .target_tok_s = target_tok_s,
        .quality_floor = quality_floor,
        .ctx_limit = model->config.n_ctx > 0 ? model->config.n_ctx : 256,
    };
    if (fast_mode && goal.ctx_limit > 256) goal.ctx_limit = 256;

    vx_runtime_profile profile = vx_choose_runtime_profile(&model->config, &device, &goal);
    if (strcmp(profile_name, "auto") != 0) {
        if (strcmp(profile_name, "mobile-low") == 0) {
            snprintf(profile.name, sizeof(profile.name), "%s", "mobile-low");
            profile.quality_floor = quality_floor < 0.88f ? 0.88f : quality_floor;
            profile.n_threads = profile.n_threads > 4 ? 4 : profile.n_threads;
            profile.ctx_cap = profile.ctx_cap > 256 ? 256 : profile.ctx_cap;
            profile.compute_level = VX_COMPUTE_ATTN_ONLY;
        } else if (strcmp(profile_name, "mobile-skip") == 0) {
            snprintf(profile.name, sizeof(profile.name), "%s", "mobile-skip");
            profile.compute_level = VX_COMPUTE_SKIP;
            if (profile.ctx_cap > 128) profile.ctx_cap = 128;
        } else if (strcmp(profile_name, "balanced") == 0) {
            snprintf(profile.name, sizeof(profile.name), "%s", "balanced");
            profile.compute_level = VX_COMPUTE_EXACT;
        }
    }

    printf("Selected runtime profile: %s\n", profile.name);
    printf("  allow_run: %s\n", profile.allow_run ? "yes" : "no");
    printf("  expected tok/s: %.1f\n", profile.expected_tok_s);
    printf("  quality floor: %.2f\n", profile.quality_floor);
    printf("  threads: %d\n", profile.n_threads);
    printf("  context cap: %d\n", profile.ctx_cap);
    if (!profile.allow_run) {
        fprintf(stderr, "Requested runtime contract cannot be met on this device/profile.\n");
        vx_model_destroy(model);
        return 1;
    }

    if (profile.ctx_cap > 0 && model->config.n_ctx > profile.ctx_cap) {
        model->config.n_ctx = profile.ctx_cap;
        printf("  Applied context cap: %d\n", model->config.n_ctx);
    }
    vx_set_n_threads(profile.n_threads);
    vx_set_compute_level(profile.compute_level);
    if (profile.compute_level != VX_COMPUTE_EXACT) {
        use_meta = false;
    }

    // Load LoRA adapter if specified
    vx_lora lora;
    memset(&lora, 0, sizeof(lora));
    if (lora_path) {
        printf("Loading LoRA adapter: %s\n", lora_path);
        vx_error lerr = vx_lora_load(lora_path, model->config.n_layers, &lora);
        if (lerr != VX_OK) {
            fprintf(stderr, "Error loading LoRA: %d\n", lerr);
            vx_model_destroy(model);
            return 1;
        }
        printf("  Adapters: %d, Rank: %d, Alpha: %.1f\n",
               lora.n_adapters, lora.rank, lora.alpha);
        vx_error merr = vx_lora_merge(&lora, model);
        if (merr != VX_OK) {
            fprintf(stderr, "Error merging LoRA: %d\n", merr);
            vx_lora_destroy(&lora);
            vx_model_destroy(model);
            return 1;
        }
        printf("LoRA merged successfully.\n");
    }

    if (use_meta) {
        model->layer_predictors = calloc(model->config.n_layers, sizeof(vx_meta_predictor *));
        if (model->layer_predictors) {
            for (int i = 0; i < model->config.n_layers; i++) {
                model->layer_predictors[i] = vx_meta_predictor_create(model->config.n_embd);
            }
        }
        model->attn_pattern_predictor = vx_meta_predictor_create(model->config.n_embd);
        model->token_exit_predictor = vx_meta_predictor_create(model->config.n_embd);
    }

    float *logits = malloc(model->config.n_vocab * sizeof(float));
    if (!logits) {
        vx_model_destroy(model);
        return 1;
    }

    // Tokenize prompt
    int prompt_ids[4096];
    int prompt_len = 0;

    if (model->tokenizer) {
        prompt_len = vx_tokenizer_encode(model->tokenizer, prompt, prompt_ids, 4096);
        if (prompt_len <= 0) {
            prompt_ids[0] = model->tokenizer->special.bos_id;
            prompt_len = 1;
        }
    } else {
        prompt_ids[0] = 1;
        prompt_len = 1;
    }

    if (benchmark) {
        vx_set_forward_timing_enabled(1);
        // Benchmark mode: cold-start timing
        printf("Benchmarking %d tokens...\n", n_tokens);
        fflush(stdout);

        // Warmup
        vx_model_forward(model, prompt_ids, prompt_len, logits);
        int token = prompt_ids[prompt_len - 1];
        for (int i = 0; i < 3; i++) {
            vx_model_forward(model, &token, 1, logits);
            token = 0;
        }

        // Timed run
        double t0 = vx_time_now_us();

        int generated = 0;
        while (generated < n_tokens) {
            vx_model_forward(model, &token, 1, logits);

            float max_val = -1e30f;
            int max_idx = 0;
            for (int j = 0; j < model->config.n_vocab; j++) {
                if (logits[j] > max_val) { max_val = logits[j]; max_idx = j; }
            }
            token = max_idx;

            if (model->tokenizer && model->tokenizer->special.eos_id > 0 &&
                token == model->tokenizer->special.eos_id)
                break;

            generated++;
        }

        double total_us = vx_time_now_us() - t0;
        double avg_us = generated > 0 ? total_us / generated : 0.0;
        double tok_s = avg_us > 0.0 ? 1e6 / avg_us : 0.0;

        printf("\nResults:\n");
        printf("  Generated:    %d tokens\n", generated);
        printf("  Total time:   %.1f ms\n", total_us / 1000.0);
        printf("  Avg latency:  %.1f us/tok\n", avg_us);
        printf("  Throughput:   %.1f tok/s\n", tok_s);
        vx_print_forward_timing();

        free(logits);
        if (lora_path) vx_lora_destroy(&lora);
        vx_model_destroy(model);
        return 0;
    }

    printf("Generating %d tokens...\n\n", n_tokens);
    fflush(stdout);

    // Run prompt through model (prefill)
    if (prompt_len > 1) {
        vx_model_forward(model, prompt_ids, prompt_len, logits);
        for (int i = 0; i < prompt_len; i++) {
            if (model->tokenizer) {
                char *piece = vx_tokenizer_decode(model->tokenizer, &prompt_ids[i], 1);
                if (piece) { printf("%s", piece); free(piece); }
            }
        }
    } else {
        vx_model_forward(model, prompt_ids, prompt_len, logits);
        if (model->tokenizer) {
            char *piece = vx_tokenizer_decode(model->tokenizer, &prompt_ids[0], 1);
            if (piece) { printf("%s", piece); free(piece); }
        }
    }
    fflush(stdout);

    // Generate
    int token = prompt_ids[prompt_len - 1];
    int generated = 0;

    while (generated < n_tokens) {
        vx_model_forward(model, &token, 1, logits);

        float max_val = -1e30f;
        int max_idx = 0;
        for (int j = 0; j < model->config.n_vocab; j++) {
            if (logits[j] > max_val) { max_val = logits[j]; max_idx = j; }
        }
        token = max_idx;

        if (model->tokenizer && model->tokenizer->special.eos_id > 0 &&
            token == model->tokenizer->special.eos_id)
            break;

        if (model->tokenizer) {
            char *piece = vx_tokenizer_decode(model->tokenizer, &token, 1);
            if (piece) { printf("%s", piece); free(piece); }
        }

        generated++;
        if (generated % 10 == 0) fflush(stdout);
    }

    printf("\n\nDone. Generated %d tokens.\n", generated);

    free(logits);
    if (lora_path) vx_lora_destroy(&lora);
    vx_model_destroy(model);
    return 0;
}
