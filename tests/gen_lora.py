"""Generate synthetic LoRA adapter for testing."""
import safetensors
import safetensors.numpy
import numpy as np
import json
import os

out_dir = os.path.dirname(__file__) or '.'

def gen_lora(n_layers=2, n_embd=64, n_ff=128, rank=4, alpha=8.0):
    tensors = {}

    for l in range(n_layers):
        for mod, dim_out in [("q", n_embd), ("k", n_embd // 4), ("v", n_embd // 4),
                             ("o", n_embd), ("gate", n_ff), ("up", n_ff), ("down", n_embd)]:
            d_in = n_embd if mod != "down" else n_ff
            d_out = dim_out

            proj_name = {
                "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
                "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
            }[mod]

            # LoRA A shape: [d_in, rank]
            a_data = np.random.randn(d_in, rank).astype(np.float32) * 0.02
            # LoRA B shape: [rank, d_out]
            b_data = np.random.randn(rank, d_out).astype(np.float32) * 0.02

            prefix = f"base_model.model.model.layers.{l}.self_attn" if mod in ("q","k","v","o") else f"base_model.model.model.layers.{l}.mlp"

            tensors[f"{prefix}.{proj_name}.lora_A.weight"] = a_data
            tensors[f"{prefix}.{proj_name}.lora_B.weight"] = b_data

    config = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    st_path = os.path.join(out_dir, "test_lora.safetensors")
    safetensors.numpy.save_file(tensors, st_path)

    cfg_path = os.path.join(out_dir, "adapter_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created {st_path} ({os.path.getsize(st_path)} bytes)")
    print(f"Created {cfg_path}")
    print(f"Tensors: {len(tensors)} ({len(tensors)//2} LoRA pairs)")
    print(f"Rank: {rank}, Alpha: {alpha}")

    # Verify
    loaded = safetensors.numpy.load_file(st_path)
    print(f"Verified: {len(loaded)} tensors loaded")

if __name__ == "__main__":
    gen_lora(n_layers=2, n_embd=64, n_ff=128, rank=4, alpha=8.0)
