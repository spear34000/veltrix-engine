# veltrix-engine

## Performance contract

Veltrix is being built as a runtime that can select a bounded execution profile for low-spec hardware before generation starts. The goal is to sustain targets like 15 tok/s on devices such as phones and N100-class mini PCs by reducing context length, thread count, and compute depth when the contract allows it.

Example:

```bash
veltrix model.gguf --target-toks 15 --quality-floor 0.85 --device auto --profile mobile-low
```

If the selected device and model cannot meet the requested throughput floor, Veltrix refuses to pretend otherwise and exits with a clear failure.
