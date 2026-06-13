# Runtime Profiles

Veltrix is moving toward a performance-contract runtime, not a best-effort demo mode.

The selection flow is:

1. Read model metadata.
2. Detect a device profile.
3. Compare the requested throughput floor and quality floor against the model and device.
4. Select a bounded runtime profile.
5. Enforce that profile before generation starts.

## Low-spec target

For phones and N100/N150-class mini PCs, the low-spec profile is designed around 1B to 3B models and a target of 15 tok/s or better.

The profile trims:

- context length
- thread count
- compute depth
- meta-learning overhead when it is not worth the cost

The objective is to keep throughput stable instead of starting with a full-context, full-depth configuration and hoping the hardware keeps up.

## Why this differs from other engines

Most local inference engines expose performance tuning as a manual knob set:

- one tool may optimize for ease of use
- another may optimize for raw kernel quality
- another may expose many model settings but leave the operator to find a safe combination

Veltrix is being shaped differently:

- it chooses a bounded runtime profile up front
- it can reject an execution request if the contract is unrealistic
- it keeps the profile visible in the CLI and runtime
- it uses the same contract for benchmarking and generation

That makes the low-spec target a runtime policy, not a user guess.
