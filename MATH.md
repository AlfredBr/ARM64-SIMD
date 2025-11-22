# Mathematical Kernel: Chaotic Logistic Map

This demo revolves around repeatedly evaluating the logistic map, a classic nonlinear recurrence that exhibits chaotic behavior:

```
x_{n+1} = r * x_n * (1 - x_n)
```

- `x` is a real number in the interval (0, 1).
- `r` is a constant multiplier (here `r = 3.96`).
- Each iteration multiplies three floating-point terms: the current state `x`, its complement `1 - x`, and the control parameter `r`.

## Why it is compute intensive

1. **High iteration counts** — Every element in the dataset (default 4,000,000 floats) is advanced through the recurrence `iterations` times (default 150). That yields 600 million logistic updates per run, each involving multiple floating-point operations.
2. **Dependency chain** — The next value depends on the previous one, so the computation for each element cannot be unrolled across iterations without keeping intermediate state; the CPU must perform each multiply/add sequentially per element.
3. **No early exits** — Unlike many workloads, chaotic iteration requires the same number of steps for every element, meaning there is no divergent control flow that could shorten computation. This makes total work proportional to `length × iterations`.
4. **Floating-point sensitivity** — Chaotic systems amplify rounding differences. To keep results deterministic, both scalar and SIMD paths must execute the multiplies in the same order. That constraint rules out fused or reordered operations that might otherwise speed things up, ensuring each update still requires separate multiply/subtract steps.
5. **Memory bandwidth pressure** — The kernel streams through a large contiguous array. Each element must be loaded, iterated in registers, then summed into a checksum. With millions of entries, the CPU constantly pulls data through caches, stressing both compute and memory subsystems.

## SIMD suitability

The logistic update consists of basic arithmetic with no branches and no data-dependent memory accesses. Those properties make it ideal for Single Instruction Multiple Data execution: the same multiply-subtract-multiply sequence applies to every element, letting the processor apply the recurrence simultaneously to multiple lanes (`Vector<float>.Count` values at once). However, because each lane still requires 150 dependent iterations, the workload remains compute-heavy even under SIMD; vectorization simply amortizes the cost across several values per instruction.
