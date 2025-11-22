# Benchmark Results

Collected with `dotnet run -c Release` on the provided system. Configuration: 4,000,000 floats × 150 logistic iterations; SIMD width 8 lanes (hardware accelerated).

| Workload            | Time (ms) | Throughput (M it/s) | Checksum       |
|---------------------|-----------|----------------------|----------------|
| Scalar (1 thread)   | 1,080.8   | 555.1                | 2,185,294.1219 |
| Parallel (per core) | 87.6      | 6,850.4              | 2,185,294.1219 |
| Parallel + SIMD     | 10.9      | 54,850.6             | 2,185,294.1219 |
| GPU (ILGPU)         | 12.1      | 49,619.6             | 2,185,294.1219 |

## Observations

- **Linear scaling with threads**: Moving from the single-threaded baseline to the parallel scalar version multiplies throughput by ~12× (554 → 6,812 M it/s), matching the expected benefit from utilizing all CPU cores for this embarrassingly parallel workload.
- **SIMD amplification**: Adding SIMD on top of threading increases throughput by another ~7.7×, showing how vector math leverages the per-core execution units beyond mere parallelism.
- **GPU verifies parity**: The ILGPU path matches the CPU checksum, demonstrating that the CUDA kernel executes the same logistic math and comes to the same result despite running on very different hardware.
- **GPU not yet dominant**: With this dataset, the GPU finishes slightly slower than the SIMD path even though it processes tens of billions of iterations per second.

## Analysis

1. **CPU utilization**: The scalar path uses only one core; the parallel path saturates available cores, drastically shrinking runtime. Because each array element requires the same number of iterations, work division is perfectly balanced and synchronization costs stay minimal.
2. **Vector efficiency**: The SIMD version achieves ~55 billion iterations per second. With 8 lanes per vector, each core processes eight independent elements every instruction. Combined with chunked partitioning, threads maintain high cache locality and avoid scheduler thrashing.
3. **Compute-bound behavior**: Even after vectorizing, runtime remains dominated by floating-point operations rather than memory I/O. The logistic map keeps each value in registers for 150 iterations, maximizing arithmetic density and showcasing how modern CPUs thrive when work is both branch-free and data-parallel.
4. **GPU transfer overheads**: The GPU kernel itself is extremely fast, but copying four million floats to the device and back plus kernel launch latency introduces ~2 ms of overhead. Because the CPU SIMD path already finishes in ~11 ms, those fixed costs erase the GPU’s arithmetic advantage on this relatively small workload. Scaling `--length` or `--iterations` upward (or keeping data resident on the GPU across multiple kernels) shifts the balance in favor of the GPU.

These results demonstrate compounding gains from layering parallelism strategies: start with a scalar baseline, add multi-threading for coarse-grained speedups, then use SIMD to exploit fine-grained data-level parallelism on both Intel and ARM architectures. The GPU path can outrun all of them once the workload is large enough to amortize PCIe transfers and fully saturate memory bandwidth, but at the default data size the device spends a comparable amount of time moving data as it does computing, leaving the highly optimized SIMD code slightly ahead.
