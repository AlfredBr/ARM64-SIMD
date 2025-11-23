# ARM64-SIMD Demo

A .NET 10 console application that explores four increasingly optimized implementations of the same CPU-intensive task:

1. **Scalar (single thread)** – traditional loop, easiest to understand.
2. **Parallel scalar** – partitions the scalar loop across CPU cores via `Parallel.For`.
3. **Parallel + SIMD** – uses `System.Numerics.Vector<float>` inside the parallel loop so each core processes multiple elements per instruction (works on Intel SSE/AVX and ARM AdvSIMD).
4. **GPU (via ILGPU)** – launches the same logistic-map update on any CUDA-capable NVIDIA GPU via ILGPU, so the workload can be offloaded onto a supported NVIDIA GPU on both Intel and ARM boxes.  (BTW: [ILGPU](https://ilgpu.net/) is awesome!  It is the only C# GPGPU framework I've found that works on both Windows and Linux, Intel and ARM alike.)

The workload iterates a chaotic logistic-map function for each element of a generated float array. Chaotic math is branch-free but multiplies aggressively, making it a good fit for SIMD demonstrations.

## Hardware used for testing

| CPU | Cores | CPU SIMD Width | RAM | GPU | GPU Architecture | SM Count | CUDA Cores per SM | Total CUDA Cores | Cost |
|-----|:-----:|:--------------:|----:|:---:|:----------------:|:--------:|:-----------------:|:----------------:|-----:|
| Intel i7-12700K | 12/20 | 8 lanes × 32-bits ([AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)) | 32 GB | RTX 3080 | Ampere | 68 | 128 | 8,704 | $2,000 |
| ARM Cortex-A57 | 4 | 4 lanes × 32-bits ([NEON](https://www.arm.com/technologies/neon)) | 4 GB | onboard | Maxwell | 1 | 128 | 128 | $100 |

## Sample Results
### Intel SIMD : width = 8 lanes, dataset = 4,000,000 floats x 150 chaos iterations
| Mode   | Time (ms) | Throughput (M iter/s) | Checksum       |
|--------------------|-----------:|-----------------------:|----------------:|
| Scalar (1 thread)  | 1,083.5   | 553.8                 | 2185294.1219   |
| Parallel (1 thread per core)| 88.0      | 6,817.8               | 2185294.1219   |
| Parallel + SIMD (8 lanes x 32-bits)   | 11.0      | 54,713.1              | 2185294.1219   |
| GPU (ILGPU, CUDA, NVIDIA RTX3080) | 9.9 | 60,752.1            | 2185294.1219   |

### ARM64 SIMD : width = 4 lanes, dataset = 4,000,000 floats x 150 chaos iterations
| Mode   | Time (ms) | Throughput (M iter/s) | Checksum       |
|--------------------|-----------:|-----------------------:|----------------:|
| Scalar (1 thread)  |    6,871.4  |  87.3  |  2185151.8441 |
| Parallel (1 thread per core) |   1,780.0  |  337.1  |  2185151.8441 |
| Parallel + SIMD (4 lanes x 32-bits)    |     473.6   |  1,266.9  |  2185151.8441 |
| GPU (ILGPU, CUDA, NVIDIA Maxwell) | 208.1   |  2,883.3  |  2185151.8441 |

### Observations
- Throughput = (dataset length × iterations) / time in seconds
- Moving from scalar to parallel execution yields a significant speedup roughly proportional to the number of CPU cores.
- Adding SIMD on top of parallelism further amplifies throughput by processing multiple data points per instruction.
- The GPU implementation achieves the highest throughput, although the advantage is less pronounced on smaller datasets due to data transfer overheads.  The GPU performance gain can be quite surprising on larger workloads.

## How It Works

### Data generation
- A `float` is 32-bits.
- `DataFactory.Create` fills an array with deterministic values derived from `sin(i * 0.000123)` so every run is reproducible.
- Values are constrained to `(0, 1)` so the logistic map stays numerically stable.

### Benchmark harness
- Command-line options: `--length=<ints>` (default 4,000,000), `--iterations=<ints>` (default 150), and `--warmup=<ints>` (default 1).
- Each workload is warmed up briefly, timed with `Stopwatch`, and reported with throughput (total iterations / seconds) and a checksum to verify identical math.

### Workload variants
- **Scalar**: tight `for` loop calling `IterateScalar` per element.
- **Parallel scalar**: uses `Parallel.For` with per-thread local accumulation, then locks to combine results. Eliminates most wall-clock latency for embarrassingly parallel work but still performs one float at a time.
- **Parallel + SIMD**: partitions the array by vector-sized ranges using `Partitioner.Create`. Each worker loads `Vector<float>` (lane count is auto-detected) and repeatedly applies `IterateSimd`. Tail elements that do not fit a full vector fall back to the scalar routine.
- **GPU (ILGPU)**: copies the dataset into an ILGPU accelerator (preferring CUDA) and runs the logistic recurrence inside a GPU kernel written in C#, then copies results back for checksum verification.

### Why performance improves
1. **Scalar ➔ Parallel**: Genuine concurrency. Each core receives a disjoint slice of the array, so wall-clock time roughly divides by the number of cores minus synchronization overhead.
2. **Parallel ➔ Parallel+SIMD**: Each core now updates `Vector<float>.Count` elements per instruction. On 64-bit hardware this is typically 8 lanes (256-bit) but automatically adapts to wider vectors on capable CPUs. Processing in chunks reduces load/store traffic and lets the CPU pipeline more fused multiply-adds per cycle.
3. **Parallel+SIMD ➔ GPU**: The GPU massively parallel architecture can handle thousands of threads simultaneously, each executing the same logistic map kernel. With high memory bandwidth and many ALUs, the GPU can outperform even SIMD-optimized CPU code on large datasets.

### Issues we encountered during development (and fixes)
- **Checksum mismatch**: The scalar logistic math originally multiplied in a different order than the SIMD vectorized function, producing different rounding noise. Aligning the scalar implementation to compute `value * (1 - value)` before multiplying by the constant ensured identical checksums and makes correctness comparisons meaningful.
- **Interlocked with doubles**: `Interlocked.Add` only supports integers on older frameworks, so we replaced it with a lightweight `lock` to combine per-thread `double` sums safely.
- **SIMD slower than parallel scalar**: The first SIMD attempt assigned single vectors to tasks, causing extreme scheduling overhead and poor cache locality. Switching to `Partitioner.Create` with chunk sizes tied to `ProcessorCount` let each worker chew through contiguous vector ranges before synchronizing, reducing contention and finally delivering the expected >50x speedup in Release builds. (NOTE: This really surprised me and I would not have discoved this without the help of AI/ChatGPT.  Amazing!)


## Building and Running

```powershell
# Build (Debug)
dotnet build

# Run with defaults (Debug)
dotnet run

# Run a release build with large workload
dotnet run -c Release -- --length=8000000 --iterations=512
```

> **GPU prerequisites:** Install the NVIDIA driver and CUDA toolkit appropriate for your platform. ILGPU will automatically fall back to a CPU accelerator if no CUDA device is detected. See [GPU.md](GPU.md) for full instructions.

Sample Release output on the authoring machine (8-lane SIMD):

```
ARM/Intel SIMD demo — dataset 4,000,000 floats x 150 chaos iterations
SIMD width: 8 lanes (hardware accelerated)
Scalar (1 thread)      1,089.2 ms  |  throughput 550.9 M it/s    |  checksum 2185294.1219
Parallel (per core)       84.0 ms  |  throughput 7,145.4 M it/s  |  checksum 2185294.1219
Parallel + SIMD           11.0 ms  |  throughput 54,746.0 M it/s |  checksum 2185294.1219
GPU (ILGPU, RTX)          14.0 ms  |  throughput 42,923.7 M it/s |  checksum 2185294.1219
```

Throughput differences illustrate how the performance benefits of multi-core execution and vectorization compound. Results vary with CPU architecture and thermal conditions, but the checksum should stay constant if the hardware supports IEEE-754 single-precision correctly.



## Tips for experimentation
- Adjust `--length` or `--iterations` to emphasize either memory bandwidth (`length`) or compute intensity (`iterations`).
- Compare Debug vs Release. JIT optimizations (loop unrolling, constant folding) are essential for SIMD performance.
- Inspect `Vector<float>.Count` on different machines to see how Intel AVX-512 or ARM SVE-ready chips expose wider lanes without code changes. (NOTE: I've tested this on Intel AVX2 and ARM NEON hardware only so far, but it should work seamlessly on wider SIMD hardware.  `Vector<T>.Count` is the number of lanes on your CPU and the code adapts automatically at runtime.  If you run this on an AVX-512 capable CPU, you should see `Vector<float>.Count` equal to 16 x 32-bit lanes!)



## Troubleshooting
- **Is Release run slower than expected?** Ensure the process is 64-bit and your CPU is not thermally throttled. Running from Visual Studio with the debugger attached may force Debug optimizations off.
- **Is SIMD accelerated on your hardware?** `Vector.IsHardwareAccelerated` returns false on very old CPUs or when targeting x86 without enabling SSE2. Switch to `net10.0` (already configured) and run on 64-bit.
- **Running out-of-memory?** Reduce `--length` or iterations; the default array consumes ~16 MB (4 million * 4 bytes).
- **Is the GPU benchmark missing?** Verify the CUDA driver/toolkit installation (`nvidia-smi`) and that the process has permission to access the device. ILGPU logs which accelerator it picked at startup.

Feel free to adapt `Program.cs` to plug in other math kernels (Fourier transforms, image filters, etc.) to explore how SIMD-friendly algorithms scale across Intel and ARM alike.

## License
Released under the [MIT License](LICENSE).
