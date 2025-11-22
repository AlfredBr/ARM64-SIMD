# ARM64-SIMD Demo

A .NET 10 console application that explores four increasingly optimized implementations of the same CPU-intensive task:

1. **Scalar (single thread)** – traditional loop, easiest to understand.
2. **Parallel scalar** – partitions the scalar loop across CPU cores via `Parallel.For`.
3. **Parallel + SIMD** – uses `System.Numerics.Vector<float>` inside the parallel loop so each core processes multiple elements per instruction (works on Intel SSE/AVX and ARM AdvSIMD).
4. **GPU (ILGPU)** – launches the same logistic-map update on any CUDA-capable NVIDIA GPU via ILGPU, so the workload can be offloaded on both Windows and Linux boxes.

The workload iterates a chaotic logistic-map function for each element of a generated float array. Chaotic math is branch-free but multiplies aggressively, making it a good fit for SIMD demonstrations.

---

## How It Works

### Data generation
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

### Issues encountered (and fixes)
- **Checksum mismatch**: The scalar logistic math originally multiplied in a different order than the SIMD vectorized function, producing different rounding noise. Aligning the scalar implementation to compute `value * (1 - value)` before multiplying by the constant ensured identical checksums and makes correctness comparisons meaningful.
- **Interlocked with doubles**: `Interlocked.Add` only supports integers on older frameworks, so we replaced it with a lightweight `lock` to combine per-thread `double` sums safely.
- **SIMD slower than parallel scalar**: The first SIMD attempt assigned single vectors to tasks, causing extreme scheduling overhead and poor cache locality. Switching to `Partitioner.Create` with chunk sizes tied to `ProcessorCount` let each worker chew through contiguous vector ranges before synchronizing, reducing contention and finally delivering the expected >50x speedup in Release builds.

---

## Building and Running

```powershell
# Build (Debug)
dotnet build

# Run with defaults (Debug)
dotnet run

# Common demo: Release build with large workload
dotnet run -c Release -- --length=4000000 --iterations=150

# Run Release build while letting ILGPU pick a CUDA device
dotnet run -c Release
```

> **GPU prerequisites:** Install the NVIDIA driver and CUDA toolkit appropriate for your platform. ILGPU will automatically fall back to a CPU accelerator if no CUDA device is detected. See [GPU.md](GPU.md) for full instructions.

Sample Release output on the authoring machine (8-lane SIMD):

```
ARM/Intel SIMD demo — dataset 4,000,000 floats x 150 chaos iterations
SIMD width: 8 lanes (hardware accelerated)
Scalar (1 thread)      1,089.2 ms  |  throughput 550.9 M it/s  |  checksum 2185294.1219
Parallel (per core)       84.0 ms  |  throughput 7,145.4 M it/s  |  checksum 2185294.1219
Parallel + SIMD           11.0 ms  |  throughput 54,746.0 M it/s |  checksum 2185294.1219
GPU (ILGPU, RTX)          14.0 ms  |  throughput 42,923.7 M it/s |  checksum 2185294.1219
```

Throughput differences illustrate how multi-core execution and vectorization compound. Results vary with CPU architecture and thermal conditions, but the checksum should stay constant if the hardware supports IEEE-754 single-precision correctly.

---

## Tips for experimentation
- Adjust `--length` or `--iterations` to emphasize either memory bandwidth (`length`) or compute intensity (`iterations`).
- Compare Debug vs Release. JIT optimizations (loop unrolling, constant folding) are essential for SIMD performance.
- Inspect `Vector<float>.Count` on different machines to see how Intel AVX-512 or ARM SVE-ready chips expose wider lanes without code changes.

---

## Troubleshooting
- **Release run slower than expected**: Ensure the process is 64-bit and your CPU is not thermally throttled. Running from Visual Studio with the debugger attached may force Debug optimizations off.
- **SIMD not hardware accelerated**: `Vector.IsHardwareAccelerated` returns false on very old CPUs or when targeting x86 without enabling SSE2. Switch to `net10.0` (already configured) and run on 64-bit.
- **Out-of-memory**: Reduce `--length` or iterations; the default array consumes ~16 MB (4 million * 4 bytes).
- **GPU benchmark missing**: Verify the CUDA driver/toolkit installation (`nvidia-smi`) and that the process has permission to access the device. ILGPU logs which accelerator it picked at startup.

Feel free to adapt `Program.cs` to plug in other math kernels (Fourier transforms, image filters, etc.) to explore how SIMD-friendly algorithms scale across Intel and ARM alike.

## License
Released under the [MIT License](LICENSE).
