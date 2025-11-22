# Prompt

This project is inspired by the final homework assignment I was given when I was first trying to learn X86 MMX instructions more than 30 years ago.  What you see here was 90% built by ChatGPT 5.1 Codex.  I tweaked the code in a few places and guided the AI via a series of prompts.  Lastly, I asked ChatGPT to take everything we have done together and generate a new, cohesive prompt that someone could use to reproduce this entire project.  It might be a interesting experiment to re-send this prompt to ChatGPT (or Claude) to see what it comes up with.

## Below is what ChatGPT generated as a prompt, unedited.

You are GPT-5.1-Codex (Preview), acting as an expert .NET performance engineer. Reproduce the ARM/Intel SIMD demo from the ARM64-SIMD repository with the following characteristics:

1. **Project setup**
   - Create a single .NET 10 console application named `ARM64-SIMD`.
   - The entry point lives in `Program.cs`.
   - Do not use external packages beyond the BCL.

2. **Workload**
   - Generate a float array (`length` default 4_000_000) using a deterministic sine-based seed scaled to the (0, 1) range.
   - For each element, iterate the logistic map `x = r * x * (1 - x)` with `r = 3.96` for `iterations` steps (default 150).
   - Provide a warmup loop (default 1 iteration) before timing.

3. **Implement three benchmarks**
   1. Scalar single-threaded reference.
   2. Parallel scalar using `Parallel.For` with per-thread local sums and a thread-safe accumulator.
   3. Parallel + SIMD using `System.Numerics.Vector<float>` within a partitioned parallel loop. Tail elements that do not fit into a vector fall back to the scalar implementation.
   - The SIMD path must work on both Intel (SSE/AVX/AVX-512) and ARM (AdvSIMD/SVE) by relying on `Vector<T>` width.

4. **Benchmark harness**
   - Parse `--length`, `--iterations`, and `--warmup` CLI arguments (all optional).
   - Report each benchmark’s elapsed time in milliseconds, throughput in millions of iterations per second, and a checksum (sum of final values) so correctness can be compared across implementations.
   - Clearly print the detected SIMD width and whether hardware acceleration is available.

5. **Performance considerations**
   - Ensure the scalar and SIMD logistic functions perform the multiplications in the same order to keep checksums identical.
   - Use chunked range partitioning (e.g., `Partitioner.Create`) in the SIMD workload to reduce scheduling overhead and improve cache locality.
   - Fall back gracefully if hardware acceleration is unavailable.

6. **Documentation**
   - Write a `README.md` summarizing the project, describing each workload, sharing sample Release-mode results, and noting any pitfalls (checksum mismatch, `Interlocked.Add` missing double overloads, initial SIMD slowdown until chunking was added).
   - Mention how to run Debug vs Release builds and how to tweak parameters.

7. **Deliverables**
   - Output the full `Program.cs`, `README.md`, and any supporting files exactly as they should appear in the repo.
   - Provide instructions for building and running (`dotnet build`, `dotnet run`, `dotnet run -c Release -- --length=... --iterations=...`).

Strive for clear explanations, concise yet instructive comments, and deterministic results so that checksums match across implementations. When in doubt, prefer readability that still showcases best practices for .NET SIMD and parallelization on both Intel and ARM architectures.
