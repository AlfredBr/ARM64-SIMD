# GPU Acceleration Guide

This project includes an optional GPU benchmark implemented with [ILGPU](https://www.ilgpu.net/). The kernel mirrors the CPU logistic-map workload but executes entirely on a CUDA-capable NVIDIA device. ILGPU compiles the C# kernel at runtime, so no separate `.cu` files are required.

## Requirements

| Platform | Minimum requirements |
|----------|----------------------|
| Windows  | NVIDIA driver + CUDA Toolkit 12.x (matching your GPU), .NET SDK 10.0, ILGPU NuGet packages (restored automatically). |
| Linux (x86_64 or ARM64) | NVIDIA driver (CUDA), `nvidia-smi` accessible, .NET SDK 10.0, ILGPU packages. |

> ILGPU falls back to a CPU accelerator if no GPU is available, so the program still runs but will log that the GPU benchmark was skipped.

## Setup steps

1. **Install CUDA**
   - Windows: use the official CUDA installer, ensuring the driver matches your GPU.
   - Linux: install the NVIDIA driver + CUDA packages from your distribution or NVIDIA repo.
2. **Verify device access**
   - Run `nvidia-smi` to confirm the GPU is visible and idle enough for compute jobs.
3. **Restore and build**
   - `dotnet restore`
   - `dotnet run -c Release`
4. **Interpret the output**
   - The app prints a line like `GPU (ILGPU): using NVIDIA GeForce RTX 3080 (Cuda).`
   - A `GPU (ILGPU)` row appears alongside the CPU benchmarks with matching checksum.

## Tips

- Multiple GPUs: set `ILGPU_DEFAULT_ACCELERATOR` to the desired accelerator type (`CUDA`, `OpenCL`, `CPU`) or use standard ILGPU environment variables to select a specific device index.
- Warmup: the shared benchmark harness runs a shorter warmup iteration before timing; expect a small one-time kernel compilation delay on the first GPU invocation.
- Troubleshooting: if ILGPU reports no compatible accelerator, double-check driver installation, ensure the process is 64-bit, and confirm the CUDA toolkit version matches your GPU architecture.

Feel free to swap in different kernels inside `Gpu/GpuKernels.cs` to experiment with other GPU-friendly algorithms.
