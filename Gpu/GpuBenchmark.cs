using System;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;

namespace ARM64_SIMD.Gpu;

internal sealed class GpuBenchmark : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, int, float> _kernel;

    private GpuBenchmark(Context context, Accelerator accelerator)
    {
        _context = context;
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, int, float>(GpuKernels.Logistic);
    }

    public static GpuBenchmark? TryCreate(out string statusMessage)
    {
        try
        {
            var context = Context.Create(builder => builder.Default());
            var device = context.Devices
                .Where(device => device.AcceleratorType != AcceleratorType.CPU)
                .OrderBy(device => device.AcceleratorType == AcceleratorType.Cuda ? 0 : 1)
                .FirstOrDefault();

            if (device is null)
            {
                context.Dispose();
                statusMessage = "GPU (ILGPU): no compatible accelerator detected. Skipping GPU benchmark.";
                return null;
            }

            var accelerator = device.CreateAccelerator(context);
            statusMessage = $"GPU (ILGPU): using {accelerator.Name} ({accelerator.AcceleratorType}).";
            return new GpuBenchmark(context, accelerator);
        }
        catch (Exception ex)
        {
            statusMessage = $"GPU (ILGPU) unavailable: {ex.Message}";
            return null;
        }
    }

    public double Execute(float[] source, int iterations)
    {
        if (source.Length == 0)
        {
            return 0;
        }

        using var deviceBuffer = _accelerator.Allocate1D<float>(source.Length);
        deviceBuffer.CopyFromCPU(source);

        _kernel(deviceBuffer.IntExtent, deviceBuffer.View, iterations, Workloads.LogisticMultiplier);
        _accelerator.Synchronize();

        var gpuResults = new float[source.Length];
        deviceBuffer.CopyToCPU(gpuResults);

        double sum = 0;
        for (int i = 0; i < gpuResults.Length; i++)
        {
            sum += gpuResults[i];
        }

        return sum;
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
