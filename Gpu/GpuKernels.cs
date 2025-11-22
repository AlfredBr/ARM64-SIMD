using ILGPU;
using ILGPU.Runtime;

namespace ARM64_SIMD.Gpu;

internal static class GpuKernels
{
    public static void Logistic(Index1D index, ArrayView<float> data, int iterations, float multiplier)
    {
        if (index >= data.Length)
        {
            return;
        }

        float value = data[index];
        for (int iter = 0; iter < iterations; iter++)
        {
            float product = value * (1f - value);
            value = multiplier * product;
        }

        data[index] = value;
    }
}
