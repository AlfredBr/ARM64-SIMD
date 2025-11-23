using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using ARM64_SIMD.Gpu;

var config = DemoConfig.FromArgs(args);
Console.WriteLine(
    $"ARM/Intel SIMD demo — dataset {config.DataLength:N0} floats x {config.Iterations} chaos iterations");

var data = DataFactory.Create(config.DataLength);
Console.WriteLine(
    $"SIMD width: {Vector<float>.Count} lanes ({(Vector.IsHardwareAccelerated ? "hardware" : "software")} accelerated)");

var benchmarks = new (string Label, Func<float[], int, double> Workload)[]
{
    ("Scalar (1 thread)", Workloads.Scalar),
    ("Parallel (per core)", Workloads.ParallelScalar),
    ("Parallel + SIMD", Workloads.ParallelSimd)
};

foreach (var (label, workload) in benchmarks)
{
    var result = Benchmark.Run(label, workload, data, config);
    Print(result);
}

var gpuBenchmark = GpuBenchmark.TryCreate(out var gpuStatusMessage);
if (!string.IsNullOrWhiteSpace(gpuStatusMessage))
{
    Console.WriteLine(gpuStatusMessage);
}

if (gpuBenchmark is not null)
{
    using (gpuBenchmark)
    {
        var gpuResult = Benchmark.Run("GPU (ILGPU)", gpuBenchmark.Execute, data, config);
        Print(gpuResult);
    }
}

static void Print(BenchmarkResult result)
{
    Console.WriteLine(
        $"{result.Label,-20} {result.Elapsed.TotalMilliseconds,9:N1} ms  |  throughput {result.Throughput / 1_000_000:N1} M it/s  |  checksum {result.Sum:F4}");
}

internal sealed record DemoConfig(int DataLength, int Iterations, int WarmupIterations)
{
    public static DemoConfig FromArgs(string[] args)
    {
        int length = 4_000_000;
        int iterations = 150;
        int warmup = 1;

        foreach (var arg in args)
        {
            if (TryParse(arg, "length", out var parsedLength))
            {
                length = parsedLength;
            }
            else if (TryParse(arg, "iterations", out var parsedIterations))
            {
                iterations = parsedIterations;
            }
            else if (TryParse(arg, "warmup", out var parsedWarmup))
            {
                warmup = parsedWarmup;
            }
        }

        return new DemoConfig(length, iterations, Math.Max(0, warmup));
    }

    private static bool TryParse(string arg, string name, out int value)
    {
        if (arg.StartsWith("--" + name + "=", StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(arg[(name.Length + 3)..], out value))
        {
            return true;
        }

        value = default;
        return false;
    }
}

internal static class DataFactory
{
    public static float[] Create(int length)
    {
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
        {
            float value = MathF.Sin(i * 0.000123f) * 0.5f + 0.5f; // keep values inside (0, 1)
            data[i] = value;
        }

        return data;
    }
}

internal static class Benchmark
{
    public static BenchmarkResult Run(string label, Func<float[], int, double> workload, float[] data, DemoConfig config)
    {
        int warmupIterations = Math.Clamp(config.WarmupIterations, 0, 5);
        for (int i = 0; i < warmupIterations; i++)
        {
            workload(data, Math.Max(1, config.Iterations / 10));
        }

        var stopwatch = Stopwatch.StartNew();
        double sum = workload(data, config.Iterations);
        stopwatch.Stop();

        double totalIterations = (double)config.DataLength * config.Iterations;
        double throughput = totalIterations / stopwatch.Elapsed.TotalSeconds;
        return new BenchmarkResult(label, sum, stopwatch.Elapsed, throughput);
    }
}

internal sealed record BenchmarkResult(string Label, double Sum, TimeSpan Elapsed, double Throughput);

internal static class Workloads
{
    internal const float LogisticMultiplier = 3.96f;
    private static readonly Vector<float> LogisticMultiplierVector = new(LogisticMultiplier);
    private static readonly Vector<float> OneVector = Vector<float>.One;

    public static double Scalar(float[] data, int iterations)
    {
        double sum = 0;
        for (int i = 0; i < data.Length; i++)
        {
            sum += IterateScalar(data[i], iterations);
        }

        return sum;
    }

    public static double ParallelScalar(float[] data, int iterations)
    {
        double total = 0;
        object gate = new();

        Parallel.For<double>(0, data.Length,
            () => 0,
            (int i, ParallelLoopState _, double localSum) => localSum + IterateScalar(data[i], iterations),
            localSum => AddThreadSafe(ref total, localSum, gate)
        );

        return total;
    }

    public static double ParallelSimd(float[] data, int iterations)
    {
        int simdWidth = Vector<float>.Count;
        int vectorLength = data.Length / simdWidth;
        int chunkSize = Math.Max(1, vectorLength / (Environment.ProcessorCount * 4));

        double total = 0;
        object gate = new();

        Parallel.ForEach(
            Partitioner.Create(0, vectorLength, chunkSize),
            () => 0.0,
            (range, _, localSum) =>
            {
                for (int vectorIndex = range.Item1; vectorIndex < range.Item2; vectorIndex++)
                {
                    int offset = vectorIndex * simdWidth;
                    var value = new Vector<float>(data, offset);
                    for (int iter = 0; iter < iterations; iter++)
                    {
                        value = IterateSimd(value);
                    }

                    localSum += SumVector(value);
                }

                return localSum;
            },
            localSum => AddThreadSafe(ref total, localSum, gate)
        );

        for (int i = vectorLength * simdWidth; i < data.Length; i++)
        {
            total += IterateScalar(data[i], iterations);
        }

        return total;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float IterateScalar(float value, int iterations)
    {
        float current = value;
        for (int i = 0; i < iterations; i++)
        {
            current = LogisticStep(current);
        }

        return current;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> IterateSimd(Vector<float> value)
    {
        var oneMinusValue = Vector.Subtract(OneVector, value);
        var product = Vector.Multiply(value, oneMinusValue);
        return Vector.Multiply(LogisticMultiplierVector, product);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float LogisticStep(float value)
    {
        float product = value * (1f - value);
        return LogisticMultiplier * product;
    }

    private static double SumVector(Vector<float> value)
    {
        double sum = 0;
        for (int lane = 0; lane < Vector<float>.Count; lane++)
        {
            sum += value[lane];
        }

        return sum;
    }

    private static void AddThreadSafe(ref double target, double value, object gate)
    {
        lock (gate)
        {
            target += value;
        }
    }
}
