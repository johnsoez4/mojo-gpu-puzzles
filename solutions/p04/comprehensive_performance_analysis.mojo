"""
Comprehensive performance analysis script for add_10_2d implementations.

This script tests multiple matrix sizes to analyze performance scaling
and identify the crossover point where GPU implementations become advantageous.
"""

from time import perf_counter_ns as now
from memory import UnsafePointer
from gpu.host import DeviceContext

alias dtype = DType.float32
alias NUM_BENCHMARK_RUNS = 5


fn add_10_2d_cpu(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """CPU-only implementation of add_10_2d function."""
    for row in range(size):
        for col in range(size):
            index = row * size + col
            output[index] = a[index] + 10.0


fn add_10_2d_gpu_unsafe(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using UnsafePointer."""
    from gpu import thread_idx

    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


fn benchmark_cpu_for_size(size: Int) -> Float64:
    """Benchmark CPU implementation for given size."""
    total_time: UInt = 0

    for run in range(NUM_BENCHMARK_RUNS):
        input_data = UnsafePointer[Scalar[dtype]].alloc(size * size)
        output_data = UnsafePointer[Scalar[dtype]].alloc(size * size)

        # Initialize input data
        for i in range(size * size):
            input_data[i] = Scalar[dtype](i)
            output_data[i] = Scalar[dtype](0.0)

        # Measure execution time
        start_time = now()
        add_10_2d_cpu(output_data, input_data, size)
        end_time = now()

        total_time += end_time - start_time

        # Clean up memory
        input_data.free()
        output_data.free()

    avg_time_ms = (
        Float64(total_time) / Float64(NUM_BENCHMARK_RUNS) / 1_000_000.0
    )
    return avg_time_ms


fn benchmark_gpu_unsafe_for_size(size: Int) raises -> Float64:
    """Benchmark GPU UnsafePointer implementation for given size."""
    total_time: UInt = 0

    # Calculate grid dimensions - simple integer arithmetic
    block_size = 16
    blocks_needed = (size + block_size - 1) // block_size  # Ceiling division

    for run in range(NUM_BENCHMARK_RUNS):
        start_time = now()

        with DeviceContext() as ctx:
            out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)

            # Initialize input data
            with a.map_to_host() as a_host:
                for i in range(size * size):
                    a_host[i] = i

            # Launch GPU kernel
            ctx.enqueue_function[add_10_2d_gpu_unsafe](
                out.unsafe_ptr(),
                a.unsafe_ptr(),
                size,
                grid_dim=blocks_needed,
                block_dim=(block_size, block_size),
            )

            ctx.synchronize()

        end_time = now()
        total_time += end_time - start_time

    avg_time_ms = (
        Float64(total_time) / Float64(NUM_BENCHMARK_RUNS) / 1_000_000.0
    )
    return avg_time_ms


fn print_performance_results(
    size: Int, cpu_time: Float64, gpu_time: Float64, gpu_success: Bool
):
    """Print performance results for a given size."""
    elements = size * size
    print(
        "Matrix size:", size, "x", size, "(" + String(elements) + " elements)"
    )
    print("  CPU Implementation:       ", cpu_time, "ms")

    if gpu_success:
        print("  GPU UnsafePointer:        ", gpu_time, "ms")

        # Calculate speedup
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            if speedup > 1.0:
                print(
                    "  GPU Speedup:              ", speedup, "x (GPU is faster)"
                )
            else:
                print(
                    "  CPU Advantage:            ",
                    1.0 / speedup,
                    "x (CPU is faster)",
                )

        # Calculate throughput
        cpu_throughput = Float64(elements) / cpu_time / 1000.0
        gpu_throughput = Float64(elements) / gpu_time / 1000.0
        print("  CPU Throughput:           ", cpu_throughput, "M elements/ms")
        print("  GPU Throughput:           ", gpu_throughput, "M elements/ms")
    else:
        print("  GPU UnsafePointer:         FAILED")

    print()


def main():
    """Main comprehensive performance analysis function."""
    print(
        "=== Comprehensive Performance Analysis: add_10_2d Implementations ==="
    )
    print("Testing multiple matrix sizes to find GPU vs CPU crossover point")
    print("Number of benchmark runs per size:", NUM_BENCHMARK_RUNS)
    print()

    # Test different matrix sizes - start small and go much larger
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    print("Performance Results:")
    print("===================")

    crossover_found = False
    crossover_size = 0

    for size in sizes:
        print("Testing size", size, "x", size, "...")

        # Benchmark CPU
        cpu_time = benchmark_cpu_for_size(size)

        # Benchmark GPU
        gpu_time = Float64(0.0)
        gpu_success = False

        try:
            gpu_time = benchmark_gpu_unsafe_for_size(size)
            gpu_success = True
        except:
            gpu_success = False
            print("  GPU benchmark failed for size", size)

        print_performance_results(size, cpu_time, gpu_time, gpu_success)

        # Check for crossover point
        if gpu_success and not crossover_found and gpu_time < cpu_time:
            crossover_found = True
            crossover_size = size
            print(
                "🎯 CROSSOVER POINT FOUND! GPU becomes faster at size",
                size,
                "x",
                size,
            )
            print()

    print("=== Analysis Summary ===")
    if crossover_found:
        print(
            "✅ GPU crossover point:",
            crossover_size,
            "x",
            crossover_size,
            "matrix",
        )
        print("📊 Recommendation:")
        print(
            "   - Use CPU implementation for matrices smaller than",
            crossover_size,
            "x",
            crossover_size,
        )
        print(
            "   - Use GPU implementation for matrices",
            crossover_size,
            "x",
            crossover_size,
            "and larger",
        )
    else:
        print("❌ No crossover point found in tested range")
        print(
            "📊 Recommendation: CPU implementation is faster for all tested"
            " sizes"
        )
        print("   - GPU overhead dominates for these matrix sizes")
        print(
            "   - Consider testing larger matrices or more complex operations"
        )

    print()
    print("=== Comprehensive Performance Analysis Complete ===")
