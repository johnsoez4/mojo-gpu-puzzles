"""
Comprehensive performance analysis script for add_10_2d implementations.

This script tests multiple matrix sizes to analyze performance scaling
and identify the crossover point where GPU implementations become advantageous.
"""

from time import perf_counter_ns as now
from memory import UnsafePointer
from gpu.host import DeviceContext
from gpu import thread_idx
from layout import Layout, LayoutTensor

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
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


fn add_10_2d_layout_tensor(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using LayoutTensor approach but with UnsafePointer for flexibility.
    """
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row * size + col] = a[row * size + col] + 10.0


fn benchmark_cpu_for_size(size: Int) -> Float64:
    """Benchmark CPU implementation for given size."""
    total_time: UInt = 0

    for run in range(NUM_BENCHMARK_RUNS):
        # Start timing before memory allocation (same as GPU benchmark)
        start_time = now()

        # Memory allocation (equivalent to GPU buffer creation)
        input_data = UnsafePointer[Scalar[dtype]].alloc(size * size)
        output_data = UnsafePointer[Scalar[dtype]].alloc(size * size)

        # Initialize input data (equivalent to GPU data initialization)
        for i in range(size * size):
            input_data[i] = Scalar[dtype](i)
            output_data[i] = Scalar[dtype](0.0)

        # CPU execution (equivalent to GPU kernel execution)
        add_10_2d_cpu(output_data, input_data, size)

        # End timing after computation (before cleanup, same as GPU benchmark)
        end_time = now()

        total_time += end_time - start_time

        # Clean up memory (not included in timing, same as GPU benchmark)
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


fn benchmark_gpu_layout_tensor_for_size(size: Int) raises -> Float64:
    """Benchmark GPU LayoutTensor implementation for given size."""
    total_time: UInt = 0

    # Calculate grid dimensions - simple integer arithmetic
    block_size = 16
    blocks_needed = (size + block_size - 1) // block_size  # Ceiling division

    for run in range(NUM_BENCHMARK_RUNS):
        start_time = now()

        with DeviceContext() as ctx:
            # Create buffers (same as UnsafePointer approach)
            out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)

            # Initialize input data
            with a.map_to_host() as a_host:
                for i in range(size * size):
                    a_host[i] = i

            # Launch GPU kernel (using LayoutTensor-style kernel)
            ctx.enqueue_function[add_10_2d_layout_tensor](
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
    size: Int,
    cpu_time: Float64,
    gpu_unsafe_time: Float64,
    gpu_unsafe_success: Bool,
    gpu_layout_time: Float64,
    gpu_layout_success: Bool,
):
    """Print performance results for a given size."""
    elements = size * size
    print(
        "Matrix size:", size, "x", size, "(" + String(elements) + " elements)"
    )
    print("  CPU Implementation:        ", cpu_time, "ms")

    if gpu_unsafe_success:
        print("  GPU UnsafePointer:         ", gpu_unsafe_time, "ms")
    else:
        print("  GPU UnsafePointer:         FAILED")

    if gpu_layout_success:
        print("  GPU LayoutTensor:          ", gpu_layout_time, "ms")
    else:
        print("  GPU LayoutTensor:          FAILED")

    # Calculate speedups and advantages only if both implementations succeeded
    if gpu_unsafe_success and cpu_time > 0 and gpu_unsafe_time > 0:
        speedup = cpu_time / gpu_unsafe_time
        if speedup > 1.0:
            print("  GPU UnsafePointer Speedup: ", speedup, "x (GPU is faster)")
        else:
            print(
                "  CPU vs GPU UnsafePointer:  ",
                1.0 / speedup,
                "x (CPU is faster)",
            )

    if gpu_layout_success and cpu_time > 0 and gpu_layout_time > 0:
        speedup = cpu_time / gpu_layout_time
        if speedup > 1.0:
            print("  GPU LayoutTensor Speedup:  ", speedup, "x (GPU is faster)")
        else:
            print(
                "  CPU vs GPU LayoutTensor:   ",
                1.0 / speedup,
                "x (CPU is faster)",
            )

    # Compare GPU implementations
    if gpu_unsafe_success and gpu_layout_success:
        if gpu_layout_time < gpu_unsafe_time:
            speedup = gpu_unsafe_time / gpu_layout_time
            print(
                "  GPU LayoutTensor is ",
                speedup,
                "x faster than GPU UnsafePointer",
            )
        else:
            speedup = gpu_layout_time / gpu_unsafe_time
            print(
                "  GPU UnsafePointer is ",
                speedup,
                "x faster than GPU LayoutTensor",
            )

    # Calculate throughput
    cpu_throughput = Float64(elements) / cpu_time / 1000.0
    print("  CPU Throughput:            ", cpu_throughput, "M elements/ms")

    if gpu_unsafe_success:
        gpu_unsafe_throughput = Float64(elements) / gpu_unsafe_time / 1000.0
        print(
            "  GPU UnsafePointer Throughput: ",
            gpu_unsafe_throughput,
            "M elements/ms",
        )

    if gpu_layout_success:
        gpu_layout_throughput = Float64(elements) / gpu_layout_time / 1000.0
        print(
            "  GPU LayoutTensor Throughput:  ",
            gpu_layout_throughput,
            "M elements/ms",
        )

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

        # Benchmark GPU UnsafePointer
        gpu_unsafe_time = Float64(0.0)
        gpu_unsafe_success = False

        try:
            gpu_unsafe_time = benchmark_gpu_unsafe_for_size(size)
            gpu_unsafe_success = True
        except:
            gpu_unsafe_success = False
            print("  GPU UnsafePointer benchmark failed for size", size)

        # Benchmark GPU LayoutTensor
        gpu_layout_time = Float64(0.0)
        gpu_layout_success = False

        try:
            gpu_layout_time = benchmark_gpu_layout_tensor_for_size(size)
            gpu_layout_success = True
        except:
            gpu_layout_success = False
            print("  GPU LayoutTensor benchmark failed for size", size)

        print_performance_results(
            size,
            cpu_time,
            gpu_unsafe_time,
            gpu_unsafe_success,
            gpu_layout_time,
            gpu_layout_success,
        )

        # Check for crossover point (either GPU implementation faster than CPU)
        if not crossover_found:
            if (gpu_unsafe_success and gpu_unsafe_time < cpu_time) or (
                gpu_layout_success and gpu_layout_time < cpu_time
            ):
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
