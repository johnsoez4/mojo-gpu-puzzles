"""
Comprehensive benchmark script for add_10_2d implementations.

This script measures and compares the execution time of three add_10_2d() function implementations:
1. CPU-only version from p04_cpu.mojo
2. GPU implementation from p04.mojo  
3. GPU implementation from p04_layout_tensor.mojo

The benchmark uses identical test data across all implementations and includes multiple test runs
to account for performance variance.
"""

from time import perf_counter_ns as now
from testing import assert_equal
from memory import UnsafePointer
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias SIZE = 2  # Back to working configuration
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)
alias NUM_BENCHMARK_RUNS = 10


# CPU implementation (copied from p04_cpu.mojo)
fn add_10_2d_cpu(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """CPU-only implementation of add_10_2d function."""
    for row in range(size):
        for col in range(size):
            var index = row * size + col
            output[index] = a[index] + 10.0


# GPU implementation from p04.mojo
fn add_10_2d_unsafe_ptr(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using UnsafePointer (from p04.mojo)."""
    from gpu import thread_idx

    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


# GPU implementation from p04_layout_tensor.mojo
fn add_10_2d_layout_tensor(
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=True, dtype, layout],
    size: Int,
):
    """GPU kernel using LayoutTensor (from p04_layout_tensor.mojo)."""
    from gpu import thread_idx

    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row, col] = a[row, col] + 10.0


struct BenchmarkResult:
    """Structure to hold benchmark results."""

    var name: String
    var total_time_ns: UInt
    var average_time_ns: Float64
    var num_runs: Int
    var success: Bool

    fn __init__(
        out self,
        name: String,
        total_time_ns: UInt,
        num_runs: Int,
        success: Bool,
    ):
        self.name = name
        self.total_time_ns = total_time_ns
        self.num_runs = num_runs
        self.success = success
        self.average_time_ns = Float64(total_time_ns) / Float64(num_runs)

    fn print_result(self):
        """Print benchmark result in a readable format."""
        if self.success:
            var avg_ms = self.average_time_ns / 1_000_000.0
            var total_ms = Float64(self.total_time_ns) / 1_000_000.0
            print("✓", self.name + ":")
            print("  Average time:", avg_ms, "ms")
            print("  Total time:  ", total_ms, "ms")
            print("  Runs:        ", self.num_runs)
        else:
            print("✗", self.name + ": FAILED")


fn benchmark_cpu_implementation() -> BenchmarkResult:
    """Benchmark the CPU-only implementation."""
    print("\n--- Benchmarking CPU Implementation ---")

    var total_time: UInt = 0
    var success = True

    for run in range(NUM_BENCHMARK_RUNS):
        # Create fresh test data for each run
        var input_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)
        var output_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)

        # Initialize input data
        for i in range(SIZE * SIZE):
            input_data[i] = Scalar[dtype](i)
            output_data[i] = Scalar[dtype](0.0)

        # Measure execution time
        var start_time = now()
        add_10_2d_cpu(output_data, input_data, SIZE)
        var end_time = now()

        total_time += end_time - start_time

        # Verify correctness on first run
        if run == 0:
            for i in range(SIZE * SIZE):
                var expected = Scalar[dtype](i + 10)
                if abs(output_data[i] - expected) > 1e-6:
                    success = False
                    break

        # Clean up memory
        input_data.free()
        output_data.free()

    return BenchmarkResult(
        "CPU Implementation", total_time, NUM_BENCHMARK_RUNS, success
    )


fn benchmark_gpu_unsafe_ptr() raises -> BenchmarkResult:
    """Benchmark the GPU implementation using UnsafePointer."""
    print("\n--- Benchmarking GPU UnsafePointer Implementation ---")

    var total_time: UInt = 0
    var success = True

    for run in range(NUM_BENCHMARK_RUNS):
        var start_time = now()

        with DeviceContext() as ctx:
            # Create GPU buffers
            var out = ctx.enqueue_create_buffer[dtype](
                SIZE * SIZE
            ).enqueue_fill(0)
            var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
                0
            )

            # Initialize input data
            with a.map_to_host() as a_host:
                for y in range(SIZE):
                    for x in range(SIZE):
                        a_host[y * SIZE + x] = y * SIZE + x

            # Launch GPU kernel
            ctx.enqueue_function[add_10_2d_unsafe_ptr](
                out.unsafe_ptr(),
                a.unsafe_ptr(),
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

        var end_time = now()
        total_time += end_time - start_time

        # Verify correctness on first run
        if run == 0:
            with DeviceContext() as ctx:
                var out_verify = ctx.enqueue_create_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)
                var a_verify = ctx.enqueue_create_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)
                var expected = ctx.enqueue_create_host_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)

                with a_verify.map_to_host() as a_host:
                    for i in range(SIZE * SIZE):
                        a_host[i] = i
                        expected[i] = a_host[i] + 10

                ctx.enqueue_function[add_10_2d_unsafe_ptr](
                    out_verify.unsafe_ptr(),
                    a_verify.unsafe_ptr(),
                    SIZE,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )

                ctx.synchronize()

                with out_verify.map_to_host() as out_host:
                    for i in range(SIZE * SIZE):
                        if abs(out_host[i] - expected[i]) > 1e-6:
                            success = False
                            break

    return BenchmarkResult(
        "GPU UnsafePointer Implementation",
        total_time,
        NUM_BENCHMARK_RUNS,
        success,
    )


fn benchmark_gpu_layout_tensor() raises -> BenchmarkResult:
    """Benchmark the GPU implementation using LayoutTensor."""
    print("\n--- Benchmarking GPU LayoutTensor Implementation ---")

    var total_time: UInt = 0
    var success = True

    for run in range(NUM_BENCHMARK_RUNS):
        var start_time = now()

        with DeviceContext() as ctx:
            # Create GPU buffers and tensors
            var out_buf = ctx.enqueue_create_buffer[dtype](
                SIZE * SIZE
            ).enqueue_fill(0)
            var out_tensor = LayoutTensor[mut=True, dtype, layout](
                out_buf.unsafe_ptr()
            ).reshape[layout]()

            var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
                0
            )
            with a.map_to_host() as a_host:
                for i in range(SIZE * SIZE):
                    a_host[i] = i

            var a_tensor = LayoutTensor[mut=True, dtype, layout](
                a.unsafe_ptr()
            ).reshape[layout]()

            # Launch GPU kernel
            ctx.enqueue_function[add_10_2d_layout_tensor](
                out_tensor,
                a_tensor,
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

        var end_time = now()
        total_time += end_time - start_time

        # Verify correctness on first run
        if run == 0:
            with DeviceContext() as ctx:
                var out_buf_verify = ctx.enqueue_create_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)
                var out_tensor_verify = LayoutTensor[mut=True, dtype, layout](
                    out_buf_verify.unsafe_ptr()
                ).reshape[layout]()

                var expected = ctx.enqueue_create_host_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)
                var a_verify = ctx.enqueue_create_buffer[dtype](
                    SIZE * SIZE
                ).enqueue_fill(0)

                with a_verify.map_to_host() as a_host:
                    for i in range(SIZE * SIZE):
                        a_host[i] = i
                        expected[i] = a_host[i] + 10

                var a_tensor_verify = LayoutTensor[mut=True, dtype, layout](
                    a_verify.unsafe_ptr()
                ).reshape[layout]()

                ctx.enqueue_function[add_10_2d_layout_tensor](
                    out_tensor_verify,
                    a_tensor_verify,
                    SIZE,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )

                ctx.synchronize()

                with out_buf_verify.map_to_host() as out_buf_host:
                    for i in range(SIZE * SIZE):
                        if abs(out_buf_host[i] - expected[i]) > 1e-6:
                            success = False
                            break

    return BenchmarkResult(
        "GPU LayoutTensor Implementation",
        total_time,
        NUM_BENCHMARK_RUNS,
        success,
    )


def main():
    """Main benchmark function."""
    print("=== Comprehensive add_10_2d Benchmark ===")
    print("Matrix size:", SIZE, "x", SIZE)
    print("Number of benchmark runs:", NUM_BENCHMARK_RUNS)
    print("GPU blocks per grid:", BLOCKS_PER_GRID)
    print(
        "GPU threads per block:",
        THREADS_PER_BLOCK[0],
        "x",
        THREADS_PER_BLOCK[1],
    )

    # Run benchmarks
    var cpu_result = benchmark_cpu_implementation()
    var gpu_unsafe_result = benchmark_gpu_unsafe_ptr()
    var gpu_layout_result = benchmark_gpu_layout_tensor()

    # Print results
    print("\n=== Benchmark Results ===")
    cpu_result.print_result()
    gpu_unsafe_result.print_result()
    gpu_layout_result.print_result()

    # Calculate and display performance comparison
    if (
        cpu_result.success
        and gpu_unsafe_result.success
        and gpu_layout_result.success
    ):
        print("\n=== Performance Comparison ===")
        var cpu_time = cpu_result.average_time_ns
        var gpu_unsafe_time = gpu_unsafe_result.average_time_ns
        var gpu_layout_time = gpu_layout_result.average_time_ns

        print("Average execution times:")
        print("  CPU:              ", cpu_time / 1_000_000.0, "ms")
        print("  GPU UnsafePointer:", gpu_unsafe_time / 1_000_000.0, "ms")
        print("  GPU LayoutTensor: ", gpu_layout_time / 1_000_000.0, "ms")

        # Calculate speedups relative to CPU
        if cpu_time > 0:
            if gpu_unsafe_time > 0:
                var unsafe_speedup = cpu_time / gpu_unsafe_time
                print("GPU UnsafePointer vs CPU speedup:", unsafe_speedup, "x")

            if gpu_layout_time > 0:
                var layout_speedup = cpu_time / gpu_layout_time
                print("GPU LayoutTensor vs CPU speedup:", layout_speedup, "x")

        # Compare GPU implementations
        if gpu_unsafe_time > 0 and gpu_layout_time > 0:
            if gpu_unsafe_time < gpu_layout_time:
                var unsafe_advantage = gpu_layout_time / gpu_unsafe_time
                print(
                    "GPU UnsafePointer is",
                    unsafe_advantage,
                    "x faster than GPU LayoutTensor",
                )
            else:
                var layout_advantage = gpu_unsafe_time / gpu_layout_time
                print(
                    "GPU LayoutTensor is",
                    layout_advantage,
                    "x faster than GPU UnsafePointer",
                )

        # Identify fastest implementation
        var fastest_time = cpu_time
        var fastest_name = String("CPU")

        if gpu_unsafe_time < fastest_time:
            fastest_time = gpu_unsafe_time
            fastest_name = String("GPU UnsafePointer")

        if gpu_layout_time < fastest_time:
            fastest_time = gpu_layout_time
            fastest_name = String("GPU LayoutTensor")

        print(
            "Fastest implementation:",
            fastest_name,
            "with",
            fastest_time / 1_000_000.0,
            "ms average",
        )

    else:
        print(
            "\n=== Some benchmarks failed - skipping performance comparison ==="
        )
        if not cpu_result.success:
            print("CPU benchmark failed")
        if not gpu_unsafe_result.success:
            print("GPU UnsafePointer benchmark failed")
        if not gpu_layout_result.success:
            print("GPU LayoutTensor benchmark failed")

    print("\n=== Benchmark Completed ===")
