"""
Simple performance analysis script for add_10_2d implementations.

This script tests multiple matrix sizes to analyze CPU performance scaling.
"""

from time import perf_counter_ns as now
from memory import UnsafePointer

alias dtype = DType.float32
alias NUM_BENCHMARK_RUNS = 3

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
        
        # Verify correctness on first run
        if run == 0:
            for i in range(size * size):
                expected = Scalar[dtype](i + 10)
                if abs(output_data[i] - expected) > 1e-6:
                    print("ERROR: Incorrect result at index", i)
                    break
        
        # Clean up memory
        input_data.free()
        output_data.free()
    
    avg_time_ms = Float64(total_time) / Float64(NUM_BENCHMARK_RUNS) / 1_000_000.0
    return avg_time_ms

fn print_performance_comparison(size: Int, cpu_time: Float64):
    """Print performance results for a given size."""
    elements = size * size
    print("Matrix size:", size, "x", size, "(" + String(elements) + " elements)")
    print("  CPU Implementation:", cpu_time, "ms")
    
    # Calculate throughput
    throughput = Float64(elements) / cpu_time / 1000.0  # Million elements per second
    print("  Throughput:", throughput, "M elements/ms")
    print()

def main():
    """Main performance analysis function."""
    print("=== Simple Performance Analysis: CPU add_10_2d Implementation ===")
    print("Testing multiple matrix sizes to analyze CPU performance scaling")
    print("Number of benchmark runs per size:", NUM_BENCHMARK_RUNS)
    print()
    
    # Test different matrix sizes
    sizes = [2, 4, 8, 16, 32, 64]
    
    print("Performance Results:")
    print("===================")
    
    for size in sizes:
        print("Testing size", size, "x", size, "...")
        cpu_time = benchmark_cpu_for_size(size)
        print_performance_comparison(size, cpu_time)
    
    print("=== Performance Analysis Complete ===")
