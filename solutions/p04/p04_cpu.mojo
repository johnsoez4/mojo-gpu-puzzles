from memory import UnsafePointer
from testing import assert_equal

alias SIZE = 2
alias dtype = DType.float32


fn add_10_2d_cpu(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """
    CPU-only implementation of add_10_2d function.

    This function replicates the exact calculation logic from the GPU implementations
    but runs entirely on CPU without any GPU operations.
    """
    # Replicate the GPU kernel logic on CPU
    # GPU version: output[row * size + col] = a[row * size + col] + 10.0
    # CPU version: iterate through all elements and add 10.0

    for row in range(size):
        for col in range(size):
            var index = row * size + col
            output[index] = a[index] + 10.0


def main():
    """Main function to test the CPU implementation."""
    print("=== CPU-only add_10_2d Implementation Test ===")

    # Create test data arrays
    var input_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)
    var output_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)
    var expected_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)

    # Initialize input data with the same pattern as GPU implementations
    # From p04.mojo: a_host[y * SIZE + x] = y * SIZE + x
    for y in range(SIZE):
        for x in range(SIZE):
            var index = y * SIZE + x
            input_data[index] = Scalar[dtype](index)
            expected_data[index] = Scalar[dtype](index + 10)
            output_data[index] = Scalar[dtype](0.0)

    print("Input data:")
    for i in range(SIZE * SIZE):
        print("  [", i, "]:", input_data[i])

    # Run the CPU implementation
    add_10_2d_cpu(output_data, input_data, SIZE)

    print("Output data:")
    for i in range(SIZE * SIZE):
        print("  [", i, "]:", output_data[i])

    print("Expected data:")
    for i in range(SIZE * SIZE):
        print("  [", i, "]:", expected_data[i])

    # Verify results
    var success = True
    for i in range(SIZE * SIZE):
        if abs(output_data[i] - expected_data[i]) > 1e-6:
            print(
                "✗ Value mismatch at index",
                i,
                ": output=",
                output_data[i],
                "expected=",
                expected_data[i],
            )
            success = False

    if success:
        print("✓ CPU implementation test PASSED")

        # Additional verification using assert_equal for each element
        try:
            for i in range(SIZE * SIZE):
                assert_equal(output_data[i], expected_data[i])
            print("✓ All assertions passed")
        except e:
            print("✗ Assertion failed:", e)
            success = False
    else:
        print("✗ CPU implementation test FAILED")

    # Clean up memory
    input_data.free()
    output_data.free()
    expected_data.free()

    if success:
        print("=== CPU Implementation Test Completed Successfully ===")
    else:
        print("=== CPU Implementation Test FAILED ===")
