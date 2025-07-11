from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

alias SEQ_LEN = 16
alias D = 16
alias TPB = SEQ_LEN


# Tiled matrix multiplication from p14 - adapted for attention
fn matmul_idiomatic_tiled[
    layout: Layout,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
):
    """Idiomatic tiled matrix multiplication from p14."""
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)

    var acc: output.element_type = 0

    alias load_a_layout = Layout.row_major(1, TPB)
    alias load_b_layout = Layout.row_major(TPB, 1)

    for idx in range((inner + TPB - 1) // TPB):
        # Get tiles from A and B matrices
        a_tile = a.tile[TPB, TPB](block_idx.y, idx)
        b_tile = b.tile[TPB, TPB](idx, block_idx.x)

        # Asynchronously copy tiles to shared memory
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result with bounds checking (needed for attention's variable sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc


# ANCHOR: transpose_kernel
fn transpose_kernel[
    layout_in: Layout,  # Layout for input matrix (seq_len, d)
    layout_out: Layout,  # Layout for output matrix (d, seq_len)
    rows: Int,
    cols: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout_out, MutableAnyOrigin],
    inp: LayoutTensor[mut=False, dtype, layout_in, MutableAnyOrigin],
):
    # FILL ME IN (roughly 18 lines)
    ...


# ANCHOR_END: transpose_kernel


# Apply softmax to attention scores taken from p16
fn softmax_kernel[
    layout: Layout,
    seq_len: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    scores: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
):
    """Apply softmax to attention scores - exact p16 pattern."""
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_i < seq_len:
        thread_max = rebind[Scalar[dtype]](scores[global_i])

    shared_max[local_i] = thread_max
    barrier()

    # Parallel reduction to find max
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_max[local_i] = max(
                shared_max[local_i], shared_max[local_i + stride]
            )
        barrier()
        stride = stride // 2

    block_max = shared_max[0]

    var exp_val: Scalar[dtype] = 0.0
    if global_i < seq_len:
        exp_val = rebind[Scalar[dtype]](exp(scores[global_i] - block_max))
        output[global_i] = exp_val

    shared_sum[local_i] = exp_val
    barrier()

    # Parallel reduction for sum
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_sum[local_i] = (
                shared_sum[local_i] + shared_sum[local_i + stride]
            )
        barrier()
        stride = stride // 2

    block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < seq_len:
        output[global_i] = output[global_i] / block_sum


# CPU implementation for vector attention
fn attention_cpu_kernel[
    layout_q: Layout,
    layout_k: Layout,
    layout_v: Layout,
    layout_out: Layout,
    seq_len: Int,
    d: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutableAnyOrigin],
    q: LayoutTensor[dtype, layout_q, MutableAnyOrigin],
    k: LayoutTensor[dtype, layout_k, MutableAnyOrigin],
    v: LayoutTensor[dtype, layout_v, MutableAnyOrigin],
):
    """CPU implementation of vector attention."""
    var scores = List[Float32]()
    var weights = List[Float32]()
    for _ in range(seq_len):
        scores.append(0.0)
        weights.append(0.0)

    # CPU: Compute attention scores K[i] · Q directly for each row i of K
    for i in range(seq_len):
        var score: Float32 = 0.0
        for dim in range(d):
            score = score + rebind[Float32](q[dim]) * rebind[Float32](k[i, dim])
        scores[i] = score

    var max_score: Float32 = scores[0]
    for i in range(1, seq_len):
        if scores[i] > max_score:
            max_score = scores[i]

    var sum_exp: Float32 = 0.0
    for i in range(seq_len):
        weights[i] = exp(scores[i] - max_score)
        sum_exp = sum_exp + weights[i]

    for i in range(seq_len):
        weights[i] = weights[i] / sum_exp

    for dim in range(d):
        var weighted_sum: Float32 = 0.0
        for i in range(seq_len):
            weighted_sum = weighted_sum + weights[i] * rebind[Float32](
                v[i, dim]
            )
        output[dim] = rebind[Scalar[dtype]](weighted_sum)


@compiler.register("attention")
struct AttentionCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        seq_len: Int,
        d: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],  # Output vector (d,)
        q: InputTensor[dtype=dtype, rank=1],  # Query vector (d,)
        k: InputTensor[dtype=dtype, rank=2],  # Key matrix (seq_len, d)
        v: InputTensor[dtype=dtype, rank=2],  # Value matrix (seq_len, d)
        ctx: DeviceContextPtr,
    ) raises:
        # Define layouts
        alias layout_q = Layout.row_major(d)
        alias layout_k = Layout.row_major(seq_len, d)
        alias layout_v = Layout.row_major(seq_len, d)
        alias layout_out = Layout.row_major(d)
        alias layout_scores = Layout.row_major(seq_len)

        # Convert to layout tensors
        var output_tensor = rebind[
            LayoutTensor[dtype, layout_out, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var q_tensor = rebind[LayoutTensor[dtype, layout_q, MutableAnyOrigin]](
            q.to_layout_tensor()
        )
        var k_tensor = rebind[LayoutTensor[dtype, layout_k, MutableAnyOrigin]](
            k.to_layout_tensor()
        )
        var v_tensor = rebind[LayoutTensor[dtype, layout_v, MutableAnyOrigin]](
            v.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            # ANCHOR: attention_orchestration
            var gpu_ctx = rebind[DeviceContext](ctx[])

            # Define layouts for matrix multiplication
            # Q reshaped to (1, d)
            alias layout_q_2d = Layout.row_major(1, d)
            # K^T is (d, seq_len)
            alias layout_k_t = Layout.row_major(d, seq_len)
            # Scores as (1, seq_len)
            alias layout_scores_2d = Layout.row_major(1, seq_len)
            # Weights as (1, seq_len)
            alias layout_weights_2d = Layout.row_major(1, seq_len)
            # Result as (1, d)
            alias layout_result_2d = Layout.row_major(1, d)

            alias scores_blocks_per_grid = (
                (seq_len + TPB - 1) // TPB,
                (1 + TPB - 1) // TPB,
            )
            alias result_blocks_per_grid = (
                (d + TPB - 1) // TPB,
                (1 + TPB - 1) // TPB,
            )
            alias matmul_threads_per_block = (TPB, TPB)
            alias transpose_blocks_per_grid = (
                (seq_len + TPB - 1) // TPB,
                (d + TPB - 1) // TPB,
            )

            # Allocate minimal temporary buffers - reuse same buffer for different shapes
            k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len * d
            )  # K^T as (d, seq_len)
            scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len
            )  # Reused for scores and weights

            k_t = LayoutTensor[mut=True, dtype, layout_k_t, MutableAnyOrigin](
                k_t_buf.unsafe_ptr()
            )

            # Step 1: Reshape Q from (d,) to (1, d) - no buffer needed
            # FILL ME IN 1 line

            # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)
            # FILL ME IN 1 function call

            # Step 3: Compute attention scores using matmul: Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len)
            # GPU: Uses matrix multiplication to compute all Q · K[i] scores in parallel
            # Reuse scores_weights_buf as (1, seq_len) for scores
            # FILL ME IN 2 lines

            # Step 4: Reshape scores from (1, seq_len) to (seq_len,) for softmax
            # FILL ME IN 1 line

            # Step 5: Apply softmax to get attention weights
            # FILL ME IN 1 function call

            # Step 6: Reshape weights from (seq_len,) to (1, seq_len) for final matmul
            # FILL ME IN 1 line

            # Step 7: Compute final result using matmul: weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d)
            # Reuse out_tensor reshaped as (1, d) for result
            # FILL ME IN 2 lines

            # ANCHOR_END: attention_orchestration

        elif target == "cpu":
            attention_cpu_kernel[
                layout_q, layout_k, layout_v, layout_out, seq_len, d, dtype
            ](output_tensor, q_tensor, k_tensor, v_tensor)

        else:
            raise Error("Unsupported target: " + target)
