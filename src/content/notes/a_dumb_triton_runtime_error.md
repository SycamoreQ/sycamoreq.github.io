---
title: "A Dumb Triton Runtime Error"
date: 2026-11-05
description: "A (somewhat) deep dive into Triton's IR verifier, scalar vs. vector pointer types in atomic operations, and kernel reductions."
tags: [Triton, Python, GPU]
draft: false
---

I have been writing some Triton code recently, and it has been very interesting. I have been writing CUDA C++ for some time, and Triton feels like it makes a lot of things much simpler than CUDA C++ does. Maybe I have not reached the harder parts of Triton yet, but at this point in my journey things have been easier.

Triton, for the uninitiated, is an open-source, Python-based domain-specific language (DSL) and MLIR-based compiler designed for writing high-performance deep learning compute kernels. Coincidentally, I have been working on Tesserae, a DSL with the exact same goal, except that it does not use MLIR—it uses a custom IR called Tirix. This is another reason I am practicing Triton: to get a feel for similar architectures and their performance. It is also interesting to see so many parallels in ideas.

In this post, I will walk you through one Triton kernel I wrote last week, and how a discussion with a fellow triton programmer led me to replace two lines of code with one. That change produced an error that made me curious about how Triton works under the hood. The short version: a scalar pointer and a vector value ended up in the same `tt.atomic_rmw` operation, and Triton's IR verifier rejected it before the kernel ever reached a GPU. (The error surfaces as a Python `RuntimeError`, but it is raised at compile time, hence the title.) Along the way I also made a second, smaller mistake with the `other=` padding value, which I cover in Step 2. I also tested two ways to make the atomic version valid. Both work, and I explain why I would still prefer the original kernel.

## The kernel itself

The kernel we will be looking at is `Tiled Triton Row Sum`, problem number `344` on [TensorTonic](https://www.tensortonic.com/problems/tiled-row-sum), an online LeetCode-like platform for practicing GPU programming. The problem heading already tells us most of what we need: a row sum, performed as a sum reduction across each row in a tiled fashion in Triton.

Conceptually, the answer is easy, and it is a familiar idea from SIMD programming, where a vector register is folded down to a single scalar (a "horizontal" reduction). The interesting part is how Triton expresses it.

Here is the code on my GitHub. TensorTonic automatically creates a repository and adds completed solutions inside it: [GitHub](https://github.com/SycamoreQ/TensorTonic-Solutions/blob/main/study-plans/language-modeling-from-scratch/cs336-l06-triton-tiled-row-sum/cs336-l06-triton-tiled-row-sum.py)
(Please dont judge my coding stats I know I have only solved 6 questions up until now :())

Here is the working code (not present in the GitHub)

```python
@triton.jit
def tiled_row_sum_kernel(x_ptr, out_ptr, n_cols, row_stride, col_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride

    # Running float32 accumulator vector for the tile
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Step across the row in tiles of width BLOCK_SIZE
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols

        ptr = row_start_ptr + col_indices * col_stride
        chunk = tl.load(ptr, mask=mask, other=0.0)

        # Convert to float32 before summation
        acc += chunk.to(tl.float32)

    # Reduce the 1D tile accumulator to a single float32 scalar and write to out[row_idx]
    row_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr + row_idx, row_sum)
```

For anybody who knows Triton, this is very straightforward to read—in fact, it was marked Medium on TensorTonic, which is pretty ironic. Either way, here is a diagram of the execution to help with both the code and what it does under the hood:

![Step 1: Initialize](/diagrams/triton/step_1.png)
*Figure 1: Step 1 — Initializing the 1D accumulator vector in registers (`acc = tl.zeros`)*

![Step 2: For loop](/diagrams/triton/step_2.png)
*Figure 2: Step 2 — Tile window boundary masking and lane-local vector accumulation*

![Step 3: Accumulate](/diagrams/triton/step_3.png)
*Figure 3: Step 3 — Final cross-lane tree reduction (`tl.sum`) and scalar store*

We can divide the execution into three steps.

## Step 1: Program Grid, Pointers, and Register Initialization

Let us say we have a single row with number of columns or `n_cols=3`:

```text
x = [1.0, 1.0, 1.0]
```

In Triton—and in any CUDA programming paradigm, this tensor resides on the **device side** in **Global Memory (HBM/VRAM)** after being copied over from the host. To process it in parallel, we index the columns of a tile with `col_indices`. While "lane" in hardware strictly refers to a thread inside a 32-thread warp, thinking of `tl.arange(0, BLOCK_SIZE)` as virtual SIMD lanes is the cleanest mental model for Triton. If our tile width is `BLOCK_SIZE = 4`, our tile has 4 lanes indexed 0 through 3. The first 3 lanes map to valid elements (`[0, 1, 2]`) and Lane 3 falls out-of-bounds (`OOB`).

Let's look at what the kernel takes in as parameters:

* `x_ptr`: A device pointer to the start of the input matrix.
* `out_ptr`: A device pointer to the output vector.
* `n_cols`: The logical number of columns ($C$) in each row.
* `row_stride` and `col_stride`: The row and column strides, in elements. These determine how far the pointer must move in physical memory to step to the next row or the next column. Passing explicit strides lets the kernel work on non-contiguous tensor views or slices without copying memory.
* `BLOCK_SIZE: tl.constexpr`: The number of input elements a single program operates on at a time. Along with the strides, this defines our "tile". Because it is marked `tl.constexpr`, the Triton compiler knows the tile width at compile time and can specialize the generated code for it. It must also be a power of two (such as $4$, $32$, or $1024$), because `tl.arange` requires power-of-two ranges.

Unlike CUDA C++, where we write code from the perspective of a single thread (`threadIdx.x`), Triton uses a **block-level (CTA-level)** programming model. When the launcher dispatches this kernel with a 1D grid of $R$ programs (`grid = (R,)`), each program instance handles one full row. Calling `row_idx = tl.program_id(axis=0)` gives the current CTA its row index, and we use it to compute `row_start_ptr`:

```python
row_idx = tl.program_id(axis=0)
row_start_ptr = x_ptr + row_idx * row_stride
```

Along with setting up our row pointer, we also initialize a 1D accumulator vector `acc` before entering the loop. Why do we do this?

* **Mixed-Precision Promotion:** Our input matrix $x$ will not necessarily be in `float32` (it could be `float16` or `bfloat16`), which can easily overflow or lose precision during summation. As the problem specification requires, we convert each tile chunk to `float32` *before* summing and accumulate into a higher-precision register buffer.
* **Register Persistence:** Calling `tl.zeros` does not touch Global Memory—it allocates `BLOCK_SIZE` registers directly inside the SM's Register File, initialized to $0.0$:

```python
acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
```

## Step 2: Sliding Tile Window, Boundary Masking, and Vector Accumulation

Step 2 is the actual logic of how tiling works: the kernel iterates across the row using a fixed-width tile window.

Why do we need a `for` loop with a sliding window instead of loading the entire row at once? For a tiny row of 3 elements, a single tile is enough. But if `n_cols` is huge (say, $131{,}072$ columns), a single thread block does not have enough physical registers on the SM to hold the entire row at once without **register spilling** to local memory. By stepping across the row in chunks of `BLOCK_SIZE`, our register footprint stays fixed at `BLOCK_SIZE` elements regardless of how wide the matrix is:

```python
for col_offset in range(0, n_cols, BLOCK_SIZE):
    col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_indices < n_cols

    ptr = row_start_ptr + col_indices * col_stride
    chunk = tl.load(ptr, mask=mask, other=0.0)

    acc += chunk.to(tl.float32)
```

Let's trace what happens inside each lane during **Tile 0 (`col_offset = 0`)** for our $n_{\text{cols}} = 3$, $\text{BLOCK\_SIZE} = 4$ example:

* **Generating Column Indices (`col_indices`):** `tl.arange(0, 4)` produces the vector `[0, 1, 2, 3]`. Adding `col_offset` (0) gives the active column indices `[0, 1, 2, 3]` for the current window.

* **Boundary Masking (`mask`):** Because `BLOCK_SIZE` ($4$) must be a power of two, our tile window overhangs the end of the 3-element row. Evaluating `col_indices < n_cols` (`[0, 1, 2, 3] < 3`) produces the boolean predicate vector `[True, True, True, False]`. Lane $3$ is flagged as `False` (out-of-bounds).

* **Predicated Load and `other=0.0` Padding (`tl.load`):** Without a mask, Lane 3 would read past the end of Row 0, either corrupting our sum with the first element of Row 1 or triggering an illegal memory access. With `mask=mask, other=0.0`, Triton emits predicated PTX global loads (`@p ld.global.b32`). Lanes 0, 1, and 2 fetch `[1.0, 1.0, 1.0]` from Global Memory, while Lane 3's memory read is suppressed in hardware and replaced with the fallback value `other=0.0`, yielding `chunk = [1.0, 1.0, 1.0, 0.0]`.

* **The `-inf` Padding Trap:** A subtle bug I ran into earlier was setting `other=-float('inf')` out of habit from writing `tl.max` and `softmax` kernels. For `tl.max`, $-\infty$ is the identity element because $\max(x, -\infty) = x$. For a sum reduction, however, padding out-of-bounds lanes with $-\infty$ means computing $1.0 + 1.0 + 1.0 + (-\infty) = -\infty$, which poisons the entire row sum. For summation, `other` must be the additive identity, $0.0$. It was a really silly error, but a quick fix once I visualized the lanes.

* **In-Register Vector Accumulation (`acc += ...`):** Finally, we cast `chunk` to `tl.float32` and add it lane-by-lane into `acc`:

```python
acc = [0.0, 0.0, 0.0, 0.0] + [1.0, 1.0, 1.0, 0.0] = [1.0, 1.0, 1.0, 0.0]
```

If `n_cols` were larger than `BLOCK_SIZE` (for instance, `n_cols = 6` with `BLOCK_SIZE = 4`), the loop would advance to `col_offset = 4`, generate `col_indices = [4, 5, 6, 7]` with `mask = [True, True, False, False]`, and accumulate the second tile on top of `acc`'s existing lanes.

## Step 3: Cross-Lane Tree Reduction (`tl.sum`) and Scalar Store

Once the `for` loop has stepped across all tiles of the row, `acc` holds `BLOCK_SIZE` partial sums distributed across the lanes of our register vector (`[1.0, 1.0, 1.0, 0.0]`). In Step 3, we collapse that vector down to a single scalar and store it to the output vector:

```python
row_sum = tl.sum(acc, axis=0)
tl.store(out_ptr + row_idx, row_sum)
```

Calling `tl.sum(acc, axis=0)` performs a **parallel tree reduction** across the lanes of the tile:

* **Intra-Warp Reduction:** Inside a 32-thread warp, threads exchange register values directly using PTX butterfly shuffle instructions (`shfl.sync.bfly.b32`), folding the vector at each of $\log_2(\text{warp\_size})$ steps without leaving the Register File.
* **Inter-Warp Reduction:** When `BLOCK_SIZE` spans multiple warps, each warp writes its partial scalar sum to Shared Memory (SRAM), the warps synchronize, and the first warp reduces those values into a single scalar: $\text{row\_sum} = 3.0$.

Finally, `tl.store(out_ptr + row_idx, row_sum)` writes that single `float32` scalar to `out[row_idx]`. Because we store one 0D scalar per row (not a vector of `BLOCK_SIZE` lanes), we don't need a mask on `tl.store`.

## Optimization Patterns in This Kernel

Triton, as a GPU DSL, has its own compiler optimizations. Many are well-known hardware optimizations that Triton hides behind its syntax, and some are pretty well hidden. This kernel is under 20 lines of Python, yet it leans on three important patterns. The first and third are choices made in how the kernel is written, while the second is something the compiler does for us.

1. **Hoisting `tl.sum` out of the loop (lane-local accumulation).** Notice that we did *not* write `row_sum += tl.sum(chunk, axis=0)` inside the `for` loop. The reason is that `acc += chunk.to(tl.float32)` is a pure lane-local vector addition (`fadd.rn.f32`): each thread adds its newly loaded register to its own running accumulator register, with no communication between threads. Calling `tl.sum(acc, axis=0)` requires cross-thread warp shuffles and shared memory synchronization. By accumulating into the register vector `acc` inside the loop and calling `tl.sum` **only once** after the loop, we pay the cross-lane reduction cost once per row instead of once per tile.

2. **Coalesced and vectorized global memory loads (`tt.divisibility`).** When `col_stride == 1`, adjacent lanes in `col_indices` map to contiguous addresses in Global Memory. Triton tracks the alignment of `x_ptr` (annotating it with `{tt.divisibility = 16 : i32}` in the generated MLIR) and can coalesce adjacent lane loads into wide vectorized PTX memory transactions (such as `ld.global.v4.b32`), which helps maximize HBM bandwidth utilization.

3. **Branchless predication.** Instead of branching with `if/else` at the tail end of a row—which can cause warp divergence in naive CUDA kernels—`mask = col_indices < n_cols` lowers to hardware predicate registers. All threads in the warp execute in lockstep, and out-of-bounds lanes simply have their global load predicated off while receiving the constant $0.0$.

## The "Dumb" Change: Replacing `tl.sum` with `tl.atomic_add`

Now we get to the core of this post. After going through Step 3, a fellow triton programmer and I got into a discussion about alternative ways to combine the accumulator.

The question came up: *"`acc` already holds our partial sums in registers. Why do we need an explicit tree reduction with `tl.sum(acc, axis=0)` followed by `tl.store`? Why can't we just atomically dump the entire `acc` vector directly into `out_ptr`?"* It was a very valid question, and it made me think.

In CUDA, we frequently use atomic operations for reductions into output buffers. So, thinking it might be a neat shortcut, I replaced the last two lines of the kernel (lines 21–22 of the listing above):

```python
# The original working code:
row_sum = tl.sum(acc, axis=0)
tl.store(out_ptr + row_idx, row_sum)
```

with a single atomic addition:

```python
# The experiment:
tl.atomic_add(out_ptr, acc, mask=None)
```

(Note that this experiment also forgot the `+ row_idx` offset, so even if it had compiled, every row would have accumulated into `out[0]`. That is a second bug hiding behind the first.)

I changed the code live in front of my peer and clicked **Run** on TensorTonic against the test suite. The failing test case that triggered the crash was:

```python
# Failing Test Case Details:
x = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda")
# Matrix shape: (2, 3) -> R = 2, C = 3
# Target output shape: (2,)
# Stride: row_stride = 3, col_stride = 1
# Tile configuration: BLOCK_SIZE = 32
```

The test ran on an **NVIDIA H200 SXM**. Instead of a failed numerical assertion or a wrong sum in the output tensor, the kernel failed to compile and run at all.

The console printed this monstrous error:

```text
exception: RuntimeError: error encountered during parsing

#loc = loc("/tmp/tensortonic-kernel/solution.py":6:0)
#loc4 = loc("/tmp/tensortonic-kernel/solution.py":10:19)
#loc5 = loc("/tmp/tensortonic-kernel/solution.py":12:39)
#loc20 = loc("x_ptr"(#loc))
#loc21 = loc("out_ptr"(#loc))
#loc22 = loc("n_cols"(#loc))
#loc23 = loc("row_stride"(#loc))
#loc27 = loc("acc"(#loc4))
#loc29 = loc("col_offset"(#loc5))
#loc37 = loc("acc"(#loc27))
"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}], function_type = (!tt.ptr<f32>, !tt.ptr<f32>, i32, i32) -> (), sym_name = "tiled_row_sum_kernel", sym_visibility = "public"}> ({
  ^bb0(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_cols: i32 loc("n_cols"(#loc)), %row_stride: i32 loc("row_stride"(#loc))):
    %i = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32 loc(#loc24)
    ...
    %acc_9 = "scf.for"(%5, %6, %7, %acc) ({
    ^bb0(%col_offset: i32 loc("col_offset"(#loc5)), %acc_10: tensor<32xf64> loc("acc"(#loc27))):
      ...
      %chunk_40 = "tt.load"(%ptr_38, %mask_23, %chunk_39) ... : (tensor<32x!tt.ptr<f32>>, tensor<32xi1>, tensor<32xf32>) -> tensor<32xf32> loc(#loc35)
      %acc_41 = "arith.extf"(%chunk_40) : (tensor<32xf32>) -> tensor<32xf64> loc(#loc36)
      %acc_42 = "arith.addf"(%acc_10, %acc_41) ... : (tensor<32xf64>, tensor<32xf64>) -> tensor<32xf64> loc(#loc36)
      "scf.yield"(%acc_42) : (tensor<32xf64>) -> () loc(#loc13)
    }) : (i32, i32, i32, tensor<32xf64>) -> tensor<32xf64> loc(#loc28)
    %9 = "arith.truncf"(%acc_9) : (tensor<32xf64>) -> tensor<32xf32> loc(#loc14)
    %10 = "arith.constant"() <{value = true}> : () -> i1 loc(#loc14)
    %11 = "tt.atomic_rmw"(%out_ptr, %9, %10) <{atomic_rmw_op = 5 : i32, scope = 1 : i32, sem = 4 : i32}> : (!tt.ptr<f32>, tensor<32xf32>, i1) -> f32 loc(#loc14)
    "tt.return"() : () -> () loc(#loc15)
  }) {noinline = false} : () -> () loc(#loc)
}) : () -> () loc(#loc)
```

The harness prints one dump per test case, and this is the first, with `float32` input. The second case used `float16` input and ends in the same `tt.atomic_rmw`. Nothing else came back: stderr contained only the `RuntimeError` line, with no separate MLIR diagnostic naming the failing operation.

At first glance, seeing raw compiler dumps while writing Python feels alarming. But if you have worked with MLIR (or are building a DSL compiler like Tesserae), this dump is gold. It shows exactly what Triton is doing before it ever touches LLVM or PTX. It was the genesis of this post, and of my interest in finding out what happened.

Two details in the dump are worth explaining before we move on:

* **The accumulator really is `float64`.** The dump contains a call to a `tl.zeros` specialized for `fp64` (`...zeros...cconstexpr_fp64`) and `tensor<32xf64>` throughout. So the kernel that failed used `tl.zeros(..., dtype=tl.float64)`, not the `float32` accumulator in the listing earlier in this post. The loaded chunk is still cast with `chunk.to(tl.float32)`, but the `+=` then promotes it to `f64`. That explains the `arith.extf` on the chunk (and, in the `float16` case, a chain of f16 → f32 → f64) and the `arith.truncf` back to `f32` before the atomic. The `f64` is incidental to the failure: when I reproduced this on a T4 (see below), a `float32` accumulator failed identically.
* **`col_stride` was folded into a constant.** The kernel takes `col_stride`, but the function signature has only four arguments. The pointer math instead contains `arith.constant 1 : i32` and `dense<1>` where the stride would be multiplied in. Triton specializes integer arguments equal to 1 as compile-time constants, and here `col_stride = 1`. This is also why the contiguity-based vectorization pattern from earlier can apply.

## Dissecting the MLIR Log: Why the Verifier Rejected the Kernel

Let's understand what the dump represents.

**MLIR dialects.** Triton lowers the Python AST into MLIR operations spanning several dialects:

* `tt`: The Triton dialect itself (`tt.func`, `tt.load`, `tt.addptr`, `tt.atomic_rmw`).
* `arith`: Standard integer and floating-point arithmetic (`arith.addi`, `arith.truncf`, `arith.constant`).
* `scf`: Structured Control Flow (`scf.for`, `scf.yield`).

Now look closely at the final three instructions before the return (`#loc14`):

```text
%9 = "arith.truncf"(%acc_9) : (tensor<32xf64>) -> tensor<32xf32> loc(#loc14)
%10 = "arith.constant"() <{value = true}> : () -> i1 loc(#loc14)
%11 = "tt.atomic_rmw"(%out_ptr, %9, %10) <{atomic_rmw_op = 5 : i32, scope = 1 : i32, sem = 4 : i32}> : (!tt.ptr<f32>, tensor<32xf32>, i1) -> f32 loc(#loc14)
```

Here is the exact mismatch:

* **`%out_ptr` (the target pointer):** Has type `!tt.ptr<f32>`. There is no shape dimension, so it is a **0D scalar pointer** to a single float in memory.
* **`%9` (the value to add):** Has type `tensor<32xf32>`, a **1D tensor of 32 floats** (matching `BLOCK_SIZE = 32`).
* **`%10` (the predicate mask):** Has type `i1`, a single 0D boolean. This is the `mask=None` default, which Triton fills in as constant `true`.
* **The operation:** `"tt.atomic_rmw"` (Atomic Read-Modify-Write). As far as I can tell, `atomic_rmw_op = 5` corresponds to the floating-point add (`FADD`), which is what `tl.atomic_add` uses on `f32` data.

### The verification failure

In the Triton dialect, `tt.atomic_rmw` requires its operands to agree in shape:

$$
\text{shape}(\text{pointer}) \equiv \text{shape}(\text{val}) \equiv \text{shape}(\text{mask}) \equiv \text{shape}(\text{return\_val})
$$

They must either be **all 0D scalars** (`!tt.ptr<f32>`, `f32`, `i1`), or **all tensors of the exact same shape** (`tensor<32x!tt.ptr<f32>>`, `tensor<32xf32>`, `tensor<32xi1>`).

When Triton's Python frontend processed `tl.atomic_add(out_ptr, acc, mask=None)`, it cast `acc` to the pointer's element type (hence the `arith.truncf` to `f32`). However, **it did not broadcast (splat) the scalar pointer `out_ptr` into a vector of pointers**. The result was a `tt.atomic_rmw` pairing a scalar address with a 32-element vector.

The only message we get is "error encountered during parsing", with no diagnostic naming the operation. But the dump is in MLIR's generic operation form, which MLIR prints when an operation fails verification. So my reading is that the verifier rejected the op, and the `RuntimeError` is just how that failure surfaces in Python. This is an inference from the IR, not something the error message states. When I reproduced the error on a Colab T4, the dump appeared in stdout and the exception message was exactly that one line, even with `MLIR_ENABLE_DIAGNOSTICS=1` set. I am not certain that is the right variable for this Triton version, so I would not read much into that.

*(By contrast, notice how pointer math inside our `for` loop worked: when we wrote `ptr = row_start_ptr + col_indices * col_stride`, adding the vector `col_indices` caused Triton to emit `"tt.splat"(%row_start_ptr_8) : (!tt.ptr<f32>) -> tensor<32x!tt.ptr<f32>>`, turning the scalar base pointer into a 32-lane vector of pointers before the load.)*

## Making It Compile, and Why I'd Still Use the Original

The error exists because the pointer is 0D while the value is 1D, so there are two ways to make the shapes agree. I tested both.

**Fix 1: reduce first, then do one scalar atomic per row.** Both the pointer and the value are 0D, so the verifier accepts it:

```python
row_sum = tl.sum(acc, axis=0)
tl.atomic_add(out_ptr + row_idx, row_sum)
```

**Fix 2: make the pointer a vector too.** Splat the row pointer across `BLOCK_SIZE` lanes, so every lane atomically adds its own partial sum:

```python
out_ptrs = (out_ptr + row_idx) + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
tl.atomic_add(out_ptrs, acc)
```

Both versions include the `+ row_idx` that the original experiment forgot.

### What I ran

I ran these on a Colab **T4** (Turing, `sm_75`; Triton 3.6.0, PyTorch 2.11.0), alongside the working `tl.sum` + `tl.store` baseline and the original failing `tl.atomic_add(out_ptr, acc)`. The four versions share one helper function for the accumulation loop, so the code differs slightly from the listing at the top, but the logic is the same. I used two inputs: the 2×3 matrix of ones with `BLOCK_SIZE = 32`, and an 8×1000 random matrix with `BLOCK_SIZE = 128`, checked against `x.sum(dim=1)`. Each working version ran twice: once with `out` zeroed, and once with `out` pre-filled with `100.0` to stand in for leftover `torch.empty` memory.

| Version | Compiles? | `out` zeroed | `out` pre-filled with 100.0 |
| ----- | ----- | ----- | ----- |
| Baseline: `tl.sum` + `tl.store` | Yes | Correct | Correct |
| Fix 1: `tl.sum`, then scalar `tl.atomic_add` | Yes | Correct | Every row off by 100 |
| Fix 2: splatted-pointer vector `tl.atomic_add` | Yes | Correct | Every row off by 100 |
| Original: scalar pointer, vector value | No: same `RuntimeError` | n/a | n/a |

With a zeroed output, both fixes give the right answers. On the 2×3 case all three working versions return `[3.0, 3.0]`. On the random case, the largest error against the reference was about 3.8e-6 for the baseline and Fix 1, and about 9.5e-6 for Fix 2. That is float32-sized rounding, and I would attribute the difference to a different summation order, though I did not isolate the cause. Fix 2 also did not double count: each lane's partial sum was added exactly once. All of the results in this table come from the T4. The original failure was on an H200 on TensorTonic.

### Back on TensorTonic

The T4 results only matter if the original platform accepts the fix, so I submitted Fix 1 there. The kernel keeps the `float64` accumulator from the original failing submission, reduces with `tl.sum`, and casts the scalar to `float32` for the atomic (TensorTonic's version of the kernel names the program id `i` and the column offsets `j`, rather than `row_idx` and `col_indices`):

```python
row_sum = tl.sum(acc, axis=0)
tl.atomic_add(out_ptr + i, row_sum.to(tl.float32))
```

TensorTonic let me modify `solve`, so I also zeroed the output before launching the kernel:

```python
out.zero_()
```

It passed all 44 of 44 tests on TensorTonic, and I ran it on both the H100 and H200 options.

![H100 test run](/diagrams/triton/H100.png)
*Figure 4: Passing 44/44 tests on NVIDIA H100 with the atomic kernel on TensorTonic*

![H200 test run](/diagrams/triton/H200.png)
*Figure 5: Passing 44/44 tests on NVIDIA H200 with the atomic kernel on TensorTonic*

I did not make note of the latencies but I suppose it would be relatively the same, considering the same hardware family. If you guys are very nit picky, I suggest running it directly on TensorTonic and 

So the atomic versions work. I would still prefer the original kernel for this problem, for two reasons, one of which matters much more for Fix 2 than for Fix 1.

### 1. Atomics accumulate onto whatever is already in `out`

* `tl.store` **overwrites** whatever bits were left in uninitialized memory.
* `tl.atomic_add` performs a **Read-Modify-Write**: it reads the existing value, adds your result to it, and writes it back.

The pre-filled run shows exactly this. Both atomic versions returned every row offset by the garbage value, while the baseline did not care. On LeetCode-style platforms like TensorTonic, the host harness plausibly allocates the output with `torch.empty(R, dtype=torch.float32)`. I have not checked that harness, but it is a common pattern. An atomic version of the solution would have to zero `out` before launching the kernel (for example with `out.zero_()`), and the baseline needs no such step.

### 2. Hardware: register shuffles vs. L2 atomics

Look at where the data lives at the end of the loop:

* In our working code, all `BLOCK_SIZE` partial sums are already in the **SM's Register File**.
* `tl.sum(acc, axis=0)` reduces them with **warp butterfly shuffles** (`shfl.sync.bfly.b32`): $\log_2(32) = 5$ shuffle steps, with threads exchanging data register-to-register and no memory traffic. A single scalar write (`tl.store`) then goes to Global Memory.
* `tl.atomic_add` instead sends updates out of the SM, through the interconnect, to the **L2 cache's atomic units**, which serialize competing updates to the same address (in Fix 2, all lanes of a row target the same address). I have not measured how much of this the hardware merges, but either way it turns an on-chip reduction into a round trip across the memory system.

This concern applies mostly to Fix 2, which issues one atomic per lane. Fix 1 issues one atomic per row, to a different address for every row, so I would expect little contention. I have not benchmarked any of the versions against each other, so this reason rests on how the hardware works, not on a measurement.

For this problem, then, the honest case for the baseline is simplicity: it needs no atomics and does not depend on `out` being zeroed. Fix 1 is a perfectly reasonable pattern, and it is exactly what you want when several programs contribute to one output element (see the table in the takeaway). Here a single program owns the whole row, so there is nothing to combine.

## Would the Behavior Change on Different GPUs?

A natural question: **would this behavior change on different hardware?** I first hit the error on an enterprise **NVIDIA H200** (Hopper, `sm_90a`). I have since reproduced it on a free Colab **NVIDIA T4** (Turing, `sm_75`, Triton 3.6.0). What about a consumer **NVIDIA GeForce RTX 5090** (Blackwell, `sm_120`)?

My prediction was that **it would not change at all**, and on the T4 that held: the same `RuntimeError`, with the same `tt.atomic_rmw` (scalar pointer, vector value) in the dump, for both test shapes and for both `float32` and `float64` accumulators. In the `float32` version there is no `arith.truncf` at all, which confirms that the `f64` in the original dump was a red herring. The RTX 5090 and AMD remain untested, so for those this is still reasoning, not a measurement.

### The compilation angle

Triton's compilation pipeline is split into stages:

$$
\text{Python AST} \longrightarrow \text{Triton MLIR (tt)} \longrightarrow \text{TritonGPU MLIR} \longrightarrow \text{LLVM IR} \longrightarrow \text{NVVM/PTX} \longrightarrow \text{SASS}
$$

The failure happens in the second stage, the Triton-dialect MLIR. The verifier runs on the **host CPU** and checks whether the operands of `"tt.atomic_rmw"` satisfy the dialect's type and shape contracts. A scalar pointer paired with a 1D tensor does not, so compilation aborts. This happens before any target-specific lowering: no register allocation for the GPU, no PTX, no CUDA driver calls. Since an invalid IR is invalid regardless of the target, the T4 result is what I expected, and I would expect the same on an RTX 5090 or even an AMD GPU through Triton's ROCm backend. I have no credits to test AMD, unfortunately.

### The microarchitectural angle

Suppose we make the kernel compile by splatting the pointer, as in the previous section. Would the hardware story differ across generations? Across all three GPUs—from 2018 Turing to Blackwell—the relevant hierarchy is the same:

1. **Registers are local to the SM.** A butterfly shuffle never leaves the execution pipeline.
2. **Atomics leave the SM.** Whether the memory behind L2 is GDDR6, HBM3e, or GDDR7, an atomic update must travel to the L2 slice that owns the address, where it is serialized against competing updates.

The memory technology changes the bandwidth numbers (roughly 320 GB/s for the T4's GDDR6 versus about 4.8 TB/s for the H200's HBM3e), but not which side of the SM boundary the reduction happens on. Register shuffles stay the cheaper option on every generation.

## The Takeaway and the Final Fix

The rule of thumb for reductions in Triton comes down to scope:

| Scope | Reduction Mechanism | Correct Primitive |
| ----- | ----- | ----- |
| **Intra-CTA reduction** (combining lanes *inside* a single thread block / tile) | Keep data on-chip: warp shuffles, plus shared memory when the block spans multiple warps | **`tl.sum(acc, axis=0)`** + **`tl.store(out_ptr + row_idx, row_sum)`** |
| **Inter-CTA reduction** (combining partial sums *across different CTAs*, which cannot see each other's registers) | Reduce to a scalar inside each CTA first, then atomically accumulate across blocks | **`tl.atomic_add(out_ptr + row_idx, block_scalar)`** (scalar pointer + scalar value; the output must be zero-initialized) |

For this kernel, where one CTA owns a whole row, stick to the intra-CTA pattern: reduce the accumulator on-chip with `tl.sum(acc, axis=0)`, then `tl.store` the final scalar. It avoids the MLIR error entirely, it does not depend on `out` being pre-zeroed (the atomic versions do), and it is the right default whenever a single program owns the whole reduction. Atomics become the right tool only when several programs must contribute to the same output element.

## Epilogue

What started as a one-line experiment with a peer ended up peeling back several layers of Triton's abstraction stack. It is easy to look at Python-based GPU DSLs like Triton and forget that beneath the clean decorator syntax is a multi-stage MLIR compiler checking dialect grammar, shapes, and type invariants. This is the kind of thing I like to do, and I would suggest it to anyone working on similar systems.

I have been writing CUDA C++ for some time, and picked up OCaml about 6 to 7 months ago. Both encourage you to look closely at compile-time and runtime errors, and OCaml, a functional-first language with a very capable compiler, has a lot more to teach.

My experience working on **Tesserae** and its intermediate representation, **Tirix**, reinforces the same lesson: a domain-specific compiler's expressiveness is only as good as the clarity of its verification passes. Here the verifier did its job. It refused an illegal operand combination before anything ran on the GPU. But the message I received, "error encountered during parsing", was far less helpful than the verdict behind it. For Tesserae, that is a useful reminder: catching bad IR is only half the job, and the diagnostic has to tell the user what went wrong. Understanding where the compiler's responsibility ends and where the hardware's memory hierarchy begins is, I think, the real secret to writing performant custom kernels.

This has also fired my enthusiasm to learn MLIR. Before this I had little to no knowledge of MLIR and I had to actually rely on an LLM to understand what this bug is. So this was a great reminder for me as well. 

### References

1. **Tillet, P., Kung, H. T., & Cox, D.** (2019). *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL). <https://dl.acm.org/doi/10.1145/3315508.3329973>
2. **LLVM / MLIR Documentation.** *MLIR: Multi-Level Intermediate Representation Overview & Dialect Concepts*. <https://mlir.llvm.org/>
3. **TensorTonic.** *Problem 344: Tiled Row Sum*. <https://www.tensortonic.com/problems/tiled-row-sum>
4. **NVIDIA Corporation.** *CUDA C++ Programming Guide: Warp Shuffle Functions & Atomic Functions*. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
5. **Triton source code.** The `tt.atomic_rmw` operation definition (`TritonOps.td`) and the atomic type-checking in the Python frontend (`semantic.py`), both in <https://github.com/triton-lang/triton>. File locations may have moved between versions.
