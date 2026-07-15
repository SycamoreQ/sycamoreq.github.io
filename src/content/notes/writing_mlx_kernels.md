---
title: Developing Apple Metal Kernels in Rust- A Deep Dive Through Attention 
date: 2026-07-15
description: --
tags: [Rust, Apple Metal]
coverImage: /cover_images/writing_mlx_kernels_1.jpg
reference: https://en.wikipedia.org/wiki/Macintosh_128K
referenceText: "cover image: Macintosh 128K: The first Mac PC"
draft: false
---

Writing a hand-rolled GPU kernel for Apple Silicon means working across two very
different layers: Metal Shading Language (MSL), the C++14-derived language the GPU
actually executes, and the host-side dispatch code that compiles, configures, and
launches that kernel. Most of the *conceptual* difficulty lives in the second layer,
not the first — MSL itself is a small, fairly boring language once you know the
attribute syntax. The launcher, on the other hand, is where you're coordinating
command queues, compute pipelines, threadgroup memory, and Rust's FFI story with
Objective-C all at once.

In this blog, I walk through both halves using the two kernels that make up scaled
dot-product attention: `attention_qk_f16`, which computes masked scaled dot products
between queries and a KV cache, and `attention_pv_f16`, which applies softmax and
accumulates the weighted sum over values. I'll spend relatively little time on the
MSL itself — it's the more familiar part if you've written CUDA — and much more time
on the launcher, since that's where the real API surface and the real bugs live.

---

## The MSL, briefly

Here is the first kernel in full. This is the entire computational core of QK^T
attention with a causal mask:

```text
#include <metal_stdlib>
using namespace metal;

kernel void attention_qk_f16(
    device const half*  Q           [[buffer(0)]], // [n_heads, head_dim]
    device const half*  K_cache     [[buffer(1)]], // [seq_len, n_heads, head_dim]
    device       half*  scores      [[buffer(2)]], // [n_heads, seq_len]
    constant     uint&  n_heads     [[buffer(3)]],
    constant     uint&  head_dim    [[buffer(4)]],
    constant     uint&  seq_len     [[buffer(5)]],
    constant     uint&  current_pos [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]] // tid.x = pos, tid.y = head
) {
    uint pos  = tid.x;
    uint head = tid.y;

    if (pos >= seq_len || head >= n_heads) return;

    float scale = 1.0f / sqrt(float(head_dim));
    float score = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        float q_val = float(Q[head * head_dim + i]);
        float k_val = float(K_cache[(pos * n_heads + head) * head_dim + i]);
        score += q_val * k_val;
    }
    score *= scale;

    if (pos > current_pos) {
        score = -INFINITY;
    }

    scores[head * seq_len + pos] = half(score);
}
```

A few syntax notes for anyone coming from CUDA rather than from Metal specifically.
`device const half*` declares a read-only pointer into device (GPU-visible) memory
holding 16-bit floats; the `[[buffer(N)]]` attribute binds that argument to the Nth
slot the host code will configure. `constant uint&` is the equivalent for small
uniform values — still read-only, but backed by Metal's constant address space
rather than device memory, which is the right space for scalars that every thread
reads identically. The thread's coordinates come from `thread_position_in_grid`, and
because we dispatch a 2D grid here (position × head), the type is `uint2` rather than
a bare `uint`. There is no explicit kernel launch configuration inside the `.metal`
file itself — that all lives on the host side, which is really the point of this
post.

The second kernel, `attention_pv_f16`, does more — it fuses softmax and the
value-weighted sum into a single dispatch using a threadgroup-level parallel
reduction:

```text
kernel void attention_pv_f16(
    device const half*  scores   [[buffer(0)]], // [n_heads, seq_len]
    device const half*  V_cache  [[buffer(1)]], // [seq_len, n_heads, head_dim]
    device       half*  output   [[buffer(2)]], // [n_heads, head_dim]
    constant     uint&  n_heads  [[buffer(3)]],
    constant     uint&  head_dim [[buffer(4)]],
    constant     uint&  seq_len  [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const uint THREADGROUP_SIZE = 32;
    uint head = gid;
    threadgroup float scratch[THREADGROUP_SIZE];

    float local_max = -INFINITY;
    for (uint i = lid; i < seq_len; i += THREADGROUP_SIZE) {
        local_max = max(local_max, float(scores[head * seq_len + i]));
    }
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] = max(scratch[lid], scratch[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = scratch[0];

    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += THREADGROUP_SIZE) {
        local_sum += exp(float(scores[head * seq_len + i]) - global_max);
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = scratch[0];

    for (uint d = lid; d < head_dim; d += THREADGROUP_SIZE) {
        float acc = 0.0f;
        for (uint i = 0; i < seq_len; i++) {
            float w = exp(float(scores[head * seq_len + i]) - global_max) / global_sum;
            acc += w * float(V_cache[(i * n_heads + head) * head_dim + d]);
        }
        output[head * head_dim + d] = half(acc);
    }
}
```

`threadgroup float scratch[THREADGROUP_SIZE]` allocates memory shared by every thread
in a single threadgroup — distinct from `device` memory (visible to the whole GPU)
and from per-thread registers. `threadgroup_barrier(mem_flags::mem_threadgroup)`
is a synchronization point: no thread proceeds past it until every thread in the
group has reached it, which is required here because threads are about to read
values other threads just wrote into `scratch`. Skipping this barrier is a
correctness bug, not a performance one — you'd read stale or half-written data.
That's essentially all the MSL syntax this project needed; the rest of the six
kernels in the engine (RMSNorm, RoPE, SwiGLU, tiled matmul) reuse the same small
vocabulary of address spaces, attributes, and barriers.

---

## The Launcher: where the real work is

Everything above compiles to something the GPU can execute, but getting Rust to
actually *run* it involves a surprisingly large API surface from `objc2_metal`,
the binding crate that wraps Apple's Objective-C Metal framework for Rust. Here is
the QK launcher in full, which we'll go through piece by piece:

```rust
pub fn attention_qk_f16(
    &self,
    ctx: &MetalContext,
    allocator: &MetalAllocator,
    q: &BlockHandle,
    k_cache: &BlockHandle,
    scores: &BlockHandle,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    current_pos: u32,
) -> Result<()> {
    let cmd_buf = ctx.command_buffer()?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

    encoder.setComputePipelineState(&self.attn_qk_pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q.gpu_buffer(allocator.buffer())), q.offset_bytes, 0);
        encoder.setBuffer_offset_atIndex(Some(k_cache.gpu_buffer(allocator.buffer())), k_cache.offset_bytes, 1);
        encoder.setBuffer_offset_atIndex(Some(scores.gpu_buffer(allocator.buffer())), scores.offset_bytes, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(&n_heads as *const u32 as *mut c_void),
            std::mem::size_of::<u32>(), 3,
        );
        // ... head_dim, seq_len, current_pos bound the same way
    }

    let grid = MTLSize { width: seq_len as usize, height: n_heads as usize, depth: 1 };
    let threadgroup = MTLSize { width: 1, height: 1, depth: 1 };
    unsafe {
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();
    }
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}
```

### `MTLCommandQueue`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder`

Metal's execution model has three layers of granularity, and understanding the
distinction between them is the single most important thing for getting a launcher
right. The **command queue** (`MTLCommandQueue`) is created once, at device
initialization, and lives for the whole program — every kernel dispatch across the
entire engine funnels through this one object. From it you request **command
buffers** (`MTLCommandBuffer`), which represent one discrete unit of GPU work you
intend to submit — think of a command buffer as a to-do list you build up and then
hand off atomically. Within a command buffer, you open a **command encoder**, and
this is the object you actually call methods on to record work: set a pipeline, bind
buffers, dispatch threads. A `MTLComputeCommandEncoder` specifically records compute
(as opposed to render or blit) commands.

The lifecycle is strict and one-directional: open an encoder, record everything you
need, call `endEncoding()`, and only then can you `commit()` the command buffer. You
cannot commit while an encoder is still open, and forgetting `endEncoding()` is one
of the most common sources of silent failures — the command buffer either refuses to
commit or the validation layer throws an error that can be non-obvious if you're not
watching for it.

In `objc2_metal`, all three of these are `Retained<ProtocolObject<dyn Trait>>` types —
`Retained<T>` being `objc2`'s answer to Objective-C's manual retain/release memory
model. It's conceptually an `Rc`: when it drops, the underlying Objective-C object's
reference count is decremented, and Metal's own internal lifecycle management frees
it when appropriate. You never call `retain` or `release` yourself; the type system
handles it.

### `MTLComputePipelineState`

Before any dispatch, you bind a pipeline state:

```rust
encoder.setComputePipelineState(&self.attn_qk_pipeline);
```

A pipeline state is the fully compiled, GPU-architecture-specific representation of
one kernel function, produced once from a `MTLFunction` (itself extracted from a
compiled `MTLLibrary`) and reused for every subsequent dispatch of that kernel. This
compile step — from MSL source, through an intermediate library, to a device-specific
pipeline — takes real wall-clock time, often tens of milliseconds per kernel, which is
why every one of the six kernels in this engine is compiled exactly once at startup,
inside a single `MetalKernels::new()` call, rather than on-demand per dispatch:

```rust
let library = device
    .newLibraryWithSource_options_error(&NSString::from_str(ATTN_QK_MSL), None)
    .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

let function = library
    .newFunctionWithName(ns_string!("attention_qk_f16"))
    .ok_or(MetalError::KernelNotLoaded("attention_qk_f16"))?;

let attn_qk_pipeline = device
    .newComputePipelineStateWithFunction_error(&function)
    .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;
```

`newLibraryWithSource_options_error` is runtime MSL compilation — the simplest path
during development, since compile errors come back as descriptive `NSError` strings
rather than a separate offline build step. Production pipelines often precompile to
a `.metallib` file via `xcrun metal`/`metallib` and load it with
`newLibraryWithData`, trading a build-time dependency for faster process startup;
for this project, runtime compilation was the right tradeoff since kernels were
still being actively iterated on.

### `setBuffer_offset_atIndex` and unified memory

```rust
encoder.setBuffer_offset_atIndex(Some(q.gpu_buffer(allocator.buffer())), q.offset_bytes, 0);
```

This is the call that binds a GPU buffer to a specific `[[buffer(N)]]` argument slot;
the index must match the MSL signature exactly. The three arguments are the
`MTLBuffer` itself, a byte offset into it, and the target index. The byte offset
matters a great deal here: several tensors in this engine are *views* into a larger
allocation — the product of a paged block allocator that hands out sub-regions of one
big buffer — so different logical tensors can share a physical `MTLBuffer` at
different offsets. Passing the offset lets the same kernel read the right region
without a separate allocation per tensor.

This whole scheme is only free because of Apple Silicon's unified memory
architecture. On a discrete GPU, host and device memory are physically separate, and
moving data between them means an explicit copy (`cudaMemcpy` and friends). On Apple
Silicon, the CPU and GPU read the same physical DRAM; a buffer allocated with
`MTLResourceStorageModeShared` is simultaneously valid from both sides. In practice
this means that after `waitUntilCompleted()` returns, you can dereference the same
`ptr` you handed to the GPU and see its output directly — no explicit device-to-host
transfer step exists in this codebase because none is needed.

For the small scalar arguments — `n_heads`, `head_dim`, and so on — a full buffer
allocation would be wasteful, so Metal provides `setBytes_length_atIndex`, which
copies inline bytes straight into the command encoder's argument table:

```rust
encoder.setBytes_length_atIndex(
    NonNull::new_unchecked(&n_heads as *const u32 as *mut c_void),
    std::mem::size_of::<u32>(),
    3,
);
```

The double cast — `*const u32` to `*mut c_void` to `NonNull<c_void>` — exists purely
to satisfy `objc2_metal`'s binding signature; it's `unsafe` because it crosses the
Rust/Objective-C FFI boundary, but sound in practice since the encoder reads the
bytes synchronously before the call returns.

### `MTLSize` and thread dispatch

```rust
let grid = MTLSize { width: seq_len as usize, height: n_heads as usize, depth: 1 };
let threadgroup = MTLSize { width: 1, height: 1, depth: 1 };
encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
```

`MTLSize` is a plain three-field struct describing a grid in up to three dimensions.
`dispatchThreads_threadsPerThreadgroup` takes two of them: the total thread count
across the whole dispatch, and how many threads should be grouped together into a
single threadgroup. For the QK kernel every thread is fully independent — thread
`(pos, head)` computes exactly one score and touches no shared state — so a
threadgroup size of `(1, 1, 1)` costs nothing and simplifies reasoning.

For the softmax+PV kernel, threadgroup size stops being a free choice, because the
reduction requires every thread contributing to one head's softmax to actually be in
the *same* threadgroup — `threadgroup` memory and `threadgroup_barrier` only
synchronize within a group, never across the whole grid. That kernel dispatches one
threadgroup per attention head, 32 threads each:

```rust
let grid = MTLSize { width: (n_heads * 32) as usize, height: 1, depth: 1 };
let threadgroup = MTLSize { width: 32, height: 1, depth: 1 };
```

### Commit, wait, and reading results back

```rust
cmd_buf.commit();
cmd_buf.waitUntilCompleted();
```

`commit()` hands the recorded command buffer to the GPU and returns immediately —
execution is asynchronous by default. `waitUntilCompleted()` blocks the calling CPU
thread until the GPU has actually finished. For an inference engine, where layer `N`'s
output tensor is layer `N+1`'s input, this synchronous wait is the correct choice —
you cannot proceed without the result. A more heavily pipelined design could overlap
CPU and GPU work across independent branches using `MTLSharedEvent`, but the
straightforward wait-per-dispatch model kept the whole system easy to reason about
during development, which mattered more early on than shaving dispatch latency. Once
`waitUntilCompleted()` returns, the output is immediately valid to read from the CPU
side, again with no explicit sync step required beyond the wait itself.

---

## Testing

Each kernel is validated independently against a CPU reference before being trusted
inside the full model. For the two attention kernels specifically, three properties
get checked: numerical correctness of the scaled dot product against an f32 CPU
implementation (within a tolerance appropriate for f16 accumulation, since
`head_dim` additions of quantized values accumulate real rounding error); correctness
of the causal mask, where every position beyond `current_pos` must read back as
exactly `-infinity`; and behavior under asymmetric, non-power-of-two dimensions,
which exercises the boundary-guard logic (`if (pos >= seq_len || head >= n_heads)
return;`) that a suspiciously-round test size like 64 could hide a bug behind.

It's worth looking at what one of these tests actually does end to end, since the
setup says as much about how the engine manages GPU memory as the kernel itself
does. Here's the harness shared by every attention test:

```rust
fn setup() -> (MetalContext, MetalAllocator, AttentionQKKernel) {
    let device = MetalDevice::system_default().unwrap();
    let ctx = MetalContext::new(device).unwrap();
    let alloc = MetalAllocator::new(&ctx, 16 * 1024 * 1024).unwrap();
    let kernel = AttentionQKKernel::new(ctx.device.raw()).unwrap();
    (ctx, alloc, kernel)
}
```

Four objects, each owning a different layer of the stack: `MetalDevice` is the thin
wrapper around `MTLCreateSystemDefaultDevice()`; `MetalContext` holds the command
queue built from that device; `MetalAllocator` pre-reserves a 16MB `MTLBuffer` up
front and hands out sub-regions of it on request, so individual tests don't each
pay the cost of a fresh Metal buffer allocation; and the kernel struct itself
compiles the MSL source and builds the pipeline state exactly once, in its own
`new()`. Every test in the suite calls `setup()` fresh, so there's no shared mutable
state between tests — a deliberate tradeoff of a little redundant setup work for
tests that can't interfere with each other.

Buffer initialization itself happens through the allocator, then gets written to
directly via the CPU pointer — the unified-memory point from earlier isn't just a
performance detail, it's what makes writing a test this simple:

```rust
#[test]
fn test_attention_qk_execution() {
    let (ctx, mut alloc, kernel) = setup();

    let head_dim = 64usize;
    let seq_len  = 64usize;
    let n_heads  = 12usize;
    let current_pos = 63usize;

    let q_f16: Vec<f16> = (0..(n_heads * head_dim))
        .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
        .collect();
    let k_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
        .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
        .collect();

    let q_bytes      = n_heads * head_dim * std::mem::size_of::<f16>();
    let k_bytes      = seq_len * n_heads * head_dim * std::mem::size_of::<f16>();
    let scores_bytes = n_heads * seq_len * std::mem::size_of::<f16>();

    let q_block      = alloc.alloc(q_bytes, 16).unwrap();
    let k_block      = alloc.alloc(k_bytes, 16).unwrap();
    let scores_block = alloc.alloc(scores_bytes, 16).unwrap();

    unsafe {
        let q_ptr = q_block.ptr as *mut f16;
        let k_ptr = k_block.ptr as *mut f16;
        for i in 0..n_heads * head_dim {
            q_ptr.add(i).write(q_f16[i]);
        }
        for i in 0..seq_len * n_heads * head_dim {
            k_ptr.add(i).write(k_f16[i]);
        }
    }

    kernel.attention_qk_f16(
        &ctx, &alloc, &q_block, &k_block, &scores_block,
        n_heads as u32, head_dim as u32, seq_len as u32, current_pos as u32,
    ).unwrap();

    let expected = attention_qk_ref(&q_f16, &k_f16, head_dim, n_heads, seq_len);
    unsafe {
        let out_ptr = scores_block.ptr as *const f16;
        for i in 0..(n_heads * seq_len) {
            let got = out_ptr.add(i).read().to_f32();
            assert!((got - expected[i]).abs() < 0.5, "mismatch at index {i}");
        }
    }
}
```

`alloc.alloc(bytes, alignment)` is the only step that talks to the allocator; it
returns a `BlockHandle` carrying both a byte offset into the pool's `MTLBuffer` and
a raw `*mut u8` into that same memory, valid because the buffer was created with
`MTLResourceStorageModeShared`. From there, writing test input is just pointer
arithmetic — cast to `*mut f16` and `.write()` each element — with no API call to
Metal at all. The kernel launch itself is the one point where the CPU-written data
actually gets handed to the GPU (via the buffer-binding calls from the previous
section), and reading the result back after `waitUntilCompleted()` is the exact same
pattern in reverse: cast the `BlockHandle`'s pointer to `*const f16` and read.

The causal-mask test reuses this harness but changes only the assertion, which is a
good illustration of why splitting setup from assertions paid off:

```rust
#[test]
fn test_attention_causal_mask() {
    let (ctx, mut alloc, kernel) = setup();
    // ... same q_block/k_block/scores_block construction as above ...

    let current_pos = 31usize; // only positions 0..=31 should be unmasked

    kernel.attention_qk_f16(
        &ctx, &alloc, &q_block, &k_block, &scores_block,
        n_heads as u32, head_dim as u32, seq_len as u32, current_pos as u32,
    ).unwrap();

    unsafe {
        let out_ptr = scores_block.ptr as *const f16;
        for i in 0..(n_heads * seq_len) {
            let got = out_ptr.add(i).read().to_f32();
            let pos = i % seq_len; // layout is [n_heads, seq_len]
            if pos > current_pos {
                assert_eq!(got, f32::NEG_INFINITY, "index {i} (pos {pos}) should be masked");
            } else {
                assert!((got - expected[i]).abs() < 0.5, "mismatch at index {i}");
            }
        }
    }
}
```

Same buffers, same kernel call, same readback pattern — only `current_pos` and the
per-element assertion change. That reuse is what made it cheap to also test
deliberately awkward, non-power-of-two dimensions (`head_dim=48`, `seq_len=35`,
`n_heads=8`) as a third variant, which is exactly the kind of case that catches an
off-by-one in the `if (pos >= seq_len || head >= n_heads) return;` guard that a
suspiciously round `head_dim=64` would never exercise.

Below is the actual test output from the attention kernel suite, run against Apple
M4 hardware:

```
running 11 tests
command buffer status: MTLCommandBufferStatus(0)
context created on: Apple M4
test metal::context::tests::test_synchronize_empty ... ok
test metal::context::tests::test_synchronize_repeated ... ok
test metal::context::tests::test_command_buffer_creation ... ok
test metal::context::tests::test_context_creation ... ok
test metal::allocator::tests::test_basic_alloc ... ok
test metal::allocator::tests::test_exhaustion ... ok
test metal::allocator::tests::test_alignment_padding ... ok
test metal::allocator::tests::test_cpu_write_read ... ok
Metal device: Apple M4
Unified memory: true
Max working set: 12124 MB
test metal::allocator::tests::test_cursor_reclaim ... ok
test metal::allocator::tests::test_free_reuse ... ok
test metal::device::tests::test_device_acquisition ... ok
test metal::kernels::tests::test_rope_multi_token ... ok
test metal::kernels::tests::test_rope_position_zero_is_identity ... ok
test metal::kernels::tests::test_rms_norm_single_token ... ok
test metal::kernels::tests::test_rms_norm_multi_token ... ok
test metal::kernels::tests::test_rms_norm_weight_scaling ... ok
test metal::kernels::tests::test_matmul_f32_execution ... ok
test metal::kernels::tests::test_matmul_non_multiple_of_blocksize ... ok
test metal::kernels::tests::test_rope_different_positions_differ ... ok
test metal::kernels::tests::test_swiglu_f16_execution ... ok
test metal::kernels::tests::test_attention_pv_execution ... ok
test metal::kernels::tests::test_attention_qk_execution ... ok

test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 570 filtered out; finished in 0.08s
```

Every kernel in the stack — device acquisition, the paged allocator, RMSNorm, RoPE,
SwiGLU, tiled matmul, and both attention passes — passing in the same run on real
Apple Silicon hardware was the first point at which the Metal backend was trustworthy
enough to wire into the rest of the inference engine.

# Whats next 

[Axiom](https://github.com/SycamoreQ/axiom) is an Inference Engine in Rust and actively developing to host it on my Apple Macbook Air. In the upcoming blogs, I will try to document my process and hopefully piece things together into a big picture with respect to the Inference Engine by basically explaining the entirety of it. I have also faced several nasty bugs while developing the engine, which I also will document, possibly will be very next blog I write.

# References 

[kleideiAI](https://github.com/ARM-software/kleidiai/tree/main/examples) is a microkernel repository and basically the bedrock of all the kernels written in axiom. I think it is a very important and quite prominent resource for anyone writing MSL kernels. Apart from that, I referred to the official [objc2_metal rust crate repository](https://docs.rs/objc2-metal/latest/objc2_metal/index.html) for the launcher code as well as the [Apple MSL documentation](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) to learn about what the APIs are. 

