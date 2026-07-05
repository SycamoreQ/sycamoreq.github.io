---
title: Optimizing The SAXPY Kernel In ARM NEON
date: 2026-07-15
description: This blog optimizes the SAXPY kernel
tags: [assembly, ARM, OCaml]
coverImage: /cover_images/writing_arm_kernels_2.jpg
reference: https://mastodon.social/@dougall/115149886886125067
referenceText: "cover image: Apple M1 Chip Die shot"
draft: false
---

# Optimizing SAXPY: A Worklog

In the last blog, we looked at how we write a simple ARM NEON assembly kernel and how the testing pipeline is done in Seraph. In this blog we will look at the most important part of writing low-level kernels: optimization. We will be going through it like a worklog — I will introduce new optimizations, explain their place in the kernel (the why), and then show the change in throughput using the test pipeline we introduced last time.

You can find the kernel code in [Seraph](https://github.com/SycamoreQ/seraph/blob/main/lib/kernels/saxpy_optim.s)

---

## Why Optimize?

A question like this is only trivial to the uninitiated. Optimization specifically for high-performance kernels is very crucial to juice out performance because in order to get that performance we need to understand what we are running these kernels on — in our case, an ARM chip. Writing an optimized kernel means that we know how the hardware is laid out and how to modify our kernel to efficiently utilize the hardware resources. This turns a very boring modification into something worth speculating upon (pun intended).

This also aligns with what I have been working on in Tesserae, my OCaml DSL for CuTE and CUTLASS GEMM kernels. NVIDIA hardware is incredibly dense and the performance of each of its GPUs is very representative of the underlying hardware. For example, post the Hopper (SM90) architectures, NVIDIA GPUs are equipped with something called Tensor Memory — memory that is shared between the different arithmetic clusters or Streaming Multiprocessors. Tensor Memory manages how memory is represented in shared memory, how it can be accessed without any bank conflicts, and how it efficiently "tiles" matrices for high throughput across all the SMs. All of this is only achievable if there is the hardware, and also when the kernel is written in a specific way that is representative of that hardware.

So in summary we should just have a sense of the architecture and also the features the library gives us to write kernels.

With that being said, let's start with the base kernel we wrote last blog:

```asm
// saxpy_neon.S
//
// void saxpy_neon(float a, const float *x, float *y, int n)
//
// y[i] = a * x[i] + y[i]   for i in [0, n)
//
// AAPCS64 argument registers for this signature:
//   s0 = a   (first FP/SIMD arg)
//   x0 = x   (first integer/pointer arg)
//   x1 = y   (second integer/pointer arg)
//   w2 = n   (third integer arg, 32-bit)
//
// This is a leaf function and only touches v0-v3, w2-w4, x0-x1 — all of
// which are caller-saved under AAPCS64, so no register save/restore is
// needed here.
//
// .S (capital S) is preprocessed by the C preprocessor before assembly,
// which is what makes the FN() macro below work — on macOS/Mach-O, C
// symbols need a leading underscore; on Linux/ELF they don't.

#if defined(__APPLE__)
#define FN(name) _##name
#else
#define FN(name) name
#endif

    .text
    .global FN(saxpy_neon)
#if !defined(__APPLE__)
    .type FN(saxpy_neon), %function
#endif

FN(saxpy_neon):
    dup     v1.4s, v0.s[0]      // broadcast a into all 4 lanes -> v1
    mov     w3, w2              // w3 = n
    lsr     w4, w3, #2          // w4 = n / 4   (number of 4-wide iterations)
    and     w3, w3, #3          // w3 = n % 4   (tail count)
    cbz     w4, .Ltail

.Lvec_loop:
    ld1     {v2.4s}, [x0], #16  // load 4 floats from x, advance x0 by 16 bytes
    ld1     {v3.4s}, [x1]       // load 4 floats from y (no advance yet)
    fmla    v3.4s, v1.4s, v2.4s // v3 = v3 + a*v2   (fused multiply-add)
    st1     {v3.4s}, [x1], #16  // store result back to y, advance x1
    subs    w4, w4, #1
    b.ne    .Lvec_loop

.Ltail:
    cbz     w3, .Ldone

.Ltail_loop:
    ldr     s2, [x0], #4
    ldr     s3, [x1]
    fmadd   s3, s0, s2, s3      // s3 = a*s2 + s3   (scalar fused multiply-add)
    str     s3, [x1], #4
    subs    w3, w3, #1
    b.ne    .Ltail_loop

.Ldone:
    ret
```

---

## The Problem With One Load At A Time

To understand the fix we first need to understand why a single `ld1` per iteration is slower than it looks.

The Apple Silicon load/store unit is a pipeline. It can *accept* a new load request every cycle, but the data does not come back immediately — there is a latency of several cycles even for L1 cache. What matters here is the distinction between **latency** (how long a single load takes end to end) and **throughput** (how many loads per cycle the unit can accept in steady state). The unit has excellent throughput but non-trivial latency.

In our base kernel's main loop:

```asm
.Lvec_loop:
    ld1     {v2.4s}, [x0], #16  // issue load for x chunk
    ld1     {v3.4s}, [x1]       // issue load for y chunk
    fmla    v3.4s, v1.4s, v2.4s // WAIT: needs v2 and v3 to have arrived
    st1     {v3.4s}, [x1], #16
    subs    w4, w4, #1
    b.ne    .Lvec_loop
```

The `fmla` instruction has a read-after-write (RAW) dependency on both `v2` and `v3`. It cannot begin executing until both loads have completed and their results are written back to the register file. So the CPU issues the first `ld1`, issues the second `ld1`, and then stalls — it cannot advance to `fmla` until both loads retire. The load pipeline sits idle during that stall.

This is the fundamental problem: we are issuing just enough loads to feed one iteration, and the arithmetic unit has to wait every single time. There is no overlap between loading data for the *next* iteration and computing the *current* one, because we never give the hardware a chance to look ahead.

Clang's autovectoriser does not do this. When you hand it `saxpy` with `-O3`, it unrolls the loop and issues loads for multiple future iterations before the current `fmla` chain completes. By the time a given `fmla` needs its data, that data has already been fetched and is sitting in a register, because the CPU's out-of-order engine saw the load far enough in advance and issued it early. This is what we need to replicate by hand.

---

## The Fix: Multi-Register Loads

The first piece of the fix is switching from single-register `ld1` to multi-register `ld1`. AArch64's `ld1` instruction has a list form:

```asm
ld1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x0], #64
```

This loads 4 vector registers — 16 floats, 64 bytes — in a single instruction. The key difference from issuing four separate `ld1` instructions is that this is a **single memory transaction**: the load unit sees one large request rather than four small ones, which it can pass to the cache subsystem more efficiently. More importantly, when we immediately follow it with a second multi-register load for `y`:

```asm
ld1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x0], #64
ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64
```

We now have two large load transactions in flight before a single `fmla` is issued. The out-of-order engine sees both requests simultaneously, can issue them to the cache in parallel, and the hardware prefetcher — recognising the regular 64-byte stride — begins fetching the *next* iteration's data while we are still computing the current one. By the time we reach the first `fmla`, the data has already arrived.

---

## Loop Unrolling: The Multiplicative Gain

Loop unrolling, although I had mentioned it explicitly as an optimization in the last blog, is quite well hidden when we do the multi-vector loading. Since we load 4 vectors at the same time, we process 4× more elements per branch — that is the unrolling. There is of course more nuance to this: I would like to think loop unrolling is a way to give independent work to the out-of-order execution engine so it can exploit concurrent loads. Unrolling the loop N times but still using single-register loads will reduce loop overhead, but will not hide load latencies — that depends on how many registers *exist* in the first place to hold in-flight data. Since not enough registers are live at any given moment with single loads, operations waiting for data will be kept waiting until the right value lands in the said register.

Even upon batching loads, processing only 4 elements per iteration (i.e. one 4-register load per branch) would reduce decoding overhead but would generate too few independent `fmla` instructions — not enough to keep the execution units busy while the next load is in flight. Therefore both loop unrolling and multi-register loads are multiplicative in terms of boosting performance: neither one alone is sufficient.

### The Register Stride Constraint

There is one assembler constraint that is easy to trip over here. The multi-register `ld1` list requires the registers to be **sequential** — `vN, vN+1, vN+2, vN+3` — because the instruction encoding only stores a starting register and a count, not an arbitrary list. You cannot write:

```asm
ld1     {v6.4s, v7.4s, v16.4s, v17.4s}, [x1], #64  // ASSEMBLER ERROR
```

The jump from `v7` to `v16` breaks the sequential stride requirement and the assembler will reject it:

```
error: registers must have the same sequential stride
```

So the register choice for the second load is constrained by whatever the first load uses. Since `v2–v5` are taken by the `x` load, the `y` load naturally falls into `v6–v9`. That means `v8` and `v9` are now live in the hot loop — and `v8`/`v9` (specifically their low 64 bits, `d8`/`d9`) are **callee-saved** under the ARM Architecture Procedure Calling Standard (AAPCS64). Any function that writes to them is obligated to restore them before returning.

This is not optional. The caller — OCaml's runtime, in our case — is entitled to assume that `d8`/`d9` survive a function call unchanged. If we clobber them without saving and restoring, we corrupt the caller's register state. The program may not crash immediately, but it will eventually: the corruption accumulates silently across calls and manifests as a SIGSEGV or wrong results at a completely unrelated point in the program.

The fix is a standard prologue/epilogue pair:

```asm
FN(saxpy_optim):
    sub     sp, sp, #32
    stp     d8, d9, [sp]
    stp     d10, d11, [sp, #16]
    ...
.Ldone:
    ldp     d8, d9, [sp]
    ldp     d10, d11, [sp, #16]
    add     sp, sp, #32
    ret
```

`stp` and `ldp` are store-pair and load-pair: they write or read two registers in a single instruction. The stack pointer adjustment is 32 bytes (16-byte aligned as AAPCS64 requires), giving us room for four 8-byte `d`-registers. The cost is 4 extra memory operations per function call — not per loop iteration — so it amortises to nothing over any non-trivial `n`.

---

## The Full Optimized Loop

Putting it together, our optimized SAXPY kernel becomes:

```asm
FN(saxpy_optim):
    sub     sp, sp, #32
    stp     d8, d9, [sp]
    stp     d10, d11, [sp, #16]
    dup     v1.4s, v0.s[0]          // broadcast alpha into all 4 lanes
    mov     w4, w2                   // w4 = n   (note: n is in w2, not w3 —
    lsr     w5, w4, #4               //           float arg a takes s0/v0,
    cbz     w5, .Lshort_setup        //           shifting integer args left)

.Lmain_loop:
    ld1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x0], #64
    ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64
    fmla    v6.4s, v2.4s, v1.4s
    fmla    v7.4s, v3.4s, v1.4s
    fmla    v8.4s, v4.4s, v1.4s
    fmla    v9.4s, v5.4s, v1.4s
    sub     x1, x1, #64
    st1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64
    subs    w5, w5, #1
    b.ne    .Lmain_loop

.Lshort_setup:
    and     w5, w2, #15
    cbz     w5, .Ldone

.Lshort_loop:
    ldr     s2, [x0], #4
    ldr     s6, [x1]
    fmadd   s6, s2, s0, s6
    str     s6, [x1], #4
    subs    w5, w5, #1
    b.ne    .Lshort_loop

.Ldone:
    ldp     d8, d9, [sp]
    ldp     d10, d11, [sp, #16]
    add     sp, sp, #32
    ret
```

Quite a few things changed compared to the one we wrote in the previous blog. We have seen quite a few powerful optimizations and they all basically combine into this multi-staged "waterfall" like structure. A few additional things worth noting:

The main loop now processes **16 elements per iteration** instead of 4. The trip count is therefore `n >> 4` (`lsr w5, w4, #4`) instead of `n >> 2`, and the remainder mask is `#15` (`0b00001111`) instead of `#3`. The store is also batched: `st1 {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64` writes all four updated `y` chunks back to memory in one instruction, symmetric with the multi-register load.

There is also a calling convention subtlety worth flagging. The function signature is `saxpy_optim(float a, const float *x, float *y, int n)`. Under AAPCS64, the float argument `a` goes into `s0`/`v0` (the first FP/SIMD register), while the three integer/pointer arguments `x`, `y`, `n` go into the general-purpose register file as `x0`, `x1`, `w2` respectively. The base kernel used `w3` for `n` — which was wrong, and was the root cause of the crash at large `n` that we debugged earlier. `w3` held garbage, the loop iterated a garbage number of times, and at n=4,000,000 it walked far off the end of the buffer. The fix is mechanical: `n` arrives in `w2`, not `w3`, so every reference to the trip count and remainder uses `w2`.

---

## Instruction Count Comparison

It is worth being explicit about what changed in terms of raw instruction count, since that directly explains the decode/dispatch throughput improvement at L1.

The base kernel's main loop body (excluding loop-control):

```asm
ld1     {v2.4s}, [x0], #16     // 1
ld1     {v3.4s}, [x1]          // 2
fmla    v3.4s, v1.4s, v2.4s   // 3
st1     {v3.4s}, [x1], #16     // 4
```

4 instructions, 4 elements processed.

The optimized kernel's main loop body:

```asm
ld1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x0], #64    // 1
ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64    // 2
fmla    v6.4s, v2.4s, v1.4s                         // 3
fmla    v7.4s, v3.4s, v1.4s                         // 4
fmla    v8.4s, v4.4s, v1.4s                         // 5
fmla    v9.4s, v5.4s, v1.4s                         // 6
sub     x1, x1, #64                                 // 7
st1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x1], #64    // 8
```

8 instructions, 16 elements processed. That is 0.5 instructions per element versus 1.0 instructions per element — a 2× improvement in instruction density, before even accounting for the load-latency hiding. Adding in the loop-control overhead (`subs` + `b.ne`) makes the picture even more favourable: the base kernel pays that overhead every 4 elements, the optimized kernel every 16.

---

## Results

Running the full benchmark suite after this change:

| Regime | n | neon baseline | saxpy_optim | clang -O3 |
|:---|---:|---:|---:|---:|
| L1 | 256 | 19.91 ns/call | 16.50 ns/call | 16.87 ns/call |
| L2 | 65,536 | 5545 ns/call | 5324 ns/call | 5260 ns/call |
| DRAM | 4,000,000 | 378,672 ns/call | 358,200 ns/call | 373,570 ns/call |

Or in throughput terms:

| Regime | baseline GFLOP/s | optim GFLOP/s | clang GFLOP/s |
|:---|---:|---:|---:|
| L1 | 25.72 | 31.03 | 30.36 |
| L2 | 23.64 | 24.62 | 24.92 |
| DRAM | 21.13 | 22.33 | 21.41 |

### Reading the Numbers

The result is clean. `saxpy_optim` now *beats* Clang at L1 (31.03 vs 30.36 GFLOP/s) and is within 1–2% at L2 and DRAM — essentially statistical noise given the run-to-run variance at those regimes. The 5–18% gap from the base kernel is gone.

The L1 result is the most satisfying. At n=256 the entire working set fits in L1 and memory latency is nearly free, so what we are measuring is purely the instruction-level throughput improvement. Going from 1.0 instructions per element to 0.5 instructions per element shows up directly: 31.03 GFLOP/s vs 25.72 GFLOP/s is a 20.6% improvement, consistent with halving the instruction count.

At L2 and DRAM, the gain is smaller (~4–5%) because both kernels are now spending a similar fraction of their time waiting on the memory hierarchy rather than on instruction throughput. But notice that `saxpy_optim` is slightly *faster* than Clang at DRAM (22.33 vs 21.41 GFLOP/s) — the multi-register load transactions give the hardware prefetcher a cleaner signal about the access pattern (two regular 64-byte stride streams rather than eight irregular 16-byte ones), which helps slightly at large working sets where prefetch accuracy matters.

---

## One Thing I Did Not Do

It is worth being honest about what this optimization is and is not. We improved instruction density and load-latency hiding, but we did not do any explicit **software prefetching** — we never issued a `prfm` instruction to tell the hardware to start fetching data for iterations that are still several steps away. For a bandwidth-bound kernel at DRAM sizes, explicit prefetching can sometimes squeeze out another 5–10% by reducing the number of cycles spent waiting on a cache miss that the hardware prefetcher did not predict early enough.

We also did not tune the unroll factor analytically — we picked 4×4s (16 elements per iteration) by intuition and it happened to work well. A more rigorous approach would sweep unroll widths (8, 16, 32 elements) and pick the one that best matches Apple Silicon's specific store-to-load forwarding and reorder buffer depth. That is an interesting experiment for another day.

---

## What Comes Next

This post covered two optimizations — load batching and loop unrolling — applied to one kernel: SAXPY. The same technique applies to dot product and sum reduction, and both of those have their own interesting constraints (dot product needs to reduce across vector lanes at the end; sum reduction needs multiple independent accumulators to break the dependency chain). In the next post we will look at those kernels and see whether the same load-batching principle transfers, or whether their structure demands a different approach.

At the time of writing this blog, I have written 4 complete kernels including SAXPY and am currently writing an Attention kernel. It is incredibly difficult, but worth the effort. I will try documenting that as well, and also my other MLX-related work I am doing in Axiom — which is probably the subject of my next blog, not Seraph.

---

## References

A lot of my knowledge on compiler and hardware optimizations — the concepts and other technicalities — comes from my experience with writing Tesserae as mentioned in the beginning of this blog, as well as the highly cited book [*Computer Architecture: A Quantitative Approach* by Hennessy and Patterson](https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-443-15406-5). Apart from this, the ARM NEON documentation helped me learn some other very interesting things like ZA storage and so on. [This 31-minute video](https://www.youtube.com/watch?v=5peYlm5j0U8) on writing a MATMUL kernel in Scalable Matrix Extension (SME2) was great in building my intuition on how MATMUL operations are actually batched and computed in ARM hardware.