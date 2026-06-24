---
title: Writing A SAXPY Kernel In ARM NEON
date: 2025-01-01
description: This blog goes over writing an ARM NEON kernel from scratch
tags: [assembly, ARM, OCaml]
---

# Writing A SAXPY Kernel in ARM NEON

## Introduction

Hello! This blog is the output of some ARM NEON I have learnt the past few days. I will be writing a SAXPY kernel in ARM NEON and then link it with an OxCaml test suite to see how it does. I will also be implementing other kernels like convolutions or even basic attention mechanisms. This will mark the beginning of it all.

I will be calling this project Seraph and I will be showing you all how I compile it and test it using an OCaml script. Then I will go through each kernel line by line and gather my research around each opcode (if it has any depth). I hope this will be a pleasant read. In order to get a good idea of ARM NEON I will also be referencing a few resources I have read as well at the end. The code is of course in the GitHub repo [Seraph](https://github.com/SycamoreQ/seraph).

I will also be writing a string of blogs on the updates on Seraph and as of writing this blog, I have already benchmarked 3 kernels along with SAXPY. So I will definitely be writing more kernels, particularly closer to Machine Learning, benchmark it and write a worklog on it. 

---

## The Setup

Seraph for now is basically an ARM NEON kernel repository and it will be so for sometime until I get the hang of things. There is some setup with respect to the tests I have written and the OCaml–C FFI files and so on which I will be going over now.

### Assembling

Given our NEON code in the `.S` format, we first **assemble** it. Assemble means to convert our asm code — in our case ARM NEON — into machine code, i.e. the `.o` format. This is done specifically by clang.

Clang, which is the default compiler frontend for the machine I am using (macOS), converts or "assembles" the code into machine code. Let us keep that aside.

We also have something called `shim.c`. Since `external` in OCaml does not use raw C ABI calling conventions, it gives us OCaml's own runtime representation. This representation has nothing in common with the hand-written ARM NEON assembly. So `shim.c` is an adapter that links them both.

More precisely, the `external` line in `bindings.ml`:

```ocaml
external dot_neon: f32arr -> f32arr -> int -> float = "caml_dot_neon"
```

This does not call our assembly function directly. It calls a C function named `caml_dot_neon`, which receives three arguments of type `value` — OCaml's universal boxed representation.

`shim.c` is an adapter sitting at this exact mismatch: it receives OCaml's representation, peels it down to the raw C types the kernel actually wants, calls the kernel with an ordinary C function, and on the way back (after the kernel executes) wraps the result back into a `value`. In our example that wrapping uses `caml_copy_double`, which is an OCaml runtime C API.

I can also propose a more direct approach and explain why that is a very bad idea: OCaml has something called *Bigarray* which is one way to pass data between OCaml and assembly, but it has a few problems. Firstly, the ABI documentation is unstable; we would also have to write a custom Bigarray representation, but in fact it is just a macro so the runtime can change it. Because it has no C interfacing, we cannot use debug tools like `lldb` or `gdb` without OCaml coming in the way. In simpler terms, it is just nicer to keep both things apart for a much cleaner assembling pipeline without worrying about inner representations.

Now that I think of it, this could be a good project idea. More on that later.

Coming back to the pipeline, we just have to compile `shim.c` and the kernel `.S` file to their `.o` representations. The code is in `lib/dune` and has the following:

```
(rule
 (target dot_neon.o)
 (deps kernels/dot.S)
 (action (run %{cc} -c %{deps} -o %{target})))

(rule
 (target shim.o)
 (deps ffi/shim.c)
 (action (run %{cc} -c -I %{ocaml_where} %{deps} -o %{target})))
```

The compilation lines are standard: we specify which file to run via `%{cc}` (the current file is substituted in) and when we hit `dune build`, it runs the rule blocks. Nothing fancy.

One thing to note is the `%{ocaml_where}` variable in the `shim.c` rule. This is specifically for the inclusion of `caml/mlvalues.h` and `caml/bigarray.h`, which live on the OCaml side of things — which is why we need a compilation flag to specify that path.

### Compiling the OCaml code

We also have some OCaml code that just links the compiler to the real C bodies present in `shim.c`. It is nothing really, just some sugar on top of the C code.

### Building the library archive

Seraph as a library is taken by dune, with all the compiled `.cmx` files, C and OCaml C stubs (`shim.o`), and handwritten kernels, and bundled into a static library (`.a` file). This is done by providing the `extra_objects` compilation line. Both `.o` and `.s` files cannot be compiled independently and need their corresponding stub function, so bringing them together was the only way.

### Linking

Linking is one of the main steps in any compiler pipeline. In Seraph, the linking is done between the archived `.a` or `.cmx` files and the OCaml runtime. This is the step where the stub functions in `shim.c` get linked with our handwritten kernels. For the linking process to happen, I created a test suite — which we will go through later — that calls the OCaml function interfaced with the C stubs, which in turn is linked with the kernels.

### Running

When I run `dune test`, the compiler builds the test executable and prints an `OK`/`FAIL` for each test case. More on this later.

### Testing

In order to get the best out of writing the kernels, I have written a testing pipeline into Seraph that compares our source code with other compiled source code. This helps us effectively devise a plan to reach SOTA.

The pipeline is basically this: there will be 4 sources of comparison, each of which will be tested on their wall-clock time — the time taken to finish the kernel execution. The 4 sources are:

1. **Our handwritten kernel**: This will be mostly unoptimized so that we can optimize it by hand and analyze each optimization to create an iterative workflow.
2. **A generated kernel from a C source**: I also write the same function in C, compile it natively in clang, store the `.S` file, and pass it into the test suite.
3. **An OCaml source**: There will also be an OCaml implementation for comparison, but this is just done to compare C code wall clock time with OCaml's so it is not that significant.
4. **Optimized kernel**: This is our goal and the subject of the blog series — we write an unoptimized kernel and slowly bring it up to match the wall clock time of the C source kernel. I will be going through all the optimizations and the resulting change in execution speed.

The test suite is also built using hyperfine, a terminal based benchmarking tool great for creating visual comparison charts which I will also be sharing in the worklog. I will also be comparing the minimum, maximum, mean and relative wall clock speeds, which is integrated into the hyperfine test suite.

---

## Kernels

Without further ado, we will see how we can write kernels in ARM NEON. I will be discussing one kernel in this blog and introduce the others later. As mentioned previously, these are handwritten kernels and most importantly, not yet optimized. We will be going through the optimized kernel in a future post and providing an ablation study between both, alongside the other sources.

The handwritten kernels exist in `lib/kernels/` and the first one we will look at is the SAXPY kernel. SAXPY is just the following operation:

$$y = \alpha \times x + y$$

We update each element of the output vector $y$ with the scaled value from $x$ and a scalar factor $\alpha$.

The kernels chosen will be mostly related to important algorithms in machine learning like dot products and sum reduction. Any other kernels are left for future work.

Before I dive into the kernel, I would like to theoretically lay out a few basic things and explain how we will approach writing it. The SAXPY algorithm has an $x$ vector, a $y$ vector, and a scalar $\alpha$. Now that we have recognized our vectors and scalars, we can discuss how they might be stored in memory and how we access them without getting the offsets wrong.

The first thing we can do with NEON is to broadcast the scalar into 4 "lanes". Lanes are data streams that NEON uses to parallelise computation, so that any computation can be divided into evenly sized lanes and performed in parallel. It is even-sized because the fundamental data layout is even, so work is divided evenly across these lanes.

From the equation, we see that both $x[i]$ and $y[i]$ need to be looped through, so we need some form of a loop to access each element. Before getting into the loop body, let us initialise the loop. We need to specify the loop counter variable (given in the function argument), the maximum limit above which the loop terminates, and the number of iterations remaining for the main loop body.

All of this information can be stored in specific registers, but there is of course a distinction between what *kind* of register they will live in. In my kernel I have used 3 kinds:

- **`w`-prefixed**: These registers are one word long (32 bits). These are general purpose registers and are actually the lower 32 bits of the much bigger 64-bit GPR registers present in ARM. They store loop counters, tail counters, and so on.
- **`s`-prefixed**: These registers are floating point registers (32-bit single precision) used for isolated scalar math. Importantly, `s0` and `v0.s[0]` name the *exact same physical bits* — AArch64's SIMD/FP register file aliases the same underlying register under several names depending on the width you want to view (`b0`/`h0`/`s0`/`d0`/`q0`/`v0` are all "register 0", just sliced to 8/16/32/64/128/128 bits respectively). So an `s`-prefixed register isn't a separate thing from the `v` registers — it's the same register file, viewed through a narrower lens, and we lean on this directly when we reach into `v0.s[0]` below.
- **`v`-prefixed**: These are the SIMD vector registers — 128-bit registers that can be chunked into 32, 64, or 128 bits depending on the instruction. This allows us to store vectors in a specific layout in memory and access them in the same way, setting up our parallel pipeline to stream vectors in parallel for computation. A lot of SIMD instruction support is built around these registers and we will see much more of them later.

In summary: `w`-registers store metadata, `s`-registers store scalar values, and `v`-registers store vector values. Combined in different ways and threaded with instructions, these achieve a fully working kernel.

Now that we know some basics, we can divide our kernel into 3 stages:

- **Prologue**: The setup stage — we load the `w`-registers with metadata needed to run the loop and set up a branch to detect when the loop has been completed.
- **Main loop**: The main vector loop where we perform the operation. We load the vectors, perform the computation, store the vectors, and update the loop counter.
- **Epilogue**: Handles the leftover elements that don't fill a complete vector. Since our main loop processes 4 elements at a time, any remainder (1–3 elements) gets processed here scalar-style.

Now that we have standardised the kernel parts, let's look at the kernel itself.

---

## Prologue

The prologue is the following block:

```asm
FN(saxpy_neon):
    dup     v1.4s, v0.s[0]      // broadcast a into all 4 lanes -> v1
    mov     w3, w2              // w3 = n
    lsr     w4, w3, #2          // w4 = n / 4   (number of 4-wide iterations)
    and     w3, w3, #3          // w3 = n % 4   (tail count)
    cbz     w4, .Ltail
```

Let us apply what we have learnt about NEON to practice. Firstly:

```asm
    dup     v1.4s, v0.s[0]      // broadcast a into all 4 lanes -> v1
```

`dup`, or duplicate, is a vector operation that duplicates the input operand into the output operand. The input operand is `v0.s[0]` — that is, vector register `v0`, accessing its first 32-bit scalar lane `s[0]`. We use `s[0]` here to get our alpha value and **broadcast** it into all 4 chunks of the output vector register `v1.4s`. This allows us to not only create 4 lanes of computation, but also inject alpha into those lanes before even doing the vector operations. The output register `v1.4s` quite literally means "take the first 128-bit register and divide it into four 32-bit chunks." This will be our basis register for further computation.

Now that we have our "computation layout" ready, we will equip it with useful metadata — loop counters and tail counters. The function receives `n` in `w2`:

```asm
    mov     w3, w2   // w3 = n
```

The `mov` instruction moves data from the source register to the destination — here from `w2` (holding `n`) into `w3`. But why move it? Why not just use `w2` directly? The answer is: we need `w3` as a working copy because we are about to overwrite it. As we will see, the `and` instruction one line later writes `n % 4` back into `w3` — so we need to have already read `n` out of it first.

Given the loop counter in `w3`, we calculate the main-loop iteration count and the tail count:

```asm
    lsr     w4, w3, #2          // w4 = n / 4   (number of 4-wide iterations)
    and     w3, w3, #3          // w3 = n % 4   (tail count)
```

`lsr`, or Logical Shift Right, shifts the source bits to the right by the immediate value and stores the result in the destination register. Per the ARM documentation: "LSR provides the unsigned value of a register divided by a variable power of two, inserting zeros into the vacated bit positions" — which means we can do an unsigned division simply by shifting right. I encourage you to try an example yourself to confirm it actually works out.

We read `w3` once to compute the main-loop count into `w4`, and only *then* overwrite `w3` itself with the tail count — since by that point the original value of `n` is no longer needed.

We also use `AND`, which performs a standard bitwise logical AND. In the context of loop unrolling, it acts as an elegant bitmask to calculate the "tail" count — the iterations left over after processing our main chunks. Mathematically this is `n % 4`.

Instead of using a slow division or modulo instruction, we use a bitmask. Because 4 is a power of two, any part of our loop count that divides evenly by 4 exists entirely in the higher-order bits. The remainder lives strictly in the lowest two bits (the 2s and 1s columns). Performing a bitwise AND with 3 (which is `0b0011`, i.e. `4 - 1`) zeroes out all the higher-order bits, perfectly isolating the remainder — calculated in a single clock cycle.

The final piece of the prologue is `cbz` — Compare and Branch if Zero. This is a branch instruction that controls the flow of execution, jumping to a specific part of the code if a specific condition is met. In our case, if `w4` (the main-loop iteration count) is zero, execution jumps to `.Ltail`, skipping the main loop entirely:

```asm
    cbz     w4, .Ltail
```

And that is the prologue. At this point we have our parallel stream ready in `v1.4s`, the main-loop iteration count in `w4`, and the tail/remainder count in `w3` (recall `w3` got reused — it no longer holds `n` at this point). Now we do the computation.

---

## Main Loop

Now we deal with the main part of the kernel: the main loop.

```asm
.Lvec_loop:
    ld1     {v2.4s}, [x0], #16  // load 4 floats from x, advance x0 by 16 bytes
    ld1     {v3.4s}, [x1]       // load 4 floats from y (no advance yet)
    fmla    v3.4s, v1.4s, v2.4s // v3 = v3 + a*v2   (fused multiply-add)
    st1     {v3.4s}, [x1], #16  // store result back to y, advance x1
    subs    w4, w4, #1
    b.ne    .Lvec_loop

.Ltail:
    cbz     w3, .Ldone
```

A lot to unpack here — let us look at it more closely.

In a very logical sense, any computation done by any CPU is: first load the registers that have the input values, do some operation on those registers, then store the updated registers back to the register file or a memory hierarchy. This is exactly what the code snippet above does.

We first load the vectors from memory via the `ld1` instruction:

```asm
    ld1    {v2.4s}, [x0], #16
```

`ld1` takes three operands: the destination, source, and an immediate offset. The destination is a familiar type — a vector operand `v2.4s`, loading into a 4-lane register that matches the layout we set up in the prologue.

What is different from the vector register usage in the prologue is the curly braces `{}` encapsulating the vector register. This is intentional and actually mandatory for every vector load instruction in ARM — the curly braces denote a *register list*, which allows us to load multiple registers all at once rather than one vector register per instruction. This is very performance-critical: load latency varies by microarchitecture (Apple Silicon's load/store unit behaves differently from, say, a Cortex-A55's), but as a rule of thumb it's a handful of cycles rather than one — so for any SIMD core, batching loads is a must to save clock cycles.

Note that I am only loading one vector register here, and that is counterintuitive to what I just said about batching — this is in fact the first ode to further optimization of the kernel, which I will talk about later.

The source operand `[x0], #16` is the memory address held in `x0`, with post-indexing by 16 bytes. Here is why 16: a vector register is 128 bits wide. We are loading 4 single-precision floats (`v2.4s`), each 4 bytes (32 bits), so 4 × 4 = 16 bytes consumed per load. The `#16` advances `x0` by 16 bytes after the load, pointing to the next chunk of `x` for the next iteration.

This type of pointer advancement is called **post-indexed addressing**, one of several ARM addressing modes. The simplest addressing mode is register indirect: `ldr Rd, [Rn]` — get the address from a register. An extension is pre-indexed addressing: `ldr Rd, [Rn, Op2]` — add an offset before the load. Post-indexed (what we use here) performs the load first and then increments the base register. (Note: those example register names use classic 32-bit ARM syntax — our actual AArch64 kernel uses `x0`/`w3`-style names, but the addressing concepts are the same.)

So far so good. We now load the `y` vector into another register:

```asm
    ld1     {v3.4s}, [x1]       // load 4 floats from y (no advance yet)
```

From the comment: `y` is not post-indexed here because we want to advance `x1` only *after* we have computed and stored the result. So we leave it as a plain indirect load into `v3.4s`.

Now we have our two input registers `v2.4s` and `v3.4s` ready. The main computation uses `fmla` — Floating Point Multiply-Add:

```asm
    fmla    v3.4s, v1.4s, v2.4s       // v3 = v3 + a*v2
```

This instruction performs a multiplication and an addition simultaneously. Without `fmla`, the CPU would have to execute a separate `fmul` followed by a `fadd`. Combining them into a single `fmla` unlocks three important architectural advantages.

**Mathematical precision.** Without FMA, the CPU multiplies `a × x`, rounds the result to fit back into a 32-bit float, then adds `y` and rounds again. Accumulating these small rounding errors over millions of iterations leads to significant precision loss. With FMA, the internal hardware calculates the product `a×x` with extended internal precision, adds `y` to that, and only performs one single rounding at the very end.

**Eliminating a dependency stall.** Consider:

```asm
    fmul  s2, s0, s1  // Step 1: Multiply (takes 3–5 clock cycles)
    fadd  s3, s3, s2  // Step 2: Add (must WAIT until Step 1 finishes)
```

The `fadd` cannot begin until `fmul` outputs its result into `s2` — the pipeline stalls. `fmla` bypasses this by passing the multiplication result directly into the internal adder circuit inside the same execution unit, without ever writing to a register first. This completely eliminates the stall.

**Throughput.** Modern processors like Apple Silicon M-series have dedicated hardware pipelines for FMA operations. By using `fmla` we cut the number of arithmetic instructions in the loop exactly in half, letting the CPU spend less energy and time fetching and decoding instructions. Combined with multi-lane processing this heavily increases throughput and reduces what we call "arithmetic intensity" — the ratio of floating-point operations to memory accesses.

So we see that we need to choose our instructions very carefully so they align not just with what the compiler wants but also with what the hardware wants. The compiler wants kernels that are parallel, not sequential, with no latency due to dependencies; the hardware wants us to model our data flow in a way that keeps all its execution units busy.

Continuing: after the `fmla` we store the result back to `y`:

```asm
    st1     {v3.4s}, [x1], #16  // store result back to y, advance x1
```

`st1` is exactly symmetric to `ld1`. We store `v3.4s` (the updated `y` chunk) to the address in `x1`, then advance `x1` by 16 bytes for the next iteration.

Register memory is not persistent — it needs to be stored back to a higher-capacity memory hierarchy. In ARM there exist quite a few custom memory caches, especially for storing multidimensional data like our vectors. For example, the ZA storage is a high-speed 2D vector register memory for storing intermediate accumulation results. But this is a topic for another blog post.

Finally, we update the loop counter and branch:

```asm
    subs    w4, w4, #1
    b.ne    .Lvec_loop
```

`subs` subtracts the immediate from `w4` and stores the result back into `w4`. Here is the general rule worth internalising: in AArch64, the `s` suffix on an instruction (`subs`, `adds`, `ands`, and so on) means "also update the condition flags based on this result." The same instruction *without* the `s` — `sub`, `add`, `and` — does the arithmetic but leaves the flags completely untouched. This matters a lot whenever the very next instruction is a conditional branch like `b.ne`, since that branch reads whatever the flags currently say — and if nothing updated them, it's reading stale, leftover state from some earlier instruction.

This is actually a subtle bug I encountered while writing this kernel: I had accidentally written `sub` instead of `subs`, and my tests failed catastrophically:

```
FAIL saxpy n=2

  Expected: `[|4.70337; 10.5952|]'
  Received: `[|4.70337; -0.157725|]'
```

ARM maintains 4 status flags — collectively the NZCV flags:

- **Zero (Z)**: set if the result was 0.
- **Negative (N)**: set if the result was negative (most significant bit set).
- **Carry (C)**: set on an unsigned overflow — e.g. the carry-out of an addition's top bit, or the absence of a borrow on a subtraction.
- **Arithmetic overflow (V)**: set on a signed overflow — like adding two positive numbers and getting a negative result because the value got too big.

A quick terminology note: this NZCV description covers the general ARM condition-flag concept. The CPSR/SPSR register names specifically belong to classic 32-bit ARM (AArch32). Our kernel is AArch64, where the equivalent flags live in `PSTATE` (with a separate `SPSR_ELx` per exception level rather than a single SPSR) — the flags behave the same way, just under different register names than most online AArch32 references use. For more depth, I recommend the Whirlwind Tour of ARM Assembly, section 23.3.4 on conditionals and branches, keeping in mind that resource describes AArch32.

So when we write just `sub`, the Z flag is never updated as `w4` counts down. The flag remains stuck at whatever value it had before our function started executing. This causes one of three failure modes:

- **Infinite loop (program hangs)**: if the Z flag happens to be 0 when our function starts, `b.ne` always sees it as 0 and loops forever. Your terminal would just freeze or eventually fail with a test timeout.
- **Segmentation fault (SIGSEGV)**: because our loop contains post-indexed loads like `ld1 {v2.4s}, [x0], #16`, the pointer `x0` advances 16 bytes every iteration. In an infinite loop, `x0` quickly marches past the end of our Bigarray and the OS kills the program.
- **Silent fall-through (wrong results)**: this is what happened to us. Our loop executed only once and then stopped, because the Z flag happened to be 1 before the loop started. When we hit `sub` (no flag update), the Z flag stays at 1, and `b.ne` sees it as "not equal to zero? false" — so it doesn't branch back. The remaining iterations never execute and we silently return wrong results.

This is one of the ways ARM teaches us to be careful with instructions and to really understand what is happening under the hood.

And that is the end of the main loop.

---

## Epilogue

The epilogue is the following block:

```asm
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

A different philosophy applies here. The primary goal of the tail loop is to compute the vector chunks that were not aligned with our main-loop width. Suppose we have vectors of length `n = 6`. The kernel divides this into one chunk of 4 (processed by the main loop) and a leftover chunk of 2. What happens if we tried to use a vector instruction to process those last 2 elements? A vector load always reads 16 bytes (4 floats) from memory. To get those last 2 elements, the CPU would be forced to read 2 valid elements plus 2 extra pieces of data that live immediately after our array in RAM. This causes two well-known issues:

- **Segmentation fault**: if our array ends exactly at the edge of a memory page, fetching those extra out-of-bounds bytes will cause the OS to kill our program with a SIGSEGV.
- **Data corruption**: when we write the result back using a vector store (`st1`), we would overwrite those next 2 slots in memory. If those slots contain other variables or OCaml heap data, we silently corrupt program memory.

We already solved half of this in the prologue: the `and w3, w3, #3` bitmask isolated the remainder count into `w3`. That value is now the loop counter for the tail, and the following computation is repeated for each leftover element:

```asm
.Ltail_loop:
    ldr     s2, [x0], #4
    ldr     s3, [x1]
    fmadd   s3, s0, s2, s3      // s3 = a*s2 + s3   (scalar fused multiply-add)
    str     s3, [x1], #4
    subs    w3, w3, #1
    b.ne    .Ltail_loop
```

It looks similar to the main loop, but with obvious changes. We do not need a vector load for the remaining elements — since there are at most 3 of them, we process each one individually using `s`-prefixed scalar registers. Everything else is the same.

The attentive reader might ask: why do we assume the tail loop only processes fewer than 4 elements? What if `n = 23`? That is a very fair question. A vector of length 23 would be divided into 5 full 4-wide chunks (covering 20 elements), with 3 remaining. Those 3 are computed in the tail loop. The `lsr #2` in the prologue computes `23 / 4 = 5` (integer division), and `and #3` gives `23 % 4 = 3`. So the main loop runs 5 times and the tail loop runs 3 times — exactly right.

The final block couples `.Ltail` with `.Ldone` to handle the case where there is no remainder at all:

```asm
.Ltail:
    cbz     w3, .Ldone

.Ldone:
    ret
```

If `w3` is zero, we skip the tail loop entirely and return immediately. Nothing much here — just a clean exit.

And that is the end of the kernel! We have finally looked piece by piece at how we write ARM NEON kernels.

---

## The Full Kernel

For reference, here is the complete `saxpy_neon` kernel:

```asm
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

This right here is what I would call the base kernel.

---

## Results

Now for the numbers. The benchmark is run across three regimes — L1 cache (n=256), L2 cache (n=65536), and DRAM (n=4,000,000) — using Alcotest's in-process timing loop, which gives the cleanest signal since it avoids process-spawn overhead. Each call is repeated enough times that the mean is stable.

| Regime | n | neon (handwritten) | clang -O3 (no restrict) | clang -O3 (restrict) |
|:---|---:|---:|---:|---:|
| L1 | 256 | 19.91 ns/call | 16.87 ns/call | 15.49 ns/call |
| L2 | 65,536 | 5545 ns/call | 5260 ns/call | 5282 ns/call |
| DRAM | 4,000,000 | 378,672 ns/call | 373,570 ns/call | 410,668 ns/call |

Or in throughput terms:

| Regime | neon GFLOP/s | clang -O3 GFLOP/s | neon GB/s | clang -O3 GB/s |
|:---|---:|---:|---:|---:|
| L1 | 25.72 | 30.36 | 154.30 | 182.14 |
| L2 | 23.64 | 24.92 | 141.83 | 149.50 |
| DRAM | 21.13 | 21.41 | 126.76 | 128.49 |

### Reading the numbers

The headline result: our handwritten base kernel is **close to Clang's output, but not quite there** — consistently around 5–18% slower depending on the regime, with the gap widest at L1 and narrowing sharply as we move into L2 and DRAM.

The L1 gap (~18%) is the most telling. At n=256, the whole working set fits in L1 cache, so memory latency is essentially free. What dominates here is purely instruction-level throughput — how efficiently the CPU can dispatch and retire our loop body. Clang's autovectoriser at `-O3` is doing something our base kernel is not, and at this small size that difference is most exposed. The most likely explanation is that Clang is issuing wider or more concurrent load instructions, letting the CPU's load/store unit stay busier per cycle.

At L2 and DRAM, the gap collapses to 1–5%. Once the bottleneck shifts to memory bandwidth — waiting for data to arrive from L2 or main memory — the instruction-level differences matter much less. Both kernels are spending most of their time waiting on the same memory hierarchy, so they converge.

The `clang_restrict` result is also worth noting. At L1 and L2, adding the `restrict` keyword (which tells the compiler the `x` and `y` pointers cannot alias each other) gives a small additional speedup — Clang uses this information to schedule loads and stores more aggressively. By DRAM size, the restrict version is actually *slower* than no-restrict in this run, which is within benchmark noise at that regime. The takeaway is that pointer aliasing hints matter most when the kernel is compute-bound, not when it is memory-bound.

For reference, the OCaml scalar implementation clocks in at ~182 ns/call at L1, roughly 9× slower than our handwritten kernel and consistent across regimes (it never gets fast because it cannot SIMD-vectorise at all and every element goes through OCaml's boxed float representation). This is the floor we are building away from.

### Where the gap comes from

Looking at our base kernel, the inefficiency is not hard to spot once you know what to look for. Our main loop processes 4 elements per iteration using a single `ld1` load for `x` and a single `ld1` load for `y`:

```asm
ld1     {v2.4s}, [x0], #16  // load 4 floats from x
ld1     {v3.4s}, [x1]       // load 4 floats from y
fmla    v3.4s, v1.4s, v2.4s
st1     {v3.4s}, [x1], #16
```

Two loads and one store, one vector at a time. The load unit issues one transaction, waits for it, issues another. Clang's output does not do this — it unrolls the loop and issues multiple loads in flight simultaneously, hiding the latency of each individual load behind the useful work of the others. This is the classic technique of **software pipelining**: by the time one load's result is needed by `fmla`, the next load's data has already arrived.

There is also the question of loop overhead. With only 4 elements per iteration, the `subs w4, w4, #1` and `b.ne` instructions at the bottom of the loop represent a non-trivial fraction of the total instruction count for small `n`. Unrolling to 16 elements per iteration (4 vector registers at a time) would cut this overhead by 4×.

Both of these — load batching and loop unrolling — are exactly the optimisations we will apply in the next post, as well some more hidden optimizations that we will discuss. 

### What comes next

The base kernel is a solid foundation and already beats OCaml's scalar implementation by a factor of 9× at L1. But we are leaving real performance on the table. The gap to Clang at L1 tells us there are straightforward micro-architectural wins available without changing any of the fundamental algorithm structure. In the next blog post, we will write `saxpy_optim` — an optimised version of this same kernel that tackles both the load-batching and unrolling gaps head-on, and we will measure each change individually to build an honest picture of what actually moved the needle and by how much.


Thanks for reading this, if any improvements or corrections please reach out to me through my socials linked in my website. I will be happy to hear from people on how to improve writing worklogs such as these as well as blog writing as a whole. Goodbye!

### References: 

I read quite a lot of ARM Neon documentation, I think almost all of it is present in the ARM documentation for the ([C/C++ instrinsics](https://github.com/ARM-software/acle/releases)) and the official [(ARM NEON ISA documentation](https://developer.arm.com/documentation/den0018/a/NEON-Instruction-Set-Architecture). I also read this goldmine of a resource which is a blog by Daniel Estévez on ([Coding NEON kernels for the Cortex-A53](https://destevez.net/2025/02/coding-neon-kernels-for-the-cortex-a53/)). This blog also implements ARM NEON SAXPY but in a different way and therefore gave me another perspective of approaching optimizations. This not only helped me with writing optimizations, but also helped me write this blog giving me a general format. 
