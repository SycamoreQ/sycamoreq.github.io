export interface Project {
  name: string;
  lang: string;
  url: string;
  description: string;
  tags: string[];
}

// ─── Add your projects here ────────────────────────────────────────────────
export const projects: Project[] = [
  {
    name: 'Tesserae',
    lang: 'OCaml, CUDA',
    url: 'https://github.com/SycamoreQ/tesserae',
    description:
      'A native OCaml DSL for writing high-performance GPU kernels (GEMM, convolutions, etc.) targeting NVIDIA CUDA via NVRTC. Tesserae lets you express kernels in a typed, functional style, then compiles them to PTX and launches them on the device without leaving the OCaml runtime.',
    tags: ['systems', 'ocaml', 'compilers' , 'gpu'],
  },
  {
    name: 'axiom',
    lang: 'Rust, C++',
    url: 'https://github.com/SycamoreQ/axiom',
    description:
      'A high-performance large language model inference engine built in Rust with custom CUDA kernels. Axiom implements the LLaMA and DeepSeek model families with support for grouped-query attention, Mixture-of-Experts with speculative pre-gating, and LoRA adapter hot-swapping. Also consists of a ForkKV disaggregated KV cache that separates shared base prefixes from per-agent residual caches',
    tags: ['systems' , 'rust' , 'gpu'],
  },
  {
    name: 'cellc',
    lang: 'C',
    url: 'https://github.com/SycamoreQ/cellc',
    description:
      'cellc is a container runtime from scratch in C ',
    tags: ['linux', 'C'],
  },
];
