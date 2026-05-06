
# LLaDA-from-Scratch Manifesto

> **Building a Language Diffusion Model from the Ground Up — No Shortcuts, No Abstractions.**

---

## 1. The Vision

Autoregressive language models have dominated natural language generation for nearly a decade. Transformers that predict the next token one at a time are fast, familiar, and effective. But they are not the only way.

**LLaDA** (Large Language Diffusion Models) flips the paradigm: instead of generating text token-by-token in order, it treats language as a distribution and learns to denoise masked sequences into coherent text — a process inspired by how diffusion models revolutionized image generation.

This project exists to **implement LLaDA from scratch**, stripping away every pre-built abstraction, understanding every matrix multiplication, and proving that diffusion-based language models can be built, trained, and understood from first principles.

---

## 2. What We Are Building

A **complete, self-contained training pipeline** for LLaDA — a diffusion model for natural language. This is not a fine-tuning hack, not a wrapper around existing code, and not a tutorial copy-paste. It is a ground-up reconstruction of:

### The Model Architecture
- **LLaDAModel** — the core transformer backbone with rotary positional embeddings (RoPE), grouped-query attention (GQA), SwiGLU activations, and configurable layer normalization (LayerNorm, RMSNorm, GemmaRMSNorm).
- **LLaDABlock / LLaDASequentialBlock / LLaDALlamaBlock** — multiple transformer block variants for flexibility.
- **LLaDAModelLM** — a Hugging Face `PreTrainedModel` wrapper enabling seamless integration with the `transformers` ecosystem.

### The Training Loop
- **Diffusion masking**: Each input sequence is randomly masked according to a sampled timestep `t ~ Uniform(0,1)`, with cross-entropy loss computed only on masked tokens and scaled by `1/t`.
- **Gradient checkpointing** strategies (whole-layer, one-in-two, fine-grained, and more) for memory-efficient training on large contexts.
- **Mixed precision** training with fp16/bfloat16 support.
- **AdamW optimizer** with linear warmup and decay scheduling.

### The Data Pipeline
- **FineWeb** (`sample-10BT`) as the training corpus — high-quality, deduplicated web text.
- **Streaming dataset loader** that shards millions of `.pt` chunks and loads them on-demand into CPU RAM, batching to GPU only when needed.
- **Noisy masking** with epsilon perturbation for stable diffusion training.

### The Inference Pipeline
- **Diffusion-based generation**: Start from a fully masked sequence and progressively unmask tokens through reverse diffusion steps.
- **Hugging Face compatibility**: Save and upload models that work with `transformers` auto-loading, pipelines, and the Hugging Face Hub.

---

## 3. Our Principles

### 1. From Scratch, Not From Shortcuts
We do not copy-paste from existing repos and call it a day. Every component — from weight initialization schemes to attention masking — is understood, implemented, and owned by this project.

### 2. Transparency Over Convenience
The code is structured to be readable and educational. Complex operations are broken into functions with clear docstrings. There are no hidden abstractions that obscure what is happening under the hood.

### 3. Minimal Dependencies
We use only what is necessary:
- **PyTorch** for tensor computation and autograd
- **Transformers** for tokenizer compatibility and Hugging Face integration
- **Datasets** for data loading utilities
- **PEFT / Accelerate** for potential future extension work
- **tilelang** for hardware-aware optimization research

### 4. Reproducibility
Every training run should be reproducible. Configuration is centralized in `configs_llada.py`. Training scripts are self-contained. Checkpoints include both model weights and tokenizer state.

### 5. Extensibility
The architecture is designed to support multiple block types, normalization strategies, activation functions, and initialization schemes. New research ideas can be plugged in without rewriting the entire pipeline.

---

## 4. Architecture Highlights

| Component | Implementation |
|---|---|
| **Attention** | Multi-head, grouped-query (GQA), FlashAttention-ready, causal and non-causal modes |
| **Positional Encoding** | Rotary Embeddings (RoPE) with full-precision computation |
| **Normalization** | LayerNorm, RMSNorm, GemmaRMSNorm — all with low-precision autocast support |
| **Activation** | GELU, ReLU, SiLU, SwiGLU |
| **Weight Initialization** | Mitchell, Normal, Kaiming Normal, Fan-in, Full Megatron strategies |
| **Checkpointing** | 10+ strategies from whole-layer to fine-grained per-operation |
| **HF Integration** | PreTrainedModel wrapper with `auto_map` registration |

---

## 5. Training Data

| Dataset | Details |
|---|---|
| **Source** | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample-10BT` |
| **Processed** | [Fredtt3/LLaDA-Sample-10BT](https://huggingface.co/datasets/Fredtt3/LLaDA-Sample-10BT) |
| **Chunks** | ~2,520,000 samples, up to 4,096 tokens each (1% random-length) |
| **Masking** | Noisy masking with ε = 1×10⁻³ |
| **Storage** | 252 sharded `.pt` files, ~166 GB total |

---

## 6. Project Structure
