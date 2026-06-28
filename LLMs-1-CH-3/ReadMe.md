# Large Language Models From Scratch

# Chapter 3: Multi-Head Self-Attention — The Mathematical Engine of GPT

---

## Overview

In Chapter 2, we turned raw text into continuous vectors. Now we get to build the part of a Transformer that actually makes it *smart* — the mechanism that lets every word in a sentence decide which other words it needs to pay attention to.

This is **self-attention**, and it's the reason modern LLMs can do things that older sequential models couldn't. Instead of reading text word-by-word and forgetting what came before, a Transformer looks at the *entire* context at once and dynamically figures out which pieces are relevant to each other. That's what allows it to learn:

- long-range dependencies between words
- contextual and grammatical structure
- semantic meaning
- and even patterns of reasoning

In this chapter, we'll build and walk through an **optimized Multi-Head Self-Attention layer** — using fused QKV projections, causal masking, and GPU-efficient tensor operations. By the end, our embedded tokens will become **context-aware representations**, the fundamental building blocks of GPT.

---

## Core Concepts at a Glance

Before diving in, here's a quick reference for the key terms we'll be working with:

| Term | What it actually means |
|---|---|
| **Self-Attention** | Every word decides which other words in the sentence matter for understanding itself |
| **Query (Q)** | What a token is *looking for* |
| **Key (K)** | What a token *contains* |
| **Value (V)** | The actual information a token *provides* |
| **Scaled Dot-Product Attention** | How strongly one token should attend to another |
| **Multi-Head Attention** | Running several independent attention mechanisms in parallel |
| **Causal Masking** | Hiding future tokens so the model can't cheat during training |
| **Fused QKV Projection** | Computing Q, K, and V in a single matrix multiply for efficiency |
| **Attention Weights** | Probability scores that say how important each token is |
| **Context Vector** | The final, enriched representation produced by attention |

---

## What You'll Understand After This Chapter

- Why attention replaced recurrent networks
- How GPT performs next-token prediction
- The mathematics behind self-attention (Q, K, V)
- Why we scale attention scores — and what goes wrong if we don't
- Why causal masking is non-negotiable
- How multi-head attention helps the model learn richer representations
- GPU optimization tricks used in production LLMs
- How all of this translates cleanly into PyTorch

---

## The Architecture at a Glance

The full Multi-Head Attention pipeline runs through seven stages:

```
Input Embeddings
        ↓
Fused QKV Projection
        ↓
Split into Heads
        ↓
Attention Score Computation
        ↓
Causal Masking
        ↓
Softmax Probability Distribution
        ↓
Weighted Value Aggregation
        ↓
Head Concatenation
        ↓
Output Projection
```

We'll work through each stage in turn.

---

## What We're Starting With

Attention takes embedded tokens as input. For the sentence `"The cat sat"`, those embeddings look like:

```
[
  [0.23, 0.44, ...],   ← "The"
  [0.91, 0.12, ...],   ← "cat"
  [0.66, 0.84, ...]    ← "sat"
]
```

Formally, the input tensor **X** has shape:

$$X \in \mathbb{R}^{B \times L \times D}$$

where **B** is the batch size, **L** is the context length, and **D** is the embedding dimension.

---

## Stage 1 — Fused QKV Projection

Traditionally, attention computes three separate projections:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

That's three distinct matrix multiplications. Our implementation collapses all three into one:

```python
self.qkv = nn.Linear(d_in, 3 * d_out, bias=False)
```

This computes $[Q, K, V] = XW_{QKV}$ in a single pass. The payoff is meaningful at scale:

- fewer GPU kernel launches
- lower memory traffic
- higher throughput

This is a production-grade optimization — the kind of thing you'll see in real LLM implementations.

---

## Stage 2 — What Are Queries, Keys, and Values?

Think of Q, K, and V like a search engine inside the model:

```
Q — "What am I searching for?"
K — "What information do I contain?"
V — "What information do I actually provide?"
```

Formally:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

For a typical GPT-scale setup — batch size 8, sequence length 1024, embedding size 768, 12 heads — each of Q, K, and V has shape `(8, 1024, 768)`.

---

## Stage 3 — Splitting Into Multiple Heads

Rather than running one big attention mechanism, we split the embedding dimension across many smaller, parallel attention heads:

```
768 dimensions
      ↓
12 heads
      ↓
64 dimensions per head
```

In general:

$$d_h = \frac{d_{\text{model}}}{h}$$

where $d_h$ is the per-head dimension, $d_{\text{model}}$ is the full embedding size, and $h$ is the number of heads.

After splitting, each of Q, K, V has shape `(8, 12, 1024, 64)` — batch, heads, sequence, head dimension.

---

## Stage 4 — Computing Attention Scores

Now each token compares itself to every other token. The raw attention scores are:

$$S = QK^T$$

If "cat" attends strongly to "animal" but weakly to "tree," you might see:

```
cat → animal: 0.95
cat → dog:    0.84
cat → tree:   0.12
```

The score matrix $S$ has shape `(8, 12, 1024, 1024)` — one score for every token pair, for every head, in every batch.

---

## Stage 5 — Scaling (And Why It Matters)

There's a subtle problem with raw dot products: they grow in magnitude with the dimension of the vectors. Specifically:

$$\text{Var}(QK^T) = d_k$$

At high dimensions, these scores can become enormous. When you pass large values through a softmax, the output saturates — almost all the probability mass collapses onto a single token, gradients vanish, and training stalls.

The fix is simple: divide by the square root of the head dimension:

$$S_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}}$$

This keeps scores well-behaved, gradients healthy, and training stable.

---

## Stage 6 — Causal Masking

GPT is trained to predict the next token given all previous tokens:

$$P(x_t \mid x_{<t})$$

That means token $t$ must not be allowed to look at tokens $t+1$, $t+2$, and so on. If it could, training would be trivially easy — the model would just copy the answer — and the learned representations would be useless at inference time.

We enforce this with a causal mask:

$$M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$$

In code:

```python
attn_scores.masked_fill(mask, -torch.inf)
```

The $-\infty$ values become 0 after softmax, so future tokens are completely invisible. The effect is exactly what we want:

```
Token 1 sees: [1]
Token 2 sees: [1, 2]
Token 3 sees: [1, 2, 3]
```

---

## Stage 7 — Softmax

We convert the masked scores into a probability distribution using softmax:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

The output is always positive, sums to 1, and is differentiable — which makes it ideal for backpropagation. For example:

```
Scores:   [2, 1, 0]
Softmax:  [0.66, 0.24, 0.10]
```

---

## Stage 8 — Weighted Value Aggregation

This is where everything comes together. The context vector for each token is:

$$C = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This is arguably **the most important equation in the Transformer architecture**. The attention weights act as a routing mechanism: they decide how much of each token's value to pull in, and the result is a new, context-aware representation for every position.

Intuitively:
```
attention weights
      ↓
decide importance
      ↓
weighted averaging of values
      ↓
new contextual meaning
```

---

## Stage 9 — Combining the Heads

Each attention head runs independently and learns to specialize. A rough picture of what different heads pick up:

| Head | What it might learn |
|---|---|
| Head 1 | Grammar |
| Head 2 | Syntax |
| Head 3 | Long-range dependencies |
| Head 4 | Coreference resolution |
| Head 5 | Semantic similarity |

After all heads compute their context vectors, we concatenate them and apply a final output projection $W_O$:

$$\text{MHA} = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)\, W_O$$

---

## Tensor Shape Walkthrough

It helps to trace exactly how the tensor dimensions evolve:

| Stage | Shape |
|---|---|
| Input | `(8, 1024, 768)` |
| After fused QKV projection | `(8, 1024, 2304)` |
| After reshape | `(8, 1024, 3, 12, 64)` |
| After permutation | `(3, 8, 12, 1024, 64)` |
| Q, K, V (each) | `(8, 12, 1024, 64)` |
| Attention scores | `(8, 12, 1024, 1024)` |
| Context vectors | `(8, 12, 1024, 64)` |
| Final output | `(8, 1024, 768)` |

The input and output shapes match — a clean, composable module.

---

## GPU Optimization Notes

Beyond the fused QKV projection, our implementation uses a few other production-grade techniques worth knowing:

**`register_buffer()`** — The causal mask is registered as a buffer rather than a parameter. This means it moves automatically to the GPU with the model, gets saved in checkpoints, and is excluded from the optimizer. It's a small detail that avoids a class of subtle bugs.

**`.contiguous()`** — After transposing or permuting tensors, PyTorch may leave the underlying memory non-contiguous. Calling `.contiguous()` ensures the memory layout is correct before reshaping, which is necessary for efficient kernel execution.

---

## Parameter Count

Our GPT implementation has **163,009,536 parameters**, corresponding to this configuration:

| Setting | Value |
|---|---|
| Layers | 12 |
| Heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 |
| Vocabulary size | 50,257 |

This is GPT-2 Small architecture — with one difference. The official GPT-2 has 124 million parameters because it ties the token embedding and LM head weights. Our implementation keeps them separate, which adds about 38.6 million parameters. Both approaches are valid; weight tying is a regularization technique that also reduces memory.

---

## The Full Learning Pipeline

To put everything in context, here's the complete journey from raw text to context vectors:

```
Raw Text
    ↓ Tokenization
Token IDs
    ↓ Sliding Window Dataset
Input / Target Pairs
    ↓ Embedding Lookup
Vectors
    ↓ QKV Projection
Queries, Keys, Values
    ↓ Dot Products + Scaling
Raw Attention Scores
    ↓ Causal Masking
Masked Scores
    ↓ Softmax
Attention Weights
    ↓ Weighted Aggregation
Context Vectors (per head)
    ↓ Concatenation + Output Projection
Transformer Representations
```

---

## Summary

In this chapter, we worked through every stage of Multi-Head Self-Attention — from the mathematical intuition behind queries, keys, and values, to the engineering decisions that make it run efficiently on GPUs. The key ideas:

- Self-attention lets every token dynamically attend to every other token in context.
- The Q, K, V decomposition separates "what I'm looking for" from "what I contain" from "what I provide."
- Scaling by $\sqrt{d_k}$ is essential for stable training.
- Causal masking enforces the autoregressive constraint that GPT depends on.
- Multiple heads allow the model to simultaneously learn different types of structure.
- Fused projections, `register_buffer`, and `contiguous()` are the kinds of optimizations that separate a textbook implementation from a production one.

This is the mathematical and computational core of the Transformer. Everything we build from here — feed-forward layers, layer normalization, positional encodings, training loops — is built around this mechanism.