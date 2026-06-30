# Large Language Models From Scratch

# Chapter 3: Multi-Head Self-Attention — The Mathematical Engine of GPT 🧠

---

## 📖 Overview

In Chapter 2, we turned raw text into continuous vectors. Now we get to build the part of a Transformer that actually makes it *smart* — the mechanism that lets every word in a sentence decide which other words it needs to pay attention to.

This is **self-attention**, and it's the reason modern LLMs can do things that older sequential models couldn't. Instead of reading text word-by-word and forgetting what came before, a Transformer looks at the *entire* context at once and dynamically figures out which pieces are relevant to each other. The mechanism comes from the 2017 paper *Attention Is All You Need* (Vaswani et al.) — and within a few years it had effectively displaced recurrent architectures (RNNs, LSTMs) as the default for almost every large-scale language task. The reason is structural: in an RNN, information from an early token has to survive step-by-step through dozens or hundreds of sequential updates before it can influence a later token, degrading (or vanishing) along the way. Attention skips the relay race entirely — every token can look directly at every other token, no matter how far apart they are. That's what allows it to learn:

- long-range dependencies between words
- contextual and grammatical structure
- semantic meaning
- and even patterns of reasoning

In this chapter, we'll build and walk through an **optimized Multi-Head Self-Attention layer** — using fused QKV projections, causal masking, and GPU-efficient tensor operations. By the end, our embedded tokens will become **context-aware representations**, the fundamental building blocks of GPT.

---

## 🗝️ Core Concepts at a Glance

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
| **Permutation Equivariance** | Without positional info, attention can't tell word order — shuffle the input, and the output just shuffles too |
| **Quadratic Complexity** | Attention's cost grows with the *square* of sequence length — the central scaling bottleneck of Transformers |

---

## 🎯 What You'll Understand After This Chapter

- Why attention replaced recurrent networks
- How GPT performs next-token prediction
- The mathematics behind self-attention (Q, K, V)
- Why attention needs positional information to understand word order
- The difference between self-attention and cross-attention
- Why we scale attention scores — and what goes wrong if we don't
- Why causal masking is non-negotiable
- How multi-head attention helps the model learn richer representations
- The computational complexity of self-attention — and why it's the main scaling bottleneck in Transformers
- GPU optimization tricks used in production LLMs
- Common implementation bugs, and how to avoid them
- How all of this translates cleanly into PyTorch, via a complete reference implementation

---

## 🏗️ The Architecture at a Glance

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

## 📥 What We're Starting With

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

## ⚙️ Stage 1 — Fused QKV Projection

Traditionally, attention computes three separate projections:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

That's three distinct matrix multiplications. Our implementation collapses all three into one:

```python
self.qkv = nn.Linear(d_in, 3 * d_out, bias=False)
```

This computes $[Q, K, V] = XW_{QKV}$ in a single pass, where $W_{QKV} \in \mathbb{R}^{D \times 3D}$. The payoff is meaningful at scale:

- fewer GPU kernel launches
- lower memory traffic
- higher throughput

This is a production-grade optimization — the kind of thing you'll see in real LLM implementations. (We'll tally up exactly how many parameters this single layer adds in the *Parameter Count* section below.)

---

## 🔍 Stage 2 — What Are Queries, Keys, and Values?

Think of Q, K, and V like a search engine inside the model:

```
Q — "What am I searching for?"
K — "What information do I contain?"
V — "What information do I actually provide?"
```

Another way to picture it: a **soft dictionary lookup**. Your Query is the search term. Every token's Key is like an index entry the dictionary checks your search term against. The better a Key matches the Query, the more weight that token's Value gets in the final blend — except instead of returning one exact definition, attention returns a *weighted average* of every entry's definition, weighted by how well each one matched.

Formally:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

For a typical GPT-scale setup — batch size 8, sequence length 1024, embedding size 768, 12 heads — each of Q, K, and V has shape `(8, 1024, 768)`.

---

## 🔀 Why Self-Attention Needs Help With Order

Here's a subtlety that trips up a lot of newcomers. Raw self-attention has no built-in sense of sequence order — mathematically, it's **permutation equivariant**. If you shuffle the input tokens, the output context vectors just get shuffled the same way; nothing about *how* each token's representation gets computed actually depends on its position in the sequence.

Look closely at $S = QK^T$: it only ever compares token *content*, never token *position*. Left on its own, attention would assign "The cat sat" and "sat The cat" the exact same set of context vectors (just reshuffled into different output slots) — it genuinely cannot tell that "The" came first.

This is exactly why positional information has to be baked into the input vectors *before* attention ever runs, rather than left for attention to figure out on its own. Once you see this, the division of labor becomes clear: positional embeddings answer "what came first?"; attention answers "what's related to what?"

---

## ↔️ Self-Attention vs. Cross-Attention

One more distinction worth knowing, since "attention" gets used as an umbrella term: everything in this chapter is **self-attention** — Q, K, and V are all computed from the *same* sequence. The original Transformer (and architectures like T5, or vision-language models that fuse images and text) also use **cross-attention**, where the queries come from one sequence while the keys and values come from a different one — for example, a decoder's Q attending over an encoder's K and V.

The decoder-only, GPT-style model we're building never needs cross-attention. Every layer is self-attention, end to end — one of the reasons GPT's architecture stays comparatively simple compared to the original encoder-decoder Transformer.

---

## 🧩 Stage 3 — Splitting Into Multiple Heads

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

In code, this is a reshape followed by a permutation:

```python
# (B, L, 3*D) -> (B, L, 3, H, d_h) -> (3, B, H, L, d_h)
qkv = qkv.view(B, L, 3, num_heads, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

After splitting, each of Q, K, V has shape `(8, 12, 1024, 64)` — batch, heads, sequence, head dimension.

---

## 📐 Stage 4 — Computing Attention Scores

Now each token compares itself to every other token. The raw attention scores are:

$$S = QK^T$$

If "cat" attends strongly to "animal" but weakly to "tree," you might see:

```
cat → animal: 0.95
cat → dog:    0.84
cat → tree:   0.12
```

The score matrix $S$ has shape `(8, 12, 1024, 1024)` — one score for every token pair, for every head, in every batch.

> 💡 **Why this matters at scale:** computing $S = QK^T$ costs $O(L^2 \cdot d_k)$ per head — $O(L^2 \cdot d_{\text{model}})$ summed across all heads — and storing $S$ costs $O(H \cdot L^2)$ memory. Double the context length, and this single step alone gets **4×** more expensive. This quadratic wall is the biggest scalability bottleneck in Transformers, and it's the reason techniques like FlashAttention and sparse/sliding-window attention exist (more in *Beyond This Chapter*, near the end).

---

## ⚖️ Stage 5 — Scaling (And Why It Matters)

There's a subtle problem with raw dot products: they grow in magnitude with the dimension of the vectors. Specifically:

$$\text{Var}(QK^T) = d_k$$

At high dimensions, these scores can become enormous. When you pass large values through a softmax, the output saturates — almost all the probability mass collapses onto a single token, gradients vanish, and training stalls.

Let's actually derive this instead of just asserting it. Assume each component of $Q$ and $K$ is independently sampled with mean $0$ and variance $1$ — a reasonable assumption right after Xavier/He-style initialization or a LayerNorm. For a single attention score:

$$S_{ij} = \sum_{k=1}^{d_k} q_{ik}\, k_{jk}$$

Since $q_{ik}$ and $k_{jk}$ are independent with mean 0, each product term $q_{ik}k_{jk}$ also has mean 0 and variance 1. Summing $d_k$ independent terms adds their variances:

$$\text{Var}(S_{ij}) = \sum_{k=1}^{d_k} \text{Var}(q_{ik}k_{jk}) = d_k$$

So the standard deviation of raw attention scores grows as $\sqrt{d_k}$ — at $d_k = 64$, scores can easily swing into the double digits. Dividing by $\sqrt{d_k}$ exactly cancels this growth, pulling the variance back down to 1 regardless of head dimension:

$$\text{Var}\!\left(\frac{S_{ij}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

The fix, restated:

$$S_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}}$$

**Illustrative example** — what this looks like in practice:

```
Unscaled (d_k = 64):  scores ≈ [38.0, 14.0, 2.0]
                       softmax ≈ [1.00, 0.00, 0.00]   ← saturated, ~zero gradient

Scaled (÷ √64 = 8):   scores ≈ [4.75, 1.75, 0.25]
                       softmax ≈ [0.94, 0.05, 0.01]   ← still peaked, but differentiable
```

This keeps scores well-behaved, gradients healthy, and training stable.

---

## 🚧 Stage 6 — Causal Masking

GPT is trained to predict the next token given all previous tokens:

$$P(x_t \mid x_{<t})$$

That means token $t$ must not be allowed to look at tokens $t+1$, $t+2$, and so on. If it could, training would be trivially easy — the model would just copy the answer — and the learned representations would be useless at inference time.

We enforce this with a causal mask:

$$M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$$

In code, the mask itself is typically built once with `torch.triu` and reused on every forward pass:

```python
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
```

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

## 🌊 Stage 7 — Softmax

We convert the masked scores into a probability distribution using softmax:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

The output is always positive, sums to 1, and is differentiable — which makes it ideal for backpropagation. For example:

```
Scores:   [2, 1, 0]
Softmax:  [0.66, 0.24, 0.10]
```

One more detail real implementations always include: computing $e^{x_i}$ directly can overflow for large $x_i$. In practice, frameworks like PyTorch's `F.softmax` subtract the row-wise maximum first:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

This is mathematically identical — the $\max(x)$ term cancels out in the ratio — but it keeps every exponent $\le 0$, so nothing overflows. You don't need to implement this yourself (PyTorch handles it internally), but it's worth knowing it's happening under the hood.

---

## 🧮 Stage 8 — Weighted Value Aggregation

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

## 🔗 Stage 9 — Combining the Heads

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

A caveat worth flagging: the table above is a simplification for intuition, not a design spec. In practice, what each head actually learns is discovered empirically, not engineered in — researchers use attention-visualization tools (e.g., BertViz) or probing studies to reverse-engineer what different heads seem to specialize in after training, and the real picture is messier than "Head 1 = grammar." Some heads even turn out to be redundant and can be pruned with little accuracy loss. The architecture *enables* specialization; it doesn't *guarantee* a clean division of labor.

---

## 📊 Tensor Shape Walkthrough

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

## ⚡ GPU Optimization Notes

Beyond the fused QKV projection, our implementation uses a few other production-grade techniques worth knowing:

**`register_buffer()`** — The causal mask is registered as a buffer rather than a parameter. This means it moves automatically to the GPU with the model, gets saved in checkpoints, and is excluded from the optimizer. It's a small detail that avoids a class of subtle bugs.

**`.contiguous()`** — After transposing or permuting tensors, PyTorch may leave the underlying memory non-contiguous. Calling `.contiguous()` ensures the memory layout is correct before reshaping, which is necessary for efficient kernel execution. (You'll see exactly where this bites in the reference implementation below — right after the heads get merged back together.)

**Fused attention kernels** — Production systems (PyTorch's built-in `scaled_dot_product_attention`, FlashAttention) go a step further and fuse the *entire* matmul → mask → softmax → matmul pipeline into a single GPU kernel, so the full $L \times L$ score matrix is never fully written out to slow GPU memory. We're keeping the stages separate here for clarity — that's the whole point of building this from scratch — but it's worth knowing that production code collapses them for speed.

---

## 🧱 Full Reference Implementation

Here's every stage above, assembled into a single working `MultiHeadAttention` module:

```python
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Stage 1: fused QKV projection
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        # Built once, reused every forward pass; register_buffer moves
        # it with the model (e.g. .to(device)) without treating it as
        # a trainable parameter.
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Stage 1: project to Q, K, V in a single matmul
        qkv = self.qkv(x)  # (b, num_tokens, 3 * d_out)

        # Stage 3: split into heads
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Stage 4: raw attention scores
        attn_scores = q @ k.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)

        # Stage 6: causal mask, applied BEFORE softmax
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # Stage 5 + 7: scale, then softmax
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # Stage 8: weighted aggregation of values
        context = (attn_weights @ v).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)

        # Stage 9: merge heads back together + output projection
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)
```

Every comment above maps back to a stage from this chapter — if anything in this class looks unfamiliar, that's a good signal for which section to revisit.

---

## ⚠️ Common Pitfalls

A few mistakes show up constantly when implementing this from scratch — worth knowing about before you hit them:

1. **Forgetting `.contiguous()` before `.view()`.** After a `.transpose()` or `.permute()`, the tensor's memory layout is no longer contiguous. Calling `.view()` directly on it raises a `RuntimeError`; `.contiguous()` (or `.reshape()`, which handles this automatically) fixes it. This is exactly the situation in Stage 9, right after merging the heads back together.
2. **`d_model` not divisible by `num_heads`.** If it isn't, `d_out // num_heads` silently produces a head dimension that doesn't actually reconstruct `d_out`, or the reshape just fails outright. Always assert this up front (see the reference implementation above).
3. **Mask dtype/device mismatches.** A mask created with `torch.ones(...)` on the CPU won't broadcast correctly once your model and inputs move to GPU. This is precisely why `register_buffer` matters — it moves the mask automatically whenever you call `.to(device)` on the model.
4. **Masking *after* softmax instead of before.** Zeroing out probabilities after the fact doesn't fully prevent information leakage and breaks the math (the remaining weights no longer sum to 1). Always apply the mask with $-\infty$ *before* softmax, never after.
5. **Using a "soft" mask value instead of true `-inf`.** Some implementations fill masked positions with a large negative number like `-1e4` instead of `-inf`, often to dodge dtype issues. In `float16`, this can be too small to fully zero out after softmax — leaking a tiny but nonzero amount of attention to future tokens. If you're not using true `-inf`, double-check the math for your specific dtype.

---

## 🔢 Parameter Count

Our GPT implementation has **163,009,536 parameters**, corresponding to this configuration:

| Setting | Value |
|---|---|
| Layers | 12 |
| Heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 |
| Vocabulary size | 50,257 |

This is GPT-2 Small architecture — with one difference. The official GPT-2 has 124 million parameters because it ties the token embedding and LM head weights. Our implementation keeps them separate, which adds about 38.6 million parameters. Both approaches are valid; weight tying is a regularization technique that also reduces memory.

For context, here's where the attention mechanism's share of those parameters comes from. Per layer, using the fused, bias-free QKV from Stage 1 and an equally bias-free output projection:

- QKV projection: $D \times 3D = 768 \times 2304 = 1{,}769{,}472$ parameters
- Output projection: $D \times D = 768 \times 768 = 589{,}824$ parameters
- **Per-layer attention total:** $\approx 2.36$M parameters

Across all 12 layers, that's roughly **28.3M parameters** — about 17% of the model's 163M total. The rest comes from the embedding tables, the (untied) LM head, and the feed-forward layers sitting between each attention block, which we'll get to in a later chapter.

---

## 🔭 Beyond This Chapter

The Multi-Head Attention layer we just built is the version you'd find in a training-focused implementation. Production inference systems push it further — worth knowing about even if we don't implement them here:

- **KV Caching** — During autoregressive generation, the Keys and Values for already-seen tokens never change. Instead of recomputing them at every new step, inference engines cache K and V and only compute them for the newest token — turning an $O(L^2)$ per-step cost into $O(L)$.
- **Multi-Query & Grouped-Query Attention (MQA / GQA)** — Share a single K/V projection across all (or groups of) heads instead of giving every head its own. This shrinks the KV cache dramatically and is what powers fast inference in models like LLaMA 2/3 and Mistral.
- **FlashAttention** — Fuses the matmul → mask → softmax → matmul pipeline into a single GPU kernel, so the full $L \times L$ score matrix is never fully materialized in slow GPU memory. Same math, far less memory traffic — a big reason today's models can handle context windows of 100K+ tokens.
- **RoPE / ALiBi** — Alternatives to absolute positional embeddings that bake position information directly into how Q and K are computed, rather than adding it to the input. These tend to generalize better to sequence lengths longer than what the model was trained on.

We'll return to some of these in later chapters once the basic mechanism is second nature.

---

## 🔄 The Full Learning Pipeline

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

## 🧪 Try It Yourself

A few exercises to make sure the mechanics actually stuck:

1. **Trace the shapes by hand.** For $B=2$, $L=5$, $D=8$, $H=2$, work out the tensor shape after every stage in the pipeline — fused QKV projection, reshape, permutation, scores, masking, softmax, aggregation, and final output.
2. **Saturation, quantified.** Using the variance derivation from Stage 5, compare the standard deviation of raw attention scores at $d_k = 64$ versus $d_k = 512$. Which head dimension saturates softmax faster if left unscaled?
3. **Build a sliding-window mask.** Modify the causal mask so each token can only attend to the previous $W$ tokens (instead of *all* previous tokens). This is the core idea behind the sparse/local attention variants used in long-context models.

---

## ✅ Summary

In this chapter, we worked through every stage of Multi-Head Self-Attention — from the mathematical intuition behind queries, keys, and values, to the engineering decisions that make it run efficiently on GPUs. The key ideas:

- Self-attention lets every token dynamically attend to every other token in context.
- The Q, K, V decomposition separates "what I'm looking for" from "what I contain" from "what I provide."
- Without positional embeddings, attention is permutation equivariant — it has no inherent sense of word order.
- Scaling by $\sqrt{d_k}$ is essential for stable training; we derived exactly why $\text{Var}(QK^T) = d_k$.
- Causal masking enforces the autoregressive constraint that GPT depends on — and it must be applied *before* softmax, not after.
- Multiple heads allow the model to simultaneously learn different types of structure.
- Fused projections, `register_buffer`, and `contiguous()` are the kinds of optimizations that separate a textbook implementation from a production one.
- Self-attention's $O(L^2)$ cost is the central scaling bottleneck of Transformers — and the motivation behind FlashAttention, KV caching, and the other techniques previewed above.

This is the mathematical and computational core of the Transformer. Everything we build from here — feed-forward layers, layer normalization, positional encodings, training loops — is built around this mechanism.