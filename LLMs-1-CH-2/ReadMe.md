

---

# 📚 Large Language Models From Scratch

## Chapter 2: The Data Preparation Pipeline (Working with Text Data)

### 🚀 Overview

Large Language Models (LLMs) cannot read or understand English; they can only process continuous mathematical matrices. This module contains the complete "digestive system" or front-end pipeline required to take raw, unstructured human language and engineer it into the exact 3D mathematical tensors required to train a neural network.

By the end of this pipeline, raw text is transformed into context-aware, position-aware continuous vectors ready for the Self-Attention mechanism.

---

### 🧠 Core Concepts: 

| Technical Term | The Simple Explanation |
| --- | --- |
| **Vocabulary Space ($V$)** | A giant dictionary that maps unique text chunks to specific ID numbers. |
| **Byte-Pair Encoding (BPE)** | A smart algorithm that chops words into pieces based on frequency (e.g., "unbelievable" $\rightarrow$ "un" + "believ" + "able") instead of using whole words or single letters. |
| **Autoregressive Sampling** | Teaching the AI by hiding the answer. We give it a sequence of words and ask it to predict the very next word. |
| **Sliding Window (Context & Stride)** | A moving magnifying glass over our text. "Context" is how wide the glass is (how many words the AI sees). "Stride" is how far we slide the glass to get the next sample. |
| **Token Embeddings** | Upgrading a basic ID number into a deep, multi-dimensional "personality profile" so the AI understands that "King" and "Queen" are mathematically related. |
| **Positional Embeddings** | Giving every word a "ticket number" in line. Without this, the AI wouldn't know the difference between "the dog bit the man" and "the man bit the dog." |

---

### ⚙️ Pipeline Architecture

This repository breaks down the data pipeline into four distinct phases:

#### 1. Tokenization (`tiktoken`)

We utilize OpenAI's BPE tokenizer to map a string of text into a sequence of discrete integers. We also introduce special context tokens to handle edge cases:

* `<|endoftext|>`: Tells the model where an independent document ends.
* `<|unk|>`: A safety net for completely unknown characters.

#### 2. PyTorch DataLoaders (The Sliding Window)

We unroll our 1D sequence of token IDs into parallel 2D tensors to create training batches.

* **Input Tensor ($\mathbf{X}$):** The context the model sees.
* **Target Tensor ($\mathbf{Y}$):** The exact same context, shifted right by exactly $+1$ time-step.

#### 3. Linear Projection (Token Embeddings)

We project discrete token IDs into a continuous space $\mathbb{R}^d$. As proven in our bonus notebooks, PyTorch's `nn.Embedding` is mathematically equivalent to generating a massive sparse One-Hot Encoded matrix and multiplying it by a `nn.Linear` layer. `nn.Embedding` acts as a highly optimized "look-up table" to save computational power.

#### 4. Injecting Time (Positional Embeddings)

Because self-attention operations process tokens simultaneously (permutation invariant), we create a secondary `nn.Embedding` layer. Instead of feeding it token IDs, we feed it the absolute sequence indices ($0, 1, 2, \dots, L-1$).

---

### 🧮 The Final Mathematical Output

The culmination of this pipeline is a single element-wise addition operation. We add the *meaning* of the token (Token Embedding) to the *location* of the token (Positional Embedding).

For any given batch $b$ and sequence position $j$, the final tensor $\mathbf{Z}$ is constructed as:

$$\mathbf{Z}_{b, j, :} = \mathbf{W}_E[\mathbf{X}_{b, j}, :] + \mathbf{W}_P[j, :]$$

This results in our final 3D tensor, shaped precisely for the transformer blocks:
**Shape:** `(Batch Size, Context Length, Embedding Dimension)`

---

### 📁 Repository Files

* `ch02.ipynb`: The main chapter code. Contains the full, step-by-step pipeline from raw text to the final 3D tensor.
* `dataloader.ipynb`: A concise, minimal version of the main data loading pipeline used for rapid implementation.
* `dataloader-intuition.ipynb`: A visual, intuitive breakdown of the sliding window using a sequence of simple numbers (0 to 1000) instead of text.
* `embeddings-and-linear-layers.ipynb`: A mathematical proof demonstrating that `nn.Embedding(idx)` is identical to `nn.Linear(one_hot(idx))`.
* `exercise-solutions.ipynb`: Solutions for creating custom strides, dynamic context lengths, and decoding unknown tokens.

### 🛠️ Dependencies

* `torch` (PyTorch)
* `tiktoken` (OpenAI Tokenizer)
* `requests` (For downloading raw text data)