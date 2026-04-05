# Self Attention from Scratch

A from-scratch implementation of the self-attention mechanism — the core of every modern LLM — built in raw PyTorch with no transformers library.

## What this implements

This is a complete forward pass of one transformer layer on the sentence **"the cat sat on the mat"**.

### Pipeline

```
raw text
→ BPE tokenization        (custom bpe.py)
→ token IDs
→ embedding lookup        (nn.Embedding, vocab=275, dim=4)
→ positional encoding     (sinusoidal, added to embeddings)
→ Q, K, V projection      (random weight matrices 4×4)
→ attention score         (softmax(Q·Kᵀ / √4) · V)
→ residual connection     (attention output + input embeddings)
→ RMSNorm
→ feedforward layer       (Linear 4→16 → ReLU → Linear 16→4)
→ residual connection     (ff output + normed embeddings)
→ repeat × 32 layers
→ final enriched vectors  (shape: 22, 4)
```

## What it does NOT implement

- Backpropagation with weight updates (no optimizer step)
- Positional encoding via RoPE (uses classic sinusoidal)
- Multi-head attention (single head only)
- Token prediction / vocabulary projection
- Training on real data

This is a pure forward pass — the goal is understanding the mechanics, not training a model.

## Files

```
Self Attention/
├── attention.ipynb   — main implementation
├── bpe.py            — custom BPE tokenizer (Level 1 project)
└── README.md         — this file
```

## How to run

```bash
pip install torch numpy
```

Open `attention.ipynb` and run all cells top to bottom.

## Key implementation details

**Embeddings** — `nn.Embedding(275, 4)` maps each BPE token ID to a 4-dimensional vector.

**Positional encoding** — sinusoidal formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
Added directly to embeddings before attention.

**Q, K, V** — three separate `(4,4)` weight matrices. Each token's embedding is multiplied through all three to produce query, key and value vectors.

**Attention score** — `softmax(Q · Kᵀ / √4, dim=-1) · V`. Softmax applied row-wise so each token's weights sum to 1.

**Feedforward** — two linear layers with ReLU: `(22,4) → (22,16) → (22,4)`. Biases of shape `(16,)` and `(4,)`.

**Residual connections** — input added back after both attention and feedforward. Prevents information loss across layers.

**RMSNorm** — applied after each residual. Normalizes along last dimension only.

**32 layers** — the full forward pass repeats 32 times, matching Llama 3 depth.

## Background

Built as Level 2 of a personal learning-AI project after developing a solid conceptual understanding of:

- Tokenization and BPE
- Vector embeddings and high dimensional space
- Why Q, K, V are three separate projections
- Attention scores and weighted V
- Residual connections and why they exist
- RMSNorm vs LayerNorm
- Feedforward layers as the "thinking" step
- Modern improvements: RoPE, GQA, SwiGLU (conceptually understood, classic versions implemented here)

## Prior ML experience

- Data preprocessing
- Linear regression
- ANN (artificial neural networks)
- Perceptron implementation