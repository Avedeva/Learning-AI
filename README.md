# 🚀 Learning-AI

> My journey of learning Artificial Intelligence by building everything from scratch.

---

## 🧠 About This Repository

This repository documents my hands-on journey into **Artificial Intelligence, Machine Learning, and Deep Learning**.

Instead of just consuming tutorials, I focus on:

* 🔨 Building things from scratch
* 📚 Understanding fundamentals deeply
* 🚀 Creating real projects that reflect genuine understanding

---

## 📂 Projects

### 🔤 Byte Pair Encoding (BPE)

📁 `BPE/`

A tokenizer built entirely from scratch — no libraries.

* Learns subword patterns from raw text data
* Implements pair frequency counting, token merging, encoding & decoding
* Builds a vocabulary of 275 tokens, reused in later projects

**Files:**
```
BPE/
├── bpe.ipynb    — full implementation
└── README.MD
```

---

### 🧠 Neural Network Refresher (MNIST)

📁 `Refresher/`

A from-scratch exploration of neural networks — no ML frameworks, just math and NumPy.

* **Perceptron** — single-neuron binary classifier, implemented manually
* **ANN (scratch_mnist)** — multi-layer neural network with forward prop, backprop, and gradient descent
* **MNIST** — additional experiments on the handwritten digit dataset
* Trained on real MNIST data (train/test CSVs included)

**Files:**
```
Refresher/
├── perceptron.ipynb      — perceptron from scratch
├── scratch_mnist.ipynb   — full ANN from scratch
├── mnist.ipynb           — MNIST experiments
├── train.csv / test.csv  — MNIST dataset
└── Readme.md
```

---

### 🔍 Self-Attention from Scratch

📁 `Self Attention/`

A complete forward pass of one transformer layer — the core of every modern LLM — built in raw PyTorch with no `transformers` library.

**Pipeline:**
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

**What it implements:**
* Single-head self-attention with Q, K, V projections
* Sinusoidal positional encoding
* Residual connections after both attention and feedforward
* RMSNorm normalization
* 32-layer forward pass (matching Llama 3 depth)

**What it does NOT implement:**
* Backpropagation / optimizer step — this is a pure forward pass
* Multi-head attention
* RoPE positional encoding
* Token prediction / vocabulary projection

> The goal is understanding the mechanics — not training a model.

**Files:**
```
Self Attention/
├── attention.ipynb   — main implementation
├── bpe.py            — custom BPE tokenizer (reused from Level 1)
└── README.md
```

**How to run:**
```bash
pip install torch numpy
```
Open `attention.ipynb` and run all cells top to bottom.

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch (raw, no high-level APIs)
* NumPy
* Jupyter Notebook
* Core ML/DL concepts — minimal framework reliance by design

---

## 🎯 Goals

* ✅ Build strong ML fundamentals (perceptron, ANN, backprop)
* ✅ Implement tokenization (BPE) from scratch
* ✅ Implement self-attention and a transformer forward pass from scratch
* 🚧 Move deeper into LLM architecture (multi-head attention, RoPE, GQA)
* 🔜 Implement training — backprop through attention, optimizer step
* 🎯 Contribute to open source (GSoC)

---

## 📈 What I'm Currently Doing

* Building transformer components from the ground up
* Understanding internals: why Q/K/V are separate, how residuals prevent collapse, what RMSNorm does vs LayerNorm
* Conceptually studied modern improvements — RoPE, GQA, SwiGLU — implementing classic versions first

---

## 💡 Philosophy

> "Don't just use AI — understand it."

---

## 📬 Connect With Me

* GitHub: [Avedeva](https://github.com/Avedeva)
* LinkedIn: [Suneet Mishra](https://www.linkedin.com/in/suneet-mishra-68652a311/)

---

## ⭐ Support

If you find this repo helpful:

* Star ⭐ it
* Fork 🍴 it
* Follow my journey 🚀
