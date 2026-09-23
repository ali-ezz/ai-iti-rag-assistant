# LAB-12-Transformers

Source notebook: LAB-12-Transformers.ipynb

# 🧠 Lab: Transformer Architecture — Deep Dive — ✅ SOLVED

**Student: Ali Ezz Ali — AI ITI Track — LAB-12**

> **Day 4 — GenAI & LLM Course (ITI)**

## Objectives
By the end of this lab you will be able to:
1. Implement **Scaled Dot-Product Attention** from scratch.
2. Build a **Multi-Head Attention** layer.
3. Understand and implement **Positional Encoding**.
4. Construct a full **Transformer Encoder Block**.
5. Construct a full **Transformer Decoder Block** with **masked self-attention** and **cross-attention**.
6. Assemble a complete **Transformer model** (Encoder-Decoder).
7. Use a pre-trained Transformer via **HuggingFace** for translation/summarization.

## Instructions
- Each exercise has a **TODO** section — write your code there.
- Cells marked with ✅ contain **tests/assertions** — run them to verify your solution.
- Do **NOT** modify the test cells.

## Prerequisites
- Python 3.8+
- Packages: `torch`, `numpy`, `matplotlib`, `transformers`

---

---
## Part 1: Scaled Dot-Product Attention

### Concept Recap

The core building block of the Transformer is **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Q** (Query), **K** (Key), **V** (Value) are projected input matrices.
- $d_k$ is the dimension of the key vectors (used for scaling).
- An optional **mask** can be applied before softmax to prevent attending to certain positions (e.g., future tokens in the decoder).

---
### Exercise 1.1: Implement Scaled Dot-Product Attention

**Task**: Complete the function below.

Steps:
1. Compute the dot product between Q and K^T.
2. Scale by $\sqrt{d_k}$.
3. If a mask is provided, set masked positions to `-1e9` (a very large negative number) so softmax drives them to ~0.
4. Apply softmax along the last dimension.
5. Multiply by V to get the output.

#### ✅ Test 1.1 — Verify Your Attention Function

---
### Exercise 1.2: Visualize Attention Weights

**Task**: Create a heatmap to visualize attention weights.

1. Use the `weights_masked` from the causal-masked attention above.
2. Plot it as a heatmap using `matplotlib`.
3. Add a title "Causal Attention Weights" and axis labels ("Query Position", "Key Position").

---
### Exercise 1.3: Why Do We Scale by $\sqrt{d_k}$?

**Task (Conceptual — answer in the markdown cell below)**:

1. What happens to the magnitude of the dot products $QK^T$ as $d_k$ grows large?
2. How does this affect the softmax output?
3. Why does dividing by $\sqrt{d_k}$ fix the problem?

**Your Answer (Ex 1.3 — Why scale by $\\sqrt{d_k}$?):**

1. As $d_k$ grows large, the variance of the dot products $QK^T$ grows too (variance $\approx d_k$ if Q, K have unit variance). So raw scores get large in magnitude (very positive / very negative).
2. Large-magnitude inputs push softmax into its saturated regions: one entry $\approx 1$, rest $\approx 0$. Gradients there are near zero, so training stalls (vanishing gradients) and attention becomes overly sharp/brittle.
3. Dividing by $\sqrt{d_k}$ re-normalizes variance back to ~1, keeping softmax inputs in a range with healthy gradients and soft, learnable attention distributions.

---
## Part 2: Multi-Head Attention

### Concept Recap

Instead of performing a single attention function, **Multi-Head Attention** runs $h$ attention heads in parallel, each with its own learned projection:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

This allows the model to attend to information from different representation subspaces at different positions.

---
### Exercise 2.1: Implement Multi-Head Attention

**Task**: Complete the `MultiHeadAttention` class below.

Steps:
1. Create linear projections for Q, K, V, and the output.
2. In `forward()`, project Q, K, V, then reshape to split into heads.
3. Apply `scaled_dot_product_attention` to each head.
4. Concatenate heads and apply the output projection.

#### ✅ Test 2.1 — Verify Multi-Head Attention

---
### Exercise 2.2: Conceptual Questions on Multi-Head Attention

**Task (answer below)**:

1. If `d_model = 512` and `num_heads = 8`, what is `d_k` (dimension per head)?
2. Why is Multi-Head Attention better than single-head attention with the same total dimension?
3. In **self-attention**, Q, K, V all come from the same input. In **cross-attention**, where do Q, K, V come from respectively?

**Your Answer (Ex 2.2):**

1. `d_k = d_model / num_heads = 512 / 8 = 64` per head.
2. Single-head attention with the same total dimension learns only one attention pattern (one weighted average). Multi-head projects into `h` different Q/K/V subspaces, so each head can attend to different relations (e.g., syntax vs coreference, short vs long range) in parallel; concatenating + `W_o` mixes them. Same total dim, richer representation.
3. Self-attention: Q, K, V all come from the **same** sequence (e.g., encoder input `x`, or decoder target `x`). Cross-attention: **Q comes from the decoder target** (query = "what do I need next?"), while **K and V come from the encoder output** (memory being attended to).

---
## Part 3: Positional Encoding

### Concept Recap

Since the Transformer has no recurrence or convolution, it needs **Positional Encoding** to inject information about the position of tokens in the sequence.

The original paper uses sinusoidal positional encoding:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---
### Exercise 3.1: Implement Sinusoidal Positional Encoding

**Task**: Complete the `PositionalEncoding` class.

#### ✅ Test 3.1 — Verify Positional Encoding

---
### Exercise 3.2: Visualize Positional Encoding

**Task**: Plot the positional encoding matrix as a heatmap.

1. Create a `PositionalEncoding` with `d_model=64` and extract the PE matrix for the first 50 positions.
2. Plot it using `plt.imshow()` with `aspect='auto'`.
3. Add axis labels: "Position" (y-axis) and "Dimension" (x-axis).

---
## Part 4: Feed-Forward Network & Layer Normalization

### Concept Recap

Each Transformer sub-layer is followed by:
1. A **Position-wise Feed-Forward Network** (two linear layers with ReLU):
   $$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$
   The inner dimension $d_{ff}$ is typically 4× $d_{model}$.

2. **Residual Connection + Layer Normalization**:
   $$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

---
### Exercise 4.1: Implement the Feed-Forward Network

**Task**: Complete the `PositionwiseFeedForward` class.

#### ✅ Test 4.1 — Verify Feed-Forward Network

---
## Part 5: Transformer Encoder Block

### Concept Recap

A single Transformer **Encoder Block** consists of:
1. Multi-Head **Self-Attention** + Residual + LayerNorm
2. **Feed-Forward Network** + Residual + LayerNorm

```
Input
  │
  ├──> Multi-Head Self-Attention ──> Add & LayerNorm
  │
  ├──> Feed-Forward Network ──────> Add & LayerNorm
  │
Output
```

---
### Exercise 5.1: Implement the Encoder Block

**Task**: Complete the `EncoderBlock` class using your `MultiHeadAttention` and `PositionwiseFeedForward`.

#### ✅ Test 5.1 — Verify Encoder Block

---
## Part 6: Transformer Decoder Block

### Concept Recap

A single Transformer **Decoder Block** consists of:
1. **Masked** Multi-Head Self-Attention + Residual + LayerNorm
2. **Cross-Attention** (attends to encoder output) + Residual + LayerNorm
3. Feed-Forward Network + Residual + LayerNorm

The masked self-attention uses a **causal mask** so each position can only attend to earlier positions.

---
### Exercise 6.1: Implement the Decoder Block

**Task**: Complete the `DecoderBlock` class.

#### ✅ Test 6.1 — Verify Decoder Block

---
## Part 7: Full Transformer Model (Encoder-Decoder)

### Concept Recap

The complete Transformer stacks $N$ encoder blocks and $N$ decoder blocks:

```
Source Tokens ──> Embedding + PE ──> [Encoder × N] ──> Encoder Output
                                                           │
Target Tokens ──> Embedding + PE ──> [Decoder × N] ──> Linear ──> Softmax ──> Output Probs
```

---
### Exercise 7.1: Assemble the Full Transformer

**Task**: Complete the `Transformer` class by combining all the components.

#### ✅ Test 7.1 — Verify Full Transformer

---
## Part 8: Conceptual Questions — Transformer Architecture

**Task (answer all questions below)**:

### Q1: Encoder vs Decoder
What is the key structural difference between an Encoder block and a Decoder block?

### Q2: Masking
Explain the two types of masks used in the Transformer:
- **Padding mask**: Why is it needed? When is it used?
- **Causal (look-ahead) mask**: Why is it needed? When is it used?

### Q3: Encoder-Only vs Decoder-Only vs Encoder-Decoder
Give an example model for each architecture variant and a task it excels at:
- Encoder-only: (model? task?)
- Decoder-only: (model? task?)
- Encoder-Decoder: (model? task?)

### Q4: Self-Attention Complexity
What is the time and memory complexity of self-attention with respect to sequence length $n$? Why does this create problems for very long sequences?

**Your Answers:**

**Q1: Encoder vs Decoder — key structural difference:**
- Encoder block = self-attention (bidirectional, sees full source) + FFN, with 1 residual+norm per sublayer (2 norms total).
- Decoder block = **masked** self-attention (causal, cannot see future target tokens) + **cross-attention** (Q from decoder, K/V from encoder output) + FFN, with 3 residual+norm stages. The cross-attention is the defining addition — it conditions generation on the encoded source.

**Q2: Masking:**
- **Padding mask:** Needed because batches pad short sequences with `<pad>` tokens. Without masking, attention/softmax would waste probability mass on meaningless pads and the loss would learn from them. Used in **both encoder self-attention and decoder cross-attention** (as `src_mask`), and optionally decoder self-attention, to ignore pad positions (set scores to `-inf` before softmax).
- **Causal (look-ahead) mask:** Needed to preserve **autoregressive** property during training — position `i` must only attend to positions `<= i`, otherwise the model cheats by seeing the future token it must predict. Used in **decoder masked self-attention** (`tgt_mask`, upper-triangular `True`). At inference it also enables step-by-step generation.

**Q3: Variants:**
- **Encoder-only (e.g., BERT):** Bidirectional self-attention, pretraining = masked-LM + NSP. Excels at **understanding/classification** — sentiment analysis, NER, QA encoding, sentence embeddings.
- **Decoder-only (e.g., GPT-2/3/4, LLaMA):** Causal masked self-attention only, pretraining = next-token prediction. Excels at **generation** — text completion, chat, code generation, summarization-as-generation.
- **Encoder-Decoder (e.g., T5, BART, original Transformer, MarianMT):** Full architecture. Excels at **sequence-to-sequence** — machine translation, abstractive summarization, where input and output lengths/domains differ and source conditioning via cross-attention helps.

**Q4: Self-attention complexity:**
- Time **O(n²·d)** and memory **O(n²)** (per head, per layer) w.r.t. sequence length `n`, because the `QK^T` score matrix is `n×n` (every query attends to every key). For very long sequences (long docs, high-res images as patches, audio) the attention matrix explodes in memory/compute and becomes the bottleneck — motivating sparse/long-range variants (Longformer, Linformer, FlashAttention, sliding-window).

---
## Part 9: Using Pre-Trained Transformers with HuggingFace 🤗

Now let's use a real pre-trained Transformer model for a practical task.

---
### Exercise 9.1: Text Summarization with a Pre-Trained T5 Model

**Task**: Use the HuggingFace `transformers` library to summarize a text passage.

Steps:
1. Load the `t5-small` model and tokenizer.
2. Tokenize the input text (prefix it with `"summarize: "`).
3. Generate a summary using `model.generate()`.
4. Decode and print the summary.

---
### Exercise 9.2: Explore Attention Patterns of a Pre-Trained Model

**Task**: Visualize the attention weights from a pre-trained BERT model.

Steps:
1. Load `bert-base-uncased` with `output_attentions=True`.
2. Tokenize a sentence.
3. Run a forward pass and extract attention weights.
4. Plot a heatmap of one attention head from the last layer.

---
## Part 10: Bonus Challenges 🏆

### Bonus 10.1: Implement a Helper Function to Generate Causal Masks

**Task**: Write a function that generates a causal (look-ahead) mask for a given sequence length.

---
### Bonus 10.2: Compare Transformer Parameter Count

**Task**: Create Transformer models with different configurations and compare their parameter counts. Fill in the table below.

| Config | d_model | num_heads | d_ff | num_layers | Total Params |
|--------|---------|-----------|------|------------|-------------|
| Tiny   | 64      | 4         | 256  | 2          | ?           |
| Small  | 128     | 8         | 512  | 4          | ?           |
| Medium | 256     | 8         | 1024 | 6          | ?           |

---
## 🎉 Congratulations!

You've completed the Transformer Deep Dive lab! You have:

- ✅ Built **Scaled Dot-Product Attention** from scratch
- ✅ Implemented **Multi-Head Attention**
- ✅ Created **Positional Encoding**
- ✅ Assembled a full **Encoder Block** and **Decoder Block**
- ✅ Combined everything into a complete **Transformer model**
- ✅ Used pre-trained Transformers via **HuggingFace** for real tasks

### Key Takeaways
1. The Transformer replaces recurrence with **self-attention**, enabling full parallelization.
2. **Multi-Head Attention** lets the model attend to different representation subspaces.
3. **Positional Encoding** injects sequence order since attention is position-agnostic.
4. **Residual connections** and **Layer Normalization** are critical for training deep models.
5. The Encoder-Decoder design is the foundation for models like T5, while encoder-only (BERT) and decoder-only (GPT) variants power many modern applications.
