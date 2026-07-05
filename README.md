# GRU from Scratch — Derivation & Implementation

A **Gated Recurrent Unit (GRU) network built from scratch in NumPy** — no
deep-learning framework for the model itself. This repository has two parts:

1. **The theory** — a complete, hand-derived account of how a GRU works: the two
   gates (update + reset), the candidate hidden state, the hidden-state
   interpolation, the softmax + cross-entropy gradient, and full
   **Backpropagation Through Time (BPTT)** through the gates, with every step shown
   explicitly and illustrated.
2. **A full-stack sentiment app** — the same GRU applied to a real task: classifying
   product-review sentiment (negative / neutral / positive). It is implemented **three
   ways on identical data** — from scratch in NumPy (the derivation in Part 1, turned
   into code), and in **PyTorch** and **TensorFlow** — across four text encoders
   (word2vec, fastText, GloVe, BERT), then served through a **FastAPI** backend and a
   **React** UI that shows every model's prediction + confidence side by side.

> Educational project: the goal is to make the mechanics of a GRU explicit and
> readable, not to be fast or state-of-the-art.

> **Note on the diagrams.** The cell diagrams in `Images/` show the GRU forward pass and
> use the same variable names as the derivation below — `h⟨t⟩` (hidden state), `z⟨t⟩`
> (update gate), `r⟨t⟩` (reset gate), `h̃⟨t⟩` (candidate), with weights `W_z, W_r, W`.

---

# Part 1 — How a GRU Works (Derivation)

A complete mathematical derivation of forward propagation and backpropagation through
time (BPTT) for a GRU, including the two gate equations, the candidate hidden state, the
softmax gradient, the cross-entropy loss gradient, and the vector/matrix gradient rules.

> **Notation.** Following the cell diagrams, the hidden state is `h⟨t⟩`, the update and
> reset gates are `z⟨t⟩` and `r⟨t⟩`, the candidate is `h̃⟨t⟩`, and the gate weights are
> `W_z, W_r, W` (with biases `b_z, b_r, b`). The diagrams use the textbook **column-vector**
> form, e.g. `z⟨t⟩ = σ(W_z·[h⟨t-1⟩, x⟨t⟩] + b_z)`. The **code derives and implements the
> equivalent batch-first / row-vector layout**: data is shaped `(m, T_x, n_x)` (examples
> are rows), the gate-input concatenation is `concat = [h_prev, x]` of shape
> `(m, n_a + n_x)`, and a gate is `σ(concat · Wᵀ + b)` — weights stored as
> `(n_a, n_a + n_x)` and applied **transposed, to the right of the data**. The two forms
> are exact transposes of each other: the gradients are identical and only the orientation
> differs. **Every backward equation in §7 is written in the batch-first form, matching the
> NumPy code line for line.**
>
> **Code mapping.** The from-scratch code in [`gru_scratch.py`](gru_scratch.py) names the
> hidden state `a` (`a_prev`, `a_next`, `a_top`), the candidate weight `Wh`/`bh` (the `W`/`b`
> here), and the hidden-state size `n_a`. The math below uses the diagrams' `h`/`W`; the two
> are the same quantities under different names.

## Table of Contents

- [GRU from Scratch — Derivation \& Implementation](#gru-from-scratch--derivation--implementation)
- [Part 1 — How a GRU Works (Derivation)](#part-1--how-a-gru-works-derivation)
  - [Table of Contents](#table-of-contents)
  - [1. GRU Architecture Overview](#1-gru-architecture-overview)
  - [2. Forward Propagation](#2-forward-propagation)
  - [3. Softmax — Definition \& Gradient](#3-softmax--definition--gradient)
  - [4. Loss Function — Cross-Entropy](#4-loss-function--cross-entropy)
  - [5. Gradient of Loss w.r.t. Logits](#5-gradient-of-loss-wrt-logits)
  - [6. Gradient of Vectors and Matrices](#6-gradient-of-vectors-and-matrices)
  - [7. Backpropagation Through Time (BPTT) — batch-first](#7-backpropagation-through-time-bptt--batch-first)
    - [Output layer (per timestep)](#output-layer-per-timestep)
    - [1 — Into the hidden state, and through the interpolation](#1--into-the-hidden-state-and-through-the-interpolation)
    - [2 — Through the candidate](#2--through-the-candidate)
    - [3 — Through the two gates](#3--through-the-two-gates)
    - [4 — Carry to the previous step](#4--carry-to-the-previous-step)
  - [8. Summary of Gradient Equations](#8-summary-of-gradient-equations)
- [Part 2 — Sentiment Classification App (Full-Stack)](#part-2--sentiment-classification-app-full-stack)
- [Reference](#reference)

---

## 1. GRU Architecture Overview

A vanilla RNN carries a single hidden state `h⟨t⟩` across time and updates it by
overwriting it at every step. Because that update repeatedly multiplies by the same
recurrent weight, gradients either **vanish** or **explode** over long sequences, so the
network struggles to learn long-range dependencies.

A **GRU** fixes this with two **gates** and a single hidden state — no separate cell
state. The **update gate** `z⟨t⟩` decides how much of the hidden state to overwrite with a
freshly computed **candidate** `h̃⟨t⟩`, and the **reset gate** `r⟨t⟩` decides how much of the
past hidden state feeds into that candidate. When the update gate stays near zero the
hidden state is carried forward almost unchanged, which is the "gradient highway" that
lets information (and gradients) travel many steps without being squashed. The GRU gets
the same benefit as an LSTM with fewer parameters (two gates instead of three, and no
cell state).

**One state carried across time:**
- `h⟨t⟩` — hidden state (the cell's output, also fed to the next step and the output layer)

**Parameters (shared across all time steps), per stacked layer:**
- `W_z, b_z` — update gate
- `W_r, b_r` — reset gate
- `W, b` — candidate hidden state
- `W_y, b_y` — output (dense) layer: top hidden state → logits

Each gate weight acts on the concatenation `[h⟨t-1⟩, x⟨t⟩]`, so a single matrix mixes the
previous hidden state and the current input.

---

## 2. Forward Propagation

At each step `t` the cell concatenates the previous hidden state with the current input,
computes the two gates and the candidate, then interpolates between the old hidden state
and the candidate.

![Single GRU cell — forward pass](Images/GRU_cell_forward.png)

**Update gate** — how much of the candidate to write into the hidden state:
```
z⟨t⟩ = σ(W_z · [h⟨t-1⟩, x⟨t⟩] + b_z)
```

**Reset gate** — how much of the past hidden state feeds the candidate:
```
r⟨t⟩ = σ(W_r · [h⟨t-1⟩, x⟨t⟩] + b_r)
```

**Candidate hidden state** — the proposed new content, computed from the *reset-gated*
previous state (element-wise `∗`):
```
h̃⟨t⟩ = tanh(W · [r⟨t⟩ ∗ h⟨t-1⟩, x⟨t⟩] + b)
```

**Hidden-state update** — interpolate between the old state and the candidate:
```
h⟨t⟩ = (1 − z⟨t⟩) ∗ h⟨t-1⟩ + z⟨t⟩ ∗ h̃⟨t⟩
```

**Output logits and probabilities** (at every step — many-to-many next-token modelling):
```
y⟨t⟩ = W_y · h⟨t⟩ + b_y
ŷ⟨t⟩ = softmax(y⟨t⟩)
```

> **Batch-first form (what the code computes).** With `concat = [h⟨t-1⟩, x⟨t⟩]` shaped
> `(m, n_a + n_x)` and weights stored as `(n_a, n_a + n_x)`, each gate is
> `σ(concat · Wᵀ + b)` and the candidate is `h̃ = tanh([r ∗ h_prev, x] · Wᵀ + b)`. The
> hidden-state interpolation is unchanged (it is element-wise). See
> [`gru_scratch.py`](gru_scratch.py), `layer_forward`.

---

## 3. Softmax — Definition & Gradient

The output layer is identical to a plain classifier, so the softmax + cross-entropy
gradient is derived once here and reused for the GRU's per-step output.

The softmax of logit vector `y` at index `i` is:

$$s_i = \frac{e^{y_i}}{\sum_{k=1}^{n} e^{y_k}}$$

We can write this as `s_i = h(y) / g(y)` where:

$$h(y) = e^{y_i}, \qquad g(y) = \sum_{k=1}^{n} e^{y_k}$$

The derivative with respect to `y_j` (quotient rule):

$$\frac{\partial s_i}{\partial y_j} = \frac{h'(y)\, g(y) - g'(y)\, h(y)}{(g(y))^2}$$

We need:

$$\frac{\partial h(y)}{\partial y_j} = h'(y) = e^{y_i} \quad \text{(if } i = j\text{, else 0 → constant)}$$

$$\frac{\partial g(y)}{\partial y_j} = \frac{\partial}{\partial y_j} \sum_{k=1}^{n} e^{y_k} = e^{y_j}$$

**Case i: j = i (diagonal).** When `i = j`, `h'(y) = e^{y_i}` and `g'(y) = e^{y_i}`:

$$\frac{\partial s_i}{\partial y_j} = \frac{e^{y_i} \cdot \sum e^{y_k} - e^{y_i} \cdot e^{y_i}}{(\sum e^{y_k})^2} = \frac{e^{y_i}}{\sum e^{y_k}} \left(1 - \frac{e^{y_j}}{\sum e^{y_k}}\right)$$

$$\boxed{\frac{\partial s_i}{\partial y_j} = s_i (1 - s_j)} \quad \text{when } j = i$$

**Case ii: j ≠ i (off-diagonal).** When `i ≠ j`, `h'(y) = 0`:

$$\frac{\partial s_i}{\partial y_j} = \frac{0 - e^{y_j} \cdot e^{y_i}}{(\sum e^{y_k})^2} = -s_i \cdot s_j$$

$$\boxed{\frac{\partial s_i}{\partial y_j} = -s_i s_j} \quad \text{when } j \neq i$$

**Combined Jacobian of softmax:**

$$\frac{\partial s_i}{\partial y_j} = \begin{cases} s_i(1 - s_j) & \text{if } j = i \\ -s_i s_j & \text{if } j \neq i \end{cases}$$

---

## 4. Loss Function — Cross-Entropy

For a correct class index `m`, the cross-entropy loss is:

$$\ell = -\log(s_m), \qquad s_m = \frac{e^{y_m}}{\sum_{k} e^{y_k}}$$

The gradient with respect to `s_m`:

$$\frac{\partial \ell}{\partial s_m} = -\frac{1}{s_m}$$

---

## 5. Gradient of Loss w.r.t. Logits

By the chain rule, the loss gradient flows back through the softmax to the logits:

$$\frac{\partial \ell}{\partial y_j} = \frac{\partial \ell}{\partial s_m} \cdot \frac{\partial s_m}{\partial y_j}$$

**Case i: j = m.** Using `∂s_m/∂y_j = s_m(1 - s_j)`:

$$\frac{\partial \ell}{\partial y_j} = -\frac{1}{s_m} \cdot s_m(1 - s_j) = s_j - 1$$

$$\boxed{\frac{\partial \ell}{\partial y_j} = s_m - 1} \quad \text{if } j = m$$

**Case ii: j ≠ m.** Using `∂s_m/∂y_j = -s_m · s_j`:

$$\frac{\partial \ell}{\partial y_j} = -\frac{1}{s_m} \cdot (-s_m \cdot s_j) = s_j$$

$$\boxed{\frac{\partial \ell}{\partial y_j} = s_j} \quad \text{if } j \neq m$$

**Combined:**

$$\frac{\partial \ell}{\partial y_j} = \begin{cases} s_m - 1 & \text{if } j = m \\ s_j & \text{if } j \neq m \end{cases}$$

> **Intuition:** This is simply `ŷ - one_hot(true_label)` — the predicted probability
> vector minus the ground-truth indicator. This is exactly the `y_pred - Y` you'll see in
> the code (`dscores = (y_pred - Y) / (m * T_x)`).

---

## 6. Gradient of Vectors and Matrices

For a linear transformation `y = Wx`, the gradients are:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T, \qquad \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y}$$

**Intuition:** The weight gradient is the outer product of the upstream gradient and the
input. The input gradient backpropagates the upstream error through the transpose of the
weight matrix. These two rules differentiate every linear step in the GRU gates.

> **Batch-first form.** With examples as rows (`y = z·Wᵀ + b`, `z` shaped `(m, ·)`), the
> same rules read `∂L/∂W = (∂L/∂y)ᵀ · z` and `∂L/∂z = (∂L/∂y) · W`, with the bias gradient
> `∂L/∂b = Σ_rows ∂L/∂y` summed over the batch. These are the exact lines used in §7.

---

## 7. Backpropagation Through Time (BPTT) — batch-first

BPTT walks the sequence in reverse, routing the gradient through each cell and
accumulating it into the shared gate weights. A GRU cell carries a single state, so it
receives one incoming gradient — `dh_next` (into the hidden state, from the next step) —
which is added to the upstream gradient from the output layer, and emits `dh_prev` and
`dx`, plus the parameter gradients.

![Single GRU cell — backward pass](Images/GRU_cell_backward.webp)

All equations below are in the **batch-first** layout the code uses: the gate concatenation
`concat = [h_prev, x_t]` of shape `(m, n_a + n_x)`, the reset-gated concatenation
`concat_r = [r⟨t⟩ ∘ h_prev, x_t]`, gates of shape `(m, n_a)`, and weights of shape
`(n_a, n_a + n_x)`. They match [`gru_scratch.py`](gru_scratch.py) `layer_backward` line for
line. (`n_a` is the hidden-state size; the code names the state `a`, the candidate weight
`Wh`, and `concat` is `concat`/`concat_r`.)

### Output layer (per timestep)

With `scores = h_top · W_yᵀ + b_y` and the softmax+CE result from §5,
`dscores = (ŷ − Y) / (m·T_x)`. Applying the batch-first rules of §6 across the batch and
time axes:

$$\frac{\partial L}{\partial W_y} = \sum_{t} (\text{dscores}^{\langle t\rangle})^T \cdot h_{\text{top}}^{\langle t\rangle}, \qquad \frac{\partial L}{\partial b_y} = \sum_{m,t}\text{dscores}, \qquad \frac{\partial L}{\partial h_{\text{top}}} = \text{dscores} \cdot W_y$$

That `dh_top` is the gradient fed into the top recurrent layer at every timestep.

### 1 — Into the hidden state, and through the interpolation

Total gradient into `h⟨t⟩` (sum of the upstream from above and the carry from the next step):

$$dh = dh_{\text{above}}^{\langle t\rangle} + dh_{\text{next}}$$

The hidden-state update `h⟨t⟩ = (1 − z) ∘ h⟨t-1⟩ + z ∘ h̃` splits `dh` three ways — into
the update gate, into the candidate, and straight back to `h⟨t-1⟩` along the skip path:

$$dz = dh \,\circ\, (\tilde{h} - h^{\langle t-1\rangle}), \qquad d\tilde{h} = dh \,\circ\, z, \qquad dh_{\text{prev}} \mathrel{+}= dh \,\circ\, (1 - z)$$

### 2 — Through the candidate

The candidate `h̃ = tanh(concat_r · Wᵀ + b)` with `concat_r = [r ∘ h_prev, x]`. Back through
the `tanh` (`1 − h̃²`), then the linear step (§6):

$$d\tilde{h}_{\text{raw}} = d\tilde{h} \,\circ\, (1 - \tilde{h}^{\,2}), \qquad \frac{\partial L}{\partial W} \mathrel{+}= d\tilde{h}_{\text{raw}}^{\,T} \cdot \text{concat}_r, \qquad \frac{\partial L}{\partial b} \mathrel{+}= \sum_{\text{batch}} d\tilde{h}_{\text{raw}}$$

$$d\,\text{concat}_r = d\tilde{h}_{\text{raw}} \cdot W \quad\Rightarrow\quad d(r \!\circ\! h_{\text{prev}}) = d\,\text{concat}_r[:, :n_a], \qquad dx^{\langle t\rangle} \mathrel{+}= d\,\text{concat}_r[:, n_a:]$$

The reset-gated term splits between the reset gate and another contribution to `h_prev`:

$$dr = d(r \!\circ\! h_{\text{prev}}) \,\circ\, h^{\langle t-1\rangle}, \qquad dh_{\text{prev}} \mathrel{+}= d(r \!\circ\! h_{\text{prev}}) \,\circ\, r$$

### 3 — Through the two gates

Each gate is `σ(concat · Wᵀ + b)` with `concat = [h_prev, x]`. Back through the sigmoid
(`σ'(x) = σ(1−σ)`) and the linear step:

$$dz_{\text{raw}} = dz \,\circ\, z(1-z), \qquad dr_{\text{raw}} = dr \,\circ\, r(1-r)$$

$$\frac{\partial L}{\partial W_z} \mathrel{+}= dz_{\text{raw}}^{\,T} \cdot \text{concat}, \qquad \frac{\partial L}{\partial W_r} \mathrel{+}= dr_{\text{raw}}^{\,T} \cdot \text{concat}, \qquad \frac{\partial L}{\partial b_\bullet} \mathrel{+}= \sum_{\text{batch}} d\bullet_{\text{raw}}$$

Each gate's input gradient splits back into the hidden-state and input halves and is
accumulated into `dh_prev` / `dx`:

$$dh_{\text{prev}} \mathrel{+}= (dz_{\text{raw}} \cdot W_z)[:, :n_a] + (dr_{\text{raw}} \cdot W_r)[:, :n_a]$$

$$dx^{\langle t\rangle} \mathrel{+}= (dz_{\text{raw}} \cdot W_z)[:, n_a:] + (dr_{\text{raw}} \cdot W_r)[:, n_a:]$$

### 4 — Carry to the previous step

`dh_prev` has now collected **four** contributions — the skip path `(1 − z)`, the
reset-gated candidate path `r`, and the hidden halves of both gate inputs. Carry it back:

$$dh_{\text{next}} \leftarrow dh_{\text{prev}} \quad \text{into step } t-1.$$

In a stacked GRU the `dx` of a layer becomes the `dh_above` of the layer below.

> **Why GRUs train better than RNNs.** When the update gate `z ≈ 0`, the hidden-state
> update is `h⟨t⟩ ≈ h⟨t-1⟩` and its gradient carry is `dh_prev ≈ dh ∘ (1 − z) ≈ dh` —
> the gradient passes back through many steps almost undamped, with **no repeated `tanh`
> factor and no fixed recurrent matrix**. That additive, gated carry is the same
> vanishing-gradient cure the LSTM's cell state provides, achieved here with one gate
> fewer and no separate state.

---

## 8. Summary of Gradient Equations

| Quantity | Gradient (batch-first) |
|---|---|
| Loss `∂ℓ/∂y` (softmax+CE) | `ŷ − Y` |
| `∂L/∂W_y` | `Σₜ dscoresᵀ · h_top` |
| `∂L/∂h_top` | `dscores · W_y` |
| `dh` (into `h⟨t⟩`) | `dh_above + dh_next` |
| `dz` (update gate) | `dh ∘ (h̃ − h⟨t−1⟩)` |
| `dh̃` (candidate) | `dh ∘ z` |
| `dh̃_raw` (candidate pre-act) | `dh̃ ∘ (1 − h̃²)` |
| `d concat_r` (reset-gated concat) | `dh̃_raw · W`  (concat_r = `[r ∘ h_prev, x]`) |
| `dr` (reset gate) | `d concat_r[:, :n_a] ∘ h⟨t−1⟩` |
| `dz_raw` / `dr_raw` (pre-act) | `dz ∘ z(1−z)` / `dr ∘ r(1−r)` |
| `∂L/∂W_z`, `∂L/∂W_r` | `dz_rawᵀ · concat`, `dr_rawᵀ · concat`;  `∂L/∂W` = `dh̃_rawᵀ · concat_r` |
| `∂L/∂b•` | `Σ_batch d•_raw` |
| `dh_prev` | `dh∘(1−z) + d concat_r[:, :n_a]∘r + (dz_raw·W_z)[:, :n_a] + (dr_raw·W_r)[:, :n_a]` |
| `dx⟨t⟩` | `d concat_r[:, n_a:] + (dz_raw·W_z)[:, n_a:] + (dr_raw·W_r)[:, n_a:]` |

---

# Part 2 — Sentiment Classification App (Full-Stack)

Part 1 derives a GRU. Part 2 turns that derivation into a working app: the same
GRU is implemented from scratch in NumPy (plus PyTorch and TensorFlow versions),
trained to classify review sentiment, and served behind a small web UI.

## What it does

Given a product review, it predicts the sentiment — **negative / neutral / positive** —
and shows how the models compare on the same sentence: **four text encoders**
(word2vec · fastText · GloVe · BERT) × **three implementations** of the same GRU:

| Implementation | File | Notes |
|---|---|---|
| **PyTorch** | `model_artifacts_generation.py` | `nn.GRU`, 2 layers + dropout, last-real-word readout |
| **TensorFlow** | `model_artifacts_generation.py` | `keras.layers.GRU` → softmax |
| **Manual (NumPy)** | `manual_gru.py` | from scratch: 2-gate cell + reset-gated candidate, BPTT, Adam — the Part 1 derivation, applied to classification |

A **FastAPI** backend loads the trained models and a **React** frontend (or the
Streamlit app) sends a review to it and displays each model's label, confidence, and
class probabilities, plus a consensus vote.

## Project structure

```
GRU/
├── code/
│   ├── model_building/                 # produces the models
│   │   ├── data_generation.py          # 1. download + split reviews -> data/raw/
│   │   ├── encoder.py                  # 2. build+trim encoders, encode splits
│   │   ├── model_artifacts_generation.py  # 3. train PyTorch + TF + manual GRUs
│   │   ├── manual_gru.py               #    the from-scratch NumPy GRU (used by step 3)
│   │   └── run_pipeline.py             #    runs steps 1-3 end to end
│   └── backend/                        # serves the models (predictor.py + FastAPI app.py)
├── frontend/                           # Vite + React UI
├── streamlit_app.py                    # Streamlit deploy entry point
└── data/                               # raw splits, trimmed encoders, embeddings, model artifacts
```

## Build + run

```bash
# 1. build the models (data -> encoder -> train). reuse existing data + encoders:
cd code/model_building && python run_pipeline.py --skip data encoder

# 2. backend + frontend (two terminals)
cd code/backend && uvicorn app:app --reload --port 8000
cd frontend && npm install && npm run dev        # http://localhost:5173

# or the Streamlit app:
streamlit run streamlit_app.py
```

Encoders are trimmed to the dataset vocabulary in memory and only the small copies are
saved (a few MB each); BERT loads from HuggingFace at runtime. See
[`code/README.md`](code/README.md) for details. The data pipeline and encoders are
identical to the Vanilla-RNN / LSTM projects — only the model architecture differs.

---

## Reference

The architecture diagrams and the overall framing of the forward/backward passes follow
the **[DeepLearning.AI Sequence Models course](https://www.coursera.org/learn/nlp-sequence-models)**
on Coursera (taught by Andrew Ng). The from-scratch NumPy implementation and the
hand-worked gradient derivations in this repository are built on the notation and intuition
from that course.
