# LAB-04-Deep-Vision

Source notebook: LAB-04-Deep-Vision.ipynb

# Computer Vision — Part 2 · Practical Notebook
## SOLVED VERSION — all TODOs completed

**Companion lab for the lecture: Classic Networks → ResNets → 1×1 Convs → Transfer Learning → Data Augmentation → Vision Transformers → Evaluation → Model Persistence**

---

This notebook is the **fully solved copy** of `STUDENT_(1).ipynb`. Every `TODO` has been filled with working code + written answers. Original structure, plots, and verifications are preserved.

## 1 · Setup & device

We install (if needed) and import everything once. The `device` line is the
standard idiom: write code once, run on GPU if available, fall back to CPU.

## 2 · Activations — a 2-minute visual sanity check

> *Lecture recap:* an activation is a **non-linear, differentiable** function
> placed between layers. Non-linear so the network can't collapse into one
> linear layer; differentiable so backprop has gradients to flow.

In practice you rarely *implement* activations — you pick one. But you should
be able to *see* why some choices cause trouble:

- **ReLU** — the default. Cheap, no saturation on the positive side. Risk:
  *dying ReLU* (negative inputs → 0 gradient → neuron stops learning).
- **Leaky ReLU** — small slope for negatives, fixes dying ReLU.
- **Sigmoid / tanh** — saturate at the tails → *vanishing gradients*. Avoid in
  hidden layers; sigmoid still fine as a final layer for binary probability.

Let's just plot them and their gradients.

**Read the bottom row — that's the row that matters for training.**
Notice the sigmoid and tanh gradients are nearly flat (≈0) once you move away from the centre. That flatness *is* the vanishing-gradient problem. ReLU's gradient is a clean 1 on the positive side — that is why it became the default.

## 3 · Convolution mechanics: shapes, stride, padding, pooling

> *Lecture recap:* a filter slides over the image computing dot products.
> **Stride** = step size (bigger stride → smaller output, less compute).
> **Padding** = a frame around the image so edge pixels get equal attention and
> (with the right padding) the output keeps the input's height/width — that's
> *"same"* padding. **Pooling** downsamples; max-pool keeps the most salient
> value in each window.

The single most useful practical skill here is **predicting output shapes**.
You will debug shape errors constantly. The formula for one spatial dimension:

$$ \text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1 $$

where `k` = kernel size, `s` = stride, `p` = padding.

Notice that `3x3, stride 1, padding 1` keeps the size at 32 — that's the
**"same" convolution** the lecture mentioned, and it's exactly why ResNet can
add a block's input to its output (the shapes line up).

Now pooling — same formula, no learnable weights:

## 4 · Using pretrained classic networks — the real-world way

> *Lecture recap:* LeNet-5 (1998, ~60k params, digits), AlexNet (2012, ~60M
> params, ReLU + ImageNet), VGG-16 (~138M params, only 3×3 convs). The trend:
> deeper networks, height/width shrink, channel count grows.

**You will almost never implement these from scratch in a real job.** You load
them with pretrained weights. `torchvision.models` gives you the whole zoo with
one line. Let's inspect a few and confirm the lecture's claims about parameter
counts and the shrink-spatial / grow-channels pattern.

VGG-16's ~138M parameters versus ResNet-18's ~11M is a big practical
point: **deeper is not the same as bigger**, and ResNet's design gets *more*
depth with *fewer* parameters. That efficiency is part of why ResNet, not VGG,
is the default backbone today.

Let's *watch* the "spatial shrinks, channels grow" pattern by pushing one image
through ResNet-18 layer by layer:

Read the `H, W` column going down — it shrinks 112 → 56 → 56 → 28 → 14 → 7
→ 1. Read the `C` column — it grows 64 → 64 → 128 → 256 → 512. **Exactly the
lecture's pattern.** The final `avgpool` collapses the 7×7 map to 1×1 — that's
*global average pooling*, the modern replacement for giant fully-connected
layers (this is the "Network in Network" idea from the lecture).

## 5 · ResNet & the residual block

> *Lecture recap:* very deep plain networks train *worse*, not better — signal
> and gradients get "scrambled" by many random weight multiplications. The fix
> is the **skip connection**: `output = F(x) + x`. If a block has nothing useful
> to add, it can easily learn `F(x) ≈ 0` and just pass `x` through (the identity
> function). Skip connections also give gradients a short path back to early
> layers.

Here's a residual block written out — this **is** worth seeing in code once,
because the `+ identity` line is the entire idea:

### Why BatchNorm lives in these blocks

> *Lecture recap:* BatchNorm normalizes each layer's activations using
> **batch statistics** during training, then uses **running (whole-training)
> statistics** at test time. It reduces *internal covariate shift*, smooths the
> loss surface, and speeds up training. In PyTorch you just add `nn.BatchNorm2d`
> — the framework tracks the running stats for you.

The one thing you **must** get right in practice: call `model.train()` vs
`model.eval()`. They switch BatchNorm (and Dropout) between the two modes.
Let's prove the modes actually behave differently:

## 6 · Transfer learning — fine-tune a pretrained model on a real dataset

> *Lecture recap:* you rarely have enough data to train from scratch. Download
> a network pretrained on ImageNet (1000 classes), **replace the final
> classification layer** with your own, and either (a) **freeze** the rest and
> train only the new head (small dataset), or (b) **fine-tune** more layers
> (larger dataset). Someone else's pretrained weights do the heavy lifting.

**This is the most important section of the notebook** — it's the workflow you
will use again and again. We'll fine-tune a pretrained **ResNet-18** to classify
**ants vs bees** (the classic `hymenoptera` dataset — small, downloads in
seconds, perfect for a lab).

### 6.1 — Data pipeline with the *correct* preprocessing

A pretrained model expects its inputs preprocessed **the same way the original
training data was**. For ImageNet models that means: resize to 224×224 and
normalize with ImageNet's channel mean/std. Getting this wrong silently wrecks
accuracy — it's a real, common bug.

Notice the **train** transform also does augmentation (random crop + flip);
the **val** transform does not. You never augment validation data.

Let's actually *look* at a batch (always look at your data — another real
habit). We have to undo the normalization for display.

### 6.2 — Build the transfer-learning model

The recipe in three steps:
1. Load ResNet-18 **with pretrained ImageNet weights**.
2. **Freeze** all existing layers (`requires_grad = False`) — they're already
   good feature extractors.
3. **Replace** the final fully-connected layer (`model.fc`) with a fresh one
   sized for *our* number of classes. New layers are trainable by default.

### 6.3 — The training loop

This is the **canonical PyTorch training loop**. Memorize its shape — you'll
write it in every project:

```
for each epoch:
    model.train()
    for each batch:
        optimizer.zero_grad()      # clear old gradients
        outputs = model(inputs)    # forward pass
        loss = criterion(...)      # compute loss
        loss.backward()            # backprop -> gradients
        optimizer.step()           # update weights
    model.eval()
    ... evaluate on validation, no gradients ...
```

### 6.4 — Inspect predictions

Always look at what the model actually predicts, including its confidence.

## 7 · Data augmentation pipelines

> *Lecture recap:* most CV tasks want more data. Augmentation makes "new" data
> from what you have: mirroring, random crops, rotation/shear, and **color
> shifting** (jitter the R/G/B channels — lighting changes the pixels but not
> the label). The golden rule: an augmentation is valid only if it **preserves
> the label**.

You already used `RandomResizedCrop` + `RandomHorizontalFlip` in Section 6.
Let's *see* what a fuller augmentation pipeline does to a single image — this is
exactly what `torchvision.transforms` is for, and it's what you'll use in real
projects.

Each panel is the *same bee*, but to the network they're different inputs. That's free extra data, and it makes the model robust to camera angle, lighting, and framing.

## 8 · Vision Transformer (ViT) vs ResNet

> *Lecture recap:* CNNs have **high inductive bias** — locality, translation invariance, hierarchy. That bias helps with small data but limits the function space. The Vision Transformer drops most of that bias: it cuts the image into patches and runs **self-attention** over them, so any patch can attend to any other patch **in a single step** (a CNN needs many layers to grow its receptive field). Attention also generalizes across modalities (text, image, audio) — that is why Transformers took over.

In practice you **use** a pretrained ViT exactly like a pretrained ResNet — `torchvision` provides both. We will load each, classify the same image with both, and then fine-tune a ViT on the ants-vs-bees dataset.

### 8.1 — Find the ViT classification head

Before fine-tuning a ViT you need to know *what its final layer is called*. For ResNet it was `model.fc`; for `vit_b_16` it is something else. Run the cell below and **scroll to the very bottom** of the printout to find the classification head.

**Hint:** the ViT classification head is `vit.heads.head` — `heads` is a container module, and `head` is the final `Linear` classifier inside it. That is the layer you replace when fine-tuning, just like `model.fc` for ResNet.

---
## 9 · Training, validation, and evaluation for image classification

> *Lecture recap:* a real training run isn't just "call `.fit()`". You need a
> **train/val split** (val data the model never trains on, used to catch
> overfitting), a **training loop** (Section 6.3), and — once training is
> done — a proper **evaluation** pass: accuracy alone hides a lot, so we also
> look at **per-class accuracy** and a **confusion matrix** to see exactly
> which classes the model confuses.

You already trained `model` (the frozen-backbone ResNet-18 from Section 6) and
collected its `history`. This section reuses that trained model and asks:
*how good is it, really, and where does it fail?*

### 9.1 — Re-reading the train/val curves

Before computing any new metric, always re-examine the learning curves you
already have. The gap between train and val lines tells you almost everything:

- Train and val loss both falling, close together → healthy fit.
- Train loss keeps falling but val loss rises → **overfitting** — the model is
  memorizing the training set.
- Both losses stay high and flat → **underfitting** — the model (or training
  setup) doesn't have enough capacity or has trained too briefly.

### 9.2 — Full validation-set evaluation

`run_epoch(val_loader, train_mode=False)` already gives you overall val loss
and accuracy. But **overall accuracy can hide a lopsided model** — e.g. 95%
overall while one class is never predicted correctly. To see that, we need
every prediction, not just an aggregate.

### 9.3 — Per-class accuracy and a confusion matrix

A **confusion matrix** is a grid: rows are true classes, columns are
predicted classes. The diagonal is correct predictions; anything off-diagonal
is a specific *kind* of mistake ("model confuses class A for class B"). This
is the standard diagnostic every real classification project reports.

### TODO 9.4 — Written check

Look at your confusion matrix and per-class accuracy above, then answer in a
comment:

1. Is either class systematically harder for the model? How can you tell from
   the matrix?
2. Suppose overall accuracy were 90% but one class had 60% accuracy and the
   other 100%. Why is reporting *only* the overall 90% misleading here?

---
## 10 · Saving and reusing trained models

> *Lecture recap:* training is expensive; deployment needs the trained
> weights **without** retraining. PyTorch's standard practice is to save a
> model's `state_dict()` — a plain dictionary mapping layer names to tensors —
> rather than pickling the whole model object. This is more portable (survives
> code refactors better) and is the format you will see in essentially every
> real PyTorch project and pretrained-weights release.

### 10.1 — Saving a trained model's `state_dict`

Two things are worth saving together: the **weights** (`state_dict()`) and
enough **metadata** to rebuild the exact same architecture later (here: the
class names, since `model.fc`'s output size depends on `len(class_names)`).

### 10.2 — Loading the checkpoint into a fresh model

Loading is the mirror image of saving, with one detail that trips people up
every time: you must **rebuild the same architecture first** (a fresh
`resnet18` with `model.fc` already replaced to match `len(class_names)`),
*then* load the weights into it. `state_dict()` has no idea what architecture
it belongs to — it's just tensor names and values.

### 10.3 — Prove the loaded model matches the original

The whole point of saving/loading is that the reloaded model behaves
**identically** to the one you trained. Verify this directly: run the same
validation batch through both `model` and `loaded_model` and confirm the
predictions (and probabilities) match.

### TODO 10.4 — Save a full "deployment-ready" checkpoint

Real deployments usually save a bit more than just weights + class names —
enough that anyone (including future-you) can load the model with **no other
context**. Extend the checkpoint dict with:

- `"architecture"`: a string, e.g. `"resnet18"`, naming which backbone to
  rebuild.
- `"input_size"`: the expected input spatial size (224 for this model).
- `"val_accuracy"`: the final validation accuracy from `history` (Section 6),
  so anyone loading the checkpoint knows how good it is *before* running it.

Save this to a new file and print its contents to confirm.
