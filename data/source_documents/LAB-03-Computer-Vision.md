# LAB-03-Computer-Vision

Source notebook: LAB-03-Computer-Vision.ipynb

# Computer Vision — Practical Notebook

**Companion lab for: Computer Vision Foundations → Neural Network Fundamentals → Image & Data Preparation → Convolutional Neural Networks → CNN Architecture → Data Augmentation → Transfer Learning**

---

This notebook is the complete hands-on companion to the lecture deck. It has two halves:

- **Part A — Classical Computer Vision** (`cv2` + NumPy): images as numbers, color
  spaces, thresholding, morphology, filters/convolution, Sobel, Canny, geometric
  transforms, histogram equalization. Nothing is "trained" here — this is the
  toolkit used in industry every day *before* any neural network gets involved.
- **Part B — Neural Networks & CNNs from scratch** (mostly plain NumPy, a little
  PyTorch at the end): a single neuron and MLP forward pass, activation
  functions, weight initialization and why it matters, batch normalization,
  residual connections, overfitting/underfitting and regularization, then the
  mechanics of **convolution, padding, stride, and pooling** built by hand
  before you ever call `nn.Conv2d`, finishing with **data augmentation** and a
  short note on **transfer learning**.

To make the *why* stick, key operations are implemented **twice**: once by hand
with NumPy so you see the mechanics, then with the one-line library call
(`cv2....` or `torch.nn....`) you would actually use in production.

> **How to run this:** Google Colab has everything pre-installed. Locally you
> need `opencv-python`, `numpy`, `matplotlib`, `scikit-image`, and `torch` (only
> for the very last section). No GPU required — everything here runs fine on CPU.

> **Legend:** This is the **SOLVED** version — all student placeholders have been filled in.

## 0 · Setup

Import everything we'll need for both parts of the notebook. We keep this
**fully offline and reproducible** — Part A uses a sample image that ships with
`scikit-image` (no downloads needed); Part B builds tiny synthetic datasets
in NumPy so nothing depends on an external download either.

---
# Part A · Classical Computer Vision

Everything in this part happens *before* deep learning: pixels, color spaces,
thresholding, morphology, convolution-based filters, edges, geometric
transforms, and histogram equalization.

## A1 · Images as grids of numbers

> *Lecture recap:* every pixel in an image is a **numerical value** with a
> location `(x, y)`. Treating an image as a grid of numbers is the basis of
> almost every classical technique: you can brighten an image by multiplying
> every pixel, shift it by adding, invert it by subtracting from the max value,
> and so on. Most color and shape transformations are just pixel-by-pixel math.

**The indexing trap:** the lecture talks about pixels at `(x, y)`, but NumPy
arrays are indexed `[row, column]`, i.e. `[y, x]`. Mixing these up is one of
the most common beginner bugs — we'll be careful and always note it.

Now the lecture's claim — *"images are numbers, so we can do math on
them"* — demonstrated. **Overflow warning:** `uint8` wraps around at 255, so
brightening must be done in a wider type before clipping back.

### TODO A1.1
Write a function `adjust(image, contrast, brightness)` implementing the
standard point operation `output = contrast * pixel + brightness`, **safely**
(no `uint8` overflow — clip to 0–255). Test it with `contrast=1.5,
brightness=30` and display next to the original.

## A2 · Color images, channels & color spaces

> *Lecture recap:* a color image is a **3D cube** — width × height × **depth**,
> where depth is the number of color channels. RGB has depth 3. **HSV** is
> useful because **Value** directly represents brightness (great for object
> tracking / color thresholding / OpenCV masking), while **HLS**'s Lightness
> behaves more perceptually balanced (useful for lane detection, illumination
> analysis). OpenCV stores loaded images as **BGR**, not RGB — a classic
> gotcha.

### TODO A2.1
The lecture asks: *to discriminate cats from dogs by fur color, should you use
grayscale or RGB, and why?* Answer in a comment. Then implement the three
RGB→grayscale formulas by hand and compare to OpenCV:

- **Lightness:** `(max(R,G,B) + min(R,G,B)) / 2`
- **Average:** `(R + G + B) / 3`
- **Luminosity** (best — matches human vision, most sensitive to green):
  `0.21*R + 0.72*G + 0.07*B`

## A3 · Color thresholding — selecting a region of interest

> *Lecture recap:* a **color threshold** isolates a region by keeping only
> pixels whose color falls in a chosen range — classic example: a green
> screen. A plain RGB threshold breaks under changing light; thresholding in
> **HSV** is more robust because hue stays fairly stable under lighting
> changes.

We build a synthetic "green screen" scene so this is fully offline and
reproducible.

## A4 · Morphological operations — cleaning binary masks

Real masks are never clean — stray pixels, small holes, jagged edges.

- **Erosion** shrinks white regions — kills bright noise, thins connections.
- **Dilation** grows white regions — fills small holes, thickens lines.
- **Opening** (erode→dilate): kills noise *without* shrinking the object.
- **Closing** (dilate→erode): fills holes *without* growing the object.

### TODO A4.1
Scanned-document scenario: black text on white paper, so the **object is the
dark region** — erosion/dilation/open/close roles are **flipped** relative to
the white-disk example above. There are:
- **Stray dark specks** in the white background (scanner dust).
- **Small gaps** inside thick dark strokes (scanner missed pixels).

Pick the right operation for each and chain them into a `clean` mask.

## A5 · Convolution, high-pass filters, and Sobel gradients

> *Lecture recap:* a **filter** is a small matrix (a convolution **kernel**).
> Convolution slides the kernel over the image; at each position it multiplies
> kernel weights by the underlying patch and sums, producing one output pixel.
> For an **edge-detection (high-pass) kernel the weights must sum to zero** —
> the filter computes a *difference*, and non-zero sum would brighten/darken
> the whole image uniformly. The **Sobel** operator uses two such kernels (one
> per axis) to get gradients `Gx`, `Gy`; combine into **magnitude =
> √(Gx²+Gy²)** and **direction = atan(Gy/Gx)**.

### TODO A5.1
Threshold the Sobel magnitude into a clean **binary edge map**: pixels above a
value become edges (white), the rest background (black). Write
`edge_map(magnitude, threshold)` and show low / medium / high thresholds to
see the trade-off.

## A6 · Low-pass filters — averaging, Gaussian, and median

> *Lecture recap:* **low-pass filters** block high-frequency content — they
> blur/smooth an image and reduce noise. Unlike an edge kernel, a low-pass
> kernel's weights **must sum to 1** (preserve overall brightness). The
> **median filter** is nonlinear (no weighted sum — it picks the median in a
> window) and is uniquely good against salt-and-pepper noise.

## A7 · Canny edge detection

> *Lecture recap:* Canny combines what we've built: **Gaussian blur** → Sobel
> gradients → **non-maximum suppression** (thin edges to local maxima along
> the gradient direction) → **hysteresis** (double threshold: strong edges kept
> outright, weak edges kept only if connected to a strong one).

## A8 · Geometric transforms — translate, rotate, scale

OpenCV expresses these with **2×3 affine matrices** fed to `cv2.warpAffine`.

### TODO A8.1
Produce an image **rotated 45° AND scaled to 70%** in **one** `warpAffine`
call (`cv2.getRotationMatrix2D` takes a `scale` argument directly). Then
answer in a comment: if rotation and translation were done as two *separate*
`warpAffine` calls, would the order matter?

## A9 · Histograms & histogram equalization

> *Lecture recap:* a **histogram** plots pixel count vs intensity. A narrow
> histogram = low contrast; one spread across 0–255 = high contrast. **Global
> equalization** redistributes intensities via the cumulative distribution
> function. Its weakness: it equalizes using the *whole* image's histogram, so
> a locally dark region in an otherwise bright photo stays dark — **CLAHE**
> (Contrast Limited Adaptive Histogram Equalization) fixes this by equalizing
> small tiles independently.

### TODO A9.1
Apply both `cv2.equalizeHist` and `cv2.createCLAHE` to the **original `gray`**
image and compare all three side by side.

---
# Part B · Neural Network & CNN Foundations

Now we move from classical pixel operations to the learning-based pipeline:
neurons, activations, initialization, batch norm, residuals,
overfitting/underfitting, and finally the CNN building blocks — convolution,
padding, stride, and pooling — built from scratch before using PyTorch.

## B1 · A single neuron and a tiny MLP

> *Lecture recap:* a **neuron** computes a weighted sum of its inputs plus a
> bias, then passes it through a non-linear **activation function**. Stacking
> neurons into layers gives a **Multi-Layer Perceptron (MLP)**. Key
> **hyperparameters**: number of layers, number of neurons, learning rate,
> batch size, optimizer, number of iterations.

### TODO B1.1
Build a tiny 2-layer MLP forward pass by hand: input size 3 → hidden size 4
(ReLU) → output size 1 (sigmoid). Use small random weights.

## B2 · Activation functions

> *Lecture recap:* activation functions take any real number as input and
> output a value in a bounded (or semi-bounded) range using a **non-linear,
> differentiable** function. Without non-linearity, stacking layers would
> collapse into one big linear function — no matter how deep.

## B3 · Vanishing & exploding gradients, and how we stabilize them

> *Lecture recap:* deep networks suffer from **vanishing** (gradients shrink
> to ~0 through many layers, early layers stop learning) or **exploding**
> (gradients blow up, training diverges) gradients. Three stabilizers:
> 1. **Good initialization** — preserves signal magnitude across layers.
> 2. **Activation function choice** — saturating functions (sigmoid/tanh) are
>    more prone to vanishing gradients than ReLU-family functions.
> 3. **Batch normalization** — normalizes activations to mean 0, variance 1
>    per batch, then applies learnable scale (γ) and shift (β).
> 4. **Residual connections** — an identity shortcut lets gradients flow
>    directly backward, sidestepping vanishing through many layers.

Let's *see* vanishing gradients happen with poor vs. good initialization.

### TODO B3.1
Implement **batch normalization**'s forward pass by hand: given a batch of
activations, compute the mean and variance, normalize to mean 0 / variance 1,
then apply learnable scale `gamma` and shift `beta`.

### TODO B3.2
Implement a **residual (skip) connection**: `output = F(x) + x`, where `F` is
some small transformation. Compare gradient flow (conceptually) by showing
that even if `F(x)` were exactly zero, the identity path still passes `x`
through unchanged — this is why residual networks are easier to train deep.

## B4 · Overfitting, underfitting, and regularization

> *Lecture recap:* **underfitting** = model too simple / undertrained (fix:
> increase capacity, tune learning rate, check gradients). **Overfitting** =
> model memorizes training data, generalizes poorly (fix: dropout, data
> augmentation, better regularization). We'll see both on a toy regression
> problem by fitting polynomials of different degree.

### TODO B4.1
Implement **dropout**'s forward pass: at training time, randomly zero out a
fraction `p` of activations and scale the rest by `1/(1-p)` (inverted dropout,
so no rescaling is needed at test time).

## B5 · Convolution for CNNs — filters and feature maps

> *Lecture recap:* a CNN **filter** slides over the image; **Image + Filter →
> Feature Map**. A filter may learn to respond to edges, corners, textures,
> shapes, or object parts. This is the *same* convolution operation from Part
> A — but now the filter's weights are **learned**, not hand-designed.

## B6 · Stride and padding

> *Lecture recap:* **Stride** controls how many pixels the filter moves each
> step — stride 1 visits every position; stride 2 skips every other position,
> halving (roughly) the output size and speeding up computation. **Padding**
> adds a border of pixels (usually zeros) around the image before convolving —
> without it, corner/edge pixels get visited far fewer times than center
> pixels, so information near the border is under-represented in the output.

### TODO B6.1
Visualize the border-attention problem padding solves: for a 3×3 image and a
2×2 filter with stride 1, count **how many times each pixel is visited** by
the sliding window (no padding). Corners should be visited once; the center
more.

## B7 · Pooling and downsampling

> *Lecture recap:* pooling lowers spatial dimension by taking the mean or max
> over small windows — reducing computation while distilling the most salient
> (max pooling) or averaged (average pooling) information. Unlike a
> convolution filter, pooling has **no learned weights** — it's a fixed window
> operation, typically with stride equal to the window size (non-overlapping).

### TODO B7.1
Apply your `max_pool2d` and `avg_pool2d` to the real `gray` image with a
larger window (e.g. 8×8) and see the "blurry, lower-resolution but still
recognizable" effect the lecture describes.

## B8 · CNN architecture hyperparameters, in PyTorch

> *Lecture recap:* a convolutional layer's key hyperparameters are **kernel
> size**, **number of kernels** (= number of output channels / feature maps),
> **stride**, and **padding**. Let's confirm our from-scratch output-size
> formula against a real `nn.Conv2d` layer, and build a tiny CNN block
> (conv → batchnorm → relu → maxpool) to see how the pieces from this notebook
> compose in practice.

## B9 · Data augmentation

> *Lecture recap:* most CV tasks benefit from more data. **Data augmentation**
> creates realistic variations of existing images so a model generalizes
> better and overfits less. Common techniques: **horizontal flip** (valid
> whenever mirroring preserves the label — a mirrored cat is still a cat),
> **random cropping** (works as long as crops are a reasonably large subset of
> the image), and **color shifting** (perturbing R/G/B channels to simulate
> lighting/camera variation).

### TODO B9.1
Build an `augment_batch(image, n)` function that returns `n` randomly
augmented versions of `image`, each with a random combination of flip
(50% chance), crop, and a small random color shift. Display a 2×3 grid of
results.

## B10 · A note on transfer learning

> *Lecture recap:* rather than training a CNN from scratch, **transfer
> learning** starts from a model pretrained on a large dataset (e.g.
> ImageNet), **freezes** most of its layers (keeping their learned filters),
> and **fine-tunes** only the last few layers on your smaller, task-specific
> dataset. This is why the filters/feature-maps intuition from Section B5
> matters — a pretrained network's early filters already detect edges,
> textures, and colors that are useful for almost *any* vision task.

---
## Wrap-up

You've now built, by hand, every core piece from the lecture:

- **Classical CV:** pixels-as-numbers, color spaces (RGB/HSV), thresholding,
  morphology, convolution, Sobel, Canny, geometric transforms, histogram
  equalization.
- **Neural nets:** a single neuron, an MLP forward pass, activation functions,
  why initialization/batchnorm/residuals stabilize gradients, and
  overfitting/underfitting with dropout as a fix.
- **CNNs:** convolution as a *learned* filter, stride, padding, pooling, and
  how they compose into a real `nn.Conv2d` block.
- **Data augmentation & transfer learning:** the practical techniques used to
  get more mileage out of limited labeled data.

Next step: put these blocks together into an actual trainable CNN classifier
on a real dataset (Part 2 territory).
