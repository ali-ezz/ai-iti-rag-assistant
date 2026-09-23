# LAB-05-Object-Detection

Source notebook: LAB-05-Object-Detection.ipynb

# Computer Vision — Part 3 · Practical Notebook — ✅ SOLVED VERSION

# Computer Vision — Part 3 · Practical Notebook


**Companion lab for the lecture: Object Detection (Bounding boxes · Classes · Confidence scores · Non-Max Suppression · IoU · Precision/Recall/mAP · Two-stage detectors) → Face Recognition (Siamese networks · Triplet loss)**

---

This is the **applications** notebook — three of the most important things
classical and modern computer vision is *used* for in the real world.

**Practical philosophy.** We use the tools people actually deploy:
- For **object detection** we use **Ultralytics YOLOv8** (the modern YOLO) for
  inference and **torchvision's Faster R-CNN** for the two-stage comparison.
  Nobody implements YOLO or Faster R-CNN from scratch in a real job — they load
  pretrained weights and use them.
- We **hand-implement IoU and NMS** because they're short, used everywhere, and
  the single best way to *understand* what detection libraries do under the hood.
- For **face recognition** we build a working **Siamese network with triplet
  loss** on a small image-pairs problem so you see the lecture's algorithm
  actually learn. We also show the production approach: use a pretrained
  embedder.
- For **style transfer** we implement the **Gatys et al.** algorithm from the
  lecture — pretrained VGG, content + Gram-matrix style loss, optimize the
  pixels of an image. It's about 50 lines of real code.

> **How to run this:** Google Colab with a GPU is strongly recommended
> (`Runtime → Change runtime type → GPU`) — particularly for Sections 10 and 12
> which actually train/optimize. CPU works for everything else but Section 12
> will take several minutes per run.

## 1 · Setup & device

Standard imports for this part.

## 2 · Intersection over Union (IoU) — from scratch

> *Lecture recap:* **IoU = area of intersection ÷ area of union** of two
> bounding boxes. It's how we judge whether a predicted box matches a
> ground-truth box. By convention a prediction is considered **correct if
> IoU ≥ 0.5** (and the class is right). Perfect overlap → IoU = 1; no overlap →
> IoU = 0.

This is a 10-line function that appears inside *every* object detection system —
training (assigning anchors to ground truth), evaluation (mAP), and inference
(NMS, next section). Let's write it.

**Box convention.** We use **`[x1, y1, x2, y2]`** — the top-left and
bottom-right corners. This is the standard in `torchvision`, `albumentations`,
and most modern code. (The lecture's `(bx, by, bh, bw)` — centre + size — is
YOLO's *output* format; we'll convert in Section 6.)

Now confirm our implementation matches `torchvision.ops.box_iou` — that's
the function every PyTorch-based detector actually uses internally:

### TODO 2.1
The lecture mentions "by convention an answer is correct if IoU ≥ 0.5". Write a
function `is_correct(pred_box, gt_box, threshold=0.5)` that returns `True` only
if both **IoU exceeds the threshold** *and* the class labels match. The function
should accept `(box, class_id)` pairs.

Then test it on three cases the instructor cell will define for you.

## 3 · Non-Max Suppression (NMS) — from scratch

> *Lecture recap:* a detector typically fires multiple times around the same
> object — many grid cells/anchors all think they see the same car. **Non-Max
> Suppression** cleans up these duplicates:
> 1. **Drop** any box with confidence below a threshold (e.g. `score < 0.6`).
> 2. **Pick** the highest-scoring remaining box, **keep it**, output it.
> 3. **Drop** every other remaining box that overlaps it heavily
>    (IoU ≥ some threshold, e.g. 0.5).
> 4. Repeat steps 2–3 until no boxes remain.
> For multi-class output, run NMS *independently per class*.

This runs at the end of every detector (YOLO, Faster R-CNN, RetinaNet, DETR's
variants…). Let's implement it.

And confirm against `torchvision.ops.nms` — the production version that's
GPU-accelerated and used inside every PyTorch detector:

### TODO 3.1
Sweep the **IoU threshold** to see its effect. Run NMS with `iou_thresh = 0.3`,
`0.5`, and `0.9` on the same `boxes`/`scores` as above. Print how many boxes
each one keeps and explain (one line each):
- Why does a *low* IoU threshold keep *fewer* boxes?
- Why does a *high* IoU threshold keep *more* boxes?
- What disaster could a too-high IoU threshold cause in practice?

---
## 3B · Bounding boxes, classes, and confidence scores — the detector output format

> *Lecture recap:* every object detector, regardless of architecture, outputs
> the same three things per detection: a **bounding box** (typically
> `[x1, y1, x2, y2]` — top-left and bottom-right corners, or sometimes
> `[cx, cy, w, h]` — center point and size), a **class label** (which of N
> categories this is), and a **confidence score** (how sure the model is).
> Understanding this raw output format — before any NMS or evaluation — is the
> foundation everything else in this notebook builds on.

### TODO 3B.1 — Box format conversions

Different libraries use different box formats. You will hit this constantly:
some codebases use corner format `[x1, y1, x2, y2]`, others use center format
`[cx, cy, w, h]`, and getting the conversion wrong silently produces boxes in
the wrong place (a very common real bug).

Implement both conversions.

### TODO 3B.2 — Decoding a raw detector output tensor

Real detectors don't hand you a friendly list of dicts — they output a raw
tensor. A typical single-image output might be shape `(num_boxes, 6)`, where
each row is `[x1, y1, x2, y2, confidence, class_id]`. Practice **parsing**
this into something usable: filter by confidence, then group by class.

### TODO 3B.3 — Class-aware Non-Max Suppression

Real detectors run NMS **per class**, not globally — a "person" box and a
"car" box can legitimately overlap heavily (a pedestrian standing next to a
car) and both should be kept. NMS should only suppress duplicates *within the
same class*. Using the `nms()` function from Section 3 and the `by_class` dict
from 3B.2, apply NMS separately to each class group.

### TODO 3B.4 — Written: why class-aware NMS matters

Suppose a detector ran **global** NMS (ignoring class labels) instead of
per-class NMS on the scene above, with `iou_thresh=0.4`. Imagine a "person"
box and a "bicycle" box that overlap with IoU = 0.6 (a person riding a
bicycle — a very common real scene). Answer in a comment:

1. What would global NMS incorrectly do in this situation?
2. Name one more real-world scenario where two *different* object classes
   commonly have high-IoU overlapping boxes.

### TODO 3B.5 — Soft-NMS (a gentler alternative)

Standard NMS **completely discards** any box with IoU above the threshold —
an all-or-nothing decision. **Soft-NMS** instead *decays* the score of
overlapping boxes proportionally to their IoU with the kept box, rather than
zeroing them out. This helps in scenes with genuinely overlapping objects of
the *same* class (e.g. a crowd of people) where standard NMS is too
aggressive and deletes correct detections.

Implement the (linear) soft-NMS scoring rule:
`new_score = score * (1 - iou)` for every box that overlaps the just-kept box,
then re-sort and repeat.

## 4 · The R-CNN family — from Selective Search to Faster R-CNN

> *Lecture recap:* object detection didn't start with a single end-to-end
> network — it evolved through three "two-stage" designs, each fixing the
> previous one's main bottleneck:
> - **R-CNN** (2014): run **Selective Search** (a classical, non-learned
>   algorithm) on the image to propose ~2000 candidate boxes, then run a
>   **separate CNN forward pass on each of the ~2000 crops**. Accurate but
>   painfully slow — ~2000 forward passes per image.
> - **Fast R-CNN**: run the CNN **once** on the whole image to get a shared
>   feature map, then use **RoI Pooling** to crop a fixed-size feature for
>   each of the ~2000 proposed regions out of that *single* feature map.
>   Selective Search is still used to generate proposals (still a bottleneck),
>   but the CNN itself only runs once.
> - **Faster R-CNN**: replace Selective Search entirely with a small learned
>   **Region Proposal Network (RPN)** that reuses the *same* shared feature
>   map to generate proposals. Now the whole pipeline — proposals AND
>   classification — is one end-to-end trainable network. This is the
>   standard "two-stage" detector architecture used in production today.

The honest practical reality: **nobody implements Faster R-CNN's RPN and RoI
pooling from scratch in a lab setting** — it's intricate low-level machinery.
`torchvision.models.detection` ships a pretrained Faster R-CNN with one line,
which is what real projects use. Let's load it and inspect its predictions
directly.

---
## 4B · LAB — R-CNN vs Fast R-CNN vs Faster R-CNN, hands-on

> *Why this lab exists:* Section 4 loaded a pretrained Faster R-CNN and used
> it. That's the right thing to do in production, but it hides *why* Faster
> R-CNN looks the way it does. This lab makes the three-stage evolution
> concrete: you will (1) reason about the bottleneck each design fixed,
> (2) implement the one piece of machinery that's actually small enough to
> code by hand — **RoI Pooling** — and (3) empirically measure the speed
> difference between "run a CNN per region" (the R-CNN way) and "run a CNN
> once, share the features" (the Fast/Faster R-CNN way) on your own machine.

**Recap table — fix in your own words as you go:**

| | R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) |
|---|---|---|---|
| Region proposals | Selective Search (classical CV, ~2000 boxes) | Selective Search (still classical) | **Region Proposal Network** (learned, shares backbone) |
| CNN forward passes per image | ~2000 (one per region crop) | **1** (whole image), then crop the *feature map* | **1** (whole image), features reused for proposals too |
| How each region gets a fixed-size feature | resize the raw image crop to 224×224 | **RoI Pooling** on the shared feature map | RoI Pooling (or RoI Align) on the shared feature map |
| End-to-end trainable? | No (separate CNN, SVMs, box regressors) | Mostly (still relies on external proposals) | **Yes** — proposals + classification in one network |
| Rough speed | ~47s/image (CPU, 2014 hardware) | ~2s/image | ~0.2s/image |

### TODO 4B.1 — Name the bottleneck each design fixes

Without looking anything up, answer in one sentence each:

1. What is the single biggest reason **R-CNN** is slow?
2. **Fast R-CNN** fixes that — but what bottleneck is still left in Fast R-CNN?
3. **Faster R-CNN** fixes *that* — what does the RPN replace, and why does
   this finally make the whole model trainable end-to-end?

### TODO 4B.2 — Implement the core of RoI Pooling

> *Lecture recap:* once you have ONE shared feature map for the whole image
> (the Fast/Faster R-CNN trick), each region proposal covers a *different
> sized* patch of that feature map. But the classifier head needs a
> **fixed-size** input (e.g. 7×7). **RoI Pooling** solves this: given a
> region `[x1, y1, x2, y2]` on the feature map, divide it into an
> `output_size x output_size` grid of bins, and **max-pool** each bin
> independently. The result is always `output_size x output_size`, no
> matter how big or small the original region was.

The setup (clipping the box, allocating the output) is done for you. You
write the **inner loop**: for each `(i, j)` cell of the output grid, work
out that bin's pixel extent on the feature map and max-pool over it.

### TODO 4B.2b — Prove it: fixed output size regardless of input size

Write two boxes of very different sizes on the fake feature map below —
one small (a few pixels), one large (most of the map) — then run `roi_pool`
on both and confirm the output shapes are identical.

### TODO 4B.3 — Measure the actual speedup: "one CNN per region" vs "one CNN, shared features"

> *Lecture recap:* R-CNN's slowness isn't theoretical — it comes from
> literally re-running a full CNN forward pass for every one of ~2000
> region proposals. Fast/Faster R-CNN instead run the CNN backbone **once**
> and reuse the resulting feature map for every region.

The setup (loading the backbone, building fake region crops) is done for
you. You write **both timed sections**: the R-CNN-style loop that runs the
backbone once per region, and the Fast-R-CNN-style single call that runs it
once on the whole image. Then compute and print the speedup ratio yourself.

### TODO 4B.4 — Written: connecting the measurement back to the architecture

Using your result from 4B.3:

1. The speedup you measured only accounts for the *backbone* forward passes.
   In real R-CNN, Selective Search itself also costs ~2 seconds per image
   **on top of** the ~2000 forward passes. Does this change which bottleneck
   you'd optimize first if you were stuck maintaining R-CNN in 2014? Why?
2. Faster R-CNN's RPN reuses the *same* shared feature map that the final
   classifier uses. Given what you now know about RoI Pooling, explain in
   one or two sentences why this reuse is only possible **after** Fast
   R-CNN's "run the CNN once" change — i.e. why couldn't you bolt an RPN
   onto original R-CNN?
3. Look back at your `roi_pool` implementation. What happens to a very
   **small** region (e.g. 2×2 feature-map pixels) when pooled up to a 7×7
   output? Is any information invented, or just repeated/coarse? Why might
   this matter for detecting small objects?

## 5 · mean Average Precision (mAP) — the standard detection metric

> *Lecture recap:* a detector outputs many boxes per image with confidence
> scores. To evaluate it, for each predicted box we check: **is it correct?**
> (class right AND IoU ≥ 0.5 with an unmatched GT). **Each GT can only be
> matched once** — extra overlapping predictions become false positives.
> Sliding the confidence threshold from high to low traces out a
> **Precision–Recall curve**. **Average Precision (AP)** is roughly the area
> under that curve, computed per class; **mAP** is the mean of AP across all
> classes.

Real codebases use `pycocotools` or `torchmetrics`; we'll compute AP **from
scratch** on a tiny toy set so the four steps (match → sort by confidence →
precision-recall curve → area under it) are unambiguous.

### TODO 5.1
A perfect detector achieves **AP = 1.0**. A random detector approaches **AP ≈ 0**.
Verify both:

1. Construct a "perfect" set of predictions: one prediction per ground truth,
   each one exactly matching (same box), all with confidence 1.0. Compute the AP.
2. Construct a "random/bad" set: same number of predictions, all in wrong
   locations (no overlap with any GT). Compute the AP.

---
## 5B · Evaluating detectors — precision, recall, IoU thresholds, and mAP in depth

> *Lecture recap:* a single accuracy number doesn't work for detection the way
> it does for classification, because a detector can be wrong in *several
> different ways at once* on the same image: it can miss an object entirely
> (**false negative**), hallucinate an object that isn't there (**false
> positive**), or find the right object with a sloppy box (**low IoU**).
> Precision, recall, and mAP each capture a different piece of this.

### TODO 5B.1 — Precision vs recall, defined from first principles

Before computing anything, write the definitions in your own words, then
verify them against a tiny concrete scenario the instructor cell sets up.

- **Precision** = "of everything I *predicted*, how much was actually right?"
- **Recall**    = "of everything that *actually exists*, how much did I find?"

### TODO 5B.2 — The precision/recall trade-off via confidence threshold

Precision and recall trade off against each other as you change the
confidence threshold used to accept a detection:
- **Low threshold** → you accept more detections → recall goes up (you find
  more true objects) but precision goes down (more junk gets through too).
- **High threshold** → you accept fewer, more confident detections →
  precision goes up but recall goes down (you start missing real objects).

Using `car_predictions` from 5B.1, sweep the confidence threshold and observe
this trade-off directly (without re-running the AP machinery — just filter
predictions by score first).

### TODO 5B.3 — How the IoU threshold changes everything

So far we always used `iou_thresh=0.5` to decide if a box "counts" as
correct. This threshold is itself a design choice. **COCO's strict "AP@[0.5:0.95]"
metric** averages AP over IoU thresholds from 0.5 to 0.95 in steps of 0.05,
specifically because a detector's score depends heavily on which threshold
you pick.

Recompute AP for `car_predictions` at several IoU thresholds and observe how
AP changes as the localization requirement gets stricter.

### TODO 5B.4 — mAP across multiple classes

Everything so far computed AP for **one** class. mAP averages AP across
**all** classes. Build a tiny 2-class scenario (cars and pedestrians) and
compute the overall mAP.
