# LAB-06-YOLO

Source notebook: LAB-06-YOLO.ipynb

# YOLO Object Detection — Mini-Project Exercise Notebook — ✅ SOLVED (Colab-ready)
### Inference → Custom Dataset → Fine-Tuning → Evaluation → Failure Analysis → Demo

**How this notebook works:**
- Setup and pretrained inference (Sections 1–2) are **solved** — you need a working environment
  and a baseline before fine-tuning makes sense, so these are given as reference.
- From Section 3 onward (dataset preparation, fine-tuning, evaluation, failure-case analysis,
  and the demo), you'll see **markdown instructions and empty code cells only** — no
  scaffolding. You write the code yourself.
- Every section ends with **Question(s)** — short written-answer questions.

**Mini-project deliverables covered by this notebook:**
1. Run inference using a pretrained YOLO model
2. Prepare a small object-detection dataset
3. Fine-tune a YOLO model on that dataset
4. Evaluate detection results and inspect common failure cases
5. Build a simple detection demo (image / video / webcam)
6. *(Optional)* Complete a Kaggle computer-vision notebook exercise

**Before you start:** Runtime → Change runtime type → **GPU** (T4 is enough). Run cells top to
bottom — later exercises depend on files and variables created earlier.

> ✅ **SOLVED VERSION:** all exercises 3.1–7.3 are fully implemented below with real,
> runnable Colab code + all Questions 1–11 answered. For the Roboflow step you only need
> to paste your free API key in ONE line (Exercise 3.1). If you run without a key, the
> notebook automatically falls back to `coco128` (a real 128-image YOLO dataset that
> auto-downloads, no key needed) so every later cell still trains, evaluates and demos
> end-to-end for submission.

## 1. Setup (solved — run as-is)

We use **Ultralytics YOLO**, the standard actively-maintained library wrapping recent YOLO
versions (v8+) behind a simple Python API — the same library used for both inference and
training in this notebook.

## 2. Deliverable 1 — Run Inference Using a Pretrained YOLO Model (solved — run as-is)

Before fine-tuning anything, we establish a baseline: how does an out-of-the-box, COCO-pretrained
YOLO model perform on a generic image? `yolov8n.pt` ("nano") is the smallest, fastest YOLOv8
checkpoint, pretrained on COCO's 80 everyday object classes.

**Question 1 — SOLVED.** The pretrained model above was trained on COCO's 80 classes. If you
fine-tune it in Section 4 on a dataset with, say, only 2 new classes, will the fine-tuned model
still be able to detect the original 80 COCO classes afterward? Why or why not? (Hint: think
about what happens to the final detection head during fine-tuning.)

> **Answer:** No — not in the standard Ultralytics fine-tuning workflow. `YOLO("yolov8n.pt")` keeps
> the backbone/neck weights (useful generic features: edges, textures, shapes) but **replaces/
> re-initialises the final detection head** so it now outputs 2 classes instead of 80 (different
> number of filters, different class mapping in `data.yaml`). Training then optimises the whole
> network (or at least the head) only on the new 2-class loss, so weights drift toward the new
> task (catastrophic forgetting). At inference the model literally has no logits for the old 80
> COCO ids (`model.names` will show your 2 new names). If you need both old + new classes you must
> either (a) train jointly on combined data, or (b) keep two separate models.

## 3. Deliverable 2 — Prepare a Small Object-Detection Dataset

For guided fine-tuning, we use **Roboflow Universe** — a public library of thousands of
ready-made, small, permissively-licensed object-detection datasets that export directly in the
YOLO annotation format (a `data.yaml` file plus `images/` and `labels/` folders), so there's no
manual annotation-format conversion needed.

**🔧 Exercise 3.1 — Choose and download a dataset.**
1. Go to [Roboflow Universe](https://universe.roboflow.com) and search for a small,
   simple dataset — good choices for a first fine-tuning exercise are things like a
   **"traffic signs"**, **"PPE / hard hat detection"**, or **"fruit detection"** dataset with
   only a handful of classes (2-5) and a few hundred to low-thousands of images. Avoid datasets
   with more than ~10 classes or tens of thousands of images for this exercise — the goal is a
   fast fine-tuning loop, not a state-of-the-art result.
2. On the dataset's Universe page, use the **"Download Dataset"** option, choose the
   **YOLOv8** export format, and select **"show download code"** to get a snippet using the
   `roboflow` Python package (an API key is required — Roboflow's free tier is sufficient;
   sign up at roboflow.com if you don't have an account).
3. Paste and run that download snippet below. It should create a local folder containing
   `data.yaml`, `train/`, `valid/`, and `test/` subfolders.

**🔧 Exercise 3.2 — Inspect the dataset.**
- Print the contents of `data.yaml` (it lists the class names and folder paths YOLO will use).
- Count how many images are in the train/valid/test splits.
- Display a grid of at least 4 sample images **with their ground-truth bounding boxes drawn**
  (the label `.txt` files use YOLO format: `class_id x_center y_center width height`, all
  normalized 0–1 — you'll need to convert these back to pixel coordinates to draw them).

**Question 2 — SOLVED.** How many classes does your chosen dataset have, and how many training
images total? Given what you saw in the CV-architectures notebook about CIFAR-10 needing
thousands of images per class to train a classifier from scratch, why is it more realistic to
fine-tune (rather than train from scratch) a detector on a dataset this small?

> **Answer (update the two numbers from the Exercise 3.2 output above):** e.g. Roboflow PPE/hard-hat
> style sets typically have **2–5 classes and a few hundred to ~1–3k train images** (coco128 fallback:
> 80 COCO names but only 128 images total — even smaller). Training a detector from random weights on
> that little data fails because detection must learn *both* localisation (box regression over anchors)
> *and* classification from very few examples per class/pose/lighting — it overfits immediately (cf.
> CIFAR-10 needing ~5k images/class even for plain classification). **Fine-tuning** reuses COCO-pretrained
> backbone+neck features (edges/textures/parts already learned from 118k COCO images) and only adapts
> the head + lightly tunes features, so a few hundred images are enough to get a useful mAP.

**Question 3 — SOLVED.** Look at a few of the sample images with ground-truth boxes you plotted. Are
the boxes tight around each object, or noticeably loose/misaligned? Why does the quality of
ground-truth annotations matter for how well a fine-tuned model can perform, regardless of
model architecture?

> **Answer:** In curated Roboflow sets (and coco128) most boxes are **tight**, but you will usually spot
> 1–2 loose/offset boxes (crowded scenes, occluded objects, ambiguous boundaries like a person's helmet
> strap vs head). Annotation quality is a **ceiling**: the box-regression loss (CIoU/DFL in YOLOv8) and the
> IoU-based matching (TP iff IoU≥0.5) both treat GT as truth. Noisy GT teaches the model noisy targets
> (it learns to predict loose boxes), and at eval time even a perfect prediction scores as FP if the GT it
> is compared against is misaligned. No architecture can beat systematically bad labels — garbage in,
> garbage out.

## 4. Deliverable 3 — Fine-Tune a YOLO Model on Your Dataset

Ultralytics makes fine-tuning straightforward: start from a pretrained checkpoint (transfer
learning, just like the ResNet transfer-learning exercise in the CV-architectures notebook) and
call `.train()` pointing at your dataset's `data.yaml`.

**🔧 Exercise 4.1 — Fine-tune.** Using `YOLO("yolov8n.pt")` as your starting point, call
`.train()` with:
- `data=` the path to your dataset's `data.yaml`
- `epochs=` a small number to start (20–30 is plenty for a small dataset — you can increase
  later if you have time budget)
- `imgsz=640` (YOLO's standard input resolution)
- `batch=16` (reduce if you hit GPU memory errors)
- a `name=` so Ultralytics saves this run under a recognizable folder in `runs/detect/`

Let it run to completion — Ultralytics will print per-epoch loss and validation mAP, and will
save the best checkpoint automatically (`runs/detect/<name>/weights/best.pt`).

**🔧 Exercise 4.2 — Load your fine-tuned model.** Load the saved `best.pt` checkpoint into a new
`YOLO(...)` object, and run inference on 2-3 images from your dataset's **test** split (images
the model did not see during training) to sanity-check it's detecting your custom classes.

**Question 4 — SOLVED.** Ultralytics' training log prints both a training loss and a validation
metric (mAP) after every epoch. If training loss keeps decreasing but validation mAP stops
improving or gets worse after some epoch, what is this a sign of, and what could you do about it
(name at least one concrete fix)?

> **Answer:** Classic **overfitting**: the model memorises training images (loss ↓) but generalises
> worse to unseen val images (mAP plateaus/drops). Concrete fixes (any one): **early stopping**
> (keep `best.pt`, stop when val mAP stalls — already enabled via `patience`), **more augmentation**
> (`mosaic/mixup/hsv` are on by default; increase them), **stronger regularisation** (weight decay,
> dropout), **lower LR / fewer epochs**, or simply **more/diverse training data**.

**Question 5 — SOLVED.** You fine-tuned starting from `yolov8n.pt` rather than training a YOLO
architecture from random weights. Based on what you learned about transfer learning in the
CV-architectures notebook, why is this almost always the right choice for a small,
few-hundred-to-few-thousand-image dataset like yours?

> **Answer:** Same logic as ResNet transfer learning: the COCO-pretrained backbone already encodes
> generic visual primitives (edges, textures, object parts, multi-scale features) learned from 118k
> diverse images. With only hundreds of images you cannot learn those from scratch without severe
> overfitting + very long training. Fine-tuning reuses them and only needs to learn the small
> dataset-specific mapping (new class head + box refinement), so it converges in ~20 epochs on a
> single T4 instead of hundreds of GPU-hours, and reaches far higher mAP.

## 5. Deliverable 4 — Evaluate Detection Results and Inspect Failure Cases

A single mAP number tells you *how well*, on average, but not *where* and *why* a model fails.
Real evaluation means looking at both.

**Exercise 5.1 — Compute metrics.** Call `.val()` on your fine-tuned model against your
dataset's validation/test split. Report `mAP@0.5`, `mAP@0.5:0.95`, mean precision, and mean
recall. Compare these numbers to what the pretrained baseline model would give on the same
images (hint: the baseline can't detect your custom classes at all, since they weren't in COCO —
so what would zero-shot performance even mean here? Explain in your answer to Question 6).

**Exercise 5.2 — Find and display failure cases.** Run inference with your fine-tuned model
across a batch of test-split images and identify:
- At least one **false negative** (an object is visibly present but the model misses it or its
  confidence is very low).
- At least one **false positive** (the model detects something that isn't actually there, or
  detects the wrong class).
- At least one case of **poor localization** (the model finds the right class, but the bounding
  box is noticeably too big, too small, or offset from the true object).

Display each failure case as an image with the predicted boxes drawn, and add a one-line caption
under each explaining which failure type it is.

**Exercise 5.3 — Confidence threshold sweep on a failure case.** Take one of your false
negatives from Exercise 5.2 and re-run inference on that same image at a lower confidence
threshold (e.g. `conf=0.1` instead of the default `0.25`). Does the object get detected now?
What does this tell you about whether the model "doesn't know" the object is there versus
"isn't confident enough" to report it?

**Question 6 — SOLVED.** Explain why comparing your fine-tuned model's mAP directly to the
pretrained baseline's mAP on your custom classes doesn't really make sense as a comparison (see
the hint in Exercise 5.1). What would be a fairer baseline to compare against, if you wanted to
argue that fine-tuning was worthwhile?

> **Answer:** The COCO baseline was never trained on your custom label space (e.g. hard-hat vs vest,
> or coco128 subset quirks) — its head outputs 80 COCO ids, so on your classes it scores ~0 by
> construction (wrong vocabulary, not wrong quality). Comparing mAPs just restates "it learned the new
> task". A fairer baseline is **same-data, no-pretraining** (train YOLO from random init on your
> data), or **frozen-backbone linear probe**, or simply **train/val loss + qualitative before/after**
> on identical images; even better, compare fine-tuned `yolov8n` vs fine-tuned larger `yolov8s/m` or
> vs different epoch/augmentation settings on the SAME custom val split.

**Question 7 — SOLVED.** For each of your three failure cases from Exercise 5.2 (false negative, false
positive, poor localization), suggest one plausible cause specific to your dataset (e.g. too few
training examples of that object's pose/lighting/scale, class imbalance, annotation
inconsistency, object too small in the image, occlusion). You don't need to be certain — a
reasonable, specific hypothesis is enough.

> **Answer:** (adapt wording to your images) **FN:** missed small/dark/occluded instance — e.g. a
> distant hard-hat occupying <2% of pixels at dusk; under-represented scale/lighting in train, so
> confidence falls below 0.25. **FP:** background texture mimics the class — e.g. a round sign/skin-tone
> patch fires the helmet/person head; too few hard negatives of that background in training.
> **Poor localisation:** loose GT in train (annotators drew generous boxes around groups) taught the
> regressor loose boxes, or heavy occlusion truncates the visible part so the box is offset; stricter
> IoU (0.75) exposes this even when IoU≥0.5 passes.

**Question 8 — SOLVED.** Based on your Exercise 5.3 result, if lowering the confidence threshold
recovers a missed detection, is retraining necessary, or could this failure mode be partly
addressed just by how the model is deployed (i.e., what threshold you choose at inference time)?
What's the trade-off of simply lowering the threshold everywhere?

> **Answer:** No retraining strictly needed for THAT image — it is a **deployment/threshold** fix:
> ship with lower `conf` (e.g. 0.10). Trade-off is the classic precision↔recall curve: lower threshold
> raises recall (fewer misses) but lowers precision (more FPs/noise get through, more operator fatigue,
> slower NMS/post-processing). In practice pick the threshold on the VAL PR curve for the operating
> point your application wants (e.g. safety-critical → favour recall; auto-counting → favour precision),
> which is exactly why Exercise 6.1 exposes it as a user slider.

## 6. Deliverable 5 — Build a Simple Detection Demo

**Exercise 6.1 — Image demo (Gradio).** Build a `gr.Interface` that:
- Takes an uploaded image and a confidence-threshold slider as inputs.
- Runs your **fine-tuned** model and returns the annotated image plus a text list of detected
  objects and their confidences (reuse the pattern of `results[0].plot()` from Section 2, but
  point at your fine-tuned model instead of the baseline).
- Launches with `share=True` so it can be tested from a phone or shared with a classmate.

**Exercise 6.2 — Video demo.** Using either a short video you upload (`google.colab.files.upload()`)
or one downloaded via URL, run your fine-tuned model's `.predict(source=..., save=True)` on the
video file and confirm the annotated output video is saved under `runs/detect/`.

**Exercise 6.3 (optional) — Webcam demo.** Colab can't access your webcam directly from
Python, so live webcam capture needs a small JavaScript bridge to grab a frame from your
browser's camera. Write (or adapt) a `take_photo()` helper using `IPython.display.Javascript`
and `google.colab.output.eval_js` that captures one frame and saves it to disk, then run your
fine-tuned model on that captured frame.

**Question 9 — SOLVED.** Your Gradio demo (Exercise 6.1) exposes a confidence-threshold slider
to the end user, rather than hard-coding one value. Based on your failure-case analysis in
Section 5, why is giving the user control over this threshold a reasonable design choice for a
demo, rather than a weakness?

> **Answer:** Because the "best" threshold is application- and image-dependent (Q8 / 5.3): low
> thresholds recover uncertain true objects (fix FNs) at the cost of more FPs; high thresholds give
> clean output but miss hard cases. No single value is optimal for all users/images. A slider makes
> the precision↔recall trade-off *visible and controllable* — exactly what a demo should teach —
> instead of hiding it behind an arbitrary hard-coded 0.25.

**Question 10 — SOLVED.** If this demo needed to run in real time on a live video stream (e.g. a
security camera) rather than single uploaded images, what would you need to check or change
about your current setup (think about the FPS benchmarking you may have done in the earlier
COCO-inference notebook, and about `yolov8n` vs. larger YOLO variants)?

> **Answer:** Check **throughput (FPS) vs latency**: measure `model.predict` FPS at `imgsz=640` on the
> deployment GPU/CPU; `yolov8n` (nano, ~3M params) is the fastest family member and the right default
> for realtime — larger `s/m/l/x` gain mAP but sharply cut FPS and may miss realtime. If too slow:
> lower `imgsz` (e.g. 480/320), raise `conf`/reduce NMS load, batch frames, use TensorRT/ONNX + FP16,
> and stream via `model.predict(source=0, stream=True)` instead of per-image Gradio calls. Also consider
> hardware (T4 vs edge CPU vs Jetson) and resolution/FPS requirements of the camera.

## 7. Optional — Kaggle Computer Vision Notebook Exercise

This section is **optional** and separate from the fine-tuning project above. Rather than
requiring you to join a live, scored Kaggle competition (which may or may not have an active
leaderboard at any given time), this section uses a small, freely downloadable **Kaggle
dataset** so you can practice the Kaggle workflow — API access, downloading data, and producing
a notebook-style submission file — without a competition dependency.

**Exercise 7.1 (optional) — Set up Kaggle API access.**
1. Create a free Kaggle account if you don't have one, go to
   Account Settings → **Create New API Token**, which downloads a `kaggle.json` file.
2. Upload `kaggle.json` to this Colab session (`google.colab.files.upload()`), then run:
   ```
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. Install the Kaggle CLI (`!pip install -q kaggle`) and confirm it works with
   `!kaggle datasets list -s "object detection"`.

**Exercise 7.2 (optional) — Download a small Kaggle CV dataset.** Pick a small image dataset
from Kaggle (search for something like "cats vs dogs" or a small object-detection dataset with
a permissive license), download it with `!kaggle datasets download -d <owner>/<dataset-name>`,
and unzip it.

**Exercise 7.3 (optional) — Run your model and produce a submission-style output.** Run
either your fine-tuned YOLO model (if the dataset has relevant objects) or the pretrained
baseline over the downloaded images, and save the predictions to a `submission.csv` in a
reasonable format (e.g. columns: `image_id, predicted_class, confidence` or, for detection,
`image_id, class, x_min, y_min, x_max, y_max, confidence`) — mirroring the structure Kaggle
competitions typically expect for a submission file, even though there's no leaderboard to
submit to here.

**Question 11 (optional) — SOLVED.** What are the practical differences between the Roboflow
dataset workflow you used in Section 3 (pre-formatted for YOLO, one function call to download)
and the Kaggle dataset workflow here (raw files, you decide the format)? Which is faster to get
started with, and which gives you more control?

> **Answer:** **Roboflow Universe is faster to start:** one `version().download("yolov8")` gives you
> `data.yaml` + `images/` + YOLO-normalised `labels/` in exactly the layout Ultralytics expects, with
> consistent annotation quality and train/val/test splits — zero format wrangling. **Kaggle is raw
> files + freedom:** you get whatever the uploader provided (CSVs, VOC XML, COCO JSON, plain folders,
> mixed licences/sizes) and YOU decide parsing, splits, label conversion, and the `submission.csv`
> schema — more control (any task, any format, any competition metric) at the cost of more glue code
> (unzipping, path handling, format conversion, writing the submission file yourself). For a first
> fine-tune, Roboflow wins on speed; for custom/competition work, Kaggle's flexibility wins.
