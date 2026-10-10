# Research brief — context for an AI reviewer

Paste this before asking for a review of any section of the manuscript. It contains
the full protocol and every headline number, so that feedback can be checked against
what was actually done.

**Note for the reviewer:** before claiming that two numbers are inconsistent, please
recompute them from the confusion counts given in §6 and state your arithmetic. A
previous review reported an F1 mismatch that did not exist.

---

## 1. One-paragraph summary

We compare two CNNs (ResNet50, EfficientNet-B0) and a Vision Transformer (ViT-B/16)
on binary endometriosis classification in laparoscopic still frames, under a
leakage-free, surgery-grouped 5-fold cross-validation. Beyond accuracy, we ask
whether each model's saliency map actually lands on the lesion, by scoring Grad-CAM
(and, for the transformer, attention rollout) against expert pixel masks and against
explicit chance-level references. The headline finding is a dissociation:
classification performance does **not** separate the three architectures once
between-surgery variability is accounted for, while explanation quality separates
them decisively.

## 2. Research question and claimed contributions

**Question.** If three architectures classify endometriosis about equally well, can
anything else distinguish them — and specifically, do their explanations point at
the lesion?

**Contributions.**
1. A leakage-free, surgery-grouped benchmark on a balanced GLENDA subset, with the
   partitioning checked programmatically and released.
2. A comparison under identical training and evaluation protocols, with uncertainty
   quantified at the surgery level rather than the frame level.
3. A quantitative explanation evaluation against expert masks **with chance-level
   and centre-prior references**, which most saliency papers omit; without them an
   IoU figure cannot be interpreted.
4. Evidence that a single train/test split can reverse the apparent ranking of
   models, demonstrated on our own data.

## 3. Dataset

- **GLENDA v1.5**, public, CC BY-NC, de-identified; no additional ethics approval.
- 373 pathology frames, each with an expert pixel-level lesion mask; 13,438
  no-pathology frames (raw ratio ≈ 1:36).
- **Balanced subset used:** all 373 pathology frames + 373 no-pathology frames
  sampled round-robin across the 27 no-pathology recordings (avoids near-duplicate
  consecutive frames and single-recording dominance). Seed 42.
- Total 746 frames in **129 groups**. A group is a surgery: a case id (`c_<id>`) for
  pathology frames, a recording-segment id for no-pathology frames.
- Lesions occupy **6.88%** of a frame on average (range 0.43–34.47%).

## 4. Protocol

- `StratifiedGroupKFold`, 5 folds, whole groups kept together; the training portion
  of each fold is split again group-wise for early stopping. Test portions are never
  used for model selection.
- Asserted per fold: no group in two partitions; the five test portions tile the
  dataset exactly once.
- Because groups are indivisible, fold composition varies: test portions hold
  126–167 frames and pathology prevalence ranges **25.7%–63.9%**. This is why
  ROC-AUC, not accuracy, is the primary metric.
- A **preliminary single case-wise 70/15/15 split** (528/111/107 frames) was run
  first; it is reported only as a comparison (see §7).

## 5. Models and training

| | ResNet50 | EfficientNet-B0 | ViT-B/16 |
|---|---|---|---|
| Weights | IMAGENET1K_V2 | IMAGENET1K_V1 | IMAGENET1K_V1 |
| Parameters | 25.6M | 5.3M | 86M |
| Learning rate | 1e-4 | 1e-4 | 2e-5 |

Full fine-tuning (nothing frozen), head replaced by a 2-unit linear layer. Input
224×224, ImageNet normalisation. Augmentation on train only: horizontal flip,
rotation ±15°, brightness and contrast jitter 0.2. Adam, batch 32, ≤30 epochs, early
stopping on validation ROC-AUC with patience 6, best-epoch weights restored. Seed 42
offset by fold index; cuDNN restricted to deterministic algorithms. Single Tesla T4
(Google Colab). 15 fits total (3 architectures × 5 folds).

## 6. Classification results

Positive class = endometriosis.

**Per-fold mean ± SD (ROC-AUC):** ResNet50 0.922 ± 0.066 · EfficientNet-B0
0.922 ± 0.058 · ViT-B/16 0.915 ± 0.054.

**Pooled out-of-fold** (all 746 frames, each predicted by the model blind to its
surgery; 95% surgery-level bootstrap intervals):

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| ResNet50 | 0.784 | 0.718 | 0.936 | 0.813 | 0.898 [0.839–0.944] |
| EfficientNet-B0 | 0.798 | 0.748 | 0.898 | 0.816 | 0.913 [0.869–0.950] |
| ViT-B/16 | 0.764 | 0.700 | 0.925 | 0.797 | 0.859 [0.782–0.923] |

**Confusion counts** (373 frames per class): ResNet50 TP 349, FN 24, FP 137, TN 236 ·
EfficientNet-B0 TP 335, FN 38, FP 113, TN 260 · ViT-B/16 TP 345, FN 28, FP 148,
TN 225.

**Pairwise tests** (pooled predictions, surgery-level resampling, Holm-corrected
within the family of three, two-sided):

| A vs B | Only A | Only B | McNemar *p* | ΔAUC [95% CI] | *p* |
|---|---|---|---|---|---|
| ResNet50 vs EfficientNet-B0 | 48 | 58 | 0.38 | −0.015 [−0.052, +0.023] | 0.45 |
| ResNet50 vs ViT-B/16 | 51 | 36 | 0.27 | +0.040 [+0.002, +0.084] | 0.074 |
| EfficientNet-B0 vs ViT-B/16 | 72 | 47 | 0.082 | +0.054 [+0.007, +0.110] | 0.054 |

**No pairwise classification difference is significant.**

## 7. Statistical methodology (important)

- 10,000 bootstrap resamples, percentile intervals, all tests two-sided, Holm
  correction within each family of three comparisons.
- **The resampling unit is the surgery, not the frame.** Frames from one recording
  share patient, camera and illumination, so frame-level resampling treats repeated
  views as independent evidence. Pathology and no-pathology groups are drawn
  separately so both classes appear in every resample.
- Effect of this choice: surgery-level intervals are **2.0–2.6× wider**. Under
  frame-level resampling both comparisons against ViT-B/16 would have been reported
  as significant (*p* < 0.001). They are not.
- **Split sensitivity.** The preliminary single split gave AUCs of 0.915 (ResNet50),
  0.860 (ViT-B/16), 0.833 (EfficientNet-B0) and would have identified
  EfficientNet-B0 as clearly the weakest. Cross-validation does not reproduce that:
  EfficientNet-B0 reaches 0.913 pooled, and its single-split value falls below all
  five of its fold values.

## 8. Explanation evaluation

Grad-CAM at `layer4[-1]` (ResNet50), `features[-1]` (EfficientNet-B0), and the last
encoder block's first layer-norm with a 14×14 token reshape (ViT-B/16). ViT-B/16 is
*additionally* explained with **attention rollout** (Abnar & Zuidema 2020), to rule
out the objection that Grad-CAM is unfair to a transformer. Maps are normalised to
[0, 1], upsampled to 224×224 and thresholded at 0.5.

Measures: **IoU** with the binarised mask; **Pointing Game** (is the map's maximum
inside the mask); **Coverage** (share of map intensity inside the mask).

**References against which the numbers are interpreted:**
- An uninformative map scores the mean mask area, 0.069, on Pointing Game and
  Coverage.
- For IoU, each method is compared against a random region of *exactly the area that
  method claimed*; expected IoU = *pm*/(*p*+*m*−*pm*), verified by simulation.
- A **centred Gaussian** (σ = 0.25×224), thresholded identically, encodes the "lesions
  are usually mid-frame" prior without looking at the image. IoU 0.151, Pointing
  0.302, Coverage 0.123.

**Results** (95% surgery-level intervals):

| | IoU | Pointing Game | Coverage |
|---|---|---|---|
| EfficientNet-B0 | 0.238 [0.178–0.293] | 0.453 [0.298–0.590] | 0.212 [0.156–0.268] |
| ResNet50 | 0.148 [0.103–0.192] | 0.264 [0.146–0.379] | 0.165 [0.116–0.217] |
| ViT-B/16, Grad-CAM | 0.032 [0.016–0.050] | 0.019 [0.000–0.056] | 0.069 [0.039–0.108] |
| ViT-B/16, rollout | 0.028 [0.015–0.044] | 0.038 [0.000–0.106] | 0.069 [0.044–0.100] |

**Comparisons** (surgery-level, Holm-corrected): EfficientNet-B0 − ResNet50
+0.090 [+0.038, +0.143], *p* = 0.0006 · ResNet50 − ViT-B/16 +0.117, *p* = 0.0006 ·
EfficientNet-B0 − ViT-B/16 +0.206, *p* = 0.0006 · Grad-CAM − rollout for ViT-B/16
+0.004, *p* = 0.80.

Against the references: ResNet50 beats area-matched random (+0.112, *p* = 0.0008)
but is **indistinguishable from the centre prior** (*p* = 1.00). EfficientNet-B0
beats both (random +0.200, *p* = 0.0008; centre prior +0.087, *p* = 0.0016).
ViT-B/16 is **indistinguishable from random** (*p* = 0.36) and significantly *worse*
than the centre prior (−0.119, *p* = 0.0008); attention rollout behaves the same.
ViT-B/16's Coverage of 0.069 equals the mean lesion area to three decimals.

## 9. Error analysis

Of the 61 surgeries contributing ≥5 frames, the **twelve with the highest error rate
are all no-pathology recordings**; in the worst, every model misclassifies >85% of
frames. The dominant failure mode is a systematic false positive on healthy tissue
in particular recordings, not missed lesions. Since folds are grouped by surgery
these frames were never trained on, so this is not memorization; we do not claim to
know what distinguishes those recordings.

## 10. Known limitations (already acknowledged — no need to re-report)

- One dataset, 129 surgeries, 746 frames. Statistical power is limited, which is
  why the classification comparisons are inconclusive.
- Still frames, not video; a surgeon uses motion and manipulation.
- Binary only: no lesion type, stage or severity.
- No calibration analysis, no decision-curve/utility analysis, no external
  validation, no clinical evaluation. We make no claim of clinical readiness.
- The numbers above come from a run in which saliency maps were scored on the 53
  pathology frames of the preliminary split, using that split's models. A rerun
  scoring every fold's model on all 373 pathology frames is in progress; the
  protocol is unchanged.

## 11. Status and target

Methodology and Results are drafted; Related Work, Discussion, Limitations,
Conclusion and Abstract are not yet written. Target venue: *Biomedical Signal
Processing and Control* (Elsevier), with an arXiv preprint first. Code, per-image
predictions, saliency scores and all figure scripts:
`github.com/Mozahid-DIU/Endometriosis`.

## 12. What feedback is most useful

1. Claims that outrun the evidence, especially any implied clinical utility.
2. Statistical errors: resampling unit, multiplicity, interval interpretation.
3. Anything a reviewer of a biomedical imaging journal would demand that is missing.
4. Places where a stated number and a stated conclusion do not match — with your
   arithmetic shown.
