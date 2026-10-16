# Methodology — Step 5

**Title:** Comparative Analysis of CNN and Vision Transformer Architectures for Explainable Endometriosis Classification in Laparoscopic Images
**Task:** Binary classification (endometriosis vs no-pathology) on balanced GLENDA + Grad-CAM explainability

> ⚠️ **This is the Step-5 *plan*, written before the experiments were run. It is kept
> as a record of the design decisions, not as a description of what was executed.**
> The executed protocol is in `paper/sections/04_methodology.md`, summarised in
> [research_brief.md](research_brief.md). Where the plan below was later changed, the
> change is flagged inline as **[as executed]**. The main departures: full
> fine-tuning instead of a frozen backbone, different learning rates, and 5-fold
> surgery-grouped cross-validation as the primary evaluation rather than a single
> held-out split.

---

## 1. Dataset Preparation

**Source:** GLENDA v1.5 (already downloaded)
- Pathology (endometriosis): **373 images** — `Glenda_v1.5_classes/.../frames/`
- No-pathology (normal): **13,438 images** — `GLENDA_v1.5_no_pathology/.../frames/`

**Balancing strategy (undersampling — simplest & standard):**
- Keep all 373 endometriosis images
- Sample 373 no-pathology images (fixed seed for reproducibility)
  — **[as executed]** round-robin across the 27 no-pathology recording folders, not
  uniformly at random, so no single recording dominates (see "Diverse no-pathology
  sampling" below)
- **Balanced set = 746 images (373 + 373)**

**⚠️ CRITICAL — Patient/Case-wise split (prevents data leakage):**
GLENDA frames come from videos; multiple frames share the same surgery/case.
Filename encodes the case ID: `c_100_v_(video_3062.mp4)_f_1458.jpg` → case = `c_100`.
- **Never** put frames of the same case in both train and test (would inflate results → reject).
- Extract case ID via regex → group images by case → split with **GroupShuffleSplit / GroupKFold** (scikit-learn).
- No-pathology frames grouped by their **video ID** (`v_2506`) the same way.
  - 🔴 **[as executed]** This is the step that went wrong first. GLENDA ships
    no-pathology frames in **per-segment folders**, so one recording appears under
    several folder names. Grouping by folder splits a recording across folds: in the
    first CV run, 6 of the 20 no-pathology recordings were split into 2–3 groups and 4
    of them landed on both sides of a split, leaking 178 of the 373 no-pathology
    frames and making EfficientNet-B0 look tied-for-best (pooled AUC 0.913 instead of
    0.852). The fix is to **strip the segment suffix** so the group is the recording.
    That gives **122 surgeries** (102 pathology cases + 20 no-pathology recordings)
    over 129 segment folders.
- Frame counts per split will NOT be exactly 70/15/15 (cases have unequal frame counts) — this is correct and expected.

**Split (patient-wise, fixed seed = 42):**
| Split | ~% | Note |
|-------|---|------|
| Train | ~70% | whole cases only |
| Validation | ~15% | whole cases only |
| Test | ~15% | whole cases only, held-out |

> ✅ Report **5-fold grouped cross-validation** (small data → CV gives credible, stable results); grouping keeps the leakage-safe guarantee across folds.
> 🔴 **[as executed]** The single 70/15/15 split above was run first, but it is **not**
> the reported protocol: the test portion is only 107 frames and its ranking of the
> three models did not survive cross-validation. The primary evaluation is
> `StratifiedGroupKFold` with 5 folds over all 746 frames, with the training portion of
> each fold split again group-wise for early stopping, and the pooled out-of-fold
> predictions as the primary estimate. The single split is reported only as a
> sensitivity analysis.

**Diverse no-pathology sampling:** the 13,438 normal frames come from only ~20 videos → sample the 373 normal images **spread across different videos** (not consecutive near-duplicate frames), to avoid redundancy.

---

## 2. Preprocessing & Augmentation

- Resize → **224×224** (standard for pretrained models)
- Normalize with **ImageNet mean/std** (transfer learning requirement)
- **Train-time augmentation** (fights small-data overfitting):
  horizontal flip, rotation (±15°), brightness/contrast jitter
- Val/Test: resize + normalize only (no augmentation)

---

## 3. Models (Transfer Learning — ImageNet pretrained)

**Group 1 — CNN baselines:**
| Model | Why |
|-------|-----|
| ResNet50 | Classic strong baseline; used in prior GLENDA work (comparability) |
| EfficientNet-B0 | Efficient, high accuracy, Colab-friendly |

**Group 2 — Vision Transformer:**
| Model | Why |
|-------|-----|
| ViT-Base/16 (or ViT-Small) | Pure transformer — the "vs" side of the comparison |

*(Optional 4th if time allows: DenseNet121 or MobileViT — only if schedule permits. Not run.)*

**Setup:** freeze backbone → train new classification head → (optional) fine-tune top layers.
- **[as executed]** No freezing: **full fine-tuning** of all three backbones, with the
  head replaced by a 2-unit linear layer. ViT-B/16 was used (not ViT-Small). Weights:
  ResNet50 `IMAGENET1K_V2`, EfficientNet-B0 and ViT-B/16 `IMAGENET1K_V1`.

---

## 4. Training Configuration

| Item | Value | [as executed] |
|------|-------|---------------|
| Loss | Cross-Entropy | same |
| Optimizer | Adam (lr 1e-3 head; 1e-5 if fine-tuning) | Adam, **1e-4** for both CNNs and **2e-5** for ViT-B/16, set per architecture in advance (not tuned per fold) |
| Batch size | 32 | same |
| Epochs | up to 30 + **early stopping** (patience 5) | up to 30, **patience 6**, best-epoch weights restored |
| Hardware | Google Colab free GPU (T4) | same; 15 fits (3 architectures × 5 folds) |
| Seed | 42 (reproducibility) | 42 offset by fold index; cuDNN restricted to deterministic algorithms |
| Model selection | best validation F1 / AUC | best **validation ROC-AUC** only |

---

## 5. Evaluation Metrics (report ALL — not just accuracy)

- Accuracy, **Precision, Recall, F1-score**, **ROC-AUC**
- **Confusion matrix** (per model)
- **ROC curves** (all models on one plot)
- Mean ± std across 5 folds
- (Optional) inference time / #params → supports "efficiency" angle

> ⚠️ On balanced data, accuracy is meaningful — but still report F1/AUC (reviewers require it).
> 🔴 **[as executed]** The **pooled out-of-fold** estimate is the primary result — every
> frame predicted once by the model that never saw its surgery — and all inferential
> comparisons are made on those predictions. Mean ± SD across folds is reported as a
> description of between-fold variability, not as a second estimate of the same
> quantity. Accuracy is *not* compared across folds, because fold prevalence ranges
> from 37.8% to 66.4% (groups are indivisible), which moves where a fixed 0.5
> threshold sits. ROC-AUC is the primary metric because it is threshold-free. Note
> also that the balanced 1:1 evaluation set makes accuracy and precision
> non-transferable to deployment prevalence (~1:36 in the source data).

---

## 5b. Statistical Significance (reviewers expect this)

- **McNemar's test** between the two best models (is the accuracy difference significant?)
- **95% confidence intervals** for AUC (DeLong method or bootstrap)
- Report mean ± std across the 5 GroupKFold folds

> 🔴 **[as executed]** Expanded considerably, and DeLong was not used:
> - **The resampling unit is the surgery, not the frame.** Frames from one recording
>   share patient, camera and illumination, so frame-level resampling counts repeated
>   views as independent evidence. All intervals come from a 10,000-resample
>   **cluster (surgery-level) bootstrap**; frame-level resampling narrows them by a
>   factor of two to two and a half.
> - All **three** pairs are tested, not just the top two, with **Holm correction**
>   within each pre-specified family of comparisons.
> - McNemar is reported because the discordant counts are informative, but it treats
>   each discordant *frame* as an independent pair, so it supports no claim.
> - Outcome: only **ResNet50 > EfficientNet-B0** on AUC survives correction
>   (+0.048, *p* = 0.030). The other two classification comparisons do not, and one
>   of them (EfficientNet-B0 vs ViT-B/16) is inconclusive rather than null.

## 6. Explainability (Grad-CAM) — the novelty

- Apply **Grad-CAM** on the best CNN and (Attention-rollout / Grad-CAM variant for) ViT
  - **[as executed]** Grad-CAM on **all three** architectures, plus **attention
    rollout** for ViT-B/16, so the transformer is also judged by its native method.
    Target layers: `layer4[-1]` (ResNet50), `features[-1]` (EfficientNet-B0), last
    encoder block's first layer-norm with a 14×14 token reshape (ViT-B/16). Maps
    normalised to [0,1], upsampled to 224×224, thresholded at 0.5.
- Overlay heatmaps on sample test images (both correct & incorrect predictions)
  - **[as executed]** `results/fig_overlays.png` was made from the earlier
    single-split weights and is illustrative only; the quantitative figure is
    `results/fig_localization.png`.
- Qualitative check: does the model focus on the **actual lesion region**?
- Compare CNN vs ViT attention patterns → discussion point

### 6b. Quantitative Grad-CAM validation (BIG novelty — we have the data!)
GLENDA ships **373 ground-truth segmentation masks** (`annots/*.png`) marking the real lesion.
- Threshold the Grad-CAM heatmap → binary attention map
- Compute **IoU / overlap** between Grad-CAM map and the ground-truth mask
- This turns explainability from "looks nice" into a **measured, quantitative claim**:
  *"the model attends to the true lesion region with mean IoU = X"*
- Strongest differentiator vs prior GLENDA papers.

> 🔴 **[as executed]** A bare IoU will not support that claim, and the plan above is
> the main thing the final analysis had to fix. Lesions cover only **7.20%** of a
> frame, so every number needs a reference level:
> - an **uninformative map** scores 0.072 on Pointing Game and Coverage by
>   construction;
> - for IoU, each method is compared against a **random region of exactly the area
>   that method claimed**;
> - a **centred Gaussian** prior ("lesions are usually mid-frame", no image input at
>   all) scores IoU 0.135, Pointing 0.274, Coverage 0.119.
>
> Against those, only **EfficientNet-B0** beats the centre prior (IoU 0.156,
> *p* = 0.0048); ResNet50 (0.128) is indistinguishable from it; ViT-B/16's Grad-CAM
> (0.039) is indistinguishable from chance on all three measures and attention
> rollout (0.014) is worse. Measures used: IoU, Pointing Game and Coverage, over
> **all 373 pathology frames** (102 surgeries), each scored by the fold model that
> never saw its surgery. Spatial agreement with a mask is not faithfulness, and the
> write-up says so.

---

## 7. Experimental Pipeline (flow)

```
GLENDA raw
   │  (balance: 373+373, seed=42)
   ▼
Balanced dataset ──► 5-fold CV grouped by surgery   [as executed: primary]
   │                   (+ the 70/15/15 single split, sensitivity only)
   │
   ▼
Preprocess + augment (224×224, ImageNet norm)
   │
   ├──► ResNet50 ┐
   ├──► EffNet-B0├─ transfer learning ─► metrics (Acc/P/R/F1/AUC)
   └──► ViT-B/16 ┘
   │
   ▼
Compare models (tables + ROC + confusion matrices)
   │
   ▼
Grad-CAM on all three + attention rollout for ViT,
scored vs masks AND vs chance/centre-prior references
   │
   ▼
Results + Discussion ──► Paper
```

---

## 8. Expected Contributions (restated for paper)

1. A reproducible **balanced GLENDA binary benchmark** (public split, fixed seed).
2. Fair **CNN vs Vision Transformer** comparison with full metrics + 5-fold CV.
3. **Grad-CAM explainability** analysis linking predictions to lesion regions.

> **[as executed]** The claimed contributions changed once the results were in. Item 3
> is now *measuring* saliency against expert masks **and against chance-level and
> centre-prior references** — the finding is that two of the four
> architecture/method combinations do **not** link predictions to lesion regions, so
> the contribution is the measurement rather than the link. A fourth contribution was
> added: the demonstration that the grouping unit (segment folder vs recording) and
> the choice of split each change which architecture appears best. See
> [research_brief.md](research_brief.md) §2.

---

## 9. Threats to Validity (add to paper — reviewers love this)

- Small dataset (746 images) → mitigated by transfer learning, augmentation, CV.
  - **[as executed]** Not fully mitigated, and the write-up does not claim it is.
    Because the resampling unit is the surgery, the effective sample size is **122
    surgeries**, and only one of the three classification comparisons was detectable.
    Limited power is reported as a limitation, not as something CV solved.
- Undersampling discards data → acknowledge; note future work with weighted-loss/GAN on full data.
- Single dataset (GLENDA) → generalization limited; state as limitation.

## 10. Reproducibility & Ethics (housekeeping — needed for acceptance)

- **GitHub repo** from day one: balanced split (case IDs), code, seeds → reviewers value this.
- **Ethics:** GLENDA is a public, de-identified dataset → state *"no IRB approval required."*
- **License:** GLENDA is **CC BY-NC 4.0** → cite the dataset paper (MMM 2020) + note non-commercial use.
- Fix **seed = 42** everywhere; log package versions.

## 11. Target Venue (decide in parallel with coding — do NOT leave to the end)

- Need **acceptance in ~1–2 months** → shortlist fast conferences / preprint now.
- Locking venue early fixes the **template, page limit, and deadline** before writing.
- Plan: submit to a conference **+** release a **medRxiv/arXiv preprint** as guaranteed record.

---

## Recap of critical additions (post-recheck)
1. 🔴 Patient/case-wise split (GroupKFold) — anti-leakage
   — **[as executed]** and the group must be the **recording**, not the segment folder
2. 🟡 Quantitative Grad-CAM vs mask (IoU) — novelty
   — **[as executed]** only interpretable alongside chance and centre-prior references
3. 🟡 Diverse no-pathology sampling
4. 🟡 McNemar test + AUC confidence intervals
   — **[as executed]** surgery-level bootstrap and Holm correction; McNemar is
   descriptive only
5. 🟢 Target venue chosen early
6. 🟢 GitHub repo for reproducibility
7. 🟢 Ethics + license statement

---

## Next: Step 6 — implement this pipeline on Colab (code).

> **[as executed]** Done, with the departures flagged above. For what was actually
> run and what it found, read `paper/sections/04_methodology.md` and
> [research_brief.md](research_brief.md) rather than this file.
