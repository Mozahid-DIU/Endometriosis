# Methodology — Step 5

**Title:** Comparative Analysis of CNN and Vision Transformer Architectures for Explainable Endometriosis Classification in Laparoscopic Images
**Task:** Binary classification (endometriosis vs no-pathology) on balanced GLENDA + Grad-CAM explainability

---

## 1. Dataset Preparation

**Source:** GLENDA v1.5 (already downloaded)
- Pathology (endometriosis): **373 images** — `Glenda_v1.5_classes/.../frames/`
- No-pathology (normal): **13,438 images** — `GLENDA_v1.5_no_pathology/.../frames/`

**Balancing strategy (undersampling — simplest & standard):**
- Keep all 373 endometriosis images
- Randomly sample 373 no-pathology images (fixed seed for reproducibility)
- **Balanced set = 746 images (373 + 373)**

**⚠️ CRITICAL — Patient/Case-wise split (prevents data leakage):**
GLENDA frames come from videos; multiple frames share the same surgery/case.
Filename encodes the case ID: `c_100_v_(video_3062.mp4)_f_1458.jpg` → case = `c_100`.
- **Never** put frames of the same case in both train and test (would inflate results → reject).
- Extract case ID via regex → group images by case → split with **GroupShuffleSplit / GroupKFold** (scikit-learn).
- No-pathology frames grouped by their **video ID** (`v_2506`) the same way.
- Frame counts per split will NOT be exactly 70/15/15 (cases have unequal frame counts) — this is correct and expected.

**Split (patient-wise, fixed seed = 42):**
| Split | ~% | Note |
|-------|---|------|
| Train | ~70% | whole cases only |
| Validation | ~15% | whole cases only |
| Test | ~15% | whole cases only, held-out |

> ✅ Use a **held-out test set** (not just val) — reviewers expect this.
> ✅ Report **5-fold GroupKFold cross-validation** (small data → CV gives credible, stable results); grouping keeps the leakage-safe guarantee across folds.

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

*(Optional 4th if time allows: DenseNet121 or MobileViT — only if schedule permits.)*

**Setup:** freeze backbone → train new classification head → (optional) fine-tune top layers.

---

## 4. Training Configuration

| Item | Value |
|------|-------|
| Loss | Cross-Entropy |
| Optimizer | Adam (lr 1e-3 head; 1e-5 if fine-tuning) |
| Batch size | 32 |
| Epochs | up to 30 + **early stopping** (patience 5) |
| Hardware | Google Colab free GPU (T4) |
| Seed | 42 (reproducibility) |
| Model selection | best validation F1 / AUC |

---

## 5. Evaluation Metrics (report ALL — not just accuracy)

- Accuracy, **Precision, Recall, F1-score**, **ROC-AUC**
- **Confusion matrix** (per model)
- **ROC curves** (all models on one plot)
- Mean ± std across 5 folds
- (Optional) inference time / #params → supports "efficiency" angle

> ⚠️ On balanced data, accuracy is meaningful — but still report F1/AUC (reviewers require it).

---

## 5b. Statistical Significance (reviewers expect this)

- **McNemar's test** between the two best models (is the accuracy difference significant?)
- **95% confidence intervals** for AUC (DeLong method or bootstrap)
- Report mean ± std across the 5 GroupKFold folds

## 6. Explainability (Grad-CAM) — the novelty

- Apply **Grad-CAM** on the best CNN and (Attention-rollout / Grad-CAM variant for) ViT
- Overlay heatmaps on sample test images (both correct & incorrect predictions)
- Qualitative check: does the model focus on the **actual lesion region**?
- Compare CNN vs ViT attention patterns → discussion point

### 6b. Quantitative Grad-CAM validation (BIG novelty — we have the data!)
GLENDA ships **373 ground-truth segmentation masks** (`annots/*.png`) marking the real lesion.
- Threshold the Grad-CAM heatmap → binary attention map
- Compute **IoU / overlap** between Grad-CAM map and the ground-truth mask
- This turns explainability from "looks nice" into a **measured, quantitative claim**:
  *"the model attends to the true lesion region with mean IoU = X"*
- Strongest differentiator vs prior GLENDA papers.

---

## 7. Experimental Pipeline (flow)

```
GLENDA raw
   │  (balance: 373+373, seed=42)
   ▼
Balanced dataset ──► train/val/test split (70/15/15) + 5-fold CV
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
Grad-CAM explainability on best models
   │
   ▼
Results + Discussion ──► Paper
```

---

## 8. Expected Contributions (restated for paper)

1. A reproducible **balanced GLENDA binary benchmark** (public split, fixed seed).
2. Fair **CNN vs Vision Transformer** comparison with full metrics + 5-fold CV.
3. **Grad-CAM explainability** analysis linking predictions to lesion regions.

---

## 9. Threats to Validity (add to paper — reviewers love this)

- Small dataset (746 images) → mitigated by transfer learning, augmentation, CV.
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
2. 🟡 Quantitative Grad-CAM vs mask (IoU) — novelty
3. 🟡 Diverse no-pathology sampling
4. 🟡 McNemar test + AUC confidence intervals
5. 🟢 Target venue chosen early
6. 🟢 GitHub repo for reproducibility
7. 🟢 Ethics + license statement

---

## Next: Step 6 — implement this exact pipeline on Colab (code).
```
```
