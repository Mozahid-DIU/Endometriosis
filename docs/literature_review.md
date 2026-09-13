# Literature Review — AI/Deep Learning for Endometriosis Diagnosis

**Research area:** Automated endometriosis detection from medical images (focus: laparoscopy / GLENDA)
**Prepared:** Step 3 of research pipeline — Literature Review
**Papers surveyed:** 24 (grouped by modality)

> ⚠️ **NOTE:** Accuracy/AUC figures below are collected from abstracts & search summaries.
> Before citing in your final paper, **verify each number against the full text/PDF**.
> This table is your *Related Work* raw material + gap-finding tool.

---

## Category A — Laparoscopy / GLENDA-based (most relevant to you)

| # | Paper (year) | Model(s) | Dataset | Key result | Limitation |
|---|--------------|----------|---------|-----------|------------|
| A1 | Explainable DL for Endometriosis Classification in Laparoscopic Images (MDPI BiomedInformatics, 2025) | ResNet50, EfficientNet-B2, EdgeNeXt-Small, ViT-Small/16 | GLENDA | Binary classification + explainability (XAI) under imbalance | Small annotated set; imbalance |
| A2 | Deep learning based detection of endometriosis lesions, 5-fold CV (ScienceDirect, 2025) | VGG19, ResNet50, Inception V3 | GLENDA | Acc 0.89 / 0.91 / 0.93 (Inception V3 best) | Only 3 CNNs; no imbalance-specific method |
| A3 | Endometriosis detection & localization in laparoscopic gynecology (Springer MTA, 2021) | Faster R-CNN, Mask R-CNN | GLENDA | Region-based lesion localization | Detection focus, small data |
| A4 | GLENDA: Gynecologic Laparoscopy Endometriosis Dataset (MMM, 2020) | — (dataset paper) | GLENDA (25k frames) | Introduced the dataset you are using | Not a model paper |
| A5 | Deep Learning Improves Accuracy of Laparoscopic Imaging Classification (Preprints.org / JCMS, 2023) | CNN | GLENDA subset (2,157 healthy / 2,291 patho) | Data-mining + CNN classification | Preprint; limited validation |
| A6 | Object Detection in Laparoscopic Surgery: Comparative Study, custom dataset (Diagnostics, 2025) | Faster R-CNN, YOLOv9 | Custom (332 videos, 17,560 annotated frames) | Compared detectors under training scenarios | Private dataset |
| A7 | Automated prediction of endometriosis using deep learning (IJNAA) | Deep CNN | GLENDA | Automated classification pipeline | Journal indexing weaker |
| A8 | Automatic segmentation of deep endometriosis in rectosigmoid (ScienceDirect, 2024) | Deep learning segmentation | MRI (rectosigmoid) | Auto-segmentation of deep lesions | Segmentation, not laparoscopy |
| A9 | SWOT Analysis of ViTs for Endometriosis from Laparoscopic Videos (ISJ Trend) | Vision Transformers | Laparoscopic video | ViT 91–97% acc in surgical tasks; feasibility/ethics | Conceptual/feasibility |
| A10 | Initial Results: Automatic Visual Recognition of Endometriosis Lesions, proof-of-concept (ScienceDirect, 2025) | DL detector | Laparoscopy | Proof-of-concept live recognition | Early-stage |
| A11 | YOLOv5 nine-class endometriosis lesion detection | YOLOv5 | Laparoscopy (9 lesion types) | Multi-class lesion detection | Needs fine-grained labels |

## Category B — Ultrasound-based

| # | Paper (year) | Model(s) | Dataset | Key result | Limitation |
|---|--------------|----------|---------|-----------|------------|
| B1 | Augmenting endometriosis analysis from ultrasound with DL (arXiv 2302.09621, 2023) | Xception, Inception-V4, ResNet50, DenseNet, EfficientNetB2 | Ultrasound | AUC 0.85 & 0.90 (5-fold CV) | Ultrasound operator-dependent |
| B2 | ConvNeXt for ovarian endometriosis cyst vs mucinous cystadenoma (2024) | ConvNeXt | Ultrasound | Discriminative diagnosis | Narrow cyst subtype |
| B3 | CNN comparison for ovarian endometriotic cyst (ResNet-152, DenseNet-161, EfficientNet-B7) | ResNet-152 best | Ultrasound | AUC 0.986, beats physicians | Single-center |
| B4 | AI in diagnosis/prediction of endometriosis via ultrasound: systematic review (Reproductive Health, 2025) | Review | — | Summarizes US-based AI evidence | Review (no new model) |

## Category C — MRI-based

| # | Paper (year) | Model(s) | Dataset | Key result | Limitation |
|---|--------------|----------|---------|-----------|------------|
| C1 | Multi-Modal Pelvic MRI Dataset for DL segmentation (Nature Scientific Data, 2025) | nnU-Net, RAovSeg | New pelvic MRI dataset | Baseline organ segmentation | Segmentation baselines only |
| C2 | AI-based MRI reading support program (AMP) for deep endometriosis (Scientific Reports, 2025) | DL reading support | MRI | Clinical decision support tool | Deployment-focused |
| C3 | U-Net endometriotic lesion segmentation (2024) | U-Net | MRI/US | Dice 0.977 | Segmentation metric only |

## Category D — Symptom / tabular ML (non-image)

| # | Paper (year) | Model(s) | Dataset | Key result | Limitation |
|---|--------------|----------|---------|-----------|------------|
| D1 | Self-report symptom-based endometriosis prediction using ML (Scientific Reports, 2023) | ML (best model) | Questionnaire (24 symptoms) | AUC 0.94, sens 0.93, spec 0.95 | Self-reported labels |
| D2 | ML algorithms as new screening approach (Scientific Reports, 2021) | Soft Voting, Random Forest, XGBoost | Clinical/questionnaire | Sens 95–98%, spec ~80% | Heterogeneous control group |

## Category E — Reviews / Meta-analyses (cite in Introduction)

| # | Paper (year) | Type | Takeaway |
|---|--------------|------|----------|
| E1 | AI in Endometriosis Imaging: A Scoping Review (MDPI AI, 2026) | PRISMA-ScR scoping review (2015–2025) | Maps the whole field; good citation backbone |
| E2 | Diagnostic accuracy of ML for endometriosis: systematic review & meta-analysis (Frontiers Endocrinology, 2025) | Meta-analysis | DL acc 0.89–0.93, AUC ~0.90 across studies |
| E3 | Current Status & Future Potential of ML in Diagnostic Imaging of Endometriosis (PMC12122278) | Literature review | Field still under-explored |
| E4 | Assessing Utility of AI in Endometriosis: Promises and Pitfalls (PMC11062212) | Critical review | Highlights data & generalization problems |

---

## Synthesis — What the field already knows

1. **Laparoscopy + CNN transfer learning is the dominant, proven approach** (VGG19/ResNet50/Inception → ~0.89–0.93 accuracy on GLENDA).
2. **GLENDA is the standard public dataset**, but it is **severely imbalanced** (373 pathology vs 13,438 no-pathology).
3. **Newer directions:** Vision Transformers (ViT), explainable AI (XAI), object detection (YOLO/Faster R-CNN), and MRI/US segmentation.
4. Symptom-based tabular ML is a separate, strong track (AUC ~0.94) but not image-based.

---

## Research GAPS (your opportunity) 🎯

These are candidate gaps — each can become your paper's contribution:

- **G1 — Class imbalance is acknowledged but rarely *solved* systematically.** Most GLENDA papers just train CNNs; few rigorously compare imbalance-handling techniques (undersampling vs weighted loss vs augmentation vs GAN). → *A focused comparison could be your contribution.*
- **G2 — Limited head-to-head of CNN vs Vision Transformer vs hybrid** on the *same balanced GLENDA split* with full metrics (F1, AUC, recall) + explainability.
- **G3 — Explainability (Grad-CAM/SHAP) is under-reported** — clinicians need to *see why* the model flags a lesion.
- **G4 — Reproducibility:** many report only accuracy on imbalanced data (misleading). A clean, balanced, reproducible benchmark with proper metrics is valuable.
- **G5 — Lightweight/efficient models** (EfficientNet, EdgeNeXt, MobileViT) for low-resource settings (relevant to Bangladesh context) are barely explored.

---

## Suggested direction (leads into Step 4)

> **Strongest, most doable for a first paper:**
> Combine **G1 + G2 + G3** → *"A balanced-data benchmark comparing CNN and Vision Transformer architectures for binary endometriosis classification on GLENDA, with class-imbalance handling and Grad-CAM explainability."*

This is:
- ✅ Doable on Colab free GPU with transfer learning
- ✅ Clear novelty (balanced benchmark + XAI + imbalance study)
- ✅ Backed by proven methods (low risk)
- ✅ Publishable at a conference / preprint in your 1–2 month window

---

## Next step (Step 4): lock the Research Question + Contribution in one sentence.
