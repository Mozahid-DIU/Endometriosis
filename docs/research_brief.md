# Research brief — context for an AI reviewer

Paste this before asking for a review of any section of the manuscript. It contains
the full protocol and every headline number, so that feedback can be checked against
what was actually done.

**Note for the reviewer:** before claiming that two numbers are inconsistent, please
recompute them from the confusion counts given in §6 and state your arithmetic. A
previous review reported an F1 mismatch that did not exist.

**Note on protocol versions.** Three evaluation protocols were run on this data. Only
the third is authoritative, and every number in this brief is from it unless the text
says otherwise. The other two appear only where the manuscript reports them as a
sensitivity analysis (§7). If a number you are checking does not appear here, check
`results/` before assuming an error — the authoritative files are
`cv_predictions_v3.csv`, `cv_fold_metrics_v3.csv`, `cv_localization_v3.csv`,
`cv_analysis_summary.csv`, `cluster_bootstrap_classification.csv` and
`cv_localization_summary.csv`.

---

## 1. One-paragraph summary

We compare two CNNs (ResNet50, EfficientNet-B0) and a Vision Transformer (ViT-B/16)
on binary endometriosis classification in laparoscopic still frames, under a
leakage-free, surgery-grouped 5-fold cross-validation. Beyond accuracy, we ask
whether each model's saliency map actually lands on the lesion, by scoring Grad-CAM
(and, for the transformer, attention rollout) against expert pixel masks and against
explicit chance-level references. The headline finding is a dissociation:
classification barely separates the three architectures once between-surgery
variability is accounted for — the only difference that survives correction is
ResNet50 over EfficientNet-B0 — while explanation quality separates them decisively
and in the opposite direction.

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
   IoU figure cannot be interpreted. Every one of the 373 pathology frames is scored
   by the fold model that never saw its surgery.
4. Evidence that the evaluation protocol can reverse the apparent ranking of models,
   demonstrated on our own data twice: once by a single train/test split, and once by
   setting the grouping unit one level too fine.

## 3. Dataset

- **GLENDA v1.5**, public, CC BY-NC, de-identified; no additional ethics approval.
- 373 pathology frames, each with an expert pixel-level lesion mask; 13,438
  no-pathology frames (raw ratio ≈ 1:36).
- **Balanced subset used:** all 373 pathology frames + 373 no-pathology frames
  sampled round-robin across the 27 no-pathology recording folders (avoids
  near-duplicate consecutive frames and single-recording dominance). Seed 42.
- Total 746 frames from **122 surgeries**. A group is a surgery: a case id (`c_<id>`)
  for pathology frames (102 cases), and a *recording* for no-pathology frames
  (20 recordings). GLENDA distributes no-pathology frames in per-segment folders, so
  one recording appears under several folder names; the segment suffix is stripped,
  because two segments of one recording are the same surgery. Treating the 129
  segment folders as groups instead is the leak documented in §7.
- Lesions occupy **7.20%** of a frame on average (range 0.17–57.39%).

## 4. Protocol

- `StratifiedGroupKFold`, 5 folds, whole surgeries kept together; the training
  portion of each fold is split again group-wise for early stopping. Test portions
  are never used for model selection.
- Asserted per fold: no group in two partitions; the five test portions tile the
  dataset exactly once.
- Because groups are indivisible, fold composition varies: test portions hold
  125–179 frames and pathology prevalence ranges **37.8%–66.4%**. This is why
  ROC-AUC, not accuracy, is the primary metric.
- The **pooled out-of-fold estimate is the primary result** — every frame predicted
  once, by the model that never saw its surgery — and all inferential comparisons are
  made on those predictions. Fold mean ± SD is reported as a description of
  between-fold variability, not as a second estimate of the same quantity.
- Two earlier protocols were run and are reported only as sensitivity analyses
  (see §7): a single case-wise 70/15/15 split (528/111/107 frames), and a 5-fold CV
  that grouped no-pathology frames by segment folder and therefore leaked.

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

**Per-fold mean ± SD (ROC-AUC):** ResNet50 0.903 ± 0.082 · EfficientNet-B0
0.884 ± 0.110 · ViT-B/16 0.925 ± 0.052. Fold-level AUC spans 0.751–0.989 across all
fifteen fits.

**Pooled out-of-fold** (all 746 frames, each predicted by the model blind to its
surgery; 95% surgery-level bootstrap intervals):

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| ResNet50 | 0.812 | 0.786 | 0.858 | 0.821 | 0.900 [0.854–0.939] |
| EfficientNet-B0 | 0.761 | 0.712 | 0.877 | 0.786 | 0.852 [0.776–0.916] |
| ViT-B/16 | 0.827 | 0.810 | 0.855 | 0.832 | 0.896 [0.832–0.944] |

**Confusion counts** (373 frames per class): ResNet50 TP 320, FN 53, FP 87, TN 286 ·
EfficientNet-B0 TP 327, FN 46, FP 132, TN 241 · ViT-B/16 TP 319, FN 54, FP 75,
TN 298.

All three recall pathology frames at a similar rate (0.855–0.877) and differ mainly
in how many normal frames they call positive. Accuracy and precision describe the
balanced 1:1 evaluation set only; the source data are about 1:36.

**Pairwise tests** (pooled predictions, surgery-level resampling, Holm-corrected
within the family of three, two-sided). Intervals are per-comparison and
uncorrected, so an interval that excludes zero can sit beside a corrected *p* above
0.05 — that is not a contradiction. McNemar's exact test is reported because the
discordant counts are informative, but it treats each discordant *frame* as an
independent pair and so ignores the clustering the other tests correct for; its
*p*-values support no claim in the manuscript.

| A vs B | ΔAUC [95% CI] | *p* raw / Holm | ΔAccuracy [95% CI] | *p* raw / Holm | Discordant (A/B), McNemar *p* (frame-level) |
|---|---|---|---|---|---|
| ResNet50 vs EfficientNet-B0 | +0.048 [+0.010, +0.095] | 0.010 / **0.030** | +0.051 [−0.001, +0.103] | 0.058 / 0.126 | 84 / 46, 0.0011 |
| ResNet50 vs ViT-B/16 | +0.004 [−0.040, +0.053] | 0.87 / 0.87 | −0.015 [−0.072, +0.044] | 0.62 / 0.62 | 58 / 69, 0.38 |
| EfficientNet-B0 vs ViT-B/16 | −0.044 [−0.087, −0.004] | 0.029 / 0.058 | −0.066 [−0.134, −0.003] | 0.042 / 0.126 | 50 / 99, 0.00007 |

**One comparison survives correction: ResNet50 discriminates better than
EfficientNet-B0 (ΔAUC +0.048, *p* = 0.030).** ResNet50 vs. ViT-B/16 shows no
evidence of a difference on either test. EfficientNet-B0 vs. ViT-B/16 favours
ViT-B/16 on both point estimates and both uncorrected intervals exclude zero, but
neither survives correction — that comparison is **inconclusive, not null**. The
manuscript claims no ordering among the three architectures.

## 7. Statistical methodology (important)

- 10,000 bootstrap resamples, percentile intervals, all tests two-sided, Holm
  correction within each pre-specified family. The families are: the three pairwise
  classification comparisons; the six pairwise localization comparisons; the eight
  comparisons of localization against a reference level; and the eight comparisons of
  Pointing Game and Coverage against chance.
- **The resampling unit is the surgery, not the frame.** Frames from one recording
  share patient, camera and illumination, so frame-level resampling treats repeated
  views as independent evidence. Pathology and no-pathology groups are drawn
  separately so both classes appear in every resample. Paired comparisons resample
  the same surgeries for both models.
- Effect of this choice: frame-level resampling narrows the intervals by a factor of
  **two to two and a half**, which is the difference between reporting several
  significant architecture differences and reporting almost none. On
  EfficientNet-B0 vs. ViT-B/16, frame-level McNemar gives *p* = 7 × 10⁻⁵ and the
  surgery-level accuracy test gives *p* = 0.126 on the same data; the discordant
  frames are concentrated in a few recordings. We report the surgery-level result.
- **Split sensitivity (protocol 1).** The preliminary single case-wise split of 107
  frames gave AUCs of 0.915 (ResNet50), 0.860 (ViT-B/16), 0.833 (EfficientNet-B0),
  and would have identified EfficientNet-B0 as clearly the weakest and ViT-B/16 as
  clearly second. Cross-validation gives 0.900 / 0.896 / 0.852 pooled: ViT-B/16 moves
  from clearly second to tied first, and ViT-B/16's single-split value falls below all
  five of its fold values.
- **Grouping sensitivity (protocol 2).** An earlier cross-validation treated each
  no-pathology *segment folder* as a group. Six of the twenty no-pathology recordings
  were thereby split into two or three groups, and four of them ended up with
  segments on both sides of a split — leakage affecting 178 of the 373 no-pathology
  frames. Under that grouping EfficientNet-B0 reached a pooled AUC of 0.913 and
  appeared tied for best; with the recording treated as one surgery it falls to 0.852
  and is the weakest model. The corrected grouping is reported throughout; the
  earlier number is recorded only to document the size of the effect.
- Three protocols on the same data and the same architectures therefore produced
  three orderings. The claim drawn is about this dataset, not about the literature.

## 8. Explanation evaluation

Grad-CAM at `layer4[-1]` (ResNet50), `features[-1]` (EfficientNet-B0), and the last
encoder block's first layer-norm with a 14×14 token reshape (ViT-B/16). ViT-B/16 is
*additionally* explained with **attention rollout** (Abnar & Zuidema 2020), to rule
out the objection that Grad-CAM is unfair to a transformer. Maps are normalised to
[0, 1], upsampled to 224×224 and thresholded at 0.5.

Scoring happens inside the cross-validation loop, so **all 373 pathology frames
(102 surgeries)** are covered by the fold model that never saw their surgery.

Measures: **IoU** with the binarised mask; **Pointing Game** (is the map's maximum
inside the mask); **Coverage** (share of map intensity inside the mask).

**References against which the numbers are interpreted:**
- An uninformative map scores the mean mask area, 0.072, on Pointing Game and
  Coverage by construction.
- For IoU, each method is compared against a random region of *exactly the area that
  method claimed*; expected IoU = *pm*/(*p*+*m*−*pm*), verified by simulation.
- A **centred Gaussian** (σ = 0.25×224), thresholded identically, encodes the "lesions
  are usually mid-frame" prior without looking at the image. IoU 0.135, Pointing
  0.274, Coverage 0.119.

**Results** (95% surgery-level intervals):

| | IoU | Pointing Game | Coverage |
|---|---|---|---|
| EfficientNet-B0 | 0.156 [0.133–0.180] | 0.287 [0.233–0.347] | 0.153 [0.128–0.181] |
| ResNet50 | 0.128 [0.111–0.147] | 0.271 [0.221–0.324] | 0.168 [0.139–0.200] |
| ViT-B/16, Grad-CAM | 0.039 [0.033–0.047] | 0.078 [0.052–0.105] | 0.070 [0.058–0.084] |
| ViT-B/16, rollout | 0.014 [0.010–0.019] | 0.054 [0.030–0.082] | 0.064 [0.051–0.078] |

**Pairwise IoU** (family of six, Holm-corrected — all six significant):
EfficientNet-B0 − ResNet50 +0.029 [+0.009, +0.049], *p* = 0.004 ·
ResNet50 − ViT-B/16 Grad-CAM +0.088 [+0.072, +0.106], *p* = 0.0006 ·
EfficientNet-B0 − ViT-B/16 Grad-CAM +0.117 [+0.096, +0.140], *p* = 0.0006 ·
ViT-B/16 Grad-CAM − rollout +0.026 [+0.019, +0.033], *p* = 0.0006.

**Against the references** (family of eight): ResNet50 beats area-matched random
(+0.096, *p* = 0.0008) but is **indistinguishable from the centre prior**
(−0.007, *p* = 0.80). EfficientNet-B0 beats both (random +0.116, *p* = 0.0008;
centre prior **+0.022, *p* = 0.0048**) and is the only method that does.
ViT-B/16's Grad-CAM is **indistinguishable from area-matched random** (+0.002,
*p* = 0.80) and significantly *below* the centre prior (−0.095, *p* = 0.0008);
attention rollout is worse still, falling below the random floor on IoU
(−0.015, *p* = 0.0008).

**Pointing Game and Coverage against chance** (family of eight): both CNNs exceed
chance on both measures (ResNet50 +0.199 and +0.096; EfficientNet-B0 +0.215 and
+0.081; all *p* = 0.0008). ViT-B/16's Grad-CAM does not differ from chance on either
(+0.006, *p* = 1.00; −0.002, *p* = 1.00); rollout does not differ on Pointing Game
(−0.018, *p* = 0.39) and falls below chance on Coverage (−0.008, *p* = 0.0008).

So the dissociation is: the evaluation finds no evidence that ViT-B/16 classifies
worse than ResNet50, yet ViT-B/16's maps are indistinguishable from chance on all
three measures; and EfficientNet-B0, the one model placed below another on AUC, is
the only one whose maps beat the centre-of-image prior.

## 9. Error analysis

Errors are not spread evenly. Of the 54 surgeries contributing ≥5 frames, the twelve
with the highest error rate are **predominantly no-pathology recordings**; in the
worst (`v_4044`, 14 frames) each of the three models misclassifies 11 to 13 frames,
and 52 frames are misclassified by all three. The dominant failure mode is a
systematic false positive on healthy tissue in particular recordings, not missed
lesions — recall is 0.855–0.877 for all three. Since folds are grouped by surgery
these frames were never trained on, so this is not memorization; we do not claim to
know what distinguishes those recordings.

## 10. Known limitations (already acknowledged — no need to re-report)

- One dataset, 122 surgeries, 746 frames. Because the resampling unit is the surgery,
  the effective sample size for every comparison is of that order, and only large
  classification differences were detectable.
- Balanced 1:1 evaluation set against source data of about 1:36, so precision and
  accuracy are not deployment predictive values.
- Still frames, not video; a surgeon uses motion and manipulation.
- Binary only: no lesion type, stage or severity.
- Spatial agreement with a mask is **not faithfulness**. GLENDA gives one annotation
  per frame, so no inter-annotator variability. The map threshold is fixed at 0.5 and
  one target layer is used per architecture; both affect the absolute numbers, though
  the ordering was stable across the three metrics. Only two explanation methods were
  tested.
- Pooling out-of-fold predictions assumes the five fold models produce scores on
  comparable scales; the fold-wise summary is reported beside the pooled one rather
  than verifying that assumption. Learning rates were set per architecture in advance,
  not tuned per fold. The exact partition depends on the `StratifiedGroupKFold`
  implementation — the same seed under a different scikit-learn version produced
  different folds — so the manifest and a hash of the frame-to-fold assignment are
  released, but the folds are not canonical.
- No calibration analysis, no decision-curve/utility analysis, no external
  validation, no clinical evaluation. We make no claim of clinical readiness.

## 11. Status and target

All eight sections are drafted: Abstract, Introduction, Related Work, Methodology,
Results, Discussion, Limitations, Conclusion (`paper/sections/`, assembled into
`paper/paper_full.md`). `scripts/check_paper_numbers.py` checks every number in the
prose against `results/`. Target venue: *Biomedical Signal Processing and Control*
(Elsevier), with an arXiv preprint first. Code, per-frame out-of-fold predictions,
saliency scores and all figure scripts:
`github.com/Mozahid-DIU/Endometriosis`.

**Figures** (`results/`): `fig_dataset_folds` (fold composition and prevalence),
`fig_roc_pooled`, `fig_confusion_pooled`, `fig_cv_auc` (per-fold AUC beside the
pooled estimate and the single-split value), `fig_localization` (localization against
both reference levels), `fig_overlays`, `fig_error_clusters` (the hardest surgeries).
Note that **`fig_overlays` was generated from the earlier single-split model weights
and is illustrative only** — it is not an output of the cross-validation run that
every number above comes from.

## 12. What feedback is most useful

1. Claims that outrun the evidence, especially any implied clinical utility, and any
   place where an inconclusive comparison is written as a null one.
2. Statistical errors: resampling unit, multiplicity, interval interpretation.
3. Anything a reviewer of a biomedical imaging journal would demand that is missing.
4. Places where a stated number and a stated conclusion do not match — with your
   arithmetic shown.
