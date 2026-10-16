## Results

Throughout, endometriosis is the positive class and every test is two-sided.
Confidence intervals are 95% percentile bootstrap intervals from 10,000 resamples
in which **whole surgeries, not individual frames, are the resampling unit**;
frames from one surgery share a patient, a camera and an illumination setting, so
resampling frames would treat repeated views of one case as independent evidence.

The **pooled out-of-fold estimate is the primary result**: every frame is predicted
once, by the model that never saw its surgery, and all inferential comparisons are
made on those predictions. Fold means and standard deviations are reported as a
description of between-fold variability, not as a second estimate of the same
quantity; the two are different estimands and are not interchangeable.

Intervals are per-comparison and uncorrected, while *p*-values are Holm-corrected
within each pre-specified family of tests. Both raw and corrected *p*-values are
given, because an uncorrected interval that excludes zero can accompany a corrected
*p* above 0.05, and the two are then easy to misread as contradictory. The families
are: the three pairwise classification comparisons; the six pairwise localization
comparisons; the eight comparisons of localization against a reference level; and
the eight comparisons of Pointing Game and Coverage against chance.

### Classification performance

Table 1 reports the three architectures under 5-fold grouped cross-validation, in
two views of the same predictions. The upper block averages each metric over the
five folds. The lower block pools the out-of-fold predictions and computes each
metric once over the resulting 746 frames, each scored by the one model that never
saw its surgery.

**Table 1.** Classification performance under 5-fold grouped cross-validation.

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| *Mean ± SD over folds* | | | | | |
| ResNet50 | 0.823 ± 0.098 | 0.791 ± 0.151 | 0.855 ± 0.132 | 0.815 ± 0.121 | 0.903 ± 0.082 |
| EfficientNet-B0 | 0.769 ± 0.113 | 0.722 ± 0.183 | **0.876 ± 0.097** | 0.783 ± 0.126 | 0.884 ± 0.110 |
| ViT-B/16 | **0.830 ± 0.092** | **0.821 ± 0.177** | 0.859 ± 0.060 | **0.829 ± 0.101** | **0.925 ± 0.052** |
| *Pooled out-of-fold (n = 746)* | | | | | |
| ResNet50 | 0.812 | 0.786 | 0.858 | 0.821 | **0.900 [0.854–0.939]** |
| EfficientNet-B0 | 0.761 | 0.712 | **0.877** | 0.786 | 0.852 [0.776–0.916] |
| ViT-B/16 | **0.827** | **0.810** | 0.855 | **0.832** | 0.896 [0.832–0.944] |

The pooled ROC curves are shown in Fig. 4.

All three models recall endometriosis frames at a similar rate (0.855–0.877) and
differ mainly in how many normal frames they call positive. The pooled confusion
counts (Fig. 2), out of 373 frames per class, are: ResNet50 TP 320, FN 53, FP 87, TN 286;
EfficientNet-B0 TP 327, FN 46, FP 132, TN 241; ViT-B/16 TP 319, FN 54, FP 75,
TN 298.

Threshold-dependent metrics vary more across folds than ROC-AUC does. Fold
prevalence is a likely contributor — because whole surgeries are kept together, the
share of pathology frames in a fold ranges from 37.8% to 66.4% (Fig. 1), which moves
where a fixed 0.5 threshold sits on the score distribution. Fold-level AUC also
varies (0.751–0.989 across all fits), so prevalence is not the only source of
variation. ROC-AUC is used as the primary metric because it is threshold-free;
accuracy is reported for completeness and is not compared across folds.

Because the evaluation set is balanced 1:1 while the source data are approximately
1:36, the precision and accuracy values above describe behaviour on this balanced
set only. They should not be read as the predictive values a deployed model would
achieve at the prevalence of unselected laparoscopy frames.

### Do the architectures differ?

Table 2 tests each pair on the pooled out-of-fold predictions. Two paired tests are
reported. The AUC difference and the accuracy difference both resample the same
surgeries for both models, so the two models always see identical data in every
resample. McNemar's exact test is also given because the discordant counts are
informative, but it treats each discordant *frame* as an independent pair and so
ignores the clustering the other tests correct for; its *p*-values are reported as
frame-level and are not used to support any claim.

**Table 2.** Pairwise comparison on the pooled out-of-fold predictions. Intervals
and the first two *p*-columns use surgery-level resampling; Holm correction is
applied within the family of three comparisons.

| Comparison (A vs B) | ΔAUC (A − B) [95% CI] | *p* raw / Holm | ΔAccuracy [95% CI] | *p* raw / Holm | Discordant frames (A/B), McNemar *p* (frame-level) |
|---|---|---|---|---|---|
| ResNet50 vs EfficientNet-B0 | **+0.048 [+0.010, +0.095]** | 0.010 / **0.030** | +0.051 [−0.001, +0.103] | 0.058 / 0.126 | 84 / 46, 0.0011 |
| ResNet50 vs ViT-B/16 | +0.004 [−0.040, +0.053] | 0.87 / 0.87 | −0.015 [−0.072, +0.044] | 0.62 / 0.62 | 58 / 69, 0.38 |
| EfficientNet-B0 vs ViT-B/16 | −0.044 [−0.087, −0.004] | 0.029 / 0.058 | −0.066 [−0.134, −0.003] | 0.042 / 0.126 | 50 / 99, 0.00007 |

One comparison survives correction: ResNet50 discriminates better than
EfficientNet-B0, by 0.048 AUC (Holm *p* = 0.030). For ResNet50 against ViT-B/16
there is no evidence of a difference on either test. For EfficientNet-B0 against
ViT-B/16 the point estimates favour ViT-B/16 on both measures and both uncorrected
intervals exclude zero, but neither survives correction (*p* = 0.058 and 0.126);
this comparison is therefore inconclusive rather than null.

The contrast between the two accuracy-based tests is worth stating plainly. On
discordant frames, ViT-B/16 is correct on 99 that EfficientNet-B0 gets wrong against
50 the other way, which a frame-level McNemar test calls decisive
(*p* = 7 × 10⁻⁵). Resampling surgeries instead of frames turns the same data into
*p* = 0.042, and 0.126 after Holm correction within the family of three. Those 149 discordant frames fall across 46 surgeries, five of which hold 46% of
them, so counting each frame as independent evidence overstates what the data
support; we report the
surgery-level result.

Taken together, this evaluation can establish only that ResNet50 outperforms
EfficientNet-B0 on AUC. It does not establish an ordering among the three
architectures, and with 122 surgeries it was unlikely to.

### Sensitivity to the protocol

The ordering of the three models is not stable across evaluation protocols. A
single case-wise held-out split of 107 frames — the protocol common in this
literature — ranked them ResNet50 (AUC 0.915), ViT-B/16 (0.860), EfficientNet-B0
(0.833). Grouped cross-validation gives ResNet50 0.900, ViT-B/16 0.896 and
EfficientNet-B0 0.852 pooled: ViT-B/16 moves from clearly second to indistinguishable from first,
and the apparent 0.08 gap between ResNet50 and EfficientNet-B0 narrows to 0.048.
Fig. 3 places the single-split value beside the five fold-level values for each
model; ViT-B/16's single-split estimate falls below all five of its folds.

A further ordering change arises from the grouping rule itself. GLENDA stores
no-pathology frames in per-segment folders, and an earlier version of this analysis
treated each folder as a group. Six of the twenty no-pathology recordings were
thereby split into two or three groups, together accounting for 178 of the 373
no-pathology frames, and four of those six ended up with segments on both sides of
a split. Under that grouping EfficientNet-B0 reached a pooled AUC of 0.913, the
highest of the three; with the recording treated as one surgery it falls to 0.852
and is the lowest. We report the corrected grouping throughout and release the
segment-grouped predictions alongside it, so that the size of the effect can be
checked.

Taken together, the single split and the surgery-grouped cross-validation produce
the same rank order, while the segment-grouped run reverses it, moving
EfficientNet-B0 from last to first. We draw a claim about this dataset rather than
about the literature: here, the ranking was more sensitive to the grouping rule
than to the choice of architecture, which is a reason to state that rule
explicitly and to prefer repeated evaluation over a single split at this sample
size.

### Explanation quality

Saliency maps were scored inside the cross-validation loop, so all 373 pathology
frames (from 102 surgeries) are covered by the fold model that never saw their
surgery. Grad-CAM is computed for all three architectures and ViT-B/16 is
additionally explained with attention rollout, the transformer-native method.

Lesions occupy 7.20% of a frame on average (range 0.17–57.39%), so a map carrying no
spatial information scores 0.072 on Pointing Game and Coverage by construction. For
IoU, which grows with the size of the region a method claims, each method is
compared against a random region of exactly the area it claimed. A centred Gaussian,
normalised and thresholded exactly as a saliency map is, provides a competing
baseline: it encodes the prior that lesions tend to lie mid-frame without looking at
the image at all, and scores IoU 0.135, Pointing Game 0.274 and Coverage 0.119.

Fig. 5 plots these values against the two reference levels.

**Table 3.** Localization against the expert masks (373 out-of-fold frames, 102
surgeries), with 95% surgery-level intervals.

| | IoU | Pointing Game | Coverage |
|---|---|---|---|
| EfficientNet-B0 | **0.156 [0.133–0.180]** | **0.287 [0.233–0.347]** | 0.153 [0.128–0.181] |
| ResNet50 | 0.128 [0.111–0.147] | 0.271 [0.221–0.324] | **0.168 [0.139–0.200]** |
| ViT-B/16, Grad-CAM | 0.039 [0.033–0.047] | 0.078 [0.052–0.105] | 0.070 [0.058–0.084] |
| ViT-B/16, attention rollout | 0.014 [0.010–0.019] | 0.054 [0.030–0.082] | 0.064 [0.051–0.078] |
| *Centred Gaussian baseline* | *0.135* | *0.274* | *0.119* |
| *Map with no spatial information* | *area-matched* | *0.072* | *0.072* |

Unlike the classification comparisons, the differences between methods are large
relative to their uncertainty. All six pairwise IoU differences are significant
after Holm correction within that family; Table 4a lists four of them, the two
omitted being ResNet50 and EfficientNet-B0 against attention rollout (+0.114 and
+0.142, both *p* = 0.0006): EfficientNet-B0 exceeds
ResNet50 by 0.029 (*p* = 0.004), both CNNs exceed ViT-B/16 by 0.088 and 0.117
(*p* = 0.0006), and ViT-B/16's Grad-CAM exceeds its own attention rollout by 0.026
(*p* = 0.0006).

**Table 4a.** Mean IoU differences between methods (family of six, Holm-corrected).

| Comparison | ΔIoU [95% CI] | *p* |
|---|---|---|
| EfficientNet-B0 − ResNet50 | +0.029 [+0.009, +0.049] | 0.004 |
| ResNet50 − ViT-B/16 (Grad-CAM) | +0.088 [+0.072, +0.106] | 0.0006 |
| EfficientNet-B0 − ViT-B/16 (Grad-CAM) | +0.117 [+0.096, +0.140] | 0.0006 |
| ViT-B/16: Grad-CAM − attention rollout | +0.026 [+0.019, +0.033] | 0.0006 |

**Table 4b.** Each method against the two reference levels (family of eight,
Holm-corrected).

| Method | vs area-matched random | vs centred Gaussian |
|---|---|---|
| EfficientNet-B0 | +0.116 [+0.097, +0.136], *p* = 0.0008 | **+0.022 [+0.008, +0.038], *p* = 0.0048** |
| ResNet50 | +0.096 [+0.081, +0.115], *p* = 0.0008 | −0.007 [−0.029, +0.017], *p* = 0.80 |
| ViT-B/16, Grad-CAM | +0.002 [−0.003, +0.006], *p* = 0.80 | −0.095 [−0.115, −0.077], *p* = 0.0008 |
| ViT-B/16, rollout | −0.015 [−0.019, −0.011], *p* = 0.0008 | −0.121 [−0.140, −0.103], *p* = 0.0008 |

Pointing Game and Coverage were tested against their analytic chance level in a
separate family of eight. Both CNNs exceed chance on both measures (ResNet50 +0.199
and +0.096; EfficientNet-B0 +0.215 and +0.081; all *p* = 0.0008). ViT-B/16's
Grad-CAM does not differ from chance on either (+0.006, *p* = 1.00; −0.002,
*p* = 1.00), and attention rollout does not differ on Pointing Game (−0.018,
*p* = 0.39) while falling below chance on Coverage (−0.008, *p* = 0.0008).

Three findings follow. First, both CNNs localize above chance on all three measures,
but only EfficientNet-B0 also exceeds the centred Gaussian baseline (+0.022 IoU,
*p* = 0.0048); ResNet50 is indistinguishable from that baseline on IoU (*p* = 0.80)
and its Pointing Game point estimate sits on the baseline's (0.271 against
0.274), a comparison we did not test. Second, ViT-B/16's
Grad-CAM maps are indistinguishable from chance on every measure we tested. Third,
attention rollout does not rescue the transformer: it is worse than ViT-B/16's own
Grad-CAM and falls below the random floor on IoU and below chance on Coverage. The
failure is therefore not an artefact of applying a CNN-oriented method to a
transformer, though it does not exclude the possibility that some other attribution
method would do better.

Fig. 6 shows frames at the best, median and worst of EfficientNet-B0's IoU
distribution. The CNN maps concentrate on a connected region near the lesion even in
the median case, while both ViT-B/16 maps scatter across unrelated tissue and
instrument surfaces.

Read with Table 2, these results separate two properties that are often assumed to
travel together. The evaluation found no evidence that ViT-B/16 classifies worse
than ResNet50, yet its explanations are indistinguishable from chance, while
EfficientNet-B0 — the one model this evaluation does place below another on AUC —
is also the only one whose explanations exceed a centre-of-image baseline.

### Where the errors concentrate

Errors are not spread evenly over the dataset. Of the surgeries contributing at
least five frames, those with the highest misclassification rate are predominantly
no-pathology recordings (Fig. 7), where every model misclassifies a large share of
the frames. The dominant failure mode is therefore a systematic false positive on
healthy tissue in particular recordings rather than scattered errors or missed
lesions. Since the folds are grouped by surgery these frames were never seen in
training, so the clustering is not memorization; identifying what distinguishes
these recordings — illumination, tissue appearance, or camera and instrument
differences — would require recording-level analysis that we did not perform.
