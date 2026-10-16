## Introduction

Endometriosis is a chronic inflammatory condition in which tissue resembling the
uterine lining grows outside the uterus — on the peritoneum, the ovaries, and in
advanced disease infiltrating deeper structures. The World Health Organization
estimates that it affects about 10% of reproductive-age women and girls worldwide,
some 190 million people [ref_who]. Its consequences are long-running rather than
acute: chronic pelvic pain, dysmenorrhoea, dyspareunia and infertility, with the
associated cost of repeated consultations, lost work and reduced quality of life
[ref_who], [ref_e3]. It remains comparatively under-researched for a condition of
that reach [ref_e4].

Diagnosis is slow. The WHO puts the average interval between symptom onset and
diagnosis at four to twelve years [ref_who], a delay driven by symptoms that
overlap with other pelvic conditions and by the absence of a reliable blood
biomarker — the 2022 ESHRE guideline recommends against using biomarkers for
diagnosis at all [ref_eshre].

The role of surgery in that pathway has recently changed, and the change matters
for where automated image analysis can help. Until recently, laparoscopy with
histological confirmation was treated as the reference standard, and parts of the
recent literature still describe it that way [ref_e3]. The 2022 ESHRE
guideline explicitly challenges that position: ultrasound or MRI is now recommended
in the diagnostic work-up, and laparoscopy is reserved for patients whose imaging
is negative or whose empirical treatment has failed or is unsuitable [ref_eshre].
Two consequences follow. First, laparoscopy is still performed — both for those
patients and for treatment — so what a surgeon sees through the camera still
determines what is found and excised. Second, and less obviously, the patients who
now reach laparoscopy are enriched for exactly the disease imaging cannot resolve:
superficial peritoneal lesions, which the guideline notes a negative scan does not
exclude [ref_eshre]. Visual recognition during surgery therefore remains
operator-dependent in precisely the cases where it is hardest, which is where
decision support has something to offer.

Deep learning has been applied to this problem from several directions.
Convolutional networks pretrained on natural images have been fine-tuned to
classify laparoscopic frames as pathological or not [ref_a2], [ref_a7] and, in a
preprint, [ref_a5],
region-based detectors have been used to localize lesions [ref_a3], [ref_a6], and
work has begun on recognition during live procedures [ref_a10]. Most of this work
uses GLENDA, the public gynecologic laparoscopy dataset, which provides
expert-annotated lesion masks alongside frame labels [ref_a4]. Vision transformers
have been examined for endometriosis laparoscopic video in a feasibility and
ethics analysis [ref_a9], and reviews of the
field summarise where the evidence currently stands [ref_e1], [ref_e2].

Three problems limit what can be concluded from this literature, and they motivate
what we do here.

The first is **evaluation protocol**. In the v1.5 release used here, GLENDA provides 373 annotated
pathology frames against 13,438 no-pathology frames [ref_glenda_web], so accuracy
on the raw set is close to meaningless: a model that never predicts the positive class scores 97%. Studies
that rebalance and studies that do not are therefore not measuring the same
quantity, and reported figures cannot be compared directly. More seriously, frames
extracted from one surgery are highly correlated, so any split that lets a
recording contribute to both training and test data will overstate performance.
This is easy to introduce accidentally: GLENDA stores no-pathology frames in
per-segment folders, and treating a folder as a group — which looks like correct
grouped validation — allows one recording onto both sides of a split.

The second is **how explanation quality is reported**. Saliency maps are commonly
shown next to predictions. The one prior GLENDA study that pairs classification
with explainability does score its maps against the expert masks, reporting IoU,
Dice and recall [ref_a1], but it compares them against no reference level, and it
reports neither the grouping unit of its cross-validation nor whether the scored
maps come from models that never saw the frame's surgery. The missing reference is
consequential: lesions occupy a small fraction of a frame, so a map with no spatial
information still scores above zero, and a fixed centre-of-image prior can outscore
a real model's map. Without such a reference, a plausible-looking overlap figure
cannot be interpreted.

The third is **what a comparison between architectures can actually establish** at
this sample size. Public laparoscopy data for this task amounts to a few hundred
annotated frames from roughly a hundred surgeries. Differences between
architectures are quantified against between-surgery variability, and treating
frames as independent observations makes those differences look far more certain
than they are.

This study addresses all three. Its contributions are:

1. **A leakage-controlled benchmark on GLENDA with the grouping unit stated
   explicitly.** All 746 frames of a balanced subset are partitioned by surgery
   under 5-fold grouped cross-validation, with the partition checked
   programmatically and released. We also report what happens when the grouping
   unit is set one level too fine, because the effect on the results is large.

2. **A comparison of ResNet50, EfficientNet-B0 and ViT-B/16 under one protocol,
   with uncertainty quantified at the level of the surgery** rather than the frame,
   and with raw and multiplicity-corrected *p*-values reported side by side.

3. **A quantitative evaluation of saliency maps against the expert lesion masks**,
   computed inside the cross-validation loop so that every pathology frame is
   scored by a model that never saw its surgery, and reported against two
   reference levels: an area-matched random region and a centre-of-image prior.
   ViT-B/16 is additionally explained with attention rollout [ref_rollout], so that
   its result does not depend on applying a convolution-oriented method to a
   transformer.

4. **A direct test of whether classification performance and explanation quality
   order the three architectures the same way.** They do not, and that dissociation
   is the study's main finding.

The remainder of the paper is organised as follows. Section 2 reviews related work.
Section 3 describes the dataset, the partitioning rule, the training protocol and
the statistical and explainability analyses. Section 4 reports the results.
Section 5 discusses what they do and do not support, Section 6 sets out the
limitations, and Section 7 concludes.
