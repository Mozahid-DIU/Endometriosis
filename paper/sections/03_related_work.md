## Related Work

### Non-image approaches

Several groups predict endometriosis without imaging at all. Symptom
questionnaires analysed with standard classifiers have been proposed as a
screening route that avoids surgery entirely [ref_d1], [ref_d2], and transcriptomic
and multi-omic signatures have been used to build predictive models of disease
status [ref_m3], [ref_m4]. Retrospective clinical records and ultrasound findings
have been mined in the same spirit [ref_m1], [ref_m2]. These approaches address a different point in the
pathway from the one studied here: they aim to decide who should be investigated,
whereas image-based work aims to support interpretation once a patient is already
in surgery. The two are complementary, and the evaluation problems differ — a
symptom model is limited mainly by label quality, an image model by how its data
are partitioned.

### Deep learning on laparoscopic images

Laparoscopy is where the diagnosis is confirmed, and it is the modality closest to
our task. Most published work uses transfer learning with ImageNet-pretrained
convolutional networks on the GLENDA dataset [ref_a4], either classifying frames as
pathological or not [ref_a2] and, in a preprint, [ref_a5], or localizing lesions
with region-based detectors such as Faster R-CNN and Mask R-CNN [ref_a3]. Related
work on comparable laparoscopic collections reports similar pipelines without
naming a public dataset [ref_a7]. Detector
comparisons on larger private collections have also been reported [ref_a6], and
early proof-of-concept work has looked at recognition during live procedures
[ref_a10].

Reported figures across these studies are difficult to place side by side. The
papers differ in how frames are sampled, whether the evaluated set is balanced,
and — critically — whether frames from one surgery can appear in both training and
test data. GLENDA is strongly imbalanced as distributed: the release used here
contains 373 pathology frames against 13,438 no-pathology frames, a ratio of
roughly 1:36, so a model that never predicts the positive class already scores 97%
accuracy. Studies that report accuracy on the imbalanced set and studies that
rebalance it are therefore not measuring the same thing. Our own results
(Section 4.3) show that the partitioning choice alone can reorder three
architectures on a single dataset, which suggests that cross-study comparisons of
headline numbers should be treated with caution.

### Other imaging modalities

Outside laparoscopy, ultrasound and MRI have received sustained attention.
Convolutional models have been applied to transvaginal ultrasound to classify
endometriosis against a surgically confirmed reference standard [ref_b1], and a
recent systematic review surveys that line of work [ref_b4]. On MRI, deep segmentation has been used to delineate
deep infiltrating lesions [ref_a8], a multi-modal pelvic MRI dataset with
segmentation baselines has been released [ref_c1], and a reading-support system has
been developed for clinical use [ref_c2]. These modalities face a different
constraint from ours: the lesion is harder to see, but acquisition is more
standardised. Laparoscopy inverts that — the lesion is directly visible, while
illumination, camera and surgical technique vary from recording to recording, which
is exactly the variation our error analysis traces (Section 4.5).

### Transformers in surgical imaging

Vision transformers [ref_vit] have been examined for surgical video, including a
feasibility analysis specific to endometriosis that also raises the ethical
questions deployment would pose [ref_a9]. Transformer architectures are attractive
here because global self-attention can integrate evidence spread across a frame,
which is plausibly useful when lesion appearance depends on surrounding tissue.
That same property, however, complicates explanation: there is no convolutional
feature map on which a class-activation method naturally operates, and attention
weights are not by themselves an attribution. Attention rollout [ref_rollout]
addresses this by propagating attention through the residual stream, and we
evaluate it alongside Grad-CAM rather than assuming either is adequate.

### Explainability in medical imaging

Grad-CAM [ref_gradcam] is the most widely used visual explanation in this
literature, with LIME [ref_lime] and SHAP [ref_shap] serving similar roles for
tabular and model-agnostic settings. Surveys of explainable medical imaging stress
transparency and clinician trust [ref_x4], and a recent meta-analysis argues that
post-hoc maps must be validated rather than displayed, reporting poor fidelity and
stability for SHAP, LIME and Grad-CAM [ref_x5],
and the one prior GLENDA study that combines classification with explainability
[ref_a1] does score its maps against the expert masks, reporting IoU, Dice and
recall. What is essentially absent from this literature is any comparison of those
overlap figures against a chance level. Without such a reference an IoU figure
cannot be interpreted: when lesions occupy a small fraction of the frame, a map
that carries no spatial information still scores above zero, and a fixed
centre-of-image prior can score higher than a model's map. Section 4.4 reports both
references, and one of our models turns out not to exceed the simpler of them.

### What is missing

Three gaps follow from the above, and they define what this study contributes.

First, **comparisons between CNN and transformer families are made under protocols
that differ between papers, and the grouping unit is seldom stated**. A prior
GLENDA study does compare four architectures under one 5-fold protocol [ref_a1];
what it does not report is how frames were grouped, which decides whether the
estimate is honest. We state the grouping unit, verify it programmatically, and
quantify uncertainty at the level of the surgery rather than the frame.

Second, **explanation quality is reported qualitatively or not at all**. We measure
the agreement between each model's saliency map and the expert lesion masks that
GLENDA provides, on every pathology frame, using out-of-fold models — and we
measure it against both a chance level and a centre-of-image baseline, so that the
numbers can be read.

Third, **leakage at the level of the surgery is easy to introduce and seldom
documented**, though a recent detector study does address it directly [ref_a6]. GLENDA distributes no-pathology frames in per-segment folders, and
treating a folder as a group — which looks like correct grouped validation — allows
one recording to contribute frames to both sides of a split. We report what that
does to the results on this dataset, because the effect is large enough to change
which architecture appears best.
