## Limitations and Future Work

**Scale and statistical power.** The balanced subset contains 746 frames from 122
surgeries, which is what the public data allow. Because the resampling unit is the
surgery, the effective sample size for every comparison is of that order, and at
most one classification difference survived correction under either aggregation
rule — a different one in each case. A study of this size can establish that
explanation quality differs sharply between architectures; it cannot settle which
of them classifies best. Larger multi-centre collections, or federated evaluation
across institutions, would be needed for that.

**How few recordings the negative class comes from.** The 122 surgeries are not
evenly informative. The 373 pathology frames come from 102 distinct cases, but all
373 no-pathology frames come from just 20 recordings. Everything the models could
learn about healthy tissue — its illumination, its camera, its instruments, the
way one surgeon moves — is therefore drawn from twenty examples, and our own error
analysis finds the errors concentrating in particular no-pathology recordings
rather than spreading out. The negative class is the binding constraint on
generalization here, more than the frame count or the 122 figure suggests, and
the most valuable additional data would be no-pathology recordings that differ in
acquisition.

**Evaluation on a balanced subset.** We rebalanced the data 1:1 so that accuracy
and precision would be interpretable; AUC does not require it, being
threshold-free and insensitive to prevalence, and that is why we made AUC the
primary metric. The source distribution is roughly 1:36, so the precision
reported here is a property of the evaluation design; nothing in this study
estimates the positive predictive value a deployed system would have, which would
depend on a prevalence we do not know.

**Which aggregation rule the comparison rests on.** Pooling out-of-fold
predictions and comparing models inside each fold are both defensible, and they
disagree here about which pairwise difference the data support (Section 4.3). We
pre-specified pooling and report it as primary rather than selecting whichever
analysis gives the cleaner result, but the honest summary is that this dataset
does not determine a classification ordering. A reader who prefers the
within-fold analysis should read the classification results as supporting
ViT-B/16 over EfficientNet-B0 and nothing else.

**Task definition.** Frames are labelled only as pathological or not. We do not
distinguish lesion type, stage or severity, all of which matter clinically, and we
do not use video: a surgeon integrates motion, probing and tactile feedback that a
still frame discards. Nor did we assess calibration or decision utility, so the
predicted probabilities should not be read as risk estimates.

**What the explanation metrics measure.** IoU, Pointing Game and Coverage quantify
spatial agreement between a saliency map and an expert mask. They do not establish
faithfulness: a map that lands on the lesion does not prove the model used lesion
evidence, and a map that does not may simply summarise the computation poorly.
GLENDA provides one annotation per frame, so we cannot report inter-annotator
variability or treat the mask as a range rather than a point. We also fixed the
map threshold at 0.5 and used a single target layer per architecture; both choices
affect the absolute numbers. The ordering of the two CNNs above both transformer
methods is stable across all three metrics, but the two CNNs themselves change
places on Coverage (ResNet50 0.168 against EfficientNet-B0 0.153) while IoU and
Pointing Game put EfficientNet-B0 first. Only two explanation methods were tested, and the
failure of attention rollout does not rule out gradient-weighted transformer
attribution performing better.

**Protocol dependencies.** We never verified that the five fold models place
their scores on a comparable scale, which is what pooling assumes. The
within-fold analysis shows how much that assumption moves the answer, but it
does not establish the assumption, and the fold-level AUCs it rests on are each
computed on 125 to 179 frames. Learning rates were set per
architecture in advance rather than tuned per fold, which keeps the comparison
simple but may not place each architecture at its best. Finally, the exact
partition depends on the `StratifiedGroupKFold` implementation: running the same
seed under a different scikit-learn version produced different folds, with
prevalence ranging differently across them. We release the split manifest and a
hash of the frame-to-fold assignment so that the reported numbers can be
reproduced, but the folds themselves are not canonical.

**Directions.** The error analysis points to recording-level appearance as the
dominant source of failure, so the most useful next data are no-pathology
recordings that differ in acquisition rather than more frames from the same ones.
The dissociation we report invites a causal test: occluding the annotated lesion
and measuring how much performance each architecture retains would show directly
how much of each model's decision rests on the lesion rather than its surroundings.
