## Limitations and Future Work

**Scale and statistical power.** The balanced subset contains 746 frames from 122
surgeries, which is what the public data allow. Because the resampling unit is the
surgery, the effective sample size for every comparison is of that order, and only
one classification difference survived correction. A study of this size can
establish that explanation quality differs sharply between architectures; it cannot
settle which of them classifies best. Larger multi-centre collections, or
federated evaluation across institutions, would be needed for that.

**Evaluation on a balanced subset.** We rebalanced the data 1:1 so that accuracy
and AUC would be informative, but the source distribution is roughly 1:36. The
precision reported here is a property of the evaluation design and would fall at
realistic prevalence; nothing in this study estimates the positive predictive value
a deployed system would have.

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

**Protocol dependencies.** Pooling out-of-fold predictions assumes the five fold
models produce scores on comparable scales; we report the fold-wise summary beside
the pooled one rather than verifying that assumption. Learning rates were set per
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
