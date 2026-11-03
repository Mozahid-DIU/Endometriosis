## Abstract

Deep learning is increasingly applied to laparoscopic images of endometriosis, and
saliency maps are routinely shown alongside the predictions. Whether those maps
actually fall on the lesion is rarely measured. We compared ResNet50,
EfficientNet-B0 and ViT-B/16 on a balanced 746-frame subset of GLENDA under 5-fold
cross-validation grouped by surgery, with confidence intervals obtained by
resampling surgeries rather than frames, and we scored each model's Grad-CAM maps —
and, for the transformer, attention rollout — against the dataset's expert lesion
masks on all 373 pathology frames, using the fold models that never saw those
surgeries. Classification barely separated the architectures, and which single
difference it did support depended on how the folds were aggregated: pooled
out-of-fold AUC was 0.900 for ResNet50, 0.896 for ViT-B/16 and 0.851 for
EfficientNet-B0, giving ResNet50 over EfficientNet-B0 (+0.048, Holm
*p* = 0.030), while comparing the models inside each fold instead left that
difference unsupported (+0.018, *p* = 0.17) and supported ViT-B/16 over
EfficientNet-B0 (+0.041, *p* = 0.0066). Explanation quality separated them
clearly, decisively, and in the opposite direction. Against an area-matched
random region and a
centre-of-image prior, EfficientNet-B0 was the only model to exceed both
(IoU 0.156, *p* = 0.0048 against the prior), ResNet50 was indistinguishable from it
(0.128 against 0.135, *p* = 0.80), and ViT-B/16 was indistinguishable from chance on IoU,
Pointing Game and Coverage; attention rollout performed worse still. That
ordering is stable: it holds across all three measures and across the full range
of widths we gave the centre prior. Ranking by discrimination and ranking by
explanation quality therefore disagree, and only the second ranking is firm. We
also show that grouping frames by recording segment rather than by recording — a
plausible reading of how the dataset is distributed — introduces leakage
sufficient to change which architecture appears best.

**Keywords:** endometriosis; laparoscopy; explainable AI; Grad-CAM; attention
rollout; vision transformer; grouped cross-validation; medical image analysis
