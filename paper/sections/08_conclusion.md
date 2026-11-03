## Conclusion

We compared ResNet50, EfficientNet-B0 and ViT-B/16 on binary endometriosis
classification in laparoscopic frames under a single protocol: a balanced GLENDA
subset, 5-fold cross-validation grouped by surgery, and uncertainty estimated by
resampling surgeries rather than frames. Alongside classification we measured what
each model's saliency map actually covers, scoring Grad-CAM — and, for the
transformer, attention rollout — against GLENDA's expert lesion masks on every
pathology frame, using the fold models that never saw those surgeries.

Three results follow. First, classification separated the architectures only
marginally, and not robustly. The pre-specified pooled analysis establishes a
single difference, ResNet50 over EfficientNet-B0 (ΔAUC +0.048, *p* = 0.030);
comparing the models inside each fold instead leaves that difference unsupported
and establishes a different one, ViT-B/16 over EfficientNet-B0 (+0.041,
*p* = 0.0066); and the ordering reverses again when the grouping unit is set to
the segment rather than the recording. Two analyst choices — how to aggregate
folds, and what counts as one surgery — each move the answer further than the
architectures move it.

Second, explanation quality did separate them, decisively and in the opposite
direction: EfficientNet-B0 is the only model whose maps exceed a centre-of-image
baseline on IoU, ResNet50's are indistinguishable from that baseline, and
ViT-B/16's are indistinguishable from chance on all three measures — a result
attention rollout does not repair, and one that holds across the full range of
widths we gave the baseline. Unlike the classification ordering, this one did not
move under any of the choices we varied.

Third, the two properties therefore do not order the architectures the same way,
so a model chosen on AUC alone may be the one whose explanations carry least
information.

For practice, two things follow. Where a system displays a saliency map
alongside its prediction, that map should be measured against the annotation and against
chance, not displayed on the assumption that a good classifier explains well. And
where frames come from recordings, the grouping unit should be stated and
justified: setting it one level too fine was enough, in our data, to change which
architecture appeared best.

The study is limited by scale — 122 surgeries — by its balanced evaluation set, and
by the fact that spatial agreement with a mask is not faithfulness. Establishing
why the dissociation arises will need an intervention rather than an observation;
occluding the annotated lesion and measuring what performance each architecture
retains is the most direct test we can suggest.
