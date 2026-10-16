## Conclusion

We compared ResNet50, EfficientNet-B0 and ViT-B/16 on binary endometriosis
classification in laparoscopic frames under a single protocol: a balanced GLENDA
subset, 5-fold cross-validation grouped by surgery, and uncertainty estimated by
resampling surgeries rather than frames. Alongside classification we measured what
each model's saliency map actually covers, scoring Grad-CAM — and, for the
transformer, attention rollout — against GLENDA's expert lesion masks on every
pathology frame, using the fold models that never saw those surgeries.

Three results follow. First, classification did not separate the architectures:
after correction, the only difference this evaluation establishes is that ResNet50
discriminates better than EfficientNet-B0 (ΔAUC +0.048, *p* = 0.030), and the
ordering of the three models reversed when the grouping unit was set to the
segment rather than the recording. Second, explanation quality did separate them, decisively and in the
opposite direction: EfficientNet-B0 is the only model whose maps exceed a
centre-of-image baseline on IoU, ResNet50's are indistinguishable from that
baseline, and
ViT-B/16's are indistinguishable from chance on all three measures — a result
attention rollout does not repair. Third, the two properties therefore do not order
the architectures the same way, so a model chosen on AUC alone may be the one whose
explanations carry least information.

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
