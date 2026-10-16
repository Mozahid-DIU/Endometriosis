## Discussion

### Two properties that do not travel together

The central result is a dissociation. The evaluation found no evidence that
ViT-B/16 classifies worse than ResNet50, yet ViT-B/16's saliency maps are
indistinguishable from chance on all three localization measures. EfficientNet-B0
is the one model this evaluation places below another on AUC, yet it is the only
one whose maps exceed a centred-Gaussian baseline on IoU. Ranking by discrimination and
ranking by explanation quality do not agree.

This matters because the two are routinely treated as one. A paper that reports a
strong AUC and prints a few heat maps invites the reading that the model has
learned the lesion. Our measurements show that the inference does not hold here: a model with no
detectable disadvantage in discrimination produced maps with no detectable spatial
signal.

One hypothesis fits both halves of the dissociation and the error analysis, and we
state it as a hypothesis because our experiments do not test it. The errors cluster
in particular no-pathology recordings rather than scattering across the dataset,
which indicates that recording-level appearance carries information the models can
use. A high-capacity model with global receptive fields, such as ViT-B/16, is well
placed to exploit that context, and a decision made from context distributed across
the frame has no compact region to highlight. EfficientNet-B0, with a sixteenth of
the parameters and a convolutional inductive bias, has less room to do so and may
be pushed toward local lesion texture — which would cost a little discrimination
while producing maps that fall on the lesion. Testing this would require
manipulating the available context, for instance by occluding the lesion region and
measuring the performance each architecture retains.

### What the localization numbers do and do not establish

IoU, Pointing Game and Coverage measure spatial agreement between a saliency map
and an expert mask. They do not measure faithfulness. A map that overlaps the
lesion does not prove the model used lesion evidence, and a map that does not
overlap it does not prove the model ignored the lesion: the map may simply be a poor
summary of the computation. The honest claim is narrower than "EfficientNet-B0
reasons about lesions": its maps agree with expert annotations more than the
alternatives do, and ViT-B/16's maps agree no more than chance.

The chance references are what make these numbers interpretable, and we would argue
for them as standard practice. Lesions cover 7.2% of a frame here, so an
uninformative map scores 0.072 on two of the three measures by construction, and an
IoU figure cannot be read without knowing how much area the method claimed. The
centred Gaussian is the more demanding reference and the more revealing one:
ResNet50's maps, which look reasonable to the eye and are significantly above
random, are statistically indistinguishable from a fixed blob in the middle of the
frame. Reported alone, its IoU of 0.128 would have suggested a model that localizes.

Attention rollout deserves a note. It was included to pre-empt the objection that
Grad-CAM is unfair to a transformer, and the result is stronger than expected: the
transformer-native method performs *worse* than Grad-CAM on ViT-B/16, and falls
below the area-matched random floor. Vanilla rollout is class-agnostic — it
aggregates attention flow without reference to the predicted class — so a diffuse
result is consistent with its known behaviour. We therefore do not claim that
ViT-B/16 cannot be explained, only that neither of the two standard options
produces lesion-aligned maps here. Gradient-weighted transformer attribution
remains to be tried.

### How fragile the comparison is

Three evaluation protocols on the same data and the same architectures produced
two different orderings, and the one that differs is the one with the grouping
mistake. A single case-wise split and a cross-validation grouped by recording both
rank ResNet50 first and EfficientNet-B0 last; a cross-validation that treated each
no-pathology segment folder as a unit puts EfficientNet-B0 first. The effect of
that single choice is larger than any difference we can establish between
architectures.

Two practical conclusions follow. First, in this dataset a single train/test split did not support a stable ranking,
so a study of comparable size that reports one should be read as provisional. Second, the grouping unit must be stated and
justified, not assumed: our own segment-level grouping looked rigorous — it was
group-aware cross-validation with assertions — yet it let four recordings place
frames on both sides of a split, which moved EfficientNet-B0's pooled AUC by 0.06.
Datasets that ship frames in per-segment folders invite exactly this error.

We would also argue for surgery-level uncertainty as the default in frame-level
studies. Resampling frames rather than surgeries narrowed our intervals by a factor
of 1.9 to 2.6, which is the difference between reporting two of the three pairwise
AUC differences as significant and reporting one. The inference a
reader cares about is to new patients, and that is the level at which the data are
independent.

### Error structure and what it implies

The surgeries with the highest error rates are predominantly no-pathology
recordings, and in the worst of them, a single no-pathology recording, every model
misclassifies most frames. The
failure mode is a systematic false positive on healthy tissue in particular
recordings rather than missed lesions — all three models recall pathology frames at
0.855 to 0.877. Since folds are grouped by surgery, these frames were never
trained on, so the clustering reflects something that differs between recordings.
We did not measure what: illumination, tissue appearance, instrument presence and
camera hardware are all plausible and all confounded in this dataset.

For a screening or second-opinion application this pattern is more actionable than
an aggregate error rate. It suggests that performance should be reported per
recording rather than pooled, and that the cases to collect next are normal
recordings that differ in acquisition, not simply more frames.

### Clinical reading

These results do not support clinical use, and we make no such claim. Three limits
are decisive. The evaluation set is balanced 1:1 while unselected laparoscopy
frames are closer to 1:36, so the precision reported here is an artefact of the
evaluation design and would fall substantially at realistic prevalence. We did not
assess calibration or decision utility, so the predicted probabilities should not
be read as risk estimates. And the task as posed — a still frame labelled
endometriosis or not — is not the clinical task, in which a surgeon integrates
motion, manipulation and tactile feedback across an entire procedure.

What the results do support is narrower and still useful: under cross-validation grouped by
surgery, frame-level discrimination between endometriosis and no-pathology
laparoscopy frames is feasible with modest data, and recall is the stronger half of
the performance profile, which is the direction a screening aid would want.

The explanation findings carry a specific warning for deployment. If a system
displays a heat map alongside its prediction, that map is part of the clinical
interface, and a map at chance level may not be a neutral addition: it can offer the
appearance of a reason where our measurements find none. Whether clinicians'
confidence is actually affected is an empirical question we did not study. On this dataset, a ViT-B/16 deployed with Grad-CAM would have produced exactly
that. Where explanations are shown to clinicians, their spatial agreement with
expert annotation should be measured and reported, not assumed — and measured
against chance, since otherwise plausible-looking numbers can mean nothing.
