## Methodology

### Dataset

We use GLENDA v1.5 [ref_a4], a public dataset of frames extracted from gynecologic
laparoscopy recordings. The v1.5 release contains 373 frames showing
endometriosis, each with a pixel-level lesion mask drawn by medical experts, and
13,438 frames from recordings with no visible pathology [ref_glenda_web]. The data are de-identified and publicly released under a
CC BY-NC licence, so no additional ethical approval was required.

The raw class ratio is approximately 1:36. Reporting accuracy on such a split is
uninformative, since a model that never predicts the positive class already scores
97%. We therefore construct a balanced subset: all 373 pathology frames, and 373
no-pathology frames drawn **round-robin across the 27 no-pathology segment
folders**, which correspond to 20 distinct recordings once the segment suffix is
stripped, so that no single recording dominates and consecutive, near-duplicate
frames are avoided. Sampling uses a fixed seed (42) and is reproduced by the script
released with this work.

### Grouping and partitioning

The unit that must not be split is the surgery, not the frame. Frames from one
recording share a patient, a camera, an illumination setting and often the same
lesion, so allowing them into both training and test sets would let a model
recognise the recording rather than the pathology, and would inflate every reported
metric. Group identity is read from the filenames: pathology frames carry a case
identifier (`c_<id>`), and no-pathology frames carry a recording-and-segment
identifier. GLENDA stores no-pathology frames in per-segment folders, so one recording can
appear under several folder names; the segment suffix is stripped, since two
segments of one recording are the same surgery. The balanced subset comprises 746
frames from 122 surgeries: 102 pathology cases and 20 no-pathology recordings.

Partitioning uses `StratifiedGroupKFold` with five folds, which keeps whole groups
together while balancing the class ratio as far as the group sizes allow. Within
each fold, the training portion is split again, group-wise, to provide a validation
set for early stopping; the fold's test portion is never used for model selection.
Every partition is checked programmatically: no group may appear in more than one
of train, validation and test, and the five test portions must tile the dataset
exactly once. Because groups are indivisible, fold composition is not uniform —
test portions contain 125 to 179 frames and pathology prevalence ranges from 37.8%
to 66.4% — and this is accounted for in the choice of evaluation metric below.

A preliminary experiment used a single case-wise 70/15/15 split (528 train, 111
validation, 107 test frames) built by the same grouping rule. We report it in
Section 4.3 as a comparison, not as a result.

### Models and training

Three ImageNet-pretrained architectures are compared, covering two convolutional
designs of very different capacity and one transformer:

- **ResNet50** [ref_resnet] (`IMAGENET1K_V2` weights), 25.6M parameters
- **EfficientNet-B0** [ref_efficientnet] (`IMAGENET1K_V1`), 5.3M parameters
- **ViT-B/16** [ref_vit] (`IMAGENET1K_V1`), 86M parameters

Parameter counts are those reported by torchvision.

In each case the classification head is replaced by a two-unit linear layer and the
whole network is fine-tuned; no layers are frozen. Frames are resized to 224×224 and
normalised with ImageNet statistics. Training augmentation is deliberately mild —
horizontal flip, rotation up to ±15°, and brightness and contrast jitter of 0.2 —
because laparoscopic frames have no canonical orientation but do carry diagnostic
colour information. Validation and test frames receive no augmentation.

All models are trained with Adam, batch size 32, for at most 30 epochs, with early
stopping when validation ROC-AUC has not improved for six epochs; the weights of the
best validation epoch are restored before testing. The learning rate is 1e-4 for the
convolutional networks and 2e-5 for ViT-B/16, the transformer requiring the smaller
step to fine-tune stably on a dataset of this size. Identical settings, seeds and
schedules are used in all 15 fits (3 architectures × 5 folds). Seeds are fixed at 42
and offset by the fold index, and cuDNN is restricted to deterministic algorithms so
that a re-run reproduces the reported numbers. Training ran on a single Tesla T4 GPU
in Google Colab.

### Evaluation

Endometriosis is the positive class throughout. We report accuracy, precision,
recall, F1 and ROC-AUC. Results are summarised in two ways: as the mean and standard
deviation of the five fold-level values, and as **pooled out-of-fold** metrics, in
which the predictions of all five folds are concatenated and each metric is computed
once over the resulting 746 frames. Pooling is possible because every frame is
predicted exactly once, by the model that never saw its surgery; it yields one
estimate over the whole dataset instead of an average of five estimates over fifths
of it.

The **pooled estimate is pre-specified as the primary result**, and every
inferential comparison is made on it; fold means and standard deviations describe
between-fold variability and are not a second estimate of the same quantity.
Pooling requires the five fold models' scores to be comparable on one scale, which
we do not verify directly; the two summaries are therefore reported side by side
rather than treated as interchangeable.

ROC-AUC is treated as the primary metric because it does not depend on the fixed
0.5 threshold, so changes in fold prevalence do not move it the way they move
accuracy and precision. Fold-level AUC still varies with case mix, and we do not
claim it is invariant across folds.

### Statistical analysis

All intervals are 95% percentile bootstrap intervals from 10,000 resamples, and all
tests are two-sided. **The resampling unit is the surgery, not the frame.** Frames
within a recording are correlated, so resampling frames would count repeated views
of one case as independent evidence; since the inference of interest is to new
surgeries, whole groups are resampled, with pathology and no-pathology groups drawn
separately so that both classes are present in every resample. For reference, we
also report how much narrower the frame-level intervals would have been.

Architectures are compared pairwise on the pooled out-of-fold predictions with two
paired bootstrap tests — of the AUC difference and of the accuracy difference — in
which both models are evaluated on the same resampled surgeries. McNemar's exact
test on discordant frames is reported alongside them because the counts are
informative, but it assumes independent pairs and therefore ignores the clustering;
we label it frame-level and do not use it to support claims. *p*-values are
Holm-corrected within each pre-specified family, and raw values are reported too,
since an uncorrected interval excluding zero can sit beside a corrected *p* above
0.05. The families are: the three pairwise classification comparisons; the six
pairwise localization comparisons; the eight localization-versus-reference
comparisons; and the eight comparisons of Pointing Game and Coverage against
chance.

Training used cross-entropy loss without class weighting — the subset is balanced —
and Adam with its torchvision defaults apart from the learning rate; scores are the
softmax probability of the positive class. The validation set inside each fold is
drawn with the same `StratifiedGroupKFold` splitter and seed as the outer loop,
taking the first split of the training portion, which yields 92 to 124 frames
depending on the fold.

### Explainability

Classification metrics say whether a model is right; they say nothing about whether
it is right for the right reason. GLENDA's expert masks allow the second question to
be asked quantitatively, by comparing each model's saliency map with the annotated
lesion.

Grad-CAM [ref_gradcam] is computed at the last
convolutional stage of each CNN (`layer4[-1]` for ResNet50, `features[-1]` for
EfficientNet-B0). For ViT-B/16 the last encoder block's first layer-norm is used,
with token activations reshaped into a 14×14 grid. Grad-CAM was designed for
convolutional feature maps, so ViT-B/16 is additionally explained with **attention
rollout** [ref_rollout], the transformer-native alternative: an identity matrix is
added to each layer's head-averaged attention to account for the residual
connection, each row is renormalised, the layers are multiplied together, and the
class token's row of the product gives each patch's contribution. Both maps are
normalised to [0, 1] and upsampled to 224×224.

Maps are scored against the binarised mask on three measures:

- **IoU** between the mask and the region where the map exceeds 0.5
- **Pointing Game**: whether the map's single highest point lies inside the mask
- **Coverage**: the share of the map's total intensity that falls inside the mask

These numbers are meaningless without a reference, because lesions occupy only 7.20%
of a frame on average. An uninformative map scores the mask's area fraction on both
Pointing Game and Coverage. IoU needs more care, since it grows with the size of the
region a method claims: each method is therefore compared against a random region of
**exactly the area that method claimed**. Writing *p* for the fraction of the frame
a method's thresholded map covers and *m* for the fraction the mask covers, a region
of area *p* placed independently of the mask overlaps it by *pm* in expectation, so
its expected IoU is *pm*/(*p*+*m*−*pm*). The construction assumed is a spatially
uniform random region, not a connected blob; we confirmed the closed form by
thresholding uniform noise at the quantile that reproduces area *p* and measuring
the IoU directly: over 25 frames at 400 random placements each, the
analytic floor averages 0.0330 and the simulated floor 0.0330 (released as
`results/iou_floor_validation.csv`). As a second, competing baseline we score a centred Gaussian
(σ = 0.25×224), min-max normalised to [0, 1] and thresholded at the same 0.5 as
every method, so its claimed area follows from the same rule rather than being
matched to anything. It encodes the prior that lesions tend to appear mid-frame
without looking at the image at all. Exceeding it is additional evidence for
lesion-specific localization; failing to exceed it means a method has not been shown
to do more than that prior.

Saliency maps are computed inside the cross-validation loop, immediately after each
fold's model is evaluated and before it is discarded, so that all 373 pathology
frames are scored by the same models that produced the classification results, each
on data from surgeries it never saw.

### Reproducibility

The dataset-construction script, the training and evaluation notebook, the split
manifest, the per-image predictions and saliency scores, and every analysis and
figure script are released at `github.com/Mozahid-DIU/Endometriosis`. Each run
records a hash of its frame-to-fold assignment, so results computed under different
groupings cannot be combined by accident. Seeds are fixed and cuDNN is restricted to
deterministic algorithms, which makes a re-run reproducible **within the tested
environment** (Google Colab, Tesla T4, PyTorch 2.x); we do not claim bitwise
reproducibility on other hardware or library versions.
