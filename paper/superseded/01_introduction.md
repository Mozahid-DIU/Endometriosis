# Comparative Analysis of CNN and Vision Transformer Architectures for Explainable Endometriosis Classification in Laparoscopic Images

## 1. Introduction

> ⚠️ **Citation note:** `[key]` markers map to `references.md`. Verify every factual number
> (prevalence, delay, accuracies) against the cited source before submission.
> Intro uses ~12 references, one component per paragraph (per required format).

### Importance of the Research Topic
Endometriosis is a chronic, estrogen-dependent inflammatory disorder in which tissue resembling
the endometrial lining grows outside the uterine cavity—on the peritoneum, ovaries, uterus, and,
in advanced cases, as deep infiltrating endometriosis. It affects an estimated **10% of women of
reproductive age** worldwide, translating to roughly 190 million individuals, and is a leading
cause of chronic pelvic pain, dysmenorrhea, dyspareunia, and infertility [E1], [E3]. Beyond its
clinical burden, the condition carries a substantial socioeconomic cost through lost
productivity, repeated medical consultations, and diminished quality of life, making it a
pressing women's-health research priority that has historically received disproportionately
little attention [E3], [D1]. Improving the speed and accuracy of endometriosis detection
therefore has direct real-world significance for a large, underserved patient population.

### Importance of the Specific Problem
Despite its prevalence, endometriosis is exceptionally difficult to diagnose, with patients
experiencing an average **diagnostic delay of 7–10 years** between symptom onset and definitive
diagnosis [E3], [D1]. This delay arises from non-specific, overlapping symptoms, the absence of a
reliable non-invasive biomarker, and continued reliance on **laparoscopy—an invasive surgical
procedure—as the diagnostic gold standard** [A3]. During laparoscopy, accurate identification of
lesions depends heavily on the operating surgeon's experience, producing inter-observer
variability and a real risk of missing subtle or atypical lesions [A3], [A10]. The consequences
of delayed or missed diagnosis are severe: disease progression, avoidable repeat surgeries, and
prolonged suffering [E4]. An effective, reliable, and objective decision-support tool to assist
lesion recognition during laparoscopy is thus a clear and urgent clinical need.

### Role of AI / Machine Learning / Deep Learning in the Domain
Artificial intelligence has emerged as a transformative aid for computer-assisted diagnosis in
gynecologic imaging. Convolutional Neural Networks (CNNs) have demonstrated strong performance in
classifying and localizing endometriotic lesions, with reported accuracies of **0.89–0.93** on
the public Gynecologic Laparoscopy ENdometriosis DAtaset (GLENDA) [A1], [A2], and meta-analytic
evidence confirms deep learning as the strongest-performing family of methods across imaging
modalities [E2]. More recently, Vision Transformers (ViTs)—which capture global context through
self-attention rather than local convolution—have shown competitive and sometimes superior
performance in surgical image tasks [A9]. Collectively, these technologies can reduce
observer dependence, standardize interpretation, and shorten the diagnostic pathway, positioning
AI as a meaningful contributor to earlier and more consistent endometriosis diagnosis.

### Existing Limitations and Proposed Contribution
Nevertheless, existing image-based studies share several limitations. First, GLENDA is
**severely class-imbalanced** (373 pathology versus 13,438 non-pathology frames), yet many works
report accuracy on imbalanced data, yielding optimistic and potentially misleading results.
Second, most models operate as **black boxes**, reporting predictive metrics without
demonstrating *where* the model attends or whether that attention aligns with the true lesion,
which undermines clinical trust [A1], [E4]. Third, comparisons between CNN and Transformer
families are often performed without a fair, leakage-free protocol, and frames from the same
surgery frequently leak across training and test sets, inflating reported performance. To address
these gaps, this study proposes a fair, leakage-controlled comparison of CNN and ViT
architectures on a **balanced** GLENDA subset, augmented with a *quantitative* explainability
analysis that measures the overlap between Grad-CAM attention [X1] and GLENDA's ground-truth
lesion masks [A4].

**The main contributions of this work are:**

1. **A reproducible, leakage-free balanced benchmark** on GLENDA for binary endometriosis
   classification, using **patient/case-wise splitting** so that frames from the same surgery
   never appear in both training and test sets—an issue overlooked in much prior work.
2. **A fair comparison of CNN (ResNet50, EfficientNet-B0) and Vision Transformer (ViT-Base/16)**
   architectures under identical transfer-learning, augmentation, and cross-validation protocols,
   evaluated with a complete metric suite (accuracy, precision, recall, F1, ROC-AUC) and
   statistical significance testing.
3. **A quantitative explainability evaluation** comparing Grad-CAM attention maps against
   ground-truth lesion masks (IoU-based overlap), assessing not merely *whether* a model is
   accurate but *whether it attends to clinically relevant regions*.

These contributions directly address the identified limitations—imbalance, data leakage, and
opaque decision-making—and thereby strengthen the reliability and clinical credibility of
automated endometriosis classification.

The remainder of this paper is organized as follows. Section 2 reviews related work, organized by
category. Section 3 describes the dataset and proposed methodology. Section 4 presents results and
statistical analysis. Section 5 discusses findings, limitations, and threats to validity, and
Section 6 concludes with future directions.
