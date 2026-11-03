"""Compare the architectures inside each fold, where score scale cannot confound.

The manuscript's primary endpoint is pooled out-of-fold AUC: the five folds'
predictions are concatenated and ranked together. That requires the five fold
models to put their scores on one scale, which we never verified - and the two
summaries disagree about who is best:

    fold-mean AUC   ViT-B/16 0.925 > ResNet50 0.903 > EfficientNet-B0 0.884
    pooled AUC      ResNet50 0.900 > ViT-B/16 0.896 > EfficientNet-B0 0.852

Look at how far each model falls from one to the other: ResNet50 -0.003,
ViT-B/16 -0.029, EfficientNet-B0 -0.032. Pooling costs almost nothing to the
model whose scores are already comparable across folds, and costs the other two.
That is a property of score calibration, not of how well a model separates
endometriosis from healthy tissue, and it pushes the pooled ranking toward
ResNet50 for a reason that has nothing to do with the task.

This computes AUC inside each fold, where only one model pair and one scale are
ever compared, and aggregates the five fold-level differences. Nothing is
pooled across folds, so cross-fold score scale cannot enter.

Aggregation. The primary figure is the unweighted mean of the five fold-level
differences: each fold is one complete evaluation on its own disjoint set of
surgeries, and weighting by size would let the largest fold speak loudest about
a quantity - AUC - whose precision depends on the smaller class, not on the
total. The frame-weighted mean is reported beside it as a check.

Uncertainty. Surgeries are resampled *within* each fold, separately by class, so
the fold structure and the clustering both survive the bootstrap and every
resampled fold still contains both classes. 10,000 resamples, two-sided, Holm
correction across the three pairs, Monte-Carlo floor 1/(B+1).

Usage (from the repo root):
    python scripts/within_fold_auc.py
Output:
    results/within_fold_auc.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from statistical_analysis import fast_auc
from cluster_bootstrap import MODELS, N_BOOT, SEED, ci, holm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PRED = RESULTS / "cv_predictions_v3.csv"
OUT_CSV = RESULTS / "within_fold_auc.csv"


def load() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    rows = list(csv.DictReader(open(PRED, encoding="utf-8")))
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["filename"]] = r
    files = sorted(set.intersection(*(set(v) for v in by_model.values())))

    first = by_model[MODELS[0]]
    y = np.array([int(first[f]["y_true"]) for f in files])
    fold = np.array([int(first[f]["fold"]) for f in files])
    surgery = np.array([first[f]["surgery_id"] for f in files])
    probs = {m: np.array([float(by_model[m][f]["prob"]) for f in files])
             for m in MODELS}
    return files, y, fold, surgery, probs


def fold_resamples(fold: np.ndarray, surgery: np.ndarray, y: np.ndarray,
                   rng) -> list[np.ndarray]:
    """Resample whole surgeries inside each fold, drawing classes separately.

    A surgery sits in exactly one fold and is class-pure, so this keeps every
    resampled fold evaluable and never moves a surgery between folds.
    """
    strata: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    for f in sorted(set(fold.tolist())):
        here = np.flatnonzero(fold == f)
        by_surgery: dict[str, list[int]] = defaultdict(list)
        for i in here:
            by_surgery[surgery[i]].append(i)
        groups = {g: np.asarray(v) for g, v in by_surgery.items()}
        pos = [g for g, idx in groups.items() if y[idx].max() == 1]
        neg = [g for g, idx in groups.items() if y[idx].max() == 0]
        strata.append((groups, pos, neg))

    out = []
    for _ in range(N_BOOT):
        per_fold = []
        for groups, pos, neg in strata:
            picked = list(rng.choice(pos, len(pos), replace=True)) + \
                     list(rng.choice(neg, len(neg), replace=True))
            per_fold.append(np.concatenate([groups[g] for g in picked]))
        out.append(per_fold)
    return out


def fold_aucs(y: np.ndarray, prob: np.ndarray,
              index_per_fold: list[np.ndarray]) -> np.ndarray:
    return np.array([fast_auc(y[i], prob[i]) for i in index_per_fold])


def main() -> None:
    files, y, fold, surgery, probs = load()
    folds = sorted(set(fold.tolist()))
    index_per_fold = [np.flatnonzero(fold == f) for f in folds]
    sizes = np.array([len(i) for i in index_per_fold], dtype=float)
    weights = sizes / sizes.sum()

    print(f"{len(files)} frames, {len(set(surgery))} surgeries, {len(folds)} folds")
    print(f"fold sizes {sizes.astype(int).tolist()}, "
          f"pathology share "
          f"{[round(float(y[i].mean()), 3) for i in index_per_fold]}\n")

    print("=== AUC inside each fold ===")
    print(f"{'':16s} " + " ".join(f"fold {f}" for f in folds) +
          "   mean    pooled   drop")
    observed: dict[str, np.ndarray] = {}
    out: list[list] = []
    for m in MODELS:
        per_fold = fold_aucs(y, probs[m], index_per_fold)
        observed[m] = per_fold
        pooled = fast_auc(y, probs[m])
        mean = float(per_fold.mean())
        print(f"{m:16s} " + " ".join(f"{a:6.3f}" for a in per_fold) +
              f"  {mean:6.3f}  {pooled:6.3f}  {pooled - mean:+6.3f}")
        out.append(["fold_auc", m, "", f"{mean:.4f}", f"{pooled:.4f}",
                    f"{pooled - mean:+.4f}", "", "", "",
                    " ".join(f"{a:.4f}" for a in per_fold)])

    print("\nThe drop column is what pooling costs each model: a model whose five")
    print("fold models disagree about what a score of 0.7 means loses the most.\n")

    rng = np.random.default_rng(SEED)
    samples = fold_resamples(fold, surgery, y, rng)

    print("=== paired within-fold AUC differences, surgeries resampled in fold ===")
    pairs = list(combinations(MODELS, 2))
    raw: list[float] = []
    detail: list[tuple] = []
    for a, b in pairs:
        per_fold = observed[a] - observed[b]
        point = float(per_fold.mean())
        weighted = float((per_fold * weights).sum())

        diffs = np.empty(N_BOOT)
        for k, per_fold_idx in enumerate(samples):
            da = fold_aucs(y, probs[a], per_fold_idx)
            db = fold_aucs(y, probs[b], per_fold_idx)
            diffs[k] = float((da - db).mean())
        lo, hi = ci(diffs)
        p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
        p = min(max(p, 1.0 / (N_BOOT + 1)), 1.0)
        raw.append(p)
        detail.append((a, b, point, weighted, lo, hi, per_fold))

    for (a, b, point, weighted, lo, hi, per_fold), p_raw, p_holm in zip(
            detail, raw, holm(raw)):
        wins = int((per_fold > 0).sum())
        verdict = ("established" if p_holm < 0.05 else
                   "inconclusive" if lo * hi > 0 else "no evidence")
        print(f"{a} vs {b}")
        print(f"   mean dAUC {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"p_raw={p_raw:.4g}  p_holm={p_holm:.4g}  ({verdict})")
        print(f"   per fold  " + " ".join(f"{d:+.3f}" for d in per_fold) +
              f"   {a} ahead in {wins}/{len(per_fold)}")
        print(f"   frame-weighted mean {weighted:+.4f}")
        out.append(["within_fold_diff", a, b, f"{point:.4f}", f"{lo:.4f}",
                    f"{hi:.4f}", f"{p_raw:.4g}", f"{p_holm:.4g}", verdict,
                    " ".join(f"{d:.4f}" for d in per_fold)])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "model_a", "model_b", "value", "lo_or_pooled",
                    "hi_or_drop", "p_raw", "p_holm", "verdict", "per_fold"])
        w.writerows(out)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
