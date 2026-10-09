"""Analyse the 5-fold grouped cross-validation output.

The single held-out split put ResNet50 (AUC 0.915) well ahead of ViT-B/16 (0.860)
and EfficientNet-B0 (0.833). Cross-validation has to answer whether that ordering
is real or an artefact of one lucky partition, so this script reports:

  1. per-fold prevalence - fold accuracy is uninterpretable without it, because
     the grouped folds are not equally balanced
  2. mean +- std across folds, for every metric
  3. pooled out-of-fold metrics: each of the 746 images predicted once, by a model
     that never saw its case. One tighter estimate instead of five noisy ones.
  4. paired significance on the pooled predictions - McNemar on the errors and a
     paired bootstrap on the AUC difference, the same tests used for the single
     split so the two analyses are directly comparable
  5. single split vs cross-validation, side by side

Usage (from the repo root):
    python scripts/analyze_cv.py
Outputs:
    results/cv_analysis_summary.csv
    results/cv_analysis_pairwise.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from statistical_analysis import fast_auc, fast_metrics

ROOT = Path(__file__).resolve().parent.parent
FOLD_CSV = ROOT / "results" / "cv_fold_metrics.csv"
PRED_CSV = ROOT / "results" / "cv_predictions.csv"
SINGLE_CSV = ROOT / "results" / "test_results.csv"
OUT_SUMMARY = ROOT / "results" / "cv_analysis_summary.csv"
OUT_PAIRWISE = ROOT / "results" / "cv_analysis_pairwise.csv"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05


def load_folds() -> dict[str, dict[int, dict[str, float]]]:
    """model -> fold -> metrics, keeping the last row when a fold was re-run."""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for r in csv.DictReader(open(FOLD_CSV, encoding="utf-8")):
        out[r["model"]][int(r["fold"])] = {m: float(r[m]) for m in METRICS} | {
            "n_train": float(r["n_train"]), "n_test": float(r["n_test"]),
            "val_AUC": float(r["val_AUC"])}
    return out


def load_predictions() -> dict[str, dict[str, tuple[int, float, int, int]]]:
    """model -> filename -> (y_true, prob, pred, fold); later rows win."""
    out: dict[str, dict[str, tuple[int, float, int, int]]] = defaultdict(dict)
    for r in csv.DictReader(open(PRED_CSV, encoding="utf-8")):
        out[r["model"]][r["filename"]] = (
            int(r["y_true"]), float(r["prob"]), int(r["pred"]), int(r["fold"]))
    return out


def paired_bootstrap_auc(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    observed = fast_auc(y, pa) - fast_auc(y, pb)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        yy = y[idx]
        diffs[i] = fast_auc(yy, pa[idx]) - fast_auc(yy, pb[idx])
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return float(observed), float(lo), float(hi), min(p, 1.0)


def bootstrap_ci(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        vals[i] = fast_auc(y[idx], prob[idx])
    return tuple(np.percentile(vals, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))


def main() -> None:
    folds, preds = load_folds(), load_predictions()
    rows_summary, rows_pairwise = [], []

    # --- 1. fold composition
    print("=== fold composition (from the pooled predictions) ===")
    ref = preds[MODELS[0]]
    per_fold: dict[int, list[int]] = defaultdict(list)
    for y_true, _, _, fold in ref.values():
        per_fold[fold].append(y_true)
    for fold in sorted(per_fold):
        ys = per_fold[fold]
        print(f"  fold {fold}: n={len(ys):3d}  endometriosis={sum(ys):3d} "
              f"({sum(ys) / len(ys):.1%})")
    prevalences = [sum(v) / len(v) for v in per_fold.values()]
    print(f"  prevalence ranges {min(prevalences):.1%} to {max(prevalences):.1%} across folds, "
          f"so accuracy and precision are not comparable fold to fold; AUC is.")

    # --- 2. mean +- std across folds
    print("\n=== 5-fold cross-validation (mean +- std) ===")
    for name in MODELS:
        got = folds.get(name, {})
        if len(got) != 5:
            print(f"  {name}: only {len(got)} folds present - rerun the missing ones")
        vals = {m: np.array([got[f][m] for f in sorted(got)]) for m in METRICS}
        print(f"{name:16s} " + "  ".join(
            f"{m} {vals[m].mean():.3f}+-{vals[m].std(ddof=1):.3f}" for m in METRICS))
        for m in METRICS:
            rows_summary.append(["cv_mean_std", name, m, f"{vals[m].mean():.4f}",
                                 f"{vals[m].std(ddof=1):.4f}",
                                 " ".join(f"{v:.4f}" for v in vals[m])])

    # --- 3. pooled out-of-fold
    print("\n=== pooled out-of-fold (every image predicted by a model blind to its case) ===")
    common = set.intersection(*(set(preds[m]) for m in MODELS if m in preds))
    files = sorted(common)
    print(f"  {len(files)} images covered by all three models")
    y = np.array([preds[MODELS[0]][f][0] for f in files])
    probs = {m: np.array([preds[m][f][1] for f in files]) for m in MODELS}
    preds_bin = {m: np.array([preds[m][f][2] for f in files]) for m in MODELS}
    for name in MODELS:
        acc, prec, rec, f1 = fast_metrics(y, preds_bin[name])
        auc = fast_auc(y, probs[name])
        lo, hi = bootstrap_ci(y, probs[name])
        print(f"{name:16s} AUC {auc:.4f} [{lo:.4f}-{hi:.4f}]  acc {acc:.4f}  "
              f"recall {rec:.4f}  F1 {f1:.4f}")
        rows_summary.append(["pooled_oof", name, "AUC", f"{auc:.4f}",
                             f"[{lo:.4f},{hi:.4f}]", f"n={len(files)}"])
        for label, value in (("Accuracy", acc), ("Precision", prec),
                             ("Recall", rec), ("F1", f1)):
            rows_summary.append(["pooled_oof", name, label, f"{value:.4f}", "", ""])

    # --- 4. paired tests on the pooled predictions
    print("\n=== paired tests on the pooled out-of-fold predictions ===")
    for a, b in combinations(MODELS, 2):
        ca, cb = preds_bin[a] == y, preds_bin[b] == y
        only_a = int(np.sum(ca & ~cb))
        only_b = int(np.sum(~ca & cb))
        n = only_a + only_b
        p_mc = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0
        d, lo, hi, p_auc = paired_bootstrap_auc(y, probs[a], probs[b])
        print(f"{a} vs {b}: only-{a}-right={only_a} only-{b}-right={only_b} "
              f"McNemar p={p_mc:.4g} | dAUC {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p_auc:.4g}")
        rows_pairwise.append([a, b, only_a, only_b, f"{p_mc:.4g}",
                              "yes" if p_mc < ALPHA else "no", f"{d:+.4f}",
                              f"{lo:+.4f}", f"{hi:+.4f}", f"{p_auc:.4g}",
                              "yes" if p_auc < ALPHA else "no"])

    # --- 5. single split vs cross-validation
    if SINGLE_CSV.exists():
        single = {r[""]: float(r["AUC"]) for r in csv.DictReader(open(SINGLE_CSV, encoding="utf-8"))}
        print("\n=== single split vs cross-validation (AUC) ===")
        print(f"{'model':16s} {'single split':>13s} {'CV mean':>9s} {'pooled OOF':>11s}")
        for name in MODELS:
            cv_mean = np.mean([folds[name][f]["AUC"] for f in sorted(folds[name])])
            print(f"{name:16s} {single.get(name, float('nan')):13.4f} {cv_mean:9.4f} "
                  f"{fast_auc(y, probs[name]):11.4f}")
            rows_summary.append(["single_vs_cv", name, "AUC",
                                 f"{single.get(name, float('nan')):.4f}",
                                 f"{cv_mean:.4f}", f"{fast_auc(y, probs[name]):.4f}"])

    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "model", "metric", "value", "spread_or_ci", "detail"])
        w.writerows(rows_summary)
    with open(OUT_PAIRWISE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_A", "model_B", "only_A_correct", "only_B_correct",
                    "mcnemar_p", "mcnemar_significant", "auc_diff", "ci_low",
                    "ci_high", "auc_p", "auc_significant"])
        w.writerows(rows_pairwise)
    print(f"\nWrote {OUT_SUMMARY.relative_to(ROOT)} and {OUT_PAIRWISE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
