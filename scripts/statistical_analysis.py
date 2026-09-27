"""Statistical analysis of the per-image test predictions.

Reads results/test_predictions.csv (written by predict_test_set.py) and produces
the rigour numbers the methodology promised but the notebook never computed:

  1. Bootstrap 95% CI for AUC, and for accuracy / recall / F1
  2. McNemar's test for every model pair (are the error patterns different?)
  3. Confusion-matrix counts and an error breakdown per case/video group
  4. DeLong-free AUC difference test via paired bootstrap

Usage (from the repo root):
    python scripts/statistical_analysis.py
Outputs:
    results/stats_confidence_intervals.csv
    results/stats_mcnemar.csv
    results/stats_error_analysis.csv
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
PRED_CSV = ROOT / "results" / "test_predictions.csv"
OUT_CI = ROOT / "results" / "stats_confidence_intervals.csv"
OUT_MCNEMAR = ROOT / "results" / "stats_mcnemar.csv"
OUT_ERRORS = ROOT / "results" / "stats_error_analysis.csv"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05


def fast_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Same value as roc_auc_score but ~100x faster,
    which matters inside a 10k-iteration bootstrap."""
    n_pos = int(np.count_nonzero(y_true == 1))
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    # average ranks within ties so tied probabilities are handled like sklearn does
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_metrics(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    """(accuracy, precision, recall, f1) for binary labels, without sklearn overhead."""
    tp = float(np.count_nonzero((y_true == 1) & (pred == 1)))
    fp = float(np.count_nonzero((y_true == 0) & (pred == 1)))
    fn = float(np.count_nonzero((y_true == 1) & (pred == 0)))
    tn = float(np.count_nonzero((y_true == 0) & (pred == 0)))
    acc = (tp + tn) / y_true.size
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def load_predictions() -> tuple[list[dict], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    y_true = np.array([int(r["y_true"]) for r in rows])
    probs = {m: np.array([float(r[f"{m}_prob"]) for r in rows]) for m in MODELS}
    preds = {m: np.array([int(r[f"{m}_pred"]) for r in rows]) for m in MODELS}
    return rows, y_true, probs, preds


def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Stratified bootstrap over images; returns {metric: (point, lo, hi)}."""
    rng = np.random.default_rng(SEED)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    point = {
        "AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = np.concatenate([
            rng.choice(pos_idx, size=pos_idx.size, replace=True),
            rng.choice(neg_idx, size=neg_idx.size, replace=True),
        ])
        yt, pr, pd_ = y_true[idx], prob[idx], pred[idx]
        acc, prec, rec, f1 = fast_metrics(yt, pd_)
        samples["AUC"].append(fast_auc(yt, pr))
        samples["Accuracy"].append(acc)
        samples["Precision"].append(prec)
        samples["Recall"].append(rec)
        samples["F1"].append(f1)

    out = {}
    for k, v in point.items():
        lo, hi = np.percentile(samples[k], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out[k] = (v, float(lo), float(hi))
    return out


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int | str]:
    """Exact (binomial) McNemar on discordant pairs, plus the chi-square version."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))   # only A right
    c = int(np.sum(~correct_a & correct_b))   # only B right
    n = b + c
    p_exact = binomtest(b, n, 0.5).pvalue if n else 1.0
    if n:
        stat = (abs(b - c) - 1) ** 2 / n      # continuity-corrected
        p_chi2 = float(chi2.sf(stat, df=1))
    else:
        stat, p_chi2 = 0.0, 1.0
    return {
        "only_A_correct": b,
        "only_B_correct": c,
        "discordant": n,
        "chi2_cc": round(stat, 4),
        "p_chi2": p_chi2,
        "p_exact": float(p_exact),
        "significant_at_0.05": "yes" if p_exact < ALPHA else "no",
    }


def paired_auc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict[str, float | str]:
    """Paired bootstrap on the AUC difference (same resampled images for both models)."""
    rng = np.random.default_rng(SEED)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    observed = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([
            rng.choice(pos_idx, size=pos_idx.size, replace=True),
            rng.choice(neg_idx, size=neg_idx.size, replace=True),
        ])
        yt = y_true[idx]
        diffs.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    # two-sided bootstrap p-value: how often the difference flips sign
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {
        "auc_diff": round(float(observed), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p_bootstrap": round(min(p, 1.0), 4),
        "significant_at_0.05": "yes" if min(p, 1.0) < ALPHA else "no",
    }


def write_confidence_intervals(y_true, probs, preds) -> None:
    with open(OUT_CI, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "value", "ci_low", "ci_high"])
        print(f"\n=== 95% CI (stratified bootstrap, n={N_BOOT}) ===")
        for m in MODELS:
            ci = bootstrap_ci(y_true, probs[m], preds[m])
            for metric, (v, lo, hi) in ci.items():
                w.writerow([m, metric, f"{v:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            print(f"{m:16s} AUC {ci['AUC'][0]:.3f} [{ci['AUC'][1]:.3f}-{ci['AUC'][2]:.3f}]  "
                  f"Acc {ci['Accuracy'][0]:.3f} [{ci['Accuracy'][1]:.3f}-{ci['Accuracy'][2]:.3f}]  "
                  f"Recall {ci['Recall'][0]:.3f} [{ci['Recall'][1]:.3f}-{ci['Recall'][2]:.3f}]")


def write_pairwise_tests(y_true, probs, preds) -> None:
    with open(OUT_MCNEMAR, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_A", "model_B", "only_A_correct", "only_B_correct", "discordant",
                    "chi2_cc", "p_chi2", "p_exact", "mcnemar_significant",
                    "auc_diff", "auc_ci_low", "auc_ci_high", "p_auc_bootstrap", "auc_significant"])
        print("\n=== McNemar + paired AUC test ===")
        for a, b in combinations(MODELS, 2):
            mc = mcnemar(y_true, preds[a], preds[b])
            au = paired_auc_test(y_true, probs[a], probs[b])
            w.writerow([a, b, mc["only_A_correct"], mc["only_B_correct"], mc["discordant"],
                        mc["chi2_cc"], f"{mc['p_chi2']:.4f}", f"{mc['p_exact']:.4f}",
                        mc["significant_at_0.05"], au["auc_diff"], au["ci_low"], au["ci_high"],
                        au["p_bootstrap"], au["significant_at_0.05"]])
            print(f"{a} vs {b}: only-{a}-right={mc['only_A_correct']} only-{b}-right={mc['only_B_correct']} "
                  f"p_exact={mc['p_exact']:.4f} ({mc['significant_at_0.05']}) | "
                  f"dAUC={au['auc_diff']:+.3f} [{au['ci_low']:+.3f},{au['ci_high']:+.3f}] "
                  f"p={au['p_bootstrap']:.4f} ({au['significant_at_0.05']})")


def write_error_analysis(rows, y_true, preds) -> None:
    print("\n=== Confusion counts ===")
    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value"])
        for m in MODELS:
            p = preds[m]
            tp = int(np.sum((y_true == 1) & (p == 1)))
            fn = int(np.sum((y_true == 1) & (p == 0)))
            fp = int(np.sum((y_true == 0) & (p == 1)))
            tn = int(np.sum((y_true == 0) & (p == 0)))
            for k, v in (("TP", tp), ("FN", fn), ("FP", fp), ("TN", tn)):
                w.writerow([f"confusion:{m}", k, v])
            print(f"{m:16s} TP={tp} FN={fn} FP={fp} TN={tn}")

        # images every model got wrong, and images only one model got wrong
        wrong_sets = {m: {rows[i]["filename"] for i in range(len(rows)) if preds[m][i] != y_true[i]}
                      for m in MODELS}
        all_wrong = set.intersection(*wrong_sets.values())
        any_wrong = set.union(*wrong_sets.values())
        print(f"\nWrong in ALL three models: {len(all_wrong)}   wrong in at least one: {len(any_wrong)}")
        for f in sorted(all_wrong):
            row = next(r for r in rows if r["filename"] == f)
            w.writerow(["hard_case:all_models_wrong", f, row["class"]])

        # per-group error rate (case / video) — shows whether errors cluster in one surgery
        print("\n=== Groups with the most errors (any model) ===")
        group_err: dict[str, Counter] = defaultdict(Counter)
        for i, row in enumerate(rows):
            g = row["group_id"]
            group_err[g]["n"] += 1
            for m in MODELS:
                if preds[m][i] != y_true[i]:
                    group_err[g][m] += 1
        ranked = sorted(group_err.items(),
                        key=lambda kv: sum(kv[1][m] for m in MODELS), reverse=True)
        for g, c in ranked:
            total_err = sum(c[m] for m in MODELS)
            w.writerow(["group_errors", g, f"n={c['n']};" + ";".join(f"{m}={c[m]}" for m in MODELS)])
            if total_err:
                print(f"  {g:28s} n={c['n']:3d}  " + "  ".join(f"{m}={c[m]}" for m in MODELS))


def main() -> None:
    if not PRED_CSV.exists():
        raise SystemExit(f"Run scripts/predict_test_set.py first ({PRED_CSV} missing)")
    rows, y_true, probs, preds = load_predictions()
    print(f"Loaded {len(rows)} test predictions "
          f"({int(np.sum(y_true == 1))} endometriosis / {int(np.sum(y_true == 0))} normal)")
    write_confidence_intervals(y_true, probs, preds)
    write_pairwise_tests(y_true, probs, preds)
    write_error_analysis(rows, y_true, preds)
    print("\nWrote:", ", ".join(str(p.relative_to(ROOT)) for p in (OUT_CI, OUT_MCNEMAR, OUT_ERRORS)))


if __name__ == "__main__":
    main()
