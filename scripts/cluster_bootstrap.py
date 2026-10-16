"""Re-estimate every confidence interval by resampling surgeries, not frames.

Frames from one surgery are not independent: they share a patient, a camera, an
illumination setting and often a lesion. Resampling individual frames therefore
understates the uncertainty, because a resample can contain the same surgery's
frames many times over while pretending they are new evidence. The inference we
actually want is to *new surgeries*, so the resampling unit must be the surgery.

This script redoes, with a cluster (group-level) bootstrap:
  - pooled out-of-fold AUC and its CI, per model
  - paired AUC and accuracy differences between models, Holm-corrected
  - McNemar's test, reported frame-level and used for no claim

Localization is handled by analyze_cv_localization.py, which works on the current
373-frame out-of-fold scores; this script no longer touches it.

Frame-level intervals are printed next to the cluster-level ones so the cost of
the correction is visible.

Usage (from the repo root):
    python scripts/cluster_bootstrap.py
Output:
    results/cluster_bootstrap_classification.csv
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from statistical_analysis import fast_auc

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PRED = RESULTS / "cv_predictions_v3.csv"
OUT_CLS = RESULTS / "cluster_bootstrap_classification.csv"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05



def group_key(row: dict) -> str:
    """v2 runs record surgery_id; the first run recorded group_id (per segment)."""
    return row.get("surgery_id") or row["group_id"]

def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    n, adjusted, running = len(pvalues), [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((n - rank) * pvalues[idx], 1.0))
        adjusted[idx] = running
    return adjusted


def case_of(filename: str) -> str:
    """Pathology frames are named c_<case>_v_(video_...)_f_<n>.jpg"""
    m = re.match(r"(c_\d+)", filename)
    return m.group(1) if m else filename


def group_resamples(groups: np.ndarray, labels: np.ndarray, rng) -> list[np.ndarray]:
    """Index sets drawn by resampling whole groups, keeping both classes present.

    Groups are class-pure here (a surgery is either a pathology case or a
    no-pathology recording), so positive and negative groups are drawn separately
    to keep every resample evaluable.
    """
    by_group: dict[str, np.ndarray] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    by_group = {g: np.asarray(v) for g, v in by_group.items()}
    pos = [g for g, idx in by_group.items() if labels[idx].max() == 1]
    neg = [g for g, idx in by_group.items() if labels[idx].max() == 0]

    out = []
    for _ in range(N_BOOT):
        picked = list(rng.choice(pos, len(pos), replace=True)) + \
                 list(rng.choice(neg, len(neg), replace=True))
        out.append(np.concatenate([by_group[g] for g in picked]))
    return out


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def frame_resamples(labels: np.ndarray, rng) -> list[np.ndarray]:
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    return [np.concatenate([rng.choice(pos, pos.size, replace=True),
                            rng.choice(neg, neg.size, replace=True)])
            for _ in range(N_BOOT)]


def classification() -> list[list]:
    rows = list(csv.DictReader(open(PRED, encoding="utf-8")))
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["filename"]] = r
    files = sorted(set.intersection(*(set(v) for v in by_model.values())))

    y = np.array([int(by_model[MODELS[0]][f]["y_true"]) for f in files])
    groups = np.array([group_key(by_model[MODELS[0]][f]) for f in files])
    probs = {m: np.array([float(by_model[m][f]["prob"]) for f in files]) for m in MODELS}
    preds = {m: np.array([int(by_model[m][f]["pred"]) for f in files]) for m in MODELS}

    n_groups = len(set(groups))
    print(f"{len(files)} frames from {n_groups} surgeries "
          f"({int(y.sum())} pathology frames)")

    rng = np.random.default_rng(SEED)
    g_samples = group_resamples(groups, y, rng)
    rng_f = np.random.default_rng(SEED)
    f_samples = frame_resamples(y, rng_f)

    out: list[list] = []
    print("\n=== pooled AUC: frame bootstrap vs surgery (cluster) bootstrap ===")
    for m in MODELS:
        point = fast_auc(y, probs[m])
        f_lo, f_hi = ci(np.array([fast_auc(y[i], probs[m][i]) for i in f_samples]))
        g_lo, g_hi = ci(np.array([fast_auc(y[i], probs[m][i]) for i in g_samples]))
        widen = (g_hi - g_lo) / (f_hi - f_lo)
        print(f"{m:16s} AUC {point:.4f}  frame [{f_lo:.3f}-{f_hi:.3f}]  "
              f"surgery [{g_lo:.3f}-{g_hi:.3f}]  ({widen:.1f}x wider)")
        out.append(["auc_ci", m, "", f"{point:.4f}", f"{f_lo:.4f}", f"{f_hi:.4f}",
                    f"{g_lo:.4f}", f"{g_hi:.4f}", "", ""])

    print("\n=== paired differences, surgery-level resampling, Holm-corrected ===")
    pairs = list(combinations(MODELS, 2))
    raw_auc, raw_mc, detail = [], [], []
    for a, b in pairs:
        observed = fast_auc(y, probs[a]) - fast_auc(y, probs[b])
        diffs = np.array([fast_auc(y[i], probs[a][i]) - fast_auc(y[i], probs[b][i])
                          for i in g_samples])
        lo, hi = ci(diffs)
        p_auc = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
        p_auc = min(max(p_auc, 1.0 / (N_BOOT + 1)), 1.0)

        ca, cb = preds[a] == y, preds[b] == y
        only_a = int(np.sum(ca & ~cb))
        only_b = int(np.sum(~ca & cb))
        n = only_a + only_b
        # frame-level: McNemar treats the discordant frames as independent pairs
        p_mc = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0

        # surgery-level equivalent: resample surgeries and compare accuracy
        acc_diff = float(ca.mean() - cb.mean())
        acc_boots = np.array([ca[i].mean() - cb[i].mean() for i in g_samples])
        acc_lo, acc_hi = ci(acc_boots)
        p_acc = 2 * min(float(np.mean(acc_boots <= 0)), float(np.mean(acc_boots >= 0)))
        p_acc = min(max(p_acc, 1.0 / (N_BOOT + 1)), 1.0)

        raw_auc.append(p_auc); raw_mc.append(p_acc)
        detail.append((a, b, observed, lo, hi, only_a, only_b,
                       acc_diff, acc_lo, acc_hi, p_mc, p_auc, p_acc))

    for (a, b, observed, lo, hi, only_a, only_b, acc_diff, acc_lo, acc_hi,
         p_mc_raw, p_auc_raw, p_acc_raw), p_auc, p_acc in zip(
            detail, holm(raw_auc), holm(raw_mc)):
        verdict = "significant" if p_auc < ALPHA else "not significant"
        print(f"{a:16s} vs {b:16s}")
        print(f"    dAUC      {observed:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"p_raw={p_auc_raw:.4g}  p_holm={p_auc:.4g}  ({verdict})")
        print(f"    dAccuracy {acc_diff:+.4f} [{acc_lo:+.4f},{acc_hi:+.4f}]  "
              f"p_raw={p_acc_raw:.4g}  p_holm={p_acc:.4g}   (surgery-level)")
        print(f"    discordant frames {only_a}/{only_b}, McNemar p={p_mc_raw:.4g} "
              f"(frame-level, ignores clustering)")
        out.append(["pairwise", a, b, f"{observed:+.4f}", "", "", f"{lo:+.4f}", f"{hi:+.4f}",
                    f"{only_a}/{only_b}; McNemar frame-level p={p_mc_raw:.4g}",
                    f"dAcc {acc_diff:+.4f} [{acc_lo:+.4f},{acc_hi:+.4f}] "
                    f"raw={p_acc_raw:.4g} holm={p_acc:.4g}; "
                    f"dAUC raw={p_auc_raw:.4g} holm={p_auc:.4g} ({verdict})"])
    return out


def main() -> None:
    cls_rows = classification()
    with open(OUT_CLS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # "auc_ci" rows carry both interval widths; "pairwise" rows carry the
        # surgery-level interval in ci_low/ci_high and leave the frame columns empty
        w.writerow(["kind", "model_A", "model_B", "value", "frame_ci_low",
                    "frame_ci_high", "ci_low", "ci_high",
                    "discordant_and_mcnemar", "accuracy_difference"])
        w.writerows(cls_rows)
    print(f"\nWrote {OUT_CLS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
