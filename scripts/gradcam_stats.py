"""Significance testing for the Grad-CAM localization results.

The headline claim of the paper is that EfficientNet-B0 localizes lesions better
than ResNet50 and ViT-B/16 even though it classifies worse. A difference in means
is not enough to claim that, so this script tests it on the per-image values:

  - Wilcoxon signed-rank test on paired IoU and Coverage (same images, no
    normality assumption, which matters because IoU is bounded and skewed)
  - McNemar's exact test on the Pointing Game (paired binary hit/miss)
  - Bootstrap 95% CI for each mean
  - Holm correction across the three model pairs

Usage (from the repo root):
    python scripts/gradcam_stats.py
Output:
    results/stats_gradcam.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
PER_IMAGE_CSV = ROOT / "results" / "gradcam_per_image.csv"
OUT_CSV = ROOT / "results" / "stats_gradcam.csv"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05


def load() -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(open(PER_IMAGE_CSV, encoding="utf-8")))
    data = {}
    for m in MODELS:
        data[m] = {
            "IoU": np.array([float(r[f"{m}_IoU"]) for r in rows]),
            "pointing": np.array([int(r[f"{m}_pointing"]) for r in rows]),
            "coverage": np.array([float(r[f"{m}_coverage"]) for r in rows]),
        }
    print(f"Loaded per-image Grad-CAM values for {len(rows)} images")
    return data


def boot_ci(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True)))
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(np.mean(values)), float(lo), float(hi)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = np.argsort(pvalues)
    n = len(pvalues)
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvalues[idx]
        running = max(running, min(val, 1.0))
        adjusted[idx] = running
    return adjusted


def main() -> None:
    if not PER_IMAGE_CSV.exists():
        raise SystemExit(f"Run scripts/gradcam_per_image.py first ({PER_IMAGE_CSV} missing)")
    data = load()

    rows_out: list[list] = []

    print("\n=== Per-model means with 95% CI (bootstrap) ===")
    for m in MODELS:
        for metric in ("IoU", "pointing", "coverage"):
            mean, lo, hi = boot_ci(data[m][metric])
            rows_out.append(["mean_ci", m, "", metric, f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}", "", ""])
        i_m, i_lo, i_hi = boot_ci(data[m]["IoU"])
        p_m, p_lo, p_hi = boot_ci(data[m]["pointing"])
        c_m, c_lo, c_hi = boot_ci(data[m]["coverage"])
        print(f"{m:16s} IoU {i_m:.3f} [{i_lo:.3f}-{i_hi:.3f}]  "
              f"Pointing {p_m:.3f} [{p_lo:.3f}-{p_hi:.3f}]  "
              f"Coverage {c_m:.3f} [{c_lo:.3f}-{c_hi:.3f}]")

    print("\n=== Paired tests between models ===")
    pairs = list(combinations(MODELS, 2))
    raw: dict[str, list[float]] = {"IoU": [], "coverage": [], "pointing": []}
    details: dict[str, list[tuple]] = {"IoU": [], "coverage": [], "pointing": []}

    for a, b in pairs:
        for metric in ("IoU", "coverage"):
            x, y = data[a][metric], data[b][metric]
            diff = x - y
            if np.any(diff != 0):
                stat, p = wilcoxon(x, y)
            else:
                stat, p = 0.0, 1.0
            raw[metric].append(float(p))
            details[metric].append((a, b, float(np.mean(diff)), float(stat), float(p)))

        # Pointing Game: paired binary -> exact McNemar
        ha, hb = data[a]["pointing"], data[b]["pointing"]
        only_a = int(np.sum((ha == 1) & (hb == 0)))
        only_b = int(np.sum((ha == 0) & (hb == 1)))
        n = only_a + only_b
        p = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0
        raw["pointing"].append(p)
        details["pointing"].append((a, b, only_a, only_b, p))

    for metric in ("IoU", "coverage"):
        adj = holm(raw[metric])
        for (a, b, mean_diff, stat, p), p_adj in zip(details[metric], adj):
            sig = "yes" if p_adj < ALPHA else "no"
            rows_out.append(["wilcoxon", a, b, metric, f"{mean_diff:+.4f}",
                             f"{stat:.1f}", f"{p:.5f}", f"{p_adj:.5f}", sig])
            print(f"{metric:9s} {a:16s} vs {b:16s} mean diff {mean_diff:+.4f}  "
                  f"W={stat:.0f}  p={p:.5f}  p_holm={p_adj:.5f}  ({sig})")

    adj = holm(raw["pointing"])
    for (a, b, only_a, only_b, p), p_adj in zip(details["pointing"], adj):
        sig = "yes" if p_adj < ALPHA else "no"
        rows_out.append(["mcnemar_pointing", a, b, "pointing",
                         f"only_{a}={only_a};only_{b}={only_b}", "", f"{p:.5f}",
                         f"{p_adj:.5f}", sig])
        print(f"pointing  {a:16s} vs {b:16s} hits only-A={only_a} only-B={only_b}  "
              f"p={p:.5f}  p_holm={p_adj:.5f}  ({sig})")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "model_A", "model_B", "metric", "value_or_diff",
                    "statistic", "p_raw", "p_holm", f"significant_at_{ALPHA}"])
        w.writerows(rows_out)
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
