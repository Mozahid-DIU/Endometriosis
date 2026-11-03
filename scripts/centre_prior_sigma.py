"""Does the centre-prior conclusion depend on the width we gave the Gaussian?

One of the paper's sharper claims is that ResNet50's Grad-CAM is statistically
indistinguishable from a fixed blob in the middle of the frame, while
EfficientNet-B0's maps beat it. That comparison rests on a number we chose
rather than derived: sigma = 0.25 x 224. A wider Gaussian claims more area and
scores a higher IoU, a narrower one less, so the baseline can be made easy or
hard by the choice - and a reviewer is entitled to ask what happens if it moves.

This sweeps sigma and recomputes, for each width:
  - the prior's own IoU, Pointing Game and Coverage over the 373 frames
  - each method's margin over it, with surgery-level intervals and a two-sided
    bootstrap p-value

The question the sweep answers is not "which sigma is right" but "over what
range of sigma does the paper's reading survive". If EfficientNet-B0 stays above
the prior and ResNet50 stays level with it across a wide band, the conclusion is
about the models. If the ordering flips inside a plausible band, the conclusion
is about the sigma, and the paper has to say so.

One caution about the p-values below. Holm is applied here within a family of
four - the four methods against the prior at one sigma. The manuscript corrects
within a family of eight, those four methods against *both* reference levels, so
its corrected p for EfficientNet-B0 at sigma 0.25 is 0.0048 where this prints
0.0032. The two are the same test under different corrections, not a discrepancy.
Read this script for whether a verdict changes across sigma, and the manuscript
for the corrected p-value to quote.

Usage (from the repo root):
    python scripts/centre_prior_sigma.py
Output:
    results/centre_prior_sigma.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_cv_localization import (CAM_THRESHOLD, LOC_CSV, MASK_DIRS, ci,
                                     holm, mask_for)
from cluster_bootstrap import N_BOOT, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "centre_prior_sigma.csv"

# 0.25 is what the manuscript used; the sweep brackets it widely on both sides.
SIGMAS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
METHODS = [("ResNet50", "grad-cam"), ("EfficientNet-B0", "grad-cam"),
           ("ViT-B/16", "grad-cam"), ("ViT-B/16", "rollout")]


def gaussian(sigma_fraction: float) -> np.ndarray:
    yy, xx = np.mgrid[0:224, 0:224]
    g = np.exp(-(((xx - 112) ** 2 + (yy - 112) ** 2) /
                 (2 * (sigma_fraction * 224) ** 2)))
    return (g - g.min()) / (g.max() - g.min())


def load_scores() -> tuple[list[str], dict, np.ndarray]:
    """Per-frame IoU for each method, plus the surgery each frame belongs to."""
    rows = list(csv.DictReader(open(LOC_CSV, encoding="utf-8")))
    per_method: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    surgery: dict[str, str] = {}
    for r in rows:
        key = (r["model"], r["method"])
        per_method[key][r["filename"]] = float(r["IoU"])
        surgery[r["filename"]] = r["surgery_id"]
    files = sorted(set.intersection(*(set(v) for v in per_method.values())))
    return files, per_method, np.array([surgery[f] for f in files])


def surgery_resamples(groups: np.ndarray, rng) -> list[np.ndarray]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    index = {g: np.asarray(v) for g, v in by_group.items()}
    return [np.concatenate([index[g] for g in rng.choice(keys, len(keys), replace=True)])
            for _ in range(N_BOOT)]


def main() -> None:
    files, per_method, groups = load_scores()
    masks = {}
    for f in files:
        m = mask_for(f)
        if m is not None:
            masks[f] = m
    files = [f for f in files if f in masks]
    print(f"{len(files)} frames, {len(set(groups))} surgeries, "
          f"mask area {np.mean([masks[f].mean() for f in files]):.4f}\n")

    rng = np.random.default_rng(SEED)
    samples = surgery_resamples(np.array([g for f, g in zip(files, groups)]), rng)

    out: list[list] = []
    for sigma in SIGMAS:
        prior = gaussian(sigma)
        region = (prior >= CAM_THRESHOLD).astype(np.uint8)
        peak = np.unravel_index(np.argmax(prior), prior.shape)

        prior_iou, hits, covs = [], [], []
        for f in files:
            m = masks[f]
            union = (region | m).sum()
            prior_iou.append(float((region & m).sum() / union) if union else 0.0)
            hits.append(int(m[peak] > 0))
            covs.append(float((prior * m).sum() / (prior.sum() + 1e-8)))
        prior_iou = np.asarray(prior_iou)

        mark = "  <- manuscript" if abs(sigma - 0.25) < 1e-9 else ""
        print(f"sigma {sigma:.2f} x 224  claims {region.mean():6.1%} of the frame   "
              f"IoU {prior_iou.mean():.4f}  Pointing {np.mean(hits):.4f}  "
              f"Coverage {np.mean(covs):.4f}{mark}")

        raw, detail = [], []
        for model, method in METHODS:
            scores = np.array([per_method[(model, method)][f] for f in files])
            delta = scores - prior_iou
            point = float(delta.mean())
            boot = np.array([float(delta[i].mean()) for i in samples])
            lo, hi = ci(boot)
            p = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
            raw.append(min(max(p, 1.0 / (N_BOOT + 1)), 1.0))
            detail.append((model, method, point, lo, hi))

        for (model, method, point, lo, hi), p_raw, p_holm in zip(
                detail, raw, holm(raw)):
            verdict = ("above" if p_holm < 0.05 and point > 0 else
                       "below" if p_holm < 0.05 else "level with")
            name = f"{model}, {method}"
            print(f"    {name:28s} {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"p_holm={p_holm:.4g}  ({verdict} the prior)")
            out.append([f"{sigma:.2f}", f"{region.mean():.4f}",
                        f"{prior_iou.mean():.4f}", f"{np.mean(hits):.4f}",
                        f"{np.mean(covs):.4f}", model, method, f"{point:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}", f"{p_raw:.4g}",
                        f"{p_holm:.4g}", verdict])
        print()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_fraction", "prior_area", "prior_iou", "prior_pointing",
                    "prior_coverage", "model", "method", "delta_iou", "ci_low",
                    "ci_high", "p_raw", "p_holm", "verdict"])
        w.writerows(out)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
