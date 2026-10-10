"""Analyse the out-of-fold explanation scores produced inside the CV loop.

Input is `results/cv_localization_v3.csv`, which the notebook writes one row per
(model, method, frame): IoU, Pointing Game, Coverage, the area the heat map
claimed, and the mask's area. Every frame was scored by the fold model that never
saw its surgery, so unlike the earlier 53-frame analysis this covers all 373
pathology frames under the same protocol as the classification results.

Everything is resampled at the surgery level, as elsewhere in the paper, and
*p*-values are Holm-corrected within each family of comparisons.

Usage (from the repo root):
    python scripts/analyze_cv_localization.py
Output:
    results/cv_localization_summary.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOC_CSV = RESULTS / "cv_localization_v3.csv"
OUT_CSV = RESULTS / "cv_localization_summary.csv"
MASK_DIRS = [ROOT / "data" / "masks" / s / "endometriosis"
             for s in ("train", "val", "test")]

N_BOOT = 10_000
SEED = 42
ALPHA = 0.05
CAM_THRESHOLD = 0.5


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    n, adjusted, running = len(pvalues), [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((n - rank) * pvalues[idx], 1.0))
        adjusted[idx] = running
    return adjusted


def centre_prior_region() -> np.ndarray:
    yy, xx = np.mgrid[0:224, 0:224]
    g = np.exp(-(((xx - 112) ** 2 + (yy - 112) ** 2) / (2 * (0.25 * 224) ** 2)))
    g = (g - g.min()) / (g.max() - g.min())
    return (g >= CAM_THRESHOLD).astype(np.uint8)


def mask_for(filename: str) -> np.ndarray | None:
    stem = Path(filename).stem + ".png"
    for d in MASK_DIRS:
        p = d / stem
        if p.exists():
            m = np.array(Image.open(p).convert("L").resize((224, 224)))
            return (m > 0).astype(np.uint8)
    return None


def expected_random_iou(claimed: np.ndarray, mask_area: np.ndarray) -> np.ndarray:
    """Two independently placed regions of area p and m overlap by p*m on average."""
    return (claimed * mask_area) / (claimed + mask_area - claimed * mask_area)


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def main() -> None:
    if not LOC_CSV.exists():
        raise SystemExit(f"{LOC_CSV.relative_to(ROOT)} not found - run the CV notebook first")
    rows = list(csv.DictReader(open(LOC_CSV, encoding="utf-8")))

    methods: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        label = r["model"] if r["method"] == "grad-cam" else f"{r['model']} (rollout)"
        methods[label][r["filename"]] = r
    names = list(methods)
    files = sorted(set.intersection(*(set(v) for v in methods.values())))
    surgeries = np.array([methods[names[0]][f]["surgery_id"] for f in files])
    print(f"{len(files)} pathology frames from {len(set(surgeries))} surgeries, "
          f"{len(names)} explanation methods")

    # centre-prior IoU has to be computed here: it depends on the mask, not the model
    centre = centre_prior_region()
    centre_iou, mask_area = [], []
    for f in files:
        m = mask_for(f)
        if m is None:
            raise SystemExit(f"no mask on disk for {f}")
        union = (centre | m).sum()
        centre_iou.append(float((centre & m).sum() / union) if union else 0.0)
        mask_area.append(float(m.mean()))
    centre_iou = np.asarray(centre_iou)
    mask_area = np.asarray(mask_area)
    print(f"lesions cover {mask_area.mean():.2%} of a frame on average "
          f"({mask_area.min():.2%}-{mask_area.max():.2%}); "
          f"centre prior scores IoU {centre_iou.mean():.3f}")

    by_surgery: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(surgeries):
        by_surgery[s].append(i)
    cases = list(by_surgery)
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([by_surgery[c] for c in rng.choice(cases, len(cases), replace=True)])
               for _ in range(N_BOOT)]

    values: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        values[name] = {
            k: np.array([float(methods[name][f][k]) for f in files])
            for k in ("IoU", "pointing", "coverage", "area")}

    out: list[list] = []
    print("\n=== means with 95% surgery-level intervals ===")
    for name in names:
        parts = []
        for metric in ("IoU", "pointing", "coverage"):
            v = values[name][metric]
            lo, hi = ci(np.array([v[i].mean() for i in samples]))
            parts.append(f"{metric} {v.mean():.3f} [{lo:.3f}-{hi:.3f}]")
            out.append(["mean_ci", name, metric, f"{v.mean():.4f}", f"{lo:.4f}", f"{hi:.4f}", ""])
        print(f"  {name:32s} " + " | ".join(parts))

    print("\n=== pairwise IoU differences (Holm-corrected) ===")
    raw, detail = [], []
    for a, b in combinations(names, 2):
        d = values[a]["IoU"] - values[b]["IoU"]
        boots = np.array([d[i].mean() for i in samples])
        lo, hi = ci(boots)
        p = 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
        raw.append(min(max(p, 1.0 / N_BOOT), 1.0))
        detail.append((a, b, float(d.mean()), lo, hi))
    for (a, b, diff, lo, hi), p in zip(detail, holm(raw)):
        verdict = "significant" if p < ALPHA else "not significant"
        print(f"  {a:28s} - {b:28s} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p={p:.4g} ({verdict})")
        out.append(["pairwise_IoU", a, b, f"{diff:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                    f"p_holm={p:.4g} ({verdict})"])

    print("\n=== against chance and the centre prior (Holm-corrected) ===")
    raw, detail = [], []
    for name in names:
        floor = expected_random_iou(values[name]["area"], mask_area)
        for label, reference in (("area-matched random", floor),
                                 ("centre prior", centre_iou)):
            d = values[name]["IoU"] - reference
            boots = np.array([d[i].mean() for i in samples])
            lo, hi = ci(boots)
            p = 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
            raw.append(min(max(p, 1.0 / N_BOOT), 1.0))
            detail.append((name, label, float(d.mean()), lo, hi, float(reference.mean())))
    for (name, label, diff, lo, hi, ref), p in zip(detail, holm(raw)):
        verdict = ("above" if diff > 0 and p < ALPHA else
                   "below" if diff < 0 and p < ALPHA else "indistinguishable")
        print(f"  {name:32s} vs {label:20s} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p={p:.4g} -> {verdict}")
        out.append(["vs_" + label.replace(" ", "_"), name, label, f"{diff:+.4f}",
                    f"{lo:+.4f}", f"{hi:+.4f}", f"p_holm={p:.4g} ({verdict}); "
                    f"reference mean {ref:.4f}"])

    # chance level for the two area-free metrics is the mean mask area
    out.append(["chance", "uninformative heat map", "pointing/coverage",
                f"{mask_area.mean():.4f}", "", "", "equals the mean lesion area"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "method_A", "method_B", "value", "ci_low", "ci_high", "note"])
        w.writerows(out)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
