"""Re-estimate every confidence interval by resampling surgeries, not frames.

Frames from one surgery are not independent: they share a patient, a camera, an
illumination setting and often a lesion. Resampling individual frames therefore
understates the uncertainty, because a resample can contain the same surgery's
frames many times over while pretending they are new evidence. The inference we
actually want is to *new surgeries*, so the resampling unit must be the surgery.

This script redoes, with a cluster (group-level) bootstrap:
  - pooled out-of-fold AUC and its CI, per model
  - paired AUC differences between models, with Holm-corrected p-values
  - McNemar's test, Holm-corrected across the three pairs
  - the localization metrics and their CIs

Frame-level intervals are printed next to the cluster-level ones so the cost of
the correction is visible.

Usage (from the repo root):
    python scripts/cluster_bootstrap.py
Outputs:
    results/cluster_bootstrap_classification.csv
    results/cluster_bootstrap_localization.csv
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
PRED = RESULTS / "cv_predictions.csv"
GRADCAM = RESULTS / "gradcam_per_image.csv"
ROLLOUT = RESULTS / "vit_rollout_per_image.csv"
OUT_CLS = RESULTS / "cluster_bootstrap_classification.csv"
OUT_LOC = RESULTS / "cluster_bootstrap_localization.csv"

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
        p_auc = min(max(p_auc, 1.0 / N_BOOT), 1.0)

        ca, cb = preds[a] == y, preds[b] == y
        only_a = int(np.sum(ca & ~cb))
        only_b = int(np.sum(~ca & cb))
        n = only_a + only_b
        p_mc = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0

        raw_auc.append(p_auc); raw_mc.append(p_mc)
        detail.append((a, b, observed, lo, hi, only_a, only_b))

    for (a, b, observed, lo, hi, only_a, only_b), p_auc, p_mc in zip(
            detail, holm(raw_auc), holm(raw_mc)):
        verdict = "significant" if p_auc < ALPHA else "not significant"
        print(f"{a:16s} vs {b:16s} dAUC {observed:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p_holm={p_auc:.4g} ({verdict}) | McNemar {only_a}/{only_b} p_holm={p_mc:.4g}")
        out.append(["pairwise", a, b, f"{observed:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                    f"{p_auc:.4g}", verdict, f"{only_a}/{only_b}", f"{p_mc:.4g}"])
    return out


def localization() -> list[list]:
    gc = list(csv.DictReader(open(GRADCAM, encoding="utf-8")))
    roll = {r["filename"]: r for r in csv.DictReader(open(ROLLOUT, encoding="utf-8"))}
    files = [r["filename"] for r in gc]
    groups = np.array([case_of(f) for f in files])
    print(f"\n=== localization: {len(files)} frames from {len(set(groups))} surgeries ===")

    methods: dict[str, dict[str, np.ndarray]] = {}
    for m in MODELS:
        methods[m] = {k: np.array([float(r[f"{m}_{k}"]) if k != "pointing"
                                   else float(int(r[f"{m}_pointing"])) for r in gc])
                      for k in ("IoU", "pointing", "coverage")}
    methods["ViT-B/16 (attention rollout)"] = {
        "IoU": np.array([float(roll[f]["rollout_IoU"]) for f in files]),
        "pointing": np.array([float(roll[f]["rollout_pointing"]) for f in files]),
        "coverage": np.array([float(roll[f]["rollout_coverage"]) for f in files])}

    # every frame here is pathology, so draw from the case list directly
    by_case: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_case[g].append(i)
    cases = list(by_case)
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([by_case[c] for c in rng.choice(cases, len(cases), replace=True)])
               for _ in range(N_BOOT)]

    out: list[list] = []
    for name, vals in methods.items():
        line = [name]
        for metric in ("IoU", "pointing", "coverage"):
            point = float(vals[metric].mean())
            lo, hi = ci(np.array([vals[metric][i].mean() for i in samples]))
            line.append(f"{metric} {point:.3f} [{lo:.3f}-{hi:.3f}]")
            out.append(["mean_ci", name, metric, f"{point:.4f}", f"{lo:.4f}", f"{hi:.4f}", ""])
        print("  " + " | ".join(line))

    print("\n  paired differences in IoU (surgery-level resampling):")
    names = list(methods)
    raw, detail = [], []
    for a, b in combinations(names, 2):
        d = methods[a]["IoU"] - methods[b]["IoU"]
        observed = float(d.mean())
        boots = np.array([d[i].mean() for i in samples])
        lo, hi = ci(boots)
        p = 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
        raw.append(min(max(p, 1.0 / N_BOOT), 1.0))
        detail.append((a, b, observed, lo, hi))
    for (a, b, observed, lo, hi), p in zip(detail, holm(raw)):
        verdict = "significant" if p < ALPHA else "not significant"
        print(f"    {a:30s} - {b:30s} {observed:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p_holm={p:.4g} ({verdict})")
        out.append(["pairwise_IoU", a, b, f"{observed:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                    f"{p:.4g} ({verdict})"])
    return out


def baselines(out: list[list]) -> None:
    """The same surgery-level resampling for the two reference levels, so the
    chance comparisons are not quoted at a different standard from the rest."""
    import numpy as np
    from PIL import Image

    gc = list(csv.DictReader(open(GRADCAM, encoding="utf-8")))
    roll = {r["filename"]: r for r in csv.DictReader(open(ROLLOUT, encoding="utf-8"))}
    mask_dir = ROOT / "data" / "masks" / "test" / "endometriosis"

    files = [r["filename"] for r in gc]
    groups = np.array([case_of(f) for f in files])
    mask_area, centre_iou = [], []
    yy, xx = np.mgrid[0:224, 0:224]
    centre = np.exp(-(((xx - 112) ** 2 + (yy - 112) ** 2) / (2 * (0.25 * 224) ** 2)))
    centre = (centre - centre.min()) / (centre.max() - centre.min())
    centre_region = (centre >= 0.5).astype(np.uint8)
    for f in files:
        m = np.array(Image.open(mask_dir / f"{Path(f).stem}.png").convert("L")
                     .resize((224, 224)))
        m = (m > 0).astype(np.uint8)
        mask_area.append(float(m.mean()))
        union = (centre_region | m).sum()
        centre_iou.append(float((centre_region & m).sum() / union) if union else 0.0)
    mask_area = np.asarray(mask_area)
    centre_iou = np.asarray(centre_iou)

    methods = {m: (np.array([float(r[f"{m}_IoU"]) for r in gc]),
                   np.array([float(r[f"{m}_area"]) for r in gc])) for m in MODELS}
    methods["ViT-B/16 (attention rollout)"] = (
        np.array([float(roll[f]["rollout_IoU"]) for f in files]),
        np.array([float(roll[f]["rollout_area"]) for f in files]))

    by_case: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_case[g].append(i)
    cases = list(by_case)
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([by_case[c] for c in rng.choice(cases, len(cases), replace=True)])
               for _ in range(N_BOOT)]

    print("\n  against chance and the centre prior (surgery-level resampling):")
    raw, detail = [], []
    for name, (iou_vals, areas) in methods.items():
        floor = (areas * mask_area) / (areas + mask_area - areas * mask_area)
        for label, reference in (("area-matched random", floor), ("centre prior", centre_iou)):
            d = iou_vals - reference
            boots = np.array([d[i].mean() for i in samples])
            lo, hi = ci(boots)
            p = 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
            raw.append(min(max(p, 1.0 / N_BOOT), 1.0))
            detail.append((name, label, float(d.mean()), lo, hi))
    for (name, label, diff, lo, hi), p in zip(detail, holm(raw)):
        verdict = ("above" if diff > 0 and p < ALPHA else
                   "below" if diff < 0 and p < ALPHA else "indistinguishable")
        print(f"    {name:30s} vs {label:20s} {diff:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"p_holm={p:.4g} -> {verdict}")
        out.append(["vs_" + label.replace(" ", "_"), name, label, f"{diff:+.4f}",
                    f"{lo:+.4f}", f"{hi:+.4f}", f"p_holm={p:.4g} ({verdict})"])


def main() -> None:
    cls_rows = classification()
    loc_rows = localization()
    baselines(loc_rows)
    with open(OUT_CLS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "model_A", "model_B", "value", "frame_ci_low",
                    "frame_ci_high", "cluster_ci_low", "cluster_ci_high",
                    "mcnemar_counts", "mcnemar_p_holm"])
        w.writerows(cls_rows)
    with open(OUT_LOC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "method_A", "method_B", "value", "ci_low", "ci_high", "note"])
        w.writerows(loc_rows)
    print(f"\nWrote {OUT_CLS.relative_to(ROOT)} and {OUT_LOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
