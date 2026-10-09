"""Chance-level baselines for the Grad-CAM localization metrics.

An IoU of 0.148 means nothing on its own: a reader cannot tell it from chance
without knowing how large the lesions are. Lesions here cover only ~7% of the
frame, so this script establishes what each metric scores when a heatmap carries
no information, and tests every method against that floor.

Chance levels, analytic
  Pointing Game  an uninformative heatmap puts its peak inside the mask with
                 probability equal to the mask's area fraction
  Coverage       and spreads that same fraction of its energy inside the mask

Chance level for IoU, which rises automatically with the predicted area, so it
must be compared area for area
  area-matched   for each image and method, a random region of exactly the area
                 that method's thresholded heatmap claimed. Expected overlap of
                 two independently placed regions of area p and m is p*m, hence
                 IoU = p*m / (p + m - p*m). Confirmed here by simulation.

Two further reference points, both thresholded like a real heatmap:
  random noise   uniform noise
  centre prior   a centred Gaussian, the "lesions are usually mid-frame" prior
                 that any saliency claim has to beat

Usage (from the repo root):
    python scripts/localization_baselines.py
Output:
    results/localization_baselines.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from gradcam_per_image import IMG_DIR, MASK_DIR, iou, load_mask, pointing_and_coverage

ROOT = Path(__file__).resolve().parent.parent
PER_IMAGE = ROOT / "results" / "gradcam_per_image.csv"
ROLLOUT = ROOT / "results" / "vit_rollout_per_image.csv"
OUT_CSV = ROOT / "results" / "localization_baselines.csv"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
SEED = 42
THRESH = 0.5
N_SIM = 200          # random placements per image, to confirm the analytic floor


def expected_random_iou(pred_area: float, mask_area: float) -> float:
    denom = pred_area + mask_area - pred_area * mask_area
    return (pred_area * mask_area) / denom if denom > 0 else 0.0


def simulate_random_iou(pred_area: float, mask: np.ndarray, rng) -> float:
    """Empirical check of the formula: threshold noise so it claims exactly
    `pred_area` of the frame, then measure IoU against the real mask."""
    out = []
    for _ in range(N_SIM):
        noise = rng.random(mask.shape)
        cut = np.quantile(noise, 1 - pred_area)
        region = (noise >= cut).astype(np.uint8)
        union = (region | mask).sum()
        out.append((region & mask).sum() / union if union else 0.0)
    return float(np.mean(out))


def load_methods(files: list[Path]) -> dict[str, dict[str, tuple[float, float]]]:
    """name -> {filename: (IoU, predicted area)}"""
    rows = {r["filename"]: r for r in csv.DictReader(open(PER_IMAGE, encoding="utf-8"))}
    if f"{MODELS[0]}_area" not in next(iter(rows.values())):
        raise SystemExit("results/gradcam_per_image.csv has no *_area column - "
                         "re-run scripts/gradcam_per_image.py first")
    methods = {name: {f.name: (float(rows[f.name][f"{name}_IoU"]),
                               float(rows[f.name][f"{name}_area"])) for f in files}
               for name in MODELS}
    if ROLLOUT.exists():
        rr = {r["filename"]: r for r in csv.DictReader(open(ROLLOUT, encoding="utf-8"))}
        if "rollout_area" in next(iter(rr.values())):
            methods["ViT-B/16 (attention rollout)"] = {
                f.name: (float(rr[f.name]["rollout_IoU"]), float(rr[f.name]["rollout_area"]))
                for f in files}
    return methods


def main() -> None:
    files = sorted(f for f in IMG_DIR.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and (MASK_DIR / f"{f.stem}.png").exists())
    masks = {f.name: load_mask(MASK_DIR / f"{f.stem}.png") for f in files}
    fracs = {k: float(m.mean()) for k, m in masks.items()}
    chance = float(np.mean(list(fracs.values())))
    rng = np.random.default_rng(SEED)
    rows = []

    print(f"{len(files)} images | lesions cover {chance:.2%} of the frame on average "
          f"(min {min(fracs.values()):.2%}, max {max(fracs.values()):.2%})")
    print(f"Chance level for an uninformative heatmap: "
          f"PointingGame = Coverage = {chance:.4f}")
    rows.append(["chance", "uninformative heatmap", "", f"{chance:.4f}", f"{chance:.4f}", "", ""])

    # --- reference heatmaps, thresholded like a real one
    yy, xx = np.mgrid[0:224, 0:224]
    centre = np.exp(-(((xx - 112) ** 2 + (yy - 112) ** 2) / (2 * (0.25 * 224) ** 2)))
    centre = (centre - centre.min()) / (centre.max() - centre.min())

    for label, make in (("random noise", lambda: rng.random((224, 224))),
                        ("centre prior", lambda: centre)):
        ious, hits, covs = [], [], []
        for f in files:
            cam, m = make(), masks[f.name]
            ious.append(iou(cam, m, THRESH))
            hit, cov = pointing_and_coverage(cam, m)
            hits.append(hit); covs.append(cov)
        print(f"{label:14s} IoU {np.mean(ious):.4f}  Pointing {np.mean(hits):.4f}  "
              f"Coverage {np.mean(covs):.4f}")
        rows.append(["baseline", label, f"{np.mean(ious):.4f}", f"{np.mean(hits):.4f}",
                     f"{np.mean(covs):.4f}", "", ""])

    # --- every method against its own area-matched floor
    methods = load_methods(files)
    print("\n--- IoU against an area-matched random region ---")
    print(f"{'method':32s} {'area':>6s} {'IoU':>7s} {'floor':>7s} {'ratio':>6s}  verdict")
    for name, values in methods.items():
        observed = np.array([values[f.name][0] for f in files])
        areas = np.array([values[f.name][1] for f in files])
        floor = np.array([expected_random_iou(a, fracs[f.name])
                          for a, f in zip(areas, files)])
        stat, p = wilcoxon(observed, floor)
        ratio = observed.mean() / floor.mean() if floor.mean() else float("inf")
        better = observed.mean() > floor.mean()
        verdict = ("above chance" if better and p < 0.05 else
                   "BELOW chance" if not better and p < 0.05 else
                   "indistinguishable from chance")
        print(f"{name:32s} {areas.mean():6.3f} {observed.mean():7.4f} {floor.mean():7.4f} "
              f"{ratio:6.2f}  p={p:.3g}  {verdict}")
        rows.append(["method", name, f"{observed.mean():.4f}", "", "",
                     f"{floor.mean():.4f}", f"ratio {ratio:.2f}, p={p:.3g}, {verdict}"])

    # --- the harder test: does a method beat the centre prior, image by image?
    # Beating random is weak. A heatmap that only says "look in the middle" would
    # also beat random, so each method is tested against that prior as well.
    centre_iou = np.array([iou(centre, masks[f.name], THRESH) for f in files])
    centre_hit = np.array([pointing_and_coverage(centre, masks[f.name])[0] for f in files])
    print("\n--- against the centre prior (per image) ---")
    per_image_rows = {r["filename"]: r for r in csv.DictReader(open(PER_IMAGE, encoding="utf-8"))}
    for name, values in methods.items():
        observed = np.array([values[f.name][0] for f in files])
        stat, p = wilcoxon(observed, centre_iou)
        better = observed.mean() > centre_iou.mean()
        verdict = ("beats the centre prior" if better and p < 0.05 else
                   "WORSE than the centre prior" if not better and p < 0.05 else
                   "no better than the centre prior")
        line = (f"{name:32s} IoU {observed.mean():.4f} vs centre {centre_iou.mean():.4f}  "
                f"p={p:.3g}  {verdict}")
        if name in MODELS:
            hits = np.array([int(per_image_rows[f.name][f"{name}_pointing"]) for f in files])
            disagree = int(np.sum(hits != centre_hit))
            line += f"  | PointingGame {hits.mean():.3f} vs {centre_hit.mean():.3f}"
        print(line)
        rows.append(["vs centre prior", name, f"{observed.mean():.4f}", "", "",
                     f"{centre_iou.mean():.4f}", f"p={p:.3g}, {verdict}"])

    # confirm the formula empirically on the first few images
    check_name = MODELS[0]
    sim, ana = [], []
    for f in files[:5]:
        _, area = methods[check_name][f.name]
        sim.append(simulate_random_iou(area, masks[f.name], rng))
        ana.append(expected_random_iou(area, fracs[f.name]))
    print(f"\nformula check on 5 images ({check_name}): "
          f"analytic {np.mean(ana):.4f} vs simulated {np.mean(sim):.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "name", "IoU", "PointingGame", "Coverage",
                    "area_matched_IoU_floor", "verdict"])
        w.writerows(rows)
    print(f"Saved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
