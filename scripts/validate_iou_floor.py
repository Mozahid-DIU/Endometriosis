"""Confirm the closed-form random-IoU floor by simulation, and record the result.

The paper's chance level for IoU is the expected overlap of a region of area *p*
placed independently of a mask of area *m*:

    E[IoU] = pm / (p + m - pm)

That formula is quoted in the Methodology, so the check behind it should be an
artifact rather than a line of console output. For each frame this draws uniform
noise, thresholds it at the quantile that reproduces that method's claimed area,
and measures the IoU against the real mask.

Note what the formula assumes: a spatially uniform random region, not a connected
blob. It is a floor for "no spatial information", not a null for contiguous CAMs.

Usage (from the repo root):
    python scripts/validate_iou_floor.py
Output:
    results/iou_floor_validation.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOC_CSV = RESULTS / "cv_localization_v3.csv"
OUT_CSV = RESULTS / "iou_floor_validation.csv"
MASK_DIRS = [ROOT / "data" / "masks" / s / "endometriosis"
             for s in ("train", "val", "test")]

N_FRAMES = 25          # enough to show agreement without a long run
N_DRAWS = 400          # random placements per frame
SEED = 42


def mask_for(filename: str) -> np.ndarray | None:
    stem = Path(filename).stem + ".png"
    for d in MASK_DIRS:
        if (d / stem).exists():
            m = np.array(Image.open(d / stem).convert("L").resize((224, 224)))
            return (m > 0).astype(np.uint8)
    return None


def analytic(p: float, m: float) -> float:
    denom = p + m - p * m
    return (p * m) / denom if denom else 0.0


def simulate(p: float, mask: np.ndarray, rng) -> float:
    out = []
    for _ in range(N_DRAWS):
        noise = rng.random(mask.shape)
        region = (noise >= np.quantile(noise, 1 - p)).astype(np.uint8)
        union = (region | mask).sum()
        out.append((region & mask).sum() / union if union else 0.0)
    return float(np.mean(out))


def main() -> None:
    rows = [r for r in csv.DictReader(open(LOC_CSV, encoding="utf-8"))
            if r["model"] == "ResNet50" and r["method"] == "grad-cam"][:N_FRAMES]
    rng = np.random.default_rng(SEED)

    out, a_all, s_all = [], [], []
    for r in rows:
        mask = mask_for(r["filename"])
        if mask is None:
            continue
        p, m = float(r["area"]), float(mask.mean())
        a, s = analytic(p, m), simulate(p, mask, rng)
        a_all.append(a); s_all.append(s)
        out.append([r["filename"], f"{p:.6f}", f"{m:.6f}", f"{a:.6f}", f"{s:.6f}",
                    f"{s - a:+.6f}"])

    a_mean, s_mean = float(np.mean(a_all)), float(np.mean(s_all))
    print(f"{len(out)} frames, {N_DRAWS} random placements each")
    print(f"  analytic mean  {a_mean:.4f}")
    print(f"  simulated mean {s_mean:.4f}")
    print(f"  difference     {s_mean - a_mean:+.5f} "
          f"({abs(s_mean - a_mean) / a_mean:.2%} of the floor)")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "claimed_area", "mask_area", "analytic_iou",
                    "simulated_iou", "difference"])
        w.writerows(out)
        w.writerow(["MEAN", "", "", f"{a_mean:.6f}", f"{s_mean:.6f}",
                    f"{s_mean - a_mean:+.6f}"])
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
