"""Per-image Grad-CAM localization metrics against the ground-truth lesion masks.

The notebook printed only the means. Significance testing needs the raw per-image
values, so this script recomputes IoU / Pointing Game / Coverage image by image
with the same configuration as the notebook:

    ResNet50        -> layer4[-1]
    EfficientNet-B0 -> features[-1]
    ViT-B/16        -> encoder.layers[-1].ln_1 with a token->grid reshape
    IoU threshold   -> CAM >= 0.5, mask binarised at > 0

CPU-only. Grad-CAM needs a backward pass, so ViT takes a few seconds per image;
53 test pathology images x 3 models is roughly 5-10 minutes.

Usage (from the repo root):
    python scripts/gradcam_per_image.py
Output:
    results/gradcam_per_image.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from predict_test_set import (MEAN, POS, STD, MODELS as MODEL_SPECS, load_model)

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "test" / "endometriosis"
MASK_DIR = ROOT / "data" / "masks" / "test" / "endometriosis"
OUT_CSV = ROOT / "results" / "gradcam_per_image.csv"

CAM_THRESHOLD = 0.5


def preprocess(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    x = torch.tensor(arr).permute(2, 0, 1).float()
    x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x.unsqueeze(0), arr


def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L").resize((224, 224)))
    return (m > 0).astype(np.uint8)


def iou(cam: np.ndarray, mask: np.ndarray, thr: float = CAM_THRESHOLD) -> float:
    pred = (cam >= thr).astype(np.uint8)
    union = (pred | mask).sum()
    return float((pred & mask).sum() / union) if union else 0.0


def pointing_and_coverage(cam: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    hit = int(mask[peak] > 0)
    coverage = float((cam * mask).sum() / (cam.sum() + 1e-8))
    return hit, coverage


def vit_reshape(t: torch.Tensor, h: int = 14, w: int = 14) -> torch.Tensor:
    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
    return r.transpose(2, 3).transpose(1, 2)


def target_layers(name: str, model) -> tuple[list, object | None]:
    if name == "ResNet50":
        return [model.layer4[-1]], None
    if name == "EfficientNet-B0":
        return [model.features[-1]], None
    if name == "ViT-B/16":
        return [model.encoder.layers[-1].ln_1], vit_reshape
    raise ValueError(name)


def main() -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        sys.exit("pip install grad-cam")

    files = sorted(f for f in IMG_DIR.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and (MASK_DIR / f"{f.stem}.png").exists())
    print(f"Test pathology images with mask: {len(files)}")

    records: dict[str, dict[str, tuple[float, int, float, float]]] = {}
    for name, (builder, weight_name) in MODEL_SPECS.items():
        model = load_model(builder, weight_name)
        layers, reshape = target_layers(name, model)
        cam_engine = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)
        per_file: dict[str, tuple[float, int, float]] = {}
        for i, f in enumerate(files, 1):
            x, _ = preprocess(f)
            cam = cam_engine(input_tensor=x, targets=[ClassifierOutputTarget(POS)])[0]
            mask = load_mask(MASK_DIR / f"{f.stem}.png")
            hit, coverage = pointing_and_coverage(cam, mask)
            area = float((cam >= CAM_THRESHOLD).mean())   # how much of the frame it claims
            per_file[f.name] = (iou(cam, mask), hit, coverage, area)
            if i % 10 == 0:
                print(f"  [{name}] {i}/{len(files)}")
        records[name] = per_file
        ious = [v[0] for v in per_file.values()]
        hits = [v[1] for v in per_file.values()]
        covs = [v[2] for v in per_file.values()]
        print(f"  [{name}] mean IoU={np.mean(ious):.4f}+-{np.std(ious):.4f}  "
              f"PointingGame={np.mean(hits):.4f}  Coverage={np.mean(covs):.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["filename"]
        for name in MODEL_SPECS:
            header += [f"{name}_IoU", f"{name}_pointing", f"{name}_coverage", f"{name}_area"]
        w.writerow(header)
        for f in files:
            row: list = [f.name]
            for name in MODEL_SPECS:
                v_iou, hit, cov, area = records[name][f.name]
                row += [f"{v_iou:.6f}", hit, f"{cov:.6f}", f"{area:.6f}"]
            w.writerow(row)
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}  ({len(files)} rows)")


if __name__ == "__main__":
    main()
