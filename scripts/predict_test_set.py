"""Run the three trained models over the test set and save per-image predictions.

The notebook only saved aggregate metrics, which is not enough for McNemar's test,
confidence intervals or error analysis. This script reloads the saved weights and
writes one row per test image per model.

Preprocessing, class order and positive class are kept identical to the notebook:
    Resize((224,224)) -> ToTensor -> Normalize(ImageNet)
    class_to_idx = {"endometriosis": 0, "normal": 1}, POS = 0

CPU-only; the test set is 107 images, so no GPU is needed.

Usage (from the repo root):
    python scripts/predict_test_set.py
Output:
    results/test_predictions.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
WEIGHTS_DIR = ROOT / "endometriosis_paper-20260927T165052Z-1-001" / "endometriosis_paper"
OUT_CSV = ROOT / "results" / "test_predictions.csv"

IMG = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_TO_IDX = {"endometriosis": 0, "normal": 1}
POS = CLASS_TO_IDX["endometriosis"]

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_resnet50() -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def build_efficientnet_b0() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def build_vit_b16() -> nn.Module:
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    return m


MODELS = {
    "ResNet50": (build_resnet50, "best_resnet50.pth"),
    "EfficientNet-B0": (build_efficientnet_b0, "best_efficientnet_b0.pth"),
    "ViT-B/16": (build_vit_b16, "best_vit_b16.pth"),
}


def load_model(builder, weight_name: str) -> nn.Module:
    path = WEIGHTS_DIR / weight_name
    if not path.exists():
        sys.exit(f"Missing weights: {path}")
    model = builder()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def group_id_of(filename: str, class_name: str) -> str:
    """Pathology frames are grouped by case (c_100_...), normals by video folder prefix."""
    if class_name == "endometriosis":
        return filename.split("_v_")[0]
    return filename.split("__")[0]


def test_images() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_name in sorted(CLASS_TO_IDX):
        for path in sorted((TEST_DIR / class_name).iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((path, class_name))
    return items


def main() -> None:
    images = test_images()
    if not images:
        sys.exit(f"No test images under {TEST_DIR}")
    print(f"Test images: {len(images)}")

    # probability of the endometriosis class, per model per image
    probs: dict[str, list[float]] = {}
    for name, (builder, weight_name) in MODELS.items():
        model = load_model(builder, weight_name)
        model_probs: list[float] = []
        with torch.no_grad():
            for i, (path, _) in enumerate(images, 1):
                x = EVAL_TF(Image.open(path).convert("RGB")).unsqueeze(0)
                p = torch.softmax(model(x), dim=1)[0, POS].item()
                model_probs.append(p)
                if i % 25 == 0:
                    print(f"  [{name}] {i}/{len(images)}")
        probs[name] = model_probs
        print(f"  [{name}] done")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["filename", "class", "group_id", "y_true"]
        for name in MODELS:
            header += [f"{name}_prob", f"{name}_pred"]
        writer.writerow(header)
        for idx, (path, class_name) in enumerate(images):
            y_true = 1 if class_name == "endometriosis" else 0
            row = [path.name, class_name, group_id_of(path.name, class_name), y_true]
            for name in MODELS:
                p = probs[name][idx]
                row += [f"{p:.6f}", int(p >= 0.5)]
            writer.writerow(row)

    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}  ({len(images)} rows)")

    # sanity check against the reported aggregate metrics
    y_true = [1 if c == "endometriosis" else 0 for _, c in images]
    for name in MODELS:
        preds = [int(p >= 0.5) for p in probs[name]]
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true, preds))
        tn = sum(t == 0 and p == 0 for t, p in zip(y_true, preds))
        acc = (tp + tn) / len(y_true)
        rec = tp / sum(y_true)
        print(f"{name:16s} acc={acc:.4f}  recall={rec:.4f}  (TP={tp} TN={tn})")


if __name__ == "__main__":
    main()
