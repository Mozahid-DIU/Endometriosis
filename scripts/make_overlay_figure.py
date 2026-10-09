"""Qualitative figure: what each model actually looks at.

Five columns per example - the frame with the expert lesion mask, then Grad-CAM
for the two CNNs, then ViT-B/16 under Grad-CAM and under attention rollout. The
numbers say ViT-B/16 does not localise; this figure lets a reader see it, and
shows that its native explanation method does not change the picture.

Three examples are chosen by EfficientNet-B0's IoU - best, median and worst - so
the figure is not a cherry-picked success.

Usage (from the repo root):
    python scripts/make_overlay_figure.py
Output:
    results/fig_overlays.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from gradcam_per_image import (IMG_DIR, MASK_DIR, load_mask, preprocess, target_layers)
from predict_test_set import MODELS as MODEL_SPECS, POS, load_model
from vit_attention_rollout import capture_attention, rollout

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "fig_overlays.png"
INK, INK_SOFT = "#0b0b0b", "#52514e"


def pick_examples() -> list[tuple[str, float]]:
    rows = list(csv.DictReader(open(RESULTS / "gradcam_per_image.csv", encoding="utf-8")))
    ranked = sorted(rows, key=lambda r: float(r["EfficientNet-B0_IoU"]))
    chosen = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]   # best, median, worst
    return [(r["filename"], float(r["EfficientNet-B0_IoU"])) for r in chosen]


def main() -> None:
    examples = pick_examples()
    print("examples:", [(f, round(v, 3)) for f, v in examples])

    models = {name: load_model(*spec) for name, spec in
              ((n, MODEL_SPECS[n]) for n in ("ResNet50", "EfficientNet-B0", "ViT-B/16"))}

    columns = ["Frame + expert mask", "ResNet50", "EfficientNet-B0",
               "ViT-B/16\nGrad-CAM", "ViT-B/16\nattention rollout"]
    fig, axes = plt.subplots(len(examples), len(columns),
                             figsize=(9.2, 2.0 * len(examples)))

    for row, (fname, _) in enumerate(examples):
        path = IMG_DIR / fname
        x, rgb = preprocess(path)
        mask = load_mask(MASK_DIR / f"{Path(fname).stem}.png")

        ax = axes[row, 0]
        ax.imshow(rgb)
        ax.contour(mask, levels=[0.5], colors="#eb6834", linewidths=1.6)
        ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.22, cmap="autumn")

        for col, name in enumerate(("ResNet50", "EfficientNet-B0", "ViT-B/16"), start=1):
            model = models[name]
            layers, reshape = target_layers(name, model)
            cam = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(POS)])[0]
            axes[row, col].imshow(show_cam_on_image(rgb.astype(np.float32), grayscale,
                                                    use_rgb=True))
            axes[row, col].contour(mask, levels=[0.5], colors="white", linewidths=1.2)

        model = models["ViT-B/16"]
        with capture_attention(model) as store:
            store.clear()
            with torch.no_grad():
                model(x)
            cam_map = rollout(store)
        axes[row, 4].imshow(show_cam_on_image(rgb.astype(np.float32), cam_map, use_rgb=True))
        axes[row, 4].contour(mask, levels=[0.5], colors="white", linewidths=1.2)

        for col in range(len(columns)):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            for side in axes[row, col].spines.values():
                side.set_visible(False)
        print(f"  row {row} done ({fname})")

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9, color=INK, pad=8)
    for row, (_, iou_v) in enumerate(examples):
        tag = ("best", "median", "worst")[row]
        axes[row, 0].set_ylabel(f"{tag} case\nEfficientNet IoU {iou_v:.2f}",
                                fontsize=8.2, color=INK_SOFT)

    fig.text(0.5, 0.045, "Lesion outlined in white (orange on the frame). "
                         "Warm colour = the region the model used for its decision.",
             ha="center", fontsize=8, color=INK_SOFT)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
