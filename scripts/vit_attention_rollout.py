"""Attention rollout for ViT-B/16, scored against the ground-truth lesion masks.

Grad-CAM was designed for convolutional feature maps, so its poor showing on
ViT-B/16 (mean IoU 0.032) invites the obvious objection that the comparison is
unfair to the transformer. This script answers it with the explanation method
built for transformers: **attention rollout** (Abnar & Zuidema, 2020).

Rollout treats attention as a flow. Per layer it adds an identity matrix for the
residual connection, row-normalises, and multiplies the layers together; the CLS
row of the product says how much each input patch contributed to the class token.

torchvision's EncoderBlock calls MultiheadAttention with need_weights=False, so
the weights are never returned. Each attention module is therefore wrapped to ask
for averaged weights and stash them, then restored afterwards.

Scored exactly like the Grad-CAM run (same images, same masks, same 0.5 threshold)
so the two are directly comparable.

Usage (from the repo root):
    python scripts/vit_attention_rollout.py
Output:
    results/vit_rollout_per_image.csv
"""
from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from predict_test_set import MEAN, POS, STD, MODELS as MODEL_SPECS, load_model
from gradcam_per_image import IMG_DIR, MASK_DIR, iou, load_mask, pointing_and_coverage, preprocess

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "vit_rollout_per_image.csv"

GRID = 14                 # 224 / 16 patches per side
N_TOKENS = GRID * GRID + 1


@contextmanager
def capture_attention(model):
    """Force torchvision's ViT blocks to hand back their attention weights."""
    store: list[torch.Tensor] = []
    originals = []

    for block in model.encoder.layers:
        attn = block.self_attention
        originals.append((attn, attn.forward))

        def patched(query, key, value, _attn=attn, **kwargs):
            kwargs.pop("need_weights", None)
            kwargs.pop("average_attn_weights", None)
            out, weights = type(_attn).forward(
                _attn, query, key, value,
                need_weights=True, average_attn_weights=True, **kwargs)
            store.append(weights.detach())
            # the block unpacks two values and ignores the second
            return out, weights

        attn.forward = patched

    try:
        yield store
    finally:
        for attn, original in originals:
            attn.forward = original


def rollout(attentions: list[torch.Tensor]) -> np.ndarray:
    """Multiply (A + I), row-normalised, across layers; return the CLS->patch map."""
    eye = torch.eye(N_TOKENS)
    result = eye.clone()
    for attn in attentions:
        a = attn[0].cpu() + eye            # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    cls_to_patches = result[0, 1:]         # drop the CLS->CLS term
    grid = cls_to_patches.reshape(1, 1, GRID, GRID)
    up = F.interpolate(grid, size=(224, 224), mode="bilinear", align_corners=False)
    cam = up[0, 0].numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 0 else np.zeros_like(cam)


def main() -> None:
    builder, weight_name = MODEL_SPECS["ViT-B/16"]
    model = load_model(builder, weight_name)

    files = sorted(f for f in IMG_DIR.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and (MASK_DIR / f"{f.stem}.png").exists())
    print(f"Test pathology images with mask: {len(files)}")

    rows, ious, hits, covs = [], [], [], []
    with capture_attention(model) as store:
        for i, f in enumerate(files, 1):
            store.clear()
            x, _ = preprocess(f)
            with torch.no_grad():
                logits = model(x)
            prob = torch.softmax(logits, 1)[0, POS].item()
            cam = rollout(store)
            mask = load_mask(MASK_DIR / f"{f.stem}.png")
            v_iou = iou(cam, mask)
            hit, cov = pointing_and_coverage(cam, mask)
            area = float((cam >= 0.5).mean())
            rows.append([f.name, f"{v_iou:.6f}", hit, f"{cov:.6f}", f"{area:.6f}", f"{prob:.6f}"])
            ious.append(v_iou); hits.append(hit); covs.append(cov)
            if i % 10 == 0:
                print(f"  {i}/{len(files)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "rollout_IoU", "rollout_pointing", "rollout_coverage",
                    "rollout_area", "prob"])
        w.writerows(rows)

    print(f"\nViT-B/16 attention rollout over {len(files)} images:")
    print(f"  mean IoU      = {np.mean(ious):.4f} +- {np.std(ious):.4f}")
    print(f"  PointingGame  = {np.mean(hits):.4f}")
    print(f"  Coverage      = {np.mean(covs):.4f}")
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
