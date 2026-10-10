"""Build the two figures the new results need.

fig_cv_auc.png
    Classification. Each fold is one dot, so the reader sees the spread instead
    of a box drawn through five points. The pooled out-of-fold AUC and its 95% CI
    sit underneath, and the old single-split value is marked separately - that
    contrast is the point: the single split ranked EfficientNet-B0 last.

fig_localization.png
    Explanation quality against the two reference levels a saliency claim has to
    clear: an area-matched random region, and a centred Gaussian prior. Without
    them an IoU of 0.148 is unreadable.

Palette: categorical slots 1-3 of the reference palette, validated for colour
vision deficiency (worst adjacent pair dE 9.2 deutan). Every value is also
direct-labelled, so colour is never the only channel.

Usage (from the repo root):
    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MODELS = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]
COLOR = {"ResNet50": "#2a78d6", "EfficientNet-B0": "#eb6834", "ViT-B/16": "#1baf7a"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})



def group_key(row: dict) -> str:
    """v2 runs record surgery_id; the first run recorded group_id (per segment)."""
    return row.get("surgery_id") or row["group_id"]

def read_csv(name: str) -> list[dict]:
    return list(csv.DictReader(open(RESULTS / name, encoding="utf-8")))


def tidy(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def figure_cv_auc() -> None:
    folds = read_csv("cv_fold_metrics.csv")
    per_model: dict[str, dict[int, float]] = {m: {} for m in MODELS}
    for r in folds:                                   # later rows win (a re-run fold)
        if r["model"] in per_model:
            per_model[r["model"]][int(r["fold"])] = float(r["AUC"])

    pooled, ci = {}, {}
    for r in read_csv("cv_analysis_summary.csv"):
        if r["section"] == "pooled_oof" and r["metric"] == "AUC":
            pooled[r["model"]] = float(r["value"])
            lo, hi = r["spread_or_ci"].strip("[]").split(",")
            ci[r["model"]] = (float(lo), float(hi))
    single = {r[""]: float(r["AUC"]) for r in read_csv("test_results.csv")}

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for i, name in enumerate(MODELS):
        y = len(MODELS) - 1 - i
        c = COLOR[name]
        aucs = [per_model[name][f] for f in sorted(per_model[name])]

        lo, hi = ci[name]
        ax.plot([lo, hi], [y - 0.22, y - 0.22], color=c, lw=2, solid_capstyle="round",
                alpha=0.55, zorder=2)
        ax.scatter([pooled[name]], [y - 0.22], s=58, color=c, zorder=4,
                   edgecolor="white", linewidth=1.2)
        ax.scatter(aucs, [y + 0.14] * len(aucs), s=34, color=c, alpha=0.8,
                   zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([single[name]], [y + 0.14], s=46, facecolor="white", zorder=4,
                   edgecolor=c, linewidth=1.6, marker="D")

        ax.text(pooled[name], y - 0.49, f"{pooled[name]:.3f}", ha="center",
                va="center", fontsize=8.5, color=INK)

    ax.text(0.0, 1.06, "each dot = one of the 5 folds      "
                       "◇ = old single split      ● = pooled over all 746 images",
            transform=ax.transAxes, fontsize=8, color=INK_SOFT, ha="left", va="bottom")
    ax.set_xlim(0.80, 0.99)
    ax.set_ylim(-0.75, 2.6)
    ax.set_yticks([len(MODELS) - 1 - i - 0.04 for i in range(len(MODELS))])
    ax.set_yticklabels(MODELS, fontsize=9.5, color=INK)
    ax.set_xlabel("ROC-AUC")
    tidy(ax)
    out = RESULTS / "fig_cv_auc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def figure_localization() -> None:
    base = {r["name"]: r for r in read_csv("localization_baselines.csv")}
    gc = read_csv("gradcam_per_image.csv")
    roll = read_csv("vit_rollout_per_image.csv")

    methods = [
        ("EfficientNet-B0", COLOR["EfficientNet-B0"], None,
         np.mean([float(r["EfficientNet-B0_IoU"]) for r in gc]),
         np.mean([int(r["EfficientNet-B0_pointing"]) for r in gc])),
        ("ResNet50", COLOR["ResNet50"], None,
         np.mean([float(r["ResNet50_IoU"]) for r in gc]),
         np.mean([int(r["ResNet50_pointing"]) for r in gc])),
        ("ViT-B/16\nGrad-CAM", COLOR["ViT-B/16"], None,
         np.mean([float(r["ViT-B/16_IoU"]) for r in gc]),
         np.mean([int(r["ViT-B/16_pointing"]) for r in gc])),
        ("ViT-B/16\nattention rollout", COLOR["ViT-B/16"], "///",
         np.mean([float(r["rollout_IoU"]) for r in roll]),
         np.mean([int(r["rollout_pointing"]) for r in roll])),
    ]
    centre = base["centre prior"]
    centre_iou, centre_point = float(centre["IoU"]), float(centre["PointingGame"])
    chance_point = float(base["uninformative heatmap"]["PointingGame"])
    floors = {m["name"]: float(m["area_matched_IoU_floor"])
              for m in read_csv("localization_baselines.csv")
              if m["kind"] == "method"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    panels = [
        (axes[0], "Grad-CAM / rollout overlap with the lesion mask (IoU)", 3, centre_iou, None),
        (axes[1], "Peak falls inside the lesion (Pointing Game)", 4, centre_point, chance_point),
    ]
    for ax, title, value_idx, centre_level, chance_level in panels:
        ys = np.arange(len(methods))[::-1]
        for y, (label, colour, hatch, iou_v, point_v) in zip(ys, methods):
            value = iou_v if value_idx == 3 else point_v
            ax.barh(y, value, height=0.52, color=colour, hatch=hatch,
                    edgecolor="white", linewidth=1.2, zorder=3)
            span = 0.33 if value_idx == 3 else 0.62
            if value > 0.3 * span:          # inside the bar, clear of the reference lines
                ax.text(value - 0.012 * span / 0.33, y, f"{value:.3f}", va="center",
                        ha="right", fontsize=8.5, color="white", zorder=5)
            else:
                ax.text(value + 0.012 * span / 0.33, y, f"{value:.3f}", va="center",
                        ha="left", fontsize=8.5, color=INK, zorder=5)
            if value_idx == 3:                       # per-method random floor
                floor = floors.get(label.replace("\n", " ").replace(
                    "ViT-B/16 Grad-CAM", "ViT-B/16").replace(
                    "ViT-B/16 attention rollout", "ViT-B/16 (attention rollout)"), None)
                if floor:
                    ax.plot([floor, floor], [y - 0.26, y + 0.26], color=INK_SOFT,
                            lw=1.4, zorder=5)

        if chance_level is not None:
            ax.axvline(chance_level, color=INK_SOFT, lw=1.2, ls=(0, (1, 2)), zorder=2)
            ax.text(chance_level, len(methods) - 0.42, " chance", fontsize=8,
                    color=INK_SOFT, ha="left", va="bottom")
        ax.axvline(centre_level, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.text(centre_level, len(methods) - 0.42, "  centre prior", fontsize=8,
                color=INK, ha="left", va="bottom")

        ax.set_yticks(ys)
        ax.set_yticklabels([m[0] for m in methods], fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=8.8, color=INK, pad=12, loc="left")
        ax.set_ylim(-0.6, len(methods) - 0.15)
        tidy(ax)

    axes[0].set_xlim(0, 0.33)
    axes[1].set_xlim(0, 0.62)
    fig.text(0.085, -0.04, "vertical tick on each bar = that method's area-matched random floor",
             fontsize=7.6, color=INK_SOFT, ha="left", va="top")
    fig.subplots_adjust(wspace=0.5)
    out = RESULTS / "fig_localization.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")




def figure_roc() -> None:
    """ROC from the pooled out-of-fold predictions, not the single split."""
    preds: dict[str, list[tuple[int, float]]] = {m: [] for m in MODELS}
    for r in read_csv("cv_predictions.csv"):
        if r["model"] in preds:
            preds[r["model"]].append((int(r["y_true"]), float(r["prob"])))
    ci = {}
    for r in read_csv("cv_analysis_summary.csv"):
        if r["section"] == "pooled_oof" and r["metric"] == "AUC":
            lo, hi = r["spread_or_ci"].strip("[]").split(",")
            ci[r["model"]] = (float(r["value"]), float(lo), float(hi))

    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for name in MODELS:
        rows = sorted(preds[name], key=lambda t: -t[1])
        pos = sum(1 for y, _ in rows if y == 1)
        neg = len(rows) - pos
        tpr, fpr = [0.0], [0.0]
        tp = fp = 0
        for y, _ in rows:
            tp, fp = tp + (y == 1), fp + (y == 0)
            tpr.append(tp / pos)
            fpr.append(fp / neg)
        auc, lo, hi = ci[name]
        ax.plot(fpr, tpr, color=COLOR[name], lw=2, solid_capstyle="round", zorder=3,
                label=f"{name}  {auc:.3f} [{lo:.3f}–{hi:.3f}]")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.01)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8, title="ROC-AUC [95% CI]",
                    handlelength=1.6, borderpad=0.2, labelspacing=0.5)
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_SOFT)
    tidy(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    out = RESULTS / "fig_roc_pooled.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def figure_confusion() -> None:
    """Pooled out-of-fold confusion matrices, one per model."""
    counts = {m: np.zeros((2, 2), dtype=int) for m in MODELS}
    for r in read_csv("cv_predictions.csv"):
        if r["model"] in counts:
            counts[r["model"]][int(r["y_true"]), int(r["pred"])] += 1

    ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ramp)
    labels = ["Normal", "Endometriosis"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, name in zip(axes, MODELS):
        cm = counts[name]
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if share[i, j] > 0.55 else INK)
        ax.set_title(name, fontsize=9, color=INK, pad=8)
        ax.set_xticks([0, 1], labels, fontsize=8)
        ax.set_yticks([0, 1], labels, fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=8.5, color=INK_SOFT)
        if name == MODELS[0]:
            ax.set_ylabel("Actual", fontsize=8.5, color=INK_SOFT)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(wspace=0.45)
    out = RESULTS / "fig_confusion_pooled.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def figure_error_clusters() -> None:
    """Where the errors live. Errors concentrate in a few whole surgeries, which
    is the argument for case-wise splitting and the lead for the discussion."""
    per_group: dict[str, dict[str, list[int]]] = {}
    label: dict[str, str] = {}
    for r in read_csv("cv_predictions.csv"):
        g = group_key(r)
        per_group.setdefault(g, {m: [] for m in MODELS})
        per_group[g][r["model"]].append(int(r["pred"] != r["y_true"]))
        label[g] = "endometriosis" if r["y_true"] == "1" else "no pathology"

    # a surgery contributing one or two frames cannot show a meaningful error
    # rate - 100% of one frame is noise, not a cluster
    MIN_FRAMES = 5
    rates = {g: {m: float(np.mean(v[m])) for m in MODELS}
             for g, v in per_group.items() if len(v[MODELS[0]]) >= MIN_FRAMES}
    worst = sorted(rates, key=lambda g: -np.mean([rates[g][m] for m in MODELS]))[:12]
    worst = sorted(worst, key=lambda g: np.mean([rates[g][m] for m in MODELS]))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = np.arange(len(worst))
    for name in MODELS:
        ax.scatter([rates[g][name] for g in worst], ys, s=42, color=COLOR[name],
                   edgecolor="white", linewidth=0.9, zorder=3, label=name)
    for y, g in zip(ys, worst):
        lo = min(rates[g][m] for m in MODELS)
        hi = max(rates[g][m] for m in MODELS)
        ax.plot([lo, hi], [y, y], color=GRID, lw=1.4, zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g}  ({len(per_group[g][MODELS[0]])} frames, {label[g]})"
                        for g in worst], fontsize=7.8, color=INK)
    kept = len(rates)
    ax.set_xlabel("share of the surgery's frames misclassified")
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.10), handletextpad=0.3, columnspacing=1.6)
    ax.set_title(f"The 12 hardest surgeries, of the {kept} with at least "
                 f"{MIN_FRAMES} frames", fontsize=9, color=INK, loc="left", pad=34)
    tidy(ax)
    out = RESULTS / "fig_error_clusters.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def figure_dataset() -> None:
    """Fold composition: equal-sized folds, but prevalence moves because whole
    surgeries are kept together. This is why AUC, not accuracy, is reported."""
    fold_rows: dict[int, list[int]] = {}
    groups: dict[int, set[str]] = {}
    for r in read_csv("cv_predictions.csv"):
        if r["model"] != MODELS[0]:
            continue
        f = int(r["fold"])
        fold_rows.setdefault(f, []).append(int(r["y_true"]))
        groups.setdefault(f, set()).add(group_key(r))

    folds = sorted(fold_rows)
    endo = np.array([sum(fold_rows[f]) for f in folds])
    normal = np.array([len(fold_rows[f]) - sum(fold_rows[f]) for f in folds])

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = np.arange(len(folds))
    ax.bar(x, endo, width=0.62, color=COLOR["EfficientNet-B0"], label="endometriosis",
           edgecolor="white", linewidth=1.2, zorder=3)
    ax.bar(x, normal, width=0.62, bottom=endo, color="#9ec5f4", label="no pathology",
           edgecolor="white", linewidth=1.2, zorder=3)
    for xi, e, n, f in zip(x, endo, normal, folds):
        ax.text(xi, e / 2, f"{e}", ha="center", va="center", fontsize=8.5, color="white")
        ax.text(xi, e + n / 2, f"{n}", ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(xi, e + n + 4, f"{e / (e + n):.0%} endo\n{len(groups[f])} surgeries",
                ha="center", va="bottom", fontsize=7.6, color=INK_SOFT)

    ax.set_xticks(x, [f"fold {f}" for f in folds], fontsize=8.5)
    ax.set_ylabel("frames")
    ax.set_ylim(0, 215)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.22), handlelength=1.2)
    tidy(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    out = RESULTS / "fig_dataset_folds.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    figure_cv_auc()
    figure_localization()
    figure_roc()
    figure_confusion()
    figure_error_clusters()
    figure_dataset()
