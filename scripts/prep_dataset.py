"""
Build a balanced, leakage-free GLENDA binary dataset for endometriosis classification.

- Pathology (endometriosis): all 373 frames  + their segmentation masks (for Grad-CAM IoU)
- Normal (no-pathology): 373 frames sampled DIVERSELY across the ~20 videos
- Patient/case-wise split (NO frame from the same case/video in two splits) -> prevents leakage
- Output: data/{train,val,test}/{endometriosis,normal}/  (ImageFolder format)
          data/masks/{train,val,test}/endometriosis/     (ground-truth masks)
          data/manifest.csv

Run locally from the repo root: `python scripts/prep_dataset.py`
(Python 3 standard library only). Then zip `data/` and upload to Colab.
Seed = 42 for full reproducibility.
"""
import os, re, csv, random, shutil
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in scripts/)
PATHO_FRAMES = ROOT / "Dataset/Glenda_v1.5_classes/Glenda_v1.5_classes/frames"
PATHO_MASKS  = ROOT / "Dataset/Glenda_v1.5_classes/Glenda_v1.5_classes/annots"
NORMAL_ROOT  = ROOT / "Dataset/GLENDA_v1.5_no_pathology/no_pathology/frames"
OUT = ROOT / "data"

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

def case_id_from_patho(fname: str) -> str:
    """c_100_v_(video_3062.mp4)_f_1458.jpg -> c_100"""
    m = re.match(r"(c_\d+)", fname)
    return m.group(1) if m else "c_unknown"

def split_groups(group_to_files, ratios, seed):
    """Assign whole groups to train/val/test, approximating image-count ratios."""
    groups = sorted(group_to_files.keys())
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    total = sum(len(v) for v in group_to_files.values())
    targets = {s: r * total for s, r in ratios.items()}
    counts = {s: 0 for s in ratios}
    assign = {}
    order = ["train", "val", "test"]
    for g in groups:
        # give this group to the split most under its target
        s = max(order, key=lambda x: targets[x] - counts[x])
        assign[g] = s
        counts[s] += len(group_to_files[g])
    return assign, counts

# ---------- 1. PATHOLOGY: gather all frames, group by case ----------
patho_by_case = defaultdict(list)
for f in sorted(os.listdir(PATHO_FRAMES)):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        patho_by_case[case_id_from_patho(f)].append(f)
n_patho = sum(len(v) for v in patho_by_case.values())
print(f"Pathology: {n_patho} frames across {len(patho_by_case)} cases")

# ---------- 2. NORMAL: gather frames grouped by video, sample DIVERSELY ----------
normal_by_video = defaultdict(list)
for video_dir in sorted(os.listdir(NORMAL_ROOT)):
    vpath = NORMAL_ROOT / video_dir
    if vpath.is_dir():
        for f in sorted(os.listdir(vpath)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                normal_by_video[video_dir].append(str(vpath / f))
n_normal_total = sum(len(v) for v in normal_by_video.values())
print(f"Normal: {n_normal_total} frames across {len(normal_by_video)} videos")

# round-robin across videos so the 373 normals are spread out (not near-duplicates)
target_normal = n_patho
for v in normal_by_video:
    random.Random(SEED).shuffle(normal_by_video[v])
sampled_normal_by_video = defaultdict(list)
pools = {v: list(files) for v, files in normal_by_video.items()}
videos_cycle = sorted(pools.keys())
i = 0
picked = 0
while picked < target_normal:
    v = videos_cycle[i % len(videos_cycle)]
    if pools[v]:
        sampled_normal_by_video[v].append(pools[v].pop())
        picked += 1
    i += 1
    if i > target_normal * 50:  # safety
        break
print(f"Sampled {picked} normal frames across {len(sampled_normal_by_video)} videos")

# ---------- 3. Patient/case-wise split (per class, keeps balance) ----------
patho_assign, patho_counts = split_groups(patho_by_case, RATIOS, SEED)
normal_assign, normal_counts = split_groups(sampled_normal_by_video, RATIOS, SEED + 1)
print("Pathology split (images):", patho_counts)
print("Normal   split (images):", normal_counts)

# ---------- 4. Copy files into ImageFolder structure + masks + manifest ----------
if OUT.exists():
    shutil.rmtree(OUT)
manifest = []

def ensure(*p):
    d = OUT.joinpath(*p); d.mkdir(parents=True, exist_ok=True); return d

# pathology
for case, files in patho_by_case.items():
    split = patho_assign[case]
    dst = ensure(split, "endometriosis")
    mdst = ensure("masks", split, "endometriosis")
    for f in files:
        shutil.copy2(PATHO_FRAMES / f, dst / f)
        mask = PATHO_MASKS / (Path(f).stem + ".png")
        has_mask = mask.exists()
        if has_mask:
            shutil.copy2(mask, mdst / (Path(f).stem + ".png"))
        manifest.append([f, "endometriosis", 1, case, split, has_mask])

# normal
for video, files in sampled_normal_by_video.items():
    split = normal_assign[video]
    dst = ensure(split, "normal")
    for src in files:
        name = f"{video}__{Path(src).name}"
        shutil.copy2(src, dst / name)
        manifest.append([name, "normal", 0, video, split, False])

with open(OUT / "manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["filename", "class", "label", "group_id", "split", "has_mask"])
    w.writerows(manifest)

# ---------- 5. Summary ----------
print("\n=== FINAL BALANCED DATASET ===")
for split in ["train", "val", "test"]:
    e = len(list((OUT / split / "endometriosis").glob("*"))) if (OUT / split / "endometriosis").exists() else 0
    n = len(list((OUT / split / "normal").glob("*"))) if (OUT / split / "normal").exists() else 0
    print(f"{split:5s}: endometriosis={e:4d}  normal={n:4d}  total={e+n:4d}")
print(f"Total rows in manifest: {len(manifest)}")
print(f"Output -> {OUT}")
