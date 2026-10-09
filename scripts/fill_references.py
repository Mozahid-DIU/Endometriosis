"""Rewrite paper/references.bib with the metadata confirmed by verify_references.py.

Three sources of truth, in order of trust:

  1. CANONICAL  - the handful of method/architecture papers whose standard citation is
                  fixed and well known. Crossref search actively mis-resolves some of
                  them (searching "EfficientNet" returns a different paper called
                  "EnhanceNet", and "An Image is Worth 16x16 Words" returns an SSRN
                  look-alike), so these are written from the canonical venue, never
                  from a search hit.
  2. MANUAL     - entries resolved from the publisher or PubMed because Crossref does
                  not index them.
  3. Crossref   - reference_check.csv rows with status VERIFIED / FOUND BY TITLE and a
                  title similarity of at least MIN_SIM.

Anything left unresolved keeps its VERIFY marker and is listed at the end.

Usage (from the repo root):
    python scripts/fill_references.py          # writes references.bib, keeps a .bak
    python scripts/fill_references.py --dry    # only report what would change
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "references.bib"
CHECK = ROOT / "paper" / "reference_check.csv"
MIN_SIM = 0.95

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.DOTALL)

# 1. canonical ML papers: fixed citations, never taken from a search hit
CANONICAL: dict[str, dict[str, str]] = {
    "ref_resnet": {
        "kind": "inproceedings",
        "title": "Deep Residual Learning for Image Recognition",
        "author": "He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian",
        "booktitle": "Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)",
        "pages": "770--778", "year": "2016", "doi": "10.1109/CVPR.2016.90",
    },
    "ref_efficientnet": {
        "kind": "inproceedings",
        "title": "{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks",
        "author": "Tan, Mingxing and Le, Quoc V.",
        "booktitle": "Proc. 36th Int. Conf. on Machine Learning (ICML)",
        "series": "PMLR", "volume": "97", "pages": "6105--6114", "year": "2019",
        "note": "arXiv:1905.11946",
    },
    "ref_vit": {
        "kind": "inproceedings",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "author": ("Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and "
                   "Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and "
                   "Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and "
                   "Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil"),
        "booktitle": "Int. Conf. on Learning Representations (ICLR)",
        "year": "2021", "note": "arXiv:2010.11929",
    },
    "ref_shap": {
        "kind": "inproceedings",
        "title": "A Unified Approach to Interpreting Model Predictions",
        "author": "Lundberg, Scott M. and Lee, Su-In",
        "booktitle": "Advances in Neural Information Processing Systems (NeurIPS)",
        "volume": "30", "pages": "4765--4774", "year": "2017",
        "note": "arXiv:1705.07874",
    },
    "ref_lime": {
        "kind": "inproceedings",
        "title": "``Why Should I Trust You?'': Explaining the Predictions of Any Classifier",
        "author": "Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos",
        "booktitle": "Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)",
        "pages": "1135--1144", "year": "2016", "doi": "10.1145/2939672.2939778",
    },
}

# 2. resolved by hand from the publisher / PubMed (not indexed in Crossref search)
MANUAL: dict[str, dict[str, str]] = {
    "ref_m1": {
        "kind": "article",
        "title": ("Machine Learning-Based Detection of Endometriosis: A Retrospective Study "
                  "in a Population of Iranian Female Patients"),
        "author": ("Nouri, Bijan and Hashemi, Seyed Hamid and Ghadimi, Delaram J. and "
                   "Roshandel, Sepideh and Akhlaghdoust, Meisam"),
        "journal": "International Journal of Fertility and Sterility",
        "year": "2024", "doi": "10.22074/ijfs.2024.2009338.1519",
        "note": "PMID 39564827; PMCID PMC11589974",
    },
    "ref_a7": {
        "kind": "article",
        "title": "Automated prediction of endometriosis using deep learning",
        "author": "Visalaxi, S. and Muthu, T. Sudalai",
        "journal": "International Journal of Nonlinear Analysis and Applications",
        "volume": "12", "number": "2", "pages": "2403--2416", "year": "2021",
        "doi": "10.22075/ijnaa.2021.5383",
    },
}

FIELD_ORDER = ["title", "author", "journal", "booktitle", "series", "publisher",
               "volume", "number", "pages", "year", "doi", "note"]


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def read_field(body: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*\{{", body)
    if not m:
        return ""
    i, depth, out = m.end(), 1, []
    while i < len(body) and depth:
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                break
        out.append(ch)
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def crossref_updates() -> dict[str, dict[str, str]]:
    """Trustworthy Crossref rows: right status and a near-exact title match."""
    out: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(open(CHECK, encoding="utf-8")):
        key, status = row["key"], row["status"]
        sim = float(row["similarity"] or 0)
        if key in CANONICAL or key in MANUAL:
            continue
        usable = status in {"VERIFIED", "FOUND BY TITLE"} and sim >= MIN_SIM
        # a9/x5 resolve to the same paper under a longer official title
        usable = usable or (status == "POSSIBLE MATCH" and key in {"ref_a9", "ref_x5"})
        if not usable:
            continue
        upd = {}
        if row["found_authors"]:
            upd["author"] = row["found_authors"]
        if row["found_journal"]:
            upd["journal"] = row["found_journal"]
        if row["found_year"]:
            upd["year"] = row["found_year"]
        if row["found_doi"]:
            upd["doi"] = row["found_doi"]
        if status == "POSSIBLE MATCH" and row["found_title"]:
            upd["title"] = row["found_title"]   # adopt the official longer title
        out[key] = upd
    return out


def render(kind: str, key: str, fields: dict[str, str]) -> str:
    lines = [f"@{kind}{{{key},"]
    present = [f for f in FIELD_ORDER if fields.get(f)]
    for i, name in enumerate(present):
        comma = "," if i < len(present) - 1 else ""
        lines.append(f"  {name:9s} = {{{fields[name]}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    dry = "--dry" in sys.argv
    text = BIB.read_text(encoding="utf-8")
    cross = crossref_updates()

    changed, unresolved, kept = [], [], []
    new_text = text
    for kind, raw_key, body in ENTRY_RE.findall(text):
        key = raw_key.strip()
        old_block = re.search(rf"@{kind}\s*\{{\s*{re.escape(key)},.*?\n\}}", text, re.DOTALL)
        assert old_block, key

        current = {name: read_field(body, name) for name in FIELD_ORDER}
        current = {k: v for k, v in current.items() if v}

        if key in CANONICAL:
            spec = dict(CANONICAL[key]); new_kind = spec.pop("kind", kind)
            fields, source = spec, "canonical"
        elif key in MANUAL:
            spec = dict(MANUAL[key]); new_kind = spec.pop("kind", kind)
            fields, source = spec, "manual"
        elif key in cross:
            new_kind = kind
            fields = {**current, **cross[key]}
            # a journal article that Crossref files under a book series stays as-is
            if fields.get("booktitle"):
                fields.pop("journal", None)
            fields, source = fields, "crossref"
        else:
            kept.append(key)
            if "VERIFY" in (current.get("author", "") + current.get("journal", "")):
                unresolved.append(key)
            continue

        fields = {k: v for k, v in fields.items() if v and "VERIFY" not in v}
        block = render(new_kind, key, fields)
        if block != old_block.group(0):
            changed.append((key, source))
            new_text = new_text.replace(old_block.group(0), block)

    print(f"{len(changed)} entries updated:")
    for key, source in changed:
        print(f"  {key:18s} <- {source}")
    if kept:
        print(f"\n{len(kept)} entries left untouched: {', '.join(kept)}")
    if unresolved:
        print(f"\nSTILL NEEDS A HUMAN ({len(unresolved)}): {', '.join(unresolved)}")

    remaining = new_text.count("VERIFY")
    if dry:
        print(f"\n[dry run] VERIFY markers would drop to {remaining}")
        return
    shutil.copy2(BIB, BIB.with_suffix(".bib.bak"))
    BIB.write_text(new_text, encoding="utf-8")
    print(f"\nWrote {BIB.relative_to(ROOT)} (backup: references.bib.bak)")
    print(f"VERIFY markers remaining: {remaining}")


if __name__ == "__main__":
    main()
