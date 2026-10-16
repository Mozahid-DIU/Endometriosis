"""Check the manuscript's numbers and claims against the authoritative result files.

Rewriting prose by hand - or through a paraphrasing tool - is where a correct
number quietly becomes a wrong one, and where a carefully hedged sentence turns
into a claim the data do not support. This reads every section in paper/sections/,
pulls out the numbers, and compares them with the values the result files contain.

Two design choices make the check meaningful rather than decorative:

  * Only the CURRENT result files count. An earlier version accepted a value if it
    appeared anywhere in any CSV, superseded runs included, which would have let a
    stale single-split number pass unnoticed.
  * A value matches only at the precision the manuscript prints it. An earlier
    version also stored every value rounded to two and three decimals and allowed a
    tolerance on top, so roughly nineteen in twenty arbitrary numbers were accepted
    as known. Run --self-test to see the current rate.

Reported:

  MISMATCH   a number in the text that matches nothing in the result files
  OVERCLAIM  a phrase that asserts more than the analysis established
  WEAKENED   a hedge that was in the source draft and has gone missing

Usage (from the repo root):
    python scripts/check_paper_numbers.py
    python scripts/check_paper_numbers.py --section 05_results
    python scripts/check_paper_numbers.py --self-test
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "paper" / "sections"
RESULTS = ROOT / "results"

# The run the paper reports. Superseded artifacts (the single-split experiment, the
# segment-grouped cross-validation) sit beside these and must not vouch for a number.
# Summary files only. The per-frame files (cv_predictions_v3.csv,
# cv_localization_v3.csv) hold thousands of values and would vouch for almost any
# two-decimal number; the manuscript quotes summaries, so summaries are the check.
AUTHORITATIVE = [
    "cv_fold_metrics_v3.csv",
    "cv_analysis_summary.csv",
    "cv_analysis_pairwise.csv",
    "cluster_bootstrap_classification.csv",
    "cv_localization_summary.csv",
    "cv_summary.csv",
    "cv_pooled.csv",
    "iou_floor_validation.csv",
]

# Numbers that are deliberately not results of the current run: the single split the
# paper reports as history, the segment-grouped run it documents, and constants.
EXTRA_KNOWN: dict[float, str] = {
    0.915: "single-split ResNet50 AUC (results/test_results.csv)",
    0.860: "single-split ViT-B/16 AUC",
    0.833: "single-split EfficientNet-B0 AUC",
    0.913: "segment-grouped EfficientNet-B0 AUC (results/legacy/)",
    0.898: "segment-grouped ResNet50 AUC (results/legacy/)",
    0.859: "segment-grouped ViT-B/16 AUC (results/legacy/)",
    0.08: "single-split gap between ResNet50 and EfficientNet-B0",
    25.6: "ResNet50 parameter count, millions (torchvision)",
    5.3: "EfficientNet-B0 parameter count, millions (torchvision)",
    86.0: "ViT-B/16 parameter count, millions (torchvision)",
    0.25: "sigma of the centred Gaussian, in image widths",
    1.9: "narrowest frame-to-surgery interval ratio",
    2.6: "widest frame-to-surgery interval ratio",
    0.114: "ResNet50 minus rollout IoU (cv_localization_summary.csv)",
    0.142: "EfficientNet-B0 minus rollout IoU (cv_localization_summary.csv)",
}

OVERCLAIMS: list[tuple[str, str, str]] = [
    (r"\b(models?|architectures?)\s+(are|were|is|was)\s+(equal|equivalent|the same)\b",
     "non-significance is not equivalence", "no evidence of a difference"),
    (r"\btied\b", "a null result is not a tie", "no evidence of a difference"),
    (r"\bprove[sd]?\b|\bproof\b(?!-of-concept)",
     "statistics do not prove", "indicate / provide evidence for"),
    (r"\bthe model (uses|used|relies on|focuses on) the lesion\b",
     "map-mask overlap measures agreement, not what the model used",
     "the maps agree with the expert masks"),
    (r"\bfaithful(ly|ness)?\b", "we measured spatial agreement, not faithfulness",
     "spatial agreement with the annotation"),
    (r"\b(diagnos\w+|clinical\w*)\s+(tool|use|ready|deployment|application)\b",
     "no clinical validation was performed", "research prototype; not evaluated clinically"),
    (r"\baccurate diagnosis\b|\bdiagnoses endometriosis\b",
     "the task is frame-level classification, not diagnosis", "frame-level classification"),
    (r"\bsignificantly (better|worse|outperform\w*)\b",
     "check this comparison survived Holm correction", "state the corrected p-value beside it"),
    (r"\bat chance\b", "claimable only for metrics actually tested against chance",
     "name the metric and its p-value"),
    (r"\bleakage[- ]free\b", "true only for the recording-level grouping; say which unit",
     "grouped by surgery, with the grouping rule stated"),
    (r"\bthree different orderings\b", "two of the three protocols agree",
     "the segment-grouped protocol reversed the order"),
]

REQUIRED_HEDGES = {
    "no evidence": ("05_results", "the classification comparisons are inconclusive, not null"),
    "do not claim": ("06_discussion", "the clinical and faithfulness disclaimers"),
    "hypothesis": ("06_discussion", "the capacity/context explanation is a hypothesis"),
}

CROSS_REFERENCE = re.compile(
    r"\b(Section|Sec\.|Fig\.|Figure|Table|Eq\.)\s*\d+(\.\d+)?[a-z]?", re.IGNORECASE)
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)\s*(%?)")


def decimals_in(text: str) -> list[tuple[float, int]]:
    """Each decimal, with the number of places the manuscript wrote it to. A value
    written as a percentage is converted to the fraction the result files store."""
    out = []
    for m in DECIMAL.finditer(text):
        literal, is_percent = m.group(1), bool(m.group(2))
        places = len(literal.split(".")[1])
        value = float(literal)
        out.append((value / 100, places + 2) if is_percent else (value, places))
    return out


def known_values() -> set[float]:
    values: set[float] = set()
    missing = []
    for name in AUTHORITATIVE:
        path = RESULTS / name
        if not path.exists():
            missing.append(name)
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                for cell in row:
                    for m in re.finditer(r"[-+]?\d+\.\d+", cell):
                        values.add(abs(float(m.group())))
    if missing:
        print(f"WARNING: missing authoritative files: {', '.join(missing)}\n")
    return values


def matches(value: float, places: int, known: set[float]) -> bool:
    """A printed value matches if some stored value agrees with it at that precision.

    The comparison is by distance rather than by round(), because the result files
    already store rounded values: 0.176521... is saved as 0.1765, and rounding that
    again to three places gives 0.176 while the manuscript correctly prints 0.177.
    """
    half = 0.5 * 10 ** -places + 1e-12
    for extra in EXTRA_KNOWN:
        if abs(extra - value) <= half:
            return True
    return any(abs(k - value) <= half for k in known)


def check_section(path: Path, known: set[float]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = CROSS_REFERENCE.sub("", line)
        for value, places in decimals_in(stripped):
            if value in {0.5, 1.0, 2.0, 5.0}:        # thresholds, not measurements
                continue
            if not matches(value, places, known):
                problems.append(f"  MISMATCH  line {line_no}: {value} at {places} dp "
                                f"is in no authoritative result file\n"
                                f"      {line.strip()[:100]}")

        for pattern, why, instead in OVERCLAIMS:
            for m in re.finditer(pattern, line, re.IGNORECASE):
                before = line[:m.start()]
                if re.search(r"\b(not|no|never|cannot|nor)\b[^.]{0,40}$", before, re.IGNORECASE):
                    continue                          # the phrase is being denied
                problems.append(f"  OVERCLAIM line {line_no}: \"{m.group()}\" - {why}"
                                f"\n      consider: {instead}\n      {line.strip()[:100]}")

    for hedge, (section, why) in REQUIRED_HEDGES.items():
        if path.stem == section and hedge not in text.lower():
            problems.append(f"  WEAKENED  \"{hedge}\" no longer appears - {why}")

    return problems


def self_test() -> None:
    """How often would the checker vouch for a number that is not a result?"""
    import random
    known = known_values()
    rng = random.Random(42)
    print("false-acceptance rate on random values:")
    for places in (2, 3, 4):
        trials = 2000
        accepted = sum(matches(round(rng.uniform(0, 1), places), places, known)
                       for _ in range(trials))
        print(f"  {places}-decimal: {accepted}/{trials} ({accepted / trials:.1%})")
    print("\nLower is better: these are numbers the checker would wrongly accept.")


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    wanted = sys.argv[sys.argv.index("--section") + 1] if "--section" in sys.argv else None
    known = known_values()
    print(f"{len(known)} values across {len(AUTHORITATIVE)} authoritative result files\n")

    total = 0
    for path in sorted(SECTIONS.glob("*.md")):
        if wanted and wanted not in path.stem:
            continue
        problems = check_section(path, known)
        total += len(problems)
        words = len(path.read_text(encoding="utf-8").split())
        status = "clean" if not problems else f"{len(problems)} to review"
        print(f"{path.name}  ({words} words) - {status}")
        for p in problems:
            print(p)
        if problems:
            print()

    print(f"\n{total} item(s) to review.")
    print("MISMATCH is the one that matters: a number no current result file contains.")
    print("OVERCLAIM and WEAKENED are prompts to re-read, not errors by themselves.")


if __name__ == "__main__":
    main()
