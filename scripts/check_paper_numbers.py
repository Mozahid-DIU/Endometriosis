"""Check the manuscript's numbers and claims against the result files.

Rewriting prose by hand — or through a paraphrasing tool — is where a correct
number quietly becomes a wrong one, and where a carefully hedged sentence turns
into a claim the data do not support. This script reads every section in
paper/sections/, pulls out the numbers, and compares them with the values actually
present in results/. It also flags wording that overstates what was measured.

It reports three things:

  MISMATCH   a number in the text that matches nothing in the result files
  OVERCLAIM  a phrase that asserts more than the analysis established
  WEAKENED   a hedge that was in the source draft and has gone missing

Nothing here is authoritative about style; it is a safety net for meaning.

Usage (from the repo root):
    python scripts/check_paper_numbers.py
    python scripts/check_paper_numbers.py --section 05_results
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "paper" / "sections"
RESULTS = ROOT / "results"

# a number is "known" if it appears in a result file at this precision
TOLERANCE = 0.0006

# Phrases that claim more than this study can support. Each entry is
# (regex, why it is a problem, what to say instead).
OVERCLAIMS: list[tuple[str, str, str]] = [
    (r"\b(models?|architectures?)\s+(are|were|is|was)\s+(equal|equivalent|the same)\b",
     "non-significance is not equivalence",
     "no evidence of a difference"),
    (r"\btied\b",
     "a null result is not a tie",
     "no evidence of a difference"),
    (r"\bprove[sd]?\b|\bproof\b(?!-of-concept)",
     "statistics do not prove",
     "indicate / provide evidence for"),
    (r"\bthe model (uses|used|relies on|focuses on) the lesion\b",
     "map-mask overlap measures agreement, not what the model used",
     "the maps agree with the expert masks"),
    (r"\bfaithful(ly|ness)?\b",
     "we measured spatial agreement, not faithfulness",
     "spatial agreement with the annotation"),
    (r"\b(diagnos\w+|clinical\w*)\s+(tool|use|ready|deployment|application)\b",
     "no clinical validation was performed",
     "research prototype; not evaluated clinically"),
    (r"\baccurate diagnosis\b|\bdiagnoses endometriosis\b",
     "the task is frame-level classification, not diagnosis",
     "frame-level classification"),
    (r"\bsignificantly (better|worse|outperform\w*)\b",
     "check this specific comparison survived Holm correction",
     "state the corrected p-value beside it"),
    (r"\bat chance\b",
     "only claimable for metrics actually tested against chance",
     "name the metric and its p-value"),
    (r"\bno leakage\b|\bleakage[- ]free\b",
     "true only for the recording-level grouping; say which unit",
     "grouped by surgery, with the grouping rule stated"),
]

# Hedges that must survive any rewrite.
REQUIRED_HEDGES: list[tuple[str, str]] = [
    ("no evidence", "the classification comparisons are inconclusive, not null"),
    ("do not claim", "the clinical and faithfulness disclaimers"),
    ("hypothesis", "the capacity/context explanation is a hypothesis"),
]


def numbers_in(text: str) -> set[float]:
    """Every decimal in the line. A value written as a percentage is also offered
    as its fraction, since the result files store fractions."""
    out = set()
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+)\s*(%?)", text):
        value = float(m.group(1))
        out.add(value if not m.group(2) else round(value / 100, 6))
    return out


def known_values() -> set[float]:
    """Every number any result file contains, plus their 3-decimal roundings."""
    values: set[float] = set()
    for path in sorted(RESULTS.glob("*.csv")):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                for cell in row:
                    for m in re.finditer(r"[-+]?\d+\.\d+", cell):
                        v = abs(float(m.group()))
                        values.add(v)
                        values.add(round(v, 3))
                        values.add(round(v, 2))
    return values


def close_to_known(value: float, known: set[float]) -> bool:
    if value in known:
        return True
    return any(abs(value - k) <= TOLERANCE for k in known)


def check_section(path: Path, known: set[float]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    for line_no, line in enumerate(text.splitlines(), 1):
        # cross-references (Section 5.4, Fig. 3, Table 4a) are not measurements
        line_for_numbers = re.sub(r"(Section|Sec\.|Fig\.|Figure|Table|Eq\.)\s*\d+(\.\d+)?",
                                  "", line, flags=re.IGNORECASE)
        for value in numbers_in(line_for_numbers):
            # years, section numbers, page counts and percentages of the frame are
            # not result values; skip the obvious non-statistics
            if value >= 100 or value in {0.5, 1.0, 2.0, 5.0, 0.05}:
                continue
            if not close_to_known(value, known):
                problems.append(f"  MISMATCH  line {line_no}: {value} "
                                f"is in no result file\n      {line.strip()[:100]}")

        for pattern, why, instead in OVERCLAIMS:
            m = re.search(pattern, line, re.IGNORECASE)
            # "does not prove", "we make no such claim": the phrase is being denied,
            # which is the hedge we want rather than an overclaim
            negated = bool(m) and bool(re.search(r"(not|no|never|cannot|nor)[^.]{0,40}$",
                                                 line[:m.start()] if m else "", re.IGNORECASE))
            if m and not negated:
                problems.append(f"  OVERCLAIM line {line_no}: \"{m.group()}\" - {why}"
                                f"\n      consider: {instead}"
                                f"\n      {line.strip()[:100]}")

    hedge_sections = {"no evidence": {"05_results"},
                      "do not claim": {"06_discussion"},
                      "hypothesis": {"06_discussion"}}
    for hedge, why in REQUIRED_HEDGES:
        if path.stem in hedge_sections.get(hedge, set()) and hedge not in text.lower():
            problems.append(f"  WEAKENED  \"{hedge}\" no longer appears - {why}")

    return problems


def main() -> None:
    wanted = None
    if "--section" in sys.argv:
        wanted = sys.argv[sys.argv.index("--section") + 1]

    known = known_values()
    print(f"{len(known)} distinct values found across {len(list(RESULTS.glob('*.csv')))} "
          f"result files\n")

    total = 0
    for path in sorted(SECTIONS.glob("*.md")):
        if wanted and wanted not in path.stem:
            continue
        problems = check_section(path, known)
        total += len(problems)
        status = "clean" if not problems else f"{len(problems)} to review"
        print(f"{path.name}  ({len(path.read_text(encoding='utf-8').split())} words) - {status}")
        for p in problems:
            print(p)
        if problems:
            print()

    print(f"\n{total} item(s) to review.")
    print("MISMATCH is the one that matters: a number the result files do not contain.")
    print("OVERCLAIM and WEAKENED are prompts to re-read, not errors by themselves.")


if __name__ == "__main__":
    main()
