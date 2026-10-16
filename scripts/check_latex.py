"""Check paper/latex/main.tex before it goes to Overleaf.

There is no LaTeX installed here, so the first compile happens on someone else's
machine. These are the failures that would waste that round trip: a citation key
the bibliography does not define, markdown that survived the conversion, a \\ref
with no \\label, a missing figure file, an unclosed environment, and - the one
that produces the most confusing error message - a character pdfLaTeX has no
definition for.

Run it after every `python scripts/build_latex.py`, and especially after the
prose has been revised, since revision tools introduce curly quotes and dashes.

Usage (from the repo root):
    python scripts/check_latex.py
Exit status is 1 if anything needs attention, so it can gate a commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "latex" / "main.tex"
BIB = ROOT / "paper" / "latex" / "references.bib"

ENVIRONMENTS = ("document", "frontmatter", "abstract", "keyword", "table",
                "figure", "tabular", "itemize", "enumerate", "equation")

COUNTS = [
    (r"\\section\{", "sections"),
    (r"\\subsection\{", "subsections"),
    (r"\\begin\{table\}", "tables"),
    (r"\\begin\{figure\}", "figures"),
    (r"\\cite\{", "citation commands"),
    (r"\\ref\{fig:", "figure references"),
    (r"\\ref\{tab:", "table references"),
    (r"\\ref\{sec:", "section references"),
]

DOLLAR = re.compile(r"(?<!\\)\$")


def main() -> int:
    if not TEX.exists():
        print(f"{TEX.relative_to(ROOT)} does not exist - run scripts/build_latex.py")
        return 1
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    problems: list[str] = []

    cited: set[str] = set()
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(","))
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    print(f"{len(cited)} keys cited, {len(defined)} defined in references.bib")
    for key in sorted(cited - defined):
        problems.append(f"  cited but not in references.bib: {key}")
    for key in sorted(defined - cited):
        problems.append(f"  in references.bib but never cited: {key}")

    print("\nstructure:")
    for pattern, label in COUNTS:
        print(f"  {label:20s} {len(re.findall(pattern, tex))}")

    leftovers = [
        (len(re.findall(r"\[ref_", tex)), "[ref_...] citation brackets"),
        (tex.count("**"), "markdown bold markers"),
        (len([l for l in tex.splitlines() if l.strip().startswith("|")]), "pipe-table rows"),
        (len([l for l in tex.splitlines() if l.startswith("#")]), "markdown headings"),
    ]
    for count, what in leftovers:
        if count:
            problems.append(f"  {count} {what} survived the conversion")

    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    for ref in sorted(refs - labels):
        problems.append(f"  \\ref{{{ref}}} has no \\label")

    for graphic in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        if not (TEX.parent / graphic).exists():
            problems.append(f"  figure file missing: {graphic}")

    delta = tex.count("{") - tex.count("}")
    if delta:
        problems.append(f"  braces are unbalanced by {delta:+d}")
    for env in ENVIRONMENTS:
        opened = len(re.findall(r"\\begin\{" + env + r"\}", tex))
        closed = len(re.findall(r"\\end\{" + env + r"\}", tex))
        if opened != closed:
            problems.append(f"  {env}: {opened} begin against {closed} end")
    for line_no, line in enumerate(tex.splitlines(), 1):
        if len(DOLLAR.findall(line)) % 2:
            problems.append(f"  line {line_no}: odd number of $, math mode left open")

    for ch in sorted({c for c in tex if ord(c) > 127}):
        line_no = next(n for n, l in enumerate(tex.splitlines(), 1) if ch in l)
        problems.append(f"  U+{ord(ch):04X} on line {line_no} has no LaTeX definition "
                        f"- add it to UNICODE in scripts/build_latex.py")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        print("\n".join(problems))
        return 1
    print("\nNothing to fix. The author block still needs filling in by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
