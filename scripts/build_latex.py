"""Generate an Elsevier (elsarticle) LaTeX manuscript from paper/sections/*.md.

The prose is still being revised, so the conversion is a script rather than a
one-off hand translation: run it again after any edit and the .tex follows. The
markdown stays the editing surface; LaTeX is only the output format.

What it handles: headings, emphasis, inline code, [ref_key] citations, pipe
tables (with booktabs rules), the figure set with captions and labels, "Fig. N"
and "Section N.N" cross-references turned into \\ref, and LaTeX-special characters.

Usage (from the repo root):
    python scripts/build_latex.py
Output:
    paper/latex/main.tex  plus copies of the figures it includes
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "paper" / "sections"
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "paper" / "latex"

TITLE = ("Comparative Analysis of CNN and Vision Transformer Architectures for "
         "Explainable Endometriosis Classification in Laparoscopic Images")

# Fill these in before submitting; they are deliberately obvious placeholders.
AUTHOR = r"\author{FIRST LAST\corref{cor1}}"
AFFILIATION = r"\affiliation{organization={DEPARTMENT, INSTITUTION}, city={CITY}, country={COUNTRY}}"
CORRESPONDING = r"\cortext[cor1]{Corresponding author: EMAIL}"

# Figures in the order the manuscript cites them. Fig. N in the text maps to the
# Nth entry here, so this list and the text must stay in step.
FIGURES = [
    ("fig_dataset_folds.png", "fig:folds",
     "Fold composition. Each bar is one cross-validation fold, split into "
     "endometriosis and no-pathology frames, with the pathology share and the "
     "number of surgeries above it. Because whole surgeries are kept together, "
     "prevalence varies from 37.8\\% to 66.4\\%, which is why ROC-AUC rather than "
     "accuracy is compared across folds."),
    ("fig_confusion_pooled.png", "fig:confusion",
     "Pooled out-of-fold confusion matrices over all 746 frames, each predicted by "
     "the model that never saw its surgery. Cell shading is the row-normalised "
     "share; counts are given inside each cell."),
    ("fig_cv_auc.png", "fig:cvauc",
     "Classification performance per architecture. Small dots are the five "
     "fold-level AUCs, the open diamond is the earlier single-split estimate, and "
     "the filled marker with its bar is the pooled out-of-fold AUC with a 95\\% "
     "surgery-level bootstrap interval."),
    ("fig_roc_pooled.png", "fig:roc",
     "Pooled out-of-fold ROC curves. Each curve aggregates the predictions of the "
     "five fold models, so every frame contributes once, scored by a model blind "
     "to its surgery. Intervals in the legend resample surgeries."),
    ("fig_localization.png", "fig:localization",
     "Agreement between each saliency map and the expert lesion mask, over all 373 "
     "out-of-fold pathology frames. The vertical tick on each bar is that method's "
     "own area-matched random floor; the dashed line is a centred Gaussian "
     "baseline and the dotted line the chance level for the Pointing Game."),
    ("fig_overlays.png", "fig:overlays",
     "Qualitative comparison on the best, median and worst frames of "
     "EfficientNet-B0's IoU distribution. Columns: frame with the expert mask, "
     "Grad-CAM for each architecture, and attention rollout for ViT-B/16. Maps "
     "shown here come from the preliminary single-split models and are "
     "illustrative; all reported numbers use the fold models."),
    ("fig_error_clusters.png", "fig:errors",
     "Per-surgery error rate for the twelve hardest surgeries among those "
     "contributing at least five frames. Errors concentrate in particular "
     "recordings rather than spreading across the dataset."),
]

# "Fig. N" as the markdown writes it, mapped to a label. Kept explicit rather
# than read off the position in FIGURES, so that moving an entry in that list
# cannot silently repoint every cross-reference.
FIGURE_NUMBER = {"1": "fig:folds", "2": "fig:confusion", "3": "fig:cvauc",
                 "4": "fig:roc", "5": "fig:localization", "6": "fig:overlays",
                 "7": "fig:errors"}

SECTION_LABELS = {
    "02_introduction": "sec:intro",
    "03_related_work": "sec:related",
    "04_methodology": "sec:method",
    "05_results": "sec:results",
    "06_discussion": "sec:discussion",
    "07_limitations": "sec:limitations",
    "08_conclusion": "sec:conclusion",
}

# "Section 4.3" and the like, written before the subsections had labels
SUBSECTION_REFS = {
    "4.1": "sec:classification", "4.2": "sec:differ", "4.3": "sec:protocol",
    "4.4": "sec:explanation", "4.5": "sec:errors",
}
SUBSECTION_LABELS = {
    "Classification performance": "sec:classification",
    "Do the architectures differ?": "sec:differ",
    "Sensitivity to the protocol": "sec:protocol",
    "Explanation quality": "sec:explanation",
    "Where the errors concentrate": "sec:errors",
}

# "Section 3" in the text, in the order the sections appear
SECTION_ORDER = ["sec:intro", "sec:related", "sec:method", "sec:results",
                 "sec:discussion", "sec:limitations", "sec:conclusion"]

# Filled by scan_tables() before conversion: the number the markdown gave a table
# ("1", "4a") mapped to its LaTeX label. LaTeX renumbers 4a/4b as 4 and 5, which is
# the point of referencing them rather than writing the number by hand.
TABLE_LABELS: dict[str, str] = {}

TABLE_CAPTION = re.compile(r"^\*\*Table\s+(\d+[ab]?)\.\*\*\s*(.*)$", re.DOTALL)


def scan_tables(md: str) -> None:
    for number in re.findall(r"^\*\*Table\s+(\d+[ab]?)\.\*\*", md, re.M):
        TABLE_LABELS[number] = "tab:" + number


SUPERSCRIPT = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
               "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}

# pdfLaTeX stops on a character it has no definition for, so every non-ASCII
# character the prose can contain needs an entry here. Revision tools are the
# usual source of new ones: they replace straight quotes with curly ones and
# hyphens with dashes, so those are covered even though the draft has none.
UNICODE = {
    "–": "--", "—": "---", "−": r"$-$", "‑": "-", "…": r"\ldots{}",
    "×": r"$\times$", "±": r"$\pm$", "≤": r"$\leq$", "≥": r"$\geq$",
    "≈": r"$\approx$", "≠": r"$\neq$", "·": r"$\cdot$",
    "σ": r"$\sigma$", "μ": r"$\mu$", "α": r"$\alpha$", "β": r"$\beta$",
    "Δ": r"$\Delta$", "°": r"$^{\circ}$",
    "‘": "`", "’": "'", "“": "``", "”": "''",
    " ": " ", " ": r"\,", "​": "",
    "™": r"\texttrademark{}", "©": r"\textcopyright{}", "€": r"\texteuro{}",
}


def escape(text: str) -> str:
    """LaTeX-special characters, leaving our own markup alone."""
    text = text.replace("\\", r"\textbackslash{}")
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, "\\" + ch)
    text = text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    # scientific notation before the character table, so 10^-5 stays one expression
    text = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
                  lambda m: "$10^{" + "".join(SUPERSCRIPT[c] for c in m.group(1)) + "}$",
                  text)
    for ch, replacement in UNICODE.items():
        text = text.replace(ch, replacement)
    return text


CITATION_RUN = re.compile(r"\[(ref_[a-z0-9_]+)\](?:,?\s*\[(ref_[a-z0-9_]+)\])*")


def inline(text: str) -> str:
    """Markdown emphasis, code and citations.

    Citations are pulled out before escaping: escape() turns `ref_a1` into
    `ref\\_a1`, which no longer looks like a citation key.
    """
    stash: list[str] = []

    def park(match: re.Match) -> str:
        keys = re.findall(r"ref_[a-z0-9_]+", match.group(0))
        stash.append(r"\cite{" + ",".join(keys) + "}")
        return f"\x00CITE{len(stash) - 1}\x00"

    text = CITATION_RUN.sub(park, text)
    text = escape(text)
    text = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1) + "}", text)
    # Non-greedy rather than [^*]+, because bold runs contain emphasis: a cell
    # reading **+0.022 [...], *p* = 0.0048** matched neither pattern cleanly and
    # left stray asterisks in the table. The emphasis pass below then handles the
    # inner *p* inside the \textbf{} this produced.
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", text)
    text = re.sub(r"\*([^*]+)\*", lambda m: r"\emph{" + m.group(1) + "}", text)
    text = re.sub(r"\x00CITE(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    text = re.sub(r"Fig\.~?\s*(\d)",
                  lambda m: (r"Fig.~\ref{" + FIGURE_NUMBER[m.group(1)] + "}"
                             if m.group(1) in FIGURE_NUMBER else m.group(0)), text)
    text = re.sub(r"Section~?\s*(\d\.\d)",
                  lambda m: r"Section~\ref{" + SUBSECTION_REFS.get(m.group(1), "sec:results") + "}",
                  text)
    text = re.sub(r"Section~?\s*(\d)(?![\d.])",
                  lambda m: (r"Section~\ref{" + SECTION_ORDER[int(m.group(1)) - 1] + "}"
                             if 1 <= int(m.group(1)) <= len(SECTION_ORDER) else m.group(0)),
                  text)
    text = re.sub(r"Table~?\s*(\d+[ab]?)",
                  lambda m: (r"Table~\ref{" + TABLE_LABELS[m.group(1)] + "}"
                             if m.group(1) in TABLE_LABELS else m.group(0)), text)
    return text


def table(lines: list[str], number: str | None = None,
          caption: str | None = None) -> str:
    """A markdown pipe table becomes a booktabs tabular.

    The caption travels separately because in the markdown it is the paragraph
    above the table, which LaTeX wants inside the float instead.
    """
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    spec = "l" + "c" * (width - 1)

    out = [r"\begin{table}[t]", r"\centering"]
    if caption:
        out.append(r"\caption{" + inline(caption) + "}")
    if number:
        out.append(r"\label{" + TABLE_LABELS[number] + "}")
    # Several of these tables are wider than the text block, and a tabular that
    # overruns is silently clipped by the PDF viewer - whole columns disappeared
    # from Tables 1, 2, 3 and 5. The \ifdim guard shrinks only what is too wide,
    # so narrow tables are not blown up to fill the line.
    out += [r"\small", r"\setlength{\tabcolsep}{4pt}",
            r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
            r"\begin{tabular}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(inline(c) for c in rows[0]) + r" \\")
    out.append(r"\midrule")
    for row in rows[1:]:
        # a row whose only content is in the first cell is a sub-heading
        if row[0] and not any(row[1:]):
            out.append(r"\addlinespace")
            out.append(r"\multicolumn{" + str(width) + "}{l}{" + inline(row[0]) + r"} \\")
        else:
            out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(out)


def figure(entry: tuple[str, str, str]) -> str:
    filename, label, caption = entry
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=\linewidth]{figures/" + filename + "}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\end{figure}", ""])


def place_figures(body: str) -> tuple[str, int]:
    """Put each figure just after the paragraph that first refers to it.

    All seven used to be emitted together at the end of the Results section.
    That is more floats than LaTeX will hold, so it deferred four of them to the
    end of the document and they came out after the bibliography. Figures keep
    the order of FIGURES, so the numbers LaTeX assigns still match the ones the
    markdown writes, and none is placed before its first mention.
    """
    lines = body.split("\n")
    insertions: list[tuple[int, str]] = []
    previous = -1
    for entry in FIGURES:
        marker = "\\ref{" + entry[1] + "}"
        cited = next((i for i, line in enumerate(lines) if marker in line), None)
        if cited is None:
            print(f"  WARNING: {entry[1]} is never referred to in the text")
            cited = len(lines) - 1
        previous = max(cited, previous + 1)
        insertions.append((previous, figure(entry)))
    for index, block in reversed(insertions):
        lines.insert(index + 1, block)
    return "\n".join(lines), len(insertions)


LIST_MARKER = re.compile(r"^(\d+\.|[-*])\s+")


def convert(md: str) -> str:
    out: list[str] = []
    buffer: list[str] = []
    table_lines: list[str] = []
    item: list[str] = []
    pending_caption: list[tuple[str, str]] = []
    item_env = ""

    def flush_item() -> None:
        """A list item can run over several indented lines, and the bold often spans
        the break, so the lines are joined before any inline conversion."""
        nonlocal item_env
        if item:
            out.append("\\begin{" + item_env + "}\\item " + inline(" ".join(item)) +
                       "\\end{" + item_env + "}")
            item.clear()
            item_env = ""

    def flush_paragraph() -> None:
        if not buffer:
            return
        text = " ".join(buffer)
        buffer.clear()
        caption = TABLE_CAPTION.match(text)
        if caption:                                # belongs inside the float below
            pending_caption.append((caption.group(1), caption.group(2)))
            return
        out.append(inline(text))
        out.append("")

    def flush_table() -> None:
        if not table_lines:
            return
        number, caption = pending_caption.pop(0) if pending_caption else (None, None)
        out.append(table(table_lines, number, caption))
        table_lines.clear()

    for line in md.splitlines():
        stripped = line.strip()
        indented = line[:1].isspace()

        if stripped.startswith("|"):
            flush_item()
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()

        if item_env and stripped and indented and not LIST_MARKER.match(stripped):
            item.append(stripped)                  # continuation of the current item
            continue

        if not stripped:
            flush_item()
            flush_paragraph()
        elif stripped.startswith("### "):
            flush_item()
            flush_paragraph()
            heading = stripped[4:]
            out.append(r"\subsection{" + inline(heading) + "}")
            if heading in SUBSECTION_LABELS:
                out.append(r"\label{" + SUBSECTION_LABELS[heading] + "}")
            out.append("")
        elif stripped.startswith("## "):
            flush_item()
            flush_paragraph()                      # the section title is set by caller
        elif LIST_MARKER.match(stripped):
            flush_item()
            flush_paragraph()
            item_env = "enumerate" if stripped[0].isdigit() else "itemize"
            item.append(LIST_MARKER.sub("", stripped))
        elif stripped.startswith(">"):
            continue                               # the draft banner
        else:
            buffer.append(stripped)

    flush_item()
    flush_paragraph()
    flush_table()
    for number, caption in pending_caption:        # a caption with no table under it
        print(f"  WARNING: caption for Table {number} has no table under it")
        out.append(r"\textbf{Table " + number + ".} " + inline(caption))
        out.append("")
    return "\n".join(out)


def merge_lists(text: str) -> str:
    """Consecutive one-item lists become a single list."""
    for env in ("itemize", "enumerate"):
        text = text.replace("\\end{" + env + "}\n\\begin{" + env + "}", "")
    return text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    # Table numbers must be known before any section is converted, because the
    # Results prose references Table 2 several paragraphs before the table itself.
    for path in sorted(SECTIONS.glob("*.md")):
        scan_tables(path.read_text(encoding="utf-8"))

    abstract_md = (SECTIONS / "01_abstract.md").read_text(encoding="utf-8")
    body = re.sub(r"^##.*$", "", abstract_md, flags=re.M)
    keywords = ""
    if "**Keywords:**" in body:
        body, _, keywords = body.partition("**Keywords:**")
    abstract = convert(body).strip()
    keywords_tex = " \\sep ".join(k.strip() for k in keywords.replace("\n", " ").split(";") if k.strip())

    parts = [
        r"\documentclass[preprint,12pt]{elsarticle}",
        # T1 is not cosmetic: without it the comma-below accents in Bļizņuks set
        # as separate commas, and the bibliography carries several such names.
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{amsmath}",
        r"\usepackage[hidelinks]{hyperref}",
        # LaTeX's defaults refuse a page that is mostly float, and a figure it
        # refuses is pushed to the end of the document - which is how four of
        # them ended up after the bibliography. These let it place a float where
        # the text asks for one.
        r"\setcounter{topnumber}{3}", r"\setcounter{bottomnumber}{2}",
        r"\setcounter{totalnumber}{5}",
        r"\renewcommand{\topfraction}{0.9}", r"\renewcommand{\bottomfraction}{0.9}",
        r"\renewcommand{\textfraction}{0.05}", r"\renewcommand{\floatpagefraction}{0.7}",
        r"\journal{Biomedical Signal Processing and Control}",
        "", r"\begin{document}", r"\begin{frontmatter}",
        r"\title{" + escape(TITLE) + "}",
        AUTHOR, AFFILIATION, CORRESPONDING, "",
        r"\begin{abstract}", abstract, r"\end{abstract}", "",
        r"\begin{keyword}", keywords_tex, r"\end{keyword}",
        r"\end{frontmatter}", "",
    ]

    body: list[str] = []
    for path in sorted(SECTIONS.glob("*.md")):
        if path.stem == "01_abstract":
            continue
        md = path.read_text(encoding="utf-8")
        heading = re.search(r"^##\s+(.+)$", md, re.M)
        body.append(r"\section{" + inline(heading.group(1)) + "}" if heading else "")
        if path.stem in SECTION_LABELS:
            body.append(r"\label{" + SECTION_LABELS[path.stem] + "}")
        body.append("")
        body.append(merge_lists(convert(md)))

    placed, figures_placed = place_figures("\n".join(body))
    parts.append(placed)

    parts += [r"\bibliographystyle{elsarticle-num}", r"\bibliography{references}",
              r"\end{document}", ""]

    (OUT_DIR / "main.tex").write_text("\n".join(parts), encoding="utf-8")
    shutil.copy(ROOT / "paper" / "references.bib", OUT_DIR / "references.bib")
    for filename, _, _ in FIGURES:
        src = RESULTS / filename
        if src.exists():
            shutil.copy(src, OUT_DIR / "figures" / filename)

    tex = (OUT_DIR / "main.tex").read_text(encoding="utf-8")
    print(f"wrote {(OUT_DIR / 'main.tex').relative_to(ROOT)}")
    print(f"  {len(tex.splitlines())} lines, {figures_placed} figures, "
          f"{tex.count(chr(92) + 'cite{')} citation commands, "
          f"{tex.count(chr(92) + 'begin{tabular}')} tables")
    print(f"  figures copied to {(OUT_DIR / 'figures').relative_to(ROOT)}")

    # A character with no LaTeX definition stops the compile with an error that
    # names a line number and nothing else, so it is worth catching here instead.
    stray = sorted({ch for ch in tex if ord(ch) > 127})
    if stray:
        print("\n  WARNING: characters pdfLaTeX will not understand, "
              "add them to UNICODE in this script:")
        for ch in stray:
            line = next(n for n, l in enumerate(tex.splitlines(), 1) if ch in l)
            print(f"    U+{ord(ch):04X}  first on line {line}")
    print("\n  Fill in the author block in main.tex before submitting.")


if __name__ == "__main__":
    main()
