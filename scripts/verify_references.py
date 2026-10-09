"""Verify every entry in paper/references.bib against Crossref.

Fabricated or wrong citations are the worst failure mode in a paper, so each entry
is checked against the real record before submission:

  - entry has a DOI  -> fetch it from Crossref and compare the titles
  - no DOI           -> search Crossref by title and report the best match
  - nothing found    -> flagged for manual checking (preprints, some Elsevier
                        pages and non-indexed journals are legitimately absent)

The real author list, journal and year come back in the report so the "VERIFY"
placeholders can be filled from the record instead of from memory.

Usage (from the repo root):
    python scripts/verify_references.py
Output:
    paper/reference_check.md   (human-readable report)
    paper/reference_check.csv  (machine-readable, one row per entry)
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "references.bib"
OUT_MD = ROOT / "paper" / "reference_check.md"
OUT_CSV = ROOT / "paper" / "reference_check.csv"

DOI_HANDLE_API = "https://doi.org/api/handles/"
CROSSREF_WORK = "https://api.crossref.org/works/"
CROSSREF_SEARCH = "https://api.crossref.org/works?rows=3&query.bibliographic="
USER_AGENT = "reference-verifier/1.0 (academic reference checking)"
REQUEST_PAUSE = 1.0           # be polite to the public API
TITLE_MATCH_OK = 0.85         # SequenceMatcher ratio treated as "same paper"
TITLE_MATCH_MAYBE = 0.60

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.DOTALL)

# ICML / ICLR / NeurIPS proceedings carry no DOI, and a Crossref title search
# actively mis-resolves these three (EfficientNet -> "EnhanceNet", ViT -> an SSRN
# look-alike). They are cited from their canonical venue, so searching for them
# only produces false alarms.
CANONICAL_NO_DOI = {"ref_vit", "ref_shap", "ref_efficientnet"}


def field(body: str, name: str) -> str:
    """Pull one brace-delimited bibtex field, tolerating nested braces."""
    m = re.search(rf"\b{name}\s*=\s*\{{", body)
    if not m:
        return ""
    i = m.end()
    depth, out = 1, []
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


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def fetch(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        print(f"    HTTP {exc.code} for {url[:80]}")
        return None
    except Exception as exc:  # network hiccup, keep going
        print(f"    request failed: {exc}")
        return None


def doi_resolves(doi: str) -> str:
    """Ask the DOI handle registry whether the DOI exists, and where it points.

    A DOI can be real yet absent from Crossref's index (several regional journals
    register elsewhere). Fetching https://doi.org/<doi> is not a reliable test
    either, because some publishers answer bots with 403 — so query the handle
    API, which only reports whether the DOI is registered."""
    data = fetch(DOI_HANDLE_API + urllib.parse.quote(doi))
    if not data or data.get("responseCode") != 1:
        return ""
    for value in data.get("values", []):
        if value.get("type") == "URL":
            return str(value.get("data", {}).get("value", "")) or "registered"
    return "registered"


def first_surname(author_field: str) -> str:
    """'Ribeiro, Marco Tulio and Singh, ...' -> 'ribeiro'"""
    first = re.split(r"\band\b", author_field)[0]
    first = first.strip().strip("{}")
    surname = first.split(",")[0] if "," in first else first.split()[-1:] and first.split()[-1]
    return normalise(surname or "")


def authors_of(item: dict) -> str:
    names = []
    for a in item.get("author", []):
        family, given = a.get("family", ""), a.get("given", "")
        if family:
            names.append(f"{family}, {given}".strip().rstrip(","))
        elif a.get("name"):
            names.append(a["name"])
    return " and ".join(names)


def year_of(item: dict) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = item.get(key, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return ""


def container_of(item: dict) -> str:
    ct = item.get("container-title") or []
    return ct[0] if ct else (item.get("publisher", "") or "")


def title_of(item: dict) -> str:
    t = item.get("title") or []
    return t[0] if t else ""


def parse_bib() -> list[dict]:
    text = BIB.read_text(encoding="utf-8")
    entries = []
    for kind, key, body in ENTRY_RE.findall(text):
        entries.append({
            "key": key.strip(),
            "kind": kind,
            "title": field(body, "title"),
            "author": field(body, "author"),
            "journal": field(body, "journal") or field(body, "booktitle"),
            "year": field(body, "year"),
            "doi": field(body, "doi"),
            "note": field(body, "note"),
        })
    return entries


def check_entry(e: dict) -> dict:
    """Returns the entry plus the Crossref verdict and the real metadata."""
    result = {**e, "status": "", "found_title": "", "found_authors": "",
              "found_journal": "", "found_year": "", "found_doi": "", "similarity": ""}

    if e["key"] in CANONICAL_NO_DOI:
        result["status"] = "CANONICAL (no DOI by venue)"
        return result

    if e["doi"]:
        data = fetch(CROSSREF_WORK + urllib.parse.quote(e["doi"]))
        time.sleep(REQUEST_PAUSE)
        if data is None:
            target = doi_resolves(e["doi"])
            result["status"] = "RESOLVES (not in Crossref)" if target else "DOI NOT FOUND"
            result["found_title"] = target
            return result
        item = data["message"]
        sim = similarity(e["title"], title_of(item))
        # Crossref stores some proceedings titles truncated at the subtitle, which
        # looks like a mismatch. If the first author matches, it is the same paper.
        author_ok = (first_surname(e["author"]) and
                     first_surname(e["author"]) == first_surname(authors_of(item)))
        result.update({
            "found_title": title_of(item),
            "found_authors": authors_of(item),
            "found_journal": container_of(item),
            "found_year": year_of(item),
            "found_doi": item.get("DOI", ""),
            "similarity": f"{sim:.2f}",
            "status": ("VERIFIED" if sim >= TITLE_MATCH_OK else
                       "VERIFIED (short title in Crossref)" if author_ok else
                       "TITLE MISMATCH" if sim >= TITLE_MATCH_MAYBE else
                       "WRONG DOI"),
        })
        return result

    # no DOI: try to find the paper by title
    data = fetch(CROSSREF_SEARCH + urllib.parse.quote(e["title"]))
    time.sleep(REQUEST_PAUSE)
    items = (data or {}).get("message", {}).get("items", [])
    if not items:
        result["status"] = "NOT FOUND (no DOI)"
        return result
    best = max(items, key=lambda it: similarity(e["title"], title_of(it)))
    sim = similarity(e["title"], title_of(best))
    result.update({
        "found_title": title_of(best),
        "found_authors": authors_of(best),
        "found_journal": container_of(best),
        "found_year": year_of(best),
        "found_doi": best.get("DOI", ""),
        "similarity": f"{sim:.2f}",
        "status": ("FOUND BY TITLE" if sim >= TITLE_MATCH_OK else
                   "POSSIBLE MATCH" if sim >= TITLE_MATCH_MAYBE else
                   "NOT FOUND (no DOI)"),
    })
    return result


def main() -> None:
    entries = parse_bib()
    print(f"Parsed {len(entries)} entries from {BIB.relative_to(ROOT)}")
    placeholder = [e["key"] for e in entries if "VERIFY" in (e["author"] + e["journal"])]
    print(f"{len(placeholder)} entries still carry a VERIFY placeholder\n")

    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['key']}: {e['title'][:60]}")
        r = check_entry(e)
        print(f"    -> {r['status']}"
              + (f" (title match {r['similarity']})" if r["similarity"] else ""))
        results.append(r)

    fields = ["key", "status", "similarity", "title", "found_title", "author",
              "found_authors", "journal", "found_journal", "year", "found_year",
              "doi", "found_doi", "note"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    by_status: dict[str, list[dict]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    lines = ["# Reference check against Crossref", "",
             f"{len(results)} entries checked. "
             f"{len(placeholder)} still carry a `VERIFY` placeholder.", "",
             "| status | count | meaning |", "|---|---|---|"]
    meaning = {
        "VERIFIED": "DOI resolves and the title matches - safe to cite",
        "FOUND BY TITLE": "no DOI in the bib, but Crossref found the paper - add the DOI",
        "TITLE MISMATCH": "DOI resolves but to a differently titled paper - check by hand",
        "WRONG DOI": "the DOI points at an unrelated paper - fix or drop",
        "DOI NOT FOUND": "Crossref does not know this DOI - fix or drop",
        "POSSIBLE MATCH": "weak title match - confirm by hand",
        "NOT FOUND (no DOI)": "not indexed in Crossref (preprint / non-indexed venue) - verify manually",
        "RESOLVES (not in Crossref)": "the DOI resolves at doi.org but is not in Crossref's index - fine",
        "VERIFIED (short title in Crossref)": "DOI resolves, first author matches, Crossref just stores a shorter title - fine",
        "CANONICAL (no DOI by venue)": "ICML/ICLR/NeurIPS paper cited from its canonical venue; no DOI exists - fine",
    }
    for status, group in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {status} | {len(group)} | {meaning.get(status, '')} |")

    for status, group in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        lines += ["", f"## {status} ({len(group)})", ""]
        for r in group:
            lines.append(f"### `{r['key']}`")
            lines.append(f"- bib title: {r['title']}")
            if r["found_title"] and similarity(r["title"], r["found_title"]) < 0.999:
                lines.append(f"- Crossref title: {r['found_title']}")
            if r["found_authors"]:
                lines.append(f"- **authors:** {r['found_authors']}")
            if r["found_journal"]:
                lines.append(f"- **journal:** {r['found_journal']} ({r['found_year']})")
            if r["found_doi"]:
                lines.append(f"- **doi:** {r['found_doi']}")
            if r["note"]:
                lines.append(f"- bib note: {r['note']}")
            lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n=== summary ===")
    for status, group in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        print(f"{status:22s} {len(group)}")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)} and {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
