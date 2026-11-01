"""Carry the yellow amendment highlighting onto the author's latest clean revision.

SOURCE_HL  my v2 highlighted build (same text as v2 clean, plus highlights and the amendment log)
TARGET     the author's latest revision (v2 clean plus the author's own edits)
OUTPUT     TARGET with the highlights transferred character by character; text the author
           added relative to SOURCE_HL is highlighted as an amendment; the amendment log is
           appended with one row for the author's edits.

Nothing in TARGET's wording is changed. Paragraphs are aligned by text; within a paragraph the
highlighted character ranges are mapped through a character-level alignment, and runs are split
only where a highlight starts or ends.
"""
import copy
import difflib
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
SOURCE_HL, TARGET, OUTPUT = (Path(p) for p in sys.argv[1:4])


def paragraphs(body):
    return [p for p in body.iter(qn("w:p"))]


def text_nodes(p):
    return [t for t in p.iter(qn("w:t"))]


def ptext(p):
    return "".join(t.text or "" for t in text_nodes(p))


def is_highlighted(node):
    run = node.getparent()
    rpr = run.find(qn("w:rPr"))
    return rpr is not None and rpr.find(qn("w:highlight")) is not None


def highlight_mask(p):
    mask = []
    for t in text_nodes(p):
        mask.extend([is_highlighted(t)] * len(t.text or ""))
    return mask


def set_highlight(run, on=True):
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        rpr = OxmlElement("w:rPr")
        run.insert(0, rpr)
    for old in rpr.findall(qn("w:highlight")):
        rpr.remove(old)
    if on:
        hl = OxmlElement("w:highlight")
        hl.set(qn("w:val"), "yellow")
        rpr.append(hl)


def apply_mask(p, mask):
    """Split runs so that exactly the characters flagged in mask are highlighted."""
    pos = 0
    for t in text_nodes(p):
        text = t.text or ""
        flags = mask[pos:pos + len(text)]
        pos += len(text)
        if not text or not any(flags):
            continue
        run = t.getparent()
        if run.tag != qn("w:r") or len(run.findall(qn("w:t"))) != 1:
            raise ValueError(f"unexpected run structure around {text[:40]!r}")
        # group consecutive characters with the same flag
        pieces, start = [], 0
        for i in range(1, len(text) + 1):
            if i == len(text) or flags[i] != flags[start]:
                pieces.append((text[start:i], flags[start]))
                start = i
        anchor = run
        for k, (piece, flag) in enumerate(pieces):
            target = run if k == 0 else copy.deepcopy(run)
            target.find(qn("w:t")).text = piece
            target.find(qn("w:t")).set(qn("xml:space"), "preserve")
            set_highlight(target, flag)
            if k > 0:
                anchor.addnext(target)
                anchor = target


def math_highlights(src, dst):
    """Copy equation highlights. A fully highlighted equation is highlighted in full even if Word
    re-split its runs or normalised characters (e.g. U+2212 to '-') on save; a partly highlighted
    equation needs an exact run-for-run match."""
    s = [r for r in src.iter(f"{M}r")]
    d = [r for r in dst.iter(f"{M}r")]
    flags = [(r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:highlight")) is not None)
             for r in s]
    if s and all(flags):
        pairs = [(s[0], rd) for rd in d]
    elif len(s) != len(d) or [r.findtext(f"{M}t") for r in s] != [r.findtext(f"{M}t") for r in d]:
        if any(flags):
            raise ValueError(f"cannot map partial equation highlight: {''.join(r.findtext(M + 't') or '' for r in s)!r}")
        return 0
    else:
        pairs = list(zip(s, d))
    n = 0
    for rs, rd in pairs:
        wr = rs.find(qn("w:rPr"))
        if wr is not None and wr.find(qn("w:highlight")) is not None:
            rpr = rd.find(qn("w:rPr"))
            if rpr is None:
                rpr = OxmlElement("w:rPr")
                mrpr = rd.find(f"{M}rPr")
                (mrpr.addnext(rpr) if mrpr is not None else rd.insert(0, rpr))
            if rpr.find(qn("w:highlight")) is None:
                hl = OxmlElement("w:highlight"); hl.set(qn("w:val"), "yellow"); rpr.append(hl)
            n += 1
    return n


def main():
    src_doc, dst_doc = Document(SOURCE_HL), Document(TARGET)
    src_body, dst_body = src_doc.element.body, dst_doc.element.body

    # the highlighted build ends with the amendment log; set it aside for re-attachment
    log_heading = next(p for p in src_body.iterchildren(qn("w:p")) if ptext(p).startswith("Amendment log"))
    log_elements = []
    el = log_heading
    while el is not None and el.tag != qn("w:sectPr"):
        log_elements.append(el)
        el = el.getnext()
    src_paras = [p for p in paragraphs(src_body)
                 if not any(p is x or x in p.iterancestors() for x in log_elements)]
    log_set = set(id(x) for x in log_elements)
    src_paras = [p for p in paragraphs(src_body)
                 if not any(id(a) in log_set for a in [p, *p.iterancestors()])]
    dst_paras = paragraphs(dst_body)

    src_text, dst_text = [ptext(p) for p in src_paras], [ptext(p) for p in dst_paras]
    matcher = difflib.SequenceMatcher(None, src_text, dst_text, autojunk=False)
    stats = {"paragraphs_matched": 0, "highlighted_chars": 0, "math_runs": 0, "author_edits": []}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for s, d in zip(src_paras[i1:i2], dst_paras[j1:j2]):
                mask = highlight_mask(s)
                if any(mask):
                    apply_mask(d, mask)
                    stats["highlighted_chars"] += sum(mask)
                stats["math_runs"] += math_highlights(s, d)
                stats["paragraphs_matched"] += 1
        elif op == "replace" and (i2 - i1) == (j2 - j1):
            for s, d in zip(src_paras[i1:i2], dst_paras[j1:j2]):
                a, b = ptext(s), ptext(d)
                src_mask = highlight_mask(s)
                mask = [False] * len(b)
                # word-level alignment: whole words the author added are highlighted
                ta, tb = re.findall(r"\w+|\s+|[^\w\s]", a), re.findall(r"\w+|\s+|[^\w\s]", b)
                oa = [0]; [oa.append(oa[-1] + len(x)) for x in ta]
                ob = [0]; [ob.append(ob[-1] + len(x)) for x in tb]
                cm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
                for cop, a1, a2, b1, b2 in cm.get_opcodes():
                    if cop == "equal":
                        mask[ob[b1]:ob[b2]] = src_mask[oa[a1]:oa[a2]]
                    elif cop in ("replace", "insert"):
                        for k in range(b1, b2):
                            if tb[k].strip():
                                mask[ob[k]:ob[k + 1]] = [True] * len(tb[k])
                apply_mask(d, mask)
                stats["highlighted_chars"] += sum(mask)
                stats["author_edits"].append((a, b))
        else:
            for d in dst_paras[j1:j2]:
                if ptext(d).strip():
                    stats["author_edits"].append(("", ptext(d)))
            for s in src_paras[i1:i2]:
                if ptext(s).strip():
                    raise ValueError(f"text in the highlighted build is missing from the target: {ptext(s)[:80]!r}")

    # re-attach the amendment log, with a row recording the author's own edits
    sect = dst_body.find(qn("w:sectPr"))
    copies = [copy.deepcopy(x) for x in log_elements]
    for x in copies:
        sect.addprevious(x)
    log_table = next(x for x in copies if x.tag == qn("w:tbl"))
    template_row = log_table.findall(qn("w:tr"))[-1]
    for old, new in stats["author_edits"]:
        row = copy.deepcopy(template_row)
        cells = row.findall(qn("w:tc"))
        values = ["3.1", "Author edit: modelling-dataset sentence reworded (new wording highlighted).",
                  "Author revision saved 25 October 2026."]
        for cell, value in zip(cells, values):
            ts = list(cell.iter(qn("w:t")))
            ts[0].text = value
            for extra in ts[1:]:
                extra.text = ""
        log_table.append(row)

    dst_doc.save(OUTPUT)
    stats["author_edits"] = [(o[:90], n[:90]) for o, n in stats["author_edits"]]
    print(stats)


if __name__ == "__main__":
    main()
