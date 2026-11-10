"""Tracked-change editing of a WordprocessingML package (lxml), for build_supervisor_corrections.py.

Every edit is written as a Word revision (w:ins / w:del / w:pPrChange / row marks) under one author, so the
document opens with Track Changes showing what was changed and each change can be accepted or rejected.
Text is matched on what Word draws: the w:t text of a paragraph's runs, citation fields included.
"""
import copy
import itertools
import re

from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
OBJ = '￼'          # stands for an equation inside paragraph text, so a match cannot silently cross one
SPECIAL = {'tab': '\t', 'br': '\n', 'noBreakHyphen': '‑', 'softHyphen': ''}   # run items shown as one character


def w(tag):
    return '{%s}%s' % (W, tag)


def m(tag):
    return '{%s}%s' % (M, tag)


def local(el):
    return etree.QName(el).localname


class Cite:
    """A citation field: Source Manager tags and the text Word shows for them."""

    def __init__(self, tags, shown):
        self.tags, self.shown = list(tags), shown


class Italic(str):
    pass


class Bold(str):
    pass


class Seq:
    """A SEQ caption field (e.g. Seq('Figure', 15))."""

    def __init__(self, name, shown):
        self.name, self.shown = name, str(shown)


class Editor:
    def __init__(self, author, date, first_id=900000):
        self.author, self.date = author, date
        self.ids = itertools.count(first_id)
        self.sdt_ids = itertools.count(1700000001)

    # ---------- revision marks ----------
    def mark(self, kind):
        el = etree.Element(w(kind))
        el.set(w('id'), str(next(self.ids)))
        el.set(w('author'), self.author)
        el.set(w('date'), self.date)
        return el

    # ---------- paragraph text ----------
    @staticmethod
    def is_text_run(r):
        return all(local(c) in ('rPr', 't', 'lastRenderedPageBreak') for c in r)

    @staticmethod
    def run_text(el):
        return ''.join(t.text or '' for t in el.iter(w('t')))

    def _explode(self, p):
        """Split runs that mix text with tabs or breaks into one run per item, so text stays editable."""
        for r in [c for c in p if local(c) == 'r']:
            items = [c for c in r if local(c) not in ('rPr',)]
            kinds = {local(c) for c in items}
            if len(items) > 1 and kinds <= {'t', 'lastRenderedPageBreak'} | set(SPECIAL) and kinds & set(SPECIAL):
                rpr = r.find(w('rPr'))
                for c in items:
                    if local(c) == 'lastRenderedPageBreak':
                        continue
                    nr = etree.Element(w('r'))
                    if rpr is not None:
                        nr.append(copy.deepcopy(rpr))
                    nr.append(copy.deepcopy(c))
                    r.addprevious(nr)
                p.remove(r)

    def tokens(self, p):
        self._explode(p)
        toks = []
        for ch in p:
            content = [c for c in ch if local(c) != 'rPr'] if local(ch) == 'r' else []
            if len(content) == 1 and local(content[0]) in SPECIAL:
                toks.append([ch, SPECIAL[local(content[0])], 'atom'])
                continue
            tag = local(ch)
            if tag == 'r':
                toks.append([ch, self.run_text(ch), 'run' if self.is_text_run(ch) else 'atom'])
            elif tag in ('sdt', 'hyperlink', 'smartTag', 'fldSimple'):
                toks.append([ch, self.run_text(ch), 'atom'])
            elif ch.tag in (m('oMath'), m('oMathPara')):
                toks.append([ch, OBJ, 'atom'])
            elif tag == 'ins':                       # earlier insertion: visible, but not split again
                toks.append([ch, self.run_text(ch), 'atom'])
        return toks                                  # w:del content is not visible text

    def text(self, p):
        return ''.join(t[1] for t in self.tokens(p))

    # ---------- run surgery ----------
    @staticmethod
    def _set_text(r, s):
        ts = r.findall(w('t'))
        for t in ts[1:]:
            r.remove(t)
        if not ts:
            t = etree.SubElement(r, w('t'))
        else:
            t = ts[0]
        t.text = s
        t.set(XML_SPACE, 'preserve')

    def _split(self, r, at):
        """Split text run r at character offset; returns the right-hand run (inserted after r)."""
        s = self.run_text(r)
        right = copy.deepcopy(r)
        for lr in right.findall(w('lastRenderedPageBreak')):
            right.remove(lr)
        self._set_text(r, s[:at])
        self._set_text(right, s[at:])
        r.addnext(right)
        return right

    def _isolate(self, p, start, end):
        """Split runs so that [start, end) covers whole tokens; return the covered tokens."""
        pos = 0
        for tok in self.tokens(p):
            el, s, kind = tok
            a, b = pos, pos + len(s)
            if kind == 'run':
                if a < start < b:
                    self._split(el, start - a)
                    return self._isolate(p, start, end)
                if a < end < b:
                    self._split(el, end - a)
                    return self._isolate(p, start, end)
            elif s and (a < start < b or a < end < b):
                raise ValueError(f'match cuts through a field or object: {s!r}')
            pos = b
        pos, covered = 0, []
        for tok in self.tokens(p):
            a, b = pos, pos + len(tok[1])
            if a >= start and b <= end and b > a:
                covered.append(tok)
            pos = b
        return covered

    def _delete_el(self, el):
        """Wrap a run / field / hyperlink / equation in a tracked deletion."""
        tag = local(el)
        if tag == 'r':
            d = self.mark('del')
            el.addprevious(d)
            d.append(el)
            for t in el.iter(w('t')):
                t.tag = w('delText')
            for t in el.iter(w('instrText')):
                t.tag = w('delInstrText')
        elif tag in ('sdt', 'hyperlink', 'smartTag'):
            holder = el.find(w('sdtContent')) if tag == 'sdt' else el
            for c in list(holder):
                if local(c) == 'r':
                    self._delete_el(c)
                elif local(c) == 'ins' and c.get(w('author')) == self.author:
                    holder.remove(c)            # deleting one's own insertion removes it
        elif el.tag in (m('oMathPara'), m('oMath')):
            for om in ([el] if el.tag == m('oMath') else el.findall(m('oMath'))):
                d = self.mark('del')
                for c in list(om):
                    if c.tag == m('oMathParaPr'):
                        continue
                    d.append(c)
                om.append(d)
                for r in d.iter(m('r')):
                    rp = r.find(w('rPr'))
                    if rp is None:
                        rp = etree.Element(w('rPr'))
                        r.insert(0 if r.find(m('rPr')) is None else 1, rp)
        else:
            raise ValueError(f'cannot delete <{tag}>')

    # ---------- building new content ----------
    @staticmethod
    def clean_rpr(rpr):
        if rpr is None:
            return None
        rpr = copy.deepcopy(rpr)
        for tag in ('ins', 'del', 'rPrChange', 'highlight'):
            for x in rpr.findall(w(tag)):
                rpr.remove(x)
        return rpr

    def make_run(self, s, rpr, italic=False, bold=False):
        r = etree.Element(w('r'))
        rp = self.clean_rpr(rpr)
        if italic or bold:
            rp = rp if rp is not None else etree.Element(w('rPr'))
            for flag, tags in ((bold, ('b', 'bCs')), (italic, ('i', 'iCs'))):
                if flag:
                    for t in tags:
                        if rp.find(w(t)) is None:
                            # b, bCs, i, iCs come after rStyle/rFonts in CT_RPr
                            anchor = None
                            for name in ('rStyle', 'rFonts'):
                                if rp.find(w(name)) is not None:
                                    anchor = rp.find(w(name))
                            el = etree.Element(w(t))
                            if anchor is None:
                                rp.insert(0, el)
                            else:
                                anchor.addnext(el)
        if rp is not None and len(rp):
            r.append(rp)
        t = etree.SubElement(r, w('t'))
        t.text = s
        t.set(XML_SPACE, 'preserve')
        return r

    def _field_runs(self, instr, shown, rpr):
        runs = []
        for kind in ('begin', 'instr', 'separate', 'shown', 'end'):
            r = etree.Element(w('r'))
            rp = self.clean_rpr(rpr)
            if rp is not None:
                r.append(rp)
            if kind in ('begin', 'separate', 'end'):
                etree.SubElement(r, w('fldChar')).set(w('fldCharType'), kind)
            elif kind == 'instr':
                t = etree.SubElement(r, w('instrText'))
                t.text = instr
                t.set(XML_SPACE, 'preserve')
            else:
                t = etree.SubElement(r, w('t'))
                t.text = shown
            runs.append(r)
        return runs

    def citation(self, cite, rpr, tracked=True):
        instr = ' CITATION ' + cite.tags[0] + ' \\l 7177 \\t' + ''.join(' \\m ' + t + ' \\t' for t in cite.tags[1:]) + ' '
        sdt = etree.Element(w('sdt'))
        pr = etree.SubElement(sdt, w('sdtPr'))
        if rpr is not None:
            pr.append(self.clean_rpr(rpr))
        etree.SubElement(pr, w('id')).set(w('val'), str(next(self.sdt_ids) % 2000000000))
        etree.SubElement(pr, w('citation'))
        content = etree.SubElement(sdt, w('sdtContent'))
        holder = self.mark('ins') if tracked else content
        if tracked:
            content.append(holder)
        for r in self._field_runs(instr, cite.shown, rpr):
            holder.append(r)
        return sdt

    def build(self, segments, rpr, tracked=True):
        """Elements for a list of segments (str, Italic, Bold, Cite, Seq), each wrapped as an insertion."""
        if isinstance(segments, (str, Cite, Seq)):
            segments = [segments]
        out, pending = [], []

        def flush():
            if pending:
                if tracked:
                    ins = self.mark('ins')
                    for r in pending:
                        ins.append(r)
                    out.append(ins)
                else:
                    out.extend(pending)
                pending.clear()

        for seg in segments:
            if isinstance(seg, Cite):
                flush()
                out.append(self.citation(seg, rpr, tracked))
            elif isinstance(seg, Seq):
                pending.extend(self._field_runs(f' SEQ {seg.name} \\* ARABIC ', seg.shown, rpr))
            elif seg:
                pending.append(self.make_run(seg, rpr, italic=isinstance(seg, Italic), bold=isinstance(seg, Bold)))
        flush()
        return out

    @staticmethod
    def template_rpr(p, near=None):
        """rPr of the text run nearest `near` (or the first text run of p)."""
        runs = [r for r in p.iter(w('r')) if r.find(w('t')) is not None and r.find(w('fldChar')) is None]
        if near is not None:
            for r in runs:
                if r is near or near in r.iterancestors() or r in near.iter():
                    return r.find(w('rPr'))
        return runs[0].find(w('rPr')) if runs else None

    def replace_field_result(self, sdt, new_text):
        """Track a change to the text a citation field shows (its result runs)."""
        content = sdt.find(w('sdtContent'))
        runs = [r for r in content if local(r) == 'r' and r.find(w('t')) is not None]
        rpr = runs[0].find(w('rPr'))
        anchor = etree.Element(w('r'))
        runs[-1].addnext(anchor)
        for r in runs:
            self._delete_el(r)
        for el in self.build(new_text, copy.deepcopy(rpr) if rpr is not None else None):
            anchor.addprevious(el)
        content.remove(anchor)

    # ---------- public edits ----------
    def replace(self, p, old, new, count=1, regex=False):
        full = self.text(p)
        found = list(re.finditer(old if regex else re.escape(old), full))
        if len(found) != count:
            raise AssertionError(f'{len(found)} matches (expected {count}) of {old!r} in: {full[:200]!r}')
        for mt in reversed(found):
            self._replace_span(p, mt.start(), mt.end(), new)

    def _replace_span(self, p, start, end, new):
        covered = self._isolate(p, start, end)
        if not covered:
            raise AssertionError('empty match')
        rpr = None
        for el, s, kind in covered:
            if kind == 'run':
                rpr = el.find(w('rPr'))
                break
        if rpr is None:
            rpr = self.template_rpr(covered[0][0], covered[0][0])
        if rpr is None:
            rpr = self.template_rpr(p)
        rpr = copy.deepcopy(rpr) if rpr is not None else None
        last = covered[-1][0]
        anchor = etree.Element(w('r'))      # placeholder for the insertion point
        last.addnext(anchor)
        for el, s, kind in covered:
            self._delete_el(el)
        for el in self.build(new, rpr):
            anchor.addprevious(el)
        anchor.getparent().remove(anchor)

    def insert(self, p, anchor_text, new, before=False, count=1):
        """Insert new content right after (or before) anchor_text, without touching the anchor."""
        full = self.text(p)
        found = [mt.start() for mt in re.finditer(re.escape(anchor_text), full)]
        if len(found) != count:
            raise AssertionError(f'{len(found)} matches of anchor {anchor_text!r} in: {full[:200]!r}')
        at = found[0] if before else found[0] + len(anchor_text)
        self._insert_at(p, at, new)

    def append(self, p, new):
        self._insert_at(p, len(self.text(p)), new)

    def _insert_at(self, p, at, new):
        pos, ref, side = 0, None, 'after'
        toks = self.tokens(p)
        for tok in toks:
            el, s, kind = tok
            a, b = pos, pos + len(s)
            if kind == 'run' and a < at < b:
                self._split(el, at - a)
                return self._insert_at(p, at, new)
            if b == at and s:
                ref, side = el, 'after'
            if a == at and s and ref is None:
                ref, side = el, 'before'
            pos = b
        if ref is None:
            if at != 0 and toks:
                raise AssertionError('insertion point not found')
            ppr = p.find(w('pPr'))
            ref, side = (ppr, 'after') if ppr is not None else (None, 'start')
        rpr = None
        if ref is not None and local(ref) == 'r':
            rpr = ref.find(w('rPr'))
        if rpr is None:
            rpr = self.template_rpr(p)
        els = self.build(new, copy.deepcopy(rpr) if rpr is not None else None)
        if side == 'start':
            for el in reversed(els):
                p.insert(0, el)
        elif side == 'after':
            for el in reversed(els):
                ref.addnext(el)
        else:
            for el in els:
                ref.addprevious(el)

    def delete_paragraph(self, p):
        for ch in list(p):
            tag = local(ch)
            if tag in ('r', 'sdt', 'hyperlink', 'smartTag') or ch.tag in (m('oMathPara'), m('oMath')):
                if tag == 'r' and not self.run_text(ch) and ch.find(w('drawing')) is None \
                        and ch.find(w('fldChar')) is None and ch.find(w('instrText')) is None \
                        and ch.find(w('footnoteReference')) is None and ch.find(w('tab')) is None \
                        and ch.find(w('br')) is None:
                    continue
                self._delete_el(ch)
        self._mark_para(p, 'del')

    def _mark_para(self, p, kind):
        ppr = p.find(w('pPr'))
        if ppr is None:
            ppr = etree.Element(w('pPr'))
            p.insert(0, ppr)
        rpr = ppr.find(w('rPr'))
        if rpr is None:
            rpr = etree.Element(w('rPr'))
            # rPr precedes sectPr and pPrChange
            after = [c for c in ppr if local(c) in ('sectPr', 'pPrChange')]
            if after:
                after[0].addprevious(rpr)
            else:
                ppr.append(rpr)
        rpr.insert(0, self.mark(kind))

    def new_paragraph(self, like, segments, ppr=None, rpr=None):
        """A tracked-inserted paragraph with the paragraph properties of `like` (or `ppr`)."""
        p = etree.Element(w('p'))
        src = ppr if ppr is not None else like.find(w('pPr'))
        if src is not None:
            src = copy.deepcopy(src)
            for x in src.findall(w('sectPr')) + src.findall(w('pPrChange')):
                src.remove(x)
            prp = src.find(w('rPr'))
            if prp is not None:
                for x in prp.findall(w('ins')) + prp.findall(w('del')):
                    prp.remove(x)
            p.append(src)
        self._mark_para(p, 'ins')
        if rpr is None and like is not None:
            rpr = self.template_rpr(like)
        for el in self.build(segments, copy.deepcopy(rpr) if rpr is not None else None):
            p.append(el)
        return p

    def change_ppr(self, p, new_ppr_children):
        """Replace a paragraph's properties (keeping its rPr), recording the old ones in w:pPrChange."""
        ppr = p.find(w('pPr'))
        if ppr is None:
            ppr = etree.Element(w('pPr'))
            p.insert(0, ppr)
        old = etree.Element(w('pPr'))
        keep_rpr = ppr.find(w('rPr'))
        for c in list(ppr):
            if local(c) not in ('rPr', 'sectPr', 'pPrChange'):
                old.append(copy.deepcopy(c))
                ppr.remove(c)
        for c in reversed(new_ppr_children):
            ppr.insert(0, c)
        chg = self.mark('pPrChange')
        chg.append(old)
        ppr.append(chg)
        if keep_rpr is not None:
            ppr.remove(keep_rpr)
            chg.addprevious(keep_rpr)

    # ---------- tables ----------
    def mark_table(self, tbl, kind):
        """Mark every row and paragraph of a table as inserted ('ins') or deleted ('del')."""
        for tr in tbl.iter(w('tr')):
            trpr = tr.find(w('trPr'))
            if trpr is None:
                trpr = etree.Element(w('trPr'))
                tcs = tr.find(w('tblPrEx'))
                (tcs.addnext if tcs is not None else (lambda e: tr.insert(0, e)))(trpr)
            trpr.append(self.mark(kind))
        for p in tbl.iter(w('p')):
            if kind == 'del':
                self.delete_paragraph(p)
            else:
                self.mark_inserted(p)

    def mark_inserted(self, p):
        """Mark an existing (copied) paragraph's content and mark as inserted."""
        for ch in list(p):
            tag = local(ch)
            if tag == 'r':
                ins = self.mark('ins')
                ch.addprevious(ins)
                ins.append(ch)
            elif tag in ('sdt',):
                content = ch.find(w('sdtContent'))
                ins = self.mark('ins')
                for r in [c for c in content if local(c) == 'r']:
                    ins.append(r)
                content.append(ins)
            elif tag == 'hyperlink':
                for r in [c for c in ch if local(c) == 'r']:
                    ins = self.mark('ins')
                    r.addprevious(ins)
                    ins.append(r)
            elif ch.tag in (m('oMathPara'), m('oMath')):
                for om in ([ch] if ch.tag == m('oMath') else ch.findall(m('oMath'))):
                    ins = self.mark('ins')
                    for c in list(om):
                        ins.append(c)
                    om.append(ins)
        self._mark_para(p, 'ins')


A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WP_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
IMAGE_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'


def swap_picture(editor, body, rels_root, media_dir, old_target, new_png, new_name, doc_pr_id):
    """Replace the picture whose relationship targets media/old_target by new_png, as a tracked deletion of the
    old drawing run and a tracked insertion of a copy that points at the new image (same size)."""
    import shutil
    from pathlib import Path
    rids = [r.get('Id') for r in rels_root if r.get('Target') == 'media/' + old_target]
    assert len(rids) == 1, (old_target, rids)
    runs = [next(a for a in blip.iterancestors() if a.tag == w('r')) for blip in
            [b for b in body.iter('{%s}blip' % A_NS) if b.get('{%s}embed' % R) == rids[0]]]
    assert len(runs) == 1, (old_target, len(runs))
    old = runs[0]
    shutil.copy(new_png, Path(media_dir) / new_name)
    new_id = 'rIdSwap' + Path(new_name).stem
    rel = etree.SubElement(rels_root, '{%s}Relationship' % REL_NS)
    rel.attrib.update({'Id': new_id, 'Type': IMAGE_REL, 'Target': 'media/' + new_name})
    new = copy.deepcopy(old)
    for b in new.iter('{%s}blip' % A_NS):
        b.set('{%s}embed' % R, new_id)
    for dp in new.iter('{%s}docPr' % WP_NS):
        dp.set('id', str(doc_pr_id))
    ins = editor.mark('ins')
    old.addnext(ins)
    ins.append(new)
    editor._delete_el(old)


def update_table(editor, tbl, rows, first_row=1):
    """Set the text of each body cell to rows[i][j] (tracked), leaving unchanged cells alone; returns changes."""
    trs = tbl.findall(w('tr'))[first_row:]
    assert len(trs) == len(rows), (len(trs), len(rows))
    n = 0
    for tr, values in zip(trs, rows):
        tcs = tr.findall(w('tc'))
        assert len(tcs) == len(values), (len(tcs), values)
        for tc, v in zip(tcs, values):
            p = tc.find(w('p'))
            cur = editor.text(p)
            if v is None or cur.strip() == str(v).strip():
                continue
            if cur.strip():
                editor.replace(p, cur.strip(), str(v))
            else:
                editor.append(p, str(v))
            n += 1
    return n


def strip_ids(el):
    """Drop w14:paraId/textId (must be unique per document) from copied content."""
    for x in el.iter():
        for a in ('{%s}paraId' % W14, '{%s}textId' % W14):
            if a in x.attrib:
                del x.attrib[a]
    return el
