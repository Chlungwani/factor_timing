"""Apply Prof. Gilbert's pre-submission review (Nov 2026) to a new copy of the thesis, as tracked changes.

Usage: python build_supervisor_corrections.py <thesis_snapshot.docx> <output.docx>

The snapshot is a `cp` of legacy/27717801_ResearchAssignment_vf - EG Mar 2026.docx (Word keeps the original
open). The original is never written. Every text, table, equation and appendix change is a Word revision by
"Claude", so the author can accept or reject each one. Not tracked (Word cannot track them either): Source
Manager entries (customXml) and citation field codes (the tags), which Word does not show; and settings.xml,
which asks Word to update fields on opening. The text that citation fields show is changed as tracked revisions.

Numbers come from legacy/results/model_outputs (the models rerun with the initial search purged at the evaluation
boundary, on the features selected from the weeks before the evaluation block, with the weeks before a series
starts filled by training-window medians and flagged), legacy/data_input, legacy/results/appendix_b (the
selection) and, regenerated from those outputs first, analysis/a4_accuracy_ci.csv,
analysis/appendix_c_after_purge.json and figures_after_purge/; they are checked in check_numbers() and where used.
Item labels (A1-A10, P = presentation, T = typography, R = references) follow the review's sections.
"""
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_tracked import (W, M, R, Editor, Cite, Italic, Bold, Seq, w, m, local, strip_ids,  # noqa: E402
                          swap_picture, update_table)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'legacy'
OUTPUTS = LEGACY / 'results' / 'model_outputs'
DATA = LEGACY / 'data_input'
CALIBRI = ROOT / 'outputs' / 'appendices_calibri_20261104' / 'Appendices_Revised_Calibri.docx'
A4_TABLE = HERE / 'analysis' / 'a4_accuracy_ci.csv'
# Appendix C tables and Figure C.1 of the purged rerun (analysis/appendix_c_values.py), Chapter 4 figures
# (legacy/results/build_chapter4_corrected_figures.py)
APPX = json.load(open(HERE / 'analysis' / 'appendix_c_after_purge.json', encoding='utf-8'))
FIGS = HERE / 'figures_after_purge'
N_TUNE = APPX['n_tune']                      # initial tuning sample after the 12-week boundary purge
# the feature selection on the weeks before the evaluation block (legacy/results/appendix_b_feature_selection.py)
APPB = LEGACY / 'results' / 'appendix_b'
SEL = json.load(open(APPB / 'summary.json', encoding='utf-8'))
N_REPS, N_FEAT = SEL['baseline_clusters'], SEL['baseline_clusters'] + 18
sys.path.insert(0, str(HERE / 'analysis'))
import results_facts  # noqa: E402
SKILL = Path(r'C:\Users\ccwis\.claude\skills\synced\397c6032-8a03-4c2d-ac67-33439714931a_0b484cf2-80cc-47fa-be32-'
             r'c76c5be2c01e\docx\scripts')
B = '{http://schemas.openxmlformats.org/officeDocument/2006/bibliography}'
NS = {'w': W, 'm': M}
ED = Editor('Claude', '2026-11-09T12:00:00Z')
LOG = []


def log(item, what):
    LOG.append((item, what))


# ======================================================================================= numbers (checked)
def check_numbers():
    """Every figure the new text states, recomputed from the model outputs and data."""
    acc = pd.read_csv(A4_TABLE).set_index(['factor_pair', 'model'])
    diff = pd.read_csv(OUTPUTS / 'bootstrap' / 'auc_paired_differences.csv').set_index(['factor_pair', 'comparison'])
    metrics = pd.read_csv(OUTPUTS / 'metrics_summary.csv').set_index(['factor_pair', 'model'])
    probs = {p: pd.read_csv(OUTPUTS / 'probs' / f'xgboost_{p.lower()}_probs.csv') for p in
             ('Momentum_vs_Value', 'Momentum_vs_Quality', 'Value_vs_Quality')}
    F = results_facts.facts(OUTPUTS)
    for k in [k for k in F if k.endswith(('|roc_auc', '|accuracy')) and k.count('|') == 2]:
        pair, model, col = k.split('|')
        assert abs(F[k] - acc.loc[(pair, model), col]) < 1e-9 and abs(F[k] - metrics.loc[(pair, model), col]) < 1e-9
    # the outputs are those of the purged design: every initial search used the N_TUNE observations
    assert N_TUNE == 746 and all(F[f'{p}|{m}|schedule'][0]['n_train'] == N_TUNE for p in PAIR_NAMES for m in MODELS)
    assert all(abs(F[f'{p}|{m}|cpcv_mean_auc'] - APPX['cpcv_mean'][f'{p}|{m}']) < 1e-12
               for p in PAIR_NAMES for m in MODELS), 'appendix_c_after_purge.json is stale'
    a = lambda pair, model, col: acc.loc[(pair, model), col]
    for pair in PAIR_NAMES:
        for model in MODELS:
            above = a(pair, model, 'auc_ci_low') > 0.5
            assert above == (pair == 'Value_vs_Quality' and model == 'XGBoost'), (pair, model)
            assert not a(pair, model, 'acc_ci_low') > a(pair, model, 'majority_class_accuracy'), (pair, model)
    assert len(diff) == 9 and all(diff.ci_low < 0) and all(diff.ci_high > 0)
    mv, mq = probs['Momentum_vs_Value'].actual.values, probs['Momentum_vs_Quality'].actual.values
    assert (mv != mq).sum() == 34 and mv.sum() == mq.sum() == 85
    # the selection uses only the weeks before the evaluation block (A2)
    reps = pd.read_csv(DATA / 'representatives.csv')['Representative']
    assert (N_REPS, N_FEAT) == (29, 47) and len(reps) == N_REPS
    assert (SEL['selection_start'], SEL['selection_end'], SEL['selection_weeks']) == ('2007-03-05', '2022-01-24', 778)
    assert (SEL['removed_missing'], SEL['screened'], SEL['removed_constant'], SEL['clustered']) == (23, 336, 10, 326)
    assert (SEL['mi_start'], SEL['mi_weeks']) == ('2011-05-09', 560)
    lag = pd.read_csv(DATA / 'lagged_data.csv', parse_dates=['Date']).set_index('Date')
    assert lag.shape[1] == N_FEAT and set(reps) <= set(lag.columns)
    # features that start after the matrix: missing, then median-filled and flagged in each training window (A1)
    gaps = lag.isna().sum()
    gaps = gaps[gaps > 0]
    assert sorted(gaps.index) == sorted(LATE) and (gaps.min(), gaps.max()) == (49, 140), gaps
    assert lag[LATE].isna().any(axis=1).loc[lambda s: s].index.max() < pd.Timestamp('2010-04-01')
    assert not lag.iloc[-170:].isna().any().any()
    assert APPX['table_c1'][0][2].endswith('May 2010'), 'the filled weeks must lie in split A'
    # sample counts (A7)
    fr = pd.read_csv(DATA / 'factor_returns.csv', parse_dates=['Date'])
    assert len(fr) == 939 and str(fr.Date.min().date()) == '2007-04-09'
    lag = pd.read_csv(DATA / 'lagged_data.csv', parse_dates=['Date'])
    assert len(lag) == 924 and str(lag.Date.min().date()) == '2007-07-23'
    lab = pd.read_csv(DATA / 'labeled_relative_factor_returns.csv', parse_dates=['Date']).dropna()
    assert int((lab.Date >= lag.Date.min()).sum()) == 914
    return dict(acc=acc, diff=diff, metrics=metrics, F=F)


LATE = ['SA_CDS_Curvature', 'ZAR18M.Curncy_QS', 'USDZARV1M.Curncy_QS', 'USDZARV1Y.Curncy_QS']   # start after the matrix
PAIR_NAMES = {'Momentum_vs_Value': 'momentum versus value', 'Momentum_vs_Quality': 'momentum versus quality',
              'Value_vs_Quality': 'value versus quality'}
MODELS = ('Logistic', 'GAM', 'XGBoost')
f3 = '{:.3f}'.format
pct = lambda x: f'{100 * x:.1f}%'  # noqa: E731
WORDS = dict(enumerate(['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']))


# ======================================================================================= package helpers
def unpack(src, dst):
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    subprocess.run([sys.executable, str(SKILL / 'merge_runs.py'), str(dst)], check=True, capture_output=True)


def pack(src_dir, out):
    tmp = out.with_suffix('.tmp.docx')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        names = sorted(p for p in src_dir.rglob('*') if p.is_file())
        ct = src_dir / '[Content_Types].xml'
        z.write(ct, '[Content_Types].xml')
        for p in names:
            if p != ct:
                z.write(p, p.relative_to(src_dir).as_posix())
    shutil.move(tmp, out)


class Doc:
    def __init__(self, path):
        self.path = path
        self.tree = etree.parse(str(path))
        self.root = self.tree.getroot()

    def save(self):
        self.tree.write(str(self.path), xml_declaration=True, encoding='UTF-8', standalone=True)


def paragraphs(root, skip_styles=('TOC1', 'TOC2', 'TOC3', 'TableofFigures')):
    for p in root.iter(w('p')):
        st = p.find('w:pPr/w:pStyle', NS)
        if st is not None and st.get(w('val')) in skip_styles:
            continue
        yield p


def find(root, snippet, count=1, regex=False):
    hits = []
    for p in paragraphs(root):
        try:
            t = ED.text(p)
        except ValueError:
            t = ''.join(x.text or '' for x in p.iter(w('t')))
        if (re.search(snippet, t) if regex else snippet in t):
            hits.append(p)
    if len(hits) != count:
        raise AssertionError(f'{len(hits)} paragraphs (expected {count}) contain {snippet!r}')
    return hits if count != 1 else hits[0]


def edit(root, snippet, old, new, item, what, regex=False, count=1):
    p = find(root, snippet)
    ED.replace(p, old, new, regex=regex, count=count)
    log(item, what)
    return p


def typed_outside_fields(p, text):
    """Occurrences of text in p that are typed, not the shown text of a citation field."""
    in_fields = sum(''.join(x.text or '' for x in s.iter(w('t'))).count(text) for s in p.iter(w('sdt')))
    return ED.text(p).count(text) - in_fields


def cite_typed(p, typed, cite):
    """Turn a typed citation into a citation field; leave it if it already is one."""
    n = typed_outside_fields(p, typed)
    if n:
        ED.replace(p, typed, cite, count=n)
    return n


# ======================================================================================= sources (R, A10)
SOURCE_DELETE = [  # uncited (R: 'listed but apparently uncited') and duplicate entries
    'Adr16', 'Alo22', 'Ang12', 'Bel57', 'Ber19', 'Bru13', 'Cho02', 'Goo94', 'Gri22', 'The', 'Hua20', 'Koi18',
    'Kri07', 'Lim19', 'Liu25', 'Nic05', 'Out', 'Sim19', 'Tob21', 'WuH21', 'Placeholder1', 'Placeholder3',
    'Placeholder5', 'Placeholder6', 'Hastie2', 'Guy031', 'Jos191', 'Jos24', 'Pen05']
RETAG = {'Hastie2': 'Has09', 'Guy031': 'Guy03', 'Jos191': 'Jos19'}

P = lambda last, first, middle=None: (last, first, middle)  # noqa: E731
SOURCE_EDITS = {
    'Mit97': dict(City='New York', Edition='1'),
    'Lop18': dict(authors=[P('López de Prado', 'M.')], City='Hoboken'),
    'Lop20': dict(authors=[P('López de Prado', 'M.')], Edition='1', City='Cambridge'),
    'Qua19': dict(authors=[P('Asness', 'C.', 'S.'), P('Frazzini', 'A.'), P('Pedersen', 'L.', 'H.')]),
    'Dan16': dict(authors=[P('Daniel', 'K.'), P('Moskowitz', 'T.', 'J.')]),
    'Tib96': dict(authors=[P('Tibshirani', 'R.')], JournalName='Journal of the Royal Statistical Society: Series B'),
    'Fam89': dict(JournalName='Journal of Financial Economics', Volume='25', Issue='1', Pages='23–49'),
    'Fam15': dict(JournalName='Journal of Financial Economics', Volume='116', Issue='1', Pages='1–22'),
    'Has09': dict(Title='The Elements of Statistical Learning'),
    'Hos13': dict(Title='Applied Logistic Regression'),
    'Leu21': dict(City='Lancaster'),
    'Štr10': dict(authors=[P('Štrumbelj', 'E.'), P('Kononenko', 'I.')],
                  Title='Explaining prediction models and individual predictions with feature contributions',
                  JournalName='Knowledge and Information Systems', Year='2014', Volume='41', Issue='3',
                  Pages='647–665'),
    'Štr101': dict(Volume='11'),
    'Pag02': dict(Year='2022', JournalName='Finance Research Letters', Volume='46', Issue='Part B', Pages='102504'),
    'Muk23': dict(authors=[P('Mukoyi', 'L.'), P('Ogujiuba', 'K.', 'K.')], Volume='16', Issue='1', Pages='4'),
    'Isr20': dict(JournalName='Journal of Investment Management', Volume='18', Issue='2', Pages='23–36'),
    'Ilm21': dict(Volume='19', Issue='4', Pages=None),
    'Bar15': dict(Volume='116', Issue='1', Pages='111–120'),
    'Had20': dict(JournalName='Review of Financial Studies', Volume='33', Issue='5', Pages='1980–2018'),
    'Asn17': dict(JournalName='Journal of Portfolio Management', Volume='43', Issue='5', Pages='72–87'),
    'Arn19': dict(JournalName='Journal of Financial Data Science', Volume='1', Issue='1', Pages='64–74'),
    'Han23': dict(JournalName='Emerging Markets Review', Volume='55', Pages='101022'),
    'Mut07': dict(JournalName='Investment Analysts Journal', Volume='36', Issue='65', Pages='15–24'),
    'Coc11': dict(JournalName='Journal of Finance', Volume='66', Issue='4', Pages='1047–1108'),
    'Arn23': dict(Pages='3034–3070'),
    'Ber12': dict(authors=[P('Bergmeir', 'C.'), P('Benítez', 'J.', 'M.')], JournalName='Information Sciences',
                  Volume='191'),
    'Cam97': dict(Edition=None, City='Princeton'),
    'Nus23': dict(Volume='155', Issue=None, Pages='104725'),
    'Che16': dict(ConferenceName='Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge '
                                 'Discovery and Data Mining', Pages='785–794', Publisher='ACM'),
    'Cot24': dict(JournalName='SSRN Working Paper No. 4698640', Volume=None, Issue=None, Pages=None),
    'Cov06': dict(Edition='2', City='Hoboken'),
    'Dix17': dict(JournalName='Algorithmic Finance', Volume='6', Issue='3–4', Pages='67–77'),
    'Fan22': dict(Year='2023', JournalName='International Review of Economics & Finance', Volume='88', Issue=None,
                  Pages='280–297'),
    'Faw06': dict(JournalName='Pattern Recognition Letters', Volume='27', Issue='8', Pages='861–874'),
    'Fen23': dict(Year='2024', Volume='59', Issue='7'),
    'Shi20': dict(JournalName='The Review of Financial Studies', Volume='33', Issue='5', Pages='2223–2273'),
    'Guy03': dict(Volume='3'),
    'Han24': dict(Volume='44', Issue=None, Pages='135–147'),
    'Kra04': dict(Volume='69', Issue='6'),
    'Kra17': dict(Volume='259', Issue='2'),
    'Lak94': dict(JournalName='The Journal of Finance'),
    'Lun17': dict(Volume='30', Pages='4765–4774'),
    'Lun20': dict(Title='From local explanations to global understanding with explainable AI for trees',
                  Volume='2', Issue='1'),
    'Mur73': dict(Volume='12', Issue='4'),
    'Pio22': dict(authors=[P('Piovezan', 'R.', 'P. B.'), P('de Andrade Júnior', 'P.', 'P.'), P('Ávila', 'S.', 'L.')],
                  Title='Machine learning method for return direction forecast of exchange traded funds (ETFs) '
                        'using classification and regression models',
                  Year='2024', JournalName='Computational Economics', Volume='63', Issue='5', Pages='1827–1852'),
    'Sae22': dict(Volume='11', Issue='1', Pages='5'),
    'Sul16': dict(authors=[P('Sulaiman', 'M.', 'A.'), P('Labadin', 'J.')], JournalName='Journal of IT in Asia',
                  Volume='6', Issue=None, Pages='11–24'),
    'War63': dict(Volume='58', Issue='301'),
}
NEW_SOURCES = [
    dict(Tag='Fra00', SourceType='JournalArticle', authors=[P('Fraser', 'E.'), P('Page', 'M.')],
         Title='Value and momentum strategies: Evidence from the Johannesburg Stock Exchange',
         JournalName='Investment Analysts Journal', Year='2000', Volume='29', Issue='51', Pages='25–35'),
    dict(Tag='Mul13', SourceType='JournalArticle', authors=[P('Muller', 'C.'), P('Ward', 'M.')],
         Title='Style-based effects on the Johannesburg Stock Exchange: A graphical time-series approach',
         JournalName='Investment Analysts Journal', Year='2013', Volume='42', Issue='77', Pages='1–16'),
    dict(Tag='Kün89', SourceType='JournalArticle', authors=[P('Künsch', 'H.', 'R.')],
         Title='The jackknife and the bootstrap for general stationary observations',
         JournalName='The Annals of Statistics', Year='1989', Volume='17', Issue='3', Pages='1217–1241'),
    dict(Tag='Pol92', SourceType='BookSection', authors=[P('Politis', 'D.', 'N.'), P('Romano', 'J.', 'P.')],
         editors=[P('LePage', 'R.'), P('Billard', 'L.')],
         Title='A circular block-resampling procedure for stationary data', BookTitle='Exploring the Limits of '
         'Bootstrap', Year='1992', Pages='263–270', City='New York', Publisher='Wiley'),
]
FIELD_ORDER = ['Title', 'BookTitle', 'ConferenceName', 'JournalName', 'Year', 'Volume', 'Issue', 'Pages', 'Edition',
               'City', 'Publisher']


def _b(tag, text=None):
    el = etree.Element(B + tag)
    if text is not None:
        el.text = text
    return el


def _names(people):
    nl = _b('NameList')
    for last, first, middle in people:
        pe = _b('Person')
        pe.append(_b('Last', last))
        pe.append(_b('First', first))
        if middle:
            pe.append(_b('Middle', middle))
        nl.append(pe)
    return nl


def fix_sources(src_root):
    by_tag = {s.findtext(B + 'Tag'): s for s in src_root.findall(B + 'Source')}
    for tag in SOURCE_DELETE:
        src_root.remove(by_tag.pop(tag))
    for tag, changes in SOURCE_EDITS.items():
        s = by_tag[tag]
        for key, value in changes.items():
            if key == 'authors':
                holder = s.find(B + 'Author/' + B + 'Author')
                holder.remove(holder.find(B + 'NameList'))
                holder.append(_names(value))
                continue
            el = s.find(B + key)
            if value is None:
                if el is not None:
                    s.remove(el)
            elif el is not None:
                el.text = value
            else:
                s.append(_b(key, value))
    # trailing punctuation that Word prints twice ('estimation. .', 'Information Sciences.,')
    for s in src_root.findall(B + 'Source'):
        for key in ('Title', 'JournalName'):
            el = s.find(B + key)
            if el is not None and el.text:
                el.text = el.text.strip().rstrip('.,').strip() if not el.text.strip().endswith('?') else el.text.strip()
        for pe in s.iter(B + 'Person'):
            for key in ('First', 'Middle', 'Last'):
                el = pe.find(B + key)
                if el is not None and el.text:
                    el.text = el.text.strip().rstrip(',').strip()
    for e in NEW_SOURCES:
        s = _b('Source')
        s.append(_b('Tag', e['Tag']))
        s.append(_b('SourceType', e['SourceType']))
        s.append(_b('Guid', '{%s}' % __import__('uuid').uuid5(__import__('uuid').NAMESPACE_URL,
                                                               'factor-timing/' + e['Tag']).hex.upper()))
        roles = _b('Author')
        for role, key in (('Author', 'authors'), ('Editor', 'editors')):
            if e.get(key):
                h = _b(role)
                h.append(_names(e[key]))
                roles.append(h)
        s.append(roles)
        for key in FIELD_ORDER:
            if key in e:
                s.append(_b(key, e[key]))
        src_root.append(s)
    log('R', f'Source Manager: {len(SOURCE_DELETE)} uncited or duplicate sources removed, '
             f'{len(SOURCE_EDITS)} corrected, {len(NEW_SOURCES)} added (Fraser & Page 2000, Muller & Ward 2013, '
             f'Künsch 1989, Politis & Romano 1992)')


# Text that Word shows inside citation fields, as it will once the sources are fixed
SHOWN_FIXES = [
    (re.compile(r'Lopez de Prado M\. , '), 'López de Prado, '),
    (re.compile(r'Lopez de Prado, '), 'López de Prado, '),
    (re.compile(r'Tibshirina'), 'Tibshirani'),
    (re.compile(r'Auret, 202\)'), 'Auret, 2022)'),
    (re.compile(r'Fang, Chen, & Ren, 2022'), 'Fang, Chen, & Ren, 2023'),
    (re.compile(r'Feng, He, Polson, & Xu, 2023'), 'Feng, He, Polson, & Xu, 2024'),
    (re.compile(r'Piovezan & de Andrea Jr\., 2022'), 'Piovezan, de Andrade Júnior, & Ávila, 2024'),
    (re.compile(r'Daniel & Moskowitz, Momentum crashes, 2016'), 'Daniel & Moskowitz, 2016'),
    (re.compile(r'Fama & French, Common risk factors in the returns on stocks and bonds, 1993'), 'Fama & French, 1993'),
    (re.compile(r'Fama & French, A five-factor asset pricing model, 2015'), 'Fama & French, 2015'),
    (re.compile(r'Fama & French, Business conditions and expected returns on stocks and bonds, 1989'),
     'Fama & French, 1989'),
    (re.compile(r'Arnott, Kalesnik, & Linnainmaa, Factor momentum, 2023'), 'Arnott, Kalesnik, & Linnainmaa, 2023'),
    (re.compile(r'Daniel, Hirshleifer, & Subrahmanyam, Investor Psychology and Security Market Under and '
                r'Overreactions, 1998'), 'Daniel, Hirshleifer, & Subrahmanyam, 1998'),
    (re.compile(r'Guyon & Elisseeff, An introduction to variable and feature selection\.? ?, 2003'),
     'Guyon & Elisseeff, 2003'),
]


def fix_citation_fields(root):
    """Retag duplicate sources in field codes and refresh the text Word shows for the corrected sources."""
    n_retag = n_shown = 0
    for sdt in root.iter(w('sdt')):
        if sdt.find('w:sdtPr/w:citation', NS) is None:
            continue
        instr = sdt.findall('.//w:instrText', NS)
        tags = []
        for it in instr:
            new = it.text
            for old, tag in RETAG.items():
                new = re.sub(r'(?<=\s)' + re.escape(old) + r'(?=\s)', tag, new)
            if new != it.text:
                n_retag += 1
            # '\t' = Suppress title: Word's APA style prints the title whenever it flags an author as repeated,
            # although no two sources here share both author and year
            new = re.sub(r'((?:CITATION|\\m)\s+[^\s\\]+)', lambda mt: mt.group(1) + ' \\t', new)
            new = re.sub(r'(\\t\s+)+\\t', r'\\t', new)
            it.text = new
            toks = new.split()
            tags += [toks[i + 1] for i, t in enumerate(toks[:-1]) if t in ('CITATION', '\\m')]
        assert not set(tags) & set(SOURCE_DELETE), tags
        ts = [t for t in sdt.iter(w('t'))]
        shown = ''.join(t.text or '' for t in ts)
        new = shown
        for pat, rep in SHOWN_FIXES:
            new = pat.sub(rep, new)
        if tags == ['Štr10']:
            new = new.replace('2010', '2014')
        if new != shown:
            ED.replace_field_result(sdt, new)
            n_shown += 1
    log('R', f'Citation fields: "Suppress title" set on every citation; {n_retag} field codes moved from duplicate '
             f'sources, {n_shown} displayed citations '
             f'refreshed (López de Prado, Tibshirani, Page et al. 2022, titles no longer needed to tell duplicates '
             f'apart, Štrumbelj & Kononenko 2014, Fang et al. 2023, Feng et al. 2024, Piovezan et al. 2024)')


C = Cite  # brevity in the text below
CITE = {
    'Fra00+Mul13': C(['Fra00', 'Mul13'], '(Fraser & Page, 2000; Muller & Ward, 2013)'),
    'Mak23+Pag241': C(['Mak23', 'Pag241'], '(Makamo, 2023; Page, McClelland, & Auret, 2024)'),
    'Mut07': C(['Mut07'], '(Mutooni & Muller, 2007)'),
    'Pag02': C(['Pag02'], '(Page, McClelland, & Auret, 2022)'),
    'Pag241': C(['Pag241'], '(Page, McClelland, & Auret, 2024)'),
    'Mak23': C(['Mak23'], '(Makamo, 2023)'),
    'Muk23': C(['Muk23'], '(Mukoyi & Ogujiuba, 2023)'),
    'Kap19': C(['Kap19'], '(Kapche Fotso, 2019)'),
    'Lop18': C(['Lop18'], '(López de Prado, 2018)'),
    'Cam97+Lop18': C(['Cam97', 'Lop18'], '(Campbell, Lo, & MacKinlay, 1997; López de Prado, 2018)'),
    'MI': C(['Kra04', 'Cov06', 'Pen051', 'Guy03'],
            '(Kraskov, Stögbauer, & Grassberger, 2004; Cover & Thomas, 2006; Peng, Long, & Ding, 2005; '
            'Guyon & Elisseeff, 2003)'),
    'logloss': C(['Goo52', 'Gne07', 'Bis06', 'Mur73'],
                 '(Good, 1952; Gneiting & Raftery, 2007; Bishop, 2006; Murphy, 1973)'),
    'boot': C(['Kün89', 'Pol92'], '(Künsch, 1989; Politis & Romano, 1992)'),
    'Shi20': C(['Shi20'], '(Gu, Kelly, & Xiu, 2020)'),
    'Dan16': C(['Dan16'], '(Daniel & Moskowitz, 2016)'),
    'Nus23': C(['Nus23'], '(Cakici, Fieberg, Metko, & Zaremba, 2023)'),
    'Fan22': C(['Fan22'], '(Fang, Chen, & Ren, 2023)'),
    'Sae22': C(['Sae22'], '(Saetia & Yokrattanasak, 2022)'),
    'Has86': C(['Has86'], '(Hastie & Tibshirani, 1986)'),
}


# ======================================================================================= front matter, Ch 1-2
def front_matter(body, res):
    F = res['F']
    title = find(body, 'In Search of the Holy Grail of Factor Timing')
    sub = find(body, 'Application of Machine Learning to the South African Equity Market')
    new = ED.new_paragraph(sub, 'Application of Machine Learning to the South African Equity Market')
    title.addnext(new)
    ED.delete_paragraph(sub)
    log('P', 'Title page: subtitle moved above "by", so the full title comes before the name')
    edit(body, 'artificial Intelligent-assisted', 'Intelligent-assisted', 'intelligence-assisted', 'T',
         'Declaration: "intelligence-assisted"')
    # the abstract rewritten in two paragraphs, the contributions of Section 1.5 and then the results; every number
    # is read from the outputs and every claim asserted
    ab = find(body, 'This study tests whether machine learning can predict')
    assert ED.text(ab).startswith('Momentum, value and quality earn long-term premia')
    acc = res['acc']
    a = lambda pair, model, col: acc.loc[(pair, model), col]  # noqa: E731
    g = lambda pair, model, what: F[f'{pair}|{model}|{what}']  # noqa: E731
    xgb_auc = [g(p, 'XGBoost', 'roc_auc') for p in PAIR_NAMES]
    for p in PAIR_NAMES:
        assert sorted(MODELS, key=lambda m: g(p, m, 'roc_auc')) == ['Logistic', 'GAM', 'XGBoost']
        for m in MODELS:
            assert (a(p, m, 'auc_ci_low') > 0.5) == ((p, m) == (VQ, 'XGBoost')), (p, m)   # only one beats chance
            assert g(p, m, 'log_loss') > np.log(2)                                        # none beats 50%
    assert all(r.ci_low < 0 < r.ci_high for _, r in res['diff'].iterrows())               # no significant difference
    cm = lambda pair, m: g(pair, m, 'cpcv_mean_auc')  # noqa: E731
    assert all(cm(p, 'XGBoost') < min(cm(p, 'Logistic'), cm(p, 'GAM')) for p in (MV, MQ))
    with_relvol = sum(bool(g(p, m, 'shap_relvol_ranks')) for p in PAIR_NAMES for m in MODELS)
    assert with_relvol == 8 and not g(MQ, 'Logistic', 'shap_relvol_ranks')
    assert SEL['removed_missing'] + SEL['screened'] == 359 and N_REPS == 29
    dates = pd.to_datetime(wf_probs('XGBoost', VQ).Date)
    y_vq = wf_probs('XGBoost', VQ).actual                     # 1 = value leads
    assert len(dates) == 156 and (f'{dates.min():%B %Y}', f'{dates.max():%B %Y}') == ('January 2022', 'January 2025')
    assert y_vq.mean() < 0.5 and abs(a(VQ, 'XGBoost', 'majority_class_accuracy') - (1 - y_vq.mean())) < 1e-12
    ED.replace(ab, ED.text(ab),
               'Momentum, value and quality earn long-term premia, but which of them leads varies over time. This study '
               'asks whether machine learning can predict relative factor leadership in the South African equity '
               'market, and whether more flexible models predict it better. It makes three contributions. First, '
               'regularised logistic regression, a generalised additive model (GAM) and XGBoost are compared on '
               'identical data, predictors and test dates, so that differences in performance isolate the effect of '
               'model flexibility. Second, information leakage is controlled at every stage: trend scanning labels '
               'leadership in three factor pairs over a horizon chosen by the data rather than fixed in advance; '
               f'screening and mutual-information clustering reduce {SEL["removed_missing"] + SEL["screened"]} '
               f'candidate predictors to {N_REPS} using only data from before the evaluation period; combinatorial '
               'purged cross-validation tunes the hyperparameters; and a purged, expanding walk-forward procedure '
               f'produces {len(dates)} weekly forecasts from {dates.min():%B %Y} to {dates.max():%B %Y}. Third, '
               'Shapley values link each model’s forecasts to economically interpretable predictors.')
    ab.addnext(ED.new_paragraph(ab,
               'The evidence of predictability is weak and pair-specific. The area under the ROC curve (ROC-AUC), which '
               'measures how well the forecasts rank the weeks, rises from logistic regression to the GAM to XGBoost '
               f'for every pair, with XGBoost reaching {f3(min(xgb_auc))} to {f3(max(xgb_auc))}, but block-bootstrap '
               'confidence intervals show no significant difference between the models. Only XGBoost’s forecasts for '
               f'value versus quality are distinguishable from chance: their ROC-AUC of {f3(g(VQ, "XGBoost", "roc_auc"))} '
               f'has an interval of {f3(a(VQ, "XGBoost", "auc_ci_low"))} to {f3(a(VQ, "XGBoost", "auc_ci_high"))}, and '
               f'they are correct in {pct(g(VQ, "XGBoost", "accuracy"))} of weeks, against '
               f'{pct(a(VQ, "XGBoost", "majority_class_accuracy"))} for always predicting quality. In the two momentum '
               'pairs, XGBoost ranked last in cross-validation, which suggests that its lead out of sample is partly '
               'specific to the evaluation period. No model’s probabilities improve on a constant 50% forecast. '
               f'Relative-volatility measures rank among the top ten features of {WORDS[with_relvol]} of the nine '
               'fitted models. Machine learning therefore offers at most a weak, pair-specific signal, better suited to '
               'moderating the tilt between two factors than to switching between them.'))
    log('A4/A9', 'Abstract rewritten in two paragraphs: the three contributions of Section 1.5 (like-for-like model '
                 'comparison, leakage control at every stage, Shapley attribution), then the results of the rerun '
                 'with the confidence intervals (ROC-AUC ordering and range, only XGBoost for value versus quality '
                 'beats chance, CPCV evidence for the momentum pairs, no model better than a constant forecast, '
                 'relative volatility in eight of nine fits)')


def chapter1(body):
    p = find(body, 'Recent South African studies have begun to apply machine learning')
    ED.replace(p, '(Makamo, 2023; Page, McClelland, & Auret, 2024; Mutooni & Muller, 2007)', CITE['Mak23+Pag241'])
    ED.replace(p, 'Using three model classes of increasing flexibility, logistic',
               'Three model classes of increasing flexibility are compared: logistic')
    log('A10/T', 'Ch 1 opening: Mutooni & Muller (2007) no longer called recent; "Using three model classes" fragment '
                 'made a sentence')
    p = find(body, 'The South African setting provides a relevant but comparatively under-researched context')
    assert cite_typed(p, '(Fraser & Page, 2000; Muller & Ward, 2013)', CITE['Fra00+Mul13']) == 1
    ED.replace(p, r'\(Mutooni & Muller, 2007\) provide early evidence on equity style timing and \(Page, McClelland, & '
                  r'Auret, 2022?\) examine style rotation, while recent studies apply machine learning models to the '
                  r'same broad problem', ['Early evidence on equity style timing ', CITE['Mut07'],
                                     ' and on short-term style rotation ', CITE['Pag02'],
                                     ' has been followed by studies that apply machine learning to the same broad '
                                     'problem'], regex=True)
    log('A10/R', '1.1: Fraser & Page (2000) and Muller & Ward (2013) are now citation fields with sources; '
                 '"(Page, McClelland, & Auret, 202)" is the 2022 Finance Research Letters paper (year fixed); '
                 'citations no longer the subject of the sentence')
    edit(body, 'the study the following specific objectives', 'the study the following', 'the study has the following',
         'T', '1.3: "the study has the following specific objectives"')
    p = find(body, 'Firstly, a sophisticated target labelling technique')
    ED.replace(p, 'Firstly, a sophisticated target labelling technique appropriate for financial datasets. Secondly, '
                  'hierarchical comparison of model architectures. Lastly, a cross-validation methodology that '
                  'preserves temporal ordering of data.',
               'The first is a target-labelling technique suited to financial time series, the second a comparison '
               'of model classes of increasing flexibility, and the third a cross-validation procedure that '
               'preserves the temporal ordering of the data.')
    log('T', '1.4: fragments ("Firstly, a sophisticated…") rewritten as a sentence')
    edit(body, 'trend-scanning methodology proposed by López de Prado', 'methodology proposed by López de Prado',
         'methodology', 'T', '1.4: typed "proposed by López de Prado" removed before the citation')
    edit(body, 'vali-dation data', 'vali-dation', 'validation', 'T', '1.4: "validation"')
    edit(body, 'regularised linear bench-mark', 'bench-mark', 'benchmark', 'T', '1.6: "benchmark"')


def chapter2(body):
    edit(body, 'an L2 penalty that stabilise', 'that stabilise ', 'that stabilises ', 'T', '2.1.1: "stabilises"')
    edit(body, 'encountered in when attempting', 'encountered in when', 'encountered when', 'T',
         '2.2.1: "encountered when"')
    p = find(body, 'International evidence indicates that machine-learning methods can sometimes improve')
    ED.append(p, '.')
    log('T', '2.3: full stop at the end of the first paragraph')
    p = find(body, 'noted in Section 2.2.2')
    ED.replace(p, 'reviewed in Section 2.2.3', 'reviewed in Section 2.2.4')
    ED.replace(p, 'noted in Section 2.2.2', 'noted in Section 2.2.3')
    log('P', '2.3: cross-references to Sections 2.2.3 (market structure) and 2.2.4 (criticisms)')
    p = find(body, 'South African factor-timing evidence predates recent ML studies')
    ED.replace(p, r'\(Mutooni & Muller, 2007\) examine equity style timing, and .*?improve on a style-momentum '
                  r'benchmark after costs\.',
               ['Section 1.1 noted that value and momentum effects have long been documented on the JSE and that '
                'local style returns vary through time. Among multifactor models, the four-factor model prices JSE '
                'returns best across market conditions, and the value factor becomes redundant in bullish markets ',
                CITE['Muk23'], '; size and momentum premia vary with the business cycle ', CITE['Kap19'],
                '; and momentum-based style rotation is strongest over short estimation and holding periods ',
                CITE['Pag02'], '. Equity style timing on the JSE has also been examined directly ', CITE['Mut07'],
                '. Machine-learning studies are more recent: LSTM networks have been used to predict local factor '
                'and sector returns, with low in-sample prediction errors ', CITE['Mak23'],
                ', and random forests, XGBoost and LightGBM have been compared for JSE style rotation, with the '
                'tree-based models improving on a style-momentum benchmark after costs ', CITE['Pag241'], '.'],
               regex=True)
    log('A10', '2.3: Page et al. (2024) no longer described twice (the style-rotation finding is the 2022 paper); '
               'two further JSE studies added (Mukoyi & Ogujiuba, 2023; Kapche Fotso, 2019); citations moved out of '
               'subject position')
    p = find(body, 'The study also improves on the design of the studies reviewed above in five ways')
    ED.replace(p, 'The models are also compared on predictive measures', 'Fifth, the models are compared on '
                                                                         'predictive measures')
    log('T', '2.3: "five ways" now lists five ("Fifth, …")')


# ======================================================================================= chapter 3
def chapter3(body):
    p = find(body, 'These models provide progressively greater functional flexibility')
    ED.append(p, '.')
    edit(body, 'binary classification task; momentum versus value', 'task; momentum versus value',
         'task for three pairs: momentum versus value', 'T', '3.1: "task for three pairs:"')
    edit(body, 'trend-scanning methodology of López de Prado', 'methodology of López de Prado', 'methodology', 'T',
         '3.1: typed author name removed before the citation')
    p = find(body, 'using a walk-forward ensuring')
    ED.replace(p, 'using a walk-forward ensuring', 'using a walk-forward procedure, ensuring')
    ED.replace(p, 'used as inputs the walk-forward', 'used as inputs to the walk-forward')
    p = find(body, 'designed to addresses')
    ED.replace(p, 'walk-forwar\td', 'walk-forward')
    ED.replace(p, 'designed to addresses', 'designed to address')
    log('T', '3.1: full stop; "walk-forward procedure, ensuring"; "inputs to"; "walk-forward" (a tab sat inside the '
             'word); "to address"')
    p = find(body, 'A 2024 review of South African factor portfolios')
    ED.replace(p, 'Their persistence across markets and economic regimes motivates their selection as predictive '
                  'targets for timing analysis. Their persistence', 'Their persistence')
    ED.replace(p, 'A 2024 review of', 'A review of')
    log('T', '3.2: repeated sentence deleted; "A 2024 review" no longer contradicts the 2025 citations')
    edit(body, 'correlations in-volving quality', 'in-volving', 'involving', 'T', '3.3: "involving"')
    p = find(body, 'The study therefore focuses on identifying changes in relative factor leadership')
    ED.append(p, '.')
    log('T', '3.3: full stop at the end of the section')
    # 3.4 (A7, A1)
    p = find(body, 'The empirical analysis uses weekly observations from March 2007 to March 2025')
    ED.replace(p, 'from March 2007 to March 2025, a total of 944 weeks', 'from April 2007 to March 2025, a total of '
                                                                         '939 weeks')
    ED.replace(p, 'Appendix A defines each feature', 'Appendix A defines each modelling feature')
    log('A7', '3.4: 939 weeks from April 2007 (factor returns), not 944 from March')
    p = find(body, 'following established practice in empirical finance and financial machine learning')
    if cite_typed(p, '(Campbell, Lo, & MacKinlay, 1997; López de Prado, 2018)', CITE['Cam97+Lop18']):
        log('R', '3.4: typed citation "(…; López de Prado, 2018)" made a citation field')
    p = find(body, 'The procedure retains 32 representative predictors')
    ED.replace(p, 'retains 32 representative predictors', f'retains {N_REPS} representative predictors')
    ED.replace(p, 'a final set of 50 predictors', f'a final set of {N_FEAT} predictors')
    log('A2', f'3.1: {N_REPS} representatives and {N_FEAT} predictors from the selection on the weeks before the '
              'evaluation block')
    p = find(body, 'Remaining gaps are forward-filled')
    ED.replace(p, r'; gaps before a series.? first observation are back-filled\.',
               [f'. Four of the {N_FEAT} modelling features begin after the feature matrix does (SA_CDS_Curvature, '
                'ZAR18M.Curncy_QS, USDZARV1M.Curncy_QS and USDZARV1Y.Curncy_QS), so their first 49 to 140 weeks, all '
                'before April 2010, have no value. Rather than copy a later value backwards, each model fills these '
                'weeks with the feature’s median in its training window and receives a 0/1 indicator for each '
                'of the four features that marks the filled weeks, so that filled and observed values can be told '
                'apart. No week of the evaluation block is missing.'], regex=True)
    log('A1', '3.4: no back-filling: the four features that start after the matrix (49-140 weeks, before April '
              '2010) take the training-window median, with a 0/1 indicator for each; no evaluation week is missing')
    p = find(body, 'The resulting feature matrix contains 929 weekly observations')
    ED.replace(p, '929', '924')
    # 3.5
    edit(body, 'trend scanning method introduced by Lopez de Prado', 'method introduced by Lopez de Prado',
         'method', 'T', '3.5: typed author name removed before the citation')
    edit(body, 'trend-scanning methodology by Lopez de Prado', 'methodology by Lopez de Prado', 'methodology', 'T',
         '3.5: typed author name removed before the citation')
    p = find(body, 'There is no prescriptive method to pick the maximum lookahead period')
    ED.replace(p, r'week\. The choice', 'weekly observations, that is, the current week and the ten weeks that '
                                        'follow. The choice', regex=True)
    log('A7/T', '3.5: H = 11 weekly observations (the current week and the ten that follow)')
    p = find(body, 'Because labels incorporate forward-looking information up to 12 weeks')
    ED.replace(p, 'Because labels incorporate forward-looking information up to 12 weeks, overlapping',
               'Because a label draws on returns up to ten weeks ahead, the last ten weeks of the sample cannot be '
               'labelled, and overlapping')
    ED.replace(p, 'a purge gap of 12 weeks is imposed', 'a purge gap of 12 weeks, longer than this horizon, is imposed')
    log('A7', '3.5: label horizon stated as ten weeks ahead (not 12), with the reason the last ten weeks are '
              'unlabelled')
    p = find(body, 'Trend-scanning is applied on the rolling 12-week relative factor returns')
    ED.replace(p, 'Trend-scanning is applied on the rolling 12-week relative factor returns. ', '')
    log('A7', '3.5: deleted "applied on the rolling 12-week relative factor returns" (the code scans the '
              'cumulative series of Equation 2)')
    # 3.6
    p = find(body, 'the factor-zoo')
    ED.replace(p, 'Section 2.1 noted that the proliferation', 'Section 2.2.1 noted that the proliferation')
    ED.replace(p, 'and Section 2.2 that many correlated', 'and Section 2.1.1 that many correlated')
    ED.replace(p, 'summarised in Table 2', 'summarised in Table 1')
    ED.replace(p, 'from 919 labelled weekly observations', 'from 914 labelled weekly observations')
    ED.replace(p, 'with each feature defined in Appendix A', 'with the candidates described and the modelling features '
                                                              'defined in Appendix A')
    log('P/A7', '3.6: Sections 2.2.1 and 2.1.1 (were 2.1 and 2.2); Table 1 (was Table 2); 914 labelled weeks')
    p = find(body, 'The missing-value screen removes the 15 candidates')
    ED.replace(p, 'The missing-value screen removes the 15 candidates with more than 30% of weekly observations '
                  'missing, leaving 344 (Appendix B, Tables B.1 and B.2).',
               f'Every step of the selection uses only the {SEL["selection_weeks"]} weeks before the evaluation block, '
               'from 5 March 2007 to 24 January 2022. The missing-value screen removes the '
               f'{SEL["removed_missing"]} candidates with more than 30% of these weeks missing, leaving '
               f'{SEL["screened"]}, and mutual information is estimated on the {SEL["mi_weeks"]} weeks from 9 May '
               f'2011, when the last of them begins. The ten features that do not vary over '
               f'those weeks are also removed, leaving {SEL["clustered"]} (Appendix B, Tables B.1 and B.2).')
    log('A2', f'3.6: the screen and the MI estimation use only the weeks before the evaluation block ('
              f'{SEL["selection_weeks"]} and {SEL["mi_weeks"]} weeks); {SEL["removed_missing"]} candidates screened '
              f'out, {SEL["removed_constant"]} constant ones removed, {SEL["clustered"]} clustered')
    p = find(body, 'reduces uncertainty about the other')
    ED.replace(p, '(i.e. How much', '(i.e. how much')
    ED.replace(p, r'\(Kraskov.*?, 2003\)(?=\. Each)', CITE['MI'], regex=True)
    log('T/R', '3.6: "how much"; one citation field instead of a typed citation plus a field that cited Guyon & '
               'Elisseeff twice')
    p = find(body, 'Hierarchical agglomeration clustering applied to the features space')
    ED.replace(p, 'Hierarchical agglomeration clustering applied to the features space',
               'Hierarchical agglomerative clustering of the feature space')
    log('T', 'Figure 6 caption: "Hierarchical agglomerative clustering of the feature space"')
    p = find(body, 'Importantly, this selection is unsupervised')
    ED.replace(p, 'resulting in a total of 32 clusters', f'resulting in a total of {N_REPS} clusters')
    ED.replace(p, 'Importantly, this selection is unsupervised and does not rely on predictive performance relative to '
                  'the target variable. As a result, feature reduction is conducted independently of the '
                  'classification task, avoiding look-ahead bias and preventing inadvertent target leakage.',
               'The selection is unsupervised: it does not use the labels, and it draws only on weeks before the '
               'evaluation block, so neither target information nor any observation of the evaluation block enters '
               'it.')
    log('A2', f'3.6: {N_REPS} clusters; the "avoiding look-ahead bias" claim replaced by what the selection uses (no '
              'labels, no evaluation-block weeks)')
    p = find(body, 'The final modelling matrix therefore consists of 50 predictors')
    ED.replace(p, 'consists of 50 predictors: 32 representative features', f'consists of {N_FEAT} predictors: '
                                                                           f'{N_REPS} representative features')
    # Table 1: the categories of the new selection (A2); BCMPEBLS is a euro-area bank lending survey (A8)
    cap = find(body, 'Summary of the 50 modelling features by category')
    ED.replace(cap, 'Summary of the 50 modelling', f'Summary of the {N_FEAT} modelling')
    tbl1 = cap.getnext()
    assert local(tbl1) == 'tbl'
    n = update_table(ED, tbl1, [
        [None, str(sum(g == 'Price-based volatility' for _, g, _ in FEATURES)),
         'Volatility term-structure level; USDZAR and EURUSD implied volatility; gold price measures', None, None],
        [None, str(sum(g == 'Fundamental valuation' for _, g, _ in FEATURES)),
         'MSCI World fundamental component, price-to-sales and trailing-to-forward price-to-earnings gap; TOP40 '
         'price-to-book and price-to-EBITDA', None, None],
        [None, str(sum(g == 'Sentiment and risk appetite' for _, g, _ in FEATURES)),
         'US consumer confidence; commodity index component; gold structural-break, complexity, skewness and '
         'kurtosis measures; global equity and bond trends; global equity return and entropy', None, None],
        [None, str(sum(g == 'Macroeconomic' for _, g, _ in FEATURES)),
         'SA and US yield-curve curvature; SA yield-curve slope; SA CDS curvature; rand PPP valuation trend; '
         'euro-area bank lending conditions; USDZAR forward pricing and entropy', None, None],
        [None] * 5, [None] * 5])
    assert n == 8, n
    log('A2/A8', f'Table 1: {N_FEAT} features; counts and examples of the four clustered categories from the new '
                 'selection (BCMPEBLS described as euro-area bank lending conditions, not energy prices)')
    for p in find(body, 'Gu et al. (2020); Daniel & Moskowitz (2016)', count=2):
        assert cite_typed(p, 'Gu et al. (2020); Daniel & Moskowitz (2016)', [CITE['Shi20'], '; ', CITE['Dan16']]) == 1
    p = find(body, 'Appendix A defines each feature, and Table B.8')
    ED.replace(p, 'Appendix A defines each feature, and Table B.8', 'Appendix A defines each feature, and Table B.6')
    log('T', 'Table 1: Evidence column in one citation style; note refers to Table B.6')
    # 3.7
    edit(body, 'The model assumes the assumes that', 'The model assumes the assumes that the log-odds of a binary '
                                                     'outcome are a linear function of the predictor:',
         'Formally, the log-odds of the outcome are a linear function of the predictors:', 'T',
         '3.7.2: duplicated "the assumes" and repeated assumption sentence')
    edit(body, 'The solver settings and the grid for', 'given in Table 4.', 'given in Table 3.', 'P',
         '3.7.2: grid is in Table 3')
    edit(body, 'Equation (9) with unit observation weights', 'Equation (9)', 'Equation (10)', 'P',
         '3.7.2: Equation (10) after the trend-scanning regression is numbered')
    p = find(body, 'is a vector of spline basis function')
    ED.replace(p, 'basis function,', 'basis functions,')
    ED.replace(p, '(Table 4)', '(Table 3)')
    log('T/P', '3.7.3: "basis functions"; Table 3')
    edit(body, 'We use the Pygam library from python', 'We use the Pygam library from python to implement this GAM '
                                                       'model.', 'We use the pyGAM library in Python to implement the '
                                                                 'GAM.', 'T', '3.7.3: "pyGAM library in Python"')
    p = find(body, 'is applied to all smooth terms and selected by cross-validation')
    ED.append(p, ' The four missing-value indicators of Section 3.4 enter as linear terms with the same penalty.')
    log('A1', '3.7.3: the missing-value indicators enter the GAM as linear terms')
    edit(body, 'with similar characteristic', 'similar characteristic.', 'similar characteristics.', 'T',
         '3.7.4: "characteristics"')
    edit(body, 'Where the logistic loss function defined as', 'Where the logistic loss function defined as',
         'where the logistic loss function is defined as', 'T', '3.7.4: "where … is defined as"')
    edit(body, 'tree-boosting framework described', 'executes the theoretical additive tree-boosting framework '
                                                    'described ', 'follows the additive tree-boosting framework set '
                                                                  'out for XGBoost ', 'T',
         '3.7.4: "framework described (Chen & Guestrin)" rephrased (citation stays in brackets, as elsewhere)')
    p = find(body, 'is the number of leaves and')
    ED.replace(p, 'the leaf weight.', ['the weight of leaf ', Italic('j'), '.'])
    # 3.8.1
    edit(body, 'Combinatorial cross-purged validation', 'Combinatorial cross-purged validation',
         'Combinatorial purged cross-validation', 'T', '3.8.1 heading: "Combinatorial purged cross-validation"')
    p = find(body, 'back-test path. Hyperparameters are selected on the mean ROC-AUC')
    ED.replace(p, 'back-test path.', 'back-test paths.')
    ED.append(p, ' The back-test paths are not used for selection. Each path pools predictions from four separately '
                 'fitted models whose training class shares differ, which shifts the level of their probabilities '
                 'and biases the pooled ROC-AUC downwards (Appendix C.2). The paths therefore serve only as a '
                 'diagnostic of how consistently the selected setting performs across the tuning sample.')
    log('A5/T', '3.8.1: "back-test paths"; says that the paths play no role in selection and what they add')
    p = find(body, 'addresses this by remove any training data')
    ED.replace(p, r'\(L.pez de Prado(?: M\. ,|,) 2018\) addresses this by remove any training data whose information '
                  r'horizon overlaps with the validation set \(i\.e\. purging\)\.',
               ['Purging addresses this by removing any training observation whose information horizon overlaps '
                'with the validation set ', CITE['Lop18'], '.'], regex=True)
    ED.replace(p, 'The initial search uses the 763 observations before the evaluation block, with predictors '
                  'standardised on the whole sample and no purge at its boundary, so the labels of its last ten '
                  'observations may draw on the first weeks of the evaluation block.',
               f'The initial search uses the {N_TUNE} observations that precede the evaluation block once the 12 '
               'weeks before it are purged, so that no label in the search draws on the evaluation block; the '
               'missing values of Section 3.4 are filled with the medians of that sample, and the predictors of '
               'logistic regression and the GAM are standardised on it.')
    log('A3/T', f'3.8.1: "removing"; the initial search is purged at the evaluation boundary ({N_TUNE} observations; '
                'models rerun with this purge)')
    edit(body, 'Final performance is there-fore assessed', 'there-fore', 'therefore', 'T', '3.8.2: "therefore"')
    edit(body, 'The training window grows from 751 observations', 'grows from 751 observations at the first prediction '
                                                                  'to 906 at the last',
         'grows from 746 observations at the first prediction to 901 at the last', 'A7', '3.8.2: 746 to 901')
    # 3.9
    p = find(body, 'ranking quality Accuracy is complemented')
    ED.replace(p, 'ranking quality Accuracy', 'ranking quality. Accuracy')
    p = find(body, 'Log-loss penalises confident but incorrect predictions')
    ED.replace(p, r'\(Good, 1952; Gneiting & Raftery, 2007\)\. \(Bishop, 2006; Murphy, 1973\)\.',
               [CITE['logloss'], '.'], regex=True)
    log('T', '3.9: missing full stop; the two log-loss citations merged into one bracket before the full stop')
    # the bootstrap as the evaluation-side counterpart of the purge, in its own paragraph; the caution sentence it
    # absorbs is removed from the paragraph on economic profitability
    p = find(body, 'small differences in these point estimates are interpreted with caution')
    ED.replace(p, ' Because the evaluation block contains only 156 weekly observations and the labels are persistent, '
                  'small differences in these point estimates are interpreted with caution.', '')
    assert ED.text(p).rstrip().endswith('falls outside the scope of this study.')
    p.addnext(ED.new_paragraph(p, [
        'The three measures are point estimates from 156 weekly predictions. Because the trend-scanned labels persist '
        'over several weeks, neighbouring observations are dependent, the same overlap that the purge removes from '
        'the training windows (Section 3.8), so resampling single weeks would understate the sampling uncertainty of '
        'the measures. It is instead estimated with a circular moving-block bootstrap, which resamples blocks of '
        'consecutive weeks and so preserves the dependence between neighbouring labels ', CITE['boot'], '. Blocks of '
        '12 weeks, the length of the purge, are drawn 10,000 times, and the 2.5th and 97.5th percentiles of each '
        'statistic form its 95% confidence interval. The same resampled weeks are applied to every model, so the '
        'interval for the difference between two models\u2019 ROC-AUCs is a paired comparison. Accuracy is also '
        'compared with the majority-class rate, the accuracy of always predicting the more frequent class. The models '
        'are not refitted in the bootstrap, so the intervals do not include the uncertainty of feature and '
        'hyperparameter selection.']))
    log('A4', '3.9: new paragraph: block-bootstrap confidence intervals (motivated by the same label dependence as '
              'the purge), paired model comparison and majority-class baseline; the caution sentence it replaces '
              'removed from the previous paragraph (method from the saved bootstrap_auc_ci.py)')
    p = find(body, 'refitted with the initially selected hyperparameters on the 763 observations')
    ED.replace(p, 'on the 763 observations', f'on the {N_TUNE} observations')
    edit(body, 'coloured by their category in Table 2', 'in Table 2', 'in Table 1', 'P', '3.10: Table 1')


# ======================================================================================= chapter 4 and 5
MV, MQ, VQ = PAIR_NAMES
X_L = 'XGBoost - Logistic'


def wf_probs(model, pair):
    return pd.read_csv(OUTPUTS / 'probs' / f'{model.lower()}_{pair.lower()}_probs.csv')


def chapter4(body, res):
    """Chapter 4 on the results of the rerun (A1-A3); every number is read from the outputs and every qualitative
    claim is asserted."""
    acc, diff, F = res['acc'], res['diff'], res['F']
    a = lambda pair, model, col: acc.loc[(pair, model), col]  # noqa: E731
    g = lambda pair, model, what: F[f'{pair}|{model}|{what}']  # noqa: E731
    sched = lambda pair, model, k: g(pair, model, 'schedule')[k]  # noqa: E731
    best_of = lambda pair, what, fn=max: fn(MODELS, key=lambda m: g(pair, m, what))  # noqa: E731
    edit(body, 'focusing on the distribution of classification errors to assesses',
         'classification errors to assesses', 'classification errors, and assesses', 'T', '4.1: "and assesses"')
    # 4.2 opening
    p = find(body, 'Table 4 reports the accuracy, ROC-AUC and log-loss of each model for the three factor pairs.')
    ED.replace(p, 'for the three factor pairs.', 'for the three factor pairs, with 95% block-bootstrap confidence '
                                                 'intervals (Section 3.9).')
    for pair in PAIR_NAMES:
        assert sorted(MODELS, key=lambda m: g(pair, m, 'roc_auc')) == ['Logistic', 'GAM', 'XGBoost'], pair
        assert best_of(pair, 'log_loss') == 'GAM'
    for pair in (MV, VQ):
        assert best_of(pair, 'accuracy') == 'XGBoost' and best_of(pair, 'log_loss', min) == 'XGBoost'
    assert best_of(MQ, 'accuracy') == 'GAM' and best_of(MQ, 'log_loss', min) == 'Logistic'
    ED.replace(p, 'XGBoost has the highest accuracy and ROC-AUC and the lowest log-loss for every pair, but the ordering '
                  'does not follow model flexibility.',
               'XGBoost has the highest ROC-AUC for every pair, and on this measure the ordering follows model '
               'flexibility.')
    ED.replace(p, 'has a lower ROC-AUC than logistic regression for two pairs and the highest log-loss for all three.',
               'ranks between logistic regression and XGBoost on ROC-AUC for every pair but has the highest log-loss '
               'for all three. XGBoost also has the highest accuracy and the lowest log-loss for momentum versus value '
               'and value versus quality, but not for momentum versus quality, where the GAM has the highest accuracy '
               'and logistic regression the lowest log-loss.')
    best = g(VQ, 'XGBoost', 'roc_auc')
    assert best == max(g(pair, m, 'roc_auc') for pair in PAIR_NAMES for m in MODELS)
    ED.replace(p, 'a ROC-AUC of 0.664 for value versus quality', f'a ROC-AUC of {f3(best)} for value versus quality')
    log('A1-A3', '4.2: model ordering and strongest ROC-AUC from the rerun (ROC-AUC now rises with flexibility for '
                 'every pair; accuracy and log-loss do not)')
    rebuild_table4(body, res)
    # confidence intervals (A4) and the CPCV evidence (A5)
    ci = lambda pair, m: f"{f3(a(pair, m, 'auc_ci_low'))} to {f3(a(pair, m, 'auc_ci_high'))}"  # noqa: E731
    neg = lambda x: f3(x).replace('-', '−')  # noqa: E731
    dci = lambda pair, c: f'{neg(diff.loc[(pair, c)].ci_low)} to {neg(diff.loc[(pair, c)].ci_high)}'  # noqa: E731
    aci = lambda pair, m: f"{100 * a(pair, m, 'acc_ci_low'):.1f}% to {100 * a(pair, m, 'acc_ci_high'):.1f}%"  # noqa
    for pair in PAIR_NAMES:
        for m in MODELS:
            assert (a(pair, m, 'auc_ci_low') > 0.5) == (pair == VQ and m == 'XGBoost'), (pair, m)
            assert a(pair, m, 'acc_ci_low') <= a(pair, m, 'majority_class_accuracy') + 1e-12, (pair, m)
    assert abs(a(VQ, 'XGBoost', 'acc_ci_low') - a(VQ, 'XGBoost', 'majority_class_accuracy')) < 1e-12
    assert all(r.ci_low < 0 < r.ci_high for _, r in diff.iterrows()) and len(diff) == 9
    assert -0.01 < diff.loc[(VQ, X_L)].ci_low < 0
    assert diff.loc[(VQ, X_L)].ci_low == max(diff.ci_low)
    cm = lambda pair, m: g(pair, m, 'cpcv_mean_auc')  # noqa: E731
    for pair in (MV, MQ):
        assert cm(pair, 'XGBoost') < min(cm(pair, 'Logistic'), cm(pair, 'GAM'))
    assert cm(VQ, 'XGBoost') > max(cm(VQ, 'Logistic'), cm(VQ, 'GAM'))
    guide = find(body, 'The subsections that follow examine, for each pair')
    paras = [
        'The intervals show that the evidence of discrimination is weak. Only for value versus quality does the '
        f'interval for the ROC-AUC of XGBoost exclude 0.5 ({ci(VQ, "XGBoost")}); for momentum versus value '
        f'({ci(MV, "XGBoost")}) and momentum versus quality ({ci(MQ, "XGBoost")}), chance-level ranking cannot be '
        'ruled out, and no other model’s interval excludes 0.5. No difference in ROC-AUC between two models is '
        'significant for any pair. The closest is the lead of XGBoost over logistic regression for value versus '
        f'quality, whose interval ({dci(VQ, X_L)}) only just includes zero; for momentum versus value and momentum '
        f'versus quality, the corresponding intervals are {dci(MV, X_L)} and {dci(MQ, X_L)}. Nor does any accuracy '
        'interval lie wholly above the majority-class rate: that of XGBoost for value versus quality '
        f'({aci(VQ, "XGBoost")}) has the rate as its lower limit.',
        'The cross-validation used for tuning did not anticipate this ranking. At the initial tuning point, the '
        'selected XGBoost settings have a mean ROC-AUC over the ten CPCV combinations of '
        f'{f3(cm(MV, "XGBoost"))} for momentum versus value and {f3(cm(MQ, "XGBoost"))} for momentum versus quality, '
        f'below logistic regression ({f3(cm(MV, "Logistic"))} and {f3(cm(MQ, "Logistic"))}) and the GAM '
        f'({f3(cm(MV, "GAM"))} and {f3(cm(MQ, "GAM"))}); XGBoost leads only for value versus quality '
        f'({f3(cm(VQ, "XGBoost"))}) (Table C.3). In the two momentum pairs, the selected XGBoost model therefore '
        'ranked last in cross-validation, yet it ranks first out of sample, which suggests that its lead in 2022 to '
        '2025 is partly specific to that period.']
    for text in paras:
        guide.addprevious(ED.new_paragraph(guide, text))
    log('A4', '4.2: confidence intervals, paired comparisons (none significant) and the majority-class baseline '
              'discussed')
    log('A5', '4.2: new paragraph on the CPCV evidence (Table C.3) not anticipating the out-of-sample ranking')

    # 4.2.1 momentum versus value
    maj = a(MV, 'XGBoost', 'majority_class_accuracy')
    xa, share = g(MV, 'XGBoost', 'accuracy'), g(MV, 'XGBoost', 'pred_pos_share')
    assert [m for m in MODELS if g(MV, m, 'roc_auc') > 0.5] == ['XGBoost'] and abs(xa - maj) < 1e-12
    assert all(g(MV, m, 'accuracy') < maj for m in ('Logistic', 'GAM')) and g(MV, 'XGBoost', 'log_loss') > np.log(2)
    assert min(g(MV, m, 'log_loss') for m in ('Logistic', 'GAM')) > g(MV, 'XGBoost', 'log_loss')
    p = find(body, "For momentum versus value, XGBoost's advantage lies in ranking")
    ED.replace(p, 'lies in ranking and classification rather than in its probabilities.',
               'lies in ranking rather than in classification or in its probabilities.')
    ED.replace(p, 'It is the only model with a ROC-AUC well above 0.5 (0.625), and it has the highest accuracy (62.8%), '
                  'consistent', f'It is the only model with a ROC-AUC above 0.5 ({f3(g(MV, "XGBoost", "roc_auc"))}), '
                                'consistent')
    ED.replace(p, 'Its log-loss (0.715), however, is above 0.693',
               f'Its accuracy ({pct(xa)}), however, equals the {pct(maj)} obtained by always predicting momentum, '
               f'because it predicts momentum in {pct(share)} of weeks, and its log-loss '
               f'({f3(g(MV, "XGBoost", "log_loss"))}) is above 0.693')
    ED.replace(p, 'Logistic regression and the GAM are close to random (ROC-AUC 0.489 and 0.528), with log-losses far '
                  'higher still (1.140 and 1.545).',
               'Logistic regression and the GAM rank the weeks no better than random (ROC-AUC '
               f'{f3(g(MV, "Logistic", "roc_auc"))} and {f3(g(MV, "GAM", "roc_auc"))}), with log-losses higher still '
               f'({f3(g(MV, "Logistic", "log_loss"))} and {f3(g(MV, "GAM", "log_loss"))}), and their accuracies '
               f'({pct(g(MV, "Logistic", "accuracy"))} and {pct(g(MV, "GAM", "accuracy"))}) are below the '
               'majority-class rate.')
    log('A1-A4', '4.2.1: momentum versus value rewritten on the rerun: XGBoost ranks best, but its accuracy only '
                 'equals the majority-class rate; logistic regression and the GAM below it')
    assert [0.3, 0.73] in g(MV, 'XGBoost', 'roc_lead_fpr')
    edit(body, 'Figure 9 shows that XGBoost achieves the highest true-positive rate', 'at false-positive rates above '
                                                                                      '0.25.',
         'at false-positive rates between about 0.30 and 0.73.', 'A1-A3', '4.2.1: Figure 9 range from the rerun')
    # Figure 10: XGBoost flat until January 2024
    pr = {m: wf_probs(m, MV) for m in MODELS}
    x = pr['XGBoost'].probability.values
    for k in (0, 1):
        assert (sched(MV, 'XGBoost', k)['learning_rate'], sched(MV, 'XGBoost', k)['reg_lambda']) == (0.01, 1000)
    assert round(x[:104].min(), 2) == 0.5 and round(x[:104].max(), 2) == 0.58 and (x[:104] >= 0.5).sum() == 102
    assert g(MV, 'XGBoost', 'period1|accuracy') < 0.5 and round(g(MV, 'XGBoost', 'period2|accuracy'), 2) == 0.73
    assert np.ptp(g(MV, 'XGBoost', 'period3|prob_range')) > 0.8
    p = find(body, "Figure 10 shows that XGBoost's probabilities are generally less extreme")
    ED.replace(p, "XGBoost's probabilities are generally less extreme than those of logistic regression and the GAM "
                  "during 2022 and 2023. Its forecasts initially favour value and subsequently momentum, remaining on "
                  "the same side of the 0.5 threshold for ex-tended intervals. This pattern indicates persistent class "
                  "decisions rather than frequent switching between the two leadership states.",
               "XGBoost's probabilities stay between 0.50 and 0.58 until January 2024, while its selected settings "
               "combine a learning rate of 0.01 with an L2 penalty of 1000 (Table C.6). It predicts momentum in all "
               "but two of these 104 weeks, wrongly in most weeks of the first period and correctly in 73% of the "
               "second, so its class decisions follow the frequency of momentum leadership rather than changes in "
               "leadership.")
    cw = {m: g(MV, m, 'confidently_wrong') for m in MODELS}
    assert cw == {'Logistic': 7, 'GAM': 23, 'XGBoost': 6}
    ED.replace(p, 'Logistic regression and the GAM more frequently approach zero or one when the realised class '
                  'contradicts the forecast, providing a visual explanation for their higher log-loss in Table 4.',
               'The GAM more frequently approaches zero or one when the realised class contradicts the forecast: it is '
               'confidently wrong in 23 weeks, against seven for logistic regression and six for XGBoost, which helps '
               'to explain its higher log-loss in Table 4.')
    log('A1-A3', '4.2.1: Figure 10 described from the rerun (XGBoost between 0.50 and 0.58 until January 2024; '
                 'confidently wrong weeks counted)')

    # 4.2.2 momentum versus quality
    tn, fp, fn, tp = g(MQ, 'XGBoost', 'confusion')
    xa, maj = g(MQ, 'XGBoost', 'accuracy'), a(MQ, 'XGBoost', 'majority_class_accuracy')
    assert xa < maj and best_of(MQ, 'accuracy', min) == 'XGBoost'
    assert all(g(MQ, m, 'log_loss') > np.log(2) for m in MODELS)
    p = find(body, "For momentum versus quality, XGBoost's higher aggregate accuracy conceals")
    ED.replace(p, ED.text(p),
               'For momentum versus quality, the three measures disagree on the best model. XGBoost ranks the weeks '
               f"best, with a ROC-AUC of {f3(g(MQ, 'XGBoost', 'roc_auc'))} against {f3(g(MQ, 'GAM', 'roc_auc'))} for "
               f"the GAM and {f3(g(MQ, 'Logistic', 'roc_auc'))} for logistic regression, but its accuracy of "
               f'{pct(xa)} is the lowest of the three and below the {pct(maj)} obtained by always predicting momentum: '
               f'at the 0.5 threshold it correctly identifies only {tp} of the {tp + fn} momentum-led weeks and {tn} of '
               f"the {tn + fp} quality-led weeks. The GAM has the highest accuracy ({pct(g(MQ, 'GAM', 'accuracy'))}) "
               f"and logistic regression the lowest log-loss ({f3(g(MQ, 'Logistic', 'log_loss'))}), although all "
               'three log-losses exceed 0.693.')
    log('A1-A3', '4.2.2: momentum versus quality rewritten on the rerun (XGBoost ranks best but classifies worst)')
    lead = {m: g(MQ, m, 'roc_lead_fpr') for m in MODELS}
    assert all(0.12 <= lo and hi <= 0.30 for lo, hi in lead['Logistic']) and [0.2, 0.3] in lead['Logistic']
    grid = np.round(np.arange(0.31, 0.451, 0.01), 2)
    gam_led = [any(lo - 1e-9 <= v <= hi + 1e-9 for lo, hi in lead['GAM']) for v in grid]
    assert np.mean(gam_led) > 0.7
    assert [0.47, 0.73] in lead['XGBoost'] and [0.75, 0.84] in lead['XGBoost']
    p = find(body, 'Figure 11 shows that logistic regression generally achieves')
    ED.replace(p, 'logistic regression generally achieves the highest true-positive rate at false-positive rates below '
                  'approximately 0.55, whereas XGBoost leads at higher false-positive rates.',
               'logistic regression and the GAM achieve the highest true-positive rates at false-positive rates below '
               'approximately 0.45, logistic regression mainly between 0.12 and 0.30 and the GAM above that, whereas '
               'XGBoost leads at most false-positive rates between 0.47 and 0.84.')
    ED.replace(p, 'so neither model dominates', 'so no model dominates')
    ED.replace(p, 'The relative performance of the two models', 'The relative performance of the models')
    ED.insert(p, 'incorrectly classifying quality-led weeks', '.')
    log('A1-A3', '4.2.2: Figure 11 described from the rerun (three-way crossing); full stop before Figure 12')
    # Figure 12
    y = wf_probs('XGBoost', MQ).actual.values
    real = [y[52 * j:52 * (j + 1)].mean() for j in range(3)]
    share = lambda m, j: g(MQ, m, f'period{j}|pred_pos_share')  # noqa: E731
    assert max(share(m, 1) for m in MODELS) <= 0.08 and round(real[0], 2) == 0.46
    assert round(real[1], 2) == 0.73 and round(share('XGBoost', 2), 2) == 0.73
    assert round(g(MQ, 'XGBoost', 'period2|accuracy'), 2) == 0.50 and max(share(m, 2) for m in MODELS[:2]) < 0.5
    s3 = [share(m, 3) for m in MODELS]
    assert (round(min(s3), 2), round(max(s3), 2), round(real[2], 2)) == (0.62, 0.77, 0.44)
    cwp = {}
    for m in MODELS:
        d = wf_probs(m, MQ)
        yy, xx = d.actual.values, d.probability.values
        wrong = ((xx > 0.9) & (yy == 0)) | ((xx < 0.1) & (yy == 1))
        cwp[m] = (int(wrong.sum()), int(wrong[:52].sum()))
    assert cwp == {'Logistic': (2, 0), 'GAM': (9, 7), 'XGBoost': (9, 7)}
    p = find(body, 'Figure 12 shows the source of the asymmetry')
    ED.replace(p, ED.text(p),
               'Figure 12 shows that all three models favour quality until January 2023, predicting momentum in at '
               'most 8% of those weeks, although momentum leads in 46% of them. In the second period, momentum leads '
               'in 73% of weeks; XGBoost predicts momentum in the same share of weeks, yet its accuracy in the period is '
               'only 50%, while logistic regression and the GAM predict momentum in fewer than half of the weeks. In the third, all three '
               'favour momentum, in 62% to 77% of weeks, although momentum leads in only 44%. The GAM and XGBoost are '
               'each confidently wrong in nine weeks, seven of them in the first period, and logistic regression in '
               'two.')
    log('A1-A3/T', '4.2.2: Figure 12 described from the rerun (the fragment "Until January 2024." goes with the old '
                   'text)')

    # 4.2.3 value versus quality
    tn, fp, fn, tp = g(VQ, 'XGBoost', 'confusion')
    assert best_of(VQ, 'accuracy') == 'XGBoost' and best_of(VQ, 'log_loss', min) == 'XGBoost'
    assert g(VQ, 'XGBoost', 'log_loss') > np.log(2) and abs(g(VQ, 'Logistic', 'roc_auc') - 0.5) < 0.02
    assert pct(g(VQ, 'XGBoost', 'accuracy')) == '60.3%'             # as the thesis states
    p = find(body, 'Value versus quality provides the strongest evidence of discrimination in the study')
    ED.replace(p, 'ROC-AUC (0.664)', f'ROC-AUC ({f3(best)})')
    ED.replace(p, 'log-loss (0.670)', f"log-loss ({f3(g(VQ, 'XGBoost', 'log_loss'))})")
    ED.replace(p, 'correctly identifies 45 of the 77 value-led weeks and 49 of the 79 quality-led weeks',
               f'correctly identifies {tp} of the {tp + fn} value-led weeks and {tn} of the {tn + fp} quality-led weeks')
    ED.replace(p, 'Logistic regression ranks second on ROC-AUC, at 0.598, while the GAM performs approximately at chance '
                  'level, with a ROC-AUC of 0.495.',
               f"The GAM ranks second on ROC-AUC, at {f3(g(VQ, 'GAM', 'roc_auc'))}, while logistic regression performs "
               f"approximately at chance level, with a ROC-AUC of {f3(g(VQ, 'Logistic', 'roc_auc'))}.")
    ED.replace(p, r'XGBoost.s advantage therefore extends across classification accuracy, ranking and probability '
                  r'quality\.',
               'XGBoost’s advantage therefore extends across all three measures, although its log-loss remains above '
               '0.693.', regex=True)
    log('A1-A3', '4.2.3: value versus quality from the rerun (the GAM now second on ROC-AUC; log-loss above 0.693)')
    lead = g(VQ, 'XGBoost', 'roc_lead_fpr')
    assert [0.14, 0.78] in lead
    fpr, tpr = roc_curve(wf_probs('XGBoost', VQ).actual, wf_probs('XGBoost', VQ).probability)[:2]
    assert (tpr >= fpr - 1e-12).all(), 'Figure 13: the XGBoost curve must stay on or above the diagonal'
    edit(body, 'Figure 13 shows that XGBoost achieves the highest true-positive rate', 'between approximately 0.08',
         'between approximately 0.14', 'A1-A3', '4.2.3: Figure 13 range from the rerun')
    edit(body, 'Figure 13 shows that XGBoost achieves the highest true-positive rate',
         'Logistic regression also displays some discrimination, consistent with its overall ROC-AUC of 0.598 in Table '
         '4, while the GAM curve remains close to the diagonal.',
         f"The GAM also displays some discrimination, consistent with its overall ROC-AUC of "
         f"{f3(g(VQ, 'GAM', 'roc_auc'))} in Table 4, while the logistic regression curve remains close to the "
         'diagonal.', 'A1-A3', '4.2.3: Figure 13: the GAM, not logistic regression, now second')
    edit(body, 'The figure therefore sup-ports', 'sup-ports', 'supports', 'T', '4.2.3: "supports"')
    p = find(body, 'Figure 13: Out-of-sample ROC curves for value versus quality')
    ED.replace(p, 'Figure 13: Out-of-sample', 'Out-of-sample')
    log('P', 'Figure 13 caption: duplicated "Figure 13:" deleted')
    # Figure 14
    d = wf_probs('GAM', VQ)
    gam_low = int(((d.probability < 0.1) & (d.actual == 1)).sum())
    assert gam_low == 18 and all(0 < g(VQ, 'XGBoost', f'period{j}|pred_pos_share') < 1 for j in (1, 2, 3))
    edit(body, 'Figure 14 shows that XGBoost', 'sometimes assigns probabilities close to zero when value subsequently '
                                               'leads',
         f'assigns probabilities below 0.1 in {gam_low} weeks in which value subsequently leads', 'A1-A3',
         '4.2.3: Figure 14: number of weeks in which the GAM was confidently wrong about value')

    # 4.3 confusion matrices
    p = find(body, 'momentum leadership is slightly more common in the two momentum pairs')
    ED.insert(p, 'slightly more common in the two momentum pairs.', ' The two momentum pairs have identical class '
                                                                    'counts, although their labels differ in 34 of '
                                                                    'the 156 weeks.')
    log('A9', '4.3: identical class counts of the two momentum pairs confirmed (labels differ in 34 weeks)')
    t6 = find(body, 'Confusion-matrix diagnostics').getnext()
    assert local(t6) == 'tbl'
    rows = [[None, None, *map(str, g(pair, m, 'confusion')), pct(g(pair, m, 'pred_pos_share')), None]
            for pair in PAIR_NAMES for m in MODELS]
    n = update_table(ED, t6, rows)
    log('A1-A3', f'Table 6: {n} cells updated from the rerun')
    cf = {m: g(MV, m, 'confusion') for m in MODELS}
    share = {m: g(MV, m, 'pred_pos_share') for m in MODELS}
    assert cf['XGBoost'][1] > max(cf['Logistic'][1], cf['GAM'][1]) and min(share.values()) > 0.545
    assert cf['Logistic'][3] == cf['GAM'][3] > cf['Logistic'][0]
    p = find(body, 'For momentum versus value, XGBoost produces fewer false positives and false negatives')
    tn, fp, fn, tp = cf['XGBoost']
    ED.replace(p, ED.text(p),
               'For momentum versus value, XGBoost displays the strongest asymmetry between the two classes. It '
               f'correctly identifies {tp} of the {tp + fn} momentum-led weeks but only {tn} of the {tn + fp} value-led '
               f'weeks, with {fp} false positives. Its predicted-positive share of {pct(share["XGBoost"])} '
               f'substantially exceeds the realised {pct((tp + fn) / 156)}, showing that its class decisions favour '
               f'momentum. Logistic regression and the GAM identify more value-led weeks, at {cf["Logistic"][0]} and '
               f'{cf["GAM"][0]}, respectively, but fewer momentum-led weeks, {cf["GAM"][3]} each. All three models '
               'predict momentum leadership more frequently than its realised share, indicating a common tendency '
               'towards the positive class.')
    cf = {m: g(MQ, m, 'confusion') for m in MODELS}
    share = {m: g(MQ, m, 'pred_pos_share') for m in MODELS}
    assert max(share.values()) < 0.545 and all(2 * cf[m][3] < 85 for m in MODELS)
    assert cf['XGBoost'][3] == cf['Logistic'][3] == cf['GAM'][3] + 1
    assert max(MODELS, key=lambda m: cf[m][0]) == 'GAM' and min(MODELS, key=lambda m: cf[m][0]) == 'XGBoost'
    p = find(body, 'For momentum versus quality, XGBoost displays the strongest asymmetry')
    ED.replace(p, ED.text(p),
               'For momentum versus quality, all three models predict momentum leadership less often than its '
               f'realised share of 54.5%, at {pct(min(share.values()))} to {pct(max(share.values()))} of weeks, and '
               f'each identifies fewer than half of the 85 momentum-led weeks: {cf["XGBoost"][3]} for logistic '
               f'regression and XGBoost and {cf["GAM"][3]} for the GAM. The models therefore differ mainly in the '
               f'quality-led weeks. The GAM identifies {cf["GAM"][0]} of the 71, which gives it the highest accuracy '
               f'for this pair, and XGBoost {cf["XGBoost"][0]}, with {cf["XGBoost"][1]} false positives.')
    cf = {m: g(VQ, m, 'confusion') for m in MODELS}
    share = {m: g(VQ, m, 'pred_pos_share') for m in MODELS}
    assert all(share[m] < 77 / 156 and cf[m][2] > cf[m][1] for m in ('Logistic', 'GAM'))
    assert abs(cf['XGBoost'][1] - cf['XGBoost'][2]) < min(abs(cf[m][1] - cf[m][2]) for m in ('Logistic', 'GAM'))
    p = find(body, 'For value versus quality, logistic regression and the GAM favour quality leadership')
    ED.replace(p, '34.0% and 41.0%', f'{pct(share["Logistic"])} and {pct(share["GAM"])}')
    ED.replace(p, 'Logistic regression correctly identifies 55 quality-led weeks but only 29 value-led weeks, while the '
                  'corresponding counts for the GAM are 46 and 31.',
               f'Logistic regression correctly identifies {cf["Logistic"][0]} quality-led weeks but only '
               f'{cf["Logistic"][3]} value-led weeks, while the corresponding counts for the GAM are {cf["GAM"][0]} and '
               f'{cf["GAM"][3]}.')
    tn, fp, fn, tp = cf['XGBoost']
    ED.replace(p, 'XGBoost identifies 49 quality-led and 45 value-led weeks, with similar numbers of false positives and '
                  'false negatives, at 30 and 32.',
               f'XGBoost identifies {tn} quality-led and {tp} value-led weeks, with {fp} false positives and {fn} false '
               'negatives.')
    log('A1-A3', '4.3: confusion-matrix paragraphs rewritten on the rerun (momentum versus quality: every model now '
                 'under-predicts momentum)')
    # A9: the high log-losses
    gs = lambda pair, k: (sched(pair, 'GAM', k)['lam'], sched(pair, 'GAM', k)['n_splines'])  # noqa: E731
    assert [gs(MV, k) for k in range(3)] == [(1, 7), (1, 10), (10, 10)]
    assert [gs(VQ, k) for k in range(3)] == [(1, 5), (1, 10), (1, 7)]
    assert all(sched(MV, 'Logistic', k)['lambda_reg'] == 0.01 for k in range(3))
    assert sched(MV, 'GAM', 0)['n_train'] == N_TUNE and N_TUNE + 155 == 901 and N_FEAT * 10 == 470
    top3 = sorted(((g(pair, m, 'log_loss'), pair, m) for pair in PAIR_NAMES for m in MODELS), reverse=True)[:3]
    assert [(pair, m) for _, pair, m in top3] == [(MV, 'GAM'), (VQ, 'GAM'), (MV, 'Logistic')]
    assert (g(MV, 'GAM', 'confidently_wrong'), g(VQ, 'GAM', 'confidently_wrong')) == (23, 19)
    gam_fixed = {pair: pd.read_csv(OUTPUTS / 'sensitivity' / f'gam_{pair.lower()}_fixed.csv') for pair in PAIR_NAMES}
    assert all(f[f.failed.isna()].log_loss.min() > 0.8 for f in gam_fixed.values())
    assert all(f[f.lam == 10000].log_loss.min() > 0.8 for f in gam_fixed.values())
    last = find(body, 'For value versus quality, logistic regression and the GAM favour quality leadership')
    last.addnext(ED.new_paragraph(last, [
        'The highest log-losses, those of the GAM for momentum versus value '
        f'({f3(g(MV, "GAM", "log_loss"))}) and value versus quality ({f3(g(VQ, "GAM", "log_loss"))}) and of logistic '
        f'regression for momentum versus value ({f3(g(MV, "Logistic", "log_loss"))}), coincide with weakly penalised '
        'selections. The GAM was tuned to the smallest penalty (λ = 1) at every tuning point for value versus quality '
        'and at the first two for momentum versus value, with five to ten splines per predictor (Table C.6): up to '
        f'470 spline coefficients estimated from {N_TUNE} to 901 observations. Logistic regression for momentum versus '
        'value was tuned to λ = 0.01 (C = 100) at every tuning point, which leaves it almost unpenalised. Such fits '
        'follow noise in the training window and give confident out-of-sample probabilities: the GAM is confidently '
        'wrong in 23 weeks for momentum versus value and in 19 for value versus quality. A heavier penalty does not '
        'resolve this for the GAM, because the penalty acts on the curvature of each fitted function and leaves its '
        'linear part unpenalised ', CITE['Has86'], ': with every setting held fixed, its log-loss exceeds 0.80 for '
        'every pair, even at λ = 10,000 (Appendix C.5). Its high log-losses therefore reflect the absence of shrinkage '
        'towards a constant forecast rather than its nonlinear effects as such.']))
    log('A9', '4.3: the high GAM and logistic log-losses diagnosed from the selected settings and the fixed-setting '
              'GAM runs (the GAM penalty leaves the linear part unpenalised)')

    # 4.4 SHAP
    relvol = {(pair, m): g(pair, m, 'shap_relvol_ranks') for pair in PAIR_NAMES for m in MODELS}
    assert [k for k, v in relvol.items() if not v] == [(MQ, 'Logistic')]
    top = lambda pair, m: g(pair, m, 'shap_top10')  # noqa: E731
    flagged = {(pair, m) for pair in PAIR_NAMES for m in MODELS if any(f.endswith('_missing') for f in top(pair, m))}
    assert flagged == {(MQ, 'Logistic'), (VQ, 'Logistic'), (VQ, 'GAM')}
    p = find(body, 'present the ten features with the largest global mean absolute SHAP values')
    ED.replace(p, 'Figures 15-17', 'Figures 15–17')
    ED.insert(p, 'for each model.', ' A relative-volatility measure, either the relative volatility or the volatility '
                                    'ratio of a pair, ranks among the ten leading features of every model for every '
                                    'pair except logistic regression for momentum versus quality.')
    ED.append(p, ' The missing-value indicators of Section 3.4 also appear in three rankings. They equal one only in '
                 'weeks before April 2010, so a large attribution shows that a reference fit treats those early weeks '
                 'of the tuning sample differently from later ones; in the evaluation block they are zero in every '
                 'week and cannot distinguish one week from another.')
    log('A1/A9/T', '4.4: "Figures 15–17"; the abstract’s relative-volatility claim stated (verified in the SHAP outputs '
                   'of the rerun); the missing-value indicators in the rankings explained')
    tops = {m: top(MV, m)[0] for m in MODELS}
    assert tops == {'XGBoost': 'relvol_momentum_vs_value_12', 'Logistic': 'relvol_value_vs_quality_12',
                    'GAM': 'relvol_momentum_vs_quality_12'}
    shared = {'ZAR18M.Curncy_QS', 'USDZAR_EntDyn', 'SA_RB_Curvature'}
    assert shared <= set(top(MV, 'Logistic')[:6]) & set(top(MV, 'GAM')[:6])
    assert top(MV, 'XGBoost')[2:6] == ['VolTermStructPC1', 'USDZARV1M.Curncy_QS', 'corr_momentum_quality_12',
                                      'volratio_value_quality_12']
    sums = {pair: g(pair, 'XGBoost', 'shap_top10_sum') for pair in PAIR_NAMES}
    assert sums[MV] < 0.1 * min(sums[MQ], sums[VQ])
    s0 = sched(MV, 'XGBoost', 0)
    assert (s0['learning_rate'], s0['reg_lambda']) == (0.01, 1000)
    p = find(body, 'reference fits for momentum versus value, although the specific measure differs')
    ED.replace(p, '; momentum-value relative volatility for XGBoost, value-quality relative volatility for logistic '
                  'regression and momentum-quality relative volatility for the GAM.',
               ': momentum–value relative volatility for XGBoost, value–quality relative volatility for logistic '
               'regression and momentum–quality relative volatility for the GAM.')
    ED.replace(p, 'and rand PPP valuation among their six leading features',
               'and South African real-yield-curve curvature among their six leading features')
    ED.replace(p, 'momentum versus quality correlation, South African real-yield-curve curvature, the value–quality '
                  'volatility ratio, global bond volatility and the commodity index component.',
               'the level of the US equity volatility term structure, short-dated USDZAR implied volatility, momentum '
               'versus quality correlation and the value–quality volatility ratio.')
    ED.append(p, " XGBoost's ten largest SHAP magnitudes for this pair sum to less than a tenth of the corresponding "
                 'totals for the other pairs. This follows from its reference setting: a learning rate of 0.01 and an '
                 'L2 penalty of 1000 (Table C.6) shrink the leaf weights until the fitted model is close to a constant, '
                 'as its flat walk-forward probabilities in the first two periods also show (Figure 10).')
    log('A1-A3', '4.4.1: SHAP rankings from the rerun; the small XGBoost SHAP magnitudes (now momentum versus value) '
                 'explained by η = 0.01 and λ = 1000; colon and en dashes')
    lt, xt, gt = top(MQ, 'Logistic'), top(MQ, 'XGBoost'), top(MQ, 'GAM')
    assert len(set(lt) & set(xt)) == 6 and {'MichgnConcIndx', 'GlobalBondTS'} <= set(lt) & set(xt)
    assert xt[0] == lt[2] == 'TOP40.Index.Price.to.Ebitda_Diff' and lt[0] == 'ZAR18M.Curncy_QS_missing'
    assert f3(g(MQ, 'XGBoost', 'roc_auc') - g(MQ, 'Logistic', 'roc_auc')) == '0.030'
    assert gt[0] == 'corr_value_quality_12' and gt[1].startswith('relvol_')
    assert gt.index('TOP40.Index.Price.to.Ebitda_Diff') == 9
    p = find(body, 'For momentum versus quality, logistic regression and XGBoost share the same two leading features')
    ED.replace(p, 'share the same two leading features: the change in the TOP40 price-to-EBITDA ratio and South African '
                  'real-yield-curve curvature. The commodity index component also appears among the four leading '
                  'features of both models.',
               'share six of their ten leading features, among them the change in the TOP40 price-to-EBITDA ratio, '
               'which ranks first for XGBoost and third for logistic regression, US consumer sentiment and the trend '
               'of the global bond index, although logistic regression ranks the missing-value indicator of '
               'medium-term USDZAR forward pricing above all three.')
    ED.replace(p, 'a difference in ROC-AUC of only 0.019', 'a difference in ROC-AUC of only 0.030')
    ED.replace(p, r'ranked seventh\. XGBoost.s ten largest SHAP magnitudes sum to less than one sixth of the '
                  r'corresponding totals for the other pairs\.', 'ranked tenth.', regex=True)
    log('A1-A3', '4.4.2: SHAP rankings of momentum versus quality from the rerun')
    lt, xt, gtv = top(VQ, 'Logistic'), top(VQ, 'XGBoost'), top(VQ, 'GAM')
    assert xt[0] == lt[0] == gtv[4] == 'mom3m_value_vs_quality'
    assert 'mom3m_momentum_vs_quality' in set(xt) & set(lt) and 'mom3m_momentum_vs_quality' not in gtv
    vr = 'volratio_value_quality_12'
    assert (gtv.index(vr), xt.index(vr), vr in lt) == (2, 8, False)
    assert {'USDZAR_EntDyn', 'GlobalEquityGret3', 'SA_CDS_Curvature_missing'} <= set(lt) & set(gtv)
    assert {'GLD_RollSD', 'US_RB_Curvature', 'SA_RB_Curvature'} <= set(xt[:5])
    p = find(body, 'For value versus quality, recent relative performance is prominent')
    ED.replace(p, 'in the models with the strongest overall discrimination', 'in all three models')
    ED.replace(p, 'ranks first for XGBoost and second for logistic regression, while the corresponding momentum–quality '
                  'measure appears among the five leading features of both.',
               'ranks first for XGBoost and logistic regression and fifth for the GAM, while the corresponding '
               'momentum–quality measure appears among the ten leading features of XGBoost and logistic regression.')
    ED.replace(p, 'The value–quality volatility ratio ranks among the four leading features in all three models.',
               'The value–quality volatility ratio ranks third for the GAM and ninth for XGBoost.')
    ED.replace(p, 'USDZAR entropy and global property volatility',
               'USDZAR entropy, the three-month return of global equities and the missing-value indicator of South '
               'African CDS-curve curvature')
    log('A1-A3', '4.4.3: SHAP rankings of value versus quality from the rerun')


def results_figures(body, rels_root, media_dir):
    """Figures 9-17 redrawn from the rerun (build_chapter4_corrected_figures.py), as tracked picture swaps."""
    files = ['roc_momentum_vs_value', 'probabilities_momentum_vs_value', 'roc_momentum_vs_quality',
             'probabilities_momentum_vs_quality', 'roc_value_vs_quality', 'probabilities_value_vs_quality',
             'shap_momentum_vs_value', 'shap_momentum_vs_quality', 'shap_value_vs_quality']
    for k, name in enumerate(files):
        swap_picture(ED, body, rels_root, media_dir, f'image{10 + k}.png', FIGS / f'figure_final_{name}.png',
                     f'rerun_figure_{9 + k}.png', 6001 + k)
    log('A3', 'Figures 9-17 redrawn from the rerun (old pictures kept as tracked deletions)')
    swap_picture(ED, body, rels_root, media_dir, 'image7.png', FIGS / 'figure_6_dendrogram.png',
                 'selection_figure_6.png', 6000)
    log('A2', f'Figure 6: dendrogram of the selection on the weeks before the evaluation block ({SEL["clustered"]} '
              f'features, {N_REPS} clusters at cut height 2)')


def rebuild_table4(body, res):
    acc, metrics = res['acc'], res['metrics']
    old = find(body, 'Out-of-sample accuracy, ROC-AUC and log-loss by factor pair and model').getnext()
    assert local(old) == 'tbl' and 'Log-loss' in ''.join(old.itertext())
    new = copy.deepcopy(old)
    strip_ids(new)
    rows = new.findall(w('tr'))
    header, body_rows = rows[0], rows[1:]
    assert len(body_rows) == 9
    widths = [1500, 1050, 1000, 2000, 2250, 1200]
    grid = new.find(w('tblGrid'))
    for g in list(grid):
        grid.remove(g)
    for wd in widths:
        etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
    names = {'Momentum_vs_Value': 'Momentum vs value', 'Momentum_vs_Quality': 'Momentum vs quality',
             'Value_vs_Quality': 'Value vs quality'}
    cells = [['Factor pair', 'Majority class', 'Model', 'Accuracy, % (95% CI)', 'ROC-AUC (95% CI)', 'Log-loss']]
    bold = []
    for pair in names:
        sub = acc.loc[pair]
        ll = metrics.loc[pair, 'log_loss']
        best = {3: sub.accuracy.idxmax(), 4: sub.roc_auc.idxmax(), 5: ll.idxmin()}   # bold: best of each measure
        assert best[4] == 'XGBoost'
        for i, model in enumerate(('Logistic', 'GAM', 'XGBoost')):
            r = sub.loc[model]
            cells.append([names[pair] if i == 0 else '', f'{r.majority_class_accuracy:.1%}' if i == 0 else '',
                          model, f'{100 * r.accuracy:.1f} ({100 * r.acc_ci_low:.1f}\u2013{100 * r.acc_ci_high:.1f})',
                          f'{r.roc_auc:.3f} ({r.auc_ci_low:.3f}\u2013{r.auc_ci_high:.3f})',
                          f'{metrics.loc[(pair, model), "log_loss"]:.3f}'])
            bold.append({k for k, m in best.items() if m == model})

    def fill(tr, values, is_bold):
        tcs = tr.findall(w('tc'))
        while len(tcs) < len(values):
            tcs[-1].addnext(copy.deepcopy(tcs[-1]))
            tcs = tr.findall(w('tc'))
        for tc, v, wd, k in zip(tcs, values, widths, range(len(values))):
            tcw = tc.find('w:tcPr/w:tcW', NS)
            if tcw is not None:
                tcw.set(w('w'), str(wd))
                tcw.set(w('type'), 'dxa')
            ps = tc.findall(w('p'))
            for extra in ps[1:]:
                tc.remove(extra)
            pp = ps[0]
            rpr = None
            for r in pp.iter(w('r')):
                if r.find(w('rPr')) is not None:
                    rpr = copy.deepcopy(r.find(w('rPr')))
                    break
            for c in list(pp):
                if local(c) != 'pPr':
                    pp.remove(c)
            if rpr is not None:
                for t in ('b', 'bCs'):
                    for x in rpr.findall(w(t)):
                        rpr.remove(x)
            if v:
                make_bold = (tr is header) or k in is_bold
                pp.append(ED.make_run(v, rpr, bold=make_bold))
    fill(header, cells[0], True)
    for tr, values, is_bold in zip(body_rows, cells[1:], bold):
        fill(tr, values, is_bold)
    tw = new.find('w:tblPr/w:tblW', NS)
    if tw is not None:
        tw.set(w('w'), str(sum(widths)))
        tw.set(w('type'), 'dxa')
    for tr in rows[:-1]:                          # keep the table on one page
        for pp in tr.iter(w('p')):
            ppr = pp.find(w('pPr'))
            if ppr is None:
                ppr = etree.SubElement(pp, w('pPr'))
                pp.insert(0, ppr)
            if ppr.find(w('keepNext')) is None:
                st = ppr.find(w('pStyle'))
                ppr.insert(0 if st is None else 1, etree.Element(w('keepNext')))
    ED.mark_table(new, 'ins')
    old.addnext(new)
    ED.mark_table(old, 'del')
    note_like = find(body, 'Note: grids are searched exhaustively')
    note = ED.new_paragraph(note_like, [Italic('Note: confidence intervals from a circular moving-block bootstrap of '
                                               'the 156 weeks (blocks of 12 weeks, 10,000 resamples; Section 3.9). The '
                                               'majority class is the accuracy of always predicting the more frequent '
                                               'class. Bold marks the best value of each measure for each pair.')])
    new.addnext(note)
    log('A3/A4', 'Table 4 rebuilt from the rerun with 95% confidence intervals for accuracy and ROC-AUC and a '
                 'majority-class column (analysis/a4_accuracy_ci.csv, metrics_summary.csv); old table kept as a '
                 'tracked deletion')


def fixed_settings():
    """The walk-forward results of every grid setting held fixed (Appendix C.5): the numbers the C.5 and Chapter 5
    text quote, with its claims asserted."""
    fx = {k: {kk: np.array(v) for kk, v in d.items()} for k, d in APPX['fixed'].items()}
    auc = lambda pair, m: fx[f'{pair}|{m}']['auc']  # noqa: E731
    ll = lambda pair, m: fx[f'{pair}|{m}']['ll']  # noqa: E731
    pc = lambda x: f'{100 * x:.0f}%'  # noqa: E731
    sel = lambda pair, m: APPX['sel_auc'][f'{pair}|{m}']  # noqa: E731
    sel_ll = lambda pair, m: APPX['sel_ll'][f'{pair}|{m}']  # noqa: E731
    below_sel = lambda pair, m: np.mean(auc(pair, m) < sel(pair, m))  # noqa: E731
    best_alt = {pair: max(auc(pair, 'Logistic').max(), auc(pair, 'GAM').max()) for pair in PAIR_NAMES}
    above_alt = {pair: np.mean(auc(pair, 'XGBoost') > best_alt[pair]) for pair in PAIR_NAMES}
    under = lambda pair, m: np.mean(ll(pair, m) < np.log(2))  # noqa: E731
    sizes = {(pair, m): len(auc(pair, m)) for pair in PAIR_NAMES for m in MODELS}
    assert sizes == {(pair, m): (19 if (pair, m) == (MQ, 'GAM') else {'Logistic': 18, 'GAM': 20, 'XGBoost': 216}[m])
                     for pair in PAIR_NAMES for m in MODELS}
    assert APPX['period_reproduced']
    n_vq, n_mv = int((auc(VQ, 'XGBoost') > 0.5).sum()), int((auc(MV, 'XGBoost') > 0.5).sum())
    assert (n_vq, n_mv) == (216, 215)
    lmax, xmax, gmax = auc(MQ, 'Logistic').max(), auc(MQ, 'XGBoost').max(), auc(MQ, 'GAM').max()
    assert lmax > xmax
    gam_beats = np.mean(auc(MQ, 'XGBoost') < gmax)
    assert all(under(pair, 'GAM') == 0 and ll(pair, 'GAM').min() > 0.8 for pair in PAIR_NAMES)
    lshare = sorted(under(pair, 'Logistic') for pair in PAIR_NAMES)
    gam_up = {pair: below_sel(pair, 'GAM') for pair in PAIR_NAMES}
    assert min(gam_up.values()) > 0.75
    assert all(sched_l['lambda_reg'] == 0.01 for sched_l in APPX['schedule'][f'{MV}|Logistic'])
    assert pd.read_csv(OUTPUTS / 'sensitivity' / 'logistic_momentum_vs_value_fixed.csv').lambda_reg.min() == 0.01
    n_mv_log = int((auc(MV, 'Logistic') < 0.5).sum())
    assert f3(sel_ll(MV, 'Logistic')) == f3(ll(MV, 'Logistic').max())
    text = {
        'intro': 'In each 52-week period, the fixed run of the setting selected for that period reproduces Chapter '
                 '4’s predictions exactly.',
        'xgb': f'XGBoost’s ranking advantage is broadly stable for value–quality and momentum–value: all {n_vq} fixed '
               f'settings exceed 0.5 for the former and {n_mv} do so for the latter, and {pc(above_alt[VQ])} and '
               f'{pc(above_alt[MV])}, respectively, also exceed the best fixed logistic and GAM results. For '
               f'momentum–quality, it is not: the best fixed logistic setting ({f3(lmax)}) exceeds every XGBoost '
               f'setting, and the best GAM setting ({f3(gmax)}) exceeds {pc(gam_beats)} of them. XGBoost settings '
               f'achieve log-loss below 0.693 in {pc(under(VQ, "XGBoost"))} of value–quality, '
               f'{pc(under(MQ, "XGBoost"))} of momentum–quality and {pc(under(MV, "XGBoost"))} of momentum–value runs, '
               f'as do {pc(lshare[0])} to {pc(lshare[-1])} of logistic settings per pair; no GAM setting does.',
        'gam': f'The GAM’s selected ROC-AUCs exceed most of its fixed-setting results: {pc(gam_up[VQ])} for '
               f'value–quality, {pc(gam_up[MQ])} for momentum–quality and {pc(gam_up[MV])} for momentum–value, whose '
               f'fixed settings all lie between {f3(auc(MV, "GAM").min())} and {f3(auc(MV, "GAM").max())}. Its '
               'log-loss exceeds 0.80 for every setting and pair, even at λ = 10,000, where the penalty leaves each '
               'fitted function almost linear (Section 4.3).',
        'logistic': f'For momentum–value, {n_mv_log} of 18 logistic settings score below 0.5. CPCV selects the weakest '
                    'penalty (λ = 0.01) at every tuning point, yielding a log-loss of '
                    f'{f3(sel_ll(MV, "Logistic"))}, equal to the worst fixed result and above the median of '
                    f'{f3(np.median(ll(MV, "Logistic")))}. For momentum–quality, the fixed settings range from '
                    f'{f3(auc(MQ, "Logistic").min())} to {f3(lmax)}, and the selected result exceeds '
                    f'{pc(below_sel(MQ, "Logistic"))} of them; for value–quality, it exceeds '
                    f'{pc(below_sel(VQ, "Logistic"))}.'}
    return dict(text=text, above_alt=above_alt, logistic_beats_all_mq=lmax > xmax)


def chapter5(body, res):
    F = res['F']
    g = lambda pair, model, what: F[f'{pair}|{model}|{what}']  # noqa: E731
    xauc = [g(pair, 'XGBoost', 'roc_auc') for pair in PAIR_NAMES]
    xacc = [g(pair, 'XGBoost', 'accuracy') for pair in PAIR_NAMES]
    p = find(body, 'relative factor leadership in the South African equity market is partly predictable out of sample')
    ED.replace(p, 'First, relative factor leadership in the South African equity market is partly predictable out of '
                  'sample, but the predictable component is modest.',
               'First, the evidence that relative factor leadership in the South African equity market is predictable '
               'out of sample is weak.')
    ED.replace(p, 'reaches ROC-AUCs between 0.589 and 0.664 and accuracies between 60.3% and 62.8% across the three '
                  'pairs (Table 4).',
               f'reaches ROC-AUCs between {f3(min(xauc))} and {f3(max(xauc))} and accuracies between {pct(min(xacc))} '
               f'and {pct(max(xacc))} across the three pairs (Table 4), but its 95% bootstrap interval for ROC-AUC '
               'excludes 0.5 only for value versus quality (Section 4.2).')
    ED.replace(p, 'relative leadership can be predicted, but only to a limited degree.',
               'relative leadership is predictable to a limited degree, and significantly so only for value versus '
               'quality.')
    log('A1-A4', '5.1 first finding: ranges from the rerun, reworded to the confidence intervals ("weak evidence, '
                 'significant only for value versus quality")')
    # second finding: ROC-AUC follows flexibility, not significantly; the GAM's probabilities; fixed settings
    for pair in PAIR_NAMES:
        assert sorted(MODELS, key=lambda m: g(pair, m, 'roc_auc')) == ['Logistic', 'GAM', 'XGBoost']
        assert max(MODELS, key=lambda m: g(pair, m, 'log_loss')) == 'GAM'
    assert min(MODELS, key=lambda m: g(MQ, m, 'accuracy')) == 'XGBoost'
    assert all(r.ci_low < 0 < r.ci_high for _, r in res['diff'].iterrows())
    fs = fixed_settings()
    p = find(body, 'Second, greater model flexibility does not improve predictive performance monotonically')
    ED.replace(p, 'XGBoost has the highest accuracy and ROC-AUC and the lowest log-loss for every pair, but the GAM ranks '
                  'below logistic regression on ROC-AUC for two of the three pairs and has the highest log-loss for all '
                  'three (Table 4).',
               'ROC-AUC rises from logistic regression to the GAM to XGBoost for every pair, though no difference '
               'between two models is statistically significant (Section 4.2). The other measures do not follow this '
               'ordering: the GAM has the highest log-loss for all three pairs, and XGBoost has the lowest accuracy for '
               'momentum versus quality (Table 4).')
    ED.replace(p, 'The gain is therefore associated with boosted trees, which can represent interactions between '
                  'predictors, rather than with smooth nonlinear effects of individual predictors. A similar pattern is '
                  'reported for US stock returns, where an additive spline model fails to outperform a linear one '
                  'robustly and the gains of more flexible models are attributed to interactions between predictors',
               'The ordering on ROC-AUC is consistent with evidence from US stock returns that the gains of more '
               'flexible models come mainly from interactions between predictors, which boosted trees can represent '
               'and an additive model cannot')
    ED.replace(p, r'\. XGBoost.s lead in ranking holds across its hyperparameter settings',
               ', but the differences here are too imprecise to confirm it. XGBoost’s lead in ranking holds across '
               'most of its hyperparameter settings', regex=True)
    assert fs['above_alt'][VQ] > 0.85 and fs['above_alt'][MV] > 0.7 and fs['logistic_beats_all_mq']
    ED.replace(p, 'where its ROC-AUC exceeds that of logistic regression by only 0.019',
               'where the best fixed setting of logistic regression ranks the weeks better than every XGBoost setting')
    ED.replace(p, r'The GAM.s weak ranking for value versus quality, by contrast, partly reflects the setting selected '
                  r'at its first tuning point\.',
               'The high log-losses of the GAM, by contrast, reflect a penalty that leaves the linear part of each '
               'fitted function unpenalised, so that its forecasts cannot be shrunk towards a constant (Section 4.3).',
               regex=True)
    log('A1-A4/A9', '5.1 second finding: ROC-AUC rises with flexibility for every pair, not significantly; the GAM’s '
                    'log-losses explained by its penalty; fixed-setting sensitivity from Appendix C.5')
    tn, fp, fn, tp = g(VQ, 'XGBoost', 'confusion')
    mv = g(MV, 'XGBoost', 'confusion')
    assert g(VQ, 'XGBoost', 'log_loss') == min(g(VQ, m, 'log_loss') for m in MODELS) > np.log(2)
    assert all(g(MV, m, 'roc_auc') < 0.5 for m in ('Logistic', 'GAM'))
    assert max(MODELS, key=lambda m: g(MQ, m, 'accuracy')) == 'GAM'
    assert min(MODELS, key=lambda m: g(MQ, m, 'log_loss')) == 'Logistic'
    p = find(body, 'Third, predictability differs markedly across the factor pairs')
    ED.replace(p, 'ranks the weeks best (ROC-AUC 0.664), recognises both leadership states (45 of the 77 value-led and '
                  '49 of the 79 quality-led weeks) and has a log-loss (0.670) below 0.693',
               f'ranks the weeks best (ROC-AUC {f3(g(VQ, "XGBoost", "roc_auc"))}), recognises both leadership states '
               f'({tp} of the {tp + fn} value-led and {tn} of the {tn + fp} quality-led weeks) and has the lowest '
               f'log-loss ({f3(g(VQ, "XGBoost", "log_loss"))}), although even this exceeds 0.693')
    ED.replace(p, r"For momentum versus value, XGBoost is the only model with a ROC-AUC well above 0\.5 \(0\.625\), but "
                  r"its log-loss \(0\.715\) exceeds 0\.693\. Momentum versus quality provides the weakest evidence of "
                  r"timing ability\. XGBoost.s accuracy of 61\.5% results from predicting momentum in 89\.1% of weeks, "
                  r"so that it identifies 82 of the 85 momentum-led weeks but only 14 of the 71 quality-led weeks "
                  r"\(Section 4\.3\)\.",
               'For momentum versus quality, the measures disagree: XGBoost ranks the weeks best (ROC-AUC '
               f'{f3(g(MQ, "XGBoost", "roc_auc"))}) but has the lowest accuracy ({pct(g(MQ, "XGBoost", "accuracy"))}), '
               f'below the {pct(F[MQ + "|majority"])} of always predicting momentum, while the GAM has the highest '
               f'accuracy ({pct(g(MQ, "GAM", "accuracy"))}) and logistic regression the lowest log-loss '
               f'({f3(g(MQ, "Logistic", "log_loss"))}). Momentum versus value provides the weakest evidence of timing '
               'ability. Logistic regression and the GAM have ROC-AUCs below 0.5, and XGBoost’s accuracy of '
               f'{pct(g(MV, "XGBoost", "accuracy"))} results from predicting momentum in '
               f'{pct(g(MV, "XGBoost", "pred_pos_share"))} of weeks, so that it identifies {mv[3]} of the '
               f'{mv[3] + mv[2]} momentum-led weeks but only {mv[0]} of the {mv[0] + mv[1]} value-led weeks '
               '(Section 4.3).', regex=True)
    log('A1-A3', '5.1 third finding rewritten on the rerun: momentum versus value is the weakest pair; the measures '
                 'disagree for momentum versus quality')
    top = lambda pair, m: g(pair, m, 'shap_top10')  # noqa: E731
    assert all(top(MV, m)[0].startswith('relvol_') for m in MODELS)
    assert top(VQ, 'XGBoost')[0] == top(VQ, 'Logistic')[0] == 'mom3m_value_vs_quality'
    assert len(set(top(MQ, 'Logistic')) & set(top(MQ, 'XGBoost'))) == 6
    p = find(body, 'Fourth, the models draw on different information for each pair')
    ED.replace(p, 'and the value–quality volatility ratio ranks among the four leading features of every model for value '
                  'versus quality, where the three-month relative return of value against quality leads XGBoost and '
                  'ranks second for logistic regression.',
               'and the three-month relative return of value against quality leads both XGBoost and logistic regression '
               'for value versus quality.')
    ED.replace(p, 'For momentum versus quality, logistic regression and XGBoost share their two leading features, the '
                  'change in the TOP40 price-to-EBITDA ratio and South African real-yield-curve curvature (Section '
                  '4.4).',
               'For momentum versus quality, logistic regression and XGBoost share six of their ten leading features, '
               'among them the change in the TOP40 price-to-EBITDA ratio (Section 4.4).')
    log('A1-A3', '5.1 fourth finding: SHAP rankings from the rerun')
    edit(body, 'First, factor timing models should be assessed and applied pair by pair',
         'give weaker support for momentum versus value and the least support for momentum versus quality',
         'give weaker support for momentum versus quality and the least support for momentum versus value', 'A1-A3',
         '5.2: order of the momentum pairs from the rerun')
    depths = {s['max_depth'] for pair in PAIR_NAMES for s in g(pair, 'XGBoost', 'schedule')}
    assert depths == {2, 3, 4}
    p = find(body, 'Second, model complexity should be matched to the evidence rather than presumed to add value')
    ED.replace(p, 'the additive nonlinear model performed worst on most measures',
               'the additive nonlinear model gave the least reliable probabilities for every pair')
    assert all(F[f'{VQ}|XGBoost|{k}'] > F[f'{VQ}|Logistic|{k}'] for k in ('accuracy', 'roc_auc'))
    assert F[f'{VQ}|XGBoost|log_loss'] < F[f'{VQ}|Logistic|log_loss'] and fs['above_alt'][VQ] > 0.85
    ED.replace(p, 'as XGBoost does for value versus quality.',
               'as XGBoost does for value versus quality, although even there its lead over logistic regression is '
               'not statistically significant (Section 4.2).')
    log('A1-A4', '5.2: second implication on the rerun (the GAM’s probabilities; XGBoost’s value–quality lead not '
                 'significant)')
    edit(body, 'the interpretability that practitioners require (Section 2.2.3)', '(Section 2.2.3)',
         '(Section 2.2.4)', 'P', '5.2: Section 2.2.4')
    assert all(g(pair, m, 'log_loss') > np.log(2) for pair in PAIR_NAMES for m in MODELS)
    assert abs(g(MV, 'XGBoost', 'accuracy') - F[MV + '|majority']) < 1e-12
    assert g(MQ, 'XGBoost', 'roc_auc') == max(g(MQ, m, 'roc_auc') for m in MODELS)
    p = find(body, 'Third, forecasts should be evaluated on more than aggregate accuracy and ROC-AUC')
    ED.replace(p, 'The momentum versus quality results show that accuracy can be raised by favouring the more frequent '
                  'class, and the momentum versus value results show that a model can rank the weeks reasonably well '
                  'while',
               'The momentum versus value results show that favouring the more frequent class raises accuracy only to '
               'the rate of always predicting it, and the momentum versus quality results show that a model can rank '
               'the weeks best while')
    ED.replace(p, 'a condition met by only three of the nine fitted models (Table 4)',
               'a condition met by none of the nine fitted models (Table 4)')
    log('A1-A3', '5.2: third implication on the rerun (pairs swapped; no model beats a constant forecast)')
    p = find(body, 'A third limitation relates to the feature selection procedure')
    ED.append(p, ' The selection is also made once, on the weeks before the evaluation block (Section 3.6), and is not '
                 'revised as the evaluation proceeds.')
    log('A2', '5.3: feature-selection limitation: one selection, made before the evaluation block and not revised')
    edit(body, 'Re-estimating the clustering on the data available at each tuning point',
         'would remove evaluation-period information from the selection',
         'would let the selection use the data that become available during the evaluation', 'A2',
         '5.4: re-estimating the clustering at each tuning point described as an extension of the design')
    p = find(body, 'Finally, the evaluation rests on a short sample')
    ED.replace(p, 'The differences between the models are therefore point estimates, and',
               'The differences between the models are therefore estimated imprecisely, and')
    ED.append(p, ' Cross-validation in the tuning sample also did not anticipate the lead of XGBoost in the two '
                 'momentum pairs (Section 4.2).')
    log('A4/A5', '5.3: short-sample limitation updated (intervals, CPCV evidence)')
    edit(body, 'A longer evaluation period would allow the differences', 'would allow the differences between the '
                                                                          'models to be tested formally',
         'would give the tests of the differences between the models more power', 'A4', '5.4: tests now exist')


# ======================================================================================= equations
def math_r(text):
    r = etree.Element(m('r'))
    rp = etree.SubElement(r, w('rPr'))
    etree.SubElement(rp, w('rFonts')).attrib.update({w('ascii'): 'Cambria Math', w('hAnsi'): 'Cambria Math'})
    t = etree.SubElement(r, m('t'))
    t.text = text
    return r


def msub(base, sub, sup=None):
    el = etree.Element(m('sSub' if sup is None else 'sSubSup'))
    e = etree.SubElement(el, m('e'))
    e.append(base if not isinstance(base, str) else math_r(base))
    s = etree.SubElement(el, m('sub'))
    s.append(math_r(sub) if isinstance(sub, str) else sub)
    if sup is not None:
        u = etree.SubElement(el, m('sup'))
        u.append(math_r(sup) if isinstance(sup, str) else sup)
    return el


def msup(base, sup):
    el = etree.Element(m('sSup'))
    etree.SubElement(el, m('e')).append(math_r(base))
    etree.SubElement(el, m('sup')).append(math_r(sup))
    return el


def mdelim(*items):
    el = etree.Element(m('d'))
    e = etree.SubElement(el, m('e'))
    for it in items:
        e.append(it)
    return el


def math_ins(*items):
    ins = ED.mark('ins')
    for it in items:
        ins.append(it)
    return ins


def math_del(*items):
    d = ED.mark('del')
    items[0].addprevious(d)
    for it in items:
        d.append(it)
    return d


def equations(body):
    kids = list(body)
    eq_tables = [k for k in kids if local(k) == 'tbl' and re.search(r'\((\d+)\)\s*$', ''.join(k.itertext()))
                 and k.find('.//m:oMath', NS) is not None]
    numbers = [int(re.search(r'\((\d+)\)\s*$', ''.join(t.itertext())).group(1)) for t in eq_tables]
    assert numbers == list(range(1, 20)), numbers
    # renumber (3)..(19) -> (4)..(20)
    for tbl, n in zip(eq_tables, numbers):
        if n < 3:
            continue
        cell = tbl.findall('.//w:tc', NS)[-1]
        p = cell.find(w('p'))
        ED.replace(p, f'({n})', f'({n + 1})')
    # the trend-scanning regression: delete the unnumbered paragraph, insert it as numbered equation (3)
    regs = [p for p in body.iter(w('p')) if 'i,j,t+h' in ''.join(p.itertext())]
    assert len(regs) == 1 and regs[0].find('.//m:oMathPara', NS) is not None
    reg = regs[0]
    new = copy.deepcopy(eq_tables[1])               # equation (2): two-cell table with the number on the right
    strip_ids(new)
    omath = new.find('.//m:oMath', NS)
    for c in list(omath):
        omath.remove(c)
    for el in (msub('CR', 'i,j,t+s'), math_r('=\u03b1+'), msub('\u03b2', 'h'), math_r('s+'),
               msub('\u03b5', 't+s'), math_r(',\u2003s=0,1,\u2026,h\u22121')):
        omath.append(el)
    num_p = new.findall('.//w:tc', NS)[-1].find(w('p'))
    for r in [c for c in num_p if local(c) == 'r']:
        num_p.remove(r)
    num_p.append(ED.make_run('(3)', None))
    ED.mark_table(new, 'ins')
    reg.addnext(new)
    ED.delete_paragraph(reg)
    log('P', 'Trend-scanning regression numbered as Equation (3), written with CR and a within-window index s; '
             'later equations renumbered (4)-(20)')
    # Equation (15) (now 16): loss argument sigma(F^(m-1)(X_t) + f_m(X_t))
    eq15 = eq_tables[14]
    d = [x for x in eq15.iter(m('d')) if x.find(m('e')).find(m('sSubSup')) is not None][0]
    e = d.find(m('e'))
    kids_e = list(e)
    i = [k.tag for k in kids_e].index(m('sSubSup'))
    old = kids_e[i:]                                 # p_t^(m-1) + f_m (X_t)
    assert len(old) == 4, [local(k) for k in old]
    math_del(*old)
    xt = lambda: mdelim(msub('X', 't'))
    e.append(math_ins(math_r('\u03c3'), mdelim(msup('F', '(m\u22121)'), xt(), math_r('+'), msub('f', 'm'), xt())))
    log('P', 'Equation (16) (was 15): loss evaluated at \u03c3(F^(m\u22121)(X_t) + f_m(X_t)), not a probability plus a '
             'tree output')
    # Equation (17) (now 18): leaf weights w_j -> omega_j, so they no longer clash with observation weights w_t
    eq17 = eq_tables[16]
    ss = eq17.find('.//m:sSubSup', NS)
    base = ss.find(m('e')).find(m('r'))
    assert base.find(m('t')).text == 'w'
    math_del(base)
    ss.find(m('e')).append(math_ins(math_r('\u03c9')))
    p = find(body, 'is the number of leaves and')
    n_inline = 0
    for om in p.iter(m('oMath')):
        for r in [r for r in om.iter(m('r')) if (r.find(m('t')).text or '').endswith('w')]:
            old = r.find(m('t')).text                  # '(w', the opening bracket is in the same run
            math_del(r)
            parent = r.getparent().getparent()
            parent.append(math_ins(math_r(old[:-1] + '\u03c9')))
            n_inline = 1
    assert n_inline == 1
    log('P', 'Equation (18) (was 17) and its text: leaf weights renamed \u03c9_j (w_t remains the observation weight)')


# ======================================================================================= acronyms, spelling
def acronyms(body):
    tbl = find(body, 'Acronym or abbreviation').getparent().getparent().getparent()
    assert local(tbl) == 'tbl'
    rows = {''.join(t.text or '' for t in tr.findall(w('tc'))[0].iter(w('t'))): tr for tr in tbl.findall(w('tr'))}
    for key in ('BRICS', 'LEI', 'MAE', 'MSE', 'PR-AUC', 'RMSE'):
        tr = rows[key]
        trpr = tr.find(w('trPr'))
        if trpr is None:
            trpr = etree.Element(w('trPr'))
            tr.insert(0, trpr)
        trpr.append(ED.mark('del'))
        for p in tr.iter(w('p')):
            ED.delete_paragraph(p)
    new = copy.deepcopy(rows['FTSE'])
    strip_ids(new)
    for tc, text in zip(new.findall(w('tc')), ('FNN', 'Feedforward Neural Network')):
        p = tc.find(w('p'))
        rpr = ED.template_rpr(p)
        for c in list(p):
            if local(c) != 'pPr':
                p.remove(c)
        p.append(ED.make_run(text, rpr))
    ED.mark_table(new, 'ins')
    rows['FTSE'].addprevious(new)
    for key, old, newtxt in (('TS', 'Time Series Level', 'Trend-Scanning t-Statistic'),
                             ('EntDyn', 'Change in Entropy', 'Entropy under Dynamic Encoding')):
        p = rows[key].findall(w('tc'))[1].find(w('p'))
        ED.replace(p, old, newtxt)
    log('P', 'Acronyms: BRICS, LEI, MAE, MSE, PR-AUC and RMSE removed (unused); FNN added; ARI and NMI kept for '
             'Appendix B; TS and EntDyn redefined to match the data (see Appendix A)')


UK = {'realized': 'realised', 'regularization': 'regularisation', 'Regularization': 'Regularisation',
      'maximize': 'maximise', 'minimizing': 'minimising', 'minimize': 'minimise', 'conceptualize': 'conceptualise',
      'penalized': 'penalised', 'optimization': 'optimisation', 'recognize': 'recognise', 'summarizes': 'summarises',
      'Generalized': 'Generalised'}


def uk_spelling(body):
    n = 0
    bib = [s for s in body.iter(w('sdt')) if s.find('w:sdtPr/w:bibliography', NS) is not None
           or s.find('w:sdtPr/w:docPartObj/w:docPartGallery[@w:val="Bibliographies"]', NS) is not None]
    skip = set()
    for s in bib:
        skip.update(s.iter(w('p')))
    pat = re.compile(r'\b(' + '|'.join(UK) + r')\b')
    for p in paragraphs(body):
        if p in skip:
            continue
        spans, pos = [], 0
        for el, s, kind in ED.tokens(p):            # only words in plain text runs (not fields, not new text)
            if kind == 'run':
                spans += [(pos + mt.start(), pos + mt.end(), mt.group(1)) for mt in pat.finditer(s)]
            pos += len(s)
        for a, b, word in reversed(spans):
            ED._replace_span(p, a, b, UK[word])
            n += 1
    log('T', f'UK spelling: {n} words changed in the body ({", ".join(sorted(UK))}); titles in the references '
             f'are left as published')


def captions(body):
    for n, pair in ((15, 'momentum versus value'), (16, 'momentum versus quality'), (17, 'value versus quality')):
        p = find(body, f'Figure {n}: Global SHAP importance of the reference models for {pair}')
        ED.replace(p, f'Figure {n}: ', ['Figure ', Seq('Figure', n), ': '])
    p = find(body, 'Model comparison', count=1)
    ED.replace(p, ' Model comparison', ': Model comparison')
    log('P', 'Figures 15-17 captions now SEQ fields (they will appear in the List of Figures); Table 2 caption colon')


# ======================================================================================= appendices
def appendix_heading(p, text, page_break):
    ppr = [etree.Element(w('pStyle'))]
    ppr[0].set(w('val'), 'AppendixHeading1')
    if page_break:
        ppr.append(etree.Element(w('pageBreakBefore')))
    num = etree.Element(w('numPr'))
    etree.SubElement(num, w('ilvl')).set(w('val'), '0')
    etree.SubElement(num, w('numId')).set(w('val'), '0')
    ppr.append(num)
    ED.change_ppr(p, ppr)
    ED.replace(p, ED.text(p), text)


def copy_from_calibri(cal_body, start_text, stop_text, include_start=False):
    kids = list(cal_body)
    txt = lambda e: ''.join(e.itertext())
    i = next(k for k, e in enumerate(kids) if txt(e).startswith(start_text))
    j = next(k for k, e in enumerate(kids) if k > i and txt(e).startswith(stop_text))
    return [copy.deepcopy(e) for e in kids[(i if include_start else i + 1):j]]


def plain_sub(el, old, new):
    """Untracked text change inside content that is about to be marked as inserted (the Calibri appendix)."""
    ts = [t for t in el.iter(w('t')) if old in (t.text or '')]
    assert len(ts) == 1 and ts[0].text.count(old) == 1, (old, len(ts))
    ts[0].text = ts[0].text.replace(old, new)


def set_text(p, text):
    """Untracked text of a plain paragraph that is about to be marked as inserted."""
    ts = list(p.iter(w('t')))
    assert ts and not list(p.iter(w('delText')))
    ts[0].text = text
    for t in ts[1:]:
        t.text = ''


def set_rows(tbl, rows, first_row=1):
    """Untracked cell text of a table that is about to be marked as inserted; surplus rows are dropped and missing
    ones copied from the last."""
    trs = tbl.findall(w('tr'))[first_row:]
    while len(trs) < len(rows):
        trs[-1].addnext(copy.deepcopy(trs[-1]))
        trs = tbl.findall(w('tr'))[first_row:]
    for tr in trs[len(rows):]:
        tbl.remove(tr)
    for tr, values in zip(trs, rows):
        tcs = tr.findall(w('tc'))
        assert len(tcs) == len(values)
        for tc, v in zip(tcs, values):
            ts = list(tc.iter(w('t')))
            if not ts:
                assert not v, v
                continue
            ts[0].text = v
            for t in ts[1:]:
                t.text = ''


def appendices(body, rels, media_dir, cal_dir):
    cal = etree.parse(str(cal_dir / 'word' / 'document.xml')).getroot().find(w('body'))
    cal_rels = etree.parse(str(cal_dir / 'word' / '_rels' / 'document.xml.rels')).getroot()
    rel_ns = '{http://schemas.openxmlformats.org/package/2006/relationships}'
    styles_ok = {'Caption'}
    next_doc_pr = [5000]
    # Figure C.1 of the rerun replaces the Calibri copy's picture
    c1_cap = next(e for e in cal if ''.join(e.itertext()).startswith('Figure C.1'))
    c1_rid = next(c1_cap.getprevious().iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip')).get(
        '{%s}embed' % R)
    c1_target = next(r.get('Target') for r in cal_rels if r.get('Id') == c1_rid)
    b1_cap = next(e for e in cal if ''.join(e.itertext()).startswith('Figure B.1:'))
    b1_rid = next(b1_cap.getprevious().iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip')).get(
        '{%s}embed' % R)
    b1_target = next(r.get('Target') for r in cal_rels if r.get('Id') == b1_rid)
    media_override = {c1_target: FIGS / 'figure_c1.png', b1_target: APPB / 'figure_b1_silhouette.png'}

    def import_elements(els, sub_like):
        out = []
        for el in els:
            strip_ids(el)
            for sp in list(el.iter(w('sectPr'))):
                sp.getparent().remove(sp)
            if local(el) == 'p':
                st = el.find('w:pPr/w:pStyle', NS)
                if st is not None and st.get(w('val')).startswith('Heading'):
                    # sub-headings in the thesis's B.x / C.x format
                    out.append(ED.new_paragraph(sub_like, ''.join(el.itertext())))
                    continue
                if st is not None and st.get(w('val')) not in styles_ok:
                    st.getparent().remove(st)
                for blip in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                    rid = blip.get('{%s}embed' % R)
                    target = next(r.get('Target') for r in cal_rels if r.get('Id') == rid)
                    name = 'supervisor_' + Path(target).name
                    shutil.copy(media_override.get(target, cal_dir / 'word' / target), media_dir / name)
                    new_id = 'rIdSup' + Path(target).stem
                    rel = etree.SubElement(rels, rel_ns + 'Relationship')
                    rel.attrib.update({'Id': new_id, 'Type': 'http://schemas.openxmlformats.org/officeDocument/2006/'
                                                             'relationships/image', 'Target': 'media/' + name})
                    blip.set('{%s}embed' % R, new_id)
                for dp in el.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr'):
                    next_doc_pr[0] += 1
                    dp.set('id', str(next_doc_pr[0]))
                ED.mark_inserted(el)
            elif local(el) == 'tbl':
                ED.mark_table(el, 'ins')
            out.append(el)
        return out

    # ---- Appendix A: rebuilt from the feature file (A8)
    a_head = find(body, 'A.1: Feature Selection')
    b_head = find(body, 'A.2: Feature selection')
    kids = list(body)
    old_a = kids[kids.index(a_head) + 1:kids.index(b_head)]
    for el in old_a:
        if local(el) == 'p':
            ED.delete_paragraph(el)
        elif local(el) == 'tbl':
            ED.mark_table(el, 'del')
    label = a_head.getprevious()                  # empty 'Appendix' paragraph numbered 'Appendix A'
    assert label.find('w:pPr/w:pStyle', NS).get(w('val')) == 'Appendix' and not ''.join(label.itertext()).strip()
    ED.delete_paragraph(label)
    appendix_heading(a_head, 'Appendix A: Feature definitions', page_break=False)
    sub_like = find(body, 'B.1: Screening of the candidate features')
    body_like = find(body, 'This appendix reproduces the 32 mutual-information')
    tbl_like = find(body, 'From candidate features to the final predictor set').getprevious()
    if local(tbl_like) != 'tbl':
        tbl_like = find(body, 'From candidate features to the final predictor set').getnext()
    cap_like = find(body, 'From candidate features to the final predictor set')
    new = [ED.new_paragraph(body_like,
                            f'Table A.1 defines the {N_FEAT} modelling features: the {N_REPS} cluster representatives '
                            '(Section 3.6) and the 18 relative-factor features added by design. Table A.2 explains '
                            'the naming '
                            'convention of the candidate features, and Table A.3 groups all 359 candidates by family; '
                            'Table B.3 lists every clustered candidate by cluster.')]
    new += feature_tables(cap_like, tbl_like)
    anchor = old_a[-1]
    for el in new:
        anchor.addnext(el)
        anchor = el
    log('A8', f'Appendix A rebuilt from featureset.csv: definitions of all {N_FEAT} modelling features (Table A.1), '
              'the '
              'naming convention (Table A.2) and the 359 candidates by family with counts (Table A.3); the old '
              'tables, which listed features not in the file (e.g. the SPX_* set) and the yield curves twice, are '
              'tracked deletions')
    # ---- Appendix B: the selection on the weeks before the evaluation block (A2), headings, renumbering, insertions
    appendix_heading(b_head, 'Appendix B: Feature selection', page_break=True)
    appendix_b(body, cal, sub_like, body_like, import_elements)
    # ---- Appendix C
    c_head = find(body, 'A.3: CPCV and walk-forward formulation')
    appendix_heading(c_head, 'Appendix C: CPCV and walk-forward validation', page_break=True)
    # the thesis's own Appendix C text and tables, on the purged initial tuning sample of the rerun (A3)
    tbl_after = lambda caption: find(body, caption).getnext()  # noqa: E731
    last_day = APPX['tune_last'].lstrip('0')
    edit(body, 'This appendix supplements Section 3.8 with the CPCV design',
         '758 weekly observations from 23 July 2007 to 24 January 2022',
         f'{N_TUNE} weekly observations from {APPX["tune_first"].lstrip("0")} to {last_day}, which end 12 weeks '
         'before the evaluation block', 'A3', f'Appendix C: initial tuning sample of {N_TUNE} weeks to {last_day}')
    sizes, (t_lo, t_hi) = APPX['split_sizes'], APPX['train_range']
    assert len(set(sizes[:4])) == 1
    p = find(body, 'The initial sample comprises four splits of 151 observations')
    ED.replace(p, 'four splits of 151 observations and a final split of 154',
               f'four splits of {sizes[0]} observations and a final split of {sizes[4]}')
    ED.replace(p, '408–444', f'{t_lo}–{t_hi}')
    n = sum(update_table(ED, tbl_after(cap), APPX[key]) for cap, key in (
        ('The five splits of the initial tuning sample and the positive share', 'table_c1'),
        ('The ten CPCV combinations of the initial search', 'table_c2'),
        ('ROC-AUCs of the settings selected at the initial tuning point', 'table_c3'),
        ('Walk-forward tuning schedule', 'table_c5'),
        ('Hyperparameters selected at each tuning point', 'table_c6')))
    base = APPX['baseline_mean_paths']
    edit(body, 'Each path combines predictions from four separately fitted models', '0.446, 0.422 and 0.406',
         f'{f3(base[MV])}, {f3(base[MQ])} and {f3(base[VQ])}', 'A3',
         f'C.1-C.4: split sizes, training range, baseline path ROC-AUCs and {n} cells of Tables C.1-C.3, C.5 and C.6 '
         'from the rerun')
    assert APPX['margin_range'][0] > 0
    edit(body, 'Table C.5 gives the tuning schedule', '14 of the', f"{APPX['changes']} of the", 'A1-A3',
         f"C.4: the selected setting changes at {APPX['changes']} of the 18 retuning points in the rerun")
    edit(body, 'The initial search uses the observations that precede the evaluation block',
         'precede the evaluation block, without a boundary purge (Section 3.8.1).',
         'precede the evaluation block, less the 12 weeks before it (Section 3.8.1).', 'A3',
         'C.4: the initial search is purged at the evaluation boundary')
    # material from the Calibri appendix copy, brought up to the rerun before it is marked as inserted
    c2_first = find(body, 'Each path combines predictions from four separately fitted models')
    extra = copy_from_calibri(cal, 'Each path combines predictions', 'Table C.3:')
    plain_sub(extra[0], '32 of the 36', f'{APPX["paths_above_baseline"]} of the 36')
    for el in reversed(import_elements(extra, sub_like)):
        c2_first.addnext(el)
    c3 = copy_from_calibri(cal, 'C.3: GAM settings excluded', 'C.4: Walk-forward', include_start=True)
    c4 = APPX['table_c4']
    assert (APPX['c4_runs'], len(c4)) == (3, 4)
    assert all(r[3] in ('λ = 1, 7 splines', 'λ = 1, 10 splines') for r in c4) and {r[3] for r in c4} == {
        'λ = 1, 7 splines', 'λ = 1, 10 splines'}
    plain_sub(c3[1], 'affecting two of its nine tuning runs. Both failures involved 10-spline settings whose penalised '
                     'iteratively reweighted least-squares fit diverged.',
              'affecting three of its nine tuning runs. All four failed settings combine the weakest penalty (λ = 1) '
              'with seven or ten splines, and their penalised iteratively reweighted least-squares fits diverged.')
    rows = [list(r) for r in c4]
    for prev, r in zip(c4, rows[1:]):           # a repeated pair or tuning point is shown once
        if r[0] == prev[0]:
            r[0] = ''
            if r[1] == prev[1]:
                r[1] = r[2] = ''
    set_rows(next(e for e in c3 if local(e) == 'tbl'), rows)
    c3_tbl = tbl_after('ROC-AUCs of the settings selected at the initial tuning point')
    assert local(c3_tbl) == 'tbl'
    anchor = c3_tbl
    for el in import_elements(c3, sub_like):
        anchor.addnext(el)
        anchor = el
    c6_tbl = tbl_after('Hyperparameters selected at each tuning point')
    assert local(c6_tbl) == 'tbl'
    anchor = c6_tbl
    kids = list(cal)
    i = next(k for k, e in enumerate(kids) if ''.join(e.itertext()).startswith('C.5: Hyperparameter sensitivity'))
    tail = [copy.deepcopy(e) for e in kids[i:] if local(e) in ('p', 'tbl')]
    txt = lambda e: ''.join(e.itertext())  # noqa: E731
    c5 = fixed_settings()['text']
    plain_sub(next(e for e in tail if txt(e).startswith('Each grid setting was held fixed')),
              'Settings selected throughout the evaluation reproduced Chapter 4’s results exactly.', c5['intro'])
    for start, key in (('XGBoost’s ranking advantage', 'xgb'), ('The GAM’s selected value', 'gam'),
                       ('For momentum–value, 13 of 18 logistic', 'logistic')):
        set_text(next(e for e in tail if txt(e).startswith(start)), c5[key])
    set_rows(next(e for e in tail if local(e) == 'tbl'), APPX['table_c7'])
    for el in import_elements(tail, sub_like):
        anchor.addnext(el)
        anchor = el
    log('A1-A3', 'C.2, C.3 (Table C.4) and C.5 (Table C.7, Figure C.1) from the Calibri copy brought up to the rerun '
                 '(paths above the baseline, four GAM exclusions in three tuning runs, C.5 text rewritten on the '
                 'fixed-setting runs, which reproduce the selected predictions period by period)')
    log('A6', 'Appendix headings are Appendix A / B / C (unnumbered, so the contents no longer shows "Chapter 7"); '
              'B.3 (Figure B.1), B.4 (Table B.4), C.2 baseline paragraphs, C.3 (Table C.4) and C.5 (Table C.7, '
              'Figure C.1) inserted from Appendices_Revised_Calibri.docx; B tables renumbered B.1-B.6')


def replace_table(old, header, rows, widths=None):
    """The old table as a tracked deletion followed by a new one in its style (tracked insertion)."""
    if widths is None:
        widths = [int(g.get(w('w'))) for g in old.find(w('tblGrid'))]
        assert len(widths) == len(header), (len(widths), header)
    new = make_table(old, header, rows, widths)
    old.addnext(new)
    ED.mark_table(old, 'del')
    return new


def merged_table(like, header, rows, widths):
    """A new table in the style of `like` whose body cells are lists of paragraphs (each a list of segments);
    a cell given as None continues the vertical merge of the cell above it."""
    tbl = make_table(like, header, [[''] * len(header)], widths)        # header row and one body-row template
    head, body_t = tbl.findall(w('tr'))
    tbl.remove(body_t)
    for x in [x for x in body_t.iter(w('ins')) if local(x.getparent()) == 'trPr']:
        x.getparent().remove(x)
    for k, values in enumerate(rows):
        tr = copy.deepcopy(body_t)
        for c, (tc, v) in enumerate(zip(tr.findall(w('tc')), values)):
            p0 = tc.find(w('p'))
            for p in tc.findall(w('p')):
                tc.remove(p)
            below = rows[k + 1][c] if k + 1 < len(rows) else 'end'
            tcpr = tc.find(w('tcPr'))
            assert tcpr is not None and tcpr.find(w('tcW')) is not None
            if v is None or below is None:
                vm = etree.Element(w('vMerge'))
                if v is not None:
                    vm.set(w('val'), 'restart')
                tcpr.find(w('tcW')).addnext(vm)
            paras = [[]] if v is None else v
            for segs in paras:
                p = etree.SubElement(tc, w('p'))
                ppr = p0.find(w('pPr'))
                if ppr is not None:
                    ppr = copy.deepcopy(ppr)
                    for x in ppr.findall('w:rPr/w:ins', NS):
                        x.getparent().remove(x)
                    p.append(ppr)
                rpr = None
                for r in p0.iter(w('r')):
                    rpr = r.find(w('rPr'))
                    break
                for el in ED.build(segs, copy.deepcopy(rpr) if rpr is not None else None, tracked=False):
                    p.append(el)
        tbl.append(tr)
    for tr in tbl.findall(w('tr'))[1:]:
        trpr = tr.find(w('trPr'))
        if trpr is None:
            trpr = etree.Element(w('trPr'))
            tr.insert(0, trpr)
        trpr.append(ED.mark('ins'))
        for p in tr.iter(w('p')):
            ED.mark_inserted(p)
    return tbl


B6 = [  # (Table 1 category, [(economic rationale, features)], evidence: paragraphs of segments, or None to merge)
    ('Price-based volatility', [
        ('Captures the level of the volatility term structure, and with it expected uncertainty.',
         ['VolTermStructPC1']),
        ('Local and gold-price volatility signal risk-on and risk-off environments.',
         ['GLD_FracDiff', 'GLD_RollSD', 'USDZARV1M.Curncy_QS', 'USDZARV1Y.Curncy_QS']),
        ('Global volatility spills over across international markets.', ['EURUSDV6M.Curncy_QS'])],
     [[CITE['Shi20'], ' Volatility consistently ranks among the most important ML predictors.'],
      [CITE['Dan16'], ' Momentum payoffs vary with market conditions.']]),
    ('Fundamental valuation', [
        ('Fundamental valuation measures (P/E, P/B, sales) capture expected mean reversion in prices.',
         ['MXWD_Fundamental_PC1', 'MXWD_PEvFwdPE']),
        ('Book-to-price differentials identify undervalued and overvalued stocks.',
         ['TOP40.Index.Price.to.Book.Ratio_Diff']),
        ('Sales- and earnings-based ratios identify cyclical undervaluation.',
         ['MXWD.Index.Price.to.Sales.Ratio_Diff', 'TOP40.Index.Price.to.Ebitda_Diff'])],
     [[CITE['Nus23'], ' Value, followed by momentum, proves to be a crucial driver of stock prices globally.']]),
    ('Sentiment and risk appetite', [
        ('Reflects household expectations and overall risk sentiment.', ['MichgnConcIndx']),
        ('Commodity PCA reflects global risk appetite, growth expectations and inflation pressure.',
         ['GenComIndexPC1_QS']),
        ('Structural breaks and the shape of gold returns signal major shifts in global uncertainty.',
         ['GLD_StructBreak', 'GLD_LempZiev', 'GLD_RollSkew', 'GLD_RollKurt']),
        ('Trends, returns and the predictability of global equity and bond markets capture momentum in investor '
         'optimism and pessimism.', ['GlobalEquityTS', 'GlobalBondTS', 'GlobalEquityGret3', 'GlobalEquity_Ent'])],
     [[CITE['Sae22'], ' Technical financial indicators and sentiment indicators showed stronger predictive power '
                      'than macroeconomic indicators.']]),
    ('Macroeconomic', [
        ('Yield-curve curvature and slope reflect growth expectations and the monetary policy stance (local and '
         'global).', ['SA_RB_Curvature', 'US_RB_Curvature', 'SA_NB_Slope']),
        ('The shape of the sovereign credit curve proxies sovereign risk, economic stress and funding conditions.',
         ['SA_CDS_Curvature']),
        ('Tracks the widening and narrowing of rand valuation.', ['RandPPP_Factor_Inst_TS_ST']),
        ('Euro-area bank lending conditions, which lead credit supply and economic activity.', ['BCMPEBLS.Index']),
        ('Captures medium-term USDZAR forward pricing, which reflects the rand–dollar interest-rate '
         'differential.', ['ZAR18M.Curncy_QS']),
        ('Information content in FX markets.', ['USDZAR_EntDyn'])],
     [[CITE['Fan22'], ' Factor predictability is macro-state dependent.']]),
    ('Factor relative performance', [
        ('Assets that have performed well in the recent past tend to continue performing.',
         ['mom1m_momentum_vs_quality', 'mom1m_momentum_vs_value', 'mom1m_value_vs_quality',
          'mom3m_momentum_vs_quality', 'mom3m_momentum_vs_value', 'mom3m_value_vs_quality']),
        ('Co-movement between factors identifies convergence and divergence regimes.',
         ['corr_momentum_quality_12', 'corr_momentum_value_12', 'corr_value_quality_12']),
        ('Standardised relative performance is useful for detecting style rotation.',
         ['zrel_value_vs_quality_12', 'zrel_momentum_vs_quality_12', 'zrel_momentum_vs_value_12'])],
     [[CITE['Shi20'], ' Momentum signals remain among the strongest predictors in both ML and traditional '
                      'frameworks.'],
      [CITE['Dan16'], ' Momentum payoffs vary with market conditions.']]),
    ('Factor relative volatility', [
        ('Identifies instability in factor relative performance.',
         ['relvol_momentum_vs_quality_12', 'relvol_momentum_vs_value_12', 'relvol_value_vs_quality_12']),
        ('Volatility differentials drive factor rotations.',
         ['volratio_momentum_quality_12', 'volratio_momentum_value_12', 'volratio_value_quality_12'])],
     None),
]


def appendix_b(body, cal, sub_like, body_like, import_elements):
    """Appendix B on the selection made from the weeks before the evaluation block (appendix_b_feature_selection.py):
    the thesis's own tables are replaced (tracked), the Calibri copy's B.3-B.4 are brought up to it before they are
    marked as inserted."""
    category = {f: c for f, c, _ in FEATURES if '<' not in f}
    b1 = pd.read_csv(APPB / 'b1_clusters.csv')
    assert set(b1.representative) == set(category) and len(b1) == N_REPS
    constant = pd.read_csv(APPB / 'b2_constant.csv').feature.tolist()
    assert constant[0] == 'PPP.ZA.Index_QS' and all('_Lambda' in c for c in constant[1:]) and len(constant) == 10
    # opening and B.1
    p = find(body, 'This appendix reproduces the 32 mutual-information')
    ED.replace(p, 'This appendix reproduces the 32 mutual-information (MI) cluster representatives used in the models '
                  'and tests their sensitivity',
               f'This appendix reports the selection of the {N_REPS} mutual-information (MI) cluster representatives '
               'used in the models, made on the weeks before the evaluation block, and tests its sensitivity')
    p = find(body, 'MI distances use 680 weekly observations')
    ED.replace(p, 'MI distances use 680 weekly observations from 26 March 2012 to 31 March 2025, after forward '
                  'filling.',
               f'The screen uses the {SEL["selection_weeks"]} weeks before the evaluation block, from 5 March 2007 to '
               f'24 January 2022, and the MI distances use the {SEL["mi_weeks"]} of them from 9 May 2011, when the '
               'last screened feature begins, after forward filling. The ten features that do not vary over these '
               'weeks, PPP.ZA.Index_QS and nine Nelson–Siegel decay parameters (Lambda), are removed before '
               'clustering, since they carry no information over them.')
    t = find(body, 'From candidate features to the final predictor set').getnext()
    assert local(t) == 'tbl'
    replace_table(t, ['Step', 'Features'], [
        ['Candidate features in the feature file', str(SEL['candidates'])],
        ['Removed: more than 30% of the weeks before the evaluation block missing (Table B.2)',
         f'−{SEL["removed_missing"]}'],
        ['Passing the missing-value screen', str(SEL['screened'])],
        ['Removed: no variation over the weeks used for MI', f'−{SEL["removed_constant"]}'],
        ['Clustered', str(SEL['clustered'])],
        ['Clusters at a cut height of 2, one representative each (Table B.3)', str(N_REPS)],
        ['Relative-factor features added by design (Section 3.6)', '18'],
        ['Final predictor set', str(N_FEAT)]])
    b2 = pd.read_csv(APPB / 'b2_screening.csv', parse_dates=['first_available'])
    assert len(b2) == SEL['removed_missing']
    t = find(body, 'Features removed by the missing-value screen').getnext()
    assert local(t) == 'tbl'
    replace_table(t, ['Feature', 'Share of weeks missing', 'First available'],
                  [[r.feature, f'{r.missing_share:.0%}', f'{r.first_available:%d %b %Y}'] for r in b2.itertuples()])
    # B.2: clusters
    t = find(body, 'Clusters, representatives and members (cut height 2)').getnext()
    assert local(t) == 'tbl'
    replace_table(t, ['Representative', 'Category', 'Size', 'Rep. mean dist.', 'Within-cluster dist.',
                      'Other members'],
                  [[r.representative, category[r.representative], str(r.size), f'{r.rep_mean_distance:.2f}',
                    f'{r.within_mean_distance:.2f}', r.other_members] for r in b1.itertuples()])
    log('A2', f'Appendix B opening and B.1-B.2: the selection uses only the {SEL["selection_weeks"]} weeks before the '
              f'evaluation block (MI on {SEL["mi_weeks"]} from 9 May 2011); Tables B.1-B.3 rebuilt '
              f'({SEL["removed_missing"]} screened out, {SEL["removed_constant"]} constant, {N_REPS} clusters), old '
              'tables kept as tracked deletions')
    # B.3 and B.4 from the Calibri copy, brought up to the selection before they are marked as inserted
    sil = pd.read_csv(APPB / 'b3_silhouette.csv').set_index('clusters').silhouette
    assert SEL['silhouette_argmax'] == 100 and sil.max() < 0.12 and sil.idxmax() == 100
    stab = pd.read_csv(APPB / 'b4_stability.csv')
    stab['not_represented'] = N_REPS - stab.kept_at_k - stab.same_cluster_substitute_at_k
    stab['changed'] = N_REPS - stab.kept_at_k
    full = stab.iloc[2]
    assert full.variant.startswith('All weeks') and (full.changed, full.same_cluster_substitute_at_k) == (7, 6)
    bins = stab.iloc[:2]
    assert list(bins.changed) == [11, 11] and sorted(N_REPS - bins.not_represented) == [26, 28]
    block = copy_from_calibri(cal, 'B.3: Choice of cut height', 'B.5: Mutual information', include_start=True)
    txt = lambda e: ''.join(e.itertext())  # noqa: E731
    para = lambda start: next(e for e in block if txt(e).startswith(start))  # noqa: E731
    plain_sub(para('Figure B.1 assesses cluster separation'), 'The mean silhouette score is 0.037 at 32 clusters',
              f'The mean silhouette score is {SEL["silhouette_at_baseline"]:.3f} at {N_REPS} clusters')
    plain_sub(para('Figure B.1 assesses cluster separation'), 'scores remain below 0.1 and reach 0.097 at 100',
              f'scores remain below 0.12 and reach {SEL["silhouette_max"]:.3f} at 100')
    e = para('Table B.4 tests the omitted-column correction')
    plain_sub(e, txt(e), 'Table B.4 tests 10 and 30 quantile bins instead of 20, and estimation on all '
                         f'{SEL["full_weeks"]} weeks to 31 March 2025, which include the evaluation block, with the '
                         f'screen also applied to all weeks. Representative comparisons use {N_REPS} clusters because '
                         'linkage heights vary with specification. A substitution retains another member of the same '
                         'cluster of the selection. The adjusted Rand index (ARI) measures partition agreement: 1 '
                         'indicates identical partitions and 0 indicates chance agreement.')
    plain_sub(para('Table B.4: Selection stability'), 'at 32 clusters', f'at {N_REPS} clusters')
    set_rows(next(e for e in block if local(e) == 'tbl'),
             [['All weeks to March 2025 (including the evaluation block)' if r.variant.startswith('All weeks')
               else r.variant, str(r.clusters_at_height_2), str(r.kept_at_k), str(r.same_cluster_substitute_at_k),
               str(r.not_represented), f'{r.ari_at_k:.2f}'] for r in stab.itertuples()])
    e = para('Correcting the omission replaces')
    plain_sub(e, txt(e), f'The bin choices change {bins.changed.iloc[0]} representatives each, while '
                         f'{min(N_REPS - bins.not_represented)} and {max(N_REPS - bins.not_represented)} of the '
                         f'{N_REPS} clusters remain represented (Table B.4). Adding the evaluation block changes '
                         f'{full.changed} representatives, {full.same_cluster_substitute_at_k} of them substitutions '
                         'within the same cluster, so the evaluation weeks would have altered the selection only '
                         'moderately.')
    b5_head = find(body, 'B.5: Mutual information and correlation')
    for el in import_elements(block, sub_like):
        b5_head.addprevious(el)
    log('A2', f'B.3 (silhouette {SEL["silhouette_at_baseline"]:.3f} at {N_REPS} clusters, Figure B.1) and B.4 (Table '
              'B.4: bins and the full sample including the evaluation block) on the new selection')
    # B.5: pairs with high MI but low correlation
    hi = pd.read_csv(APPB / 'b5_high_mi_low_correlation.csv').sort_values('nmi', ascending=False, kind='stable')
    assert len(hi) == SEL['hi_mi_low_r_pairs'] == 40
    intro = ED.new_paragraph(body_like,
                             f'Of the {SEL["pairs"]:,} pairs of clustered features, {SEL["hi_mi_low_r_pairs"]} have '
                             'an NMI of at least 0.4 but an absolute Pearson correlation (|r|) below 0.2; Table B.5 '
                             'lists the 15 with the highest NMI. Across the 40, the median NMI falls from '
                             f'{SEL["hi_mi_low_r_median_nmi_levels"]:.2f} in levels to '
                             f'{SEL["hi_mi_low_r_median_nmi_changes"]:.2f} in weekly changes, and the lower lag-1 '
                             f'autocorrelation of the two series has a median of '
                             f'{SEL["hi_mi_low_r_median_autocorr"]:.2f}. Clustering on |r| instead of NMI would keep '
                             f'only {SEL["corr_kept"]} of the {N_REPS} representatives (ARI {SEL["corr_ari"]:.2f}).')
    b5_head.addnext(intro)
    edit(body, 'Feature pairs with high MI but low correlation', 'Table B.7', 'Table B.5', 'A6', 'Table B.7 -> B.5')
    t = find(body, 'Feature pairs with high MI but low correlation').getnext()
    assert local(t) == 'tbl'
    replace_table(t, ['Feature pair', 'NMI (levels)', 'NMI (weekly changes)', '|r| (levels)',
                      'Lag-1 autocorrelation'],
                  [[f'{r.feature_a} – {r.feature_b}', f'{r.nmi:.2f}', f'{r.nmi_weekly_changes:.2f}',
                    f'{r.abs_pearson:.2f}', f'{r.autocorr_a:.2f} / {r.autocorr_b:.2f}'] for r in hi.head(15).itertuples()])
    log('A2', f'B.5: opening paragraph and Table B.5 on the new selection ({SEL["hi_mi_low_r_pairs"]} pairs with high '
              f'NMI and low |r|, the 15 highest listed; |r| clustering keeps {SEL["corr_kept"]} of {N_REPS})')
    # B.6: economic classification of the modelling features
    p = find(body, 'lists the 50 modelling features, the 32 cluster representatives')
    ED.replace(p, 'Table B.8 lists the 50 modelling features, the 32 cluster representatives',
               f'Table B.6 lists the {N_FEAT} modelling features, the {N_REPS} cluster representatives')
    ED.replace(p, 'by the categories of Table 2', 'by the categories of Table 1')
    cap = find(body, 'Economic classification of the 50 modelling features')
    ED.replace(cap, 'Table B.8', 'Table B.6')
    ED.replace(cap, 'of the 50 modelling', f'of the {N_FEAT} modelling')
    old = cap.getnext()
    assert local(old) == 'tbl'
    rows = []
    for cat, groups, evidence in B6:
        for g, (why, feats) in enumerate(groups):
            for k, f in enumerate(feats):
                rows.append([[[cat]] if g == k == 0 else None, [[f]], [[why]] if k == 0 else None,
                             evidence if (g == k == 0 and evidence is not None) else None])
    rel = pd.read_csv(DATA / 'factor_relative_features.csv', nrows=1).columns[1:]
    listed = [r[1][0][0] for r in rows]
    assert sorted(listed) == sorted(set(category) | set(rel)) and len(listed) == N_FEAT
    widths = [int(g.get(w('w'))) for g in old.find(w('tblGrid'))]
    new = merged_table(old, ['Feature category', 'Feature', 'Economic rationale', 'Empirical findings'], rows, widths)
    old.addnext(new)
    ED.mark_table(old, 'del')
    log('A2/A8/T', f'Table B.6 rebuilt for the {N_FEAT} modelling features in the categories of Table 1, with citation '
                   'fields (BCMPEBLS: euro-area bank lending survey; "proxies sovereign risk"; "Value, followed by '
                   'momentum, proves"); old table kept as a tracked deletion')


FEATURES = [  # (feature, Table 1 category, definition)
    ('VolTermStructPC1', 'Price-based volatility', 'First principal component of the VIX, VIX3M, VIX6M and VIX1Y '
                                                   'indices (level of the US equity volatility term structure)'),
    ('USDZARV1M.Curncy_QS', 'Price-based volatility', 'One-month at-the-money implied volatility of USD/ZAR, quantile '
                                                      'scaled'),
    ('USDZARV1Y.Curncy_QS', 'Price-based volatility', 'One-year at-the-money implied volatility of USD/ZAR, quantile '
                                                      'scaled'),
    ('EURUSDV6M.Curncy_QS', 'Price-based volatility', 'Six-month at-the-money implied volatility of EUR/USD, quantile '
                                                      'scaled'),
    ('GLD_RollSD', 'Price-based volatility', 'Rolling standard deviation of gold returns'),
    ('GLD_FracDiff', 'Price-based volatility', 'Fractionally differenced gold price'),
    ('MXWD_Fundamental_PC1', 'Fundamental valuation', 'First principal component of the valuation ratios of the MSCI '
                                                      'World Index'),
    ('MXWD_PEvFwdPE', 'Fundamental valuation', 'Trailing price-to-earnings ratio of the MSCI World Index less its '
                                               'forward price-to-earnings ratio'),
    ('MXWD.Index.Price.to.Sales.Ratio_Diff', 'Fundamental valuation', 'Price-to-sales ratio of the MSCI World Index, '
                                                                      'differenced'),
    ('TOP40.Index.Price.to.Book.Ratio_Diff', 'Fundamental valuation', 'Price-to-book ratio of the FTSE/JSE Top 40 '
                                                                      'Index, differenced'),
    ('TOP40.Index.Price.to.Ebitda_Diff', 'Fundamental valuation', 'Price-to-EBITDA ratio of the FTSE/JSE Top 40 '
                                                                  'Index, differenced'),
    ('MichgnConcIndx', 'Sentiment and risk appetite', 'University of Michigan Consumer Sentiment Index'),
    ('GenComIndexPC1_QS', 'Sentiment and risk appetite', 'First principal component of commodity spot prices, '
                                                         'quantile scaled'),
    ('GLD_StructBreak', 'Sentiment and risk appetite', 'Structural-break statistic of the gold price'),
    ('GLD_LempZiev', 'Sentiment and risk appetite', 'Lempel–Ziv complexity of gold returns'),
    ('GLD_RollSkew', 'Sentiment and risk appetite', 'Rolling skewness of gold returns'),
    ('GLD_RollKurt', 'Sentiment and risk appetite', 'Rolling kurtosis of gold returns'),
    ('GlobalEquityTS', 'Sentiment and risk appetite', 'Trend-scanning t-statistic of the global equity index '
                                                      '(direction and strength of its recent trend)'),
    ('GlobalBondTS', 'Sentiment and risk appetite', 'Trend-scanning t-statistic of the global bond index'),
    ('GlobalEquityGret3', 'Sentiment and risk appetite', 'Three-month geometric return of the global equity index'),
    ('GlobalEquity_Ent', 'Sentiment and risk appetite', 'Entropy of global equity returns under static encoding'),
    ('SA_NB_Slope', 'Macroeconomic', 'Slope factor of a Nelson–Siegel fit to the South African nominal bond '
                                     'yield curve'),
    ('SA_RB_Curvature', 'Macroeconomic', 'Curvature factor of a Nelson–Siegel fit to the South African real '
                                         '(inflation-linked) bond yield curve'),
    ('US_RB_Curvature', 'Macroeconomic', 'Curvature factor of a Nelson–Siegel fit to the US real '
                                         '(inflation-protected) bond yield curve'),
    ('SA_CDS_Curvature', 'Macroeconomic', 'Curvature factor of a Nelson–Siegel fit to the South African '
                                          'sovereign credit default swap curve'),
    ('RandPPP_Factor_Inst_TS_ST', 'Macroeconomic', 'Short-horizon trend-scanning t-statistic of the rand '
                                                   'purchasing-power-parity valuation'),
    ('BCMPEBLS.Index', 'Macroeconomic', 'Euro-area bank lending survey index (bank lending conditions)'),
    ('ZAR18M.Curncy_QS', 'Macroeconomic', '18-month USD/ZAR forward points, quantile scaled'),
    ('USDZAR_EntDyn', 'Macroeconomic', 'Entropy of USD/ZAR returns under dynamic encoding'),
    ('mom1m_<pair>', 'Factor relative performance', 'Difference between the two factors\u2019 compounded returns over '
                                                    'the last 4 weeks (three features, one per pair)'),
    ('mom3m_<pair>', 'Factor relative performance', 'Difference between the two factors\u2019 compounded returns over '
                                                    'the last 12 weeks (three features)'),
    ('zrel_<pair>_12', 'Factor relative performance', 'Weekly return spread of the pair, standardised by its 12-week '
                                                      'rolling mean and standard deviation (three features)'),
    ('corr_<a>_<b>_12', 'Factor relative performance', '12-week rolling correlation of the two factors\u2019 weekly '
                                                       'returns (three features)'),
    ('relvol_<pair>_12', 'Factor relative volatility', '12-week rolling standard deviation of the weekly return '
                                                       'spread of the pair (three features)'),
    ('volratio_<a>_<b>_12', 'Factor relative volatility', 'Ratio of the two factors\u2019 12-week rolling '
                                                          'volatilities (three features)'),
]
CONVENTIONS = [
    ('_QS, _QuantileScale, _QuantScale', 'Quantile scaled: the percentile of the value within the series\u2019 '
                                         'history, between 0 and 1'),
    ('_Diff', 'Differenced series, a transformation towards stationarity'),
    ('FracDiff', 'Fractionally differenced series, which removes the trend but keeps long memory'),
    ('TS (_TS, _TS_ST, _TS_LT)', 'Backward-looking trend-scanning t-statistic of the series (ST and LT: short and '
                                 'long horizon)'),
    ('Gret1/3/6/12, Vol1/3/6/12, RV_3/12', 'Geometric return and realised volatility over 1, 3, 6 or 12 months'),
    ('SB, StructBreak', 'Structural-break statistic'),
    ('Ent, EntDyn, LempZiev', 'Entropy of returns under static or dynamic encoding; Lempel\u2013Ziv complexity'),
    ('RollSD, RollSkew, RollKurt', 'Rolling standard deviation, skewness and kurtosis of returns'),
    ('PC1, PC2, PC3', 'Principal components of a group of related series'),
    ('Level, Slope, Curvature, Lambda, R2', 'Nelson\u2013Siegel factors, decay parameter and goodness of fit of a '
                                            'curve; NB and RB denote nominal and real bond curves, CDS the sovereign '
                                            'credit default swap curve'),
    ('V<tenor>.Curncy (e.g. USDZARV1M)', 'At-the-money implied volatility of the currency pair at that tenor'),
    ('ZAR<tenor>.Curncy, ZARI<tenor>.Curncy', 'USD/ZAR forward points and implied rand yields at that tenor'),
    ('_YoY', 'Year-on-year change'),
]
FAMILIES = [
    ('Equity and bond volatility indices', r'^(VolTermStruct|VIX|VolSpread|VixMove|MOVEStructBreak|VIXStructBreak|'
                                           r'OVXStructBreak)'),
    ('Asset-class returns, volatility, trends and breaks', r'^(Local|Global)(Equity|Bond|Bonds|Property)(TS|Gret|Vol|SB)'
                                                           r'|^USDZAR(TS|Gret|Vol\d|SB)|_RV_'),
    ('Entropy', r'_Ent($|Dyn|_TS)|^MultiAssetEnt'),
    ('Cross-asset correlations', r'_Corr'),
    ('Yield and credit curves', r'^(SA|US)_(NB|RB|CDS)_|^SABreakEvenInfl|BndStructBreak|^RepSouCDS'),
    ('Currency volatility, forward points and implied yields', r'^(USDZAR_Vol|USDZARV|USDEUR_Vol|EURUSDV|SAFwdPts|ZAR\d|'
                                                               r'ZAR_FwdPts|SAImplYld|ZARI|DXY)'),
    ('Commodities', r'^(GLD_|XAU|XAG|LA1|HG1|NG1|GenComIndex)'),
    ('Equity valuation ratios (JSE All Share, Top 40, MSCI World)', r'^(JALSH|MXWD|TOP40|LocalGlobalPE)'),
    ('Macroeconomic, survey and sentiment indicators', r'.'),
]


def feature_tables(cap_like, tbl_like):
    cols = pd.read_csv(DATA / 'featureset.csv', nrows=1).columns[1:]
    assert len(cols) == 359
    reps = set(pd.read_csv(DATA / 'representatives.csv')['Representative'])
    listed = {f for f, _, _ in FEATURES if '<' not in f}
    assert listed == reps, listed ^ reps
    rel = pd.read_csv(DATA / 'factor_relative_features.csv', nrows=1).columns[1:]
    assert len(rel) == 18
    fam = {name: [] for name, _ in FAMILIES}
    for c in cols:
        name = next(n for n, pat in FAMILIES if re.search(pat, c))
        fam[name].append(c)
    assert sum(len(v) for v in fam.values()) == 359
    out = []
    out.append(ED.new_paragraph(cap_like, f'Table A.1: The {N_FEAT} modelling features'))
    out.append(make_table(tbl_like, ['Feature', 'Category (Table 1)', 'Definition'],
                          [[f, c, d] for f, c, d in FEATURES], [3000, 2350, 3676]))
    out.append(ED.new_paragraph(cap_like, [Italic('Note: <pair> stands for momentum_vs_value, momentum_vs_quality or '
                                                  'value_vs_quality, and <a>_<b> for momentum_value, momentum_quality '
                                                  'or value_quality. The 18 relative-factor features are computed from '
                                                  f'the weekly factor returns; the {N_REPS} representatives come '
                                                  'from the candidate file and are lagged by four weeks with the '
                                                  'rest. The models also receive a 0/1 indicator for each of the '
                                                  'four representatives that begin after the matrix (Section '
                                                  '3.4).')]))
    out.append(ED.new_paragraph(cap_like, 'Table A.2: Naming convention of the candidate features'))
    out.append(make_table(tbl_like, ['Element of the name', 'Meaning'], [list(r) for r in CONVENTIONS], [3300, 5726]))
    out.append(ED.new_paragraph(cap_like, 'Table A.3: The 359 candidate features by family'))
    rows = []
    for name, _ in FAMILIES:
        ex = fam[name]
        rows.append([name, str(len(ex)), '; '.join(ex[:6]) + ('; \u2026' if len(ex) > 6 else '')])
    rows.append(['Total', '359', ''])
    out.append(make_table(tbl_like, ['Family', 'Features', 'Examples'], rows, [3000, 1000, 5026]))
    return out


def make_table(like, header, rows, widths):
    """A new table in the style of `like` (first two rows used as header / body templates)."""
    tbl = copy.deepcopy(like)
    strip_ids(tbl)
    trs = tbl.findall(w('tr'))
    head_t, body_t = trs[0], trs[1] if len(trs) > 1 else trs[0]
    for tr in trs:
        tbl.remove(tr)
    grid = tbl.find(w('tblGrid'))
    for g in list(grid):
        grid.remove(g)
    for wd in widths:
        etree.SubElement(grid, w('gridCol')).set(w('w'), str(wd))
    tw = tbl.find('w:tblPr/w:tblW', NS)
    if tw is not None:
        tw.set(w('w'), str(sum(widths)))
        tw.set(w('type'), 'dxa')

    def row(template, values, bold):
        tr = copy.deepcopy(template)
        tcs = tr.findall(w('tc'))
        for extra in tcs[1:]:
            tr.remove(extra)
        tc0 = tcs[0]
        for x in tc0.findall('w:tcPr/w:gridSpan', NS) + tc0.findall('w:tcPr/w:vMerge', NS):
            x.getparent().remove(x)
        for i, (v, wd) in enumerate(zip(values, widths)):
            tc = copy.deepcopy(tc0) if i else tc0
            tcw = tc.find('w:tcPr/w:tcW', NS)
            if tcw is not None:
                tcw.set(w('w'), str(wd))
                tcw.set(w('type'), 'dxa')
            ps = tc.findall(w('p'))
            for extra in ps[1:]:
                tc.remove(extra)
            p = ps[0]
            rpr = ED.template_rpr(p)
            rpr = copy.deepcopy(rpr) if rpr is not None else None
            for c in list(p):
                if local(c) != 'pPr':
                    p.remove(c)
            jc = p.find('w:pPr/w:jc', NS)
            if jc is not None:
                jc.set(w('val'), 'left')
            if rpr is not None:
                for t in ('b', 'bCs'):
                    for x in rpr.findall(w(t)):
                        rpr.remove(x)
            p.append(ED.make_run(v, rpr, bold=bold))
            if not bold:                                  # body rows may break across pages
                for kn in p.findall('w:pPr/w:keepNext', NS):
                    kn.getparent().remove(kn)
            if i:
                tr.append(tc)
        return tr

    tbl.append(row(head_t, header, True))
    for values in rows:
        tbl.append(row(body_t, values, False))
    ED.mark_table(tbl, 'ins')
    return tbl


# ======================================================================================= footnotes, bibliography
def footnotes(fn_root):
    p = find(fn_root, 'The feature file spans 944 weeks')
    ED.replace(p, ED.text(p).strip(),
               'The factor returns span 939 weeks, from 9 April 2007 to 31 March 2025 (the candidate feature file '
               'covers 944 weeks from 5 March 2007). The 12-week rolling windows of the relative-factor features '
               'remove the first 11 (939 \u2212 11 = 928), and the four-week lag removes a further four (928 \u2212 4 = '
               '924), so the matrix begins on 23 July 2007. Trend-scanning labels cannot be resolved for the final ten '
               'weeks (Section 3.5), leaving 914 labelled observations per factor pair (23 July 2007 to 20 January '
               '2025).')
    log('A7/T', 'Footnote 2: counts from the current data (939, 928, 924, 914; 23 July 2007), with a full stop')
    p = find(fn_root, 'which shows a neutral choice')
    ED.replace(p, 'which shows a neutral choice', 'which expresses no view')
    log('T', 'Footnote 4: "which expresses no view"')


BIB_DELETE_PREFIX = {
    'Adr16': 'Adrian, T.', 'Alo22': 'Alonso, M. N.', 'Ang12': 'Ang, A.', 'Bel57': 'Bellman, R.', 'Ber19': 'Beraha, M.',
    'Bru13': 'Brunnermeier, M.', 'Cho02': 'Chordia, T.', 'Goo94': 'Goodfellow, I., Bengio, Y., & Courville, A. (1994)',
    'Gri22': 'Grinsztajn, L.', 'The': 'Hanley, J. A.', 'Hua20': 'Huang, D.', 'Koi18': 'Koijen, R.',
    'Kri07': 'Krier, C.', 'Lim19': 'Lim, B., Zohren, S., & Roberts', 'Liu25': 'Liu, S.', 'Nic05': 'Niculescu-Mizil',
    'Out': 'Tashman, L. J.', 'Sim19': 'Simonian, J.', 'Tob21': 'Tobek, O.', 'WuH21': 'Wu, H.',
    'Placeholder6': 'Shihao, G.', 'Hastie2': 'Hastie, T. T., & Friedman',
}
# entries printed twice: (prefix, which printed copy to delete) - the copy whose source was removed
BIB_DELETE_DUP = [('Carhart, M. M. (1997)', 1), ('Fama, E., & French, K. (2015)', 1),
                  ('Guyon, I., & Elisseeff, A. (2003)', 1), ('Hastie, T., Tibshirani, R., & Friedman, J. (2009)', 1),
                  ('Joseph, A. (2019)', 1), ('Joseph, A. (2024)', 0), ('Peng, H., Long, F., & Ding, C. (2005)', 0)]
# printed entry prefix for each source corrected in build_chapter3_references.CORRECTED (Word's APA, verified there)
CORRECTED_PREFIX = {
    'Arn23': 'Arnott, R., Kalesnik', 'Ber12': 'Bergmeir, C.', 'Bis06': 'Bishop, C. M.', 'Cam97': 'Campbell, J. Y.',
    'Nus23': 'Cakici, N.', 'Che16': 'Chen, T., & Guestrin', 'Coc11': 'Cochrane, J.', 'Cot24': 'Cotturo, P.',
    'Cov06': 'Cover, T.', 'Dix17': 'Dixon, M.', 'Fan22': 'Fang, Y.', 'Faw06': 'Fawcett, T.', 'Fen23': 'Feng, G., He',
    'Shi20': 'Gu, S., Kelly', 'Guy03': 'Guyon, I.', 'Han24': 'Han, Y.', 'Kra04': 'Kraskov, A.',
    'Kra17': 'Krauss, C., Do', 'Lak94': 'Lakonishok, J.', 'Lun17': 'Lundberg, S. M., & Lee',
    'Lun20': 'Lundberg, S., Erion', 'Mur73': 'Murphy, A.', 'Pio22': 'Piovezan, M.', 'Sae22': 'Saetia, K.',
    'Sul16': 'Sulaiman, N.', 'War63': 'Ward, J.'}
EXTRA_BIB = {
    'Gneiting, T.': ['Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and '
                     'estimation. ', ('Journal of the American Statistical Association, 102',), '(477), 359–378.'],
    'Good, I. J.': ['Good, I. J. (1952). Rational decisions. ', ('Journal of the Royal Statistical Society: Series B, '
                                                                 '14',), '(1), 107–114.'],
}
I_ = lambda s: (s,)  # noqa: E731
BIB_REPLACE = {  # printed entry prefix -> corrected entry (APA as Word prints it; 1-tuples are italic)
    'Mitchell, T. (1997)': ['Mitchell, T. (1997). ', I_('Machine Learning'), ' (1 ed.). New York: McGraw-Hill.'],
    'Lopez de Prado, M. (2018)': ['López de Prado, M. (2018). ', I_('Advances in Financial Machine Learning'),
                                  ' (1 ed.). Hoboken: John Wiley & Sons.'],
    'Lopez de Prado, M. (2020)': ['López de Prado, M. (2020). ', I_('Machine Learning for Asset Managers'),
                                  ' (1 ed.). Cambridge: Cambridge University Press.'],
    'Asness, C., Frazzini, A., & Pedersen': ['Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus '
                                             'junk. ', I_('Review of Accounting Studies, 24'), '(1), 34-112.'],
    'Daniel, K., & Moskowitz, D. J.': ['Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. ',
                                       I_('Journal of Financial Economics, 122'), '(2), 221-247.'],
    'Tibshirina, R.': ['Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. ',
                       I_('Journal of the Royal Statistical Society: Series B, 58'), '(1), 267-288.'],
    'Fama, E., & French, K. (1989)': ['Fama, E., & French, K. (1989). Business conditions and expected returns on '
                                      'stocks and bonds. ', I_('Journal of Financial Economics, 25'), '(1), 23–49.'],
    'Fama, E., & French, K. (2015)': ['Fama, E., & French, K. (2015). A five-factor asset pricing model. ',
                                      I_('Journal of Financial Economics, 116'), '(1), 1–22.'],
    'Hastie, T., Tibshirani, R., & Friedman, J. (2009)': ['Hastie, T., Tibshirani, R., & Friedman, J. (2009). ',
                                                          I_('The Elements of Statistical Learning'),
                                                          ' (2 ed.). New York: Springer.'],
    'Hosmer, D.': ['Hosmer, D., Lemeshow, S., & Sturdivant, R. (2013). ', I_('Applied Logistic Regression'),
                   ' (3 ed.). Hoboken: Wiley.'],
    'Leung, E.': ['Leung, E., Lohre, H., Mischlich, D., Shea, Y., & Stroh, M. (2021). ',
                  I_('The Promises and Pitfalls of Machine Learning for Predicting Stock Returns'),
                  '. Lancaster: SSRN.'],
    'Štrumbelj, E., & Kononenko, I. (2010). Explaining': [
        'Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature '
        'contributions. ', I_('Knowledge and Information Systems, 41'), '(3), 647–665.'],
    'Štrumbelj, E., & Kononenko, I. (2010). An efficient': [
        'Štrumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using game '
        'theory. ', I_('Journal of Machine Learning Research, 11'), ', 1-18.'],
    'Mukoyi, L.': ['Mukoyi, L., & Ogujiuba, K. K. (2023). Comparison of Multifactor Asset Pricing Models in the South '
                   'African Stock Market [2000–2016]. ', I_('Journal of Risk and Financial Management, 16'), '(1), 4.'],
    'Israel, R.': ['Israel, R., Kelly, B., & Moskowitz, T. J. (2020). Can Machines \u201cLearn\u201d Finance? ',
                   I_('Journal of Investment Management, 18'), '(2), 23–36.'],
    'Ilmanen, A., Israel': ['Ilmanen, A., Israel, R., Lee, R., Moskowitz, T. J., & Thapar, A. (2021). How do factor '
                            'premia vary over time? A century of evidence. ', I_('Journal of Investment Management, 19'),
                            '(4).'],
    'Barroso, P.': ['Barroso, P., & Santa-Clara, P. (2015). Momentum has its moments. ',
                    I_('Journal of Financial Economics, 116'), '(1), 111–120.'],
    'Haddad, V.': ['Haddad, V., Kozak, S., & Santosh, S. (2020). Factor timing. ',
                   I_('Review of Financial Studies, 33'), '(5), 1980–2018.'],
    'Asness, C. S., Chandra': ['Asness, C. S., Chandra, S., Ilmanen, A., & Israel, R. (2017). Contrarian factor timing '
                               'is deceptively difficult. ', I_('Journal of Portfolio Management, 43'), '(5), 72–87.'],
    'Arnott, R., Harvey': ['Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of '
                           'machine learning. ', I_('Journal of Financial Data Science, 1'), '(1), 64–74.'],
    'Hanauer, M. X.': ['Hanauer, M. X., & Kalsbach, T. (2023). Machine learning and the cross-section of emerging '
                       'market stock returns. ', I_('Emerging Markets Review, 55'), ', 101022.'],
    'Mutooni, R.': ['Mutooni, R., & Muller, C. (2007). Equity style timing. ', I_('Investment Analysts Journal, 36'),
                    '(65), 15–24.'],
}
BIB_INSERT = [  # (insert before entry starting with, entry)
    ('Page, D., McClelland, D., & Auret, C. (2024)', ['Page, D., McClelland, D., & Auret, C. (2022). Style rotation on '
                                                      'the JSE. ', I_('Finance Research Letters, 46'),
                                                      '(Part B), 102504.']),
    ('Friedman, J.', ['Fraser, E., & Page, M. (2000). Value and momentum strategies: Evidence from the Johannesburg '
                      'Stock Exchange. ', I_('Investment Analysts Journal, 29'), '(51), 25–35.']),
    ('Murphy, A.', ['Muller, C., & Ward, M. (2013). Style-based effects on the Johannesburg Stock Exchange: A graphical '
                    'time-series approach. ', I_('Investment Analysts Journal, 42'), '(77), 1–16.']),
    ('Lakonishok, J.', ['Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. ',
                        I_('The Annals of Statistics, 17'), '(3), 1217–1241.']),
    ('Rapach, D. E.', ['Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary '
                       'data. In R. LePage, & L. Billard (Eds.), ', I_('Exploring the Limits of Bootstrap'),
                       ' (pp. 263–270). New York: Wiley.']),
]


def bibliography(body):
    sdt = next(s for s in body.iter(w('sdt')) if s.find('w:sdtPr/w:bibliography', NS) is not None)
    paras = [p for p in sdt.iter(w('p'))]
    text = lambda p: ''.join(t.text or '' for t in p.iter(w('t')))
    gone = 0
    for tag, prefix in BIB_DELETE_PREFIX.items():
        hits = [p for p in paras if text(p).startswith(prefix)]
        assert len(hits) == 1, (tag, prefix, len(hits))
        delete_bib_entry(hits[0])
        gone += 1
    for prefix, which in BIB_DELETE_DUP:
        hits = [p for p in paras if text(p).startswith(prefix) and alive(p)]
        assert len(hits) == 2, (prefix, len(hits))
        delete_bib_entry(hits[which])
        gone += 1
    fixed = 0
    sys.path.insert(0, str(LEGACY / 'results'))
    from build_chapter3_references import CORRECTED
    for tag, prefix in CORRECTED_PREFIX.items():
        BIB_REPLACE[prefix] = CORRECTED[tag][0]
    BIB_REPLACE.update(EXTRA_BIB)
    for prefix, entry in BIB_REPLACE.items():
        hits = [p for p in paras if text(p).startswith(prefix) and alive(p)]
        assert len(hits) == 1, (prefix, len(hits))
        replace_bib_entry(hits[0], entry)
        fixed += 1
    for before, entry in BIB_INSERT:
        hits = [p for p in paras if text(p).startswith(before) and alive(p)]
        assert len(hits) == 1, before
        new = ED.new_paragraph(hits[0], [Italic(x[0]) if isinstance(x, tuple) else x for x in entry],
                               rpr=bib_rpr(hits[0]))
        hits[0].addprevious(new)
    log('R', f'Reference list: {gone} uncited or duplicate entries deleted, {fixed} corrected (e.g. Mitchell '
             f'"Not Sure ed.", López de Prado 1st ed. Cambridge, Pedersen L. H., Moskowitz T. J., Tibshirani, '
             f'"Journal", "Statistical", "Logistic", "Lancaster", Štrumbelj & Kononenko 2014, missing volumes and '
             f'issues), {len(BIB_INSERT)} added (Fraser & Page; Muller & Ward; Page et al. 2022; Künsch; Politis & Romano)')


def alive(p):
    """False once the paragraph mark carries a tracked deletion."""
    return p.find('w:pPr/w:rPr/w:del', NS) is None


def bib_rpr(p):
    for r in p.iter(w('r')):
        if r.find(w('t')) is not None and r.find(w('rPr')) is not None and r.find('w:rPr/w:i', NS) is None:
            return r.find(w('rPr'))
    return None


def delete_bib_entry(p):
    """Delete an entry's text and paragraph mark, keeping any BIBLIOGRAPHY field-code runs."""
    for r in [c for c in p if local(c) == 'r']:
        if r.find(w('t')) is not None and r.find(w('fldChar')) is None:
            ED._delete_el(r)
    ED._mark_para(p, 'del')


def replace_bib_entry(p, entry):
    rpr = bib_rpr(p)
    runs = [c for c in p if local(c) == 'r' and c.find(w('t')) is not None and c.find(w('fldChar')) is None]
    last = runs[-1]
    anchor = etree.Element(w('r'))
    last.addnext(anchor)
    for r in runs:
        ED._delete_el(r)
    for el in ED.build([Italic(x[0]) if isinstance(x, tuple) else x for x in entry], copy.deepcopy(rpr)):
        anchor.addprevious(el)
    anchor.getparent().remove(anchor)


# ======================================================================================= main
def main():
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = check_numbers()
    work = Path(tempfile.mkdtemp(prefix='thesis_corr_'))
    cal = Path(tempfile.mkdtemp(prefix='calibri_'))
    unpack(snap, work)
    with zipfile.ZipFile(CALIBRI) as z:
        z.extractall(cal)
    doc = Doc(work / 'word' / 'document.xml')
    body = doc.root.find(w('body'))
    fn = Doc(work / 'word' / 'footnotes.xml')
    src_path = next(p for p in (work / 'customXml').glob('item*.xml') if b'<b:Sources' in p.read_bytes()[:400])
    src = Doc(src_path)
    rels = Doc(work / 'word' / '_rels' / 'document.xml.rels')

    fix_sources(src.root)
    fix_citation_fields(body)
    front_matter(body, res)
    chapter1(body)
    chapter2(body)
    chapter3(body)
    chapter4(body, res)
    results_figures(body, rels.root, work / 'word' / 'media')
    chapter5(body, res)
    equations(body)
    acronyms(body)
    captions(body)
    appendices(body, rels.root, work / 'word' / 'media', cal)
    footnotes(fn.root)
    bibliography(body)
    uk_spelling(body)
    # ask Word to update fields (captions, contents, lists, citations, bibliography) when the file is opened
    settings = Doc(work / 'word' / 'settings.xml')
    if settings.root.find(w('updateFields')) is None:
        uf = etree.Element(w('updateFields'))
        uf.set(w('val'), 'true')
        later = [c for c in settings.root if local(c) in ('hdrShapeDefaults', 'footnotePr', 'endnotePr', 'compat',
                                                          'docVars', 'rsids')]
        later[0].addprevious(uf)
    for d in (doc, fn, src, rels, settings):
        d.save()
    pack(work, out)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(cal, ignore_errors=True)
    with open(HERE / 'change_log.txt', 'w', encoding='utf-8') as f:
        for item, what in LOG:
            f.write(f'[{item}] {what}\n')
    print(f'wrote {out} ({len(LOG)} log entries, {next(ED.ids) - 900000} revision marks)')


if __name__ == '__main__':
    main()
