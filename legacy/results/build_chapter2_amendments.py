"""Build the Chapter 2 (Literature Review) amendment document.

Section 2.1 restated much of the revised Section 1.1 (the CAPM opening, the factor models, the rational and
behavioural explanations, the cyclicality of factor returns, the motivation for timing and the mixed evidence)
with the same citations, and repeated itself: an older version of its first paragraph was left above the
rewritten one. At the author's request (2 November 2026) it no longer repeats Chapter 1 and focuses on the
challenges of factor timing, under the heading "Challenges of factor timing": an opening that names five
challenges, then one paragraph on each (identifying genuine premia, their uncertain drivers, the length of factor
cycles, the fragility of reported timing gains, and the complexity of the relationship between predictors and
factor returns). It is built from the material the section already had, with references back to Section 1.1.
Citations are Word citation fields; narrative citations are the authors' names followed by a field that shows the
year only (switches \\n \\t), so that they still reach the reference list.

Starts from the current thesis (a snapshot), keeps only Chapter 2 and applies the amendments in place,
highlighted, with an amendment log at the end.

Usage: python build_chapter2_amendments.py [out_dir] [thesis_snapshot.docx]
"""
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from build_chapter1_amendments import B_NS, norm
from build_chapter4_amendments import compose, find_para, fresh_copy, heading, style, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter2_Literature_Review_Minimal_Amendments_Highlighted.docx"


def cite(*pairs):
    """A parenthetical citation field: Source Manager tags and the APA text Word shows for them."""
    return [tag for tag, _ in pairs], "(" + "; ".join(shown for _, shown in pairs) + ")"


def narrative(authors, tag, year):
    """'Cochrane (2011)': the names typed, the year a citation field (author and title suppressed)."""
    return [f"{authors} ", ([f"{tag} \\n  \\t"], f"({year})")]


def build():
    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)

    # the thesis reference list (Word's bibliography) and the Source Manager, to check the cited works
    paras = list(body.iter(qn("w:p")))
    ref_head = [p for p in paras if text(p).strip() == "References" and style(p).startswith("Heading")][-1]
    master = []
    for p in paras[paras.index(ref_head) + 1:]:
        if style(p).startswith("Heading"):
            break
        if text(p).strip():
            master.append(text(p))
    with zipfile.ZipFile(SOURCE) as z:
        bib = next(etree.fromstring(z.read(n)) for n in z.namelist()
                   if n.startswith("customXml/item") and "Props" not in n and b"Sources" in z.read(n)[:400])
    sources = {s.findtext(B_NS + "Tag"): s for s in bib.findall(B_NS + "Source")}

    # keep only Chapter 2: from its "Chapter" label to the element before the next chapter label
    children = list(body)
    start = children.index(heading(body, "Literature Review"))
    end = next(i for i, el in enumerate(children) if i > start and el.tag == qn("w:p") and style(el) == "Heading1")
    first = start - 1 if style(children[start - 1]) == "Chapter" else start
    last = end - 2 if style(children[end - 1]) == "Chapter" else end - 1
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not first <= i <= last:
            body.remove(el)
    set_heading_start(document, 2)

    e = Editor(True)
    log = []
    note = lambda section, change, reason: log.append((section, change, reason))

    head = heading(body, "Foundational framework on factor timing")
    old_first = find_para(body, contains="Subsequently, empirical anomalies demonstrated")
    paragraphs = [find_para(body, contains="Subsequent empirical anomalies demonstrated"),
                  find_para(body, startswith="This study focuses on three of these academically validated factors"),
                  find_para(body, startswith="Although the theoretical underpinnings of factor premia remain contested"),
                  find_para(body, startswith="This insight underpins the concept of factor timing"),
                  find_para(body, startswith="Some studies report that valuation and momentum-based timing rules"),
                  find_para(body, contains="identify several difficulties in forecasting factor premia")]
    last_old = find_para(body, startswith="The structural challenges inherent in timing factor premia")

    ilm21 = ("Ilm21", "Ilmanen, Israel, Lee, Moskowitz, & Thapar, 2021")
    opening, genuine, drivers, cycles, fragile, complexity = paragraphs
    compose(e, opening, [
        "Section 1.1 set out the case for factor timing; this section reviews why it is difficult in practice. Five "
        "challenges recur in the literature: identifying which premia are genuine, the uncertain economic drivers of "
        "those premia, the length of factor cycles, the fragility of reported timing gains, and the complexity of the "
        "relationship between predictors and factor returns."])
    compose(e, genuine, [
        "The first challenge is identifying which premia are genuine. Asset pricing research has uncovered a rapidly "
        "expanding set of cross-sectional return predictors, which ", *narrative("Cochrane", "Coc11", 2011),
        " described as a “zoo” of factors, and subsequent work attributes much of this proliferation to "
        "insufficiently stringent statistical standards: many proposed factors lack economic justification and fail "
        "to replicate out of sample ", cite(("Har16", "Harvey, Liu, & Zhu, 2016"), ("Fen20", "Feng, Giglio, & Xiu, 2020")),
        ". Even genuine premia shrink once they are published, as investors trade on them (McLean & Pontiff, 2016). "
        "Timing a factor whose premium is spurious cannot add value, and a premium that has decayed leaves less to "
        "time. A small core set of factors nevertheless "
        "remains widely used in research and practice ",
        cite(("Asn13", "Asness, Moskowitz, & Pedersen, 2013"), ("Fam15", "Fama & French, 2015")),
        ", and the three factors in this study, value, momentum and quality, belong to it."])
    compose(e, drivers, [
        "The second challenge is that the economic drivers of the premia are uncertain. Section 1.1 noted that "
        "risk-based and behavioural explanations both have empirical support, and the two point to different "
        "predictors. If the premia compensate for risk, they should vary with business conditions, as expected "
        "returns on stocks and bonds do ", cite(("Fam89", "Fama & French, 1989")),
        "; on the JSE, size and momentum premia have been found to vary with the business cycle (Kapche Fotso, "
        "2019). If they reflect investor biases, they should be largest when mispricing is hardest to arbitrage away, "
        "since real-world constraints limit arbitrage ", cite(("Shl97", "Shleifer & Vishny, 1997")),
        ". Without a settled theory, a timing model cannot draw its predictors from a single explanation, and factor "
        "returns appear to be influenced by a multifaceted set of variables that extends beyond economic "
        "fundamentals."])
    compose(e, cycles, [
        "The third challenge is the length of factor cycles. An equal allocation across factors is a demanding "
        "benchmark: ", *narrative("DeMiguel, Garlappi and Uppal", "DeM09", 2009),
        " show that equal weighting is difficult to beat out of sample when allocating across assets. Yet value "
        "crashes and growth rallies recur, and even a diversified factor portfolio can endure multi-year "
        "underperformance ",
        cite(("Asn00", "Asness, Friedman, Krail, & Liew, 2000"), ("Arn16", "Arnott, Beck, Kalesnik, & West, 2016")),
        ". Because factor cycles are asynchronous, dynamic reallocation could improve on static diversification ",
        cite(("Ilm12", "Ilmanen & Kizer, 2012")),
        ", but cycles that last several years also mean that a sample of a few decades contains few independent "
        "episodes from which a timing model can learn, a limitation that ", *narrative("Ilmanen et al.", "Ilm21", 2021),
        " address by examining a century of data."])
    compose(e, fragile, [
        "The fourth challenge is that reported timing gains are fragile. Section 1.1 noted that the evidence on "
        "factor timing is mixed, and several features of the problem explain why. The predictable component of factor "
        "returns is small relative to their noise ", cite(("Isr20", "Israel, Kelly, & Moskowitz, 2020")),
        ", and return predictors that perform well over a full sample often fail out of sample and in later "
        "subperiods, as Welch and Goyal (2008) show for the equity premium. Because many predictors and "
        "specifications are tested on the same data, the best reported result is biased upwards unless the search "
        "itself is taken into account (White, 2000). Results are also sensitive to the predictor, evaluation period, "
        "benchmark and treatment of implementation costs ",
        cite(("Asn17", "Asness, Chandra, Ilmanen, & Israel, 2017"), ilm21), ": ",
        *narrative("Arnott et al.", "Arn16", 2016), " find relative valuations useful for timing factors, whereas ",
        *narrative("Asness et al.", "Asn17", 2017), " find that value timing adds little beyond a static allocation, "
        "and Cederburg et al. (2020) find that volatility-managed portfolios do not systematically outperform their "
        "unmanaged counterparts out of sample."])
    compose(e, complexity, [
        "The fifth challenge is the complexity of the relationship between predictors and factor returns. ",
        *narrative("Cotturo, Liu and Proner", "Cot24", 2024),
        " note that the drivers of factor premia may operate over different horizons, that the relationship between "
        "predictors and factor performance may be nonlinear and interactive ",
        cite(("Leu21", "Leung, Lohre, Mischlich, Shea, & Stroh, 2021"), ("Bry23", "Bryzgalova, Pelger, & Zhu, 2023")),
        ", and that these relationships can change across market states ", cite(ilm21),
        ". The set of candidate predictors is also large relative to the available sample (Rapach & Zhou, 2013; "
        "Feng et al., 2018). These difficulties have led recent research to machine learning methods, which do not "
        "require a pre-specified functional form and can approximate nonlinear relationships between predictors and "
        "returns ", cite(("Shi20", "Gu, Kelly, & Xiu, 2020")),
        ", but they do not imply that the most flexible model will perform best, which motivates the comparison of "
        "models with different levels of functional flexibility."])
    for p in (old_first, last_old):
        p.getparent().remove(p)
    e.replace(head, "Foundational framework on factor timing", "Challenges of factor timing")

    # Section 2.3.1 is general to factor timing, not to machine learning: what it adds is now in Section 2.1
    old_231 = [heading(body, "Challenges with time-series data:"),
               find_para(body, startswith="Literature establishes that momentum, value and quality are economically"),
               find_para(body, startswith="Predictive relationships shift over time."),
               find_para(body, startswith="Timing results depend on which predictors are tested")]
    for p in old_231:
        p.getparent().remove(p)

    # Chapter opening: what the review is for, and an outline of Sections 2.1-2.4 (as for Chapter 1)
    chapter_opening = fresh_copy(complexity)
    compose(e, chapter_opening, [
        "This chapter reviews the literature that motivates the study's design. Its purpose is to establish why factor "
        "timing is difficult, what machine learning can contribute and where it tends to fail, and what has not yet "
        "been tested in the South African market. Section 2.1 sets out the challenges of factor timing that apply "
        "whatever forecasting method is used. Section 2.2 reviews the machine learning model classes relevant to the "
        "study, from regularised linear models to tree ensembles and deep learning. Section 2.3 reviews the "
        "international evidence on machine learning for factor timing, including the trade-off between model "
        "complexity and robustness, differences across markets and the way predictive and economic value are "
        "evaluated. Section 2.4 reviews the South African evidence and sets out the gap that the study addresses."])
    head.addprevious(chapter_opening)
    note("2 (opening)", "New chapter opening: what the review is for (why factor timing is difficult, what machine "
         "learning can contribute and where it fails, what has not been tested in South Africa) and an outline of "
         "Sections 2.1–2.4. The chapter title was followed directly by Section 2.1.", "Consistency with Chapter 1 "
         "and Section 4.1: a chapter opens by saying what it does.")

    note("2.1", "Heading changed from 'Foundational framework on factor timing' to 'Challenges of factor timing', "
         "and the section rewritten around five challenges: an opening that names them, then one paragraph on each "
         "(identifying genuine premia; their uncertain economic drivers; the length of factor cycles; the fragility "
         "of reported timing gains; the complexity of the relationship between predictors and factor returns). Six "
         "paragraphs, was eight.", "Author: the section should focus on the challenges of factor timing and not "
         "repeat Chapter 1.")
    note("2.1", "Kept from the old text: the factor zoo (Cochrane, 2011; Harvey, Liu, & Zhu, 2016; Feng, Giglio, & "
         "Xiu, 2020) and the core set of factors; limits to arbitrage (Shleifer & Vishny, 1997) and the multifaceted "
         "drivers of factor returns; naïve diversification (DeMiguel, Garlappi, & Uppal, 2009), value crashes and "
         "asynchronous cycles; the sensitivity of timing gains; Cotturo et al.'s difficulties in forecasting factor "
         "premia and the case for machine learning. Moved in from Section 2.3.1 (deleted): the decay of premia after "
         "publication (McLean & Pontiff, 2016); the JSE premia's variation with the business cycle (Kapche Fotso, "
         "2019); the low signal-to-noise ratio of factor returns (now cited to Israel, Kelly, & Moskowitz, 2020); the "
         "failure of full-sample predictors in later subperiods (Welch & Goyal, 2008); data snooping (White, 2000); "
         "and valuation timing as an example of predictor sensitivity (Arnott et al., 2016, against Asness et al., "
         "2017). Added: that risk-based premia should vary with business conditions (Fama & French, 1989, already a "
         "source in the thesis) and behavioural premia with the limits to arbitrage; that long cycles leave few "
         "independent episodes to learn from (Ilmanen et al., 2021, examine a century of data); what Cederburg et al. "
         "(2020) find, which the old text cited without saying.", "Author: focus on the challenges.")
    note("2.3.1", "Section 'Challenges with time-series data' deleted: it reviewed challenges of factor timing in "
         "general, not of machine learning, and repeated Section 1.1 and the new Section 2.1. What it added is now "
         "in Section 2.1 (see above). Not kept: its first paragraph (the factors and their variation over time, the "
         "motivation for timing; Section 1.1); the momentum-crash example (cited in Section 1.1); the duplicated "
         "citation group '(Harvey, Liu, & Zhu, 2016) (White, 2000; Harvey et al., 2016)'; and 'JSE style-rotation "
         "alphas largely disappear once momentum is added to the benchmark (Page, McClelland, & Auret, 2024)', "
         "which could not be confirmed: Page et al. (2024) report that their tree models beat a style-momentum "
         "benchmark. Restore it with the correct source if the paper (or Page et al., 2022) supports it.",
         "Author: no repetition; Section 2.3 is about machine learning applications.")
    note("2.1", "Deleted as repeats of Chapter 1: the CAPM opening and 'single-factor models could not adequately "
         "explain the cross-section of returns'; the account of the factor models (Fama and French, 1992, 1993, "
         "2015; Jegadeesh and Titman, 1993; Carhart, 1997) and of the quality and low-volatility premia (the factors "
         "are introduced in Section 1.1 and defined in Section 3.2); the rational and behavioural explanations with "
         "the same citations and the conclusion that the premia have no single origin; the cyclicality of factor "
         "returns; 'This insight underpins the concept of factor timing …'; 'Some studies report that valuation and "
         "momentum-based timing rules improve … The literature therefore supports testing factor timing out of "
         "sample'. Huang et al. (2020) is no longer cited: its sentence repeated Section 1.1.",
         "Author: no repetition of Chapter 1.")
    note("2.1", "Deleted repeats within the section: the older version of the first paragraph, left above the "
         "rewritten one (Prof Gilbert's comments C42–C44 were on that copy); 'This study focuses on three of these "
         "academically validated factors …', which repeated the end of the paragraph before; the last paragraph "
         "('The structural challenges inherent in timing factor premia …'), merged into the fifth challenge because "
         "both cited Gu, Kelly and Xiu (2020) and Leung et al. (2021) for nonlinearity and Ilmanen et al. (2021) for "
         "changing relationships; the duplicated citation groups of the timing-evidence paragraph (each given once "
         "as a field and once typed).", "Repetition.")
    note("2.1", "Narrative citations written as 'Cochrane (2011)', not '(Cochrane, 2011) described …': the names are "
         "typed and the year is a citation field that suppresses the author, so the work still reaches the reference "
         "list. The factor zoo is attributed to Cochrane (2011), the presidential address Prof Gilbert refers to "
         "(source Coc11). Fama and French (2015) is cited from source Fam15, not the duplicate Placeholder1 that "
         "printed the title.", "Prof Gilbert's comments C42 (narrative citations), C43 (remove title) and C44 "
         "(reference the factor zoo properly).")
    # cited works that are missing from the thesis reference list, or have no source at all
    tags = []
    for it in (it for p in paragraphs for it in p.iter(qn("w:instrText"))):
        if "CITATION" in (it.text or ""):
            tags += [it.text.split("CITATION")[1].split()[0]] + re.findall(r"\\m (\S+)", it.text)
    off = []
    for tag in dict.fromkeys(tags):
        s = sources[tag]
        last_name, year = next(s.iter(B_NS + "Last")).text, s.findtext(B_NS + "Year")
        if not any(norm(t).startswith(norm(last_name)) and f"({year}" in t for t in master):
            off.append(f"{last_name} ({year}, {tag})")
    note("2.1 / references", "Cited with a source but missing from the thesis reference list (they appear when the "
         "bibliography is updated, References > Update Citations and Bibliography): " + ("; ".join(off) or "none")
         + ". Typed because the thesis has no source for them: Cederburg, O'Doherty, Wang and Yan (2020), Rapach and "
         "Zhou (2013), 'Feng et al. (2018)', McLean and Pontiff (2016), Kapche Fotso (2019), Welch and Goyal (2008) "
         "and White (2000). Add them in the Source Manager and replace the typed citations with fields; the details "
         "of Kapche Fotso (2019) are also marked as outstanding in the literature-review reference file. 'Feng et al. (2018)' may be the 2018 working-paper version of Feng, He, Polson and Xu (2023), which "
         "is in the Source Manager (Fen23); if so, cite Fen23.", "Reference list.")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
