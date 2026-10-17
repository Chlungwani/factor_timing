"""Apply clean thesis formatting to a Pandoc-generated Chapter 4 DOCX."""

from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


PAGE_USABLE_INCHES = 6.27


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=65, start=100, bottom=65, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def remove_heading_numbering(document):
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[style_name]
        if style.element.pPr is not None:
            num_pr = style.element.pPr.find(qn("w:numPr"))
            if num_pr is not None:
                style.element.pPr.remove(num_pr)


def clear_headers_and_footers(document):
    """Remove template-only header/footer artwork from the standalone draft."""
    seen = set()
    for section in document.sections:
        story_parts = (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        )
        for story in story_parts:
            element = story._element
            marker = id(element)
            if marker in seen:
                continue
            seen.add(marker)
            for child in list(element):
                element.remove(child)
            element.append(OxmlElement("w:p"))


def configure_styles(document):
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11.5)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    tokens = {
        "Heading 1": (18, 14, 8),
        "Heading 2": (14, 12, 5),
        "Heading 3": (12, 9, 3),
    }
    for style_name, (size, before, after) in tokens.items():
        style = document.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = document.styles["Caption"]
    caption.font.name = "Calibri"
    caption.font.size = Pt(9.5)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor(0, 0, 0)
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)


def table_widths(table):
    column_count = len(table.columns)
    weights = []
    for column_index in range(column_count):
        max_length = max(
            len(cell.text.replace("\n", " ").strip())
            for cell in table.columns[column_index].cells
        )
        weights.append(max(8, min(max_length, 34)))

    if column_count >= 4:
        weights[0] = max(weights[0], 18)

    total = sum(weights)
    widths = [PAGE_USABLE_INCHES * weight / total for weight in weights]
    minimum = 0.62 if column_count >= 6 else 0.78
    deficit = sum(max(0.0, minimum - width) for width in widths)
    widths = [max(minimum, width) for width in widths]
    if deficit:
        widest = max(range(column_count), key=lambda index: widths[index])
        widths[widest] = max(minimum, widths[widest] - deficit)
    return widths


def format_tables(document):
    for table in document.tables:
        table_style_names = {style.name for style in document.styles if style.type == 3}
        table.style = "Table Grid" if "Table Grid" in table_style_names else "Table"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        widths = table_widths(table)

        for column_index, width in enumerate(widths):
            for cell in table.columns[column_index].cells:
                cell.width = Inches(width)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT
                        if column_index == 0
                        else WD_ALIGN_PARAGRAPH.CENTER
                    )
                    for run in paragraph.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(8.5 if len(table.columns) >= 6 else 9.5)

        for row in table.rows:
            tr_pr = row._tr.get_or_add_trPr()
            cant_split = OxmlElement("w:cantSplit")
            tr_pr.append(cant_split)

        if table.rows:
            header = table.rows[0]
            tr_pr = header._tr.get_or_add_trPr()
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
            for cell in header.cells:
                set_cell_shading(cell, "D9E2F3")
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True


def format_paragraphs_and_images(document):
    for index, paragraph in enumerate(document.paragraphs):
        text = paragraph.text.strip()
        is_table_caption = bool(re.match(r"^Table \d+\.\d+\.\s", text))
        is_figure_caption = bool(re.match(r"^Figure \d+\.\d+\.\s", text))

        if is_table_caption:
            paragraph.style = document.styles["Caption"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                run.font.bold = True

        if is_figure_caption:
            paragraph.style = document.styles["Caption"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_together = True
            for run in paragraph.runs:
                run.font.bold = True

        if text.startswith(("Panel A:", "Panel B:")):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_after = Pt(3)
            for run in paragraph.runs:
                run.font.italic = True

        if paragraph.style.name == "Normal" and text and not (is_table_caption or is_figure_caption or text.startswith(("Panel A:", "Panel B:"))):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        if index == 0:
            paragraph.paragraph_format.space_after = Pt(12)

        if any(run._r.xpath(".//w:drawing") for run in paragraph.runs):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(3)

    max_width = Inches(PAGE_USABLE_INCHES)
    max_height = Inches(8.6)
    for shape in document.inline_shapes:
        width_scale = max_width / shape.width
        height_scale = max_height / shape.height
        scale = min(1.0, width_scale, height_scale)
        shape.width = int(shape.width * scale)
        shape.height = int(shape.height * scale)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: format_chapter4_draft_docx.py INPUT.docx OUTPUT.docx")

    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    document = Document(source)

    document.core_properties.title = "Chapter 4: Results"
    document.core_properties.subject = "Final CPCV and walk-forward model results"

    for section in document.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.3)
        section.bottom_margin = Cm(2.3)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    remove_heading_numbering(document)
    clear_headers_and_footers(document)
    configure_styles(document)
    format_tables(document)
    format_paragraphs_and_images(document)

    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(target)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
