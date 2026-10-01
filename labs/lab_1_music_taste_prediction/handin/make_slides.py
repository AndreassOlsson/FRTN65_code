"""Builds handin/slides.pptx from handin/slides.md with python-pptx.

    uv run --with python-pptx python handin/make_slides.py

python-pptx is pulled in for this one run rather than added to the
package, since only the hand-in needs it. slides.md is the source: each
slide is a `# title`, then any of a `## subtitle`, bullets, one image,
one table, stat lines (`> 0.830 | label`) and `Note:` lines for the
speaker notes. The layout follows from what a slide holds: the first is
the title slide, an image goes right of the bullets (above them when it
is a wide panel), a table spans the
slide, stats sit as a row of cards above the bullets.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
SOURCE, OUT = HERE / "slides.md", HERE / "slides.pptx"

INK = RGBColor(0x26, 0x1C, 0x33)  # deep plum, the dominant colour
MUTED = RGBColor(0x6B, 0x5E, 0x78)
AMBER = RGBColor(0xC9, 0x8A, 0x12)  # the accent, numbers only
CARD = RGBColor(0xF3, 0xEF, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY, CODE = "Cambria", "Calibri", "Courier New"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


@dataclass
class Slide:
    title: str = ""
    subtitle: str = ""
    bullets: list[str] = field(default_factory=list)
    image: str | None = None
    table: list[list[str]] = field(default_factory=list)
    stats: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse(text: str) -> list[Slide]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    slides = []
    for chunk in re.split(r"^---$", text, flags=re.M):
        s = Slide()
        for line in chunk.strip().splitlines():
            line = line.strip()
            if line.startswith("## "):
                s.subtitle = line[3:]
            elif line.startswith("# "):
                s.title = line[2:]
            elif line.startswith("- "):
                s.bullets.append(line[2:])
            elif m := re.match(r"!\[.*?\]\((.+?)\)", line):
                s.image = m.group(1)
            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    s.table.append(cells)
            elif line.startswith("> "):
                value, label = (p.strip() for p in line[2:].split("|", 1))
                s.stats.append((value, label))
            elif line.startswith("Note:"):
                s.notes.append(line[5:].strip())
        if s.title:
            slides.append(s)
    return slides


def runs(paragraph, text: str, size: int, color: RGBColor, bold: bool = False):
    """Text with `code` spans set in a monospace face."""
    for i, part in enumerate(re.split(r"`([^`]*)`", text)):
        if not part:
            continue
        r = paragraph.add_run()
        r.text = part
        r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
        r.font.name = CODE if i % 2 else BODY


def textbox(slide, x, y, w, h):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    return tf


def bullet(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.28)))
    pPr.set("indent", str(-Inches(0.28)))
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•")


def add_bullets(slide, items, x, y, w, h, size=17, color=INK):
    tf = textbox(slide, x, y, w, h)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet(p)
        p.space_after = Pt(10)
        runs(p, item, size, color)


def add_title(slide, s: Slide, color=INK):
    tf = textbox(slide, MARGIN, Inches(0.45), W - 2 * MARGIN, Inches(0.9))
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = s.title
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(32), True, color


def add_stats(slide, stats, y, dark=False):
    n = len(stats)
    gap = Inches(0.35)
    w = min((W - 2 * MARGIN - gap * (n - 1)) / n, Inches(4.2))
    for i, (value, label) in enumerate(stats):
        x = MARGIN + i * (w + gap)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, Inches(1.45))
        card.adjustments[0] = 0.08
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(0x3A, 0x2C, 0x4A) if dark else CARD
        card.line.fill.background()
        card.shadow.inherit = False
        tf = textbox(slide, x + Inches(0.25), y + Inches(0.12), w - Inches(0.5), Inches(1.25))
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = value
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(34), True, AMBER
        q = tf.add_paragraph()
        runs(q, label, 13, WHITE if dark else MUTED)


def add_image(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, Emu(pw), Emu(ph))


def aspect(path) -> float:
    with Image.open(path) as im:
        return im.size[0] / im.size[1]


def add_table(slide, rows, y):
    n_rows, n_cols = len(rows), len(rows[0])
    height = Inches(0.42) * n_rows
    shape = slide.shapes.add_table(n_rows, n_cols, MARGIN, y, W - 2 * MARGIN, height)
    table = shape.table
    # first column and any long text column get the room
    lengths = [max(len(r[c]) for r in rows) for c in range(n_cols)]
    total = sum(max(l, 6) for l in lengths)
    for c, l in enumerate(lengths):
        table.columns[c].width = int((W - 2 * MARGIN) * max(l, 6) / total)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (CARD if r % 2 else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.1)
            p = cell.text_frame.paragraphs[0]
            runs(p, text, 14, WHITE if r == 0 else INK, bold=(r == 0 or c == 0))


def notes(slide, s: Slide):
    if s.notes:
        slide.notes_slide.notes_text_frame.text = " ".join(s.notes)


def background(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def build(slides: list[Slide], out: Path = OUT) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    for i, s in enumerate(slides):
        slide = prs.slides.add_slide(blank)
        if i == 0:
            background(slide, INK)
            tf = textbox(slide, MARGIN, Inches(1.5), W - 2 * MARGIN, Inches(2.2))
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = s.title
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(44), True, WHITE
            q = tf.add_paragraph()
            q.space_before = Pt(14)
            runs(q, s.subtitle, 20, RGBColor(0xC9, 0xBF, 0xD4))
            if s.stats:
                add_stats(slide, s.stats, Inches(4.6), dark=True)
            notes(slide, s)
            continue
        background(slide, WHITE)
        add_title(slide, s)
        top = Inches(1.5)
        if s.stats:
            add_stats(slide, s.stats, top)
            top += Inches(1.8)
        if s.table:
            add_table(slide, s.table, top)
        elif s.image and aspect(HERE / s.image) > 2:
            # a wide panel figure gets the slide's width, the bullets go under it
            fig_h = Inches(3.3)
            add_image(slide, HERE / s.image, MARGIN, top, W - 2 * MARGIN, fig_h)
            below = top + fig_h + Inches(0.3)
            add_bullets(slide, s.bullets, MARGIN, below, W - 2 * MARGIN, H - below - Inches(0.3), size=15)
        elif s.image:
            split = Inches(5.4)
            add_bullets(slide, s.bullets, MARGIN, top, split - MARGIN, H - top - MARGIN, size=16)
            add_image(slide, HERE / s.image, split + Inches(0.3), top, W - split - Inches(0.3) - MARGIN, H - top - Inches(0.4))
        else:
            size = 18 if s.stats else 19
            add_bullets(slide, s.bullets, MARGIN, top, W - 2 * MARGIN, H - top - MARGIN, size=size)
        notes(slide, s)
    prs.save(out)
    return prs


def main(argv=None) -> None:
    slides = parse(SOURCE.read_text())
    build(slides)
    print(f"{len(slides)} slides written to {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main(sys.argv[1:])
