"""Render the working paper to PDF.

    python paper/build_paper.py                 # -> paper/lifecycle_asset_allocation.pdf
    python paper/build_paper.py --out other.pdf

Two passes are used so that the table of contents can carry real page numbers.
Every quoted number is resolved from ``results/tables`` at build time (see
``paper/facts.py``), so the paper cannot drift away from the pipeline that
produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors                                   # noqa: E402
from reportlab.lib.units import cm                                 # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame,  # noqa: E402
                                KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph,
                                Spacer, Table)
from reportlab.platypus.tableofcontents import TableOfContents     # noqa: E402

import content                                                     # noqa: E402
import style as st                                                 # noqa: E402
from facts import Facts                                            # noqa: E402

SHORT_TITLE = "Beyond the Status Quo: A Computational Re-Examination"


class PaperDoc(BaseDocTemplate):
    """A document with a plain title page and a running-head body page."""

    def __init__(self, filename: str, running_head: str = "",
                 **kwargs: Any) -> None:
        super().__init__(filename, pagesize=st.PAGE_SIZE,
                         leftMargin=st.MARGIN_LEFT, rightMargin=st.MARGIN_RIGHT,
                         topMargin=st.MARGIN_TOP, bottomMargin=st.MARGIN_BOTTOM,
                         title=kwargs.pop("doc_title", SHORT_TITLE),
                         author=kwargs.pop("doc_author", ""),
                         subject=kwargs.pop("doc_subject", ""), **kwargs)
        #: What the running head prints. Each paper carries its own short
        #: title; hard-coding one paper's here put the companion study's
        #: name on every page of the other.
        self.running_head = running_head or SHORT_TITLE
        frame = Frame(self.leftMargin, self.bottomMargin, self.width,
                      self.height, id="text", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="title", frames=[frame], onPage=self._title_page),
            PageTemplate(id="body", frames=[frame], onPage=self._body_page),
        ])
        self.styles: Dict[str, Any] = {}

    # -- page furniture ---------------------------------------------------
    def _title_page(self, canvas: Any, doc: Any) -> None:
        # The front matter can spill onto a second page, which is still a
        # title page; the provenance note belongs under the title once, not
        # under every page it happens to cover.
        if canvas.getPageNumber() != 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(st.SOFT_RULE)
        canvas.setLineWidth(0.5)
        y = doc.bottomMargin - 0.7 * cm
        canvas.line(doc.leftMargin, y, doc.leftMargin + doc.width, y)
        canvas.setFont(self.styles["running"].fontName, 7.6)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(doc.leftMargin + doc.width / 2.0,
                                 y - 0.42 * cm,
                                 "Every figure in this document is computed "
                                 "from the sources described in Section 3.")
        canvas.restoreState()

    def _body_page(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(self.styles["running"].fontName, 7.6)
        canvas.setFillColor(colors.HexColor("#666666"))
        top = doc.bottomMargin + doc.height + 0.55 * cm
        canvas.drawString(doc.leftMargin, top, self.running_head)
        canvas.drawRightString(doc.leftMargin + doc.width, top,
                               f"{canvas.getPageNumber()}")
        canvas.setStrokeColor(st.SOFT_RULE)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, top - 0.18 * cm,
                    doc.leftMargin + doc.width, top - 0.18 * cm)
        canvas.restoreState()

    # -- table of contents ------------------------------------------------
    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        name = getattr(flowable, "style", None)
        if name is None:
            return
        level = {"h1": 0, "h2": 1}.get(name.name)
        if level is None:
            return
        text = flowable.getPlainText()
        self.notify("TOCEntry", (level, text, self.page))


class Context:
    """Everything the content module needs: styles, numbers, counters."""

    def __init__(self, facts: Facts, styles: Dict[str, Any]) -> None:
        self.f = facts
        self.s = styles
        self.width = st.FRAME_WIDTH
        self._figure_no = 0
        self._table_no = 0
        self._equation_no = 0
        self.figure_index: List[str] = []
        self.table_index: List[str] = []
        #: ``anchor -> (kind, number)`` for floats prose refers to by name.
        #: A literal "Table 4" in a note is a reference nothing checks, and
        #: this paper shipped one that had pointed at the wrong table since
        #: the numbering last moved. Named floats are resolved after the
        #: story is assembled, so they survive both a renumbering and a
        #: forward reference.
        self.float_anchors: Dict[str, Tuple[str, int]] = {}

    # -- text -------------------------------------------------------------
    @staticmethod
    def resolve(text: str) -> str:
        """Public name for :meth:`_r`, for callers that build their own
        flowables. The references list did exactly that and printed a raw
        ``#section`` token on the page for it."""
        return Context._r(text)

    @staticmethod
    def _r(text: str) -> str:
        """Resolve ``#section`` tokens against the paper's reading order.

        Every string that reaches the page goes through here, so a heading
        and a cross-reference to it can never disagree: both are rendered
        from ``content.SECTION_ORDER``. Code blocks are the one exception --
        a shell command may legitimately contain a ``#``.
        """
        return content.resolve_sections(text)

    def h1(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["h1"])

    def h2(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["h2"])

    def h3(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["h3"])

    def p(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["body"])

    def note(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["note"])

    def quote(self, text: str) -> Paragraph:
        return Paragraph(self._r(text), self.s["quote"])

    def code(self, text: str) -> Paragraph:
        body = text.strip("\n").replace("&", "&amp;").replace("<", "&lt;") \
            .replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;")
        return Paragraph(body, self.s["code"])

    def bullets(self, items: Sequence[str]) -> List[Paragraph]:
        return [Paragraph(f"•&nbsp;&nbsp;{self._r(item)}", self.s["bullet"])
                for item in items]

    def equation(self, text: str, tag: bool = True) -> Paragraph:
        if tag:
            self._equation_no += 1
            text = f"{text}&nbsp;&nbsp;&nbsp;&nbsp;({self._equation_no})"
        return Paragraph(text, self.s["equation"])

    def gap(self, height: float = 6.0) -> Spacer:
        return Spacer(1, height)

    # -- floats -----------------------------------------------------------
    def figure(self, name: str, caption: str, max_height: float = 21.0 * cm,
               width_scale: float = 1.0, anchor: str = "") -> List[Flowable]:
        # Figures are authored at the width of this text column (see
        # ``PAGE_WIDTH_IN`` in ``src/plots.py``), so width binds and the
        # image lands on the page at 1:1 with its labels the size they were
        # drawn. ``max_height`` is only a guard against a runaway figure.
        self._figure_no += 1
        self._anchor(anchor, "Figure", self._figure_no)
        caption = self._r(caption)
        self.figure_index.append(f"Figure {self._figure_no}. {caption}")
        image = st.figure_flowable(self.f.figure(name),
                                   self.width * width_scale, max_height)
        head = Paragraph(f"Figure {self._figure_no}.", self.s["caption_head"])
        body = Paragraph(caption, self.s["caption"])
        return [KeepTogether([image, head, body])]

    def table(self, rows: Sequence[Sequence[str]], caption: str,
              note: str | None = None, anchor: str = "",
              **kwargs: Any) -> List[Flowable]:
        self._table_no += 1
        self._anchor(anchor, "Table", self._table_no)
        caption = self._r(caption)
        note = self._r(note) if note else note
        # Cells carry cross-references too -- Appendix A cites the section
        # each parameter is varied in -- so they are resolved as well.
        rows = [[self._r(cell) if isinstance(cell, str) else cell
                 for cell in row] for row in rows]
        self.table_index.append(f"Table {self._table_no}. {caption}")
        head = Paragraph(f"Table {self._table_no}. {caption}",
                         self.s["caption_head"])
        table = st.make_table(rows, self.s, total_width=self.width, **kwargs)
        parts: List[Flowable] = [head, self.gap(2), table]
        if note:
            parts.append(Paragraph(f"<i>Notes.</i> {note}", self.s["note"]))
        else:
            parts.append(self.gap(10))
        return [KeepTogether(parts)] if len(rows) <= 14 else parts

    def _anchor(self, name: str, kind: str, number: int) -> None:
        """Record a float under a name prose can refer to.

        Two floats sharing a name would make ``@table:name`` mean whichever
        was built last, which is the silent-wrong-pointer failure the anchors
        exist to remove, so it is an error rather than a last-write-wins.
        """
        if not name:
            return
        if name in self.float_anchors:
            kind_was, number_was = self.float_anchors[name]
            raise SystemExit(
                f"float anchor {name!r} is claimed twice: by {kind_was} "
                f"{number_was} and again by {kind} {number}")
        self.float_anchors[name] = (kind, number)

    def rule(self) -> Flowable:
        return st.HorizontalRule(self.width)


def _walk(flowables: Sequence[Any]) -> List[Any]:
    """Flatten nested KeepTogether/Table containers into their leaf flowables."""
    out: List[Any] = []
    for item in flowables:
        out.append(item)
        for attr in ("_content", "_cellvalues"):
            nested = getattr(item, attr, None)
            if isinstance(nested, (list, tuple)):
                flat = [x for row in nested
                        for x in (row if isinstance(row, (list, tuple)) else [row])]
                out.extend(_walk(flat))
    return out


def build(out_path: str, config_path: str = "config.yaml") -> Path:
    fonts = st.register_fonts()
    styles = st.build_styles(fonts)
    facts = Facts(config_path=config_path)
    ctx = Context(facts, styles)

    story = content.story(ctx)

    prose = "".join(flowable.getPlainText() for flowable in _walk(story)
                    if isinstance(flowable, Paragraph))
    missing = st.missing_glyphs(prose, fonts["Serif"])
    if missing:
        print(f"warning: the text face cannot render {missing!r}; these would "
              "be drawn as black boxes", file=sys.stderr)

    doc = PaperDoc(out_path,
                   doc_title="Beyond the Status Quo: A Computational "
                             "Re-Examination of Lifecycle Asset Allocation",
                   doc_author="StockChartIR replication project",
                   doc_subject="Lifecycle asset allocation, block bootstrap, "
                               "certainty equivalent consumption")
    doc.styles = styles
    # An identity pass here -- nothing is trimmed from the companion study --
    # but it is the guard that keeps the two documents building the same way,
    # so a float dropped from this one later cannot leave a hole either.
    renumber_floats(story, ctx.float_anchors)
    doc.multiBuild(story)
    return Path(out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(Path(__file__).resolve().parent
                                    / "lifecycle_asset_allocation.pdf"))
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    path = build(args.out, args.config)
    dangling = check_cross_references(str(path))
    if dangling:
        print("Cross-references that point at nothing:")
        for problem in dangling:
            print("  -", problem)
        raise SystemExit(
            "the document cites sections that do not exist; fix them rather "
            "than shipping a pointer a reader cannot follow")
    skipped = check_float_numbering(str(path))
    if skipped:
        for problem in skipped:
            print("  -", problem)
        raise SystemExit("the float numbering has holes in it")
    size = path.stat().st_size / 1024.0
    print(f"wrote {path} ({size:,.0f} KB)")
    return 0


def check_cross_references(pdf_path: str) -> list[str]:
    """Every "Section N" and "Section N.M" reference must resolve to a heading.

    Sections have been renumbered several times as extensions were added, and
    a dangling cross-reference is invisible to the author and misleading to a
    reader who follows it. This reads the built document back and checks them,
    so the failure mode is a build error rather than a wrong pointer.

    **What it cannot catch.** A reference that resolves to the wrong heading.
    "Section 16.1" pointed at the cross-section limitation for several drafts
    after that subsection had been renumbered to 19.1; because 16.1 still
    existed -- as an unrelated housing subsection -- no existence check could
    have flagged it. That class of error needs a reader, and this function is
    not one. It catches the reference that points at nothing.
    """
    import re
    from pypdf import PdfReader

    text = "\n".join((page.extract_text() or "")
                     for page in PdfReader(pdf_path).pages)
    headings = set()
    for match in re.finditer(r"\n\s*(\d{1,2}(?:\.\d{1,2}){0,2})[ .]", text):
        headings.add(match.group(1))
    for match in re.finditer(r"\n(\d{1,2})\. [A-Z]", text):
        headings.add(match.group(1))

    problems = []
    for match in re.finditer(r"Section (\d{1,2}(?:\.\d{1,2}){0,2})", text):
        ref = match.group(1)
        # A short build resolves a section it does not carry against the
        # companion document and says so. That is a pointer a reader can
        # follow, not a dangling one, so it is not this check's business.
        tail = " ".join(text[match.end():match.end() + 40].split())
        if tail.startswith("of the companion study"):
            continue
        if ref not in headings:
            context = text[max(0, match.start() - 90):match.start() + 60]
            problems.append(f"Section {ref} -> no such heading "
                            f"(...{context.strip()!r})")
    return sorted(set(problems))


# ---------------------------------------------------------------------------
# Float renumbering
# ---------------------------------------------------------------------------
#: How a caption head announces its float. ``Context.figure`` writes
#: ``"Figure 7."`` on a line of its own; ``Context.table`` writes
#: ``"Table 7. The caption"`` in one paragraph. One pattern covers both.
_CAPTION_HEAD = re.compile(r"^(Figure|Table) (\d+)\.")

#: An in-text reference to a float. Deliberately narrower than the caption
#: pattern -- it must not need a trailing full stop, because prose says
#: "Table 8 reports" as often as "Table 8."
_FLOAT_REFERENCE = re.compile(r"\b(Figure|Table) (\d+)\b")

#: A reference to a float by name: ``@table:bootstrap_fidelity``. Prose uses
#: these instead of literal numbers, and they are resolved once the story is
#: assembled and renumbered, so they hold across a trim and across a forward
#: reference to a float that has not been built yet.
_FLOAT_ANCHOR_REFERENCE = re.compile(r"@(figure|table):([a-z0-9_]+)\b")


def _slots(parts: Iterable[Flowable]) -> Iterator[Tuple[List[Any], int]]:
    """Every flowable in the story, as the list holding it and its index.

    Distinct from :func:`_walk`, which flattens the story into leaves for
    reading. Yielding the container and the index rather than the flowable
    is what lets a caller replace a paragraph *in place*, which renumbering
    has to do. ``KeepTogether`` is the only nesting the paper builds, and it
    keeps its children in ``_content``.
    """
    items = parts if isinstance(parts, list) else list(parts)
    for i in range(len(items)):
        yield items, i
        # Depth-first, in place: a caption inside a ``KeepTogether`` has to
        # be visited where it sits, because the numbering is assigned in
        # document order and a stack would hand back the nested ones last.
        # Re-read the slot -- the caller is allowed to have replaced it, and
        # it is the replacement whose children the document will show.
        inner = getattr(items[i], "_content", None)
        if isinstance(inner, list):
            yield from _slots(inner)


def renumber_floats(story: List[Flowable],
                    anchors: Dict[str, Tuple[str, int]] | None = None,
                    ) -> Dict[str, Dict[int, int]]:
    """Close the gaps a trimmed section leaves in the float numbering.

    The counters live on the :class:`Context` and advance when a figure or
    table is *built*. The short paper builds a section in full and then drops
    subsections from it, which removes the floats but cannot rewind the
    counter -- so the document printed "Figure 1" and then "Figure 5", and
    jumped from Table 3 to Table 8 to Table 23. Renumbering at source is not
    an option: the sections are shared with the companion study, and a
    counter that knew what was going to be trimmed would have to be told
    twice, once per paper.

    So this runs on the assembled story instead, where what survived is
    known. It reads the caption heads in document order, builds the old-to-new
    map, and rewrites both the heads and every reference in the prose.

    Returns the map, keyed ``"Figure"`` and ``"Table"``, so the caller can
    report what moved. A document with nothing trimmed gets an identity map
    and no rewritten paragraphs.
    """
    order: Dict[str, List[int]] = {"Figure": [], "Table": []}
    for items, i in _slots(story):
        flowable = items[i]
        if getattr(getattr(flowable, "style", None), "name", "") \
                != "caption_head":
            continue
        match = _CAPTION_HEAD.match(flowable.getPlainText().strip())
        if match:
            order[match.group(1)].append(int(match.group(2)))

    mapping = {kind: {old: new for new, old in enumerate(olds, start=1)}
               for kind, olds in order.items()}

    #: ``name -> "Table 8"``, at the numbers the document will actually
    #: print. A float that was trimmed away has no entry, so a reference to
    #: one is reported rather than left pointing into the companion study's
    #: numbering.
    resolved = {}
    for name, (kind, old) in (anchors or {}).items():
        if old in mapping[kind]:
            resolved[name] = f"{kind} {mapping[kind][old]}"

    dangling: List[str] = []

    def rewrite_number(match: "re.Match[str]") -> str:
        kind, old = match.group(1), int(match.group(2))
        return f"{kind} {mapping[kind].get(old, old)}"

    def rewrite_anchor(match: "re.Match[str]") -> str:
        name = match.group(2)
        if name not in resolved:
            dangling.append(name)
            return match.group(0)
        return resolved[name]

    for items, i in _slots(story):
        flowable = items[i]
        text = getattr(flowable, "text", None)
        if not isinstance(text, str):
            continue
        fixed = _FLOAT_ANCHOR_REFERENCE.sub(rewrite_anchor, text)
        fixed = _FLOAT_REFERENCE.sub(rewrite_number, fixed)
        if fixed != text:
            items[i] = Paragraph(fixed, flowable.style,
                                 bulletText=flowable.bulletText)

    if dangling:
        known = ", ".join(sorted(resolved)) or "none"
        raise SystemExit(
            "the document refers to floats that it does not print: "
            + ", ".join(sorted(set(dangling)))
            + f" (anchors this build registered: {known})")
    return mapping


def check_float_numbering(pdf_path: str) -> list[str]:
    """No figure or table number may be skipped, and none may repeat.

    :func:`renumber_floats` runs on the story; this reads the built document
    back, so a float whose caption the renumberer failed to recognise is
    caught rather than shipped. The two checks are deliberately independent:
    the first is a rewrite, the second is evidence it worked.
    """
    from pypdf import PdfReader

    text = "\n".join((page.extract_text() or "")
                     for page in PdfReader(pdf_path).pages)
    problems = []
    unresolved = sorted({m.group(0)
                         for m in _FLOAT_ANCHOR_REFERENCE.finditer(text)})
    if unresolved:
        problems.append(
            "these float anchors reached the page unresolved: "
            + ", ".join(unresolved))
    for kind in ("Figure", "Table"):
        seen = sorted({int(m.group(2)) for m in _FLOAT_REFERENCE.finditer(text)
                       if m.group(1) == kind})
        if not seen:
            continue
        missing = [n for n in range(1, max(seen) + 1) if n not in seen]
        if missing:
            problems.append(
                f"{kind} numbering skips {missing} (highest is {max(seen)}), "
                f"so the document cites floats it does not print")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
