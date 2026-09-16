"""The Internet Appendix: the apparatus the paper points at and does not print.

Six subsections were written for the short paper and moved out of it. They
are not offcuts -- each answers an objection the paper raises against
itself, and each is cited from the body -- but each costs a page a referee
would rather spend on the result, and none of them is the argument.

They cannot go to the companion study the way the inherited trims do,
because the companion does not contain them: they were written here. So
they are rendered into their own document, from the same writers that would
have put them in the paper, numbered A.1 to A.6 in the order the paper
refers to them. :data:`paper.short.APPENDIX_NUMBER` is the single source of
those numbers, so a reference in the paper and a heading in the appendix
cannot disagree.

Built by ``python -m paper.appendix``.
"""

from __future__ import annotations

from typing import Any, List

from reportlab.platypus import Flowable, NextPageTemplate, PageBreak, Paragraph

from . import build_paper as _bp
from . import short as sh

ct = _bp.content

TITLE: str = "Internet Appendix"
SUBTITLE: str = ("Supporting material for “The Pension and the Drawdown "
                 "Rule: Portfolio Choice under a Means-Tested Public "
                 "Pension”")


def front(ctx: Any) -> List[Flowable]:
    """Title, and what this document is for."""
    s = ctx.s
    out: List[Flowable] = [
        Paragraph(TITLE, s["title"]),
        Paragraph(SUBTITLE, s["subtitle"]),
        Paragraph("This version: 16 September 2026", s["subtitle"]),
    ]
    out.append(ctx.p(
        "This appendix carries six pieces of apparatus the paper cites and "
        "does not print. Each answers an objection the paper raises against "
        "itself rather than one a reader is left to raise: what the panel "
        "costs the corner the mechanism predicts, whether that corner "
        "survives a different objective, whether the opposite corner is a "
        "corner at all, what an interval on the <i>difference</i> between "
        "two cells looks like when an interval on their levels is not the "
        "claim, what the grid does when scored over a real lifespan rather "
        "than a fixed horizon, and how much of the delete-one interval's "
        "precision was the independence it assumes."))
    out.append(ctx.p(
        "Every number here is read from the same pipeline output at build "
        "time as the corresponding number in the paper, so the two cannot "
        "drift apart. Section numbers are the paper's own: a reference to "
        "Section A.4 in the paper is the section headed A.4 here."))
    out.append(NextPageTemplate("body"))
    out.append(PageBreak())
    return out


def _renumber(parts: List[Flowable], key: str, number: str) -> List[Flowable]:
    """Retitle a moved subsection as its appendix number.

    The heading arrives as ``7.3 What the panel costs this corner``; here it
    has to read ``A.1 What the panel costs this corner``, and the number has
    to come from the map the paper resolves its references against rather
    than from a second list kept in step by hand.
    """
    out = list(parts)
    for i, flowable in enumerate(out):
        if sh._rank_of(flowable) != 2:
            continue
        text = sh._text_of(flowable)
        body = text.split(" ", 1)[1] if " " in text else text
        out[i] = ctx_h2(flowable, f"{number} {body}")
        break
    return out


def ctx_h2(template: Any, text: str) -> Flowable:
    """A heading in the same style as the one it replaces."""
    return Paragraph(text, template.style)


def story(ctx: Any) -> List[Flowable]:
    """The appendix, assembled from the subsections the paper moved out."""
    ct.COMPANION = {"name": "the companion study",
                    "numbers": sh.LONG_NUMBER_ALL}
    ct.ABSENT = sh.TRIMMED_ANCHORS
    ct.RENUMBERED = sh.RENUMBERED
    with sh.renumbered():
        parts: List[Flowable] = []
        parts += front(ctx)
        built = {"incidence": sh.incidence(ctx), "ordering": sh.ordering(ctx)}
        moved = {key: sh._appendix_only(value, key)
                 for key, value in built.items()}
        for anchor, _ in sh.APPENDIX_ORDER:
            key, tail = anchor.split(".")
            number = sh.APPENDIX_NUMBER[anchor]
            block = _one(moved.get(key, []), key, tail)
            if not block:
                raise SystemExit(
                    f"the appendix has nothing for {anchor}: the paper "
                    f"moved a different set of subsections than this "
                    f"document expects to print")
            parts += _renumber(block, key, number)
    ct.COMPANION = {}
    ct.ABSENT = ()
    ct.RENUMBERED = {}
    return parts


def _one(parts: List[Flowable], key: str, tail: str) -> List[Flowable]:
    """The single subsection numbered ``key.tail`` out of a moved block."""
    want = f"{sh.SHORT_ORDER.index(key) + 1}.{tail}"
    out: List[Flowable] = []
    taking = False
    for flowable in parts:
        rank = sh._rank_of(flowable)
        if rank <= 2:
            text = sh._text_of(flowable)
            starts = text.startswith(want + " ")
            if taking and not starts:
                break
            taking = starts
        if taking:
            out.append(flowable)
    return out


def _paper_floats(config_path: str) -> dict:
    """``anchor -> "Table 8 of the paper"``, from the paper's own numbering.

    Built by assembling the paper's story and renumbering it exactly as its
    own build does, so the appendix quotes the number a reader will find
    rather than one this module decided on.
    """
    from .facts import Facts
    from . import style as st

    styles = st.build_styles(st.register_fonts())
    ctx = _bp.Context(Facts(config_path=config_path), styles)
    parts = sh.story(ctx)
    mapping = _bp.renumber_floats(parts, ctx.float_anchors)
    out = {}
    for name, (kind, old) in ctx.float_anchors.items():
        if old in mapping[kind]:
            out[name] = f"{kind} {mapping[kind][old]} of the paper"
    return out


def build(out_path: str = "paper/internet_appendix.pdf",
          config_path: str = "config.yaml") -> str:
    """Render the Internet Appendix and check its cross-references."""
    from .facts import Facts
    from . import style as st

    fonts = st.register_fonts()
    styles = st.build_styles(fonts)
    ctx = _bp.Context(Facts(config_path=config_path), styles)
    doc = _bp.PaperDoc(
        out_path, running_head=TITLE, doc_title=f"{TITLE}: {SUBTITLE}",
        doc_author="Lifecycle investing study",
        doc_subject="Supporting material on means-tested pensions and "
                    "portfolio choice")
    doc.styles = styles
    parts = story(ctx)
    # Prose that moved here still cites tables that stayed in the body. The
    # numbers come from the paper's own build rather than being guessed,
    # which is why this reads them off a freshly numbered story rather than
    # naming them here.
    _bp.ELSEWHERE = _paper_floats(config_path)
    try:
        _bp.renumber_floats(parts, ctx.float_anchors)
    finally:
        _bp.ELSEWHERE = {}
    doc.multiBuild(parts)
    return out_path


if __name__ == "__main__":
    print("wrote", build())
