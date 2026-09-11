"""The short paper: one thesis, drawn from the sections that carry it.

The long study answers twenty-eight questions. A journal wants one, and the
one worth asking here is the only extension that overturns the result being
replicated: an asset-tested public pension reverses the all-equity
prescription, and it does so by removing a floor rather than by taxing
wealth.

This module does not fork the prose. It selects the sections that carry
that argument, renumbers them for the shorter reading order, and writes new
front matter, an introduction and a conclusion around them. Every number
still comes from the same tables the long paper reads, so the two can never
disagree about a figure.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Sequence, Tuple

from reportlab.platypus import Flowable, PageBreak, Paragraph

# `build_paper` puts its own directory on sys.path and does a top-level
# `import content`, so `paper.content` and `content` are two different module
# objects. The renderer resolves cross-references against *its* copy, so this
# module has to reach through to the same one -- patching the other silently
# does nothing, which is exactly what it did.
from . import build_paper as _bp

ct = _bp.content

#: The reading order of the short paper. The middle three sections are the
#: argument; the rest is the minimum a reader needs to evaluate it.
SHORT_ORDER: Tuple[str, ...] = (
    "introduction",
    "data",
    "methods",
    "baseline",
    "pension",
    "leisure",
    "longevity",
    "tax",
    "limitations",
    "conclusion",
)

#: Every section's number in the long study, so a reference this paper does
#: not contain can still point a reader somewhere real.
LONG_NUMBER_ALL: Dict[str, int] = {
    k: ct.section_number(k) for k in ct.SECTION_ORDER}

TITLE = "The Floor Beneath the Portfolio"
SUBTITLE = ("Public Pension Design and the All-Equity Lifecycle Portfolio")

#: Added because the long paper's eighteen do not cover this argument. The
#: referee's list, in the order a reader meets the ideas.
EXTRA_REFERENCES: Tuple[str, ...] = (
    'Cocco, J. F., Gomes, F. J., and Maenhout, P. J. (2005). "Consumption '
    'and Portfolio Choice over the Life Cycle." <i>Review of Financial '
    'Studies</i>, 18(2), 491–533. The income profile used here is taken '
    'from this estimation.',
    'Gomes, F., and Michaelides, A. (2005). "Optimal Life-Cycle Asset '
    'Allocation: Understanding the Empirical Evidence." <i>Journal of '
    'Finance</i>, 60(2), 869–904.',
    'Viceira, L. M. (2001). "Optimal Portfolio Choice for Long-Horizon '
    'Investors with Nontradable Labor Income." <i>Journal of Finance</i>, '
    '56(2), 433–470.',
    'Campbell, J. Y., and Viceira, L. M. (2002). <i>Strategic Asset '
    'Allocation: Portfolio Choice for Long-Term Investors.</i> Oxford '
    'University Press.',
    'Benzoni, L., Collin-Dufresne, P., and Goldstein, R. S. (2007). '
    '"Portfolio Choice over the Life-Cycle when the Stock and Labor Markets '
    'are Cointegrated." <i>Journal of Finance</i>, 62(5), 2123–2167.',
    'Dahlquist, M., Setty, O., and Vestman, R. (2018). "On the Asset '
    'Allocation of a Default Pension Fund." <i>Journal of Finance</i>, '
    '73(4), 1893–1936.',
    'Hubbard, R. G., Skinner, J., and Zeldes, S. P. (1995). "Precautionary '
    'Saving and Social Insurance." <i>Journal of Political Economy</i>, '
    '103(2), 360–399. The mechanism this paper measures in a portfolio '
    'setting is the one they identify for saving.',
    'Milevsky, M. A., and Huang, H. (2011). "Spending Retirement on Planet '
    'Vulcan: The Impact of Longevity Risk Aversion on Optimal Withdrawal '
    'Rates." <i>Financial Analysts Journal</i>, 67(2), 45–58. The '
    'amortisation rule that wins in Section '
    f'{SHORT_ORDER.index("longevity") + 1} is theirs.',
    'Dimson, E., Marsh, P., and Staunton, M. (2002). <i>Triumph of the '
    'Optimists: 101 Years of Global Investment Returns.</i> Princeton '
    'University Press.',
    'Yaari, M. E. (1965). "Uncertain Lifetime, Life Insurance, and the '
    'Theory of the Consumer." <i>Review of Economic Studies</i>, 32(2), '
    '137–150.',
)


@contextlib.contextmanager
def renumbered():
    """Number the sections for *this* paper while the story is built.

    ``content`` resolves every ``#key`` against one module-level map. The
    short paper is a different reading order, so the map is swapped for the
    duration and restored afterwards -- which keeps a cross-reference
    written for the long paper pointing at the right place in this one,
    rather than at a number that is not in the document.
    """
    original = ct._SECTION_NUMBER
    ct._SECTION_NUMBER = {k: i + 1 for i, k in enumerate(SHORT_ORDER)}
    try:
        yield
    finally:
        ct._SECTION_NUMBER = original


def _resolvable(ctx: Any, text: str) -> Paragraph:
    return ctx.p(text)


def front(ctx: Any) -> List[Flowable]:
    """Title, abstract and keywords for the short paper."""
    f, s = ctx.f, ctx.s
    gaps = f.table("pension_gap").set_index("system")
    au = gaps.loc["australia_as_legislated"]
    base = gaps.loc["us_social_security"]
    untested = gaps.loc["age_pension_untested"]
    bite = f.table("leisure_means_test_bite")
    legis = bite[bite["household"] == "as legislated"].iloc[0]
    ret = f.table("leisure_return_sweep")
    wide = ret.pivot(index="assumed_return", columns="system", values="cec")
    best_us = float(wide["us"].idxmax())
    best_au = float(wide["au_as_legislated"].idxmax())
    best_gap = 100.0 * (wide["au_as_legislated"].max()
                        / wide["us"].max() - 1.0)

    out: List[Flowable] = [
        Paragraph(TITLE, s["title"]),
        Paragraph(SUBTITLE, s["subtitle"]),
        Paragraph("This version: 11 September 2026", s["subtitle"]),
        Paragraph("Abstract", s["h1_plain"]),
    ]
    out.append(ctx.p(
        f"A lifecycle investor holding only equities, split between domestic "
        f"and international markets, is said to dominate the age-declining "
        f"glide path embedded in target-date funds. We show that this "
        f"prescription is a property of the public pension the investor "
        f"retires onto, not of the return process. On a 16-country panel of "
        f"real returns spanning 1890–2020, simulated with a calendar-joint "
        f"block bootstrap, the all-equity portfolio leads the target-date "
        f"fund by {float(base['gap_pct']):.2f}% in certainty-equivalent "
        f"retirement consumption under the United States' earnings-related "
        f"schedule. Replacing that schedule with Australia's — a "
        f"means-tested Age Pension alongside a compulsory 12% "
        f"Superannuation Guarantee — reverses the ordering to "
        f"{float(au['gap_pct']):.2f}%, and the target-date fund wins."))
    out.append(ctx.p(
        f"The mechanism is not the means test's taper. This household holds "
        f"{float(legis['median_wealth_multiple']):.1f} times average "
        f"earnings against a cut-off of "
        f"{float(legis['cutoff_multiple']):.1f}, and "
        f"{float(legis['share_above_cutoff']):.0%} of simulated households "
        f"are past the test before it is applied. What the means test "
        f"removes is an unconditional real annuity, and with it the floor "
        f"under bad outcomes: mean retirement consumption <i>rises</i> "
        f"{float(au['mean_lift_pct']):+.0f}% while the fifth percentile "
        f"<i>falls</i> {float(au['p5_lift_pct']):+.0f}%. Paying the same "
        f"flat pension unconditionally, means-testing nothing, leaves the "
        f"ranking intact at {float(untested['gap_pct']):+.2f}%, which "
        f"isolates the test rather than the pension's level."))
    out.append(ctx.p(
        f"A floor need not come from a pension. Under an amortisation "
        f"withdrawal rule — which divides the balance by the years "
        f"remaining and therefore cannot deplete it — ruin falls to zero "
        f"and the ordering reverses again: scored at each system's own best "
        f"assumed return ({best_us:.0%} for the United States, "
        f"{best_au:.0%} for Australia) the Australian household leads by "
        f"{best_gap:+.1f}%. The all-equity prescription is therefore "
        f"conditional on two institutions at once, the pension and the "
        f"drawdown rule, and is silent about a household that has neither a "
        f"guaranteed income nor a rule that cannot run out."))
    out.append(ctx.p(
        "<b>Keywords:</b> lifecycle asset allocation; public pension "
        "design; means testing; target-date funds; certainty-equivalent "
        "consumption; decumulation. <b>JEL:</b> G11, G51, H55, D14."))
    out.append(PageBreak())
    return out


def introduction(ctx: Any) -> List[Flowable]:
    f = ctx.f
    gaps = f.table("pension_gap").set_index("system")
    au, base = gaps.loc["australia_as_legislated"], gaps.loc["us_social_security"]
    out: List[Flowable] = [ctx.h1("#introduction. Introduction")]
    out.append(ctx.p(
        "The default investment vehicle of the modern retirement system is "
        "the target-date fund, and it embodies one proposition: that an "
        "investor should hold less equity as they age. Anarkulova, Cederburg "
        "and O'Doherty (2023) challenge it directly, showing that when "
        "returns are drawn in blocks from the international historical "
        "record rather than i.i.d. from post-war US data, a fixed all-equity "
        "portfolio beats the glide path on almost every metric an investor "
        "would care about."))
    out.append(ctx.p(
        "That result is derived for an investor who receives the United "
        "States' Social Security: an earnings-related benefit, paid whatever "
        "else the retiree owns. This paper asks what the prescription "
        "becomes for an investor who does not."))
    out.append(ctx.h2("#introduction.1 What we find"))
    out.append(ctx.p(
        f"<b>The ordering reverses under an asset-tested pension.</b> The "
        f"all-equity lead of {float(base['gap_pct']):.2f}% becomes "
        f"{float(au['gap_pct']):.2f}% when the American schedule is replaced "
        f"by Australia's, and the de-risking glide path takes first place. "
        f"Nothing about the return panel changes between those two runs; the "
        f"objective function does."))
    out.append(ctx.p(
        "<b>The mechanism is the floor, not the taper.</b> A means test is "
        "usually discussed as an implicit tax on wealth, and Australia's is "
        "steep enough to qualify — a dollar of assessable assets costs 7.8% "
        "of pension a year, more than domestic equity earns on average in "
        "this panel. But that reading does not survive checking where the "
        "household stands: it is past the cut-off before the test is "
        "applied, so the taper reaches it only in the left tail, after a "
        "portfolio has already fallen. What the means test removes is the "
        "unconditional annuity itself. A retiree standing on a guaranteed "
        "floor can carry the equity tail; one standing on their portfolio "
        "alone cannot, and a risk-averse objective prices that difference "
        "heavily."))
    out.append(ctx.p(
        "<b>A floor need not come from a pension.</b> The reversal is "
        "conditional on a withdrawal rule that can run out. Under an "
        "amortisation rule the balance is divided by the years remaining, "
        "ruin falls to zero, and the household manufactures its own floor "
        "out of the larger portfolio compulsory saving bought it. The "
        "ordering then reverses a second time. Which means the all-equity "
        "prescription is conditional on two institutions rather than one."))
    out.append(ctx.h2("#introduction.2 Relation to the literature"))
    out.append(ctx.p(
        "The lifecycle portfolio-choice literature since Cocco, Gomes and "
        "Maenhout (2005) and Gomes and Michaelides (2005) treats public "
        "pension income as a bond-like endowment that crowds fixed income "
        "out of the financial portfolio. That is right for an "
        "earnings-related schedule. It is the wrong intuition for an "
        "asset-tested one, where the endowment is withdrawn precisely as "
        "the portfolio grows, and where — as Hubbard, Skinner and Zeldes "
        "(1995) show for saving — the binding feature is the floor the "
        "transfer provides rather than its expected level. Dahlquist, Setty "
        "and Vestman (2018) solve for a default fund's allocation given a "
        "pension system; we vary the system and hold the fund menu fixed, "
        "which isolates the same interaction from the other side."))
    out.append(ctx.p(
        "On the decumulation side the rule that reverses our reversal is "
        "the actuarial or amortisation rule of Milevsky and Huang (2011). "
        "Its interaction with pension design appears not to have been "
        "measured: the withdrawal-rule literature conditions on a fixed "
        "income floor, and the pension literature conditions on a fixed "
        "withdrawal rule."))
    out.append(ctx.p(
        f"Our panel is the Jordà–Schularick–Taylor macrohistory database, "
        f"the same source the replicated study uses, and the same 16 "
        f"developed markets Dimson, Marsh and Staunton (2002) cover for a "
        f"comparable period. Every number below is regenerated from that "
        f"panel by the pipeline described in Section "
        f"{SHORT_ORDER.index('methods') + 1}; the fuller robustness "
        f"apparatus, and the twenty-seven extensions not needed for this "
        f"argument, are in the companion study."))
    return out


def conclusion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    gaps = f.table("pension_gap").set_index("system")
    au, base = gaps.loc["australia_as_legislated"], gaps.loc["us_social_security"]
    untested = gaps.loc["age_pension_untested"]
    out: List[Flowable] = [ctx.h1("#conclusion. Conclusion")]
    out.append(ctx.p(
        f"The case for a fixed all-equity lifecycle portfolio is a case "
        f"about an investor with a guaranteed real income underneath it. "
        f"Give the same investor the same returns and an asset-tested "
        f"pension instead, and the lead of {float(base['gap_pct']):.2f}% "
        f"becomes {float(au['gap_pct']):.2f}% and the de-risking glide path "
        f"— the design the literature spends its length arguing against — "
        f"wins."))
    out.append(ctx.p(
        f"The reason is worth separating from the obvious one. It is "
        f"tempting to attribute the reversal to the means test's taper, "
        f"which is steep enough to act as a wealth tax. It cannot be that: "
        f"this household is past the cut-off before the test applies. It is "
        f"the loss of the floor. Paying the same pension to everyone, with "
        f"no test at all, leaves the ordering intact "
        f"({float(untested['gap_pct']):+.2f}%) even though it pays this "
        f"household less than the American schedule does. Withdrawing a "
        f"pension at the margin and paying a smaller one are different "
        f"interventions, and only the first reorders portfolios."))
    out.append(ctx.p(
        "The second finding qualifies the first and should travel with it. "
        "A withdrawal rule that cannot deplete the portfolio supplies the "
        "floor the pension no longer does, and the ordering reverses again. "
        "So the reversal is not a fact about Australia. It is a fact about "
        "any retiree whose income in the bad states depends on the "
        "portfolio itself — which includes a saver under an asset test, and "
        "excludes one drawing on an annuity, a defined-benefit scheme, or a "
        "rule that divides by the years remaining."))
    out.append(ctx.p(
        "Two implications follow for default design. A plan sponsor "
        "choosing a glide path is implicitly choosing it against a pension "
        "schedule, and the same fund is not the right default in a "
        "means-tested system and an earnings-related one. And the "
        "drawdown default matters as much as the portfolio default: in a "
        "system without a floor, supplying one through the withdrawal rule "
        "recovers what the pension no longer provides."))
    out.append(ctx.p(
        "We state these as implications of a calibrated model. It carries "
        "no behavioural constraints, prices no annuity, and — as Section "
        f"{SHORT_ORDER.index('limitations') + 1} sets out — rests on a "
        "sixteen-country panel whose jackknife interval is wide enough that "
        "the direction of these results is far better established than "
        "their size."))
    return out


def story(ctx: Any) -> List[Flowable]:
    """The short paper, assembled from the sections that carry its thesis."""
    ct.COMPANION = {"name": "the companion study", "numbers": LONG_NUMBER_ALL}
    with renumbered():
        parts: List[Flowable] = []
        parts += front(ctx)
        parts += introduction(ctx)
        parts += ct.section_data(ctx)
        parts += ct.section_methods(ctx)
        parts += ct.section_baseline(ctx)
        parts += ct.section_pension(ctx)
        parts += ct.section_leisure(ctx)
        parts += ct.section_longevity(ctx)
        parts += ct.section_tax(ctx)
        parts += ct.section_limitations(ctx)
        parts += conclusion(ctx)
        parts += references(ctx)
    ct.COMPANION = {}
    return parts


def references(ctx: Any) -> List[Flowable]:
    out: List[Flowable] = [PageBreak(),
                           Paragraph("References", ctx.s["h1_plain"])]
    for entry in sorted(set(ct.REFERENCES) | set(EXTRA_REFERENCES),
                        key=lambda e: e.split("(")[0].strip()):
        out.append(Paragraph(ctx.resolve(entry), ctx.s["reference"]))
    return out


def build(out_path: str = "paper/floor_beneath_the_portfolio.pdf",
          config_path: str = "config.yaml") -> str:
    """Render the short paper and check its cross-references."""
    from .facts import Facts
    from . import style as st

    fonts = st.register_fonts()
    styles = st.build_styles(fonts)
    ctx = _bp.Context(Facts(config_path=config_path), styles)
    doc = _bp.PaperDoc(
        out_path, doc_title=f"{TITLE}: {SUBTITLE}",
        doc_author="Lifecycle investing study",
        doc_subject="Public pension design and lifecycle asset allocation")
    doc.styles = styles
    doc.multiBuild(story(ctx))
    dangling = _bp.check_cross_references(out_path)
    if dangling:
        raise SystemExit("dangling cross-references:\n  "
                         + "\n  ".join(dangling))
    return out_path


if __name__ == "__main__":
    print("wrote", build())
