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
    "model",
    "data",
    "methods",
    "baseline",
    "pension",
    "leisure",
    "incidence",
    "longevity",
    "tax",
    "limitations",
    "conclusion",
)

#: Sections this paper writes itself rather than selecting from the long
#: study. Every other entry in :data:`SHORT_ORDER` is an inherited section,
#: reopened by :func:`_reopen` on this paper's argument.
OWN: Tuple[str, ...] = ("introduction", "model", "incidence", "limitations",
                        "conclusion")

#: The inherited sections whose long-study title does not say what they are
#: doing in *this* paper, and the title that does.
RETITLED: Dict[str, str] = {
    "pension": "One Country's Pension Reverses the Result",
    "leisure": "What a Year of Retirement Is Worth Without a Floor",
    "longevity": "A Rule That Cannot Run Out Restores It",
}

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
    optimum = f.table("incidence_optimum")
    cost = 100.0 * (float(optimum["cec_lifetime"].iloc[-1])
                    / float(optimum["cec_lifetime"].iloc[0]) - 1.0)
    band = f.table("incidence_band_profile")
    arms = list(dict.fromkeys(band["arm"])) if "arm" in band else []
    base_arm = band[band["arm"] == arms[0]] if arms else band
    rule_arm = band[band["arm"] == arms[-1]] if len(arms) > 2 else base_arm
    bound = base_arm[base_arm["median_wealth"] <= float(band["cutoff"].iloc[0])]
    taper_equity = float(bound["equity"].median()) if len(bound) else 0.0
    rule_equity = float(rule_arm["equity"].median())
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
        f"Two further results are needed before that reading can be "
        f"trusted. Charging the Superannuation Guarantee to wages rather "
        f"than to the employer costs {abs(cost):.1f}% of lifetime "
        f"certainty-equivalent consumption and leaves the balance at "
        f"retirement, and therefore the retiree's whole problem, exactly "
        f"unchanged — so the cross-system comparisons are bracketed by "
        f"that figure rather than pinned. And scaling the balance until "
        f"the assets test does bind, the household wants "
        f"{taper_equity:.0%} equity at every position under a fixed real "
        f"withdrawal and {rule_equity:.0%} under a "
        f"percentage-of-balance rule. The portfolio a means-tested retiree "
        f"should hold is not a property of the means test; it is a "
        f"property of the means test and the withdrawal rule jointly."))
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
    untested = gaps.loc["age_pension_untested"]
    bite = f.table("leisure_means_test_bite")
    legis = bite[bite["household"] == "as legislated"].iloc[0]
    band = f.table("incidence_band_profile")
    arms = list(dict.fromkeys(band["arm"])) if "arm" in band else []
    base_arm = band[band["arm"] == arms[0]] if arms else band
    rule_arm = band[band["arm"] == arms[-1]] if len(arms) > 2 else base_arm
    optimum = f.table("incidence_optimum")
    cost = abs(100.0 * (float(optimum["cec_lifetime"].iloc[-1])
                        / float(optimum["cec_lifetime"].iloc[0]) - 1.0))
    taper = float(f.cfg["lifecycle"].get("pension_taper", 0.078))

    out: List[Flowable] = [ctx.h1("#introduction. Introduction")]
    out.append(ctx.p(
        "The default investment vehicle of the modern retirement system is "
        "the target-date fund, and it embodies one proposition: that an "
        "investor should hold less equity as they age. Anarkulova, Cederburg "
        "and O'Doherty (2023) challenge it directly, showing that when "
        "returns are drawn in blocks from the international historical "
        "record rather than i.i.d. from post-war US data, a fixed all-equity "
        "portfolio beats the glide path on almost every metric an investor "
        "would care about. The finding has been influential precisely "
        "because it is robust to the things a reader first reaches for: it "
        "survives higher risk aversion, a longer horizon, and a return "
        "panel that includes every disaster the twentieth century "
        "supplied."))
    out.append(ctx.p(
        "It is not robust to the thing nobody reaches for, which is the "
        "pension. That result is derived for an investor who receives the "
        "United States' Social Security: an earnings-related benefit, paid "
        "whatever else the retiree owns. Roughly a third of OECD members "
        "instead pay a residence-based benefit that is withdrawn against "
        "the retiree's own assets or income, Australia's Age Pension among "
        "them. This paper asks what the all-equity prescription becomes "
        "for an investor under that second kind of scheme, and finds that "
        "it reverses."))
    out.append(ctx.p(
        "The exercise is deliberately narrow. Nothing about the return "
        "process changes between our two arms: the same panel, the same "
        "bootstrap, the same simulated lifetimes, the same household, the "
        "same fund menu. Only the benefit formula moves. Whatever "
        "difference appears is therefore attributable to the pension and "
        "to nothing else, which is what lets the paper say <i>why</i> the "
        "prescription reverses rather than only <i>that</i> it does."))
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
    out.append(ctx.p(
        f"<b>Who pays for compulsory saving changes the level and not the "
        f"portfolio.</b> Australia's Superannuation Guarantee is levied on "
        f"the employer, and modelling it that way hands the Australian arm "
        f"income the American arm never receives. Charging it to wages "
        f"instead costs {cost:.1f}% of lifetime certainty-equivalent "
        f"consumption — so every comparison here is bracketed by that "
        f"figure — but it leaves the balance at the pension age, and "
        f"therefore the entire retirement problem, exactly unchanged. The "
        f"contribution is made either way. That is an identity rather than "
        f"an estimate, and it means the incidence question and the "
        f"portfolio question are separate questions."))
    out.append(ctx.p(
        f"<b>On a household the test actually binds, the portfolio is a "
        f"property of the withdrawal rule, not of the pension.</b> The "
        f"household simulated here retires on "
        f"{float(legis['median_wealth_multiple']):.1f} times average "
        f"earnings, so we scale the arriving balance until it sits inside "
        f"the taper band and ask again. Spending a fixed real amount, it "
        f"wants no equity at any balance the test can reach; spending a "
        f"share of the current balance, it wants "
        f"{float(rule_arm['equity'].median()):.0%} equity at every one of "
        f"them. The reason is mechanical and, once seen, obvious: under a "
        f"fixed real rule a good return is never spent, so it accumulates "
        f"into assessable assets, the pension is withdrawn against them at "
        f"{taper:.1%} a year, and consumption does not rise at all. The "
        f"upside is confiscated and the downside is not."))
    out.append(ctx.h2("#introduction.2 Contribution"))
    out.append(ctx.p(
        "Three things here are new, and it is worth separating them from "
        "what is replication. That an all-equity portfolio beats a glide "
        "path on an international block bootstrap is Anarkulova, Cederburg "
        "and O'Doherty's result, and Section "
        f"{SHORT_ORDER.index('baseline') + 1} reproduces it rather than "
        f"claiming it."))
    out.append(ctx.p(
        "First, the <i>conditionality</i>: the prescription is a property "
        "of the pension the investor retires onto, and we can say which "
        "feature of the pension by paying the same benefit "
        "unconditionally, which leaves the ordering intact at "
        f"{float(untested['gap_pct']):+.2f}%. Withdrawing a pension at the "
        f"margin and paying a smaller one are different interventions, and "
        f"only the first reorders portfolios. Second, the <i>mechanism</i>: "
        f"Section {SHORT_ORDER.index('model') + 1} writes the retiree's "
        f"budget line as three regimes and shows that inside the taper band "
        f"a means test is insurance rather than a wealth tax — a "
        f"prediction sharp enough to fail, which under one of the two "
        f"withdrawal rules it does. Third, the <i>interaction</i>: the "
        f"portfolio and the drawdown rule are one decision when the pension "
        f"is asset-tested, and we measure how much of one."))
    out.append(ctx.h2("#introduction.3 Relation to the literature"))
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
        "The means-testing literature itself is largely about saving and "
        "labour supply rather than portfolio choice, and it has long "
        "understood the taper as an implicit tax. Hubbard, Skinner and "
        "Zeldes (1995) is the closest antecedent to what we find: their "
        "asset-tested transfer discourages saving not because of its "
        "expected level but because of the floor it puts under "
        "consumption. We measure the same feature acting on the "
        "composition of the portfolio rather than its size, and find that "
        "it acts in the opposite direction to the tax intuition — the "
        "taper, inside the band, makes equity cheaper rather than dearer."))
    out.append(ctx.p(
        f"Our panel is the Jordà–Schularick–Taylor macrohistory database, "
        f"the same source the replicated study uses, and the same 16 "
        f"developed markets Dimson, Marsh and Staunton (2002) cover for a "
        f"comparable period. Every number below is regenerated from that "
        f"panel by the pipeline described in Section "
        f"{SHORT_ORDER.index('methods') + 1}; the fuller robustness "
        f"apparatus, and the extensions not needed for this argument, are "
        f"in the companion study."))
    out.append(ctx.h2("#introduction.4 Roadmap"))
    out.append(ctx.p(
        f"Section {SHORT_ORDER.index('model') + 1} sets out the retiree's "
        f"budget line under an asset test and derives the prediction the "
        f"rest of the paper tests. Sections "
        f"{SHORT_ORDER.index('data') + 1} and "
        f"{SHORT_ORDER.index('methods') + 1} describe the panel and the "
        f"simulation. Section {SHORT_ORDER.index('baseline') + 1} "
        f"reproduces the all-equity result under the American schedule, "
        f"and Section {SHORT_ORDER.index('pension') + 1} reverses it under "
        f"the Australian one. Section "
        f"{SHORT_ORDER.index('leisure') + 1} separates the floor from the "
        f"taper; Section {SHORT_ORDER.index('incidence') + 1} charges the "
        f"guarantee and moves the household onto the test; Section "
        f"{SHORT_ORDER.index('longevity') + 1} restores the floor through "
        f"the withdrawal rule; and Section "
        f"{SHORT_ORDER.index('tax') + 1} rules out the tax treatment as an "
        f"explanation. Section {SHORT_ORDER.index('limitations') + 1} says "
        f"what would change these conclusions."))
    return out


def model(ctx: Any) -> List[Flowable]:
    """The kinked budget line, and the prediction it makes.

    The referee's first objection to the earlier draft was that the paper
    had a result and no model: the reversal was reported, not derived, and
    a reader had no way to tell which feature of the pension was doing the
    work. Two lines of algebra fix that, and they make a prediction sharp
    enough to fail -- which, in one of the three arms below, it does.
    """
    f = ctx.f
    cfg = f.cfg
    ss = cfg["lifecycle"]
    taper = float(ss.get("pension_taper", 0.078))
    free = float(ss.get("pension_free_area", 3.01))
    rate = float(ss.get("pension_full_rate", 0.293))
    profile = f.table("incidence_band_profile")
    cut = float(profile["cutoff"].iloc[0])

    out: List[Flowable] = [ctx.h1("#model. A Kinked Budget Line")]
    out.append(ctx.p(
        "Write <i>W</i> for the balance a household brings to retirement, "
        "<i>R</i> for the gross real return it earns, <i>A</i> for the "
        "assets-test free area, <i>b</i> for the full pension and "
        "<i>&tau;</i> for the rate at which the pension is withdrawn "
        "against assets above <i>A</i>. Retirement consumption is the "
        "portfolio plus the benefit, and the benefit is a kinked function "
        "of the portfolio. Three regimes follow directly:"))
    out.append(ctx.equation(
        "<i>c</i> = <i>WR</i> + <i>b</i>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(below the free area)"))
    out.append(ctx.equation(
        "<i>c</i> = (1 &minus; <i>&tau;</i>)<i>WR</i> + "
        "(<i>b</i> + <i>&tau;A</i>)"
        "&nbsp;&nbsp;&nbsp;&nbsp;(inside the taper band)"))
    out.append(ctx.equation(
        "<i>c</i> = <i>WR</i>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(above the cut-off)"))
    out.append(ctx.p(
        "The middle line is the one worth reading slowly, because both of "
        "its terms push the same way and the usual intuition has only one "
        "of them. Inside the band the household's exposure to the "
        "portfolio is scaled <i>down</i> by (1 &minus; <i>&tau;</i>), and "
        "the guaranteed part of consumption is raised <i>up</i> from "
        "<i>b</i> to <i>b</i> + <i>&tau;A</i>. A means test is usually "
        "described as an implicit tax on wealth, and Australia's is steep "
        "enough to qualify: at "
        f"{taper:.1%} a year it exceeds what domestic equity earns on "
        "average in this panel. But a proportional tax on a risky payoff "
        "is not a deterrent to risk-taking. It is insurance. The household "
        "inside the band keeps less of a good outcome and is held further "
        "off the floor in a bad one, which is exactly the trade a "
        "risk-averse investor pays for."))
    out.append(ctx.p(
        "So the model predicts something the wealth-tax reading does not. "
        "Optimal equity should be <i>non-monotone</i> in wealth: high below "
        "the free area, where the full pension is an unconditional floor; "
        "higher again inside the band, where the taper damps the portfolio "
        "and lifts the floor together; and lowest <i>above</i> the "
        "cut-off, where the pension is gone and the portfolio is the only "
        "thing between the household and nothing. The minimum belongs "
        "where the pension has been withdrawn entirely, not where it is "
        "being withdrawn."))
    out.append(ctx.p(
        f"On this calibration the free area is {free:.2f} times average "
        f"earnings, the full rate {rate:.1%} of them, and the cut-off "
        f"{cut:.2f}. Those three numbers place every household in the "
        f"sections that follow, and Section "
        f"{SHORT_ORDER.index('incidence') + 1} tests the prediction "
        f"directly by moving a household across all three regimes and "
        f"asking what it wants to hold in each."))
    out.append(ctx.h2("#model.1 What the model does not settle"))
    out.append(ctx.p(
        "The three lines are an accounting identity for one period, not a "
        "solved lifecycle problem. They say which way two forces point; "
        "they do not say which is larger, because that depends on the "
        "return distribution, the horizon, the risk aversion and the rule "
        "the household spends by. Nor do they say anything about the "
        "region a real means test spends most of its time in, where a "
        "household near the cut-off in one year is over it in the next and "
        "the test is re-assessed annually against a drawn-down balance. "
        "Both of those are simulation questions, and the rest of the paper "
        "is the simulation."))
    out.append(ctx.p(
        "One implication is worth flagging before the results arrive, "
        "because it is the reason the prediction can fail. The insurance "
        "reading assumes the household <i>consumes</i> what the portfolio "
        "earns. Under a withdrawal rule that spends a fixed real amount, "
        "it does not: a good return accumulates into assessable assets, "
        "the pension is withdrawn against them, and consumption does not "
        "rise at all. Then the taper is a pure loss on the upside and the "
        f"insurance term never arrives. Section "
        f"{SHORT_ORDER.index('incidence') + 1} runs both rules for exactly "
        f"this reason."))
    return out


def incidence(ctx: Any) -> List[Flowable]:
    """Who pays for the guarantee, and what a household near the test wants.

    The referee's other two objections: the Superannuation Guarantee was
    free money, and the household was too rich for the means test to bind.
    They look like one objection and are not, which is the first thing this
    section reports.
    """
    f = ctx.f
    swept = f.table("incidence_sweep")
    optimum = f.table("incidence_optimum")
    profile = f.table("incidence_band_profile")
    free_end, paid_end = optimum.iloc[0], optimum.iloc[-1]
    cost = 100.0 * (float(paid_end["cec_lifetime"])
                    / float(free_end["cec_lifetime"]) - 1.0)
    work = 100.0 * (float(paid_end["mean_working_consumption"])
                    / float(free_end["mean_working_consumption"]) - 1.0)
    cut = float(profile["cutoff"].iloc[0])
    arms = list(dict.fromkeys(profile["arm"])) if "arm" in profile else []
    base_arm = profile[profile["arm"] == arms[0]] if arms else profile
    rule_arm = (profile[profile["arm"] == arms[-1]] if len(arms) > 2
                else base_arm)
    in_band = base_arm[base_arm["position"] == "inside the taper band"]
    above = base_arm[base_arm["position"] == "above the cut-off"]
    below = base_arm[base_arm["position"] == "below the free area"]

    out: List[Flowable] = [
        ctx.h1("#incidence. Who Pays, and Whom the Test Binds")]
    out.append(ctx.p(
        "Two objections stand against everything above, and they look like "
        "one. The first is that the Superannuation Guarantee has been free: "
        "its statutory incidence is on the employer, so the Australian arm "
        "has been handed a tenth of income the American arm never got, and "
        "most of the empirical literature puts the economic incidence on "
        "wages instead. The second is that the household simulated here "
        f"retires on {float(free_end['median_wealth']):.0f} times average "
        f"earnings against a cut-off of {cut:.1f}, so the means test at the "
        "centre of the paper never binds it."))
    out.append(ctx.p(
        "The tempting move is to answer both at once: charge the "
        "guarantee, watch the balance fall, and let the household walk "
        "down onto the test. It does not work, and the reason is the "
        "cleanest thing in this section. Under full economic incidence the "
        "worker funds the contribution out of wages, so take-home pay "
        "falls; but the contribution is still made, so the fund receives "
        "the same money and compounds it identically. <b>Incidence changes "
        "what the household gave up, not what it arrives with.</b> The "
        "balance at the pension age is the same to the last cent at either "
        "end of the dial, and the two objections need two answers."))
    out.append(ctx.h2("#incidence.1 Charging the guarantee"))
    out.append(ctx.p(
        f"The first dial charges a share of the employer contribution "
        f"against wages, from nothing to all of it, crossed with every "
        f"allocation: {len(swept):,} combinations on common random "
        f"numbers. Charging it in full costs {abs(cost):.1f}% of lifetime "
        f"certainty-equivalent consumption and lowers mean consumption "
        f"during the working years by {abs(work):.1f}%, which is the "
        f"transfer itself. That is the size of the free lunch, and the "
        f"honest reading of every cross-system comparison above is that it "
        f"is bracketed by the two ends of this dial rather than pinned at "
        f"either."))
    out.append(ctx.p(
        f"Measured the way the rest of this paper measures — a certainty "
        f"equivalent over the retirement window — the answer is exactly "
        f"zero, {float(free_end['cec']):.4f} at both ends to every digit. "
        f"The retiree's problem is untouched by who paid for the balance: "
        f"same wealth, same pension, same returns. And the allocation does "
        f"not move either: {float(free_end['equity']):.0%} equity is "
        f"optimal whether the guarantee is free or funded out of forty "
        f"years of wages. What the free guarantee inflated was the level "
        f"of Australian consumption, not the portfolio chosen to deliver "
        f"it."))
    out.append(ctx.h2("#incidence.2 A household the test does bind"))
    out.append(ctx.p(
        "The second dial moves the balance directly. It scales what "
        "reaches the pension age, applied at the retirement boundary and "
        "nowhere else, so the career is identical across the grid and a "
        "change in the retiree's allocation is a response to the balance "
        "rather than to what was given up for it. It is a counterfactual "
        "and worth being blunt about: not a claim that Australians save "
        "less than this household, but the observation that a household "
        "near the assets test holds a fraction of what this one holds, and "
        "that the retiree's problem at that balance is the one the model "
        "in Section "
        f"{SHORT_ORDER.index('model') + 1} speaks to. Nothing near the "
        f"test is reachable otherwise."))
    rows = [["Median balance at retirement", "Position",
             "Wanted equity, fixed real rule",
             "Wanted equity, share of balance"]]
    for _, row in base_arm.iterrows():
        match = rule_arm[rule_arm["scale"] == row["scale"]]
        rows.append([
            f"{float(row['median_wealth']):.2f}&times;",
            str(row["position"]),
            f"{float(row['equity']):.0%}",
            f"{float(match['equity'].iloc[0]):.0%}" if len(match) else "—"])
    out += ctx.table(
        rows,
        "The equity share a retiree wants, by where the balance leaves "
        "them against the assets test, under two withdrawal rules. "
        "Balances are multiples of economy-wide average earnings; the free "
        f"area is {float(profile['free_area'].iloc[0]):.2f} and the "
        f"cut-off {cut:.2f}. The pension begins the day work stops in both "
        "columns, so every retirement year is under the test.",
        note="Each row is the best of twenty-one equity shares at that "
             "balance, scored on certainty-equivalent retirement "
             "consumption over the same simulated lifetimes.")
    if len(in_band) and len(above) and len(below):
        out.append(ctx.p(
            f"<b>Under a fixed real withdrawal the prediction fails, and "
            f"it fails in the direction the wealth-tax reading would have "
            f"picked.</b> Wanted equity is "
            f"{float(below['equity'].max()):.0%} below the free area, "
            f"{float(in_band['equity'].max()):.0%} inside the band and "
            f"{float(above['equity'].max()):.0%} above the cut-off — "
            f"monotone in wealth, with the minimum where the model said the "
            f"maximum should be. A household the means test actually binds "
            f"wants no equity at all."))
    out.append(ctx.p(
        "The reason is the withdrawal rule, and it is worth stating "
        "precisely because it is the most useful thing in this section. A "
        "fixed real rule spends a set amount and never spends what the "
        "portfolio earns. For a household near the assets test that is "
        "close to the worst available arrangement: a good equity outcome "
        "converts into assessable assets, the pension is withdrawn against "
        "them at "
        f"{float(ctx.f.cfg['lifecycle'].get('pension_taper', 0.078)):.1%} a "
        "year, and consumption does not rise by a cent. The upside is "
        "confiscated and the downside is not. Equity is then dominated, "
        "and no amount of insurance in the shape of the budget line can "
        "rescue it."))
    if len(rule_arm) and "arm" in profile and len(arms) > 2:
        out.append(ctx.p(
            f"<b>Change the rule and the answer changes completely.</b> "
            f"Spending a share of the <i>current</i> balance instead — so "
            f"the gain is consumed as it arrives rather than assessed — "
            f"the same household wants "
            f"{float(rule_arm['equity'].max()):.0%} equity at every "
            f"position, below the free area, inside the band and past the "
            f"cut-off alike. The swing between the two rules averages "
            f"{100 * abs(float(base_arm['equity'].mean()) - float(rule_arm['equity'].mean())):.0f} "
            f"percentage points, which is the whole allocation. So the "
            f"portfolio a means-tested retiree should hold is not a fact "
            f"about the means test. It is a fact about the means test and "
            f"the drawdown rule together — and the rule is the half the "
            f"retiree can choose."))
    out += ctx.figure(
        "fig66_incidence",
        "Two dials on the Australian arm. Charging the Superannuation "
        "Guarantee to wages (top row) moves lifetime consumption and "
        "leaves the balance untouched; moving the balance (bottom row) is "
        "the only way to put a household on the assets test. The lower "
        "right panel is the model's prediction, tested.")
    out.append(ctx.p(
        "This is the same lesson the withdrawal-rule section reaches from "
        "the other side, and the two should be read together. There it was "
        "a rule that cannot deplete restoring the floor an asset test "
        "removed; here it is a rule that cannot spend destroying the case "
        "for equity that a floor would otherwise support. Both say the "
        "drawdown default and the portfolio default are one decision, and "
        "that a plan sponsor choosing either without the other is choosing "
        "in the dark."))
    return out


def limitations(ctx: Any) -> List[Flowable]:
    """This paper's limitations, not the companion study's.

    The inherited section catalogues the threats to twenty-eight results.
    Most of them are irrelevant here and the pointers to sections this
    paper does not contain are worse than useless, so it is rewritten
    around the four things that would actually change these conclusions.
    """
    f = ctx.f
    # The same leave-one-out runs the companion study reports, read here
    # for the one number this section needs: how far the headline moves
    # when a single country is removed.
    infl = f.table("panel_influence")
    long_number = LONG_NUMBER_ALL

    out: List[Flowable] = [ctx.h1("#limitations. What Would Change These "
                                  "Conclusions")]
    out.append(ctx.p(
        "The results above are the output of a calibrated simulation, and "
        "the honest question about any such thing is what would have to be "
        "different for it to come out the other way. Four things would."))
    out.append(ctx.h2("#limitations.1 The cross-section is sixteen "
                      "countries"))
    line = (
        "The panel is sixteen developed markets, and they are not sixteen "
        "independent draws: equity returns co-move, and the twentieth "
        "century happened to all of them at once. Dropping one country at "
        "a time moves the headline gap enough that the <i>direction</i> of "
        "every result here is far better established than its size.")
    if "gap_pct" in infl:
        line += (
            f" Across the {len(infl)} deletions the baseline gap runs "
            f"{float(infl['gap_pct'].min()):+.2f}% to "
            f"{float(infl['gap_pct'].max()):+.2f}%, and "
            f"{'stays positive throughout' if bool((infl['gap_pct'] > 0).all()) else 'changes sign for at least one'}.")
    line += (
        " Nothing in this paper should be read as a point estimate, and "
        "the reversal claims are claims about signs.")
    out.append(ctx.p(line))
    out.append(ctx.h2("#limitations.2 One pension system, stylised"))
    out.append(ctx.p(
        "Australia's Age Pension is modelled as an assets test alone: a "
        "flat maximum rate, a free area, and a taper, re-assessed every "
        "retirement year against the drawn-down balance. The real system "
        "also applies an income test and a deeming rule, and pays the "
        "lower of the two results; it treats the family home as exempt, "
        "which moves a large share of Australian household wealth outside "
        "the test entirely; and it pays couples at a different rate on "
        "different thresholds. Each of those would change where a given "
        "household sits against the cut-off. None of them changes the "
        "shape of the budget line in Section "
        f"{SHORT_ORDER.index('model') + 1}, which is what the argument "
        f"turns on — but a reader wanting the number for a particular "
        "Australian household will not find it here."))
    out.append(ctx.h2("#limitations.3 The household is one household"))
    out.append(ctx.p(
        f"A single earner, a deterministic wage profile with a permanent "
        f"shock, no unemployment, no career break, no divorce, no "
        f"bequest motive beyond a fixed weight, and a home that is not "
        f"modelled. Section {SHORT_ORDER.index('incidence') + 1} moves the "
        f"balance across the whole width of the assets test, which "
        f"addresses the objection that the results were demonstrated on a "
        f"household the test could not bind; it does not turn one "
        f"household into a distribution of them. The households that "
        f"actually cluster near the cut-off differ from this one in more "
        f"than their balance, and whether those differences reinforce or "
        f"offset the effect measured here is not something this design can "
        f"say."))
    out.append(ctx.h2("#limitations.4 No annuity, and no behaviour"))
    out.append(ctx.p(
        "The instrument that most directly supplies the floor this paper "
        "is about — a life annuity — is not in the choice set, and if it "
        "were it would dominate part of the grid. Nor is there any "
        "behavioural constraint: the household rebalances annually without "
        "hesitation, never panics, and follows its withdrawal rule for "
        "thirty years. The rule comparison in Section "
        f"{SHORT_ORDER.index('longevity') + 1} should be read as what the "
        f"rules deliver if followed, which is an upper bound on what they "
        f"deliver."))
    out.append(ctx.h2("#limitations.5 What is settled, and what is not"))
    out.append(ctx.p(
        "Two things in this paper are robust to all of the above, because "
        "they are identities rather than estimates. Charging the "
        "Superannuation Guarantee to wages cannot change the balance at "
        "retirement, so it cannot change the retiree's problem — that is "
        "arithmetic, and the simulation reproduces it to every digit. And "
        "a withdrawal rule that spends a share of the balance converts a "
        "portfolio gain into consumption, where a fixed real rule converts "
        "it into assessable assets; the sign of that difference does not "
        "depend on the panel."))
    out.append(ctx.p(
        f"Everything else is an estimate. The fuller robustness apparatus "
        f"— the bootstrap-design sweep, the out-of-sample split, the "
        f"valuation conditioning, the fee and turnover ladders, the "
        f"mortality calibration and the parameter sensitivities — is in "
        f"the companion study, whose Sections "
        f"{long_number.get('panel', 19)} and "
        f"{long_number.get('limitations', 37)} carry it. This paper "
        f"inherits those results; it does not re-derive them."))
    return out


def conclusion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    gaps = f.table("pension_gap").set_index("system")
    au, base = gaps.loc["australia_as_legislated"], gaps.loc["us_social_security"]
    untested = gaps.loc["age_pension_untested"]
    band = f.table("incidence_band_profile")
    arms = list(dict.fromkeys(band["arm"])) if "arm" in band else []
    rule_arm = band[band["arm"] == arms[-1]] if len(arms) > 2 else band
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
        f"The third finding is the one we did not expect and would now put "
        f"first for a practitioner. Scaling the balance until the assets "
        f"test genuinely binds — which no household in the sections above "
        f"does, and which is the objection this design was built to answer "
        f"— the portfolio the retiree wants stops being a fact about the "
        f"pension at all. Spending a fixed real amount, they want no "
        f"equity anywhere the test reaches. Spending a share of the "
        f"current balance, they want "
        f"{float(rule_arm['equity'].median()):.0%} everywhere. The "
        f"mechanism is not subtle: a rule that never spends what the "
        f"portfolio earns turns a good return into assessable assets, and "
        f"the pension is withdrawn against them while consumption stands "
        f"still. Near a means test, the withdrawal rule does not modify "
        f"the portfolio decision. It is the portfolio decision."))
    out.append(ctx.p(
        "Three implications follow for default design. A plan sponsor "
        "choosing a glide path is implicitly choosing it against a pension "
        "schedule, and the same fund is not the right default in a "
        "means-tested system and an earnings-related one. The drawdown "
        "default matters as much as the portfolio default: in a system "
        "without a floor, supplying one through the withdrawal rule "
        "recovers what the pension no longer provides. And for the "
        "households a means test actually binds — which are not the "
        "wealthy ones the taper is usually discussed around — the two "
        "defaults cannot be set separately at all, because the rule "
        "decides whether an extra dollar of return reaches consumption or "
        "is taken back by the test."))
    out.append(ctx.p(
        f"We state these as implications of a calibrated model. It carries "
        f"no behavioural constraints, prices no annuity, models one "
        f"household rather than a distribution of them, and — as Section "
        f"{SHORT_ORDER.index('limitations') + 1} sets out — rests on a "
        f"sixteen-country panel whose leave-one-out range is wide enough "
        f"that the direction of these results is far better established "
        f"than their size. Two of them are exceptions, because they are "
        f"identities rather than estimates: charging a compulsory "
        f"contribution to wages cannot change the balance it produces, and "
        f"a rule that spends a share of the balance converts a portfolio "
        f"gain into consumption where a fixed real rule converts it into "
        f"assessable assets. Neither depends on the panel."))
    return out


#: Why each inherited section is in *this* paper. The long study's opening
#: paragraph explains why the section exists at all, which is a different
#: question and, to a reader who has just been given one argument, the
#: wrong one. Each of these replaces the section's first line.
REOPENING: Dict[str, str] = {
    "data": (
        "Everything below rests on one panel, and the argument of this "
        "paper is a comparison of two objective functions over it rather "
        "than a claim about the returns themselves. So this section is "
        "kept to what a reader needs to judge that comparison: what the "
        "panel is, which markets are in it, and what is known to be wrong "
        "with it. The construction detail, the source audit and the "
        "coverage arithmetic are in the companion study."),
    "methods": (
        "The pipeline that turns the panel into a certainty equivalent is "
        "the replicated study's, with one change that matters here: the "
        "public pension is a parameter rather than a fixture, so the same "
        "simulated lifetimes can be scored under two systems with nothing "
        "else moving between them."),
    "baseline": (
        "Before the pension can be shown to reverse the result, the result "
        "has to be reproduced. This section is the replication: the "
        "all-equity portfolio against the age-declining glide path, under "
        "the earnings-related schedule the original study assumes."),
    "pension": (
        "This is the paper's first finding. Nothing changes but the "
        "pension the household retires onto, and the ordering the previous "
        "section established reverses."),
    "leisure": (
        "The reversal has to be attributed to something, and there are two "
        "candidates: the means test's taper, and the loss of the "
        "unconditional floor beneath it. This section separates them, and "
        "the answer is not the one the wealth-tax reading of a means test "
        "would predict."),
    "longevity": (
        "If the mechanism is a missing floor rather than a tax on wealth, "
        "a household should be able to build its own floor out of the "
        "withdrawal rule and recover what the pension no longer supplies. "
        "This section tests that, and the reversal reverses again."),
    "tax": (
        "One last thing could be doing the work. The two systems differ in "
        "their tax treatment as well as their benefit formula, and if the "
        "reversal were a tax result rather than a pension result it would "
        "show up here."),
}


#: Paragraphs an inherited section carries that do not belong in a journal
#: submission. Matched on a distinctive phrase and dropped whole. The
#: companion study keeps them: a reader of a replication archive wants to
#: know how the code is tested, and a referee reading a forty-page paper
#: does not want to be told the test count.
DROPPED: Dict[str, Tuple[str, ...]] = {
    "methods": ("automated tests",),
}

#: Whole subsections an inherited section carries that this paper does not
#: need. Matched on a phrase in the subsection heading; everything from that
#: heading up to the next heading of the same rank or higher goes with it.
#: The companion study keeps them -- a replication archive is judged partly
#: on its audit trail -- but a reader here needs to know what the panel is,
#: not how each series was reconciled against its source.
TRIMMED: Dict[str, Tuple[str, ...]] = {
    "data": ("Auditing the data",),
}

#: The anchors those trims remove, so a cross-reference into one resolves
#: to the companion study rather than to a heading this paper does not
#: print. Kept alongside :data:`TRIMMED` and checked against it, because the
#: two going out of step is the failure the build cannot see.
TRIMMED_ANCHORS: Tuple[str, ...] = ("data.6",)


def _reopen(parts: List[Flowable], ctx: Any, key: str) -> List[Flowable]:
    """Retitle an inherited section and open it on this paper's argument.

    The section bodies are shared with the companion study so the two can
    never disagree about a number. Their headings and first paragraphs are
    not: a section that opens by explaining why it exists in a
    twenty-eight-section study is answering a question this reader has not
    asked.
    """
    out = list(parts)
    for phrase in TRIMMED.get(key, ()):
        out = _without_subsection(out, phrase, key)
    for phrase in DROPPED.get(key, ()):
        kept = [fl for fl in out if phrase not in _text_of(fl)]
        if len(kept) == len(out):
            raise SystemExit(
                f"short paper: nothing in section {key!r} contains "
                f"{phrase!r}, so the paragraph it was meant to drop has "
                f"been reworded and the filter is silently doing nothing")
        out = kept
    if key in RETITLED and out:
        out[0] = ctx.h1(f"#{key}. {RETITLED[key]}")
    if key in REOPENING:
        head = 1 if out else 0
        out.insert(head, ctx.p(REOPENING[key]))
    return out


def _text_of(flowable: Any) -> str:
    """The rendered text of a flowable, or empty for a non-text one."""
    return getattr(flowable, "text", "") or ""


def _rank_of(flowable: Any) -> int:
    """1 for a section heading, 2 for a subsection, 9 for anything else."""
    style = getattr(getattr(flowable, "style", None), "name", "")
    return {"h1": 1, "h1_plain": 1, "h2": 2}.get(style, 9)


def _without_subsection(parts: List[Flowable], phrase: str,
                        key: str) -> List[Flowable]:
    """Drop a subsection and everything under it, up to the next heading.

    Tables and figures inside the span go with it, which is the point --
    a trimmed subsection that left its two figures stranded under the
    following heading would be worse than not trimming at all.
    """
    start = next((i for i, fl in enumerate(parts)
                  if _rank_of(fl) == 2 and phrase in _text_of(fl)), None)
    if start is None:
        raise SystemExit(
            f"short paper: no subsection of {key!r} is headed {phrase!r}, "
            f"so the trim is silently doing nothing -- the heading has been "
            f"reworded or the subsection removed upstream")
    end = next((j for j in range(start + 1, len(parts))
                if _rank_of(parts[j]) <= 2), len(parts))
    return parts[:start] + parts[end:]


def story(ctx: Any) -> List[Flowable]:
    """The short paper, assembled from the sections that carry its thesis."""
    ct.COMPANION = {"name": "the companion study", "numbers": LONG_NUMBER_ALL}
    ct.ABSENT = TRIMMED_ANCHORS
    with renumbered():
        parts: List[Flowable] = []
        parts += front(ctx)
        parts += introduction(ctx)
        parts += model(ctx)
        for key in ("data", "methods", "baseline", "pension", "leisure"):
            parts += _reopen(getattr(ct, f"section_{key}")(ctx), ctx, key)
        parts += incidence(ctx)
        for key in ("longevity", "tax"):
            parts += _reopen(getattr(ct, f"section_{key}")(ctx), ctx, key)
        parts += limitations(ctx)
        parts += conclusion(ctx)
        parts += references(ctx)
    ct.COMPANION = {}
    ct.ABSENT = ()
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
