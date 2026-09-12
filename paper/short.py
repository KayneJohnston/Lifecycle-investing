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

import numpy as np

from reportlab.platypus import (Flowable, NextPageTemplate, PageBreak,
                                Paragraph)

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
    "ordering",
    "limitations",
    "conclusion",
)

#: Sections this paper writes itself rather than selecting from the long
#: study. Every other entry in :data:`SHORT_ORDER` is an inherited section,
#: reopened by :func:`_reopen` on this paper's argument.
OWN: Tuple[str, ...] = ("introduction", "model", "incidence", "ordering",
                        "limitations", "conclusion")

#: The inherited sections whose long-study title does not say what they are
#: doing in *this* paper, and the title that does.
RETITLED: Dict[str, str] = {
    "pension": "What One Country's Pension Does to the Result",
    "leisure": "Which Feature of the Pension Does the Work",
    "longevity": "The Withdrawal Rule a Retiree Should Actually Use",
}

#: Every section's number in the long study, so a reference this paper does
#: not contain can still point a reader somewhere real.
LONG_NUMBER_ALL: Dict[str, int] = {
    k: ct.section_number(k) for k in ct.SECTION_ORDER}

TITLE = "The Pension and the Drawdown Rule"
SUBTITLE = ("What a Means-Tested Pension Does to the All-Equity "
            "Lifecycle Portfolio, and Under Which Rule")

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
    'Iskhakov, F., Thorp, S., and Bateman, H. (2015). "Optimal Annuity '
    'Purchases for Australian Retirees." <i>Economic Record</i>, 91(293), '
    '139–154.',
    'Bateman, H., Eckert, C., Iskhakov, F., Louviere, J., Satchell, S., and '
    'Thorp, S. (2018). "Individual Capability and Effort in Retirement '
    'Benefit Choice." <i>Journal of Risk and Insurance</i>, 85(2), 483–512.',
    'Sefton, J., van de Ven, J., and Weale, M. (2008). "Means Testing '
    'Retirement Benefits: Fostering Equity or Discouraging Savings?" '
    '<i>Economic Journal</i>, 118(528), 556–590.',
    'Braun, R. A., Kopecky, K. A., and Koreshkova, T. (2017). "Old, Sick, '
    'Alone, and Poor: A Welfare Analysis of Old-Age Social Insurance '
    'Programmes." <i>Review of Economic Studies</i>, 84(2), 580–612.',
    'Bütler, M., Peijnenburg, K., and Staubli, S. (2017). "How Much Do '
    'Means-Tested Benefits Reduce the Demand for Annuities?" <i>Journal of '
    'Pension Economics and Finance</i>, 16(4), 419–449.',
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
    # And this cut carries no appendices, so a reference to one has to be
    # sent to the companion for the same reason a reference to an absent
    # section is. Two shipped pointing at appendices this paper does not
    # have, because a letter does not go through the `#key` resolver.
    had = ct.HAS_APPENDICES
    ct.HAS_APPENDICES = False
    try:
        yield
    finally:
        ct._SECTION_NUMBER = original
        ct.HAS_APPENDICES = had


def _resolvable(ctx: Any, text: str) -> Paragraph:
    return ctx.p(text)


def front(ctx: Any) -> List[Flowable]:
    """Title, abstract and keywords for the short paper."""
    f, s = ctx.f, ctx.s
    # Only what the abstract and the highlights quote. Every read here used
    # to feed a page-long abstract; a table left behind when the prose that
    # needed it went is how the wrong leave-one-out ended up in Section
    # #limitations.1, so the list is kept to what is used.
    bite = f.table("leisure_means_test_bite")
    legis = bite[bite["household"] == "as legislated"].iloc[0]
    # The household the abstract's own comparisons are made on. It is not
    # the legislated one, and a draft quoted the legislated one's share
    # past the cut-off beside the matched one's pair of leads.
    sched = bite[bite["household"] == "pension schedule only"].iloc[0]
    optimum = f.table("incidence_optimum")
    cost = 100.0 * (float(optimum["cec_lifetime"].iloc[-1])
                    / float(optimum["cec_lifetime"].iloc[0]) - 1.0)
    # The *portfolio* comparison -- which of two funds a retiree should
    # hold -- and not the comparison between two countries' consumption
    # levels, which is a different quantity that can move the other way.
    gapped = f.table("ordering_gaps")
    baseline_rule = str(f.cfg["lifecycle"]["retirement"]["rule"])
    au_rows = gapped[gapped["system"] == "australia_as_legislated"]
    us_rows = gapped[gapped["system"] == "us_social_security"]
    au_base = au_rows[au_rows["rule"] == baseline_rule]
    us_base = us_rows[us_rows["rule"] == baseline_rule]
    au_best = au_rows.loc[au_rows["gap_pct"].idxmax()]
    # How many fifteen-country panels reproduce the contested sign, which
    # is not the count the bias diagnostic reports and was once confused
    # with it. Counted rather than typed, so a rerun cannot leave the
    # sentence behind.
    split = _split_found(f, baseline_rule)
    # The rule Section #longevity selects rather than the best cell of
    # the grid. Leading with the maximiser would make that section
    # decorative at the one place its answer should bind, so the
    # recommendation is the headline and the maximum sits beside it.
    rec = _recommended(f)
    rec_gap = (f"{rec['gap_pct']:+.2f}%" if rec.get("measured")
               else f"{float(au_best['gap_pct']):+.2f}%")
    rec_rule = (rule_label(str(rec["rule"])) if rec.get("measured")
                else rule_label(str(au_best["rule"])))
    au_base_gap = float(au_base["gap_pct"].iloc[0]) if len(au_base) \
        else float("nan")
    # The widest rule effect, which is the number the abstract leads with
    # because it is the one the panel resolves.
    widest_pp = float("nan")
    if _has(f, "ordering_differences"):
        diffs = f.table("ordering_differences")
        if len(diffs):
            widest_pp = float(
                diffs.loc[diffs["difference_pp"].abs().idxmax(),
                          "difference_pp"])
    flat = gapped[(gapped["system"] == "age_pension_untested")
                  & (gapped["rule"] == baseline_rule)]
    flat_gap = float(flat["gap_pct"].iloc[0]) if len(flat) else float("nan")
    us_base_gap = float(us_base["gap_pct"].iloc[0]) if len(us_base) \
        else float("nan")

    out: List[Flowable] = [
        Paragraph(TITLE, s["title"]),
        Paragraph(SUBTITLE, s["subtitle"]),
        Paragraph("This version: 11 September 2026", s["subtitle"]),
        Paragraph("Abstract", s["h1_plain"]),
    ]
    matched = _matched(f, baseline_rule)
    mt_iv = (_interval_for(f, matched["systems"]["means_tested/voluntary"],
                           baseline_rule)
             if matched.get("measured") else {"measured": False})
    out.append(ctx.p(
        f"An all-equity lifecycle portfolio is said to dominate the "
        f"age-declining glide path of a target-date fund. We show that the "
        f"prescription is conditional on the retiree's institutions rather "
        f"than on the return process. On a 16-country panel of real "
        f"returns spanning 1890\u20132020, crossing five public pension "
        f"regimes with eight withdrawal rules on common simulated "
        f"lifetimes, the all-equity portfolio leads the target-date fund "
        f"by {us_base_gap:+.2f}% in certainty-equivalent retirement "
        f"consumption under an earnings-related pension and "
        f"{matched['cells']['means_tested/voluntary']:+.2f}% under a "
        f"means-tested one paid to the same household on the same "
        f"contributions \u2014 a reversal whose sign holds in all sixteen "
        f"delete-one-country sub-panels. It needs two further things: a "
        f"fixed real withdrawal, and a household risk-averse enough to pay "
        f"for a floor. The mechanism is the loss of an unconditional floor "
        f"rather than the means test's taper, and either of the two "
        f"institutions a country can change \u2014 a compulsory "
        f"contribution, worth "
        f"{matched['contribution_effect_means_tested']:+.1f} points here, "
        f"or a withdrawal rule that cannot deplete, worth up to "
        f"{abs(widest_pp):.0f} points \u2014 offsets most of it. Near the test, "
        f"the drawdown default and the portfolio default are one decision."
        if matched.get("measured") else
        f"An all-equity lifecycle portfolio is said to dominate the "
        f"age-declining glide path of a target-date fund. We show that the "
        f"prescription is conditional on the retiree's institutions rather "
        f"than on the return process. On a 16-country panel of real "
        f"returns spanning 1890\u20132020, the all-equity portfolio leads "
        f"the target-date fund by {us_base_gap:+.2f}% under an "
        f"earnings-related pension and {au_base_gap:+.2f}% under a "
        f"means-tested one."))
    out.append(ctx.p(
        f"<b>What the paper shows.</b> (i) Replacing an earnings-related "
        f"pension with a means-tested one, holding contributions fixed, "
        f"reverses the portfolio ordering under the withdrawal rule the "
        f"literature assumes, and under no other rule in an eight-rule "
        f"menu. (ii) The mechanism is the "
        f"floor, not the taper: paying the <i>same</i> flat pension to the "
        f"same household without testing it leaves the all-equity portfolio "
        f"ahead by {flat_gap:+.2f}% against "
        f"{matched['cells']['means_tested/voluntary']:+.2f}% with the test "
        f"applied \u2014 the rate, the contributions and the panel all held "
        f"\u2014 and "
        f"{float(sched['share_above_cutoff']):.0%} of that household is "
        f"past the cut-off before the test applies, so the taper reaches "
        f"it only in the left tail. (iii) A withdrawal "
        f"rule that cannot deplete restores the ordering: under the rule "
        f"Section {SHORT_ORDER.index('longevity') + 1} selects, the lead "
        f"under the Australian system as legislated is {rec_gap}, and "
        f"every rate in the family restores it. (iv) "
        f"Sixteen delete-one-country sub-panels sign the reversal at "
        f"matched contributions and sign the rule effect under both "
        f"pensions; what they cannot sign is the net of the two "
        f"institutions Australia combines. (v) Who pays for compulsory "
        f"saving changes lifetime consumption "
        f"by {abs(cost):.1f}% and the retiree's problem not at all. (vi) On "
        f"a household the test does bind, the wanted equity share is a "
        f"property of the withdrawal rule rather than of the pension."))
    out.append(ctx.p(
        "<b>Keywords:</b> lifecycle asset allocation; public pension "
        "design; means testing; target-date funds; certainty-equivalent "
        "consumption; decumulation. <b>JEL:</b> G11, G51, H55, D14."))
    out.append(NextPageTemplate("body"))
    out.append(PageBreak())
    return out


def introduction(ctx: Any) -> List[Flowable]:
    f = ctx.f
    bite = f.table("leisure_means_test_bite")
    legis = bite[bite["household"] == "as legislated"].iloc[0]
    # The matched household -- this paper's own savings rate, Australia's
    # pension. Every comparative claim is made on it, and a draft quoted
    # the legislated household's balance beside the matched household's
    # headline, which is two different workers in one paragraph.
    sched = bite[bite["household"] == "pension schedule only"].iloc[0]
    saving = float(f.cfg["lifecycle"]["savings_rate"])
    _sg = float(f.cfg.get("pension", {}).get("sg_rate", 0.12))
    _sg_tax = float(f.cfg.get("pension", {}).get("sg_contributions_tax",
                                                 0.15))
    total = saving + _sg * (1.0 - _sg_tax)
    band = f.table("incidence_band_profile")
    arms = list(dict.fromkeys(band["arm"])) if "arm" in band else []
    base_arm = band[band["arm"] == arms[0]] if arms else band
    rule_arm = band[band["arm"] == arms[-1]] if len(arms) > 2 else base_arm
    optimum = f.table("incidence_optimum")
    cost = abs(100.0 * (float(optimum["cec_lifetime"].iloc[-1])
                        / float(optimum["cec_lifetime"].iloc[0]) - 1.0))
    taper = float(f.cfg["lifecycle"].get("pension_taper", 0.078))
    gapped = f.table("ordering_gaps")
    au_rows = gapped[gapped["system"] == "australia_as_legislated"]
    baseline_rule = str(f.cfg["lifecycle"]["retirement"]["rule"])
    au_base = au_rows[au_rows["rule"] == baseline_rule]
    us_rows = gapped[gapped["system"] == "us_social_security"]
    us_base = us_rows[us_rows["rule"] == baseline_rule]
    au_best = au_rows.loc[au_rows["gap_pct"].idxmax()]
    # How many fifteen-country panels reproduce the contested sign, which
    # is not the count the bias diagnostic reports and was once confused
    # with it. Counted rather than typed, so a rerun cannot leave the
    # sentence behind.
    split = _split_found(f, baseline_rule)
    # The rule Section #longevity selects rather than the best cell of
    # the grid. Leading with the maximiser would make that section
    # decorative at the one place its answer should bind, so the
    # recommendation is the headline and the maximum sits beside it.
    rec = _recommended(f)
    rec_gap = (f"{rec['gap_pct']:+.2f}%" if rec.get("measured")
               else f"{float(au_best['gap_pct']):+.2f}%")
    rec_rule = (rule_label(str(rec["rule"])) if rec.get("measured")
                else rule_label(str(au_best["rule"])))
    au_base_gap = float(au_base["gap_pct"].iloc[0]) if len(au_base) \
        else float("nan")
    us_base_gap = float(us_base["gap_pct"].iloc[0]) if len(us_base) \
        else float("nan")
    gamma_list, gamma_walk = _gamma_walk(f, baseline_rule)
    flat = gapped[(gapped["system"] == "age_pension_untested")
                  & (gapped["rule"] == baseline_rule)]
    flat_gap = float(flat["gap_pct"].iloc[0]) if len(flat) else float("nan")
    recovers = bool(float(au_best["gap_pct"]) > 0.0)
    iv = f.table("ordering_intervals") \
        if _has(f, "ordering_intervals") else None
    contested = None
    if iv is not None and len(iv):
        _hit = iv[(iv["system"] == "australia_as_legislated")
                    & (iv["rule"] == baseline_rule)]
        contested = _hit.iloc[0] if len(_hit) else None


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
    out.append(ctx.p(
        f"<b>Two households appear below and it is worth separating them "
        f"here rather than leaving a reader to notice.</b> The first "
        f"saves this paper's own {saving:.0%} of income and is the "
        f"household every comparative statement is made on, because "
        f"holding the contribution rate still is what lets a difference "
        f"be attributed to the pension. The second adds Australia's "
        f"compulsory Superannuation Guarantee on top, for a total of "
        f"{total:.1%}, and is the household an Australian actually is. "
        f"They are the same worker with two savings rates, and they arrive "
        f"at retirement in different places: "
        f"{float(sched['median_wealth_multiple']):.1f} times average "
        f"earnings against {float(legis['median_wealth_multiple']):.1f}, "
        f"with {float(sched['share_above_cutoff']):.0%} and "
        f"{float(legis['share_above_cutoff']):.0%} of them respectively "
        f"past the point where the assets test stops paying."))
    out.append(ctx.p(
        f"That distance matters for the mechanism and it is the reason to "
        f"lead with the first. Even the {saving:.0%} saver is mostly past "
        f"the cut-off before the test is applied — the pension "
        f"replaces {float(sched['benefit_replacement']):.1%} of their "
        f"career income against the American schedule's "
        f"{float(sched['us_benefit_replacement']):.0%}, and "
        f"{float(legis['benefit_replacement']):.1%} of the Australian's "
        f"— so neither headline is mainly a comparison of two "
        f"means-test designs. Both are comparisons between a household "
        f"that receives an unconditional pension and the same household "
        f"receiving almost none, which is exactly the intervention the "
        f"mechanism section says should matter and the one the taper "
        f"reading says should not. The households a means test actually "
        f"binds are a third population, and Section "
        f"{SHORT_ORDER.index('incidence') + 1} reaches them by scaling "
        f"the balance rather than by pretending either of these is among "
        f"them."))
    out.append(ctx.h2("#introduction.1 What we find"))
    matched = _matched(f, baseline_rule)
    mt_iv = (_interval_for(f, matched["systems"]["means_tested/voluntary"],
                           baseline_rule)
             if matched.get("measured") else {"measured": False})
    if matched.get("measured") and mt_iv.get("measured"):
        split_mt = _split_found(
            f, baseline_rule, matched["systems"]["means_tested/voluntary"])
        out.append(ctx.p(
            f"<b>The ordering reverses when a pension is means-tested, and "
            f"the cleanest form of that is a pair of runs differing in "
            f"nothing else.</b> Pay this household the Age Pension's own "
            f"flat rate with no test attached and the all-equity portfolio "
            f"leads the target-date fund by {flat_gap:+.2f}%; apply the "
            f"test to that identical benefit and the lead becomes "
            f"{matched['cells']['means_tested/voluntary']:+.2f}%, and the "
            f"de-risking glide path takes first place. The rate, the "
            f"contribution rate, the panel and the simulated lifetimes are "
            f"the same in both; only the test is switched on. Against the "
            f"American schedule instead the same household leads by "
            f"{matched['cells']['earnings_related/voluntary']:+.2f}%, which "
            f"is the anchor to the literature and bundles the level change "
            f"with the test — worth "
            f"{flat_gap - matched['cells']['earnings_related/voluntary']:+.2f} "
            f"points, so nothing in what follows turns on which of the two "
            f"references is used."))
        out.append(ctx.p(
            f"<b>That sign is one the cross-section resolves.</b> All "
            f"{_spelled(int(split_mt['deletions']))} of the fifteen-country "
            f"sub-panels put the target-date fund ahead, and the least "
            f"negative of them is "
            f"{split_mt['largest_flip_value']:+.2f}% — clear of zero "
            f"rather than close to it. The delete-one interval, "
            f"[{mt_iv['ci'][0]:+.2f}, {mt_iv['ci'][1]:+.2f}], agrees, and "
            f"we lead with the count rather than the interval for a reason "
            f"Section {SHORT_ORDER.index('ordering') + 1}.3 gives: this "
            f"cell's deletions are skewed enough that the standard error "
            f"built on them is the weaker of the two statistics."))
        out.append(ctx.p(
            f"<b>Australia's own combination is a different and smaller "
            f"number, because a second institution pushes back.</b> The "
            f"Superannuation Guarantee doubles the contribution rate, and "
            f"under a means-tested pension that is worth "
            f"{matched['contribution_effect_means_tested']:+.1f} points "
            f"— a bigger balance is a partial substitute for the floor "
            f"the test removed. As legislated the two land at "
            f"{au_base_gap:+.2f}%, and that residual is the one cell of "
            f"the table this cross-section cannot sign. We report both, in "
            f"that order: the first is a statement about pension design "
            f"and is what the mechanism sections are about; the second is "
            f"a statement about one country and is the more fragile of the "
            f"two."))
    out.append(ctx.p(
        f"Two qualifications belong with those sentences rather than after "
        f"them. The reversal holds under "
        f"{_spelled(int((au_rows['gap_pct'] < 0).sum()))} of the "
        f"{_spelled(int(len(au_rows)))} withdrawal rules we run; under "
        f"every other "
        f"one the all-equity portfolio leads in Australia too. And the "
        f"legislated cell is the one this cross-section cannot "
        f"resolve: a delete-one-country jackknife over the sixteen markets "
        f"puts a standard error of "
        f"{float(contested['standard_error']):.1f} points on a point "
        f"estimate of {float(contested['gap_pct']):+.2f}%, and the sign "
        f"changes when {_spelled(int(split['sign_flips']))} of the "
        f"{_spelled(int(split['deletions']))} are removed one at a time. "
        f"Every other "
        f"cell in that table has a standard error between 2 and 5 points "
        f"and an interval excluding zero."
        if split.get("measured") else
        f"estimate of {float(contested['gap_pct']):+.2f}%, and the interval "
        f"contains zero. Every other cell in that table has a standard "
        f"error between 2 and 5 points and an interval excluding zero."))
    if gamma_walk:
        out.append(ctx.p(
            f"<b>A third condition is the risk aversion.</b> Re-scoring "
            f"the same simulated lifetimes at risk aversions of "
            f"{gamma_list}, the matched means-tested cell runs "
            f"{_gamma_walk(f, baseline_rule, matched['systems']['means_tested/voluntary'])[1] or gamma_walk}"
            f" and the legislated one runs {gamma_walk}: in both the "
            f"reversal is absent at the lowest of the three and deepens "
            f"through the other two, while every other withdrawal rule "
            f"stays positive at all three. So the failure of the "
            f"all-equity prescription is located precisely — an "
            f"asset-tested pension, a fixed real withdrawal, and a "
            f"household risk-averse enough to pay for the floor — "
            f"and two of those three are chosen by whoever sets the "
            f"defaults."
            if matched.get("measured") else
            f"<b>A third condition is the risk aversion.</b> Re-scoring "
            f"the same simulated lifetimes at risk aversions of "
            f"{gamma_list}, the contested cell runs {gamma_walk}."))
    out.append(ctx.p(
        "<b>The mechanism is the floor, not the taper.</b> A means test is "
        "usually discussed as an implicit tax on wealth, and Australia's is "
        "steep enough to qualify — a dollar of assessable assets costs 7.8% "
        "of pension a year, more than domestic equity earns on average in "
        "this panel. But that reading does not survive checking where the "
        f"household stands. {float(sched['share_above_cutoff']):.0%} of "
        f"the matched households and "
        f"{float(legis['share_above_cutoff']):.0%} of the Australian ones "
        "are past the cut-off before the test is applied, so the taper "
        "reaches either only in the left tail, after a portfolio has "
        "already fallen. What the means test removes is the "
        "unconditional annuity itself. A retiree standing on a guaranteed "
        "floor can carry the equity tail; one standing on their portfolio "
        "alone cannot, and a risk-averse objective prices that difference "
        "heavily."))
    out.append(ctx.p(
        f"<b>A floor need not come from a pension.</b> The reversal is "
        f"conditional on a withdrawal rule that can run out. Under an "
        f"amortisation rule the balance is divided by the years remaining, "
        f"ruin falls to zero, and the household manufactures its own floor "
        f"out of the larger portfolio compulsory saving bought it. Holding "
        f"the pension and the returns fixed and changing only the rule, the "
        f"all-equity lead in the Australian system moves from "
        f"{au_base_gap:+.2f}% to {rec_gap} under the "
        f"{rec_rule} rule Section "
        f"{SHORT_ORDER.index('longevity') + 1} selects \u2014 and to "
        f"{float(au_best['gap_pct']):+.2f}% at the most generous rate in "
        f"the family \u2014 so "
        f"{'the portfolio ordering reverses a second time' if recovers else 'the target-date fund keeps the lead even then'}. "
        f"The all-equity prescription is therefore conditional on two "
        f"institutions rather than one \u2014 and, as the next paragraph "
        f"but one reports, on a preference as well."))
    out.append(ctx.p(
        "It is worth being exact about what that sentence compares, because "
        "two orderings are easy to run together. One is between "
        "<i>portfolios</i>: which of two funds a given retiree should hold. "
        "The other is between <i>countries</i>: which pension system "
        "delivers more retirement consumption. They can move in opposite "
        "directions, and the claim above is about the first. Every "
        "percentage in this paper is a portfolio comparison unless it is "
        "labelled otherwise."))
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
        f"two households above retire on "
        f"{float(sched['median_wealth_multiple']):.1f} and "
        f"{float(legis['median_wealth_multiple']):.1f} times average "
        f"earnings, both far outside the taper band, so we scale the "
        f"arriving balance until it sits inside and ask again. Spending a fixed real amount, it "
        f"wants no equity at any balance the test can reach; spending a "
        f"share of the current balance, it wants at least "
        f"{float(rule_arm['equity'].median()):.0%} equity at every one of "
        f"them, and more than that wherever it can borrow. The reason is "
        f"mechanical and, once seen, obvious: under a "
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
        "of the pension the investor retires onto, and the design says "
        "which feature of the pension it is. Paying the Age Pension's own "
        "flat rate to the same household on the same contributions, but "
        "without testing it, leaves the all-equity portfolio ahead by "
        f"{flat_gap:+.2f}%; applying the test to that identical benefit "
        f"takes it to "
        f"{matched['cells']['means_tested/voluntary']:+.2f}%. Withdrawing "
        f"a pension at the margin and paying a smaller one are different "
        f"interventions, and only the first reorders portfolios. Second, "
        f"the <i>mechanism</i>: "
        f"Section {SHORT_ORDER.index('model') + 1} writes the retiree's "
        f"budget line as three regimes and shows that inside the taper band "
        f"a means test is insurance rather than a wealth tax. That "
        f"prediction is sharp enough to fail and it does: under a fixed "
        f"real withdrawal the retiree wants no equity anywhere the test "
        f"reaches, and under a percentage-of-balance rule \u2014 once the "
        f"grid is widened far enough to have an opinion \u2014 the "
        f"maximum sits below the free area rather than inside the band. "
        f"The taper does insure, and it insures less than the guarantee "
        f"it withdraws. We report the refutation because a mechanism that "
        f"cannot be wrong is not a mechanism. Third, the "
        f"<i>interaction</i>: the "
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
        "The Australian assets test has its own literature, and it reaches "
        "the pension-as-endowment question from the institutional side. "
        "Iskhakov, Thorp and Bateman (2015) solve an Australian retiree's "
        "annuity purchase with the means test in the constraint set, and "
        "find the test reshaping a decision that is about buying a floor "
        "\u2014 which is the same margin this paper reaches through the "
        "withdrawal rule, with the annuity itself left out of the choice "
        "set; Bateman et al. (2018) document what Australian members "
        "actually hold. On the "
        "theory side, Sefton, van de Ven and Weale (2008) and Braun, "
        "Kopecky and Koreshkova (2017) analyse means-tested transfers as "
        "insurance rather than as taxes, and Bütler, Peijnenburg and "
        "Staubli (2017) show a means-tested benefit crowding out voluntary "
        "annuitisation for exactly the reason the floor matters here. What "
        "we add to that work is the comparative statement — the same "
        "household, the same returns, two pension systems — and the "
        "interaction with the drawdown rule, which we have not found "
        "measured."))
    out.append(ctx.p(
        "On the decumulation side the rule that supplies the floor is the "
        "actuarial or amortisation rule of Milevsky and Huang (2011). The "
        "withdrawal-rule literature typically conditions on a fixed income "
        "floor and the pension literature on a fixed withdrawal rule, which "
        "is why the interaction between them has room to be new."))
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
        f"{SHORT_ORDER.index('longevity') + 1} finds the withdrawal rule a "
        f"retiree should actually use, and Section "
        f"{SHORT_ORDER.index('ordering') + 1} asks what that rule does to "
        f"the portfolio ordering and how precisely sixteen countries can "
        f"resolve each answer. Section "
        f"{SHORT_ORDER.index('limitations') + 1} says what would change "
        f"these conclusions. The two systems also differ in their tax "
        f"treatment, and Section {LONG_NUMBER_ALL['tax']} of the companion "
        f"study rules that out as an explanation; we do not repeat it here."))
    return out


def model(ctx: Any) -> List[Flowable]:
    """The retiree's budget line, and what the withdrawal rule does to it.

    The first version of this section derived a kinked budget line, showed
    that the taper is insurance inside the band, and predicted that optimal
    equity would be non-monotone in wealth with its minimum above the
    cut-off. The simulation then refuted it, and a model whose prediction
    fails and which is not used again is a model doing no work.

    The error was in what the model held fixed. It priced the *test* while
    assuming the household consumes what the portfolio earns -- which is a
    property of the withdrawal rule, not of the pension. Written with the
    rule in it, the same two lines predict what the sweep actually finds.
    """
    f = ctx.f
    ss = f.cfg["lifecycle"]
    taper = float(ss.get("pension_taper", 0.078))
    free = float(ss.get("pension_free_area", 3.01))
    rate = float(ss.get("pension_full_rate", 0.293))
    profile = f.table("incidence_band_profile")
    cut = float(profile["cutoff"].iloc[0])

    out: List[Flowable] = [ctx.h1("#model. What an Asset Test Does to a "
                                  "Retiree's Budget Line")]
    out.append(ctx.p(
        "Write <i>W</i> for the assessable balance a household holds at the "
        "start of a retirement year, <i>R</i> for the gross real return it "
        "earns over that year, <i>x</i> for what its withdrawal rule pays "
        "out, <i>A</i> for the assets-test free area, <i>b</i> for the full "
        "pension and <i>&tau;</i> for the rate at which the pension is "
        "withdrawn against assets above <i>A</i>. The test is assessed on "
        "the assets that remain once the year's withdrawal has been taken, "
        "which is the convention the Australian scheme uses and the one the "
        "simulation implements. Consumption in the year is the withdrawal "
        "plus whatever pension the test leaves:"))
    out.append(ctx.equation(
        "<i>c</i> = <i>x</i> + min{<i>b</i>, "
        "max[0, <i>b</i> &minus; <i>&tau;</i>(<i>WR</i> &minus; <i>x</i> "
        "&minus; <i>A</i>)]}"))
    out.append(ctx.p(
        "The three regimes are the three branches of that expression, and "
        "writing them out is what makes the mechanism visible. Let "
        "<i>S</i> = <i>WR</i> &minus; <i>x</i> be the assets the test sees. "
        "Below the free area (<i>S</i> &le; <i>A</i>) the pension is paid in "
        "full; inside the band (<i>A</i> &lt; <i>S</i> &lt; <i>A</i> + "
        "<i>b</i>/<i>&tau;</i>) it is withdrawn at the taper; above the "
        "cut-off it is gone:"))
    out.append(ctx.equation(
        "<i>c</i> = <i>x</i> + <i>b</i>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(below the free area)"))
    out.append(ctx.equation(
        "<i>c</i> = <i>x</i>(1 + <i>&tau;</i>) &minus; <i>&tau;WR</i> + "
        "(<i>b</i> + <i>&tau;A</i>)"
        "&nbsp;&nbsp;&nbsp;&nbsp;(inside the taper band)"))
    out.append(ctx.equation(
        "<i>c</i> = <i>x</i>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(above the cut-off)"))
    out.append(ctx.h2("#model.1 Why the withdrawal rule decides the sign"))
    out.append(ctx.p(
        f"A means test is usually described as an implicit tax on wealth, "
        f"and Australia's is steep enough to qualify: at {taper:.1%} a year "
        f"it exceeds what domestic equity earns on average in this panel. "
        f"The middle line says that description is incomplete, and what it "
        f"puts in its place depends entirely on how <i>x</i> responds to "
        f"<i>R</i>. Differentiate it with respect to the return:"))
    out.append(ctx.equation(
        "&part;<i>c</i>/&part;<i>R</i> = "
        "(1 + <i>&tau;</i>)&nbsp;&part;<i>x</i>/&part;<i>R</i> &minus; "
        "<i>&tau;W</i>"))
    out.append(ctx.p(
        f"Take the rule that spends a fixed real amount first, because it "
        f"is the one the literature assumes and the one this paper spends "
        f"by. Here <i>x</i> is a constant, so &part;<i>x</i>/&part;<i>R</i> "
        f"= 0 and the derivative is &minus;<i>&tau;W</i>, negative at every "
        f"withdrawal rate. A good return raises assessable assets and "
        f"leaves the withdrawal exactly where it was — this year and every "
        f"year after it, because the rule does not read the balance at all. "
        f"The gain is taxed at {taper:.1%} a year for as long as it is held "
        f"and reaches consumption in no year at all; the only place it can "
        f"surface is the estate. The upside is confiscated and the downside "
        f"is not."))
    out.append(ctx.p(
        f"Now the rule that spends a fixed share <i>k</i> of the current "
        f"balance. Here <i>x</i> = <i>kWR</i>, so "
        f"&part;<i>x</i>/&part;<i>R</i> = <i>kW</i> and the derivative is "
        f"<i>W</i>[<i>k</i>(1 + <i>&tau;</i>) &minus; <i>&tau;</i>]. This "
        f"is positive only above a threshold withdrawal rate, "
        f"<i>k</i> &gt; <i>&tau;</i>/(1 + <i>&tau;</i>), which on the "
        f"Australian taper is {taper / (1 + taper):.2%}. It is worth being "
        f"blunt that the proportional rule this paper actually simulates "
        f"draws 4% and therefore sits <i>below</i> that threshold: in the "
        f"year a good return arrives, this household too hands the test "
        f"more in withdrawn pension than it takes in extra spending."))
    out.append(ctx.p(
        "The two rules nonetheless differ in the way that matters, and the "
        "threshold is what makes the difference visible rather than what "
        "creates it. Under the proportional rule the gain is deferred: the "
        "larger balance is still there, and every later withdrawal is a "
        "share of it, so the return reaches consumption eventually whatever "
        "<i>k</i> is. Under the fixed real rule the deferral never ends, "
        "because no future withdrawal reads the balance either. A "
        "proportional rule spends the gain late; a fixed real rule never "
        "spends it. That is the distinction which survives from one period "
        "to a retirement, and the threshold only says how late."))
    out.append(ctx.h3("#model.1.1 What that implies for the portfolio"))
    out.append(ctx.p(
        "The three regimes then order the retiree's problem, and the "
        "comparison is two lines. Exposure to the return is "
        "&part;<i>x</i>/&part;<i>R</i> below the free area, "
        "(1 + <i>&tau;</i>)&part;<i>x</i>/&part;<i>R</i> &minus; "
        "<i>&tau;W</i> inside the band, and &part;<i>x</i>/&part;<i>R</i> "
        "again above the cut-off. The part of consumption that does not "
        "move with the return goes <i>b</i>, then <i>b</i> + "
        "<i>&tau;A</i>, then nothing. So inside the band the household "
        "faces strictly less exposure than below the free area and carries "
        "a strictly larger riskless component; above the cut-off it faces "
        "the same exposure as below the free area and carries none at all."))
    out.append(ctx.p(
        "Two words are worth avoiding here, and the reason is a real "
        "objection rather than fastidiousness. It is tempting to call "
        "<i>b</i> + <i>&tau;A</i> a <i>floor</i>, and it is not one: "
        "consumption inside the band is <i>decreasing</i> in the return, so "
        "that quantity is the intercept of a downward-sloping line and is "
        "attained at the top of the band rather than guaranteed across it. "
        "What the algebra delivers is the riskless component of "
        "consumption, which is the quantity the endowment result is about "
        "and which does rise. Elsewhere in this paper “floor” "
        "means the unconditional pension — a payment that arrives "
        "whatever the portfolio does — and that is a floor in the "
        "ordinary sense."))
    out.append(ctx.p(
        "For an investor with constant relative risk aversion, a larger "
        "riskless component of consumption raises the optimal share of the "
        "risky asset — the standard result for a bond-like endowment, and "
        "the one the lifecycle literature invokes to explain why a pension "
        "crowds fixed income out of the financial portfolio. Applying it to "
        "the ordering above gives the prediction directly: <b>optimal "
        "equity should be highest inside the taper band, lower below the "
        "free area, and lowest above the cut-off</b> — non-monotone in "
        "wealth, with its minimum where the pension has been withdrawn "
        "entirely rather than where it is being withdrawn."))
    out.append(ctx.p(
        "That is the prediction under a rule that spends the balance. Under "
        "a fixed real rule the insurance term never arrives, because the "
        "household never converts the return into anything it eats, and the "
        "prediction is a corner rather than a shape: no equity anywhere the "
        "taper operates, and position against the test should barely "
        "matter, because what does the damage is the rule and not the "
        "band."))
    out.append(ctx.p(
        f"On this calibration the free area is {free:.2f} times average "
        f"earnings, the full rate {rate:.1%} of them, and the cut-off "
        f"{cut:.2f}. Section {SHORT_ORDER.index('incidence') + 1} runs "
        f"both rules across all three regimes and reports what happens to "
        f"each prediction. Neither survives intact, and saying so is the "
        f"point of writing one down."))
    out.append(ctx.h2("#model.2 What the model does not settle"))
    out.append(ctx.p(
        f"This is one period of accounting, not a solved lifecycle "
        f"problem. It says which way the forces point; it does not say "
        f"how large the non-monotonicity is, or whether it is there at "
        f"all, because that depends on the return distribution, the "
        f"horizon and the risk aversion. Section "
        f"{SHORT_ORDER.index('incidence') + 1}.5 finds it is not there: "
        f"under either withdrawal rule wanted equity turns out to be "
        f"monotone in wealth, and the hump this section predicts does not "
        f"appear once the grid is wide enough to show one. Nor does "
        "one period describe the region a real means test spends most of "
        "its time in, where a household inside the band this year is over "
        "the cut-off next year and the test is re-assessed annually against "
        "a drawn-down balance. Both are simulation questions, and the "
        "simulation is the rest of the paper."))
    out.append(ctx.p(
        "The dependence on risk aversion turns out to be the sharper of "
        "those two omissions, and worth flagging here rather than letting "
        "it arrive as a surprise. One period of accounting says the "
        "curvature of the objective scales the effect; Section "
        "#ordering.5 finds that over the range of risk aversions "
        "commonly used it changes the effect's <i>sign</i>. Nothing in "
        "the algebra above rules that out — the floor and the exposure "
        "move in opposite directions across the band, and which "
        "dominates is exactly the question a felicity function answers — "
        "but the algebra does not predict it either, and we did not "
        "expect it."))
    out.append(ctx.p(
        "One caution belongs with the second prediction rather than with "
        "the results. It is a corner at zero, and a "
        "constant-relative-risk-aversion objective with a consumption floor "
        "near zero is unbounded below — a handful of near-starvation years "
        "can move a certainty equivalent further than a decade of ordinary "
        "ones. A prediction of \u201cno equity\u201d is exactly the kind one "
        "should distrust, so Section "
        f"{SHORT_ORDER.index('incidence') + 1} re-scores it at several risk "
        f"aversions and several consumption floors before reporting it."))
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
        f"retires far past the assets test -- on "
        f"{float(free_end['median_wealth']):.1f} times average earnings at "
        f"the all-equity portfolio this sweep holds, against a cut-off of "
        f"{cut:.1f} -- so the means test at the centre of the paper never "
        f"binds it."))
    # m5 of the referee report. Three balances are quoted in three places
    # and the old parenthetical called the difference "a little", which is
    # not what 13.8 against 36.4 is.
    out.append(ctx.note(
        f"Three balances at retirement appear in this paper and they are "
        f"not the same household measured three ways. Section "
        f"{SHORT_ORDER.index('leisure') + 1}.1's is the fixed-contribution "
        f"household of that section's 2×2, which saves only this "
        f"paper's own 10% because holding contributions still is what "
        f"separates the pension's timing from its formula. Section "
        f"{SHORT_ORDER.index('incidence') + 1}'s carries the Superannuation "
        f"Guarantee and holds the all-equity portfolio this sweep is run "
        f"on. Section {SHORT_ORDER.index('introduction') + 1}'s is the same "
        f"household at the portfolio the headline compares. Each is stated "
        f"where it is used; what they share, and all the objection above "
        f"needs, is that every one of them is several times the cut-off."))
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
        anchor="band_profile",
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
            f"the same household wants at least "
            f"{float(rule_arm['equity'].max()):.0%} equity at every "
            f"position, below the free area, inside the band and past the "
            f"cut-off alike \u2014 at least, because that is the top of "
            f"this grid and Section "
            f"{SHORT_ORDER.index('incidence') + 1}.5 shows the household "
            f"would take more if it were offered. The swing between the "
            f"two rules averages "
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
    if _has(f, "incidence_influence"):
        infl = f.table("incidence_influence")
        if len(infl):
            out.append(ctx.h2("#incidence.3 What the panel costs this "
                              "corner"))
            cols = [c for c in ("equity_where_the_test_binds",
                                "equity_overall") if c in infl]
            steady = all(float(infl[c].max()) == float(infl[c].min())
                         for c in cols)
            out.append(ctx.p(
                f"The two results above are checked against the "
                f"preference specification and not against the panel, "
                f"which is the objection Section "
                f"{SHORT_ORDER.index('ordering') + 1} answers for its own "
                f"numbers. So the balance sweep is recomputed "
                f"{len(infl)} times, once with each country's history "
                f"removed, and the equity share the retiree wants is read "
                f"off each one."))
            if steady:
                held = float(infl[cols[0]].iloc[0])
                out.append(ctx.p(
                    f"<b>The answer is {held:.0%} equity in every one of "
                    f"the {_spelled(len(infl))} deletions.</b> Not close to "
                    f"{held:.0%} — identical, to every digit the sweep "
                    f"resolves. A jackknife standard error would be "
                    f"exactly zero and the interval a point, which is "
                    f"why we report the deletions rather than an interval "
                    f"computed from them. That is what a corner solution "
                    f"looks like when it is checked: the objective is "
                    f"not nearly best at the boundary, it is pinned "
                    f"there, and no fifteen-country subsample of this "
                    f"panel moves it. It is also the reason to state the "
                    f"finding as a direction rather than a number. The "
                    f"panel resolves <i>which corner</i> with no "
                    f"uncertainty at all and says nothing about how far "
                    f"inside the boundary a real retiree, facing a real "
                    f"assets test rather than this stylisation, would "
                    f"land."))
            else:
                spans = ", ".join(
                    f"{c.replace('_', ' ')} runs "
                    f"{float(infl[c].min()):.0%} to {float(infl[c].max()):.0%}"
                    for c in cols)
                out.append(ctx.p(
                    f"<b>The corner does move.</b> Across the "
                    f"{len(infl)} deletions, {spans}. The result is "
                    f"therefore a statement about this panel and not only "
                    f"about the means test, and we report the span rather "
                    f"than the central value alone."))
    # The other half of the promise made in the model section: a corner at
    # zero under an objective that is unbounded below is exactly the kind
    # of answer to distrust, so it is re-scored against the curvature and
    # against a floor before it is reported.
    out += _corner_robustness(ctx, f)
    if _has(f, "ceiling_optimum"):
        lev_opt = f.table("ceiling_optimum")
        cheap = float(lev_opt["spread"].min())
        free = lev_opt[np.isclose(lev_opt["spread"], cheap)] \
            .sort_values("scale")
        bands = f.table("ceiling_by_band") \
            if _has(f, "ceiling_by_band") else None
        if len(free):
            out.append(ctx.h2("#incidence.5 Whether the other corner is a "
                              "corner at all"))
            out.append(ctx.p(
                f"The zero is not at a grid edge in any sense that "
                f"matters \u2014 there is nothing below no equity to want, "
                f"and the deletions above resolve it exactly. The 100% "
                f"is different, and the difference is easy to miss. It "
                f"sits at the top of the "
                f"grid it was chosen from, so it is consistent with a "
                f"household that wants exactly the whole portfolio in "
                f"equity and with one that wants half as much again and "
                f"was never asked. That is not a quibble about a "
                f"truncated number: the claim Section "
                f"{SHORT_ORDER.index('model') + 1}.1.1 makes is about a "
                f"<i>shape</i> across the assets test, and three regions "
                f"pinned against a shared ceiling cannot be ranked "
                f"against one another at all."))
            out.append(ctx.p(
                f"So the ceiling comes off. The retiree may borrow against "
                f"the equity sleeve at the realised bill return plus a "
                f"spread, in retirement only, so the balance they arrive "
                f"with is the balance this section scaled rather than one "
                f"borrowing helped build. The sleeve stays all-equity; the "
                f"dial is how much of it they hold. The household is the "
                f"one in the right-hand column of "
                f"@table:band_profile and not a household built to "
                f"resemble it \u2014 the same specification, the "
                f"same pension starting the day work stops, the same "
                f"fourteen balances, differing from the left-hand column "
                f"in the withdrawal rule and in nothing else. That is what "
                f"licenses reading the two columns against each other "
                f"below."))
            rows = [["Position against the test", "Balances",
                     "Wanted holding, median", "Lowest", "Highest"]]
            if bands is not None and len(bands):
                order_of = {"below the free area": 0,
                            "inside the taper band": 1,
                            "above the cut-off": 2}
                for _, row in bands.sort_values(
                        "position",
                        key=lambda c: c.map(
                            lambda v: order_of.get(str(v), 9))).iterrows():
                    rows.append([
                        str(row["position"]), f"{int(row['balances'])}",
                        f"{float(row['median_leverage']):.2f}\u00d7",
                        f"{float(row['low']):.2f}\u00d7",
                        f"{float(row['high']):.2f}\u00d7"])
                out += ctx.table(
                    rows,
                    "How much of the portfolio a means-tested retiree wants "
                    "in equity once the grid runs past the whole of it, by "
                    "position against the assets test, at a borrowing "
                    "spread of zero.",
                    note="A holding above 1.00\u00d7 is borrowing against "
                         "the sleeve. Zero is the cheapest borrowing there "
                         "is and therefore the hardest test for the "
                         "unlevered corner; the paragraph after next prices "
                         "it properly.")
            lo = float(free["leverage"].min())
            hi = float(free["leverage"].max())
            out.append(ctx.p(
                f"<b>The corner was the grid's.</b> At a borrowing spread "
                f"of zero the retiree wants between {lo:.2f}\u00d7 and "
                f"{hi:.2f}\u00d7 the portfolio, and never exactly "
                f"1.00\u00d7 at any of the {len(free)} balances. The "
                f"100% reported above is a floor on what this household "
                f"wants, not the amount they want. Everything the previous "
                f"subsection says about the <i>difference</i> between the "
                f"two withdrawal rules survives that \u2014 it is a "
                f"difference between a corner at zero and a corner at or "
                f"above the whole portfolio, which is if anything wider "
                f"than reported \u2014 but the level should be read as a "
                f"bound."))
            if bands is not None and len(bands):
                wanted = {str(r["position"]): float(r["median_leverage"])
                          for _, r in bands.iterrows()}
                band_v = wanted.get("inside the taper band")
                below_v = wanted.get("below the free area")
                above_v = wanted.get("above the cut-off")
                if None not in (band_v, below_v, above_v):
                    holds = band_v > below_v and below_v > above_v
                    out.append(ctx.p(
                        f"<b>And the shape is not the predicted one.</b> "
                        f"With room above the whole portfolio the three "
                        f"regions do separate \u2014 "
                        f"{below_v:.2f}\u00d7 below the free area, "
                        f"{band_v:.2f}\u00d7 inside the band, "
                        f"{above_v:.2f}\u00d7 past the cut-off \u2014 so "
                        f"the ceiling was hiding something. But the "
                        f"ordering runs the wrong way for Section "
                        f"{SHORT_ORDER.index('model') + 1}.1.1, which puts "
                        f"the maximum inside the band. "
                        f"{'It holds here.' if holds else 'The maximum is below the free area instead, where the pension is paid in full, and wanted equity falls monotonically as the balance rises.'} "
                        f"What survives of the prediction is its lower "
                        f"end: the minimum does sit above the cut-off, "
                        f"where the pension has been withdrawn entirely. "
                        f"What does not survive is the claim that the "
                        f"taper band is where a retiree most wants "
                        f"equity."))
                    out.append(ctx.p(
                        "The reading we take from that is the one the "
                        "one-period model was always weakest on. A larger "
                        "guaranteed component raises the optimal risky "
                        "share; below the free area the guarantee is the "
                        "full pension and inside the band it is less. The "
                        "insurance the taper supplies against a falling "
                        "portfolio is real, and it is smaller than the "
                        "guarantee it is withdrawing. Section "
                        f"{SHORT_ORDER.index('model') + 1} gets the "
                        "direction of the taper's insurance right and its "
                        "size wrong, and only a simulation with room above "
                        "the grid could have said so."))
            ordered = base_arm.sort_values("median_wealth")
            if len(ordered) > 1 and len(free) > 1:
                first = float(ordered["equity"].iloc[0])
                last = float(ordered["equity"].iloc[-1])
                lev_lo = float(free["leverage"].iloc[0])
                lev_hi = float(free["leverage"].iloc[-1])
                if (last - first) * (lev_hi - lev_lo) < 0:
                    out.append(ctx.p(
                        f"<b>The two rules do not merely want different "
                        f"amounts. They want them in opposite "
                        f"directions.</b> Read down the same fourteen "
                        f"balances, wanted equity under a fixed real "
                        f"withdrawal runs {first:.0%} at the poorest and "
                        f"{last:.0%} at the richest, rising with wealth; "
                        f"under a percentage-of-balance rule it runs "
                        f"{lev_lo:.2f}\u00d7 and {lev_hi:.2f}\u00d7, "
                        f"falling. The withdrawal rule does not shift the "
                        f"level of a common profile. It reverses the sign "
                        f"of the wealth gradient itself."))
                    out.append(ctx.p(
                        "That is the sharpest form of this section's "
                        "claim, and it is worth separating from the "
                        "swing in levels. Under a rule that spends the "
                        "balance, a poorer retiree is the one standing on "
                        "the largest guaranteed floor relative to their "
                        "portfolio, so they take the most risk \u2014 the "
                        "textbook endowment result, working normally. "
                        "Under a rule that does not, the same floor is "
                        "unreachable: the return cannot be eaten, so it "
                        "becomes assessable assets and the pension is "
                        "withdrawn against it, and the poorer retiree is "
                        "the one with most to lose from holding equity at "
                        "all. Position against a means test does not have "
                        "a portfolio implication on its own. It has one "
                        "sign under one rule and the opposite sign under "
                        "another, and a plan sponsor who sets a glide "
                        "path without setting a drawdown default has not "
                        "chosen between them."))
            out += _borrowing_price(ctx, f)
            out.append(ctx.p(
                f"Two things keep this from overturning the section rather "
                f"than qualifying it. The borrowing is worth little: "
                f"against the unlevered portfolio on the same paths it "
                f"buys a few per cent of certainty-equivalent "
                f"consumption, so the household is close to indifferent "
                f"across a wide range and the ranking of the two "
                f"withdrawal rules is nowhere near it. And the grid is "
                f"still a grid: {int((free['at_ceiling']).sum())} of "
                f"{len(free)} balances want the top of this one too, so "
                f"what is established is that the optimum lies above the "
                f"whole portfolio and not how far above. A retiree who "
                f"cannot borrow \u2014 which is most of them \u2014 "
                f"holds 100% equity and the section's advice is "
                f"unchanged."))
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


def ordering(ctx: Any) -> List[Flowable]:
    """Which portfolio wins, under which pension, under which rule.

    The section the previous draft needed and did not have. Its headline
    was a comparison of two portfolios; its second finding was a comparison
    of two countries; and it ran the two together in the phrase "the
    ordering reverses again". This is the portfolio comparison, rule by
    rule.
    """
    f = ctx.f
    gapped = f.table("ordering_gaps")
    baseline_rule = str(f.cfg["lifecycle"]["retirement"]["rule"])
    au = gapped[gapped["system"] == "australia_as_legislated"]
    us = gapped[gapped["system"] == "us_social_security"]
    au_base = au[au["rule"] == baseline_rule]
    us_base = us[us["rule"] == baseline_rule]
    au_best = au.loc[au["gap_pct"].idxmax()]
    au_worst = au.loc[au["gap_pct"].idxmin()]
    # The rule Section #longevity selects rather than the best cell of
    # the grid. Leading with the maximiser would make that section
    # decorative at the one place its answer should bind, so the
    # recommendation is the headline and the maximum sits beside it.
    rec = _recommended(f)
    rec_gap = (f"{rec['gap_pct']:+.2f}%" if rec.get("measured")
               else f"{float(au_best['gap_pct']):+.2f}%")
    rec_rule = (rule_label(str(rec["rule"])) if rec.get("measured")
                else rule_label(str(au_best["rule"])))
    recovers = bool(float(au_best["gap_pct"]) > 0.0)
    winners = au[au["gap_pct"] > 0.0]
    swept_all = f.table("ordering_sweep")
    au_eq = swept_all[(swept_all["system"] == "australia_as_legislated")
                      & (swept_all["strategy"] == "balanced_all_equity")]
    au_eq = au_eq.set_index("rule").loc[list(au["rule"])].reset_index()
    band = f.table("ordering_intervals") if _has(f, "ordering_intervals") \
        else None
    diffs = f.table("ordering_differences") \
        if _has(f, "ordering_differences") else None
    spread = f.table("ordering_by_gamma") \
        if _has(f, "ordering_by_gamma") else None
    gammas = sorted({float(g) for g in spread["gamma"]}) \
        if spread is not None and len(spread) else []
    gamma_list = _join([f"{g:g}" for g in gammas])
    au_spread = spread[spread["system"] == "australia_as_legislated"] \
        if spread is not None and len(spread) else None
    au_eq_base = au_eq[au_eq["rule"] == baseline_rule].iloc[0]
    au_eq_best = au_eq.loc[au_eq["cec"].idxmax()]
    cec_ratio = float(au_eq_best["cec"]) / float(au_eq_base["cec"])
    # The whole factorial, read down the contribution rate and across the
    # benefit formula, so the headline contrast is a difference in one
    # institution rather than in two. A three-column version of this table
    # compared a 10%-saving American household with a 20.2%-saving
    # Australian one and called it the pension.
    order = [x for x in ("us_social_security", "us_matched_saving",
                         "age_pension_untested", "age_pension_matched",
                         "australia_as_legislated")
             if x in set(gapped["system"])]
    label = {"us_social_security": "United States, 10% saving",
             "us_matched_saving": "United States, 20.2% saving",
             "age_pension_untested": "Age Pension, no means test",
             "age_pension_matched": "Means-tested, 10% saving",
             "australia_as_legislated": "Australia as legislated"}

    out: List[Flowable] = [
        ctx.h1("#ordering. Which Portfolio Wins, and Under Which Rule")]
    out.append(ctx.p(
        "The previous section found the withdrawal rule a retiree should "
        "use. This one asks what that rule does to the question the paper "
        "opened with, and it is worth being precise about why the question "
        "needs asking separately."))
    out.append(ctx.p(
        f"Two orderings have been in play. The first is between "
        f"<i>portfolios</i>: all-equity against the target-date fund, which "
        f"is what Sections {SHORT_ORDER.index('baseline') + 1} and "
        f"{SHORT_ORDER.index('pension') + 1} report and what the "
        f"target-date literature is about. The second is between "
        f"<i>countries</i>: whether an Australian household consumes more "
        f"than an American one, which is what a comparison of certainty "
        f"equivalents across pension systems reports. Section "
        f"{SHORT_ORDER.index('leisure') + 1}.2 reports the second, rule by "
        f"rule, and it is tempting to read a system moving up that table "
        f"as the portfolio ordering reversing. It is not the same "
        f"quantity, and the two can move in opposite directions. This "
        f"section reports the portfolio ordering directly, under the rule "
        f"Section {SHORT_ORDER.index('longevity') + 1} selects as well as "
        f"under the one the literature assumes."))
    out.append(ctx.p(
        f"Two things about the grid before the numbers. The menu is "
        f"{_spelled(len(set(swept_all['strategy'])))} strategies rather "
        f"than the six of Section {SHORT_ORDER.index('baseline') + 1}: "
        f"cash is dropped, "
        f"because nothing here turns on it and a portfolio that no reader "
        f"is choosing between costs a column in every table. And every "
        f"cell is scored twice \u2014 once over the fixed horizon the rest "
        f"of the paper uses, once weighted by the survival curve Section "
        f"{SHORT_ORDER.index('longevity') + 1} argues for. Section "
        f"{SHORT_ORDER.index('ordering') + 1}.6 reports what the second "
        f"objective does; the tables below are the first, because that is "
        f"the objective Sections "
        f"{SHORT_ORDER.index('baseline') + 1} and "
        f"{SHORT_ORDER.index('pension') + 1} report and the one a reader "
        f"comparing them will expect."))
    # m3 and m9 of the referee report: the eight rules here are drawn from
    # the twenty-four ranked two sections earlier, and one of them is run
    # at a rate that section rejects. Both are defensible and neither was
    # stated.
    out.append(ctx.p(
        f"The rule menu needs the same treatment, because it is eight rules "
        f"where Section {SHORT_ORDER.index('longevity') + 1} ranks "
        f"{_spelled(int(len(f.table('longevity_ranking'))))}. It is not a "
        f"sample of them. It is the rule the literature assumes and this "
        f"paper spends by, the amortisation family across every assumed "
        f"return the grid carries — which is the family that supplies "
        f"its own floor, and therefore the whole of the mechanism being "
        f"tested — and one percentage-of-balance rule as the "
        f"representative of the rules that scale with the portfolio without "
        f"amortising. Running all "
        f"{int(len(f.table('longevity_ranking')))} against "
        f"{_spelled(len(order))} pension regimes, "
        f"{_spelled(len(set(swept_all['strategy'])))} portfolios and "
        f"sixteen deletions would cost a great deal to populate rows "
        f"nothing in the argument reads. The "
        f"percentage rule is run at "
        f"{float(f.cfg['ordering']['percent_rate']):.0%} rather than at the "
        f"{float(_best_percent_rate(f)):.0%} Section "
        f"{SHORT_ORDER.index('longevity') + 1} finds best for it, because "
        f"here it stands in for the four-per-cent convention the "
        f"withdrawal-rate literature is written about; at its own best rate "
        f"it would be a second amortisation-like rule rather than the "
        f"contrast it is included to provide."))
    rows = [["Withdrawal rule"] + [label.get(x, x) for x in order]]
    for rule in dict.fromkeys(gapped["rule"]):
        cells = [rule_label(str(rule))]
        for system in order:
            hit = gapped[(gapped["system"] == system)
                         & (gapped["rule"] == rule)]
            cells.append(f"{float(hit['gap_pct'].iloc[0]):+.2f}%"
                         if len(hit) else "—")
        rows.append(cells)
    out += ctx.table(
        rows,
        "The all-equity portfolio's lead over the target-date fund, in "
        "certainty-equivalent retirement consumption, for every pension "
        "system and every withdrawal rule. Positive means the all-equity "
        "portfolio wins.",
        note="Every cell is scored on the same simulated lifetimes, so a "
             "difference between two cells is the system, the rule or the "
             "portfolio and never a different draw of returns. The estate "
             "is included, because an amortisation rule spends the "
             "portfolio to zero by construction and a fixed real rule does "
             "not.")
    if len(us_base) and len(au_base):
        out.append(ctx.p(
            f"<b>The first finding survives being re-derived on this "
            f"grid.</b> Under a {rule_label(baseline_rule)} withdrawal "
            f"the all-equity portfolio "
            f"leads by {float(us_base['gap_pct'].iloc[0]):+.2f}% in the "
            f"American system and {float(au_base['gap_pct'].iloc[0]):+.2f}% "
            f"in the Australian one, on the same regimes and the same path "
            f"count as Section "
            f"{SHORT_ORDER.index('pension') + 1} — "
            f"{int(f.cfg['ordering']['n_paths']):,} lifetimes rather than "
            f"the {int(f.cfg['bootstrap']['n_paths']):,} of Section "
            f"{SHORT_ORDER.index('baseline') + 1}, which is why the same "
            f"American comparison reads "
            f"{f.advantage('balanced_all_equity', 'target_date_fund'):.1f}% "
            f"there and {float(us_base['gap_pct'].iloc[0]):.2f}% here. The "
            f"two are the same quantity at two Monte Carlo sample sizes and "
            f"neither is the sampling error that matters, which is the "
            f"panel's and is measured in Section "
            f"{SHORT_ORDER.index('ordering') + 1}.3."))
        out.append(ctx.p(
            f"These are not the percentages in that section's own table, "
            f"and the difference is a labelling one worth stating rather "
            f"than smoothing over. The column headed \u2018All-intl over "
            f"50/50\u2019 there compares two <i>all-equity</i> portfolios "
            f"with each other — a hundred per cent international against "
            f"the fifty-fifty domestic-international split — which is a "
            f"question about the international sleeve, not about equity "
            f"versus a glide path. Its ranking column carries the "
            f"portfolio comparison, and agrees with the table above: the "
            f"target-date fund is first in Australia and is not in the "
            f"United States. The table above puts a number on it."))
    if recovers:
        out.append(ctx.p(
            f"<b>And the second finding holds as a statement about "
            f"portfolios, not only about countries.</b> Under "
            f"{rec_rule} \u2014 the rule Section "
            f"{SHORT_ORDER.index('longevity') + 1} selects, which is not "
            f"the most generous rate in the family and is the one a "
            f"retiree is being advised to use \u2014 the all-equity "
            f"portfolio leads by {rec_gap} in the Australian system, "
            f"against {float(au_base['gap_pct'].iloc[0]):+.2f}% under the "
            f"fixed real rule. The family spans "
            f"{float(au['gap_pct'].drop(au['gap_pct'].idxmin()).min()):+.2f}% "
            f"to {float(au_best['gap_pct']):+.2f}%, so nothing here turns "
            f"on which rate is picked. {_spelled(len(winners))} of "
            f"{_spelled(len(au))} rules in the menu return the lead. A rule that supplies its own floor "
            f"restores the all-equity prescription, and that is a claim "
            f"about portfolios rather than an inference from one about "
            f"countries."))
    else:
        out.append(ctx.p(
            f"<b>The second finding does not hold as a statement about "
            f"portfolios.</b> No rule in the menu returns the lead to the "
            f"all-equity portfolio in the Australian system: the gap runs "
            f"from {float(au_worst['gap_pct']):+.2f}% under "
            f"{au_worst['rule']} to {float(au_best['gap_pct']):+.2f}% under "
            f"{au_best['rule']}, and the target-date fund leads throughout. "
            f"That an amortisation rule lets the Australian <i>household</i> "
            f"overtake the American one is true and is reported in Section "
            f"{SHORT_ORDER.index('leisure') + 1}; it is a statement about "
            f"the level of consumption in two countries, and we withdraw "
            f"the portfolio reading of it."))
    out.append(ctx.p(
        f"Either way the rule belongs in the statement of the result. "
        f"Within the Australian system alone, changing nothing but the "
        f"withdrawal rule moves the all-equity lead by "
        f"{float(au['gap_pct'].max() - au['gap_pct'].min()):.1f} percentage "
        f"points. A plan sponsor choosing a portfolio default without also "
        f"choosing a drawdown default is choosing on an axis that explains "
        f"less than the one they left open."))
    # The 2x2 inside the five columns, which is what lets the headline be
    # about one institution rather than two.
    out += _matched_factorial(ctx, f, baseline_rule)
    out.append(ctx.h2("#ordering.2 The reversal lives in the worst cell of "
                      "the table"))
    out.append(ctx.p(
        f"One row of the table deserves to be read against the rest. "
        f"{_spelled(int((gapped['gap_pct'] < 0).sum())).capitalize()} of "
        f"the {int(len(gapped))} cells are negative, both of them "
        f"means-tested and both of them in the same row: the fixed real "
        f"withdrawal — the 4% rule, and the one the lifecycle "
        f"literature and this paper's earlier sections all spend by. Every "
        f"other rule in the menu leaves the all-equity portfolio ahead "
        f"under every pension in the grid. Section "
        f"{SHORT_ORDER.index('longevity') + 1} has already found that rule "
        f"is not the one a retiree should use. This table says what it "
        f"costs the household that does."))
    lived = "prob_ruin_survival" in au_eq
    rows = [["Withdrawal rule", "CEC", "Ruin, fixed horizon"]
            + (["Ruin, real lifespan"] if lived else [])]
    for _, row in au_eq.iterrows():
        rows.append([rule_label(str(row["rule"])),
                     f"{float(row['cec']):.4f}",
                     f"{float(row['prob_ruin']):.1%}"]
                    + ([f"{float(row['prob_ruin_survival']):.1%}"]
                       if lived else []))
    out += ctx.table(
        rows,
        "The all-equity portfolio under each withdrawal rule, in the "
        "Australian system. The first row is the rule under which the "
        "pension reverses the portfolio ordering.",
        note="Same simulated lifetimes as the table above. Ruin under a "
             "fixed horizon is the share of paths whose portfolio is "
             "exhausted before the terminal age; under a real lifespan it "
             "is the share exhausted before the retiree dies, integrated "
             "over the survival curve, which is the measure Section "
             f"#longevity argues for and the smaller of the two by "
             f"construction.")
    ruin_gap_fixed = (float(au_eq_base['prob_ruin'])
                      - float(au_eq_best['prob_ruin']))
    line = (
        f"The fixed real rule leaves this household on "
        f"{float(au_eq_base['cec']):.4f} against "
        f"{float(au_eq_best['cec']):.4f} under "
        f"{rule_label(str(au_eq_best['rule']))} \u2014 a "
        f"reduction of {100 * (1 - 1 / cec_ratio):.0f}% in "
        f"certainty-equivalent consumption \u2014 and exhausts the "
        f"portfolio on {float(au_eq_base['prob_ruin']):.1%} of paths "
        f"against {float(au_eq_best['prob_ruin']):.1%}. It is dominated on "
        f"both margins by every other rule in the menu.")
    if lived:
        ruin_gap_lived = (float(au_eq_base['prob_ruin_survival'])
                          - float(au_eq_best['prob_ruin_survival']))
        line += (
            f" The ruin margin is the one to state carefully, because "
            f"Section {SHORT_ORDER.index('longevity') + 1} objects to the "
            f"measure it is stated in: counting a portfolio exhausted at "
            f"ninety-one as a failed retirement for a household that most "
            f"likely died before then overstates what went wrong. "
            f"Integrated over the survival curve instead, the rule ruins "
            f"{float(au_eq_base['prob_ruin_survival']):.1%} of the time "
            f"against {float(au_eq_best['prob_ruin_survival']):.1%} \u2014 "
            f"a gap of {100 * ruin_gap_lived:.1f} points rather than "
            f"{100 * ruin_gap_fixed:.1f}. The dominance holds on either "
            f"measure and is worth about half as much on the better one, "
            f"which is the honest way to carry it.")
    out.append(ctx.p(line))
    out.append(ctx.p(
        "That is not a reason to discard the reversal, and we are not "
        "discarding it. A fixed real withdrawal is what the withdrawal-rate "
        "literature is written about, what most retirement calculators "
        "implement, and what the study this paper replicates assumes "
        "throughout; a result about the households actually following it is "
        "a result about a large number of real people. It is a reason to "
        "state the finding with its condition attached. The all-equity "
        "prescription survives an asset-tested pension for a retiree who "
        "spends a share of the balance, and does not survive it for one who "
        "spends a fixed real amount. Which of those a household is doing is "
        "a choice, and on this evidence it matters more than the portfolio "
        "choice it is usually treated as subordinate to."))
    out.append(ctx.h2("#ordering.3 How precisely the panel resolves each "
                      "sign"))
    out.append(ctx.p(
        "Every claim in this paper is a claim about a sign, and the panel "
        "is sixteen developed markets whose twentieth centuries were not "
        "independent of one another. So each cell above is recomputed "
        "sixteen times, once with each country's history removed, and the "
        "delete-one jackknife gives the sampling error the panel carries. "
        "That is the error to weigh a sign against. Monte Carlo error is "
        "not: a hundred thousand paths drive it close to zero without "
        "adding a single country of evidence."))
    if band is not None and len(band):
        rows = [["Pension system", "Withdrawal rule", "Lead (%)",
                 "Jackknife s.e.", "95% interval", "Sign holds in all 16"]]
        for _, row in band.iterrows():
            rows.append([
                label.get(str(row["system"]), str(row["system"])),
                rule_label(str(row["rule"])),
                f"{float(row['gap_pct']):+.2f}",
                f"{float(row['standard_error']):.2f}",
                f"[{float(row['ci_low']):+.2f}, {float(row['ci_high']):+.2f}]",
                "yes" if bool(row["sign_survives_every_deletion"]) else "no"])
        out += ctx.table(
            rows,
            "Delete-one-country intervals on the all-equity lead, for "
            "every system a headline is drawn from.",
            note="The jackknife standard error over sixteen sub-panels, "
                 "each holding fifteen markets. An interval containing zero "
                 "is a cell in which this panel cannot say which portfolio "
                 "wins.")
    out += _both_reversals(ctx, f, baseline_rule)
    out.append(ctx.h2("#ordering.4 The interval on the difference, which "
                      "is the claim"))
    out.append(ctx.p(
        "The claim the evidence does support is a difference rather than a "
        "level \u2014 that changing the withdrawal rule moves the lead by "
        "tens of points \u2014 and an interval on two levels is not an "
        "interval on their difference. The sixteen sub-panels are shared "
        "between cells, so the differences are paired and get a jackknife "
        "of their own rather than one assembled out of two marginal "
        "standard errors."))
    systems_here = ([str(x) for x in dict.fromkeys(diffs["system"])]
                    if diffs is not None and "system" in diffs else [])
    if diffs is not None and len(diffs):
        header = ["Rule, against the fixed real rule"]
        for system in systems_here:
            header += [f"{label.get(system, system)}: difference (pp)",
                       "95% interval"]
        if not systems_here:
            header = ["Rule, against the fixed real rule", "Difference (pp)",
                      "Jackknife s.e.", "95% interval", "Cell correlation"]
        rows = [header]
        keys = list(dict.fromkeys(diffs["rule"]))
        for key in keys:
            cells = [rule_label(str(key))]
            if systems_here:
                for system in systems_here:
                    hit = diffs[(diffs["system"] == system)
                                & (diffs["rule"] == key)]
                    if len(hit):
                        r = hit.iloc[0]
                        cells += [f"{float(r['difference_pp']):+.1f}",
                                  f"[{float(r['ci_low']):+.1f}, "
                                  f"{float(r['ci_high']):+.1f}]"]
                    else:
                        cells += ["\u2014", "\u2014"]
            else:
                r = diffs[diffs["rule"] == key].iloc[0]
                cells += [f"{float(r['difference_pp']):+.1f}",
                          f"{float(r['standard_error']):.1f}",
                          f"[{float(r['ci_low']):+.1f}, "
                          f"{float(r['ci_high']):+.1f}]",
                          f"{float(r['correlation']):.2f}"]
            rows.append(cells)
        out += ctx.table(
            rows,
            "Delete-one-country intervals on the difference between each "
            "withdrawal rule and the fixed real rule, under every pension "
            "the jackknife covers.",
            anchor="rule_differences",
            note="Paired across the sixteen sub-panels, so the pairing can "
                 "cancel whatever the deletions do to both cells at once. "
                 "How much it helps is a property of the cells: where the "
                 "level is badly resolved the difference built on it "
                 "inherits some of that.",
            font_size=7.0)
        # Named rather than ranked. A draft picked the systems by which
        # had the smallest median standard error and then labelled them
        # "matched" and "legislated", which put the American column's
        # numbers under both names.
        matched_here = _matched(f, baseline_rule)
        pair = ((matched_here["systems"]["means_tested/voluntary"],
                 "australia_as_legislated")
                if matched_here.get("measured") else ())
        if len(pair) == 2 and all(x in systems_here for x in pair):
            def _stat(system: str) -> Tuple[float, int, int]:
                block = diffs[diffs["system"] == system]
                return (float(block["standard_error"].median()),
                        int(block["ci_excludes_zero"].sum()), len(block))

            mt_se, mt_n, mt_of = _stat(pair[0])
            au_se, au_n, au_of = _stat(pair[1])
            out.append(ctx.p(
                f"<b>The rule effect is resolved under both means-tested "
                f"regimes, and far more tightly at matched "
                f"contributions.</b> There the standard errors run about "
                f"{mt_se:.1f} points and {mt_n} of {mt_of} differences "
                f"exclude zero; as legislated they run about {au_se:.1f} "
                f"and {au_n} of {au_of} do. The gap between those two "
                f"columns is not a fact about withdrawal rules — the "
                f"point estimates in them differ by only a few points. It "
                f"is the badly behaved cell of Section "
                f"{SHORT_ORDER.index('ordering') + 1}.3 propagating into "
                f"every difference built against it, which is what one "
                f"should expect and is the reason to print both."))
        diffs = (diffs[diffs["system"] == "australia_as_legislated"]
                 if systems_here else diffs)
        widest = diffs.loc[diffs["difference_pp"].abs().idxmax()]
        resolved_n = int(diffs["ci_excludes_zero"].sum())
        out.append(ctx.p(
            f"<b>The widest rule effect is "
            f"{rule_label(str(widest['rule']))} at "
            f"{float(widest['difference_pp']):+.1f} percentage points, with "
            f"a standard error of {float(widest['standard_error']):.1f} and "
            f"an interval of [{float(widest['ci_low']):+.1f}, "
            f"{float(widest['ci_high']):+.1f}].</b> "
            f"{resolved_n} of {len(diffs)} differences exclude zero. The "
            f"pairing helps less than one might hope: the cells correlate "
            f"{float(diffs['correlation'].median()):.2f} across deletions "
            f"at the median, so these intervals are not much tighter than "
            f"the levels they come from. We report that too, because the "
            f"reason to compute a difference interval at all is that it "
            f"<i>can</i> be far tighter, and here it is not."))
        out.append(ctx.p(
            f"So the rule effect is resolved under both pensions, the "
            f"reversal is resolved at matched contributions, and what "
            f"remains unresolved is the legislated cell alone. The honest "
            f"summary is that sixteen countries can separate a "
            f"{abs(float(widest['difference_pp'])):.0f}-point effect from "
            f"zero and cannot separate a two-point one \u2014 which is what "
            f"one would expect, and is worth having established rather than "
            f"assumed."))
    if spread is not None and len(spread):
        out.append(ctx.h2("#ordering.5 And the same grid at other risk "
                          "aversions"))
        out.append(ctx.p(
            f"The interval above is what the panel costs in precision. It "
            f"says nothing about the other objection, which is that a "
            f"certainty equivalent is a statement about a felicity "
            f"function as much as about a portfolio. Scoring an outcome "
            f"again costs nothing next to producing it, so the whole grid "
            f"is re-scored at risk aversions of {gamma_list}. "
            f"This is the check "
            f"Section {SHORT_ORDER.index('incidence') + 1} runs against "
            f"preferences and this section had been running only against "
            f"the panel; each result is now checked against both."))
        rows = [["Withdrawal rule"]
                + [f"\u03b3 = {g:g}" for g in gammas]]
        for rule in dict.fromkeys(au_spread["rule"]):
            hit = au_spread[au_spread["rule"] == rule].set_index("gamma")
            rows.append([rule_label(str(rule))]
                        + [f"{float(hit.loc[g, 'gap_pct']):+.2f}"
                           if g in hit.index else "\u2014" for g in gammas])
        out += ctx.table(
            rows,
            "The all-equity portfolio's lead over the target-date fund in "
            "the Australian system, at three risk aversions.",
            note="The same simulated lifetimes throughout: only the "
                 "objective they are scored under changes, so the "
                 "differences across a row are preference effects and "
                 "nothing else.")
        flips = [str(r) for r in dict.fromkeys(au_spread["rule"])
                 if len({_sign(float(v)) for v in
                         au_spread[au_spread["rule"] == r]["gap_pct"]}) > 1]
        base_row = au_spread[au_spread["rule"] == baseline_rule] \
            .sort_values("gamma")
        walk = _join([f"{v:+.2f}% at \u03b3 = {g:g}"
                      for g, v in zip(base_row["gamma"],
                                      base_row["gap_pct"])])
        if flips:
            out.append(ctx.p(
                f"<b>One row changes sign, and it is the contested one.</b> "
                f"Under a {rule_label(baseline_rule)} withdrawal the lead "
                f"runs "
                f"{walk}. "
                f"Every other row in the table is positive at every risk "
                f"aversion. So the reversal this paper reports is not a "
                f"property of the means test on its own, and not of the "
                f"withdrawal rule on its own: it needs a fixed real "
                f"withdrawal <i>and</i> a household risk-averse enough for "
                f"the loss of the floor to outweigh the equity premium. "
                f"Between \u03b3 = "
                f"{min(base_row['gamma']):g} and \u03b3 = 5 the sign "
                f"turns over."))
            out.append(ctx.p(
                "That is a sharper statement of the paper's thesis than "
                "the one we set out to make, and a more conditional one. "
                "The interaction is not two-way but three-way, and the "
                "third term is the one a plan sponsor cannot observe. A "
                "default portfolio is chosen for a population, and this "
                "table says the right default in a means-tested system "
                "depends on where in that population the sponsor thinks "
                "the marginal member sits."))
        else:
            out.append(ctx.p(
                f"<b>No row changes sign.</b> The contested cell runs "
                f"{walk}, "
                f"so the ordering reported here is a property of the "
                f"pension and the rule rather than of the curvature of "
                f"the objective."))
    if _has(f, "ordering_by_objective"):
        objs = f.table("ordering_by_objective")
        wide = objs.pivot_table(index=["system", "rule"],
                                columns="objective",
                                values="gap_pct")
        if {"cec", "cec_survival"} <= set(wide.columns) and len(wide):
            moved = (wide["cec_survival"] - wide["cec"]).abs()
            flips = int(((np.sign(wide["cec"].round(6))
                          != np.sign(wide["cec_survival"].round(6)))).sum())
            key = ("australia_as_legislated", baseline_rule)
            out.append(ctx.h2("#ordering.6 And the same grid on a real "
                              "lifespan"))
            out.append(ctx.p(
                f"One objection remains, and it is one this paper raises "
                f"against itself. Section "
                f"{SHORT_ORDER.index('longevity') + 1} rejects a "
                f"retirement that ends at ninety-three with certainty as "
                f"not neutral <i>between</i> withdrawal rules, and "
                f"re-solves the rule against a survival curve. This "
                f"section compares portfolios <i>within</i> a rule, which "
                f"is a different exposure \u2014 a horizon that flatters "
                f"the rules dividing by it flatters both portfolios in a "
                f"cell alike, so the gap between them should be close to "
                f"insulated even where the levels are not. Should be is "
                f"not is, so every outcome above was scored a second time "
                f"under the objective that section argues for."))
            lvl = ""
            if "cec" in swept_all and "cec_survival" in swept_all:
                shift = (swept_all["cec_survival"] / swept_all["cec"]
                         - 1.0) * 100.0
                lvl = (f"The levels do move: across the grid the "
                       f"certainty equivalents shift over a span of "
                       f"{float(shift.max() - shift.min()):.1f} points, "
                       f"from {float(shift.min()):+.1f}% to "
                       f"{float(shift.max()):+.1f}%. ")
            out.append(ctx.p(
                f"<b>{lvl}The gaps do not.</b> Across all "
                f"{len(wide)} cells the all-equity lead moves by "
                f"{float(moved.median()):.2f} percentage points at the "
                f"median and {float(moved.max()):.2f} at the worst, and "
                f"{'no sign changes' if not flips else f'{flips} signs change'} "
                f"\u2014 the dial here is the horizon, and the one that "
                f"does move a sign is the risk aversion of Section "
                f"{SHORT_ORDER.index('ordering') + 1}.5. "
                + (f"The contested cell goes "
                   f"{float(wide.loc[key, 'cec']):+.2f}% to "
                   f"{float(wide.loc[key, 'cec_survival']):+.2f}%. "
                   if key in wide.index else "")
                + f"So the insulation is not an argument but a "
                f"measurement: pricing a real lifespan moves the levels by "
                f"several times what it moves the comparison this paper is "
                f"about, and moves no sign at all."))
            out.append(ctx.p(
                "We report it because the alternative was to ask the "
                "reader to believe it. A paper that spends a section "
                "arguing one objective is the right one, and then reports "
                "its central table under another, owes the reader the "
                "difference rather than the reasoning."))
    out += ctx.figure(
        "fig67_ordering",
        "The all-equity portfolio's lead over the target-date fund, by "
        "pension system and withdrawal rule. One panel per system on a "
        "shared scale; the zero line is the ranking, the outlined bar is "
        "the rule the rest of this paper spends by, and the whiskers are "
        "delete-one-country 95% intervals where they were computed.")
    return out


def limitations(ctx: Any) -> List[Flowable]:
    """This paper's limitations, not the companion study's.

    The inherited section catalogues the threats to twenty-eight results.
    Most of them are irrelevant here and the pointers to sections this
    paper does not contain are worse than useless, so it is rewritten
    around the four things that would actually change these conclusions.
    """
    f = ctx.f
    # This section's job is to say how far the panel can be trusted, so it
    # has to quote *this paper's* headline and not the companion's. It read
    # `panel_influence` for several drafts, which is the leave-one-out on
    # the all-international-against-50/50 comparison: a different quantity,
    # positive in all sixteen deletions, and therefore a sentence claiming
    # robustness that Section #ordering.3 spends two pages denying.
    baseline_rule = str(f.cfg["lifecycle"]["retirement"]["rule"])
    band = f.table("ordering_intervals") \
        if _has(f, "ordering_intervals") else None
    contested = resolved = None
    seat = None
    if _has(f, "ordering_pseudo"):
        pseudo = f.table("ordering_pseudo")
        hit = pseudo[(pseudo["system"] == "australia_as_legislated")
                     & (pseudo["rule"] == baseline_rule)]
        seat = hit.iloc[0] if len(hit) else None
    if band is not None and len(band):
        hit = band[(band["system"] == "australia_as_legislated")
                   & (band["rule"] == baseline_rule)]
        contested = hit.iloc[0] if len(hit) else None
        resolved = band[band["ci_excludes_zero"].astype(bool)]
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
        "century happened to all of them at once. Section "
        f"#ordering.3 recomputes every cell of the ordering table sixteen "
        "times, once with each country's history removed, and the answer "
        "is not uniform across the table \u2014 which is why this paper "
        "separates the institutional effects it can sign from the "
        "residual it cannot.")
    _matched_here = _matched(f, str(f.cfg["lifecycle"]["retirement"]["rule"]))
    _mt_iv = (_interval_for(
        f, _matched_here["systems"]["means_tested/voluntary"],
        str(f.cfg["lifecycle"]["retirement"]["rule"]))
        if _matched_here.get("measured") else {"measured": False})
    if _mt_iv.get("measured") and _mt_iv.get("excludes_zero"):
        line += (
            f" The reversal itself, measured against a household matched "
            f"on contributions, is one of the cells it settles: "
            f"{_mt_iv['gap_pct']:+.2f}% with an interval of "
            f"[{_mt_iv['ci'][0]:+.2f}, {_mt_iv['ci'][1]:+.2f}] and the same "
            f"ordering in all sixteen sub-panels.")
    if contested is not None:
        line += (
            f" What it does not settle is Australia's own combination of "
            f"two institutions that offset. That cell is "
            f"{float(contested['gap_pct']):+.2f}% "
            f"with a delete-one standard error of "
            f"{float(contested['standard_error']):.2f} and an interval of "
            f"[{float(contested['ci_low']):+.2f}, "
            f"{float(contested['ci_high']):+.2f}], which contains zero; "
            f"across the sixteen sub-panels it runs "
            f"[{float(contested['loo_low']):+.2f}, "
            f"{float(contested['loo_high']):+.2f}] and its sign "
            f"{'holds throughout' if bool(contested['sign_survives_every_deletion']) else 'does not survive every deletion'}.")
    if resolved is not None and len(resolved):
        line += (
            f" The rest of the table is not like that: "
            f"{len(resolved)} of {len(band)} cells have intervals "
            f"excluding zero, the matched reversal among them. So the "
            f"cross-section is small enough to leave one sign undetermined "
            f"and large enough to settle every other, and the two should "
            f"not be reported in the same voice.")
    if seat is not None:
        line += (
            f" And the interval is the weaker half of that, in <i>both</i> "
            f"the cells where the ordering reverses. Only "
            f"{int(seat['below_point'])} of {int(seat['deletions'])} "
            f"deletions fall below the legislated estimate, the mean "
            f"deletion is {float(seat['loo_mean']):+.2f}%, and the matched "
            f"cell is skewed the same way")
        _mt_ps = None
        if _has(f, "ordering_pseudo") and _matched_here.get("measured"):
            _p = f.table("ordering_pseudo")
            _hit = _p[(_p["system"]
                       == _matched_here["systems"]["means_tested/voluntary"])
                      & (_p["rule"]
                         == str(f.cfg["lifecycle"]["retirement"]["rule"]))]
            _mt_ps = _hit.iloc[0] if len(_hit) else None
        if _mt_ps is not None:
            line += (
                f" — {int(_mt_ps['below_point'])} of "
                f"{int(_mt_ps['deletions'])}, an implied bias of "
                f"{float(_mt_ps['bias_estimate']):+.1f} points against the "
                f"legislated cell's {float(seat['bias_estimate']):+.1f}. "
                f"So neither estimate is a typical member of its own "
                f"sub-panels and neither standard error carries what it "
                f"looks like it carries. What separates the two cells is "
                f"not the interval but the count of deletions that keep "
                f"the sign, which assumes nothing about how they scatter: "
                f"sixteen of sixteen at matched contributions, eight as "
                f"legislated.")
        else:
            line += (
                ", so the estimate is not a typical member of its own "
                "sub-panels and the standard error built on them carries "
                "less than it looks like it does.")
    split = (_split_found(f, str(f.cfg["lifecycle"]["retirement"]["rule"]))
             if _has(f, "ordering_influence") else {"measured": False})
    if split.get("measured"):
        line += (
            f" That is a statement about the estimator and not about how "
            f"many countries reverse: {_spelled(int(split['sign_holds']))} of "
            f"{int(split['deletions'])} fifteen-country panels still put the "
            f"target-date fund ahead and "
            f"{_spelled(int(split['sign_flips']))} do not, with removing "
            f"{COUNTRY_NAME.get(str(split['largest_flip']), str(split['largest_flip']))} "
            f"alone moving the cell "
            f"{split['largest_flip_swing']:+.1f} points. Section "
            f"#ordering.3 gives the arithmetic; the short form is that the "
            f"reversal is carried by a handful of identifiable markets in "
            f"this panel, and no wider panel is available to say whether "
            f"they are representative.")
    line += (
        " Nothing in this paper should be read as a point estimate, and "
        "every claim it makes is a claim about a sign.")
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
    band = f.table("incidence_band_profile")
    arms = list(dict.fromkeys(band["arm"])) if "arm" in band else []
    rule_arm = band[band["arm"] == arms[-1]] if len(arms) > 2 else band
    gapped = f.table("ordering_gaps")
    baseline_rule = str(f.cfg["lifecycle"]["retirement"]["rule"])
    au_rows = gapped[gapped["system"] == "australia_as_legislated"]
    us_rows = gapped[gapped["system"] == "us_social_security"]
    au_best = au_rows.loc[au_rows["gap_pct"].idxmax()]
    # How many fifteen-country panels reproduce the contested sign, which
    # is not the count the bias diagnostic reports and was once confused
    # with it. Counted rather than typed, so a rerun cannot leave the
    # sentence behind.
    split = _split_found(f, baseline_rule)
    # The rule Section #longevity selects rather than the best cell of
    # the grid. Leading with the maximiser would make that section
    # decorative at the one place its answer should bind, so the
    # recommendation is the headline and the maximum sits beside it.
    rec = _recommended(f)
    rec_gap = (f"{rec['gap_pct']:+.2f}%" if rec.get("measured")
               else f"{float(au_best['gap_pct']):+.2f}%")
    rec_rule = (rule_label(str(rec["rule"])) if rec.get("measured")
                else rule_label(str(au_best["rule"])))
    recovers = bool(float(au_best["gap_pct"]) > 0.0)
    swept_all = f.table("ordering_sweep")
    au_eq = swept_all[(swept_all["system"] == "australia_as_legislated")
                      & (swept_all["strategy"] == "balanced_all_equity")]
    au_eq_base = au_eq[au_eq["rule"] == baseline_rule].iloc[0]
    au_eq_best = au_eq.loc[au_eq["cec"].idxmax()]
    ruin_gap = float(au_eq_base["prob_ruin"]) - float(au_eq_best["prob_ruin"])
    # The same margin on the measure Section #longevity argues for. Quoting
    # only the fixed-horizon one would state this paper's case for
    # abandoning the rule in the units this paper says are wrong.
    ruin_gap_lived = (
        float(au_eq_base["prob_ruin_survival"])
        - float(au_eq_best["prob_ruin_survival"])
        if "prob_ruin_survival" in au_eq else float("nan"))
    cec_ratio = float(au_eq_best["cec"]) / float(au_eq_base["cec"])
    iv = f.table("ordering_intervals") \
        if _has(f, "ordering_intervals") else None
    contested = None
    if iv is not None and len(iv):
        _hit = iv[(iv["system"] == "australia_as_legislated")
                    & (iv["rule"] == baseline_rule)]
        contested = _hit.iloc[0] if len(_hit) else None
    gamma_list, gamma_walk = _gamma_walk(f, baseline_rule)
    matched = _matched(f, baseline_rule)
    mt_iv = (_interval_for(f, matched["systems"]["means_tested/voluntary"],
                           baseline_rule)
             if matched.get("measured") else {"measured": False})
    au_base_gap = float(
        au_rows[au_rows["rule"] == baseline_rule]["gap_pct"].iloc[0])
    flat = gapped[(gapped["system"] == "age_pension_untested")
                  & (gapped["rule"] == baseline_rule)]
    flat_gap = float(flat["gap_pct"].iloc[0]) if len(flat) else float("nan")
    us_base_gap = float(
        us_rows[us_rows["rule"] == baseline_rule]["gap_pct"].iloc[0])
    out: List[Flowable] = [ctx.h1("#conclusion. Conclusion")]
    out.append(ctx.p(
        f"The case for a fixed all-equity lifecycle portfolio is not one "
        f"case but a family of them, and which member you are in is "
        f"settled by two institutions and one preference rather than by "
        f"the return process. Give an "
        f"investor an earnings-related pension and a fixed real withdrawal "
        f"and the all-equity portfolio leads the target-date fund by "
        f"{us_base_gap:+.2f}%. Change the pension to an asset-tested one, "
        f"holding the household and its contributions still, and the lead "
        f"becomes "
        f"{matched['cells']['means_tested/voluntary']:+.2f}%: the "
        f"de-risking glide path, the design the literature spends its "
        f"length arguing against, wins. Change the withdrawal rule instead "
        f"and it does not."
        if matched.get("measured") else
        f"and the all-equity portfolio leads the target-date fund by "
        f"{us_base_gap:+.2f}%. Change the pension to an asset-tested one "
        f"and the lead becomes {au_base_gap:+.2f}%."))
    if mt_iv.get("measured") and mt_iv.get("excludes_zero"):
        out.append(ctx.p(
            f"That sign the panel can carry: the delete-one interval is "
            f"[{mt_iv['ci'][0]:+.2f}, {mt_iv['ci'][1]:+.2f}] and the "
            f"ordering is the same in all sixteen sub-panels. What it "
            f"cannot carry is the number an Australian actually faces. "
            f"Australia pairs the means test with a compulsory "
            f"contribution, which under that test is worth "
            f"{matched['contribution_effect_means_tested']:+.1f} points "
            f"because a larger balance is a partial substitute for the "
            f"floor the test removed; the two nearly cancel, the system as "
            f"legislated lands at {float(contested['gap_pct']):+.2f}%, and "
            f"sixteen countries put a standard error of "
            f"{float(contested['standard_error']):.1f} points on that "
            f"remainder — the sign changes when "
            f"{_spelled(int(split['sign_flips'])) if split.get('measured') else 'half'} "
            f"of the "
            f"{_spelled(int(split['deletions'])) if split.get('measured') else 'sixteen'} "
            f"markets are removed one at a time. A cross-section of "
            f"sixteen can sign an institution and cannot sign the "
            f"difference between two of them that happen to offset."))
    else:
        out.append(ctx.p(
            f"We would not have a reader stop at that sentence, because "
            f"the panel does not support it. Sixteen countries put a "
            f"standard error of "
            f"{float(contested['standard_error']):.1f} points on a gap of "
            f"{float(contested['gap_pct']):+.2f}%."))
    _mt_rows = (gapped[gapped["system"]
                       == matched["systems"]["means_tested/voluntary"]]
                if matched.get("measured") else au_rows)
    out.append(ctx.p(
        f"The other thing the panel resolves comfortably, and in every "
        f"cell, is the withdrawal rule's effect. Under the system as "
        f"legislated the rule moves the all-equity lead by "
        f"{float(au_rows['gap_pct'].max() - au_rows['gap_pct'].min()):.0f} "
        f"percentage points across the menu, and at matched contributions "
        f"by "
        f"{float(_mt_rows['gap_pct'].max() - _mt_rows['gap_pct'].min()):.0f} "
        f"— larger than either institutional effect under either "
        f"pension, and an order of magnitude larger than the remainder in "
        f"dispute. It is the one quantity in this paper that no choice "
        f"among the arms makes small."))
    if gamma_walk:
        out.append(ctx.p(
            f"A second check points the same way from a different "
            f"direction, and it reaches the resolved cell as well as the "
            f"unresolved one. Re-scoring the identical simulated lifetimes "
            f"at risk aversions of {gamma_list}, the matched means-tested "
            f"cell runs "
            f"{_gamma_walk(f, baseline_rule, matched['systems']['means_tested/voluntary'])[1] or gamma_walk} "
            f"and the legislated one {gamma_walk} — neither is negative at "
            f"the lowest of the three — while every other withdrawal rule "
            f"in the menu stays positive throughout. The reversal "
            f"therefore needs three things at once: an asset-tested "
            f"pension, a fixed real withdrawal, and a household "
            f"risk-averse enough to pay for the floor. Its absence needs "
            f"only one of the three to fail, and that is as true of the "
            f"cell the panel signs as of the cell it cannot."
            if matched.get("measured") else
            f"A second check points the same way from a different "
            f"direction. Re-scoring the identical simulated lifetimes at "
            f"risk aversions of {gamma_list}, the contested cell runs "
            f"{gamma_walk} — it is not negative at the lowest one — while "
            f"every other withdrawal rule in the menu stays positive "
            f"throughout."))
    out.append(ctx.p(
        f"That last sentence is the one we would put first. The reversal "
        f"holds in {_spelled(int((au_rows['gap_pct'] < 0).sum()))} of the "
        f"{_spelled(int(len(au_rows)))} withdrawal rules we run, and the "
        f"rule it "
        f"holds under is the fixed real withdrawal — which costs the same "
        f"Australian household {100 * (1 - 1 / cec_ratio):.0f}% of its "
        f"certainty-equivalent consumption against the best rule in the "
        f"same menu and runs out "
        + (f"{100 * ruin_gap_lived:.1f} percentage points more often over "
           f"a real lifespan \u2014 {100 * ruin_gap:.1f} if ruin is "
           f"counted to a terminal age nobody is promised. "
           if ruin_gap_lived == ruin_gap_lived else
           f"{100 * ruin_gap:.0f} percentage points more often. ")
        + f"A retiree who follows Section "
        f"{SHORT_ORDER.index('longevity') + 1}'s advice never meets the "
        f"reversal. We report it because it is what the literature's "
        f"standard assumption produces, not because we think a retiree "
        f"should be in that cell."))
    out.append(ctx.p(
        f"The reason is worth separating from the obvious one. It is "
        f"tempting to attribute the reversal to the means test's taper, "
        f"which is steep enough to act as a wealth tax. It cannot be that: "
        f"this household is past the cut-off before the test applies. It is "
        f"the loss of the floor, and the cleanest evidence for that is a "
        f"pair of cells differing in one thing. Paying the Age Pension's "
        f"own flat rate to this household without testing it leaves the "
        f"all-equity portfolio ahead by {flat_gap:+.2f}%; testing that "
        f"identical benefit, on the same contributions and the same "
        f"panel, takes it to "
        f"{matched['cells']['means_tested/voluntary']:+.2f}%. Nothing else "
        f"moves between those two runs — not the rate, not the saving "
        f"rate, not a single simulated return. Withdrawing a pension at "
        f"the margin and paying a smaller one are different "
        f"interventions, and only the first reorders portfolios."
        if matched.get("measured") else
        f"the loss of the floor. Paying the same pension to everyone, with "
        f"no test at all, leaves the all-equity portfolio ahead by "
        f"{flat_gap:+.2f}%."))
    out.append(ctx.p(
        f"The second finding qualifies the first and should travel with it. "
        f"A withdrawal rule that cannot deplete the portfolio supplies the "
        f"floor the pension no longer does, and holding everything else "
        f"fixed it moves the all-equity lead in the Australian system to "
        f"{rec_gap} under the {rec_rule} rule Section "
        f"{SHORT_ORDER.index('longevity') + 1} selects, and to "
        f"{float(au_best['gap_pct']):+.2f}% at the most generous rate in "
        f"the family"
        f"{', either of which returns the ordering the pension had reversed' if recovers else ', neither of which returns the lead'}. "
        f"So the reversal is not a fact about Australia. It is a fact about "
        f"any retiree whose income in the bad states depends on the "
        f"portfolio itself — which includes a saver under an asset test, "
        f"and excludes one drawing on an annuity, a defined-benefit scheme, "
        f"or a rule that divides by the years remaining."))
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
        "This is where the paper starts rather than where it ends, and it "
        "is worth saying which of the two before the numbers arrive. "
        "Nothing changes but the pension the household retires onto, and "
        "the ordering the previous section established reverses. Two "
        "things qualify that, both established later and both worth "
        "carrying through this section: the reversal appears under one of "
        "the eight withdrawal rules this paper runs, and the cell reported "
        "here -- Australia as legislated, which pairs the means test with "
        "a compulsory contribution -- is the one cell of that table whose "
        "sign sixteen countries cannot resolve. Holding the contribution "
        "rate still instead gives a reversal the panel does sign, and that "
        "is the version the ordering section leads with; this section "
        "reports the country as it is. Neither qualification is a reason "
        "to skip the section -- the reversal is what the literature's "
        "standard assumption produces, and understanding why it happens is "
        "what the rest of the paper is built on -- but every number below "
        "should be read as a point estimate with the interval reported in "
        "the ordering section attached."),
    "leisure": (
        "The point estimate has to be attributed to something, and there "
        "are two candidates: the means test's taper, and the loss of the "
        "unconditional floor beneath it. Australia's pension differs from "
        "the American schedule in two ways at once -- it is worked out by a "
        "means test rather than from a career of earnings, and it starts on "
        "a fixed birthday rather than when work stops -- so this section "
        "crosses the two features and reports each alone against both "
        "together. Because the second feature is about timing, the "
        "retirement date is re-solved in every cell rather than held fixed, "
        "which is what the date columns in the table report and why these "
        "certainty equivalents are aggregated from the start of working "
        "life. They are not comparable with the levels in the previous "
        "section; only the differences within this one should be read."),
    "longevity": (
        "If the mechanism is a missing floor rather than a tax on wealth, a "
        "household should be able to build its own floor out of the "
        "withdrawal rule. Before that can be tested, the rules have to be "
        "compared on an honest footing: every comparison so far scores a "
        "retirement that ends on a known date, which hands the rules that "
        "amortise to one the answer. This section re-solves the rule, the "
        "rate and the allocation together against a real survival curve, "
        "and finds which rule a retiree should actually use. What that rule "
        "then does to the pension comparison is the next section."),
}


#: Paragraphs an inherited section carries that do not belong in a journal
#: submission. Matched on a distinctive phrase and dropped whole. The
#: companion study keeps them: a reader of a replication archive wants to
#: know how the code is tested, and a referee reading a forty-page paper
#: does not want to be told the test count.
DROPPED: Dict[str, Tuple[str, ...]] = {
    # Section 7's opening two paragraphs set up the value of leisure and the
    # parameter L. Both subsections that survive the trim below are about
    # the pension's features, so the setup is left explaining machinery the
    # section no longer uses.
    "leisure": ("the third leg of a retirement plan",
                "disutility of labour in utils",
                "A retirement-window objective cannot price a retirement",
                # 7.1's own opener restates the setup the reopening
                # paragraph now carries, and points at a comparison the
                # trim removed.
                "The comparison above changes two things at once"),
}

#: Subsections the trim leaves stranded at their original numbers. Section 7
#: keeps its fifth and sixth and loses the rest, and a section whose first
#: heading is 7.5 tells the reader four subsections are missing.
RENUMBERED: Dict[str, Dict[int, int]] = {"leisure": {5: 1, 6: 2},
                                         "baseline": {4: 3}}

#: Whole subsections an inherited section carries that this paper does not
#: need. Matched on a phrase in the subsection heading; everything from that
#: heading up to the next heading of the same rank or higher goes with it.
#: The companion study keeps them -- a replication archive is judged partly
#: on its audit trail -- but a reader here needs to know what the panel is,
#: not how each series was reconciled against its source.
TRIMMED: Dict[str, Tuple[str, ...]] = {
    "data": ("Auditing the data",),
    # Verification belongs to a replication archive, and the archive is the
    # companion study. A reader here needs to believe the simulator, not to
    # audit it.
    "methods": ("Implementation and verification",),
    # Both of these argue the companion's thesis rather than this one.
    # Stochastic dominance and the country-draw diagnostic establish that
    # the all-equity portfolio wins on the panel; this paper takes that as
    # given from Section #baseline.1 and asks what a pension does to it.
    "baseline": ("Distributional dominance", "How the countries are drawn"),
    # The inherited section spends five subsections pricing the retirement
    # date before it reaches the floor-versus-taper decomposition this paper
    # needs. That is a good study and a different paper; here it buries the
    # argument under nine pages, and its certainty equivalents are computed
    # over a window that is not comparable with the rest of this one. What
    # is kept is the 2x2 and the withdrawal-rule comparison.
    "leisure": ("The pension you claim early", "When to stop",
                "The break-even, which is the number to carry",
                "The same question under Australia's pension",
                "The rule decides which system wins",
                "What this changes"),
}

#: The anchors those trims remove, so a cross-reference into one resolves
#: to the companion study rather than to a heading this paper does not
#: print. Kept alongside :data:`TRIMMED` and checked against it, because the
#: two going out of step is the failure the build cannot see.
TRIMMED_ANCHORS: Tuple[str, ...] = (
    "data.6", "methods.5", "baseline.3", "baseline.5",
    "leisure.1", "leisure.2", "leisure.3", "leisure.4",
    "leisure.7", "leisure.8")


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


#: The shared label map, so both papers name a rule the
#: same way.
rule_label = ct.rule_label


def _join(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- prose, not a Python list."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _sign(value: float) -> int:
    """-1, 0 or +1, with a tolerance, so a gap of 1e-15 is not a sign."""
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def _gamma_walk(f: Any, rule: str,
                system: str = "australia_as_legislated") -> Tuple[str, str]:
    """The contested cell read at each risk aversion, as prose.

    Three sections quote it -- the abstract, the introduction and the
    conclusion -- and a sign that moves with the curvature of the objective
    is not a thing to state three ways. Returns ``("2, 5 and 10", "+21.91%
    at gamma = 2, ...")``, or two empty strings when the sweep was not run.
    """
    if not _has(f, "ordering_by_gamma"):
        return "", ""
    spread = f.table("ordering_by_gamma")
    walked = spread[(spread["system"] == system)
                    & (spread["rule"] == rule)].sort_values("gamma")
    if len(walked) < 2:
        return "", ""
    return (_join([f"{g:g}" for g in walked["gamma"]]),
            _join([f"{v:+.2f}% at \u03b3 = {g:g}"
                   for g, v in zip(walked["gamma"], walked["gap_pct"])]))


#: Small counts are spelled in this paper's prose, and a sentence that
#: sets a numeral beside a spelled number for the same kind of quantity
#: reads as a typographical accident rather than a choice.
_NUMBER_WORDS: Dict[int, str] = {
    0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
}


def _spelled(n: int) -> str:
    """``five`` for 5, ``24`` for 24 -- spelled to twenty, then digits.

    The panel's own size is sixteen and it turns up in sentences beside
    spelled counts ("eight of the sixteen"), so stopping at twelve put a
    numeral next to a word for the same kind of quantity.
    """
    return _NUMBER_WORDS.get(int(n), str(int(n)))


def _borrowing_price(ctx: Any, f: Any) -> List[Flowable]:
    """What a real borrowing spread does to the levered corner.

    A corner found at a zero spread is found at a price nobody is offered,
    so it establishes that the unlevered answer was the grid's and not that
    a retiree should borrow. The sweep prices the margin loan; reporting
    only the free end would be reporting the flattering half of a check
    this paper ran.
    """
    if not _has(f, "ceiling_optimum") or not _has(f, "ceiling_sweep"):
        return []
    optima = f.table("ceiling_optimum")
    if not len(optima) or "spread" not in optima:
        return []
    from src import ceiling as cl

    even = cl.break_even_spread(optima)
    spreads = sorted(float(x) for x in optima["spread"].unique())
    if len(spreads) < 2:
        return []
    top = optima[np.isclose(optima["spread"], spreads[-1])]
    unlevered = int((~top["levered"].astype(bool)).sum())
    by_spread = ", ".join(
        f"{100.0 * sp:.0f}% → {float(optima[np.isclose(optima['spread'], sp)]['leverage'].median()):.2f}×"
        for sp in spreads)

    if np.isfinite(even):
        closing = (
            f"The unlevered portfolio comes back at a spread of "
            f"{100.0 * float(even):.0f}%, which is inside the range a "
            f"retail margin loan is written at. So the levered corner is "
            f"real at a price no household is offered and gone at one they "
            f"are, and the practical reading of this subsection is the "
            f"bound rather than the level.")
    else:
        closing = (
            f"The sweep does not reach the price at which the last balance "
            f"stops borrowing: at {100.0 * spreads[-1]:.0f}% the median is "
            f"{float(top['leverage'].median()):.2f}× and "
            f"{unlevered} of {len(top)} balances want exactly the unlevered "
            f"portfolio, but one still does not. The break-even is "
            f"therefore above {100.0 * spreads[-1]:.0f}% and this paper "
            f"does not say where — which is the honest form of the "
            f"result, and enough for the claim being made: the corner "
            f"reported above is the grid's, and a household paying a real "
            f"price for leverage is close to the unlevered portfolio "
            f"rather than far above it.")

    return [ctx.p(
        f"<b>And at a price a household would actually pay, most of the "
        f"borrowing goes away.</b> The spread over the realised bill "
        f"return is swept as well as the leverage, and the median wanted "
        f"holding falls with it: {by_spread}. {closing}")]


#: Country codes as a reader would rather see them. Only the markets that
#: turn up in the sign split need naming; the rest of the paper refers to
#: the panel collectively.
COUNTRY_NAME: Dict[str, str] = {
    "AUS": "Australia", "BEL": "Belgium", "CHE": "Switzerland",
    "DEU": "Germany", "DNK": "Denmark", "ESP": "Spain", "FIN": "Finland",
    "FRA": "France", "GBR": "the United Kingdom", "ITA": "Italy",
    "JPN": "Japan", "NLD": "the Netherlands", "NOR": "Norway",
    "PRT": "Portugal", "SWE": "Sweden", "USA": "the United States",
}


def _best_percent_rate(f: Any, rule: str = "constant_percent") -> float:
    """The rate the longevity sweep picks for the percentage-of-balance rule.

    Quoted so the ordering grid can say plainly that it runs the rule at a
    different one, and why. A paper that sweeps a dial in one section and
    pins it in the next owes the reader that sentence.
    """
    if not _has(f, "longevity_sweep"):
        return float("nan")
    swept = f.table("longevity_sweep")
    hit = swept[swept["rule"] == rule]
    if not len(hit) or "rate" not in hit:
        return float("nan")
    return float(hit.loc[hit["cec_mortality"].idxmax(), "rate"])


def _matched(f: Any, rule: str) -> Dict[str, Any]:
    """The 2x2 of pension formula against contribution rate, at one rule."""
    if not _has(f, "ordering_gaps"):
        return {"measured": False}
    factorial = dict(f.cfg.get("ordering", {}).get("factorial", {}))
    if not factorial:
        return {"measured": False}
    from src import ordering as odr
    return odr.matched_contrast(f.table("ordering_gaps"), factorial, rule)


def _interval_for(f: Any, system: str, rule: str) -> Dict[str, Any]:
    """One cell's delete-one interval, or nothing measured."""
    if not _has(f, "ordering_intervals"):
        return {"measured": False}
    table = f.table("ordering_intervals")
    hit = table[(table["system"] == system) & (table["rule"] == rule)]
    if not len(hit):
        return {"measured": False}
    row = hit.iloc[0]
    return {"measured": True,
            "gap_pct": float(row["gap_pct"]),
            "se": float(row["standard_error"]),
            "ci": (float(row["ci_low"]), float(row["ci_high"])),
            "excludes_zero": bool(row["ci_excludes_zero"]),
            "sign_survives": bool(row["sign_survives_every_deletion"])}


def _matched_factorial(ctx: Any, f: Any, rule: str) -> List[Flowable]:
    """Which institution moves the ordering, holding the other still.

    Australia differs from the United States in the benefit formula *and*
    in the compulsory contribution, and a row headed "Australia as
    legislated" carries both. An earlier draft of this paper compared an
    American household saving 10% with an Australian one saving 20.2% and
    attributed the difference to the pension. The contribution rate is
    what carries a household past the assets test, which is the treatment
    rather than a detail of it, so the two have to be crossed -- which is
    what the pension section's own design does and what this table had
    stopped doing.
    """
    found = _matched(f, rule)
    if not found.get("measured"):
        return []
    cells, systems = found["cells"], found["systems"]
    er, mt = "earnings_related", "means_tested"
    vol, com = "voluntary", "compulsory"

    out: List[Flowable] = [
        ctx.h2("#ordering.1 Which institution the reversal belongs to")]
    out.append(ctx.p(
        f"Two of the columns above move two things at once. Australia's "
        f"pension is means-tested <i>and</i> its contribution is "
        f"compulsory, so a comparison between the first column and the "
        f"last changes the benefit formula and doubles the saving rate in "
        f"the same step. That is not a detail of the treatment: Section "
        f"{SHORT_ORDER.index('leisure') + 1}.1 shows the guarantee is what "
        f"carries this household past the assets test in the first place. "
        f"The five columns are a 2×2 with a control, and read as one "
        f"they say something the three-column version could not."))
    out.extend(ctx.table(
        [["All-equity lead over the target-date fund",
          "Saving 10%", "Saving 20.2%", "Contribution effect"],
         ["Earnings-related pension",
          f"{cells[f'{er}/{vol}']:+.2f}%", f"{cells[f'{er}/{com}']:+.2f}%",
          f"{found['contribution_effect_earnings_related']:+.2f} pp"],
         ["Means-tested pension",
          f"{cells[f'{mt}/{vol}']:+.2f}%", f"{cells[f'{mt}/{com}']:+.2f}%",
          f"{found['contribution_effect_means_tested']:+.2f} pp"],
         ["Pension effect",
          f"{found['pension_effect_voluntary']:+.2f} pp",
          f"{found['pension_effect_compulsory']:+.2f} pp",
          f"{found['interaction']:+.2f} pp"]],
        f"The benefit formula crossed with the contribution rate, under a "
        f"{rule_label(rule)} withdrawal. Each cell is the all-equity "
        f"portfolio's lead over the target-date fund; the margins are "
        f"differences along a row and down a column, and the corner is "
        f"their interaction.",
        anchor="ordering_factorial",
        note="Same simulated lifetimes in every cell. The bottom-right "
             "cell of the body is the row headed ‘Australia as "
             "legislated’ above; the top-left is ‘United States, "
             "10% saving’."))

    iv = _interval_for(f, systems[f"{mt}/{vol}"], rule)
    legislated = _interval_for(f, systems[f"{mt}/{com}"], rule)
    out.append(ctx.p(
        f"<b>Held apart, both institutions are large and they work against "
        f"each other.</b> Replacing an earnings-related pension with a "
        f"means-tested one, at the same contribution rate, moves the lead "
        f"{found['pension_effect_voluntary']:+.2f} points. Doubling the "
        f"contribution rate, under the same means-tested pension, moves it "
        f"{found['contribution_effect_means_tested']:+.2f} the other way. "
        f"The Australian system does both, and what a single "
        f"Australia-against-America row reports is the residual: "
        f"{found['confounded']:+.2f} points, which is smaller than either "
        f"piece it is made of. The interaction is "
        f"{found['interaction']:+.2f} points, and it has the sign the "
        f"paper's mechanism predicts — compulsory saving is worth more "
        f"under a means test than under an unconditional pension, because "
        f"a larger balance is a partial substitute for a floor that has "
        f"been withdrawn."))

    if iv.get("measured"):
        out.append(ctx.p(
            f"<b>And the reversal, held apart, is a sign this panel can "
            f"resolve.</b> At matched contributions the means-tested "
            f"household's all-equity lead is {iv['gap_pct']:+.2f}% with a "
            f"delete-one standard error of {iv['se']:.2f}, an interval of "
            f"[{iv['ci'][0]:+.2f}, {iv['ci'][1]:+.2f}]"
            + (f" that excludes zero, and the sign holds in every one of "
               f"the sixteen deletions."
               if iv["excludes_zero"] and iv["sign_survives"] else
               f", which still contains zero.")
            + (f" The cell this paper opened with \u2014 the system as "
               f"legislated, at {legislated['gap_pct']:+.2f}% with an "
               f"interval of [{legislated['ci'][0]:+.2f}, "
               f"{legislated['ci'][1]:+.2f}] \u2014 is still not "
               f"resolvable, and now there is a reason for that rather "
               f"than a shrug. It is what is left when an institutional "
               f"effect of {found['pension_effect_voluntary']:+.1f} points "
               f"and one of "
               f"{found['contribution_effect_means_tested']:+.1f} points "
               f"are allowed to offset inside a single row. The panel can "
               f"sign the cell those two are measured from; it cannot sign "
               f"the remainder."
               if legislated.get("measured")
               and not legislated["excludes_zero"] else "")))
        out.append(ctx.p(
            f"That is the form the paper's first finding should be stated "
            f"in. A means test reverses the portfolio ordering under a "
            f"fixed real withdrawal — measured against a matched "
            f"household, signed, and robust to dropping any one market. "
            f"Whether an <i>Australian</i> ends up on the wrong side of "
            f"the ordering depends on a second institution pushing back, "
            f"and on this panel the two nearly cancel. The first is a "
            f"statement about pension design and is what the rest of this "
            f"paper is about; the second is a statement about one country's "
            f"particular combination, and it is the one that cannot be "
            f"signed."))
    return out


def _split_found(f: Any, rule: str,
                 system: str = "australia_as_legislated") -> Dict[str, Any]:
    """:func:`src.ordering.sign_split` on this paper's contested cell."""
    if not (_has(f, "ordering_influence") and _has(f, "ordering_gaps")):
        return {"measured": False}
    from src import ordering as odr
    return odr.sign_split(f.table("ordering_influence"),
                          f.table("ordering_gaps"), rule, system)


def _both_reversals(ctx: Any, f: Any, rule: str) -> List[Flowable]:
    """The two negative cells given the same three statistics apiece.

    A draft of this section reported the jackknife diagnostics for the
    legislated cell alone, because on a three-system grid nothing else was
    negative. Holding the contribution rate still produced a second
    reversal, and it is the one the paper now leads with -- so applying
    the scepticism to the old cell and not the new one would be choosing
    where to be careful.

    The three statistics are not interchangeable and the section says
    which it trusts. An *interval* assumes the deletions scatter around
    the point estimate; a *bias diagnostic* says whether they do; a
    *count of deletions that keep the sign* assumes nothing at all. Where
    the second says the first is unreliable, the third is what a claim
    about a sign should rest on.
    """
    if not (_has(f, "ordering_intervals") and _has(f, "ordering_pseudo")):
        return []
    band, pseudo = f.table("ordering_intervals"), f.table("ordering_pseudo")
    matched = _matched(f, rule)
    order = []
    if matched.get("measured"):
        order.append((matched["systems"]["means_tested/voluntary"],
                      "means-tested at matched contributions"))
    order.append(("australia_as_legislated", "the system as legislated"))

    rows = [["Cell", "Lead (%)", "95% interval",
             "Deletions below the point", "Implied bias (pp)",
             "Deletions keeping the sign"]]
    seen: List[Dict[str, Any]] = []
    for system, name in order:
        iv = band[(band["system"] == system) & (band["rule"] == rule)]
        ps = pseudo[(pseudo["system"] == system) & (pseudo["rule"] == rule)]
        split = _split_found(f, rule, system)
        if not (len(iv) and len(ps) and split.get("measured")):
            continue
        a, b = iv.iloc[0], ps.iloc[0]
        seen.append({"name": name, "iv": a, "ps": b, "split": split})
        rows.append([
            name.capitalize(),
            f"{float(a['gap_pct']):+.2f}",
            f"[{float(a['ci_low']):+.2f}, {float(a['ci_high']):+.2f}]",
            f"{int(b['below_point'])} of {int(b['deletions'])}",
            f"{float(b['bias_estimate']):+.1f}",
            f"{int(split['sign_holds'])} of {int(split['deletions'])}"])
    if len(seen) < 2:
        return []

    out: List[Flowable] = list(ctx.table(
        rows,
        "The two cells in which the ordering reverses, on the three "
        "statistics the sixteen deletions support.",
        anchor="both_reversals",
        note="The interval assumes the deletions scatter around the point "
             "estimate. The implied bias, (n−1) times the gap between "
             "the mean deletion and the point, says whether they do. The "
             "count against zero assumes neither."))

    first, second = seen[0], seen[1]
    out.append(ctx.p(
        f"<b>The two reversals are not equally well established, and the "
        f"statistic that separates them is the last column.</b> At matched "
        f"contributions every one of the sixteen fifteen-country panels "
        f"puts the target-date fund ahead, and the least negative of them "
        f"is {_split_found(f, rule, order[0][0])['largest_flip_value']:+.2f}% "
        f"— clear of zero rather than close to it. As legislated the "
        f"sub-panels divide "
        f"{int(second['split']['sign_holds'])}–"
        f"{int(second['split']['sign_flips'])}, which is no evidence about "
        f"a sign at all."))

    out.append(ctx.p(
        f"<b>The middle two columns are the reason to lead with the last "
        f"one rather than with the interval.</b> A jackknife standard "
        f"error assumes the deletions sit around the point estimate, "
        f"roughly half above and half below. In neither of these cells do "
        f"they: {int(first['ps']['below_point'])} of "
        f"{int(first['ps']['deletions'])} fall below the matched estimate "
        f"and {int(second['ps']['below_point'])} of "
        f"{int(second['ps']['deletions'])} below the legislated one, "
        f"against seven of sixteen in almost every other cell of the "
        f"table. The implied biases are "
        f"{float(first['ps']['bias_estimate']):+.1f} and "
        f"{float(second['ps']['bias_estimate']):+.1f} points. So the "
        f"intervals we print for these two cells are the weakest of the "
        f"three statistics, in both, and we print them because withholding "
        f"an unflattering number is worse than qualifying it."))

    out.append(ctx.p(
        f"That leaves the two cells in genuinely different positions, and "
        f"the difference is not one of degree. The matched cell's claim "
        f"survives its own diagnostic, because the diagnostic bears on the "
        f"interval and the claim rests on the count — sixteen of "
        f"sixteen sub-panels, none of them near zero. The legislated "
        f"cell's claim has nothing left once the interval is discounted: "
        f"eight of sixteen, an implied bias "
        f"{abs(float(second['ps']['bias_estimate'])) / max(abs(float(first['ps']['bias_estimate'])), 1e-9):.1f} "
        f"times the matched cell's, and a mean deletion of "
        f"{float(second['ps']['loo_mean']):+.2f}% that does not reverse "
        f"the ordering at all. We report the first as a finding and the "
        f"second as a number we cannot sign."))

    rest = pseudo[~((pseudo["rule"] == rule)
                    & pseudo["system"].isin([o[0] for o in order]))]
    if len(rest):
        out.append(ctx.p(
            f"None of this is a property of jackknives. Across the "
            f"{len(rest)} cells in which the ordering does not reverse, "
            f"the deletions split "
            f"{int(rest['below_point'].min())}-to-"
            f"{int(rest['below_point'].max())} around their own point "
            f"estimates and no implied bias exceeds "
            f"{float(rest['bias_estimate'].abs().max()):.1f} points — "
            f"the same machinery, the same panel, the same sixteen "
            f"deletions, well behaved wherever the sign is not close to "
            f"zero. It is the two cells nearest a sign change that the "
            f"diagnostic fires on, which is where a jackknife is most "
            f"likely to be nonlinear and least likely to be trusted."))

    # And which countries carry the legislated cell, which is the more
    # useful thing sixteen deletions can say about a sign they cannot fix.
    out += _sign_split(ctx, f, rule)
    return out


def _sign_split(ctx: Any, f: Any, rule: str,
                system: str = "australia_as_legislated") -> List[Flowable]:
    """How many sub-panels reproduce the contested sign, and which do not.

    Two counts are easy to run together and a draft of this paper ran them
    together. ``below_point`` asks where the deletions sit relative to the
    *point estimate*, which is what a jackknife standard error is built on
    and what the bias diagnostic reads. Whether a deletion reproduces the
    *reversal* is a question about zero. In this cell the answers differ
    sharply -- four of sixteen deletions fall below the point estimate and
    eight of sixteen stay negative -- and the draft reported the first as
    though it were the second, concluding that essentially no
    fifteen-country subsample reproduces the reversal when half of them do.

    So the count against zero is measured here, and the movers are named.
    Which countries carry a sign is more useful to a reader than how wide
    an interval around it is, and it is the same arithmetic.
    """
    if not (_has(f, "ordering_influence") and _has(f, "ordering_gaps")):
        return []
    from src import ordering as odr

    found = odr.sign_split(f.table("ordering_influence"),
                           f.table("ordering_gaps"), rule, system)
    if not found.get("measured"):
        return []
    name = lambda c: COUNTRY_NAME.get(str(c), str(c))
    holds, flips = int(found["sign_holds"]), int(found["sign_flips"])
    n = int(found["deletions"])

    out = [ctx.p(
        f"<b>Which fifteen-country panels reverse, in the cell where that "
        f"is in doubt.</b> The question is only live for the system as "
        f"legislated — at matched contributions every sub-panel "
        f"reverses and there is nothing to enumerate. Of the "
        f"{_spelled(n)} fifteen-country panels built on the legislated "
        f"arm, "
        f"{_spelled(holds)} "
        f"still put the target-date fund ahead and {_spelled(flips)} do not: "
        f"the reversal is reproduced by half the subsamples of the panel it "
        f"was found in, not by none of them and not by all. The skew the "
        f"paragraph above reports is real and it is a statement about the "
        f"estimator, not a licence to say the reversal belongs to the whole "
        f"panel alone.")]

    out.append(ctx.p(
        f"Which countries do it is the more useful fact, and the deletions "
        f"give it directly. Removing {name(found['largest_flip'])} moves the "
        f"cell from {found['point']:+.2f}% to "
        f"{found['largest_flip_value']:+.2f}%, a swing of "
        f"{found['largest_flip_swing']:+.2f} points and more than any other "
        f"market; removing {name(found['largest_hold'])} moves it the other "
        f"way, to {found['largest_hold_value']:+.2f}%. The markets whose "
        f"removal overturns the reversal are "
        f"{_join([name(c) for c in found['flippers']])}. Two of the largest "
        f"movers among them are the panel's weakest equity markets over the "
        f"period, which is the mechanism rather than a coincidence: the "
        f"all-equity portfolio is hurt more by a century-long domestic "
        f"underperformance than a glide path is, so dropping one helps it. "
        f"The reversal is carried by the left tail of the country "
        f"cross-section, and a panel that happened to exclude Italy or "
        f"Portugal would not have reported it."))

    out.append(ctx.p(
        f"We take three things from that rather than one. The sign is not "
        f"resolved: {flips} of {n} sub-panels overturn it and the interval "
        f"contains zero. The estimate is not a typical member of its own "
        f"deletions either, which is what the bias diagnostic says and why "
        f"the interval is the weakest of the three statistics here. And the "
        f"reversal has an identifiable source in the cross-section rather "
        f"than being noise about a point — which is a fact a wider "
        f"panel could confirm or destroy, and the most useful thing sixteen "
        f"countries can say about it."))
    return out


def _corner_robustness(ctx: Any, f: Any) -> List[Flowable]:
    """The zero corner re-scored against the curvature and against a floor.

    Section 2 warns that a corner at zero is exactly the answer to
    distrust, because a constant-relative-risk-aversion objective with a
    consumption floor near zero is unbounded below: a handful of
    near-starvation years can move a certainty equivalent further than a
    decade of ordinary ones, and a portfolio chosen to avoid them is
    choosing against the aggregator rather than for the household. It then
    promises this check. The promise went unkept in a draft, which is the
    worse of the two ways to handle a caveat one has already thought of.
    """
    if not _has(f, "incidence_robustness"):
        return []
    table = f.table("incidence_robustness")
    if not len(table):
        return []
    bands = [c for c in ("below the free area", "inside the taper band",
                         "above the cut-off") if c in table.columns]
    if not bands:
        return []

    def label(row: str) -> str:
        if row == "cec":
            return "the baseline objective"
        if row.startswith("cec_gamma"):
            return f"\u03b3 = {row[len('cec_gamma'):]}"
        if row.startswith("cec_floor"):
            return f"a consumption floor of {float(row[len('cec_floor'):]):.2f}"
        return row

    out: List[Flowable] = [
        ctx.h2("#incidence.4 And the corner against the objective")]
    out.append(ctx.p(
        f"The deletions above vary the panel. They do not touch the second "
        f"objection, which Section {SHORT_ORDER.index('model') + 1}.2 raises "
        f"against this result rather than waiting for a reader to: a corner "
        f"at zero equity is the kind of answer a "
        f"constant-relative-risk-aversion objective produces when the "
        f"objective, not the household, cannot tolerate the left tail. "
        f"Scoring an outcome again costs nothing beside producing it, so the "
        f"same balance sweep is re-read under "
        f"{_spelled(len(table))} objectives: the baseline, two other risk "
        f"aversions, and two consumption floors below which the aggregator "
        f"is not allowed to look."))
    out.extend(ctx.table(
        [["Scored under"] + [b.capitalize() for b in bands]]
        + [[label(str(r["scoring"]))]
           + [f"{float(r[b]):.0%}" for b in bands]
           for _, r in table.iterrows()],
        "The equity share a means-tested retiree wants under a fixed real "
        "withdrawal, by position against the assets test, under five "
        "objectives.",
        anchor="incidence_robustness",
        note="A floor replaces consumption below it before the felicity "
             "function is applied, which is what removes the unboundedness "
             "the corner would otherwise be exploiting."))

    values = table[bands].to_numpy(dtype=float)
    unchanged = bool((values == values[0, 0]).all())
    if unchanged:
        out.append(ctx.p(
            f"<b>The corner is not the aggregator's.</b> Every one of the "
            f"{values.size} cells is {values[0, 0]:.0%}: the answer does not "
            f"move at half the baseline risk aversion, at twice it, or with "
            f"the aggregator forbidden to look below a floor. Whatever else "
            f"is true of a certainty equivalent at this curvature, the "
            f"finding here is not made of it. The mechanism in the paragraph "
            f"above — that a fixed real rule converts a good return "
            f"into assessable assets and never into consumption — is "
            f"arithmetic on the budget line and does not consult a felicity "
            f"function, which is what one would expect to survive this and "
            f"is what does."))
    else:
        out.append(ctx.p(
            f"<b>The corner does depend on the objective.</b> Across the "
            f"{len(table)} scorings the wanted share runs "
            f"{values.min():.0%} to {values.max():.0%}, so the result "
            f"reported above is conditional on the aggregator as well as on "
            f"the panel, and Section {SHORT_ORDER.index('model') + 1}.2's "
            f"caution was warranted. We report the span rather than the "
            f"baseline alone."))
    return out


def _recommended(f: Any, system: str = "australia_as_legislated"
                 ) -> Dict[str, Any]:
    """What the rule Section #longevity selects does to the ordering grid.

    The grid's best cell and the rule this paper recommends are not the
    same cell, and the difference is the point: a section exists to choose
    a rule, so the headline should quote the choice and give the maximum
    beside it rather than lead with the maximum and leave the choice
    unmentioned.
    """
    if not (_has(f, "longevity_ranking") and _has(f, "ordering_gaps")):
        return {"measured": False}
    from src import ordering as odr

    return odr.recommended_rule(f.table("longevity_ranking"),
                                f.table("ordering_gaps"), system)


def _has(f: Any, name: str) -> bool:
    """Whether a results table exists, so an optional block can be skipped."""
    try:
        f.table(name)
    except FileNotFoundError:
        return False
    return True


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
    ct.RENUMBERED = RENUMBERED
    with renumbered():
        parts: List[Flowable] = []
        parts += front(ctx)
        parts += introduction(ctx)
        parts += model(ctx)
        for key in ("data", "methods", "baseline", "pension", "leisure"):
            parts += _reopen(getattr(ct, f"section_{key}")(ctx), ctx, key)
        parts += incidence(ctx)
        parts += _reopen(ct.section_longevity(ctx), ctx, "longevity")
        parts += ordering(ctx)
        parts += limitations(ctx)
        parts += conclusion(ctx)
        parts += references(ctx)
    ct.COMPANION = {}
    ct.ABSENT = ()
    ct.RENUMBERED = {}
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
        out_path, running_head=TITLE, doc_title=f"{TITLE}: {SUBTITLE}",
        doc_author="Lifecycle investing study",
        doc_subject="Public pension design and lifecycle asset allocation")
    doc.styles = styles
    parts = story(ctx)
    # The trims above remove floats whose numbers the context has already
    # issued, so the story is renumbered before it is laid out and the built
    # document is read back to confirm nothing was missed.
    _bp.renumber_floats(parts, ctx.float_anchors)
    doc.multiBuild(parts)
    problems = (_bp.check_cross_references(out_path)
                + _bp.check_float_numbering(out_path))
    if problems:
        raise SystemExit("the short paper does not hang together:\n  "
                         + "\n  ".join(problems))
    return out_path


if __name__ == "__main__":
    print("wrote", build())
