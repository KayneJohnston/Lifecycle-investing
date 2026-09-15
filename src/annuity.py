"""The instrument that supplies the floor, put in the choice set.

The paper's mechanism is that a means test reverses the portfolio ordering
by removing an *unconditional floor* rather than by taxing wealth. Section
#longevity then shows a household can build a floor of its own out of the
withdrawal rule. Both of those leave the obvious question standing: the
market sells floors. A life annuity is the instrument, and until this
section it was not in the choice set -- the paper conceded its absence in
the limitations and left a reader to guess what would have happened.

Guessing is not available here, because two forces pull in opposite
directions and neither is small.

**For the annuity.** Under a means test the household's problem is that its
income in the bad states is its own portfolio. An annuity replaces part of
that portfolio with a payment that arrives whether the market cooperated or
not, which is exactly the floor the test withdrew. And where the scheme
exempts annuitised wealth from the assets test, buying one *also* raises the
pension, so the means test subsidises the very instrument that undoes it.
That is the crowding-in side of the mechanism Bütler, Peijnenburg and
Staubli (2017) find running the other way.

**Against it.** An annuity is bought at a price. The insurer prices a real
lifetime income on a survival curve and adds a load, so a household that
annuitises hands over more than the actuarial value of what it gets back,
and it gives up the estate -- which this paper's objective prices, because
an amortisation rule spends the portfolio to zero and a fixed real rule does
not. Where the scheme *assesses* annuitised wealth, the shelter disappears
and only the floor is left.

Which side wins is a measurement, so this module measures it.

**Pricing.** The premium for one unit of real income a year for life is the
expected present value of that stream on the same Gompertz curve
:mod:`src.mortality` uses, discounted at a real rate, divided by a money's
worth ratio. A money's worth below one is the load: the fraction of the
actuarial value the buyer gets back. Real-world money's worth on real
annuities in developed markets runs roughly 0.75 to 0.95, and the ratio is
swept rather than asserted.

**The corner is not an answer.** A household that annuitises its whole
balance holds no retirement portfolio, so the comparison this paper is
about -- which of two funds a retiree should hold -- has nothing left to
compare. What a gap at a full share measures is which *accumulation*
strategy bought the larger income, both arms having liquidated at the
pension age; both also register as ruined on the portfolio measure, because
there is no portfolio. That cell is worth reporting and is not worth
reading as the reversal being cured, so every verdict below is taken over
the shares at which a portfolio still exists and the corner is labelled
separately.

**The horizon trap, which is why this section is scored twice.** Every
outcome in this project is also aggregated over a fixed thirty-year
retirement. An annuity priced on a survival curve whose life expectancy is
under nineteen years, and then *paid* for thirty years with certainty, is
being handed the insurer's margin and the mortality credit both. Scored that
way an annuity would look wonderful for a reason that has nothing to do with
the means test. So every cell here carries the survival-weighted certainty
equivalent Section #longevity argues for as its headline, and the
fixed-horizon one beside it to show the size of the distortion.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: How the assets test sees annuitised wealth, as the share of the annuity's
#: remaining actuarial value that counts as an assessable asset. Real
#: schedules sit between the two: Australia assesses a lifetime income
#: stream on a declining purchase price with a partial exemption, and
#: income-tests the payments. This model carries no income test -- the
#: limitations say so -- so the exempt arm flatters the annuity by exactly
#: the test it omits, and both arms are reported.
TREATMENTS: Tuple[Tuple[str, float], ...] = (
    ("exempt from the assets test", 0.0),
    ("assessed at its remaining value", 1.0),
)


def price(survive: np.ndarray, real_rate: float,
          moneys_worth: float = 1.0) -> float:
    """Premium for one unit of real income a year for life.

    ``survive[h]`` is the probability of being alive at the start of
    simulated year ``h``, indexed from the start of *working* life the way
    :func:`src.mortality.survival` returns it; the caller slices it to the
    retirement years before passing it in. The payment is made at the start
    of each year the annuitant is alive, so year zero is undiscounted and
    certain.

    A money's worth of one is actuarially fair. Below one the buyer gets
    back that fraction of the actuarial value, so the premium is larger:
    ``price = EPV / moneys_worth``.
    """
    alive = np.asarray(survive, dtype=float)
    if alive.ndim != 1 or not len(alive):
        raise ValueError("survive must be a non-empty one-dimensional array")
    if not 0.0 < float(moneys_worth) <= 1.0:
        raise ValueError("moneys_worth must lie in (0, 1]")
    if float(real_rate) <= -1.0:
        raise ValueError("real_rate must exceed -100%")
    discount = (1.0 + float(real_rate)) ** -np.arange(len(alive), dtype=float)
    return float((alive * discount).sum() / float(moneys_worth))


def assessable_path(survive: np.ndarray, real_rate: float,
                    income: float = 1.0) -> np.ndarray:
    """The annuity's remaining actuarial value at the start of each year.

    An assets test that looks at an income stream has to put a number on it,
    and the defensible number is what the remaining payments are worth:
    the expected present value of the income still to come, conditional on
    the annuitant being alive to collect it. It starts near the premium net
    of the load and declines to one year's income, which is the shape a real
    schedule's declining purchase-price rule approximates.
    """
    alive = np.asarray(survive, dtype=float)
    if alive.ndim != 1 or not len(alive):
        raise ValueError("survive must be a non-empty one-dimensional array")
    rate = float(real_rate)
    out = np.empty(len(alive), dtype=float)
    for h in range(len(alive)):
        rest = alive[h:] / max(alive[h], 1e-12)
        discount = (1.0 + rate) ** -np.arange(len(rest), dtype=float)
        out[h] = float(income) * float((rest * discount).sum())
    return out


def layer(survive: np.ndarray, n_retired: int, real_rate: float,
          moneys_worth: float, fraction: float) -> Dict[str, Any]:
    """Everything the simulator needs to carry one annuity purchase.

    Returns the premium as a share of the balance at retirement, the real
    income that share buys per unit of balance, and the assessable value of
    the remaining payments in each retirement year per unit of balance. The
    quantities are per unit so the simulator can scale them by each path's
    own wealth without re-pricing.
    """
    retired = np.asarray(survive, dtype=float)[-int(n_retired):] \
        if len(survive) > int(n_retired) else np.asarray(survive, dtype=float)
    # Conditional on reaching the pension age, which is what a retiree buying
    # at that age is quoted.
    retired = retired / max(float(retired[0]), 1e-12)
    unit = price(retired, real_rate, moneys_worth)
    share = float(np.clip(fraction, 0.0, 1.0))
    income = share / unit if unit > 0.0 else 0.0
    return {"premium_share": share,
            "unit_price": unit,
            "income_per_wealth": income,
            "assessable_per_wealth": assessable_path(retired, real_rate,
                                                     income),
            "life_expectancy_years": float(retired.sum())}


def sweep(simulate: Callable[[str, Any, str, float, float], Any],
          systems: Sequence[str], rules: Sequence[Tuple[str, Any]],
          strategies: Sequence[str], fractions: Sequence[float],
          treatments: Sequence[Tuple[str, float]],
          score: Callable[[Any], Dict[str, Any]],
          log_every: int = 25) -> pd.DataFrame:
    """The ordering grid, with the annuitised share as a fifth dimension.

    ``simulate(system, rule, strategy, fraction, assessed)`` returns one
    outcome. A fraction of zero is the paper's existing cell, so the sweep
    contains its own control and the comparison is a difference inside one
    table rather than against a number quoted from another section.
    """
    rows: List[Dict[str, Any]] = []
    total = (len(systems) * len(rules) * len(strategies) * len(fractions)
             * len(treatments))
    n = 0
    for system in systems:
        for rule_key, rule in rules:
            for treat_label, assessed in treatments:
                for fraction in fractions:
                    for strategy in strategies:
                        row: Dict[str, Any] = {
                            "system": str(system),
                            "rule": str(rule_key),
                            "strategy": str(strategy),
                            "fraction": float(fraction),
                            "treatment": str(treat_label),
                            "assessed": float(assessed)}
                        row.update(score(simulate(system, rule, strategy,
                                                  float(fraction),
                                                  float(assessed))))
                        rows.append(row)
                        n += 1
                        if log_every and n % int(log_every) == 0:
                            LOGGER.info("  scored %d of %d", n, total)
    return pd.DataFrame.from_records(rows)


def gaps(frame: pd.DataFrame, pair: Tuple[str, str],
         column: str = "cec_survival") -> pd.DataFrame:
    """The challenger's lead over the incumbent, in each cell of the sweep.

    Keyed on everything but the strategy, so a row is one portfolio
    comparison at one annuitised share.
    """
    challenger, incumbent = pair
    keys = ["system", "rule", "treatment", "assessed", "fraction"]
    wide = frame.pivot_table(index=keys, columns="strategy", values=column)
    if challenger not in wide.columns or incumbent not in wide.columns:
        return pd.DataFrame()
    out = wide.reset_index()
    out["gap_pct"] = 100.0 * (out[challenger] / out[incumbent] - 1.0)
    out["winner"] = np.where(out["gap_pct"] > 0.0, challenger, incumbent)
    return out.sort_values(keys).reset_index(drop=True)


def wanted(frame: pd.DataFrame, strategy: str,
           column: str = "cec_survival") -> pd.DataFrame:
    """The annuitised share each cell would choose, and what it is worth.

    One row per (system, rule, treatment) for the named portfolio: the share
    that maximises the objective, the share the paper's own cells assume
    (zero), and the gain between them.
    """
    block = frame[frame["strategy"] == strategy]
    rows: List[Dict[str, Any]] = []
    keys = ["system", "rule", "treatment", "assessed"]
    for key, part in block.groupby(keys, sort=False):
        part = part.sort_values("fraction")
        best = part.loc[part[column].idxmax()]
        none = part[part["fraction"] <= 0.0]
        base = float(none[column].iloc[0]) if len(none) else float("nan")
        rows.append({
            **dict(zip(keys, key)),
            "strategy": strategy,
            "best_fraction": float(best["fraction"]),
            "best_value": float(best[column]),
            "no_annuity_value": base,
            "gain_pct": (100.0 * (float(best[column]) / base - 1.0)
                         if base and base == base else float("nan")),
            "at_the_corner": bool(
                float(best["fraction"]) >= float(part["fraction"].max())),
        })
    return pd.DataFrame.from_records(rows)


def reversal(gapped: pd.DataFrame, system: str, rule: str,
             treatment: str) -> Dict[str, Any]:
    """Whether the portfolio ordering still reverses once a floor is buyable.

    The paper's finding is a sign in one cell at a zero annuitised share.
    The question is whether that sign survives when the household may buy
    the floor the means test withdrew.

    Read over the *interior* of the grid rather than over all of it. At a
    share of one there is no portfolio: both arms liquidated at the pension
    age, both hold nothing, and the number is a comparison of what the two
    accumulation strategies bought rather than of what a retiree should
    hold. A sign that changes only there has not been cured, and this
    reports the two separately so the prose cannot run them together.
    """
    block = gapped[(gapped["system"] == system) & (gapped["rule"] == rule)
                   & (gapped["treatment"] == treatment)].sort_values("fraction")
    if not len(block):
        return {"measured": False}
    none = block[block["fraction"] <= 0.0]
    if not len(none):
        return {"measured": False}
    base = float(none["gap_pct"].iloc[0])
    sign = np.sign(round(base, 6))
    flipped = block[np.sign(block["gap_pct"].round(6)) != sign]
    inside = block[block["fraction"] < 1.0]
    flipped_inside = inside[np.sign(inside["gap_pct"].round(6)) != sign]
    corner = block[block["fraction"] >= 1.0]
    deepest = inside.loc[inside["gap_pct"].idxmin()] if len(inside) else None
    return {
        "measured": True,
        "system": str(system),
        "rule": str(rule),
        "treatment": str(treatment),
        "no_annuity_gap_pct": base,
        "reverses_without_annuity": bool(base < 0.0),
        "gap_at_full_pct": float(block["gap_pct"].iloc[-1]),
        "full_fraction": float(block["fraction"].iloc[-1]),
        "sign_survives_every_share": bool(not len(flipped)),
        "first_share_that_flips_it": (float(flipped["fraction"].min())
                                      if len(flipped) else float("nan")),
        # The two that the prose is allowed to lean on.
        "sign_survives_every_interior_share": bool(not len(flipped_inside)),
        "first_interior_share_that_flips_it": (
            float(flipped_inside["fraction"].min())
            if len(flipped_inside) else float("nan")),
        "flip_needs_the_corner": bool(len(flipped) and not
                                      len(flipped_inside)),
        "corner_gap_pct": (float(corner["gap_pct"].iloc[0]) if len(corner)
                           else float("nan")),
        # How much portfolio is left where the sign turns. A share of 0.90
        # leaves a tenth of the balance to allocate, which is a thin
        # version of the question the paper asks rather than an answer to
        # it, and the prose has to be able to say so with a number.
        "portfolio_left_at_the_flip": (
            1.0 - float(flipped["fraction"].min()) if len(flipped)
            else float("nan")),
        "widest_interior_share": (float(inside["fraction"].max())
                                  if len(inside) else float("nan")),
        "deepest_interior_gap_pct": (float(deepest["gap_pct"])
                                     if deepest is not None else float("nan")),
        "deepest_at_share": (float(deepest["fraction"])
                             if deepest is not None else float("nan")),
        "annuity_deepens_it": bool(
            deepest is not None and float(deepest["gap_pct"]) < base),
        "widest_move_pp": float(block["gap_pct"].max()
                                - block["gap_pct"].min()),
    }


def verdict(gapped: pd.DataFrame, wanted_frame: pd.DataFrame,
            headline_system: str, legislated_system: str,
            baseline_rule: str, strategy: str) -> Dict[str, Any]:
    """What the annuity does to the paper's finding, classified from the grid.

    Three questions, and the prose has to report whichever way each comes
    out: does the household want an annuity at all, does it want more of one
    under a means test than under an earnings-related pension, and does the
    portfolio reversal survive the purchase.
    """
    if not len(gapped) or not len(wanted_frame):
        return {"measured": False}
    out: Dict[str, Any] = {"measured": True,
                           "treatments": [], "strategy": str(strategy),
                           "rule": str(baseline_rule)}
    for label in list(dict.fromkeys(gapped["treatment"])):
        rev = reversal(gapped, headline_system, baseline_rule, label)
        want = wanted_frame[(wanted_frame["treatment"] == label)
                            & (wanted_frame["rule"] == baseline_rule)]
        by_system = want.set_index("system")["best_fraction"].to_dict()
        mt = by_system.get(headline_system, float("nan"))
        er = by_system.get("us_social_security", float("nan"))
        gain = want.set_index("system")["gain_pct"].to_dict()
        out["treatments"].append({
            "treatment": str(label),
            **{k: v for k, v in rev.items() if k != "measured"},
            "wanted_under_means_test": float(mt),
            "wanted_under_earnings_related": float(er),
            "means_test_wants_more": bool(mt == mt and er == er and mt > er),
            "gain_under_means_test_pct": float(
                gain.get(headline_system, float("nan"))),
            "gain_under_earnings_related_pct": float(
                gain.get("us_social_security", float("nan"))),
            "wanted_at_the_corner": bool(want["at_the_corner"].any()),
        })
    # Judged on the interior, because a sign that only turns where the
    # portfolio has been liquidated has not been turned by anything this
    # paper is about.
    survived = [t["sign_survives_every_interior_share"]
                for t in out["treatments"]
                if t.get("sign_survives_every_interior_share") is not None]
    out["reversal_survives_under_every_treatment"] = bool(survived
                                                          and all(survived))
    out["reversal_survives_under_no_treatment"] = bool(survived
                                                       and not any(survived))
    out["every_flip_needs_the_corner"] = bool(
        out["treatments"]
        and all(t.get("flip_needs_the_corner") for t in out["treatments"]))
    out["annuity_deepens_it_everywhere"] = bool(
        out["treatments"]
        and all(t.get("annuity_deepens_it") for t in out["treatments"]))
    # The wanted share against the top of the grid. A model with no health
    # shock, no liquidity need and no bequest motive beyond a fixed weight
    # reproduces Yaari's full-annuitisation corner, and a section that
    # reported the corner as advice would be reporting the omission.
    corners = [t.get("wanted_at_the_corner") for t in out["treatments"]]
    out["wanted_share_is_a_corner"] = bool(corners and any(corners))
    shares = [t.get("wanted_under_means_test") for t in out["treatments"]]
    out["lowest_wanted_share"] = (float(min(s for s in shares if s == s))
                                  if any(s == s for s in shares)
                                  else float("nan"))
    out["legislated_system"] = str(legislated_system)
    return out


def horizon_distortion(frame: pd.DataFrame,
                       fixed: str = "cec",
                       lived: str = "cec_survival") -> Dict[str, Any]:
    """How much the fixed horizon overpays an annuity.

    An annuity priced on a survival curve and then paid for thirty years
    with certainty collects the mortality credit twice. This measures the
    size of that, so the choice to score the section on the survival-weighted
    objective is a reported number rather than an assertion.
    """
    if fixed not in frame.columns or lived not in frame.columns:
        return {"measured": False}
    block = frame.dropna(subset=[fixed, lived])
    if not len(block):
        return {"measured": False}
    none = block[block["fraction"] <= 0.0]
    full = block[block["fraction"] >= float(block["fraction"].max())]

    def _lift(part: pd.DataFrame, column: str) -> float:
        return float(part[column].mean()) if len(part) else float("nan")

    fixed_gain = 100.0 * (_lift(full, fixed) / _lift(none, fixed) - 1.0)
    lived_gain = 100.0 * (_lift(full, lived) / _lift(none, lived) - 1.0)
    return {"measured": True,
            "full_fraction": float(block["fraction"].max()),
            "gain_fixed_horizon_pct": fixed_gain,
            "gain_real_lifespan_pct": lived_gain,
            "overpaid_pp": fixed_gain - lived_gain,
            "fixed_horizon_flatters_it": bool(fixed_gain > lived_gain)}


def by_load(swept: pd.DataFrame, pair: Tuple[str, str], system: str,
            rule: str) -> pd.DataFrame:
    """The same cell re-read at every price the annuity was offered at.

    The obvious way for this section to be wrong is a pricing artefact: an
    annuity cheap enough rescues any portfolio, and one dear enough rescues
    none. So the load is swept and the crossing share is read at each
    setting, including an actuarially fair one that no insurer offers and
    that is here as the bound rather than as a calibration.
    """
    rows: List[Dict[str, Any]] = []
    for worth in sorted({float(x) for x in swept.get("moneys_worth", [])}):
        gapped = gaps(swept[swept["moneys_worth"] == worth], pair)
        if not len(gapped):
            continue
        for label in list(dict.fromkeys(gapped["treatment"])):
            found = reversal(gapped, system, rule, label)
            if not found.get("measured"):
                continue
            rows.append({
                "moneys_worth": worth,
                "treatment": label,
                "no_annuity_gap_pct": found["no_annuity_gap_pct"],
                "deepest_gap_pct": found["deepest_interior_gap_pct"],
                "deepest_at_share": found["deepest_at_share"],
                "flips_at_share": found["first_interior_share_that_flips_it"],
                "deepens": found["annuity_deepens_it"],
            })
    return pd.DataFrame.from_records(rows)


def load_verdict(frame: pd.DataFrame) -> Dict[str, Any]:
    """Whether the price changes the answer, classified from the sweep."""
    if not len(frame):
        return {"measured": False}
    flips = frame["flips_at_share"].dropna()
    fair = frame[frame["moneys_worth"] >= 1.0]
    return {
        "measured": True,
        "settings": int(len(frame)),
        "loads": int(frame["moneys_worth"].nunique()),
        "deepens_at_every_price": bool(frame["deepens"].all()),
        "one_crossing_share": bool(flips.nunique() <= 1),
        "earliest_crossing": (float(flips.min()) if len(flips)
                              else float("nan")),
        "latest_crossing": (float(flips.max()) if len(flips)
                            else float("nan")),
        "holds_at_an_actuarially_fair_price": bool(
            len(fair) and fair["deepens"].all()),
        "cheapest_price": float(frame["moneys_worth"].max()),
        "dearest_price": float(frame["moneys_worth"].min()),
    }


def channel(swept: pd.DataFrame, pair: Tuple[str, str], rule: str,
            treatment: str, ceiling: float = 0.75) -> pd.DataFrame:
    """How far the mean and the tail move as the household annuitises.

    The gap this section reports is a ratio of certainty equivalents, and a
    certainty equivalent bundles a mean with a tail. Splitting them says
    which one the annuity is acting on, and comparing the split across
    pension regimes says whether it is acting through the instrument or
    through the test.

    Read over the interior only -- up to ``ceiling`` -- because past it the
    household holds too little portfolio for the comparison to be one
    between portfolios. A draft of this section asserted a mechanism in
    prose that this measurement does not support; the numbers are here so
    the prose has to be read off them.
    """
    challenger, incumbent = pair
    block = swept[(swept["rule"] == rule)
                  & (swept["treatment"] == treatment)
                  & (swept["fraction"] <= float(ceiling))]
    rows: List[Dict[str, Any]] = []
    for system in list(dict.fromkeys(block["system"])):
        part = block[block["system"] == system]
        row: Dict[str, Any] = {"system": str(system)}
        for column, name in (("mean_consumption", "mean"),
                             ("p5_consumption", "tail")):
            wide = part.pivot_table(index="fraction", columns="strategy",
                                    values=column)
            if challenger not in wide.columns or incumbent not in wide.columns:
                continue
            ratio = (wide[challenger] / wide[incumbent]).sort_index()
            row[f"{name}_ratio_at_zero"] = float(ratio.iloc[0])
            row[f"{name}_ratio_lowest"] = float(ratio.min())
            row[f"{name}_ratio_span"] = float(ratio.max() - ratio.min())
            row[f"{name}_moves_against_the_challenger"] = bool(
                ratio.min() < ratio.iloc[0])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def channel_verdict(frame: pd.DataFrame, tested: str,
                    untested: str) -> Dict[str, Any]:
    """Whether the annuity acts through the instrument or through the test.

    If the annuitised share moved the portfolio comparison by itself, it
    would move it under an earnings-related pension too. Whether it does is
    the discriminating measurement, and it is the one the prose is allowed
    to lean on.
    """
    if not len(frame) or "mean_ratio_span" not in frame:
        return {"measured": False}
    keyed = frame.set_index("system")
    if tested not in keyed.index or untested not in keyed.index:
        return {"measured": False}
    hit, control = keyed.loc[tested], keyed.loc[untested]
    span_tested = float(hit["mean_ratio_span"])
    span_control = float(control["mean_ratio_span"])
    return {
        "measured": True,
        "tested_system": str(tested),
        "untested_system": str(untested),
        "mean_span_under_the_test": span_tested,
        "mean_span_without_it": span_control,
        "tail_span_under_the_test": float(hit["tail_ratio_span"]),
        "tail_span_without_it": float(control["tail_ratio_span"]),
        "ratio_of_spans": (span_tested / span_control
                           if span_control > 0 else float("inf")),
        "mean_moves_against_it_under_the_test": bool(
            hit["mean_moves_against_the_challenger"]),
        "tail_moves_against_it_under_the_test": bool(
            hit["tail_moves_against_the_challenger"]),
        # The discriminating fact: an instrument effect would show up under
        # both pensions, and an interaction shows up under one.
        "it_is_the_test_and_not_the_instrument": bool(
            span_tested > 3.0 * span_control),
    }
