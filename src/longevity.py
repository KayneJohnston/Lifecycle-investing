"""The withdrawal rule when the horizon is not known.

Every comparison of spending rules in this project -- Section #spending's
sweep, Section #plan's joint optimisation -- scores a retirement that ends at
ninety-three with certainty.  That is not neutral between rules, and the
direction of the distortion is not subtle:

* A rule that **amortises to a fixed date** knows the date. Dividing the
  balance by the years remaining is very nearly optimal when the years
  remaining are a known constant, so a fixed horizon hands it the answer.
* A rule that **hedges longevity** -- spending a fixed share of the balance,
  or dividing by an actuarial expectancy -- is paying a premium against a
  risk the model has switched off, and looks needlessly cautious.
* And **the probability of ruin** is measured against a certainty of living
  to ninety-three, so it counts as failure a portfolio that runs out at
  ninety-one for somebody who, in a life table, most likely died at
  eighty-six.

Section #mortality already re-weights the aggregation by a Gompertz survival
curve, and shows it does not change which *allocation* wins.  It says in
terms that the re-weighting is only approximate for the horizon-based rules,
"which would themselves change if they knew the mortality table".  This
section is that unfinished sentence: it re-scores the *rule* and the *rate*
as well as the allocation, and solves the three together.

Three decisions, one objective
------------------------------
The grid is the cross product of

``allocation``
    an equity share and, within it, a domestic share, held constant through
    retirement;
``rule``
    the withdrawal policies of Section #spending, including the two that
    already read a mortality table -- so "should the rule know how long you
    are likely to live?" is a comparison inside the grid rather than an
    assumption behind it;
``rate``
    the level the rule spends at, where it has one to set.

Each combination is scored twice on the same simulated lifetimes: once over
the fixed horizon this project has used throughout, and once survival-
weighted.  Nothing is re-simulated between the two, so the difference is the
aggregation and only the aggregation.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Combination", "describe", "front_load", "correlation_strength", "allocation_grid", "plan_grid", "sweep", "optimum",
    "by_objective", "ranking_shift", "ablation", "verdict",
    "FIXED", "MORTALITY",
]

#: Column names for the two aggregations, used everywhere rather than typed
#: out, because a study whose whole point is the difference between two
#: objectives cannot afford to mix them up.
FIXED: str = "cec_fixed"
MORTALITY: str = "cec_mortality"

#: Share of the joint gain the interaction may take and still let the three
#: decisions be reported one at a time.
SEPARABLE_SHARE: float = 0.25

#: Rank-correlation bands for how much of a reordering one explanation
#: accounts for. A moderate correlation is a contributing cause and must not
#: be written up as *the* cause, which is the mistake these exist to stop.
STRONG_CORR: float = 0.70
MODERATE_CORR: float = 0.35


def correlation_strength(value: float) -> str:
    """``strong``, ``moderate``, ``weak`` or ``none``, for prose to branch on."""
    magnitude = abs(float(value))
    if not np.isfinite(magnitude) or magnitude < 0.15:
        return "none"
    if magnitude >= STRONG_CORR:
        return "strong"
    if magnitude >= MODERATE_CORR:
        return "moderate"
    return "weak"


@dataclasses.dataclass(frozen=True)
class Combination:
    """One (allocation, rule, rate) triple."""

    equity: float
    domestic: float
    rule: str
    rate: float | None = None
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    suffix: str = ""

    @property
    def rule_label(self) -> str:
        """The rule and its variant, without the rate."""
        return f"{self.rule} ({self.suffix})" if self.suffix else self.rule

    def label(self) -> str:
        rate = "" if self.rate is None else f" @ {self.rate:.1%}"
        return (f"{self.rule_label}{rate}, "
                f"{self.equity:.0%} equity / {self.domestic:.0%} domestic")

    @property
    def assumed_return(self) -> float | None:
        """The real return this rule amortises at, if it takes one."""
        from . import spending as spg

        if self.rule not in spg.RETURN_PARAMETERISED:
            return None
        value = self.params.get(spg.RETURN_PARAMETER)
        return None if value is None else float(value)

    def build(self) -> Any:
        """The :class:`src.spending.SpendingRule` this combination names."""
        from . import spending as spg

        params = dict(self.params)
        if self.rate is not None:
            params["rate"] = float(self.rate)
        return spg.build(self.rule, **params)


def allocation_grid(equity: Sequence[float], domestic: Sequence[float],
                    ) -> List[Tuple[float, float]]:
    """Every ``(equity, domestic)`` pair on the grid."""
    return [(float(e), float(d)) for e in equity for d in domestic]


def plan_grid(rule_specs: Sequence[Mapping[str, Any]],
              rates: Sequence[float],
              returns: Sequence[float] = (),
              ) -> List[Tuple[str, float | None, Mapping[str, Any], str]]:
    """``(rule, rate, params, suffix)`` for every policy worth scoring.

    Three families, and each is swept on whatever dial actually sets its
    spending level. A rate-parameterised rule is crossed with ``rates``. A
    rule dialled by an assumed real return -- the annuity factor is a
    function of it, so it front-loads exactly as a higher rate does -- is
    crossed with ``returns``, *overriding* any value the spec pinned: a
    level chosen by hand in the config is not a swept dimension, and an
    optimum found among three hand-picked values is not an optimum. Only a
    rule with no dial at all appears once.

    Passing no ``returns`` leaves the pinned variants exactly as configured,
    so callers that want the configured menu rather than a sweep keep it.
    """
    from . import spending as spg

    out: List[Tuple[str, float | None, Mapping[str, Any], str]] = []
    seen: set = set()

    def _add(key: str, rate: float | None, params: Mapping[str, Any],
             suffix: str) -> None:
        # Expanding a dial collapses several configured variants onto the
        # same grid, so the same policy can be produced more than once.
        mark = (key, rate, tuple(sorted(params.items())), suffix)
        if mark not in seen:
            seen.add(mark)
            out.append((key, rate, dict(params), suffix))

    for spec in rule_specs:
        key = str(spec["key"])
        params = dict(spec.get("params", {}) or {})
        suffix = str(spec.get("suffix", "") or "")
        if key in spg.RATE_PARAMETERISED:
            for rate in rates:
                _add(key, float(rate), params, suffix)
        elif key in spg.RETURN_PARAMETERISED and len(returns):
            for value in returns:
                _add(key, None,
                     {**params, spg.RETURN_PARAMETER: float(value)},
                     return_suffix(float(value)))
        else:
            _add(key, None, params, suffix)
    return out


def return_suffix(value: float) -> str:
    """`4% assumed return`, matching the hand-written variants it replaces.

    ``:g`` rather than ``:.0%`` so that a half-point step on the grid reads
    as 2.5% instead of collapsing onto its neighbour.
    """
    return f"{100.0 * float(value):g}% assumed return"


#: Rules whose divisor comes from a survival model rather than a fixed
#: planning horizon. Whether being in this set is what wins under an
#: uncertain lifespan is the question, so it is a column rather than an
#: assumption.
MORTALITY_AWARE: frozenset = frozenset({"gompertz"})


def front_load(combo: "Combination", wealth: float = 1.0) -> float:
    """The first retirement year's withdrawal, as a share of the balance.

    One number that puts every rule on a comparable footing regardless of
    how it arrives at its level -- a rate it was told, a fixed horizon it
    amortises over, or a survival curve it reads. It is the obvious
    candidate for *why* an uncertain lifespan reorders the rules, so it is
    carried and correlated rather than argued about.
    """
    from . import lifecycle as lc

    spec = lc.LifecycleSpec()
    rule = combo.build()
    draw = rule.initial_withdrawal(np.array([float(wealth)]),
                                   spec.n_retired, spec.age_retire)
    return float(np.asarray(draw, dtype=float).ravel()[0]) / float(wealth)


def sweep(simulate: Callable[[Combination], Any],
          score_fixed: Callable[[Any], float],
          score_mortality: Callable[[Any], float],
          ruin_mortality: Callable[[Any], float],
          combinations: Sequence[Combination],
          log_every: int = 200) -> pd.DataFrame:
    """Every combination, scored under both aggregations.

    The two scores come off the *same* simulated outcome, which is the point:
    a difference between them cannot be sampling noise, because there is no
    second sample.
    """
    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(combinations):
        outcome = simulate(combo)
        rows.append({
            "equity": combo.equity, "domestic": combo.domestic,
            "rule": combo.rule, "rule_label": combo.rule_label,
            "rate": np.nan if combo.rate is None else float(combo.rate),
            "has_rate": combo.rate is not None,
            "assumed_return": (np.nan if combo.assumed_return is None
                               else float(combo.assumed_return)),
            "has_assumed_return": combo.assumed_return is not None,
            "label": combo.label(),
            "front_load": float(front_load(combo)),
            "reads_mortality": bool(combo.rule in MORTALITY_AWARE),
            FIXED: float(score_fixed(outcome)),
            MORTALITY: float(score_mortality(outcome)),
            "ruin_fixed": float(np.mean(outcome.ruin)),
            "ruin_mortality": float(ruin_mortality(outcome)),
            "mean_consumption": float(outcome.consumption.mean()),
        })
        if log_every and (i + 1) % int(log_every) == 0:
            LOGGER.info("  scored %d of %d combinations", i + 1,
                        len(combinations))
    return pd.DataFrame.from_records(rows)


def optimum(frame: pd.DataFrame, column: str = MORTALITY) -> pd.Series:
    """The single best combination under one objective."""
    if not len(frame):
        raise ValueError("no combinations to choose between")
    return frame.loc[frame[column].idxmax()]


def by_objective(frame: pd.DataFrame) -> pd.DataFrame:
    """The winner under each aggregation, side by side.

    Reported as one row per objective rather than as a difference, because
    the interesting case is a *different combination* winning, and a
    difference of certainty equivalents would hide that.
    """
    rows: List[Dict[str, Any]] = []
    for name, column in (("fixed horizon", FIXED),
                         ("survival-weighted", MORTALITY)):
        best = optimum(frame, column)
        rows.append({
            "objective": name,
            "rule": best["rule_label"],
            "rate": best["rate"],
            "equity": best["equity"],
            "domestic": best["domestic"],
            "cec_fixed": best[FIXED],
            "cec_mortality": best[MORTALITY],
            "ruin_fixed": best["ruin_fixed"],
            "ruin_mortality": best["ruin_mortality"],
        })
    return pd.DataFrame.from_records(rows)


def ranking_shift(frame: pd.DataFrame, column: str = MORTALITY,
                  reference: str = FIXED) -> pd.DataFrame:
    """Rules ranked under each objective, at each rule's own best settings.

    Each rule is given its best rate and allocation under the objective being
    ranked, so a rule is not penalised for a setting chosen to suit a
    different horizon.
    """
    rows: List[Dict[str, Any]] = []
    for label, block in frame.groupby("rule_label"):
        rows.append({
            "rule_label": str(label),
            reference: float(block[reference].max()),
            column: float(block[column].max()),
        })
    out = pd.DataFrame.from_records(rows)
    out["rank_fixed"] = out[reference].rank(ascending=False).astype(int)
    out["rank_mortality"] = out[column].rank(ascending=False).astype(int)
    out["rank_change"] = out["rank_fixed"] - out["rank_mortality"]
    return out.sort_values("rank_mortality").reset_index(drop=True)


def _restrict(frame: pd.DataFrame, base: pd.Series,
              free: Sequence[str]) -> pd.DataFrame:
    """Rows matching ``base`` on every dimension except those in ``free``."""
    held = [d for d in ("allocation", "rule", "rate") if d not in free]
    mask = pd.Series(True, index=frame.index)
    for dimension in held:
        if dimension == "allocation":
            mask &= (np.isclose(frame["equity"], float(base["equity"]))
                     & np.isclose(frame["domestic"], float(base["domestic"])))
        elif dimension == "rule":
            mask &= frame["rule_label"] == base["rule_label"]
        else:
            # The rate is a property *of a rule*, not an independent dial: a
            # policy that divides by a planning horizon has none to hold. So
            # holding the rate means holding it wherever the candidate has
            # one, and never excluding a rule that sets none. Without this
            # the "free the rule" row could only ever reach other
            # rate-parameterised rules, and the ablation could not show the
            # very substitution this section is about.
            rate = base["rate"]
            if not pd.isna(rate):
                mask &= (frame["rate"].isna()
                         | np.isclose(frame["rate"].fillna(-1.0),
                                      float(rate)))
    return frame[mask]


def ablation(frame: pd.DataFrame) -> pd.DataFrame:
    """What re-choosing each decision for a real horizon is worth.

    The baseline is the combination a fixed horizon would have picked, scored
    under survival weighting -- which is what an investor who took this
    project's earlier advice and then went on living would actually get. Each
    row then frees one decision, and the last frees all three. The gap
    between the three singles and the joint is the interaction, and a large
    one would mean none of the singles can be read on its own.
    """
    if not len(frame):
        return pd.DataFrame(columns=["freed", "cec", "gain_pct"])
    base = optimum(frame, FIXED)
    start = float(base[MORTALITY])
    rows: List[Dict[str, Any]] = [{
        "freed": "nothing (the fixed-horizon choice)",
        "rule": base["rule_label"], "rate": base["rate"],
        "equity": base["equity"], "domestic": base["domestic"],
        "cec": start}]
    for name, free in (("allocation", ("allocation",)),
                       ("rule", ("rule",)),
                       ("rate", ("rate",)),
                       ("all three", ("allocation", "rule", "rate"))):
        block = _restrict(frame, base, free)
        if not len(block):
            continue
        best = optimum(block, MORTALITY)
        rows.append({
            "freed": name, "rule": best["rule_label"], "rate": best["rate"],
            "equity": best["equity"], "domestic": best["domestic"],
            "cec": float(best[MORTALITY])})
    out = pd.DataFrame.from_records(rows)
    out["gain_pct"] = 100.0 * (out["cec"] / start - 1.0)
    return out


def describe(rule: str, rate: float) -> str:
    """`constant real at 4.0%`, or `gompertz, which sets no rate`.

    A rule that derives its level from a planning horizon has no rate, and
    printing `nan%` for it -- which an earlier version did -- reads as a bug
    rather than as the fact it is.
    """
    if rate is None or not np.isfinite(float(rate)):
        return f"{rule}, which sets no rate of its own"
    return f"{rule} at {float(rate):.1%}"


def rate_preference(frame: pd.DataFrame,
                    can_deplete: Mapping[str, bool]) -> pd.DataFrame:
    """The rate each rate-setting rule wants, and whether it can run out.

    One row per rule, scored at that rule's best allocation, so no rule is
    held to a mix chosen to suit another.  Rules with no rate of their own
    are absent by construction rather than carried as ``NaN``.  Two rates
    that score identically resolve to the lower one, which is both the
    conservative reading and what the figure's peak marker picks.
    """
    if not len(frame) or "rule_label" not in frame:
        return pd.DataFrame(
            columns=["rule_label", "family", "rate", "cec", "can_deplete"])
    rated = frame[frame["has_rate"]] if "has_rate" in frame else frame
    rows = []
    for rule, block in rated.groupby("rule_label"):
        curve = block.groupby("rate")[MORTALITY].max()
        if not len(curve):
            continue
        # The label carries the variant in brackets; the deplete flag is a
        # property of the rule family, so the bracket is stripped first.
        family = str(rule).split(" (")[0]
        rows.append({"rule_label": str(rule), "family": family,
                     "rate": float(curve.idxmax()),
                     "cec": float(curve.max()),
                     "can_deplete": bool(can_deplete.get(family, True))})
    out = pd.DataFrame.from_records(rows)
    return out.sort_values("rate", ascending=False).reset_index(drop=True) \
        if len(out) else out


def rate_split_verdict(preference: pd.DataFrame) -> Dict[str, Any]:
    """Does "can run out" separate the rates, or only correlate with them?

    The tempting story is that the rules which cannot deplete want a high
    rate and the rules which can want a low one.  It is worth checking
    rather than telling, because a rule can sit on the wrong side of it: a
    lightly smoothed endowment rule is nearly a percentage of balance and
    behaves like one, while still being able to run the portfolio to zero.
    ``separates`` is true only when every non-depleting rule wants strictly
    more than every depleting one.
    """
    if not len(preference):
        return {"measured": False}
    top = preference.iloc[0]
    bottom = preference.iloc[-1]
    safe = preference[~preference["can_deplete"]]
    risky = preference[preference["can_deplete"]]
    separates = bool(len(safe) and len(risky)
                     and safe["rate"].min() > risky["rate"].max())
    found: Dict[str, Any] = {
        "measured": True,
        "rules": int(len(preference)),
        "top_rule": str(top["rule_label"]), "top_rate": float(top["rate"]),
        "bottom_rule": str(bottom["rule_label"]),
        "bottom_rate": float(bottom["rate"]),
        "spread_pp": 100.0 * (float(top["rate"]) - float(bottom["rate"])),
        "top_can_deplete": bool(top["can_deplete"]),
        "separates": separates,
    }
    # When the split fails, name the depleting rule that reaches highest --
    # that rule is the counter-example, and the paper should print it.
    if not separates and len(risky) and len(safe):
        highest = risky.iloc[0]
        found["crossing_rule"] = str(highest["rule_label"])
        found["crossing_rate"] = float(highest["rate"])
        found["crossing_ties_top"] = bool(
            abs(float(highest["rate"]) - float(top["rate"])) < 1e-9)
    return found


def verdict(frame: pd.DataFrame, shift: pd.DataFrame,
            ablated: pd.DataFrame) -> Dict[str, Any]:
    """What an uncertain lifespan changes, classified rather than assumed."""
    if not len(frame):
        return {"measured": False}
    fixed_best = optimum(frame, FIXED)
    mortality_best = optimum(frame, MORTALITY)
    found: Dict[str, Any] = {
        "measured": True,
        "combinations": int(len(frame)),
        "fixed_rule": str(fixed_best["rule_label"]),
        "mortality_rule": str(mortality_best["rule_label"]),
        "fixed_rate": float(fixed_best["rate"]),
        "mortality_rate": float(mortality_best["rate"]),
        "fixed_equity": float(fixed_best["equity"]),
        "mortality_equity": float(mortality_best["equity"]),
        "fixed_domestic": float(fixed_best["domestic"]),
        "mortality_domestic": float(mortality_best["domestic"]),
    }
    found["rule_changes"] = bool(
        found["fixed_rule"] != found["mortality_rule"])
    found["allocation_changes"] = bool(
        not np.isclose(found["fixed_equity"], found["mortality_equity"])
        or not np.isclose(found["fixed_domestic"],
                          found["mortality_domestic"]))
    rate_pair = (found["fixed_rate"], found["mortality_rate"])
    if all(np.isfinite(rate_pair)):
        found["rate_changes"] = bool(not np.isclose(*rate_pair))
        found["rate_rises"] = bool(rate_pair[1] > rate_pair[0])
    else:
        # A horizon-based rule has no rate to set. That is itself the
        # finding when one wins, so it is recorded rather than left as a
        # NaN for the prose to print.
        found["winner_sets_no_rate"] = bool(
            not np.isfinite(found["mortality_rate"]))
    # An optimum on the boundary of its own grid is a truncation, not an
    # optimum. `src.accumulation.at_grid_edge` says so, and every dial that
    # sets a spending level has to be put through it -- the withdrawal rate
    # and the assumed real return alike. A dial left off this check is a
    # corner waiting to be reported as a peak.
    from .accumulation import at_grid_edge

    rates = sorted(frame.loc[frame["has_rate"], "rate"].dropna().unique())
    if rates:
        found["rate_grid_low"] = float(min(rates))
        found["rate_grid_high"] = float(max(rates))
        for label, value in (("fixed", found["fixed_rate"]),
                             ("mortality", found["mortality_rate"])):
            if np.isfinite(value):
                found[f"{label}_rate_at_edge"] = bool(
                    at_grid_edge(rates, float(value)))
        # The winner may set no rate at all, in which case the corner that
        # matters is the best *rate-setting* rule's, since that is what the
        # rate panel plots.
        rated = frame[frame["has_rate"]]
        if len(rated):
            best_rated = rated.loc[rated[MORTALITY].idxmax()]
            found["best_rated_rule"] = str(best_rated["rule_label"])
            found["best_rated_rate"] = float(best_rated["rate"])
            found["best_rated_at_edge"] = bool(
                at_grid_edge(rates, float(best_rated["rate"])))
            found["rate_optimum_interior"] = bool(
                not found["best_rated_at_edge"])

    # The same check on the other dial. An amortisation rule is levelled by
    # the return it assumes, so a grid of three hand-picked values can hand
    # back its own top end and call it the winner.
    if "has_assumed_return" in frame:
        dialled = frame[frame["has_assumed_return"]]
        returns = sorted(dialled["assumed_return"].dropna().unique())
        if len(returns) and len(dialled):
            best_dialled = dialled.loc[dialled[MORTALITY].idxmax()]
            found["return_grid_low"] = float(min(returns))
            found["return_grid_high"] = float(max(returns))
            found["best_return_rule"] = str(best_dialled["rule_label"])
            found["best_return"] = float(best_dialled["assumed_return"])
            found["best_return_at_edge"] = bool(
                at_grid_edge(returns, float(best_dialled["assumed_return"])))
            found["return_optimum_interior"] = bool(
                not found["best_return_at_edge"])
            # Whether the overall winner is the rule sitting on that edge is
            # the question the section's headline depends on.
            found["winner_is_return_dialled"] = bool(
                str(mortality_best["rule_label"])
                == str(best_dialled["rule_label"]))

    found["anything_changes"] = bool(
        found["rule_changes"] or found["allocation_changes"]
        or found.get("rate_changes", False))

    # Ruin is the headline number every retirement study quotes, and a fixed
    # horizon inflates it by counting failures that happen after most people
    # have died. Measured across the rules that *can* run out: the winning
    # combination often cannot, and a ratio of zero to zero would say
    # nothing about the distortion.
    found["ruin_fixed_at_optimum"] = float(mortality_best["ruin_fixed"])
    found["ruin_mortality_at_optimum"] = float(
        mortality_best["ruin_mortality"])
    found["optimum_can_deplete"] = bool(
        float(mortality_best["ruin_fixed"]) > 0.0)
    depleting = frame[frame["ruin_fixed"] > 0.0]
    if len(depleting):
        ratios = (depleting["ruin_fixed"]
                  / depleting["ruin_mortality"].replace(0.0, np.nan))
        found["depleting_combinations"] = int(len(depleting))
        found["median_ruin_ratio"] = float(ratios.median())
        best_depleting = depleting.loc[depleting[MORTALITY].idxmax()]
        found["best_depleting_rule"] = str(best_depleting["rule_label"])
        found["best_depleting_ruin_fixed"] = float(
            best_depleting["ruin_fixed"])
        found["best_depleting_ruin_mortality"] = float(
            best_depleting["ruin_mortality"])
        if float(best_depleting["ruin_mortality"]) > 0.0:
            found["best_depleting_ratio"] = float(
                best_depleting["ruin_fixed"]
                / best_depleting["ruin_mortality"])

    if len(shift):
        moved = shift[shift["rank_change"] != 0]
        found["rules_ranked"] = int(len(shift))
        found["rules_that_move"] = int(len(moved))
        if len(moved):
            biggest = moved.loc[moved["rank_change"].abs().idxmax()]
            found["biggest_mover"] = str(biggest["rule_label"])
            found["biggest_move_places"] = int(biggest["rank_change"])

    # Why the order changes. Two candidate explanations, and the data is
    # asked which: that a rule wins by *reading a mortality table*, or that
    # it wins by *spending faster*. The second is measurable on any rule,
    # including the ones that have never heard of a survival curve.
    if len(shift) > 2 and "front_load" in frame.columns:
        best = frame.loc[frame.groupby("rule_label")[MORTALITY].idxmax()]
        ranked = best.set_index("rule_label")
        joined = shift.set_index("rule_label").join(
            ranked[["front_load", "reads_mortality"]])
        joined = joined.dropna(subset=["front_load"])
        if len(joined) > 2:
            found["front_load_corr"] = float(
                joined["front_load"].corr(-joined["rank_mortality"],
                                          method="spearman"))
            found["front_load_strength"] = correlation_strength(
                found["front_load_corr"])
            found["front_load_corr_fixed"] = float(
                joined["front_load"].corr(-joined["rank_fixed"],
                                          method="spearman"))
            aware = joined[joined["reads_mortality"].astype(bool)]
            blind = joined[~joined["reads_mortality"].astype(bool)]
            if len(aware) and len(blind):
                found["mortality_aware_best_rank"] = int(
                    aware["rank_mortality"].min())
                found["blind_best_rank"] = int(blind["rank_mortality"].min())
                found["a_blind_rule_wins"] = bool(
                    found["blind_best_rank"] < found["mortality_aware_best_rank"])

    if len(ablated) > 1:
        singles = ablated[ablated["freed"].isin(
            ("allocation", "rule", "rate"))]
        joint = ablated[ablated["freed"] == "all three"]
        if len(joint):
            total = float(joint["gain_pct"].iloc[0])
            found["joint_gain_pct"] = total
            found["single_gains_pct"] = {
                str(r["freed"]): float(r["gain_pct"])
                for _, r in singles.iterrows()}
            summed = float(singles["gain_pct"].sum())
            found["interaction_pct"] = total - summed
            found["separable"] = bool(
                abs(total - summed) <= SEPARABLE_SHARE * abs(total)
                if total else True)
            if len(singles):
                top = singles.loc[singles["gain_pct"].idxmax()]
                found["dominant_decision"] = str(top["freed"])
                found["dominant_gain_pct"] = float(top["gain_pct"])
    return found
