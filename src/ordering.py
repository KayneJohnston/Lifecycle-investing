"""Does the withdrawal rule change which *portfolio* wins, or only which system?

This module exists because of an equivocation the paper was caught making.

The headline is a statement about portfolios: the all-equity portfolio leads
the target-date fund under an earnings-related pension and trails it under
an asset-tested one. A later section then reports that under an amortisation
withdrawal rule Australia overtakes the United States, and the paper glossed
that as "the ordering reverses again".

It is not the same ordering. One compares two *strategies* inside a system;
the other compares two *systems*. Australia beating America on the level of
retirement consumption says nothing about whether an Australian retiree
should hold equities, and the paper had no table that did.

So this sweep asks the question directly: for every pension system, under
every withdrawal rule, which portfolio wins and by how much. The answer is
whatever it is -- if the all-equity portfolio does not recover its lead when
the rule supplies the floor, the claim has to be withdrawn rather than
reworded, and :func:`verdict` is written so the prose has to say so.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: The pair the paper's headline is about, challenger first.
HEADLINE: Tuple[str, str] = ("balanced_all_equity", "target_date_fund")

#: How large a strategy gap has to be before it is called a lead rather than
#: a tie. Ten basis points of certainty-equivalent consumption: below that
#: the bootstrap cannot tell the two portfolios apart and "wins" would be
#: reporting noise as a ranking.
TIE_BAND: float = 0.001


def sweep(simulate: Callable[[str, Any, str], Any],
          systems: Sequence[str], rules: Sequence[Tuple[str, Any]],
          strategies: Sequence[str],
          score: Callable[[Any], Dict[str, Any]],
          log_every: int = 10) -> pd.DataFrame:
    """Every system, crossed with every rule, crossed with every portfolio.

    ``simulate(system, rule, strategy)`` returns one outcome. The three
    dimensions are swept together rather than one at a time because the
    whole question is whether they interact.
    """
    rows: List[Dict[str, Any]] = []
    total = len(systems) * len(rules) * len(strategies)
    n = 0
    for system in systems:
        for rule_key, rule in rules:
            for strategy in strategies:
                row: Dict[str, Any] = {"system": str(system),
                                       "rule": str(rule_key),
                                       "strategy": str(strategy)}
                row.update(score(simulate(system, rule, strategy)))
                rows.append(row)
                n += 1
                if log_every and n % int(log_every) == 0:
                    LOGGER.info("  scored %d of %d", n, total)
    return pd.DataFrame.from_records(rows)


def gaps(frame: pd.DataFrame, pair: Tuple[str, str] = HEADLINE,
         column: str = "cec", tie: float = TIE_BAND) -> pd.DataFrame:
    """The challenger's lead over the incumbent, per system and rule.

    Reported as a percentage of the incumbent's certainty equivalent, which
    is the same convention the rest of the project uses for a strategy gap
    -- so these numbers are comparable with the headline and the reader is
    not asked to hold two definitions at once.
    """
    if not len(frame):
        return frame
    challenger, incumbent = pair
    wide = frame.pivot_table(index=["system", "rule"], columns="strategy",
                             values=column)
    missing = [s for s in pair if s not in wide.columns]
    if missing:
        raise ValueError(f"the sweep carries no strategy {missing!r}; "
                         f"it has {sorted(wide.columns)}")
    out = wide.reset_index()[["system", "rule"]].copy()
    out["challenger"] = wide[challenger].to_numpy(dtype=float)
    out["incumbent"] = wide[incumbent].to_numpy(dtype=float)
    out["gap_pct"] = 100.0 * (out["challenger"] / out["incumbent"] - 1.0)
    out["leader"] = np.where(
        out["gap_pct"] > 100.0 * tie, challenger,
        np.where(out["gap_pct"] < -100.0 * tie, incumbent, "tie"))
    # The best of the whole menu, not just the pair: a rule under which some
    # third portfolio wins is a different finding from one under which the
    # pair swaps, and the prose should be able to tell them apart.
    best = frame.loc[frame.groupby(["system", "rule"])[column].idxmax()]
    out = out.merge(
        best[["system", "rule", "strategy", column]].rename(
            columns={"strategy": "best_strategy", column: "best_cec"}),
        on=["system", "rule"], how="left")
    # Sorted by the order the rules were swept in, not alphabetically:
    # "amortisation at 10%" sorts between 0% and 2% as a string, and a table
    # whose rows read 0, 10, 2, 4, 6, 8 invites the reader to misread the
    # trend that is the point of the column.
    order = list(dict.fromkeys(frame["rule"]))
    out["rule"] = pd.Categorical(out["rule"], categories=order, ordered=True)
    out = out.sort_values(["system", "rule"]).reset_index(drop=True)
    out["rule"] = out["rule"].astype(str)
    return out


def verdict(frame: pd.DataFrame, baseline_rule: str,
            baseline_system: str = "us",
            contender_system: str = "au_as_legislated",
            pair: Tuple[str, str] = HEADLINE) -> Dict[str, Any]:
    """Whether the withdrawal rule can restore the all-equity ordering.

    The field the paper turns on is ``recovers``: under the rule the paper
    spends by, the challenger trails in the contender system; does any rule
    in the menu give it the lead back? If none does, the paper's second
    headline is a claim about the level of consumption in two countries and
    must be written as one.
    """
    if not len(frame):
        return {"measured": False}
    challenger, incumbent = pair
    found: Dict[str, Any] = {"measured": True, "rules": int(
        frame["rule"].nunique()), "systems": int(frame["system"].nunique()),
        "baseline_system": str(baseline_system),
        "contender_system": str(contender_system)}

    def _row(system: str, rule: str) -> Any:
        hit = frame[(frame["system"] == system) & (frame["rule"] == rule)]
        return hit.iloc[0] if len(hit) else None

    base = _row(baseline_system, baseline_rule)
    contend = _row(contender_system, baseline_rule)
    if base is None or contend is None:
        return {"measured": False}
    found["baseline_rule"] = str(baseline_rule)
    found["baseline_gap_pct"] = float(base["gap_pct"])
    found["contender_gap_pct"] = float(contend["gap_pct"])
    found["baseline_leader"] = str(base["leader"])
    found["contender_leader"] = str(contend["leader"])
    # The paper's first finding, re-derived here so the two cannot drift.
    found["pension_reverses_the_ordering"] = bool(
        base["leader"] == challenger and contend["leader"] != challenger)

    block = frame[frame["system"] == contender_system]
    found["contender_gaps"] = {str(r["rule"]): float(r["gap_pct"])
                               for _, r in block.iterrows()}
    winners = block[block["leader"] == challenger]
    found["recovers"] = bool(len(winners))
    if len(winners):
        best = winners.loc[winners["gap_pct"].idxmax()]
        found["recovering_rule"] = str(best["rule"])
        found["recovering_gap_pct"] = float(best["gap_pct"])
        found["recovering_rules"] = sorted(str(r) for r in winners["rule"])
    # Even where the pair does not swap, the rule may move the gap a long
    # way, and a reader deciding a default cares about that.
    found["contender_gap_range_pp"] = float(
        block["gap_pct"].max() - block["gap_pct"].min())
    found["contender_best_rule"] = str(
        block.loc[block["gap_pct"].idxmax(), "rule"])
    found["contender_worst_rule"] = str(
        block.loc[block["gap_pct"].idxmin(), "rule"])
    # Which portfolio the contender actually wants, rule by rule. When it is
    # neither member of the pair, saying only "the target-date fund wins"
    # would be false.
    found["contender_best_strategies"] = {
        str(r["rule"]): str(r["best_strategy"]) for _, r in block.iterrows()}
    found["a_third_portfolio_ever_wins"] = bool(
        any(s not in pair for s in found["contender_best_strategies"].values()))
    signs = {np.sign(round(float(g), 6))
             for g in found["contender_gaps"].values()}
    found["sign_depends_on_the_rule"] = bool(len(signs - {0.0}) > 1)
    return found


def by_system(frame: pd.DataFrame) -> pd.DataFrame:
    """The gap table pivoted for printing: rules down, systems across."""
    if not len(frame):
        return frame
    return frame.pivot(index="rule", columns="system",
                       values="gap_pct").reset_index()


def influence(gaps_for: Callable[[Sequence[str]], pd.DataFrame],
              countries: Sequence[str]) -> pd.DataFrame:
    """The whole gap table, recomputed once per country removed.

    ``gaps_for(kept)`` returns a gap table built on a panel holding only
    ``kept``. Sixteen developed markets are not sixteen independent draws --
    equity returns co-move and the twentieth century happened to all of them
    at once -- so the delete-one jackknife is the sampling error the *panel*
    carries. It is the number a sign should be weighed against, and Monte
    Carlo error is not: a hundred thousand paths drive that close to zero
    without adding a single country of evidence.
    """
    frames: List[pd.DataFrame] = []
    for dropped in countries:
        kept = [c for c in countries if c != dropped]
        LOGGER.info("leave-one-out: dropping %s (%d markets left)",
                    dropped, len(kept))
        block = gaps_for(kept).copy()
        block.insert(0, "dropped", str(dropped))
        block["n_markets"] = len(kept)
        frames.append(block)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def intervals(influence_frame: pd.DataFrame, gapped: pd.DataFrame,
              column: str = "gap_pct") -> pd.DataFrame:
    """A delete-one jackknife interval for every system-and-rule cell.

    The sign of a cell is what this paper's claims are made of, so the field
    that matters is ``sign_survives_every_deletion``: whether the ordering
    the point estimate reports is the ordering every one of the sixteen
    sub-panels reports. A cell whose confidence interval straddles zero is a
    cell that cannot carry a claim about which portfolio wins.
    """
    from .panel_robustness import jackknife

    if not len(influence_frame) or not len(gapped):
        return pd.DataFrame()
    base = {(str(r["system"]), str(r["rule"])): float(r[column])
            for _, r in gapped.iterrows()}
    rows: List[Dict[str, Any]] = []
    for (system, rule), block in influence_frame.groupby(["system", "rule"]):
        point = base.get((str(system), str(rule)), float("nan"))
        found = jackknife(block.rename(columns={column: "gap_pct"}),
                          baseline_gap=point)
        values = block[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append({
            "system": str(system), "rule": str(rule),
            "gap_pct": point,
            "standard_error": float(found.get("standard_error", np.nan)),
            "ci_low": float(found.get("ci_low", np.nan)),
            "ci_high": float(found.get("ci_high", np.nan)),
            "loo_low": float(values.min()) if values.size else np.nan,
            "loo_high": float(values.max()) if values.size else np.nan,
            "deletions": int(values.size),
            "sign_survives_every_deletion": bool(
                values.size and np.all(np.sign(values) == np.sign(point))),
            "ci_excludes_zero": bool(
                np.isfinite(found.get("ci_low", np.nan))
                and found["ci_low"] * found["ci_high"] > 0.0),
        })
    out = pd.DataFrame.from_records(rows)
    order = list(dict.fromkeys(gapped["rule"]))
    out["rule"] = pd.Categorical(out["rule"], categories=order, ordered=True)
    out = out.sort_values(["system", "rule"]).reset_index(drop=True)
    out["rule"] = out["rule"].astype(str)
    return out


def precision_verdict(table: pd.DataFrame, baseline_rule: str,
                      baseline_system: str = "us_social_security",
                      contender_system: str = "australia_as_legislated",
                      ) -> Dict[str, Any]:
    """Whether the reversal is a sign the panel can actually resolve.

    The paper's headline is a change of sign in one cell. If that cell's
    interval straddles zero, the headline is a point estimate wearing a
    claim's clothing, and this says so.
    """
    if not len(table):
        return {"measured": False}
    hit = table[(table["system"] == contender_system)
                & (table["rule"] == baseline_rule)]
    base = table[(table["system"] == baseline_system)
                 & (table["rule"] == baseline_rule)]
    if not len(hit) or not len(base):
        return {"measured": False}
    row, ref = hit.iloc[0], base.iloc[0]
    found: Dict[str, Any] = {
        "measured": True,
        "cells": int(len(table)),
        "reversal_gap_pct": float(row["gap_pct"]),
        "reversal_se": float(row["standard_error"]),
        "reversal_ci": (float(row["ci_low"]), float(row["ci_high"])),
        "reversal_loo": (float(row["loo_low"]), float(row["loo_high"])),
        "reversal_sign_survives": bool(row["sign_survives_every_deletion"]),
        "reversal_resolved": bool(row["ci_excludes_zero"]),
        "baseline_gap_pct": float(ref["gap_pct"]),
        "baseline_se": float(ref["standard_error"]),
        "baseline_resolved": bool(ref["ci_excludes_zero"]),
    }
    # Which cells the panel can and cannot resolve at all. A table where
    # only the contested one is unresolved says something different from a
    # table where half of them are.
    found["resolved_cells"] = int(table["ci_excludes_zero"].sum())
    unresolved = table[~table["ci_excludes_zero"]]
    found["unresolved"] = [f"{r['system']} / {r['rule']}"
                           for _, r in unresolved.iterrows()]
    return found


def difference_intervals(influence_frame: pd.DataFrame, gapped: pd.DataFrame,
                         reference_rule: str, system: str,
                         column: str = "gap_pct") -> pd.DataFrame:
    """Jackknife the *difference* between each rule and a reference rule.

    Intervals on two levels do not give an interval on their difference,
    and the difference is what the paper's surviving claim is about: that
    changing the withdrawal rule moves the all-equity lead by tens of
    points. The two cells are computed on the same sixteen sub-panels, so
    the differences are paired and the jackknife is taken on the paired
    series rather than assembled out of the two marginal standard errors.
    How much that pairing helps depends on how correlated the cells are,
    which is itself reported: where the correlation is low the difference
    is no better resolved than the levels, and saying so is the point.
    """
    if not len(influence_frame) or not len(gapped):
        return pd.DataFrame()
    block = influence_frame[influence_frame["system"] == system]
    if not len(block) or reference_rule not in set(block["rule"]):
        return pd.DataFrame()
    wide = block.pivot(index="dropped", columns="rule", values=column)
    if reference_rule not in wide:
        return pd.DataFrame()
    base_series = wide[reference_rule].to_numpy(dtype=float)
    point_of = {str(r["rule"]): float(r[column])
                for _, r in gapped[gapped["system"] == system].iterrows()}
    reference_point = point_of.get(reference_rule, float("nan"))
    rows: List[Dict[str, Any]] = []
    for rule in wide.columns:
        if str(rule) == reference_rule:
            continue
        values = wide[rule].to_numpy(dtype=float)
        paired = values - base_series
        ok = np.isfinite(paired)
        n = int(ok.sum())
        if n < 2:
            continue
        centred = paired[ok] - paired[ok].mean()
        se = float(np.sqrt((n - 1) / n * float((centred ** 2).sum())))
        point = point_of.get(str(rule), float("nan")) - reference_point
        rows.append({
            "system": str(system), "rule": str(rule),
            "reference_rule": reference_rule,
            "difference_pp": point,
            "standard_error": se,
            "ci_low": point - 1.96 * se,
            "ci_high": point + 1.96 * se,
            "loo_low": float(paired[ok].min()),
            "loo_high": float(paired[ok].max()),
            "correlation": float(np.corrcoef(base_series[ok],
                                             values[ok])[0, 1])
            if n > 2 else float("nan"),
            "deletions": n,
            "sign_survives_every_deletion": bool(
                np.all(np.sign(paired[ok]) == np.sign(point))),
            "ci_excludes_zero": bool(
                (point - 1.96 * se) * (point + 1.96 * se) > 0.0),
        })
    out = pd.DataFrame.from_records(rows)
    if not len(out):
        return out
    order = [r for r in dict.fromkeys(gapped["rule"]) if r != reference_rule]
    out["rule"] = pd.Categorical(out["rule"], categories=order, ordered=True)
    out = out.sort_values("rule").reset_index(drop=True)
    out["rule"] = out["rule"].astype(str)
    return out


def difference_verdict(table: pd.DataFrame) -> Dict[str, Any]:
    """Whether the rule effect is resolved, and by how much.

    ``all_resolved`` is the field the prose turns on. The paper's claim is
    that the withdrawal rule matters more than the pension does, and that
    is a claim about these differences rather than about the levels the
    previous table reports.
    """
    if not len(table):
        return {"measured": False}
    widest = table.loc[table["difference_pp"].abs().idxmax()]
    found: Dict[str, Any] = {
        "measured": True,
        "comparisons": int(len(table)),
        "reference_rule": str(table["reference_rule"].iloc[0]),
        "resolved": int(table["ci_excludes_zero"].sum()),
        "all_resolved": bool(table["ci_excludes_zero"].all()),
        "widest_rule": str(widest["rule"]),
        "widest_pp": float(widest["difference_pp"]),
        "widest_se": float(widest["standard_error"]),
        "widest_ci": (float(widest["ci_low"]), float(widest["ci_high"])),
        "median_correlation": float(table["correlation"].median()),
        "smallest_resolved_pp": float(
            table.loc[table["ci_excludes_zero"], "difference_pp"].abs().min())
        if bool(table["ci_excludes_zero"].any()) else float("nan"),
    }
    unresolved = table[~table["ci_excludes_zero"]]
    found["unresolved"] = [str(r["rule"]) for _, r in unresolved.iterrows()]
    return found


def gamma_verdict(by_gamma: pd.DataFrame, rule: str, system: str,
                  column: str = "gap_pct") -> Dict[str, Any]:
    """Whether one cell's sign holds across the risk aversions scored.

    The contested cell is the paper's headline, and a sign that depends on
    the curvature of the felicity function is a sign about the objective
    rather than about the pension. Cheap to check, because scoring an
    outcome again costs nothing next to simulating it.
    """
    if not len(by_gamma) or "gamma" not in by_gamma:
        return {"measured": False}
    hit = by_gamma[(by_gamma["system"] == system)
                   & (by_gamma["rule"] == rule)].sort_values("gamma")
    if len(hit) < 2:
        return {"measured": False}
    values = hit[column].to_numpy(dtype=float)
    return {
        "measured": True,
        "gammas": [float(g) for g in hit["gamma"]],
        "gaps": [float(v) for v in values],
        "low": float(values.min()),
        "high": float(values.max()),
        "sign_holds": bool(len(set(np.sign(np.round(values, 6)))
                               - {0.0}) <= 1),
        "spread_pp": float(values.max() - values.min()),
    }


def pseudo_values(influence_frame: pd.DataFrame, gapped: pd.DataFrame,
                  column: str = "gap_pct") -> pd.DataFrame:
    """How well behaved each cell's jackknife actually is.

    :func:`intervals` reports a standard error built on the delete-one
    values, and that construction assumes the statistic is close to linear
    in the units being deleted: the sixteen sub-panel estimates should
    scatter around the full-sample one, roughly half above and half below.
    Whether they do is a fact about the data, not an assumption, and it is
    cheap to check once the deletions have been run.

    Two fields carry it. ``below_point`` counts the deletions that fall
    under the full-sample estimate, which for a smooth statistic on
    sixteen units should sit near eight. ``bias_estimate`` is the classical
    jackknife bias, ``(n-1)(mean of deletions - point)``: small relative to
    the estimate when the pseudo-values behave, and not otherwise.

    Reporting this matters here because a jackknife interval that straddles
    zero reads as imprecision, and a cell whose deletions almost all lie on
    one side of the point estimate is saying something different and more
    specific -- that the estimate belongs to the whole panel rather than to
    any subsample of it.
    """
    if not len(influence_frame) or not len(gapped):
        return pd.DataFrame()
    point_of = {(str(r["system"]), str(r["rule"])): float(r[column])
                for _, r in gapped.iterrows()}
    rows: List[Dict[str, Any]] = []
    for (system, rule), block in influence_frame.groupby(["system", "rule"],
                                                         sort=False):
        values = block[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        point = point_of.get((str(system), str(rule)), float("nan"))
        if n < 2 or not np.isfinite(point):
            continue
        mean = float(values.mean())
        rows.append({
            "system": str(system), "rule": str(rule),
            "point": point, "deletions": n,
            "loo_mean": mean,
            "loo_sd": float(values.std(ddof=1)),
            "below_point": int((values < point).sum()),
            "bias_estimate": float((n - 1) * (mean - point)),
            # Against the estimate itself, because a bias of one point
            # means something different beside a gap of two than beside a
            # gap of twenty-five.
            "bias_over_point": float(abs((n - 1) * (mean - point))
                                     / abs(point)) if point else float("inf"),
        })
    return pd.DataFrame.from_records(rows)


def bias_verdict(table: pd.DataFrame, baseline_rule: str, system: str,
                 ratio: float = 3.0) -> Dict[str, Any]:
    """Whether one cell's jackknife misbehaves where the others do not.

    A diagnostic that fired on every cell would be a property of the
    method and worth little. The claim worth making is a comparative one:
    the same machinery on the same panel is well behaved everywhere except
    where the paper's contested sign lives. ``isolated`` is that claim, and
    it is false unless the contested cell is the worst by ``ratio``.
    """
    if not len(table):
        return {"measured": False}
    hit = table[(table["system"] == system) & (table["rule"] == baseline_rule)]
    if not len(hit):
        return {"measured": False}
    row = hit.iloc[0]
    others = table.drop(hit.index)
    found: Dict[str, Any] = {
        "measured": True,
        "system": system, "rule": baseline_rule,
        "point": float(row["point"]),
        "deletions": int(row["deletions"]),
        "below_point": int(row["below_point"]),
        "loo_mean": float(row["loo_mean"]),
        "bias_estimate": float(row["bias_estimate"]),
        "bias_over_point": float(row["bias_over_point"]),
        # The mean deletion crossing zero is the sharpest way to say it:
        # the average fifteen-country panel does not reproduce the sign.
        "mean_deletion_flips_sign": bool(
            np.sign(float(row["loo_mean"])) != np.sign(float(row["point"]))
            and abs(float(row["loo_mean"])) > 1e-9),
    }
    if len(others):
        worst = others["bias_estimate"].abs().max()
        found["others_worst_bias"] = float(worst)
        found["others_median_bias"] = float(
            others["bias_estimate"].abs().median())
        found["others_below_low"] = int(others["below_point"].min())
        found["others_below_high"] = int(others["below_point"].max())
        found["isolated"] = bool(
            abs(found["bias_estimate"]) > ratio * worst)
    return found


def by_objective(frame: pd.DataFrame, pair: Tuple[str, str] = HEADLINE,
                 columns: Sequence[str] = ("cec", "cec_survival"),
                 tie: float = TIE_BAND) -> pd.DataFrame:
    """The same grid's gaps under each objective it was scored on.

    The paper rejects a fixed retirement horizon as not neutral *between
    rules* and then compares portfolios *within* a rule. Those are
    different exposures: a horizon that flatters the rules which divide by
    it flatters both portfolios in a cell equally, so the gap between them
    should be close to insulated even where the levels are not. Close to,
    not exactly -- the two portfolios leave different estates and ruin at
    different rates, and the survival weighting prices both.

    Whether the insulation holds is a fact about this grid rather than an
    argument, which is the reason to compute it. Returns one row per cell
    per objective, so a caller can put the two side by side.
    """
    blocks = []
    for column in columns:
        if column not in frame:
            continue
        block = gaps(frame, pair=pair, column=column, tie=tie)
        block = block.assign(objective=column)
        blocks.append(block)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def objective_verdict(table: pd.DataFrame, baseline_rule: str, system: str,
                      fixed: str = "cec", survival: str = "cec_survival",
                      tolerance: float = 2.0) -> Dict[str, Any]:
    """Whether changing the objective changes what the grid says.

    Three things are separable and the paper needs all three. Whether any
    *sign* moves, which is what its claims are made of. How far the
    contested cell moves, which is the cell a reader will check. And how
    far the *gaps* move on average against how far the *levels* do, which
    is the evidence for or against the insulation argument above.
    """
    if not len(table) or "objective" not in table:
        return {"measured": False}
    wide = table.pivot_table(index=["system", "rule"], columns="objective",
                             values="gap_pct")
    if fixed not in wide or survival not in wide:
        return {"measured": False}
    wide = wide.dropna(subset=[fixed, survival])
    if not len(wide):
        return {"measured": False}
    moved = (wide[survival] - wide[fixed]).abs()
    flipped = wide[(np.sign(wide[fixed].round(6))
                    != np.sign(wide[survival].round(6)))]
    found: Dict[str, Any] = {
        "measured": True,
        "cells": int(len(wide)),
        "median_move_pp": float(moved.median()),
        "worst_move_pp": float(moved.max()),
        "worst_cell": [str(x) for x in moved.idxmax()],
        "signs_flipped": int(len(flipped)),
        "flipped_cells": [f"{a} / {b}" for a, b in flipped.index],
        "insulated": bool(len(flipped) == 0
                          and float(moved.max()) <= tolerance),
        "tolerance_pp": float(tolerance),
    }
    key = (system, baseline_rule)
    if key in wide.index:
        found["contested_fixed"] = float(wide.loc[key, fixed])
        found["contested_survival"] = float(wide.loc[key, survival])
        found["contested_move_pp"] = float(
            wide.loc[key, survival] - wide.loc[key, fixed])
        found["contested_sign_holds"] = bool(
            np.sign(round(float(wide.loc[key, fixed]), 6))
            == np.sign(round(float(wide.loc[key, survival]), 6)))
    return found


def level_shift(frame: pd.DataFrame, fixed: str = "cec",
                survival: str = "cec_survival") -> Dict[str, Any]:
    """How far the *levels* move when the objective does.

    The counterpart to :func:`objective_verdict`. If the levels move a lot
    and the gaps move little, the insulation argument is supported by the
    data rather than by assertion; if both move alike, it is not.
    """
    if fixed not in frame or survival not in frame:
        return {"measured": False}
    block = frame[[fixed, survival]].dropna()
    if not len(block):
        return {"measured": False}
    shift = (block[survival] / block[fixed] - 1.0) * 100.0
    return {
        "measured": True,
        "rows": int(len(block)),
        "median_level_shift_pct": float(shift.median()),
        "low_level_shift_pct": float(shift.min()),
        "high_level_shift_pct": float(shift.max()),
        # The span is what the comparison needs. A median near zero can
        # hide levels moving ten per cent in both directions, and it is
        # the movement rather than its average that the gaps are being
        # held against.
        "level_span_pct": float(shift.max() - shift.min()),
    }
