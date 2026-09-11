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
