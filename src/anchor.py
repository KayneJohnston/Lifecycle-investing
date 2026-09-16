"""A prediction put to an institution that chose independently.

Everything this paper reports is the output of a calibrated simulation. The
honest objection is that nothing in it has been confronted with anything
outside itself: the mechanism is derived, the grid is swept, the signs are
jackknifed, and at no point does a fact from the world get a chance to
disagree.

There is one confrontation available that does not need microdata, and it
is a sharp one. The paper's mechanism says a means-tested system needs a
withdrawal rule whose payment reads the balance -- a rule that never does
turns a good return into assessable assets while consumption stands still,
and the portfolio ordering reverses. That is a claim about which *shape* of
drawdown rule an asset-tested system requires.

Australia means-tests its public pension against assets. It also compels
every retiree drawing an account-based private pension to pay out, each
year, at least a legislated percentage of the account balance, on a rate
that rises with age: four per cent under 65, rising in steps to fourteen at
95 and over (Superannuation Industry (Supervision) Regulations 1994,
Schedule 7). That is the balance-reading family, in statute, chosen by a
legislature that had never seen this model.

So the prediction has an out-of-sample test that costs nothing to run: take
the rule that country actually mandates, at the rates it actually mandates,
and ask whether the all-equity portfolio keeps its lead under the means
test. If the mechanism is right it should, and it should do so without the
paper having tuned anything -- the rates are not ours.

**What this establishes and what it does not.** It is an institutional
anchor, not a behavioural one. It says the rule shape a means-testing
country legislates is the shape the model says such a country needs; it
does not say retirees behave as the model's household does, and it cannot,
because a design fact is not a holdings fact. The behavioural implication
-- that retirement-phase portfolios in a system mandating this shape should
sit at the high-equity end relative to a target-date glide path at the same
age -- is stated by :func:`observable_prediction` in falsifiable form, with
the data that would settle it named. That data is not in this repository
and could not be fetched into it: the egress policy this project runs under
denies every bulk statistical host, which `src.observed` documents at
length.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: The source of the schedule, quoted where the paper quotes it so that the
#: statute and the simulation cannot drift apart in a rerun.
STATUTE: str = ("Superannuation Industry (Supervision) Regulations 1994, "
                "Schedule 7")


def schedule_frame(schedule: Sequence[Tuple[int, float]]) -> pd.DataFrame:
    """The statutory minimum, as a table a reader can check against the law."""
    rows: List[Dict[str, Any]] = []
    steps = list(schedule)
    for i, (age, rate) in enumerate(steps):
        upper = steps[i + 1][0] - 1 if i + 1 < len(steps) else None
        band = (f"Under {steps[1][0]}" if i == 0 and len(steps) > 1
                else f"{age} and over" if upper is None
                else f"{age} to {upper}")
        rows.append({"age_band": band, "minimum_share": float(rate)})
    return pd.DataFrame.from_records(rows)


def compare(simulate: Callable[[str, Any, str], Any],
            systems: Sequence[str], rules: Sequence[Tuple[str, Any]],
            strategies: Sequence[str],
            score: Callable[[Any], Dict[str, Any]],
            log_every: int = 6) -> pd.DataFrame:
    """Every regime crossed with the legislated rule and its comparators."""
    rows: List[Dict[str, Any]] = []
    total = len(systems) * len(rules) * len(strategies)
    n = 0
    for system in systems:
        for label, rule in rules:
            for strategy in strategies:
                row: Dict[str, Any] = {"system": str(system),
                                       "rule": str(label),
                                       "strategy": str(strategy)}
                row.update(score(simulate(system, rule, strategy)))
                rows.append(row)
                n += 1
                if log_every and n % int(log_every) == 0:
                    LOGGER.info("  scored %d of %d", n, total)
    return pd.DataFrame.from_records(rows)


def gaps(frame: pd.DataFrame, pair: Tuple[str, str],
         column: str = "cec_survival") -> pd.DataFrame:
    """The challenger's lead over the incumbent, by regime and rule."""
    challenger, incumbent = pair
    wide = frame.pivot_table(index=["system", "rule"], columns="strategy",
                             values=column)
    if challenger not in wide.columns or incumbent not in wide.columns:
        return pd.DataFrame()
    out = wide.reset_index()
    out["gap_pct"] = 100.0 * (out[challenger] / out[incumbent] - 1.0)
    out["winner"] = np.where(out["gap_pct"] > 0.0, challenger, incumbent)
    return out


def verdict(gapped: pd.DataFrame, system: str, legislated: str,
            assumed: str) -> Dict[str, Any]:
    """Whether the rule a means-testing country mandates passes the test.

    The prediction is directional and was made before this rule was run:
    under an assets test, a balance-reading rule keeps the all-equity
    portfolio ahead and a balance-blind one does not. The legislated
    schedule is the first; the rule the literature assumes is the second.
    """
    block = gapped[gapped["system"] == system]
    if not len(block):
        return {"measured": False}
    keyed = block.set_index("rule")["gap_pct"]
    if legislated not in keyed.index or assumed not in keyed.index:
        return {"measured": False}
    law = float(keyed.loc[legislated])
    lit = float(keyed.loc[assumed])
    return {
        "measured": True,
        "system": str(system),
        "legislated_rule": str(legislated),
        "assumed_rule": str(assumed),
        "gap_under_the_legislated_rule_pct": law,
        "gap_under_the_assumed_rule_pct": lit,
        "legislated_rule_keeps_the_lead": bool(law > 0.0),
        "assumed_rule_loses_it": bool(lit < 0.0),
        # The whole prediction in one flag: the statute's rule shape is on
        # the side the mechanism says an asset-tested system needs, and the
        # literature's is not.
        "prediction_holds": bool(law > 0.0 and lit < 0.0),
        "swing_pp": law - lit,
    }


def observable_prediction(gapped: pd.DataFrame, system: str,
                          legislated: str) -> Dict[str, Any]:
    """The behavioural claim this institutional anchor does not establish.

    Stated so it can be falsified by someone with the data. A design fact is
    not a holdings fact, and a paper that let the first stand in for the
    second would be doing what it spends its length objecting to.
    """
    block = gapped[(gapped["system"] == system)
                   & (gapped["rule"] == legislated)]
    if not len(block):
        return {"measured": False}
    row = block.iloc[0]
    return {
        "measured": True,
        "implied_winner": str(row["winner"]),
        "gap_pct": float(row["gap_pct"]),
        "claim": ("Retirement-phase portfolios in a system that both "
                  "means-tests the public pension and mandates a "
                  "percentage-of-balance drawdown should sit at the "
                  "high-equity end relative to a target-date glide path at "
                  "the same age."),
        "falsified_by": ("Fund-level retirement-phase asset allocation by "
                         "member age, against target-date glide paths at "
                         "matched ages. Neither is carried here."),
    }
