"""What a retiree under an assets test should actually do, solved.

Everything in this project up to here compares things off a menu. The
headline is two funds; the withdrawal-rule section ranks eight rules; the
ordering grid crosses five pension regimes with eight of them. That answers
"which of these is best" and leaves standing the question a referee asks
first: what is the *optimal* policy under a means test, and how much does
the menu cost the household that picks from it?

This module runs the alternating search of :mod:`src.plan` -- choose the
plan for the current allocation schedule, re-solve the schedule for that
plan, stop when a round returns the plan it began with -- once under each
pension regime. Running it per regime is what turns a solved policy into a
statement about the institution: if the optimal drawdown rule is the same
under an earnings-related pension and under an assets-tested one, the
paper's interaction is about menus and not about optima, and the section
has to say so.

**The menu gap is the number this exists to produce.** For each regime it
is the certainty-equivalent distance between the solved policy and the best
the paper's own two-fund, one-rule menu can do. A large gap under the means
test and a small one without it says the test is what makes the menu
expensive -- which is the paper's thesis, restated as a cost rather than as
an ordering.

**What it does not do.** This is a solved policy over a rule family and a
free-form deterministic allocation schedule, not a dynamic program over the
assets test. The household commits to a rule at retirement and does not
re-optimise as the balance crosses the taper; a true state-contingent
policy would condition the withdrawal on where the balance sits against the
threshold, and would do at least as well. What is solved here is therefore
a lower bound on what the optimal policy achieves, and the menu gap it
reports is a lower bound on the menu's cost.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def solve_by_system(bench_for: Callable[..., Any],
                    systems: Sequence[str], plans: Sequence[Any],
                    gamma: float, equity_grid: Sequence[float],
                    domestic_grid: Sequence[float],
                    start_equity: float = 1.0, start_domestic: float = 0.1,
                    bond_share: float = 0.7, domestic_band_years: int = 5,
                    n_sweeps: int = 2, max_rounds: int = 4,
                    leverage_grid: Sequence[float] = (1.0,),
                    spread: float = 0.0) -> pd.DataFrame:
    """The joint optimum under each pension regime, one row apiece.

    ``bench_for(system, leverage)`` returns a :class:`src.plan.PlanBench`
    built on that regime's spec at that borrowing level. The search itself is
    :func:`src.plan.alternate`; what is new here is running it per regime and
    keeping the solved schedule so a caller can ask whether the optimal
    *allocation* moves as well as the optimal rule.

    **Why the leverage grid is not optional decoration.** The question this
    module exists to answer is whether the assets test moves the allocation
    or only the rule, and an allocation grid that stops at the whole
    portfolio cannot answer it: two regimes that both want everything they
    are allowed score identically whether they want the same thing or not.
    So the ceiling comes off, exactly as :mod:`src.ceiling` lifts it for the
    balance sweep, and the reported holding is the one the household chose
    rather than the largest one on offer. A holding that lands strictly
    inside the grid is an answer; one that lands on its edge is reported as
    censored rather than as an optimum.
    """
    from . import plan as pl

    ladder = [float(x) for x in leverage_grid] or [1.0]
    rows: List[Dict[str, Any]] = []
    rungs: List[Dict[str, Any]] = []
    for system in systems:
        best: Dict[str, Any] | None = None
        flat: Dict[str, Any] | None = None
        for leverage in ladder:
            bench = bench_for(str(system), leverage)
            joint = pl.alternate(bench, plans, gamma, equity_grid,
                                 domestic_grid, start_equity, start_domestic,
                                 bond_share, domestic_band_years, n_sweeps,
                                 max_rounds)
            if joint.get("plan") is None:
                continue
            LOGGER.info("  %s at %.2fx: %s, CEC %.6f", system, leverage,
                        joint["plan"].label(), float(joint["cec"]))
            rungs.append({"system": str(system), "leverage": leverage,
                          "rule": joint["plan"].rule,
                          "label": joint["plan"].label(),
                          "cec": float(joint["cec"]),
                          "converged": bool(joint.get("converged", False))})
            if leverage == 1.0:
                flat = dict(joint, n_working=bench.spec.n_working)
            if best is None or float(joint["cec"]) > float(best["cec"]):
                best = dict(joint, leverage=leverage,
                            n_working=bench.spec.n_working)
        if best is None:
            continue
        plan = best["plan"]
        leverage = float(best["leverage"])
        equity = np.asarray(best["equity"], dtype=float)
        n_working = int(best["n_working"])
        # What the household actually holds: the share of the portfolio in
        # equity, times what it borrowed to hold it. At leverage one these
        # are the same number, which is why the earlier grid could not tell
        # a corner from a ceiling.
        holding = equity.copy()
        holding[n_working:] = holding[n_working:] * leverage
        retired = holding[n_working:]
        rows.append({
            "system": str(system),
            "rule": plan.rule,
            "rate": np.nan if plan.rate is None else float(plan.rate),
            "label": plan.label(),
            "cec": float(best["cec"]),
            # The same search with borrowing switched off. The paper's menu
            # is two unlevered funds, so a menu gap taken against the
            # levered answer would be pricing the rule and the borrowing
            # together and calling the sum the cost of a drawdown default.
            "unlevered_cec": (float(flat["cec"]) if flat is not None
                              else float("nan")),
            "unlevered_rule": (str(flat["plan"].rule) if flat is not None
                               else ""),
            "leverage_premium_pct": (
                100.0 * (float(best["cec"]) / float(flat["cec"]) - 1.0)
                if flat is not None and float(flat["cec"]) > 0
                else float("nan")),
            "rounds": int(len(best.get("rounds", []))),
            "converged": bool(best.get("converged", False)),
            "leverage": leverage,
            "mean_equity": float(equity.mean()),
            "mean_holding_in_retirement": float(retired.mean())
            if len(retired) else float("nan"),
            "holding_at_retirement": float(retired[0]) if len(retired) else
            float("nan"),
            "equity_at_retirement": float(equity[n_working:][0])
            if len(retired) else float("nan"),
            "mean_equity_in_retirement": float(equity[n_working:].mean())
            if len(retired) else float("nan"),
            "equity_falls_through_retirement": bool(
                len(retired) > 1 and retired[-1] < retired[0] - 1e-9),
            # An optimum on the edge of the grid is not an optimum, and the
            # paper has to say which it has rather than print the number
            # either way.
            "holding_is_censored": bool(
                len(retired) and abs(float(retired.max())
                                     - max(ladder) * max(equity_grid))
                < 1e-9),
            "rule_reads_the_balance": _reads_balance(plan.rule),
        })
        LOGGER.info("  %s: %s, CEC %.6f, holding %.2fx (%d rounds%s)",
                    system, plan.label(), float(best["cec"]),
                    float(retired.mean()) if len(retired) else float("nan"),
                    len(best.get("rounds", [])),
                    "" if best.get("converged") else ", not converged")
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(rungs)


def _reads_balance(rule: str) -> bool:
    """Whether a rule's payment is a function of the portfolio that year.

    The paper's division. A rule whose payment never reads the balance turns
    a good return into assessable assets while consumption stands still,
    which is the whole of the mechanism, so whether the *solved* policy
    lands on one side of that line is the question this section asks.
    """
    from . import plan as pl

    blind = {"constant_real", "fixed_real_rule"}
    if rule in blind:
        return False
    # Everything the plan module knows can deplete or amortise reads the
    # balance in some form; the blind rules are the fixed-real family.
    return rule in pl.CAN_DEPLETE or rule in {"amortisation", "gompertz",
                                              "life_expectancy",
                                              "constant_percent"}


def menu_gap(score: Callable[[str, str, Any, np.ndarray, np.ndarray], float],
             solved: pd.DataFrame, strategies: Sequence[str],
             schedules: Mapping[str, Tuple[np.ndarray, np.ndarray]],
             rules: Sequence[Tuple[str, Any]],
             headline_rule: str | None = None,
             column: str = "unlevered_cec") -> pd.DataFrame:
    """What the paper's own menu costs, regime by regime.

    ``score(system, strategy, plan, equity, domestic)`` returns the
    certainty equivalent of one menu item under one regime at one rule.

    Two gaps, because they answer different objections. The first is
    against the rule the *literature* assumes -- a fixed real four per cent
    on a target-date fund, which is what defaults actually offer. The
    second is against the best cell of the paper's own eight-rule grid,
    which is the fair comparison: a reader who suspects the menu was chosen
    to lose should see the solved policy beaten against the best thing the
    paper itself reports. A gap that survives the second is a gap the menu
    cannot close by picking better from within it.
    """
    rows: List[Dict[str, Any]] = []
    for _, row in solved.iterrows():
        system = str(row["system"])
        best_key, best_rule, best_cec = None, None, float("-inf")
        headline_cec = float("nan")
        for label, plan in rules:
            for strategy in strategies:
                equity, domestic = schedules[str(strategy)]
                value = float(score(system, str(strategy), plan, equity,
                                    domestic))
                if value > best_cec:
                    best_key, best_rule, best_cec = (str(strategy),
                                                     str(label), value)
                if headline_rule is not None and str(label) == headline_rule \
                        and value > (headline_cec if headline_cec ==
                                     headline_cec else float("-inf")):
                    headline_cec = value
        # Scored against the solved policy with borrowing switched off,
        # because the menu it is beaten against cannot borrow either. The
        # levered answer is reported beside it as its own quantity rather
        # than folded into the cost of a default.
        solved_cec = float(row[column] if column in row
                           and row[column] == row[column] else row["cec"])
        rows.append({
            "system": system,
            "solved_rule": str(row["rule"]),
            "solved_cec": solved_cec,
            "best_menu_item": best_key,
            "best_menu_rule": best_rule,
            "best_menu_cec": best_cec,
            "menu_gap_pct": 100.0 * (solved_cec / best_cec - 1.0)
            if best_cec > 0 else float("nan"),
            "default_menu_cec": headline_cec,
            "default_gap_pct": 100.0 * (solved_cec / headline_cec - 1.0)
            if headline_cec == headline_cec and headline_cec > 0
            else float("nan"),
        })
    return pd.DataFrame.from_records(rows)


def verdict(solved: pd.DataFrame, gaps: pd.DataFrame, headline: str,
            control: str = "us_social_security",
            ladder: pd.DataFrame | None = None) -> Dict[str, Any]:
    """What the solved policy says, classified from the search.

    Three questions the prose has to answer whichever way they come out.
    Does the optimal drawdown rule change with the pension? Does the optimal
    *allocation* change with it? And does the menu cost more under a test
    than without one?
    """
    if not len(solved) or not len(gaps):
        return {"measured": False}
    by = solved.set_index("system")
    gap = gaps.set_index("system")
    if headline not in by.index or control not in by.index:
        return {"measured": False}
    hit, base = by.loc[headline], by.loc[control]
    rules = list(dict.fromkeys(solved["rule"]))
    out: Dict[str, Any] = {
        "measured": True,
        "headline_system": str(headline),
        "control_system": str(control),
        "rule_under_the_test": str(hit["rule"]),
        "rule_without_it": str(base["rule"]),
        "rule_changes_with_the_pension": bool(hit["rule"] != base["rule"]),
        "one_rule_wins_everywhere": bool(len(rules) == 1),
        "rules_chosen": rules,
        # The division the paper's mechanism rests on. A solved policy that
        # lands on a balance-reading rule under the test is the mechanism
        # arriving from the optimisation rather than from the menu.
        "solved_rule_reads_the_balance": bool(hit["rule_reads_the_balance"]),
        "every_solved_rule_reads_the_balance": bool(
            solved["rule_reads_the_balance"].all()),
        # The *holding* rather than the share, because the share is capped
        # at the whole portfolio and two households pinned against that cap
        # are indistinguishable. Section A.3's finding is exactly that the
        # cap binds here, so a comparison made on the share would be
        # answering the grid's question rather than the paper's.
        "holding_under_the_test": float(hit["mean_holding_in_retirement"]),
        "holding_without_it": float(base["mean_holding_in_retirement"]),
        "holding_gap": abs(float(hit["mean_holding_in_retirement"])
                           - float(base["mean_holding_in_retirement"])),
        # Against the ladder's own step rather than a number chosen here.
        # A gap the search could not have resolved is not a difference the
        # paper may report, and the step is what the search can resolve.
        "ladder_step": _ladder_step(solved, ladder),
        "holding_moves_with_the_pension": bool(
            abs(float(hit["mean_holding_in_retirement"])
                - float(base["mean_holding_in_retirement"]))
            > _ladder_step(solved, ladder)),
        # And whether either answer is a number or an edge. A censored
        # optimum cannot support a claim about invariance in either
        # direction, and the prose has to say so rather than print it.
        "any_holding_is_censored": bool(solved["holding_is_censored"].any()),
        "every_holding_is_censored": bool(solved["holding_is_censored"].all()),
        "equity_under_the_test": float(hit["mean_equity"]),
        "equity_without_it": float(base["mean_equity"]),
        "equity_moves_with_the_pension": bool(
            abs(float(hit["mean_equity"]) - float(base["mean_equity"]))
            > 0.05),
        "menu_gap_under_the_test_pct": float(gap.loc[headline,
                                                     "menu_gap_pct"]),
        "menu_gap_without_it_pct": float(gap.loc[control, "menu_gap_pct"]),
        "default_gap_under_the_test_pct": float(
            gap.loc[headline, "default_gap_pct"]),
        "default_gap_without_it_pct": float(
            gap.loc[control, "default_gap_pct"]),
        "best_menu_rule_under_the_test": str(
            gap.loc[headline, "best_menu_rule"]),
        "best_menu_rule_without_it": str(gap.loc[control, "best_menu_rule"]),
        "every_search_converged": bool(solved["converged"].all()),
        "most_rounds": int(solved["rounds"].max()),
        # What borrowing is worth on its own, kept apart from what solving
        # the rule is worth. The two were briefly reported as one number
        # here, which priced a drawdown default at the sum of the default
        # and the leverage the menu was never offered.
        "leverage_premium_under_the_test_pct": float(
            hit.get("leverage_premium_pct", float("nan"))),
        "leverage_premium_without_it_pct": float(
            base.get("leverage_premium_pct", float("nan"))),
        "solved_rule_survives_switching_borrowing_off": bool(
            str(hit.get("unlevered_rule", "")) == str(hit["rule"])),
    }
    out["menu_costs_more_under_the_test"] = bool(
        out["menu_gap_under_the_test_pct"] > out["menu_gap_without_it_pct"])
    out["menu_gap_ratio"] = (
        out["menu_gap_under_the_test_pct"] / out["menu_gap_without_it_pct"]
        if out["menu_gap_without_it_pct"] > 0 else float("inf"))
    out["default_costs_more_under_the_test"] = bool(
        out["default_gap_under_the_test_pct"]
        > out["default_gap_without_it_pct"])
    out["default_gap_ratio"] = (
        out["default_gap_under_the_test_pct"]
        / out["default_gap_without_it_pct"]
        if out["default_gap_without_it_pct"] > 0 else float("inf"))
    return out


def _ladder_step(solved: pd.DataFrame,
                 ladder: pd.DataFrame | None) -> float:
    """The finest difference in holding the borrowing search can resolve.

    Two regimes whose solved holdings differ by less than one rung are two
    regimes this search cannot tell apart, and reporting the difference as
    a finding would be reading precision off a grid that does not have it.
    Taken from the swept ladder when one is available and from the solved
    rows otherwise; a single-rung search resolves nothing, so the step is
    infinite and no difference is reportable.
    """
    source = ladder if ladder is not None and len(ladder) else solved
    if "leverage" not in getattr(source, "columns", ()):
        return float("inf")
    rungs = sorted({float(x) for x in source["leverage"]})
    if len(rungs) < 2:
        return float("inf")
    return float(min(b - a for a, b in zip(rungs, rungs[1:])))


def schedule_frame(solved: pd.DataFrame,
                   schedules: Mapping[str, Tuple[np.ndarray, np.ndarray]],
                   n_working: int) -> pd.DataFrame:
    """The solved equity path through retirement, one column per regime.

    Printed because the paper's claim is about a *portfolio*, and a section
    that reported only which rule won would have solved the drawdown problem
    and left the allocation one to a sentence.
    """
    rows: List[Dict[str, Any]] = []
    for _, row in solved.iterrows():
        system = str(row["system"])
        if system not in schedules:
            continue
        equity, domestic = schedules[system]
        equity = np.asarray(equity, dtype=float)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        for offset, share in enumerate(equity[int(n_working):]):
            rows.append({"system": system, "retirement_year": offset,
                         "equity": float(share),
                         "leverage": leverage,
                         "holding": float(share) * leverage,
                         "domestic": float(np.asarray(domestic)[
                             int(n_working) + offset])})
    return pd.DataFrame.from_records(rows)
