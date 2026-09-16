"""The policy solved rather than picked, and the two ways of pricing a menu.

The referee objection this module answers is that the paper compares menu
items and calls the comparison a result. The defence is a solved policy --
and a solved policy invites its own failure mode, which is a menu chosen to
lose. So the gap is measured twice: against the rule defaults actually
offer, and against the best cell of the paper's own grid. Most of what is
tested here is that the second measurement is taken and reported, because
without it the first is rhetoric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plan as pl  # noqa: E402
from src import policy as pol  # noqa: E402


def _solved(rows) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {"system": s, "rule": r, "rate": rate, "label": f"{r}",
         "cec": cec, "rounds": 2, "converged": True,
         "mean_equity": eq, "equity_at_retirement": eq,
         "mean_equity_in_retirement": eq,
         "equity_falls_through_retirement": False,
         "rule_reads_the_balance": pol._reads_balance(r)}
        for s, r, rate, cec, eq in rows])


class TestWhichRulesReadTheBalance:
    """The division Section 2 turns on. A solved policy that lands on one
    side of it is the mechanism arriving from the optimisation; a
    classifier that got the side wrong would report that either way."""

    def test_a_fixed_real_rule_does_not(self) -> None:
        assert not pol._reads_balance("constant_real")
        assert not pol._reads_balance("fixed_real_rule")

    @pytest.mark.parametrize("rule", ["constant_percent", "amortisation",
                                      "gompertz", "life_expectancy",
                                      "endowment", "guyton_klinger",
                                      "vanguard_dynamic"])
    def test_every_other_family_does(self, rule) -> None:
        assert pol._reads_balance(rule)

    def test_every_registered_rule_is_classified_one_way(self) -> None:
        """A rule the classifier has never heard of would silently be
        called balance-blind, which is the side the paper's finding sits
        on -- so the failure would flatter the result."""
        from src import spending as spg

        for rule in spg.REGISTRY:
            assert isinstance(pol._reads_balance(rule), bool)


class TestTheVerdictReportsEitherAnswer:
    """The section exists to find out whether the interaction is about
    optima or about menus, so the classifier has to be able to say the
    second."""

    def test_a_rule_that_changes_with_the_pension_is_reported_as_changing(
            self) -> None:
        solved = _solved([("us_social_security", "gompertz", np.nan, 1.36, 1.0),
                          ("age_pension_matched", "constant_percent", 0.10,
                           1.05, 1.0)])
        gaps = pd.DataFrame.from_records([
            {"system": "us_social_security", "menu_gap_pct": 9.6,
             "default_gap_pct": 31.1, "best_menu_rule": "amortisation at 4%"},
            {"system": "age_pension_matched", "menu_gap_pct": 3.0,
             "default_gap_pct": 115.4, "best_menu_rule": "amortisation at 6%"}])
        got = pol.verdict(solved, gaps, "age_pension_matched")
        assert got["measured"]
        assert got["rule_changes_with_the_pension"]
        assert not got["one_rule_wins_everywhere"]
        assert got["solved_rule_reads_the_balance"]

    def test_a_rule_that_does_not_change_is_reported_as_not_changing(
            self) -> None:
        """If both regimes solve to the same rule the paper's interaction
        is about defaults rather than optima, and the section has to say
        so rather than find a way not to."""
        solved = _solved([("us_social_security", "constant_percent", 0.10,
                           1.36, 1.0),
                          ("age_pension_matched", "constant_percent", 0.10,
                           1.05, 1.0)])
        gaps = pd.DataFrame.from_records([
            {"system": "us_social_security", "menu_gap_pct": 9.6,
             "default_gap_pct": 31.1, "best_menu_rule": "x"},
            {"system": "age_pension_matched", "menu_gap_pct": 3.0,
             "default_gap_pct": 115.4, "best_menu_rule": "y"}])
        got = pol.verdict(solved, gaps, "age_pension_matched")
        assert not got["rule_changes_with_the_pension"]
        assert got["one_rule_wins_everywhere"]

    def test_an_allocation_that_does_not_move_is_reported_as_not_moving(
            self) -> None:
        solved = _solved([("us_social_security", "gompertz", np.nan, 1.36, 1.0),
                          ("age_pension_matched", "constant_percent", 0.10,
                           1.05, 1.0)])
        gaps = pd.DataFrame.from_records([
            {"system": s, "menu_gap_pct": 1.0, "default_gap_pct": 2.0,
             "best_menu_rule": "x"}
            for s in ("us_social_security", "age_pension_matched")])
        got = pol.verdict(solved, gaps, "age_pension_matched")
        assert not got["equity_moves_with_the_pension"]

    def test_a_missing_control_is_not_measured(self) -> None:
        solved = _solved([("age_pension_matched", "constant_percent", 0.10,
                           1.05, 1.0)])
        gaps = pd.DataFrame.from_records([
            {"system": "age_pension_matched", "menu_gap_pct": 3.0,
             "default_gap_pct": 115.4, "best_menu_rule": "y"}])
        assert not pol.verdict(solved, gaps,
                               "age_pension_matched").get("measured")


class TestTheMenuIsPricedTwice:
    """A solved policy beaten against one badly chosen comparator proves
    nothing. The second comparator is the paper's own grid, and the ratio
    between the two gaps is what says whether the menu or the default is
    the expensive thing."""

    @staticmethod
    def _score_factory(values):
        def score(system, strategy, plan, equity, domestic):
            return values[(system, strategy, plan)]
        return score

    def test_it_reports_the_best_cell_and_the_default_separately(self) -> None:
        solved = _solved([("s", "constant_percent", 0.1, 1.10, 1.0)])
        schedules = {"a": (np.ones(3), np.zeros(3)),
                     "b": (np.ones(3), np.zeros(3))}
        rules = [("default", "planD"), ("amortisation at 6%", "planA")]
        values = {("s", "a", "planD"): 0.50, ("s", "b", "planD"): 0.55,
                  ("s", "a", "planA"): 1.00, ("s", "b", "planA"): 0.90}
        got = pol.menu_gap(self._score_factory(values), solved, ["a", "b"],
                           schedules, rules, headline_rule="default")
        row = got.iloc[0]
        assert row["best_menu_rule"] == "amortisation at 6%"
        assert row["best_menu_cec"] == pytest.approx(1.00)
        assert row["menu_gap_pct"] == pytest.approx(10.0)
        # The default is the *best* item at the default rule, not the worst.
        assert row["default_menu_cec"] == pytest.approx(0.55)
        assert row["default_gap_pct"] == pytest.approx(100.0)

    def test_the_two_gaps_can_disagree_about_which_is_expensive(self) -> None:
        """They did, in the run this paper ships: the default costs far
        more under the means test and the menu costs slightly less. A
        section that reported only the first would have had the sign of its
        own comparison wrong."""
        solved = _solved([("tested", "r", 0.1, 1.0, 1.0),
                          ("untested", "r", 0.1, 1.0, 1.0)])
        gaps = pd.DataFrame.from_records([
            {"system": "tested", "menu_gap_pct": 3.0,
             "default_gap_pct": 115.0, "best_menu_rule": "x"},
            {"system": "untested", "menu_gap_pct": 9.6,
             "default_gap_pct": 31.0, "best_menu_rule": "y"}])
        got = pol.verdict(gaps=gaps, solved=solved, headline="tested",
                          control="untested")
        assert not got["menu_costs_more_under_the_test"]
        assert got["default_costs_more_under_the_test"]
        assert got["default_gap_ratio"] == pytest.approx(115.0 / 31.0)


class TestTheShippedRun:
    """What the run in the repository found, so a rerun that moved it
    cannot leave the paper's prose behind."""

    @staticmethod
    def _table(name: str) -> pd.DataFrame:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / f"{name}.csv"),
                         recursive=True)
        if not hits:
            pytest.skip(f"{name} has not been generated")
        return pd.read_csv(hits[0])

    def test_every_search_reached_a_fixed_point(self) -> None:
        """An unconverged search is a policy nobody solved, and reporting
        one as an optimum would be the section's worst failure."""
        assert self._table("policy_solved")["converged"].all()

    def test_the_solved_rule_reads_the_balance_everywhere(self) -> None:
        assert self._table("policy_solved")["rule_reads_the_balance"].all()

    def test_the_menu_gap_is_smaller_than_the_default_gap(self) -> None:
        """The paper says its own grid is close to optimal and the default
        is not. If a rerun inverted that, the claim has to change."""
        gaps = self._table("policy_menu_gap")
        assert (gaps["menu_gap_pct"] < gaps["default_gap_pct"]).all()

    def test_the_solved_allocation_is_flat_within_each_regime(self) -> None:
        """The paper prints one equity figure per regime and says it holds
        in every retirement year. If the schedule sloped, that sentence
        would be describing an average as a constant."""
        schedule = self._table("policy_schedule")
        for system, part in schedule.groupby("system"):
            assert part["equity"].nunique() == 1, system
