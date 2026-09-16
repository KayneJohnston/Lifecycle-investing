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


def _solved(rows, leverage: float = 1.0, censored: bool = False
            ) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {"system": s, "rule": r, "rate": rate, "label": f"{r}",
         "cec": cec, "rounds": 2, "converged": True,
         "leverage": leverage,
         "mean_equity": eq, "equity_at_retirement": eq,
         "mean_equity_in_retirement": eq,
         "mean_holding_in_retirement": eq * leverage,
         "holding_at_retirement": eq * leverage,
         "equity_falls_through_retirement": False,
         "holding_is_censored": censored,
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
        assert not got["holding_moves_with_the_pension"]

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

    def test_the_paper_does_not_call_a_sloping_schedule_a_constant(self
                                                                    ) -> None:
        """The printed figure is one number per regime, and the schedule
        behind it stopped being flat when the allocation grid was allowed
        past the whole portfolio. So either the schedule is flat, or the
        table says the number is a mean -- what must not happen is the
        earlier sentence, which described an average as a constant.
        """
        import inspect

        from paper import short as sh

        schedule = self._table("policy_schedule")
        flat = all(part["equity"].nunique() == 1
                   for _, part in schedule.groupby("system"))
        note = inspect.getsource(sh._solved_table)
        if flat:
            return
        assert "mean over them" in note, (
            "the solved schedule varies across retirement years and the "
            "table note still describes the printed figure as holding in "
            "every one of them")
        assert "every year of every regime" not in note


class TestTheCeilingComesOff:
    """The claim this section makes is that the allocation does not move
    with the pension. A grid stopping at the whole portfolio cannot support
    that claim in either direction -- two regimes pinned at the cap are
    indistinguishable however different they are -- and the Internet
    Appendix already establishes that the cap binds for this household. So
    the search runs past it, and the flag that says whether it is still
    pinned has to work."""

    def test_an_answer_on_the_edge_of_the_ladder_is_called_censored(self
                                                                    ) -> None:
        got = pol.verdict(
            _solved([("a", "gompertz", np.nan, 1.0, 1.0),
                     ("b", "gompertz", np.nan, 1.0, 1.0)],
                    leverage=2.0, censored=True),
            pd.DataFrame.from_records([
                {"system": s, "menu_gap_pct": 1.0, "default_gap_pct": 2.0,
                 "best_menu_rule": "x"} for s in ("a", "b")]),
            "a", control="b")
        assert got["every_holding_is_censored"]

    def test_an_interior_answer_is_not(self) -> None:
        got = pol.verdict(
            _solved([("a", "gompertz", np.nan, 1.0, 1.0),
                     ("b", "gompertz", np.nan, 1.0, 1.0)],
                    leverage=1.5, censored=False),
            pd.DataFrame.from_records([
                {"system": s, "menu_gap_pct": 1.0, "default_gap_pct": 2.0,
                 "best_menu_rule": "x"} for s in ("a", "b")]),
            "a", control="b")
        assert not got["any_holding_is_censored"]

    def test_the_holding_is_the_share_times_what_was_borrowed(self) -> None:
        """A share of 1.0 at 1.75x is a holding of 1.75, and it is the
        holding the paper compares across regimes. Reading the share
        instead would report two different households as the same one."""
        frame = _solved([("a", "gompertz", np.nan, 1.0, 1.0)], leverage=1.75)
        assert frame["mean_holding_in_retirement"].iloc[0] == pytest.approx(
            1.75)

    def test_two_regimes_at_different_holdings_are_reported_as_differing(
            self) -> None:
        """The failure mode the ladder exists to rule out: at a shared
        ceiling this comparison came out equal by construction."""
        solved = pd.concat([
            _solved([("tested", "constant_percent", 0.1, 1.0, 1.0)],
                    leverage=1.0),
            _solved([("untested", "gompertz", np.nan, 1.3, 1.0)],
                    leverage=2.0)], ignore_index=True)
        ladder = pd.DataFrame.from_records(
            [{"system": s, "leverage": r, "cec": 1.0}
             for s in ("tested", "untested")
             for r in (1.0, 1.25, 1.5, 1.75, 2.0)])
        gaps = pd.DataFrame.from_records([
            {"system": s, "menu_gap_pct": 1.0, "default_gap_pct": 2.0,
             "best_menu_rule": "x"} for s in ("tested", "untested")])
        got = pol.verdict(solved, gaps, "tested", control="untested",
                          ladder=ladder)
        assert got["holding_moves_with_the_pension"]
        # ...where the share alone would have said the opposite.
        assert not got["equity_moves_with_the_pension"]


class TestTheMenuIsPricedAgainstSomethingItCouldHaveChosen:
    """A solved policy allowed to borrow, beaten against a menu of two
    unlevered funds, prices the drawdown default at the default plus the
    leverage. That is what the first levered run of this section did, and
    the number it produced was the one the paper would have quoted."""

    @staticmethod
    def _score_factory(values):
        def score(system, strategy, plan, equity, domestic):
            return values[(system, strategy, plan)]
        return score

    def _frame(self, column):
        solved = _solved([("s", "constant_percent", 0.1, 1.40, 1.0)])
        solved["unlevered_cec"] = 1.10
        schedules = {"a": (np.ones(3), np.zeros(3))}
        rules = [("default", "planD"), ("amortisation at 6%", "planA")]
        values = {("s", "a", "planD"): 0.50, ("s", "a", "planA"): 1.00}
        return pol.menu_gap(self._score_factory(values), solved, ["a"],
                            schedules, rules, headline_rule="default",
                            column=column).iloc[0]

    def test_it_uses_the_unlevered_solved_policy_by_default(self) -> None:
        row = self._frame("unlevered_cec")
        assert row["menu_gap_pct"] == pytest.approx(10.0)
        assert row["default_gap_pct"] == pytest.approx(120.0)

    def test_the_levered_answer_gives_a_bigger_and_wrong_gap(self) -> None:
        """Kept as the contrast, so the default above is not an arbitrary
        choice of column but the one that answers the question asked."""
        row = self._frame("cec")
        assert row["menu_gap_pct"] == pytest.approx(40.0)
        assert row["menu_gap_pct"] > self._frame("unlevered_cec")["menu_gap_pct"]

    def test_a_missing_unlevered_column_falls_back_rather_than_crashing(
            self) -> None:
        solved = _solved([("s", "constant_percent", 0.1, 1.40, 1.0)])
        got = pol.menu_gap(
            self._score_factory({("s", "a", "planA"): 1.00}), solved, ["a"],
            {"a": (np.ones(3), np.zeros(3))}, [("amortisation at 6%", "planA")],
            column="unlevered_cec")
        assert got["solved_cec"].iloc[0] == pytest.approx(1.40)

    def test_the_verdict_reports_the_borrowing_on_its_own(self) -> None:
        solved = pd.concat([
            _solved([("tested", "constant_percent", 0.1, 1.10, 1.0)],
                    leverage=2.0),
            _solved([("untested", "gompertz", np.nan, 1.30, 1.0)],
                    leverage=1.5)], ignore_index=True)
        solved["unlevered_cec"] = [1.00, 1.20]
        solved["unlevered_rule"] = ["constant_percent", "gompertz"]
        solved["leverage_premium_pct"] = [10.0, 8.3]
        gaps = pd.DataFrame.from_records([
            {"system": s, "menu_gap_pct": 3.0, "default_gap_pct": 100.0,
             "best_menu_rule": "x"} for s in ("tested", "untested")])
        got = pol.verdict(solved, gaps, "tested", control="untested")
        assert got["leverage_premium_under_the_test_pct"] == pytest.approx(
            10.0)
        assert got["leverage_premium_without_it_pct"] == pytest.approx(8.3)
        assert got["solved_rule_survives_switching_borrowing_off"]


class TestTheShippedLadder:
    """What the run in the repository found once the ceiling came off."""

    @staticmethod
    def _table(name: str) -> pd.DataFrame:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / f"{name}.csv"),
                         recursive=True)
        if not hits:
            pytest.skip(f"{name} has not been generated")
        return pd.read_csv(hits[0])

    def test_the_search_was_given_room_above_the_whole_portfolio(self
                                                                 ) -> None:
        """If every solved row sits at leverage one the ladder was never
        exercised, and the censoring objection stands unanswered."""
        solved = self._table("policy_solved")
        assert "leverage" in solved.columns
        assert float(solved["leverage"].max()) > 1.0

    def test_the_two_means_tested_regimes_are_not_the_same_run(self) -> None:
        """They differ in the contribution rate, so a certainty equivalent
        identical to the last digit is a dropped contribution rather than a
        finding. It was, once."""
        solved = self._table("policy_solved").set_index("system")
        pair = ("age_pension_matched", "australia_as_legislated")
        if not all(k in solved.index for k in pair):
            pytest.skip("the legislated regime is not in the solved set")
        a, b = (float(solved.loc[k, "cec"]) for k in pair)
        assert not np.isclose(a, b, rtol=1e-9), (a, b)

    def test_the_ladder_was_actually_swept(self) -> None:
        """One row per (regime, borrowing level). Without it the paper
        cannot say the solved answer sits inside the range rather than at
        its edge, and cannot separate the rule from the borrowing."""
        ladder = self._table("policy_ladder")
        assert {"system", "leverage", "cec"} <= set(ladder.columns)
        for system, part in ladder.groupby("system"):
            assert len(part) >= 2, system
            assert float(part["leverage"].min()) == pytest.approx(1.0), system

    def test_the_menu_gap_is_taken_against_the_unlevered_answer(self) -> None:
        """The menu cannot borrow, so the thing it is beaten against must
        not have been allowed to either."""
        solved = self._table("policy_solved").set_index("system")
        gaps = self._table("policy_menu_gap").set_index("system")
        assert "unlevered_cec" in solved.columns
        for system in gaps.index:
            assert float(gaps.loc[system, "solved_cec"]) == pytest.approx(
                float(solved.loc[system, "unlevered_cec"]), rel=1e-9), system


class TestTheLadderSetsTheResolution:
    """Whether two regimes' allocations differ is judged against what the
    borrowing search can resolve, not against a tolerance picked here. The
    first levered run put the two holdings 0.06 apart on a ladder whose
    rungs are 0.25 wide, which is not a difference this search found."""

    @staticmethod
    def _pair(a: float, b: float, rungs) -> tuple:
        solved = pd.concat([
            _solved([("tested", "constant_percent", 0.1, 1.0, 1.0)],
                    leverage=a),
            _solved([("untested", "gompertz", np.nan, 1.0, 1.0)],
                    leverage=b)], ignore_index=True)
        ladder = pd.DataFrame.from_records(
            [{"system": s, "leverage": r, "cec": 1.0}
             for s in ("tested", "untested") for r in rungs])
        gaps = pd.DataFrame.from_records([
            {"system": s, "menu_gap_pct": 1.0, "default_gap_pct": 2.0,
             "best_menu_rule": "x"} for s in ("tested", "untested")])
        return solved, ladder, gaps

    def test_a_gap_inside_one_rung_is_not_a_difference(self) -> None:
        solved, ladder, gaps = self._pair(
            1.40, 1.33, [1.0, 1.25, 1.5, 1.75, 2.0])
        got = pol.verdict(solved, gaps, "tested", "untested", ladder)
        assert got["ladder_step"] == pytest.approx(0.25)
        assert not got["holding_moves_with_the_pension"]

    def test_a_gap_wider_than_a_rung_is(self) -> None:
        solved, ladder, gaps = self._pair(
            2.00, 1.25, [1.0, 1.25, 1.5, 1.75, 2.0])
        got = pol.verdict(solved, gaps, "tested", "untested", ladder)
        assert got["holding_moves_with_the_pension"]

    def test_a_finer_ladder_can_resolve_what_a_coarse_one_cannot(self
                                                                 ) -> None:
        """The same two holdings, judged against two searches. This is the
        property that makes the threshold a statement about the search
        rather than about the answer."""
        solved, ladder, gaps = self._pair(1.40, 1.33, [1.0, 1.5, 2.0])
        coarse = pol.verdict(solved, gaps, "tested", "untested", ladder)
        solved, fine, gaps = self._pair(
            1.40, 1.33, [1.0, 1.02, 1.04, 1.06, 1.08])
        got = pol.verdict(solved, gaps, "tested", "untested", fine)
        assert not coarse["holding_moves_with_the_pension"]
        assert got["holding_moves_with_the_pension"]

    def test_a_single_rung_search_resolves_nothing(self) -> None:
        solved, ladder, gaps = self._pair(1.0, 2.0, [1.0])
        got = pol.verdict(solved, gaps, "tested", "untested", ladder)
        assert got["ladder_step"] == float("inf")
        assert not got["holding_moves_with_the_pension"]
