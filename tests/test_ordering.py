"""Tests for the strategy-ordering sweep.

The section exists to settle an equivocation between two comparisons that
were being glossed as one -- a strategy gap inside a system, and a level gap
between systems. So most of these check that the module keeps the two apart,
and that it will report a *failure* to recover the ordering as clearly as a
success. A verdict that could only say "yes" would be worthless here.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from src import ordering as odr


def _frame(rows: list) -> pd.DataFrame:
    """`(system, rule, strategy, cec)` tuples as the sweep would emit them."""
    return pd.DataFrame.from_records(
        [{"system": s, "rule": r, "strategy": t, "cec": c,
          "prob_ruin": 0.1, "mean_consumption": 1.0, "p5_consumption": 0.5}
         for s, r, t, c in rows])


CHALLENGER, INCUMBENT = odr.HEADLINE


class TestSweep:
    def test_it_is_the_full_cross_product(self) -> None:
        seen = []

        def simulate(system, rule, strategy):
            seen.append((system, rule, strategy))
            return object()

        frame = odr.sweep(simulate, ["us", "au"], [("a", 1), ("b", 2)],
                          ["x", "y", "z"], lambda o: {"cec": 1.0},
                          log_every=0)
        assert len(frame) == 12 == len(seen)
        assert set(frame["rule"]) == {"a", "b"}

    def test_the_rule_key_is_recorded_not_the_rule_object(self) -> None:
        """The rule objects are unhashable policy instances; the tables key
        on the label, and a table keyed on `<object at 0x...>` would be
        unreadable and unstable between runs."""
        frame = odr.sweep(lambda s, r, t: None, ["us"],
                          [("amortisation at 6%", object())], ["x"],
                          lambda o: {"cec": 1.0}, log_every=0)
        assert list(frame["rule"]) == ["amortisation at 6%"]

    def test_an_empty_grid_gives_an_empty_frame(self) -> None:
        frame = odr.sweep(lambda s, r, t: None, [], [("a", 1)], ["x"],
                          lambda o: {"cec": 1.0}, log_every=0)
        assert not len(frame)


class TestGaps:
    def test_the_gap_is_the_challengers_lead_in_per_cent(self) -> None:
        gapped = odr.gaps(_frame([
            ("us", "fixed", CHALLENGER, 1.10),
            ("us", "fixed", INCUMBENT, 1.00)]))
        assert float(gapped["gap_pct"].iloc[0]) == pytest.approx(10.0)
        assert gapped["leader"].iloc[0] == CHALLENGER

    def test_a_negative_gap_names_the_incumbent(self) -> None:
        gapped = odr.gaps(_frame([
            ("au", "fixed", CHALLENGER, 0.90),
            ("au", "fixed", INCUMBENT, 1.00)]))
        assert float(gapped["gap_pct"].iloc[0]) == pytest.approx(-10.0)
        assert gapped["leader"].iloc[0] == INCUMBENT

    def test_a_gap_inside_the_tie_band_is_a_tie(self) -> None:
        """Two portfolios a hundredth of a per cent apart are not ranked by
        this bootstrap, and calling one a winner would report noise."""
        gapped = odr.gaps(_frame([
            ("us", "fixed", CHALLENGER, 1.00005),
            ("us", "fixed", INCUMBENT, 1.00000)]))
        assert gapped["leader"].iloc[0] == "tie"

    def test_it_also_reports_the_best_of_the_whole_menu(self) -> None:
        gapped = odr.gaps(_frame([
            ("us", "fixed", CHALLENGER, 1.10),
            ("us", "fixed", INCUMBENT, 1.00),
            ("us", "fixed", "sixty_forty", 1.30)]))
        assert gapped["best_strategy"].iloc[0] == "sixty_forty"
        assert float(gapped["best_cec"].iloc[0]) == pytest.approx(1.30)

    def test_a_missing_strategy_is_an_error_not_a_nan(self) -> None:
        """Silently dropping the pair would produce an empty gap column and
        a section that reported nothing while looking like it had."""
        with pytest.raises(ValueError, match="no strategy"):
            odr.gaps(_frame([("us", "fixed", "sixty_forty", 1.0)]))

    def test_an_empty_frame_passes_through(self) -> None:
        assert not len(odr.gaps(pd.DataFrame()))


class TestVerdict:
    @staticmethod
    def _gapped(au_fixed: float, au_amort: float,
                us_fixed: float = 10.0) -> pd.DataFrame:
        return odr.gaps(_frame([
            ("us", "fixed", CHALLENGER, 1.0 + us_fixed / 100),
            ("us", "fixed", INCUMBENT, 1.0),
            ("us", "amort", CHALLENGER, 1.0 + us_fixed / 100),
            ("us", "amort", INCUMBENT, 1.0),
            ("au_as_legislated", "fixed", CHALLENGER, 1.0 + au_fixed / 100),
            ("au_as_legislated", "fixed", INCUMBENT, 1.0),
            ("au_as_legislated", "amort", CHALLENGER, 1.0 + au_amort / 100),
            ("au_as_legislated", "amort", INCUMBENT, 1.0)]))

    def test_it_sees_the_pension_reverse_the_ordering(self) -> None:
        found = odr.verdict(self._gapped(-5.0, -5.0), "fixed")
        assert found["pension_reverses_the_ordering"]
        assert found["baseline_gap_pct"] == pytest.approx(10.0)
        assert found["contender_gap_pct"] == pytest.approx(-5.0)

    def test_it_sees_the_ordering_recovered(self) -> None:
        found = odr.verdict(self._gapped(-5.0, +8.0), "fixed")
        assert found["recovers"]
        assert found["recovering_rule"] == "amort"
        assert found["recovering_gap_pct"] == pytest.approx(8.0)
        assert found["sign_depends_on_the_rule"]

    def test_it_reports_a_failure_to_recover(self) -> None:
        """The branch that matters. If no rule returns the lead, the paper's
        second headline has to be reworded, and a verdict that could not say
        so would let it stand."""
        found = odr.verdict(self._gapped(-5.0, -2.0), "fixed")
        assert not found["recovers"]
        assert "recovering_rule" not in found
        assert not found["sign_depends_on_the_rule"]
        assert found["contender_gap_range_pp"] == pytest.approx(3.0)

    def test_it_names_the_best_and_worst_rule_either_way(self) -> None:
        found = odr.verdict(self._gapped(-5.0, -2.0), "fixed")
        assert found["contender_best_rule"] == "amort"
        assert found["contender_worst_rule"] == "fixed"

    def test_it_notices_a_third_portfolio_winning(self) -> None:
        frame = _frame([
            ("us", "fixed", CHALLENGER, 1.10),
            ("us", "fixed", INCUMBENT, 1.00),
            ("au_as_legislated", "fixed", CHALLENGER, 0.95),
            ("au_as_legislated", "fixed", INCUMBENT, 1.00),
            ("au_as_legislated", "fixed", "sixty_forty", 1.40)])
        found = odr.verdict(odr.gaps(frame), "fixed")
        assert found["a_third_portfolio_ever_wins"]
        assert found["contender_best_strategies"]["fixed"] == "sixty_forty"

    def test_the_pair_winning_everywhere_is_reported_as_such(self) -> None:
        found = odr.verdict(self._gapped(-5.0, +8.0), "fixed")
        assert not found["a_third_portfolio_ever_wins"]

    def test_a_missing_baseline_rule_is_not_measured(self) -> None:
        found = odr.verdict(self._gapped(-5.0, +8.0), "no_such_rule")
        assert found == {"measured": False}

    def test_an_empty_frame_is_not_measured(self) -> None:
        assert odr.verdict(pd.DataFrame(), "fixed") == {"measured": False}


class TestBySystem:
    def test_it_pivots_rules_down_and_systems_across(self) -> None:
        wide = odr.by_system(TestVerdict._gapped(-5.0, +8.0))
        assert set(wide.columns) >= {"rule", "us", "au_as_legislated"}
        assert len(wide) == 2


class TestRuleOrder:
    """Rules are swept in a meaningful order; the table must keep it.

    Sorted as strings, "amortisation at 10%" falls between 0% and 2%, so a
    column whose whole point is the trend across assumed returns reads
    0, 10, 2, 4, 6, 8.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        rows = []
        for rule in ("fixed real rule", "amortisation at 0%",
                     "amortisation at 2%", "amortisation at 10%"):
            for strategy, cec in ((CHALLENGER, 1.1), (INCUMBENT, 1.0)):
                rows.append(("us", rule, strategy, cec))
        return _frame(rows)

    def test_the_sweep_order_is_kept(self) -> None:
        out = odr.gaps(self._frame())
        assert list(out["rule"]) == ["fixed real rule", "amortisation at 0%",
                                     "amortisation at 2%",
                                     "amortisation at 10%"]

    def test_the_rule_column_is_still_plain_strings(self) -> None:
        """A categorical leaks into `to_csv` and into every consumer that
        groups on it, so it is cast back once the sort is done."""
        out = odr.gaps(self._frame())
        assert not isinstance(out["rule"].dtype, pd.CategoricalDtype)
        assert all(isinstance(x, str) for x in out["rule"])


class TestInfluence:
    """Delete-one-country recomputation of the whole gap table."""

    def test_it_runs_once_per_country_with_that_one_removed(self) -> None:
        seen = []

        def gaps_for(kept):
            seen.append(tuple(kept))
            return pd.DataFrame([{"system": "us", "rule": "r",
                                  "gap_pct": 1.0}])

        out = odr.influence(gaps_for, ["AUS", "USA", "GBR"])
        assert len(seen) == 3
        assert all(len(k) == 2 for k in seen)
        assert ("USA", "GBR") in seen and ("AUS", "GBR") in seen

    def test_the_dropped_country_is_tagged_on_every_row(self) -> None:
        out = odr.influence(
            lambda kept: pd.DataFrame([{"system": "us", "rule": "a",
                                        "gap_pct": 1.0},
                                       {"system": "us", "rule": "b",
                                        "gap_pct": 2.0}]),
            ["AUS", "USA"])
        assert len(out) == 4
        assert set(out["dropped"]) == {"AUS", "USA"}
        assert set(out["n_markets"]) == {1}

    def test_no_countries_gives_an_empty_frame(self) -> None:
        assert not len(odr.influence(lambda kept: pd.DataFrame(), []))


class TestIntervals:
    @staticmethod
    def _pair(values, point):
        """`values` are the per-deletion gaps for one cell."""
        infl = pd.DataFrame([{"dropped": f"C{i}", "system": "au",
                              "rule": "fixed", "gap_pct": v}
                             for i, v in enumerate(values)])
        gapped = pd.DataFrame([{"system": "au", "rule": "fixed",
                                "gap_pct": point}])
        return infl, gapped

    def test_a_tight_cluster_away_from_zero_is_resolved(self) -> None:
        out = odr.intervals(*self._pair([-2.0, -2.1, -1.9, -2.05], -2.0))
        row = out.iloc[0]
        assert row["sign_survives_every_deletion"]
        assert row["ci_excludes_zero"]
        assert row["standard_error"] < 1.0

    def test_a_spread_straddling_zero_is_not_resolved(self) -> None:
        """The branch the paper's headline turns on."""
        out = odr.intervals(*self._pair([-2.0, 4.0, -6.0, 3.0], -2.0))
        row = out.iloc[0]
        assert not row["sign_survives_every_deletion"]
        assert not row["ci_excludes_zero"]

    def test_a_consistent_sign_can_still_be_unresolved(self) -> None:
        """Every deletion negative, but so widely spread that the interval
        crosses zero. Reporting only the sign check would call this settled."""
        out = odr.intervals(*self._pair([-0.5, -9.0, -0.2, -8.0], -2.0))
        row = out.iloc[0]
        assert row["sign_survives_every_deletion"]
        assert not row["ci_excludes_zero"]

    def test_it_reports_the_raw_deletion_span(self) -> None:
        out = odr.intervals(*self._pair([-2.0, -5.0, -1.0], -2.0))
        assert float(out["loo_low"].iloc[0]) == pytest.approx(-5.0)
        assert float(out["loo_high"].iloc[0]) == pytest.approx(-1.0)
        assert int(out["deletions"].iloc[0]) == 3

    def test_the_point_estimate_is_the_full_panel_not_the_mean(self) -> None:
        """The jackknife is centred on the full-panel gap; using the mean of
        the deletions instead would quietly report a different number from
        the one in the headline table."""
        out = odr.intervals(*self._pair([-3.0, -3.0, -3.0], -2.0))
        assert float(out["gap_pct"].iloc[0]) == pytest.approx(-2.0)

    def test_empty_inputs_give_an_empty_frame(self) -> None:
        assert not len(odr.intervals(pd.DataFrame(), pd.DataFrame()))


class TestPrecisionVerdict:
    @staticmethod
    def _table(au_values, us_values, au_point=-2.0, us_point=11.0):
        rows = []
        for system, values, point in (("australia_as_legislated", au_values,
                                       au_point),
                                      ("us_social_security", us_values,
                                       us_point)):
            rows += [{"dropped": f"C{i}", "system": system, "rule": "fixed",
                      "gap_pct": v} for i, v in enumerate(values)]
        gapped = pd.DataFrame([
            {"system": "australia_as_legislated", "rule": "fixed",
             "gap_pct": au_point},
            {"system": "us_social_security", "rule": "fixed",
             "gap_pct": us_point}])
        return odr.intervals(pd.DataFrame(rows), gapped)

    def test_a_resolved_reversal_is_reported_as_resolved(self) -> None:
        found = odr.precision_verdict(
            self._table([-2.0, -2.1, -1.9], [11.0, 11.1, 10.9]), "fixed")
        assert found["reversal_resolved"]
        assert found["baseline_resolved"]
        assert found["unresolved"] == []

    def test_an_unresolved_reversal_is_reported_as_unresolved(self) -> None:
        found = odr.precision_verdict(
            self._table([-2.0, 5.0, -8.0], [11.0, 11.1, 10.9]), "fixed")
        assert not found["reversal_resolved"]
        assert found["baseline_resolved"]
        assert "australia_as_legislated / fixed" in found["unresolved"]
        assert found["resolved_cells"] == 1

    def test_a_missing_cell_is_not_measured(self) -> None:
        table = self._table([-2.0, -2.1], [11.0, 11.1])
        assert odr.precision_verdict(table, "no_such_rule") == {
            "measured": False}

    def test_an_empty_table_is_not_measured(self) -> None:
        assert odr.precision_verdict(pd.DataFrame(), "fixed") == {
            "measured": False}
