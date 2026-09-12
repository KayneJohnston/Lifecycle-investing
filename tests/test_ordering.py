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


class TestDifferenceIntervals:
    """The paper's surviving claim is a difference between two cells, and
    an interval on two levels is not an interval on their difference. The
    deletions are shared, so the difference gets a paired jackknife."""

    @staticmethod
    def _inputs(base, other, base_point=-2.0, other_point=26.0):
        infl = pd.DataFrame(
            [{"dropped": f"C{i}", "system": "au", "rule": "fixed",
              "gap_pct": v} for i, v in enumerate(base)]
            + [{"dropped": f"C{i}", "system": "au", "rule": "amort",
                "gap_pct": v} for i, v in enumerate(other)])
        gapped = pd.DataFrame([
            {"system": "au", "rule": "fixed", "gap_pct": base_point},
            {"system": "au", "rule": "amort", "gap_pct": other_point}])
        return infl, gapped

    def test_the_difference_is_the_difference_of_the_points(self) -> None:
        out = odr.difference_intervals(
            *self._inputs([-2.0, -3.0, -1.0], [26.0, 25.0, 27.0]),
            reference_rule="fixed", system="au")
        assert float(out["difference_pp"].iloc[0]) == pytest.approx(28.0)

    def test_perfectly_correlated_cells_leave_no_difference_error(self
                                                                   ) -> None:
        """When two cells move together across deletions the difference is
        constant, and the paired jackknife should see that even though each
        level is noisy on its own."""
        out = odr.difference_intervals(
            *self._inputs([-2.0, -8.0, 4.0], [26.0, 20.0, 32.0]),
            reference_rule="fixed", system="au")
        assert float(out["standard_error"].iloc[0]) == pytest.approx(0.0,
                                                                     abs=1e-9)
        assert bool(out["ci_excludes_zero"].iloc[0])
        assert float(out["correlation"].iloc[0]) == pytest.approx(1.0)

    def test_uncorrelated_cells_leave_the_difference_noisy(self) -> None:
        out = odr.difference_intervals(
            *self._inputs([-2.0, -8.0, 4.0], [26.0, 32.0, 20.0]),
            reference_rule="fixed", system="au")
        assert float(out["standard_error"].iloc[0]) > 5.0
        assert float(out["correlation"].iloc[0]) < 0.0

    def test_the_reference_rule_is_not_compared_with_itself(self) -> None:
        out = odr.difference_intervals(
            *self._inputs([-2.0, -3.0], [26.0, 25.0]),
            reference_rule="fixed", system="au")
        assert "fixed" not in set(out["rule"])
        assert list(out["rule"]) == ["amort"]

    def test_a_missing_reference_rule_gives_nothing(self) -> None:
        out = odr.difference_intervals(
            *self._inputs([-2.0, -3.0], [26.0, 25.0]),
            reference_rule="no_such_rule", system="au")
        assert not len(out)

    def test_a_missing_system_gives_nothing(self) -> None:
        out = odr.difference_intervals(
            *self._inputs([-2.0, -3.0], [26.0, 25.0]),
            reference_rule="fixed", system="no_such_system")
        assert not len(out)


class TestDifferenceVerdict:
    def test_it_names_the_widest_effect(self) -> None:
        table = odr.difference_intervals(
            *TestDifferenceIntervals._inputs([-2.0, -3.0, -1.0],
                                             [26.0, 25.0, 27.0]),
            reference_rule="fixed", system="au")
        found = odr.difference_verdict(table)
        assert found["widest_rule"] == "amort"
        assert found["widest_pp"] == pytest.approx(28.0)
        assert found["all_resolved"]

    def test_an_unresolved_difference_is_listed(self) -> None:
        table = odr.difference_intervals(
            *TestDifferenceIntervals._inputs([-2.0, -20.0, 18.0],
                                             [26.0, 40.0, 6.0], -2.0, 1.0),
            reference_rule="fixed", system="au")
        found = odr.difference_verdict(table)
        assert not found["all_resolved"]
        assert "amort" in found["unresolved"]

    def test_an_empty_table_is_not_measured(self) -> None:
        assert odr.difference_verdict(pd.DataFrame()) == {"measured": False}


class TestGammaVerdict:
    @staticmethod
    def _frame(values):
        return pd.DataFrame([{"system": "au", "rule": "fixed",
                              "gap_pct": v, "gamma": g}
                             for g, v in values])

    def test_a_stable_sign_is_reported_as_stable(self) -> None:
        found = odr.gamma_verdict(self._frame([(2.0, -1.0), (5.0, -2.0),
                                               (10.0, -4.0)]), "fixed", "au")
        assert found["sign_holds"]
        assert found["spread_pp"] == pytest.approx(3.0)

    def test_a_sign_that_flips_with_gamma_is_flagged(self) -> None:
        """A sign that depends on the curvature of the objective is a
        statement about the objective, not about the pension."""
        found = odr.gamma_verdict(self._frame([(2.0, 4.0), (5.0, -2.0),
                                               (10.0, -9.0)]), "fixed", "au")
        assert not found["sign_holds"]

    def test_one_risk_aversion_is_not_a_check(self) -> None:
        assert odr.gamma_verdict(self._frame([(5.0, -2.0)]), "fixed",
                                 "au") == {"measured": False}


class TestThePseudoValues:
    """A jackknife standard error assumes the statistic is close to linear
    in the units deleted, so the sixteen sub-panel estimates should scatter
    around the full-sample one. Whether they do is a fact about the data.
    Reporting it matters because an interval straddling zero reads as
    imprecision, and deletions that nearly all lie on one side of the point
    estimate say something more specific."""

    @staticmethod
    def _frames(values, point, system="au", rule="fixed"):
        influence = pd.DataFrame({
            "system": [system] * len(values), "rule": [rule] * len(values),
            "dropped": [f"C{i}" for i in range(len(values))],
            "gap_pct": list(values)})
        gapped = pd.DataFrame({"system": [system], "rule": [rule],
                               "gap_pct": [point]})
        return influence, gapped

    def test_a_symmetric_jackknife_has_no_bias(self) -> None:
        influence, gapped = self._frames([-1.0, 1.0, -2.0, 2.0], 0.0)
        row = odr.pseudo_values(influence, gapped).iloc[0]
        assert row["bias_estimate"] == pytest.approx(0.0)
        assert row["below_point"] == 2

    def test_the_bias_is_the_classical_one(self) -> None:
        """``(n-1)(mean of deletions - point)``, by hand."""
        influence, gapped = self._frames([2.0, 4.0, 6.0], 1.0)
        row = odr.pseudo_values(influence, gapped).iloc[0]
        assert row["bias_estimate"] == pytest.approx(2 * (4.0 - 1.0))
        assert row["below_point"] == 0

    def test_it_scales_the_bias_by_the_estimate(self) -> None:
        influence, gapped = self._frames([2.0, 4.0, 6.0], 1.0)
        row = odr.pseudo_values(influence, gapped).iloc[0]
        assert row["bias_over_point"] == pytest.approx(6.0)

    def test_a_cell_with_one_deletion_is_skipped(self) -> None:
        influence, gapped = self._frames([1.0], 1.0)
        assert not len(odr.pseudo_values(influence, gapped))

    def test_a_cell_the_point_estimates_do_not_carry_is_skipped(self) -> None:
        influence, _ = self._frames([1.0, 2.0], 1.0)
        gapped = pd.DataFrame({"system": ["other"], "rule": ["other"],
                               "gap_pct": [1.0]})
        assert not len(odr.pseudo_values(influence, gapped))

    def test_empty_in_empty_out(self) -> None:
        assert not len(odr.pseudo_values(pd.DataFrame(), pd.DataFrame()))


class TestWhetherTheBiasIsIsolated:
    """A diagnostic that fires on every cell is a property of the method.
    The claim worth making is comparative: well behaved everywhere except
    where the contested sign lives."""

    @staticmethod
    def _table(rows):
        return pd.DataFrame.from_records([
            {"system": s, "rule": r, "point": p, "deletions": 4,
             "loo_mean": m, "loo_sd": 1.0,
             "below_point": b, "bias_estimate": 3 * (m - p),
             "bias_over_point": abs(3 * (m - p)) / abs(p)}
            for s, r, p, m, b in rows])

    def test_one_misbehaving_cell_among_well_behaved_ones_is_isolated(self
                                                                      ) -> None:
        table = self._table([("au", "fixed", -2.0, 0.5, 1),
                             ("au", "amort", 20.0, 20.1, 2),
                             ("us", "fixed", 11.0, 10.9, 2)])
        found = odr.bias_verdict(table, "fixed", "au")
        assert found["isolated"]
        assert found["mean_deletion_flips_sign"]

    def test_a_cell_no_worse_than_its_neighbours_is_not(self) -> None:
        table = self._table([("au", "fixed", -2.0, -2.1, 2),
                             ("au", "amort", 20.0, 21.0, 2)])
        found = odr.bias_verdict(table, "fixed", "au")
        assert not found["isolated"]

    def test_a_mean_that_keeps_the_sign_does_not_flip_it(self) -> None:
        table = self._table([("au", "fixed", -2.0, -8.0, 0),
                             ("au", "amort", 20.0, 20.1, 2)])
        found = odr.bias_verdict(table, "fixed", "au")
        assert not found["mean_deletion_flips_sign"]

    def test_it_reports_the_spread_of_the_other_cells(self) -> None:
        table = self._table([("au", "fixed", -2.0, 0.5, 1),
                             ("au", "amort", 20.0, 20.1, 3),
                             ("us", "fixed", 11.0, 10.9, 2)])
        found = odr.bias_verdict(table, "fixed", "au")
        assert found["others_below_low"] == 2
        assert found["others_below_high"] == 3

    def test_a_cell_that_is_not_there_is_not_measured(self) -> None:
        table = self._table([("au", "amort", 20.0, 20.1, 2)])
        assert odr.bias_verdict(table, "fixed", "au") == {"measured": False}

    def test_empty_in_not_measured_out(self) -> None:
        assert odr.bias_verdict(pd.DataFrame(), "fixed", "au") == {
            "measured": False}


class TestTheSecondObjective:
    """Section 9 rejects a fixed retirement horizon as not neutral between
    rules; Section 10 compares portfolios within a rule. Whether the gap
    survives the change of objective is a fact about the grid, and these
    tests hold the machinery that measures it to cases whose answer is
    known by construction."""

    @staticmethod
    def _swept(rows):
        """``rows`` is ``(system, rule, strategy, cec, cec_survival)``."""
        return pd.DataFrame.from_records([
            {"system": s, "rule": r, "strategy": t,
             "cec": c, "cec_survival": v}
            for s, r, t, c, v in rows])

    def test_it_returns_a_gap_under_each_objective(self) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
            ("au", "fixed", "target_date_fund", 0.8, 1.6),
        ])
        table = odr.by_objective(swept)
        assert set(table["objective"]) == {"cec", "cec_survival"}
        # 1.0/0.8 and 2.0/1.6 are the same ratio, so the same gap.
        assert table["gap_pct"].nunique() == 1

    def test_an_objective_the_grid_lacks_is_skipped(self) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
            ("au", "fixed", "target_date_fund", 0.8, 1.6),
        ]).drop(columns=["cec_survival"])
        table = odr.by_objective(swept)
        assert set(table["objective"]) == {"cec"}

    def test_a_gap_that_does_not_move_is_insulated(self) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
            ("au", "fixed", "target_date_fund", 0.8, 1.6),
        ])
        found = odr.objective_verdict(odr.by_objective(swept), "fixed", "au")
        assert found["insulated"]
        assert found["worst_move_pp"] == pytest.approx(0.0, abs=1e-9)
        assert found["contested_sign_holds"]

    def test_a_sign_that_flips_is_not(self) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.05, 0.95),
            ("au", "fixed", "target_date_fund", 1.00, 1.00),
        ])
        found = odr.objective_verdict(odr.by_objective(swept), "fixed", "au")
        assert not found["insulated"]
        assert found["signs_flipped"] == 1
        assert found["flipped_cells"] == ["au / fixed"]
        assert not found["contested_sign_holds"]

    def test_a_move_beyond_the_tolerance_is_not_insulated(self) -> None:
        """Same sign, but the gap moves ten points."""
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.10, 1.20),
            ("au", "fixed", "target_date_fund", 1.00, 1.00),
        ])
        found = odr.objective_verdict(odr.by_objective(swept), "fixed", "au")
        assert found["signs_flipped"] == 0
        assert not found["insulated"]
        assert found["worst_move_pp"] == pytest.approx(10.0)

    def test_it_names_the_cell_that_moves_most(self) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
            ("au", "fixed", "target_date_fund", 0.8, 1.6),
            ("au", "amort", "balanced_all_equity", 1.10, 1.20),
            ("au", "amort", "target_date_fund", 1.00, 1.00),
        ])
        found = odr.objective_verdict(odr.by_objective(swept), "fixed", "au")
        assert found["worst_cell"] == ["au", "amort"]

    def test_nothing_measured_is_said_so(self) -> None:
        assert odr.objective_verdict(pd.DataFrame(), "fixed", "au") == {
            "measured": False}

    def test_the_levels_are_reported_separately(self) -> None:
        """The insulation argument needs both halves: gaps that hold still
        while levels move is evidence; both holding still is not."""
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
            ("au", "fixed", "target_date_fund", 0.8, 1.6),
        ])
        found = odr.level_shift(swept)
        assert found["median_level_shift_pct"] == pytest.approx(100.0)

    def test_a_grid_without_the_second_objective_reports_nothing(self
                                                                 ) -> None:
        swept = self._swept([
            ("au", "fixed", "balanced_all_equity", 1.0, 2.0),
        ]).drop(columns=["cec_survival"])
        assert odr.level_shift(swept) == {"measured": False}


class TestTheRecommendedRule:
    """A whole section upstream exists to choose a withdrawal rule. Quoting
    the best cell of this grid instead makes that choice decorative and the
    headline flattering, so the two studies are joined -- and because they
    spell the same policy differently and always have, the join is the part
    that can rot."""

    RANKING = pd.DataFrame({
        "rule_label": ["amortisation (6% assumed return)",
                       "amortisation (8% assumed return)",
                       "constant percent"],
        "rank_mortality": [1, 2, 3]})
    GAPS = pd.DataFrame({
        "system": ["au"] * 4,
        "rule": ["fixed_real_rule", "amortisation at 6%",
                 "amortisation at 8%", "constant_percent at 4%"],
        "gap_pct": [-2.08, 24.87, 26.63, 18.43]})

    def test_it_finds_the_rule_the_longevity_study_ranks_first(self) -> None:
        found = odr.recommended_rule(self.RANKING, self.GAPS, "au")
        assert found["rule"] == "amortisation at 6%"
        assert found["gap_pct"] == pytest.approx(24.87)

    def test_it_reports_the_maximum_separately(self) -> None:
        """Both numbers, so the paper can lead with one and give the other
        rather than choosing which to omit."""
        found = odr.recommended_rule(self.RANKING, self.GAPS, "au")
        assert found["best_rule"] == "amortisation at 8%"
        assert found["best_gap_pct"] == pytest.approx(26.63)
        assert not found["is_best"]

    def test_it_says_so_when_the_recommendation_is_the_maximum(self) -> None:
        ranking = self.RANKING.assign(rank_mortality=[2, 1, 3])
        found = odr.recommended_rule(ranking, self.GAPS, "au")
        assert found["rule"] == "amortisation at 8%"
        assert found["is_best"]

    def test_the_join_matches_on_the_rate_not_the_wording(self) -> None:
        """`amortisation (6% assumed return)` against `amortisation at 6%`.
        Matching on the family alone would take whichever rate came first."""
        found = odr.recommended_rule(self.RANKING, self.GAPS, "au")
        assert "6%" in found["rule"]

    def test_a_join_that_resolves_nothing_is_reported(self) -> None:
        """Silently finding nothing would send the paper back to quoting
        the maximum, which is the error this exists to prevent."""
        ranking = pd.DataFrame({"rule_label": ["gompertz"],
                                "rank_mortality": [1]})
        assert odr.recommended_rule(ranking, self.GAPS, "au") == {
            "measured": False}

    def test_a_rate_the_grid_does_not_carry_is_not_silently_swapped(self
                                                                    ) -> None:
        ranking = pd.DataFrame({
            "rule_label": ["amortisation (7% assumed return)"],
            "rank_mortality": [1]})
        assert not odr.recommended_rule(ranking, self.GAPS,
                                        "au")["measured"]

    def test_a_system_the_grid_does_not_carry_is_reported(self) -> None:
        assert odr.recommended_rule(self.RANKING, self.GAPS,
                                    "nowhere") == {"measured": False}

    def test_empty_in_not_measured_out(self) -> None:
        assert odr.recommended_rule(pd.DataFrame(), pd.DataFrame(),
                                    "au") == {"measured": False}

    def test_the_real_tables_still_join(self) -> None:
        """The guard that matters: the two studies' labels agree today. If
        either renames its rules this fails here rather than in the paper."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "results/tables"
        if not (root / "longevity_ranking.csv").exists():
            pytest.skip("the longevity study has not been run")
        found = odr.recommended_rule(
            pd.read_csv(root / "longevity_ranking.csv"),
            pd.read_csv(root / "ordering_gaps.csv"),
            "australia_as_legislated")
        assert found["measured"], "the two studies no longer name the same rule"


class TestTheSignSplit:
    """Where the deletions sit against *zero*, which is not where they sit
    against the point estimate.

    The bug this exists to prevent: a draft read a skewed ``below_point``
    count as evidence that essentially no fifteen-country subsample
    reproduced the paper's reversal. Half of them did. The two counts
    answer different questions and the paper needs the one about the sign.
    """

    #: Point -2. Four deletions below it, and yet six of ten still negative
    #: -- exactly the shape that misled the draft.
    GAPS = pd.DataFrame([{"system": "au", "rule": "fixed_real", "gap_pct": -2.0}])
    INFLUENCE = pd.DataFrame([
        {"dropped": d, "system": "au", "rule": "fixed_real", "gap_pct": v}
        for d, v in (("AAA", -5.0), ("BBB", -4.0), ("CCC", -3.0),
                     ("DDD", -2.5), ("EEE", -1.5), ("FFF", -0.5),
                     ("GGG", 0.5), ("HHH", 1.0), ("III", 3.0),
                     ("JJJ", 9.0))])

    def _found(self):
        return odr.sign_split(self.INFLUENCE, self.GAPS, "fixed_real", "au")

    def test_it_counts_the_deletions_that_keep_the_sign(self) -> None:
        found = self._found()
        assert found["deletions"] == 10
        assert found["sign_holds"] == 6
        assert found["sign_flips"] == 4
        assert found["holds_share"] == pytest.approx(0.6)
        assert not found["unanimous"]

    def test_it_disagrees_with_the_below_point_count(self) -> None:
        """The whole reason it exists. Four deletions fall below the point
        estimate and six keep its sign; a paper that reads the first as the
        second says the opposite of the truth."""
        found = self._found()
        pseudo = odr.pseudo_values(self.INFLUENCE, self.GAPS)
        assert int(pseudo.iloc[0]["below_point"]) == 4
        assert found["sign_holds"] == 6

    def test_it_names_the_country_that_moves_it_furthest(self) -> None:
        found = self._found()
        assert found["largest_flip"] == "JJJ"
        assert found["largest_flip_swing"] == pytest.approx(11.0)
        assert found["largest_hold"] == "AAA"
        assert found["largest_hold_swing"] == pytest.approx(-3.0)

    def test_the_two_lists_partition_the_deletions(self) -> None:
        found = self._found()
        assert set(found["flippers"]) & set(found["holders"]) == set()
        assert len(found["flippers"]) + len(found["holders"]) == 10

    def test_a_positive_cell_reads_the_extremes_the_other_way(self) -> None:
        """The mover that matters is the one heading *toward* zero, which
        is the low end for a positive cell and the high end for a negative
        one."""
        gaps = pd.DataFrame([{"system": "au", "rule": "amort",
                              "gap_pct": 20.0}])
        inf = pd.DataFrame([
            {"dropped": d, "system": "au", "rule": "amort", "gap_pct": v}
            for d, v in (("AAA", -1.0), ("BBB", 18.0), ("CCC", 30.0))])
        found = odr.sign_split(inf, gaps, "amort", "au")
        assert found["sign_holds"] == 2 and found["sign_flips"] == 1
        assert found["largest_flip"] == "AAA"
        assert found["largest_hold"] == "CCC"

    def test_a_unanimous_cell_says_so(self) -> None:
        gaps = pd.DataFrame([{"system": "au", "rule": "amort",
                              "gap_pct": 20.0}])
        inf = pd.DataFrame([
            {"dropped": d, "system": "au", "rule": "amort", "gap_pct": v}
            for d, v in (("AAA", 18.0), ("BBB", 21.0), ("CCC", 25.0))])
        found = odr.sign_split(inf, gaps, "amort", "au")
        assert found["unanimous"] and found["sign_flips"] == 0
        assert found["flippers"] == []

    def test_a_missing_cell_is_not_measured(self) -> None:
        assert not odr.sign_split(self.INFLUENCE, self.GAPS,
                                  "nope", "au").get("measured")
        assert not odr.sign_split(pd.DataFrame(), pd.DataFrame(),
                                  "fixed_real", "au").get("measured")

    def test_a_zero_point_is_not_measured(self) -> None:
        """A sign split around a cell with no sign is not a quantity."""
        gaps = pd.DataFrame([{"system": "au", "rule": "r", "gap_pct": 0.0}])
        inf = pd.DataFrame([{"dropped": "AAA", "system": "au", "rule": "r",
                             "gap_pct": 1.0}])
        assert not odr.sign_split(inf, gaps, "r", "au").get("measured")

    def test_the_real_contested_cell_is_split_in_half(self) -> None:
        """The correction itself, on the live tables. If a rerun ever makes
        the reversal unanimous the paper's wording has to change again, and
        this is where that surfaces."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "results/tables"
        if not (root / "ordering_influence.csv").exists():
            pytest.skip("the ordering study has not been run")
        found = odr.sign_split(
            pd.read_csv(root / "ordering_influence.csv"),
            pd.read_csv(root / "ordering_gaps.csv"),
            "fixed_real_rule", "australia_as_legislated")
        assert found["measured"]
        assert found["sign_flips"] > 0, (
            "the reversal now survives every deletion; the paper says it "
            "does not")
        assert found["sign_holds"] > 0, (
            "no sub-panel reproduces the reversal; the paper says half do")
