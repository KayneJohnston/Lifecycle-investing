"""Tests for the three-way sweep under an uncertain horizon.

The section exists because a fixed death age is not neutral between
withdrawal rules, so most of these check that the machinery can *tell the
two objectives apart* -- a bug that silently scored both the same way would
produce a section concluding, plausibly and wrongly, that nothing changes.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from src import longevity as lv


class TestCombination:
    def test_a_rate_rule_carries_its_rate_into_the_label(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1,
                               rule="constant_real", rate=0.04)
        assert "4.0%" in combo.label()
        assert "100% equity" in combo.label()

    def test_a_horizon_rule_has_no_rate_in_its_label(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1, rule="gompertz")
        assert "%" in combo.label()          # the allocation still shows
        assert "@" not in combo.label()      # but no rate does

    def test_a_variant_is_named_by_its_suffix(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1,
                               rule="guyton_klinger", rate=0.04,
                               suffix="tight band")
        assert combo.rule_label == "guyton_klinger (tight band)"

    def test_it_builds_the_rule_it_names(self) -> None:
        from src import spending as spg

        built = lv.Combination(equity=1.0, domestic=0.1,
                               rule="constant_real", rate=0.045).build()
        assert isinstance(built, spg.ConstantRealRule)
        assert built.rate == pytest.approx(0.045)

    def test_params_reach_the_rule(self) -> None:
        built = lv.Combination(equity=1.0, domestic=0.1, rule="gompertz",
                               params={"buffer_years": 5.0}).build()
        assert built.buffer_years == pytest.approx(5.0)


class TestGrids:
    def test_the_allocation_grid_is_the_cross_product(self) -> None:
        grid = lv.allocation_grid([0.8, 1.0], [0.0, 0.1, 0.2])
        assert len(grid) == 6
        assert (1.0, 0.2) in grid

    def test_only_rate_rules_are_crossed_with_the_rates(self) -> None:
        """A rule that derives its level from a planning horizon has no rate
        to sweep, and crossing it with one would score the same policy many
        times and let it win on repetition."""
        specs = [{"key": "constant_real"}, {"key": "gompertz"}]
        plans = lv.plan_grid(specs, [0.03, 0.04, 0.05])
        rates = [r for k, r, _, _ in plans if k == "constant_real"]
        horizon = [r for k, r, _, _ in plans if k == "gompertz"]
        assert rates == [0.03, 0.04, 0.05]
        assert horizon == [None]

    def test_variants_keep_their_params_and_suffix(self) -> None:
        specs = [{"key": "gompertz", "params": {"buffer_years": 5.0},
                  "suffix": "+5y buffer"}]
        (key, rate, params, suffix), = lv.plan_grid(specs, [0.04])
        assert key == "gompertz" and rate is None
        assert params == {"buffer_years": 5.0}
        assert suffix == "+5y buffer"


def _outcome(consumption: float, ruin: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        consumption=np.full((4, 10), float(consumption)),
        ruin=np.full(4, float(ruin)))


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"equity": e, "domestic": d, "rule": r, "rule_label": r,
         "rate": rate, "has_rate": not pd.isna(rate),
         "assumed_return": np.nan, "has_assumed_return": False,
         "label": f"{r}-{e}-{d}-{rate}",
         lv.FIXED: cf, lv.MORTALITY: cm,
         "ruin_fixed": rf, "ruin_mortality": rm, "mean_consumption": 1.0}
        for e, d, r, rate, cf, cm, rf, rm in rows])


def _dialled(rows) -> pd.DataFrame:
    """Rows for a rule levelled by an assumed return rather than a rate."""
    return pd.DataFrame([
        {"equity": e, "domestic": d, "rule": r,
         "rule_label": f"{r} ({100 * ret:g}% assumed return)",
         "rate": np.nan, "has_rate": False,
         "assumed_return": ret, "has_assumed_return": True,
         "label": f"{r}-{e}-{d}-{ret}",
         lv.FIXED: cf, lv.MORTALITY: cm,
         "ruin_fixed": rf, "ruin_mortality": rm, "mean_consumption": 1.0}
        for e, d, r, ret, cf, cm, rf, rm in rows])


class TestSweep:
    def test_both_objectives_come_off_one_outcome(self) -> None:
        """The difference between the two scores has to be the aggregation
        and nothing else, so the sweep must not simulate twice."""
        calls = []

        def simulate(combo):
            calls.append(combo)
            return _outcome(1.0, 0.1)

        combos = [lv.Combination(equity=1.0, domestic=0.1,
                                 rule="constant_real", rate=0.04),
                  lv.Combination(equity=0.8, domestic=0.1, rule="gompertz")]
        frame = lv.sweep(simulate, lambda o: 1.0, lambda o: 0.9,
                         lambda o: 0.05, combos, log_every=0)
        assert len(calls) == 2
        assert len(frame) == 2
        assert set(frame[lv.FIXED]) == {1.0}
        assert set(frame[lv.MORTALITY]) == {0.9}

    def test_a_horizon_rule_records_no_rate(self) -> None:
        frame = lv.sweep(lambda c: _outcome(1.0, 0.0), lambda o: 1.0,
                         lambda o: 1.0, lambda o: 0.0,
                         [lv.Combination(equity=1.0, domestic=0.1,
                                         rule="gompertz")], log_every=0)
        assert not bool(frame["has_rate"].iloc[0])
        assert np.isnan(frame["rate"].iloc[0])


class TestOptimumAndRanking:
    ROWS = [
        (1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.12, 0.05),
        (1.0, 0.1, "constant_real", 0.05, 0.95, 0.85, 0.20, 0.09),
        (1.0, 0.1, "gompertz", np.nan, 0.90, 0.95, 0.00, 0.00),
        (0.8, 0.1, "gompertz", np.nan, 0.88, 0.92, 0.00, 0.00),
    ]

    def test_each_objective_can_pick_a_different_winner(self) -> None:
        frame = _frame(self.ROWS)
        assert lv.optimum(frame, lv.FIXED)["rule"] == "constant_real"
        assert lv.optimum(frame, lv.MORTALITY)["rule"] == "gompertz"

    def test_by_objective_reports_both_winners(self) -> None:
        table = lv.by_objective(_frame(self.ROWS))
        assert list(table["objective"]) == ["fixed horizon",
                                            "survival-weighted"]
        assert table["rule"].tolist() == ["constant_real", "gompertz"]

    def test_each_rule_is_ranked_at_its_own_best_settings(self) -> None:
        """Otherwise a rule is penalised for a rate chosen to suit a
        different horizon, which is the comparison this section exists to
        avoid making."""
        shift = lv.ranking_shift(_frame(self.ROWS))
        row = shift[shift["rule_label"] == "constant_real"].iloc[0]
        assert row[lv.FIXED] == pytest.approx(1.00)   # its best, not its last
        assert row[lv.MORTALITY] == pytest.approx(0.85)

    def test_the_rank_change_is_signed_toward_promotion(self) -> None:
        shift = lv.ranking_shift(_frame(self.ROWS))
        gompertz = shift[shift["rule_label"] == "gompertz"].iloc[0]
        assert gompertz["rank_change"] > 0        # promoted by a real horizon

    def test_an_empty_frame_has_no_optimum(self) -> None:
        with pytest.raises(ValueError, match="no combinations"):
            lv.optimum(pd.DataFrame())


class TestAblation:
    ROWS = [
        # the fixed-horizon winner, and a poor performer once survival-weighted
        (1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.12, 0.05),
        # freeing the rate alone
        (1.0, 0.1, "constant_real", 0.06, 0.90, 0.84, 0.30, 0.14),
        # freeing the allocation alone
        (0.8, 0.1, "constant_real", 0.04, 0.92, 0.83, 0.10, 0.04),
        # freeing the rule alone
        (1.0, 0.1, "gompertz", np.nan, 0.85, 0.90, 0.00, 0.00),
        # all three
        (0.8, 0.3, "gompertz", np.nan, 0.80, 0.97, 0.00, 0.00),
    ]

    def test_the_baseline_is_the_fixed_choice_scored_the_new_way(self
                                                                 ) -> None:
        out = lv.ablation(_frame(self.ROWS))
        base = out.iloc[0]
        assert base["freed"].startswith("nothing")
        assert base["cec"] == pytest.approx(0.80)
        assert base["gain_pct"] == pytest.approx(0.0)

    def test_freeing_one_decision_holds_the_others(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["rate", "cec"] == pytest.approx(0.84)
        assert out.loc["allocation", "cec"] == pytest.approx(0.83)
        assert out.loc["rule", "cec"] == pytest.approx(0.90)

    def test_freeing_everything_finds_the_overall_best(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["all three", "cec"] == pytest.approx(0.97)

    def test_gains_are_measured_against_the_baseline(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["rule", "gain_pct"] == pytest.approx(
            100.0 * (0.90 / 0.80 - 1.0))


class TestVerdict:
    def test_it_names_what_changed(self) -> None:
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["rule_changes"]
        assert found["anything_changes"]

    def test_a_winner_with_no_rate_is_flagged_not_printed_as_nan(self
                                                                 ) -> None:
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["winner_sets_no_rate"]
        assert "rate_changes" not in found

    def test_ruin_is_compared_only_on_rules_that_can_run_out(self) -> None:
        """The winning rule often cannot deplete, and zero over zero would
        say nothing about how much a fixed horizon overstates the risk."""
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert not found["optimum_can_deplete"]
        assert found["best_depleting_rule"] == "constant_real"
        assert found["best_depleting_ratio"] == pytest.approx(0.20 / 0.09)

    def test_a_fixed_horizon_that_changes_nothing_is_reported_as_such(self
                                                                      ) -> None:
        """The control: if both objectives pick the same combination the
        section has to say so rather than manufacture a difference."""
        rows = [(1.0, 0.1, "gompertz", np.nan, 1.00, 1.00, 0.0, 0.0),
                (0.8, 0.1, "gompertz", np.nan, 0.90, 0.90, 0.0, 0.0)]
        frame = _frame(rows)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert not found["anything_changes"]

    def test_separability_is_judged_against_the_joint_gain(self) -> None:
        frame = _frame(TestAblation.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert "interaction_pct" in found
        assert found["interaction_pct"] == pytest.approx(
            found["joint_gain_pct"] - sum(found["single_gains_pct"].values()))

    def test_an_empty_frame_reports_nothing(self) -> None:
        assert lv.verdict(pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame()) == {"measured": False}


class TestDescribe:
    def test_a_rate_rule_reads_with_its_rate(self) -> None:
        assert lv.describe("constant real", 0.04) == "constant real at 4.0%"

    def test_a_horizon_rule_says_so_rather_than_printing_nan(self) -> None:
        assert "no rate" in lv.describe("gompertz", float("nan"))
        assert "nan" not in lv.describe("gompertz", float("nan")).lower()

    def test_none_is_handled_like_a_missing_rate(self) -> None:
        assert "no rate" in lv.describe("gompertz", None)


class TestWhyTheOrderChanges:
    """Two candidate explanations for the reshuffle -- that a rule wins by
    reading a mortality table, or that it wins by spending faster -- and the
    code has to let the data choose rather than assume the first."""

    def test_front_load_is_the_first_year_share_of_the_balance(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1,
                               rule="constant_real", rate=0.045)
        assert lv.front_load(combo) == pytest.approx(0.045)

    def test_a_horizon_rule_still_has_a_front_load(self) -> None:
        """Which is the point: it puts rules that set a rate and rules that
        derive one on the same footing."""
        assert 0.0 < lv.front_load(
            lv.Combination(equity=1.0, domestic=0.1, rule="gompertz")) < 1.0

    def test_a_longer_planning_horizon_spends_more_slowly(self) -> None:
        plain = lv.front_load(
            lv.Combination(equity=1.0, domestic=0.1, rule="gompertz"))
        buffered = lv.front_load(
            lv.Combination(equity=1.0, domestic=0.1, rule="gompertz",
                           params={"buffer_years": 5.0}))
        assert buffered < plain

    def test_only_the_actuarial_rules_are_marked_mortality_aware(self) -> None:
        assert "gompertz" in lv.MORTALITY_AWARE
        assert "amortisation" not in lv.MORTALITY_AWARE
        assert "life_expectancy" not in lv.MORTALITY_AWARE

    @staticmethod
    def _ordered(front_loads, cecs) -> pd.DataFrame:
        return pd.DataFrame([
            {"equity": 1.0, "domestic": 0.1, "rule": f"r{i}",
             "rule_label": f"r{i}", "rate": np.nan, "has_rate": False,
             "label": f"r{i}", lv.FIXED: 1.0, lv.MORTALITY: c,
             "ruin_fixed": 0.0, "ruin_mortality": 0.0,
             "mean_consumption": 1.0, "front_load": f,
             "reads_mortality": i == 0}
            for i, (f, c) in enumerate(zip(front_loads, cecs))])

    def test_a_perfect_match_reports_a_correlation_of_one(self) -> None:
        frame = self._ordered([0.03, 0.04, 0.05, 0.06],
                              [0.80, 0.85, 0.90, 0.95])
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["front_load_corr"] == pytest.approx(1.0)

    def test_no_relationship_reports_no_correlation(self) -> None:
        """The control: if spending speed did not explain the order, the
        section must not claim it does."""
        frame = self._ordered([0.03, 0.06, 0.04, 0.05],
                              [0.90, 0.85, 0.95, 0.80])
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert abs(found["front_load_corr"]) < 0.9

    def test_a_blind_rule_beating_the_actuarial_one_is_flagged(self) -> None:
        # r0 is the mortality-aware one and scores worst.
        frame = self._ordered([0.03, 0.04, 0.05, 0.06],
                              [0.80, 0.85, 0.90, 0.95])
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["a_blind_rule_wins"]
        assert found["mortality_aware_best_rank"] > found["blind_best_rank"]

    def test_an_actuarial_winner_is_not_flagged(self) -> None:
        frame = self._ordered([0.03, 0.04, 0.05, 0.06],
                              [0.95, 0.90, 0.85, 0.80])
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert not found["a_blind_rule_wins"]


class TestCorrelationStrength:
    """A moderate correlation is a contributing cause. Writing it up as *the*
    cause is the mistake this banding exists to stop, and it is one an
    earlier draft of this section made."""

    def test_the_bands_run_in_order(self) -> None:
        assert lv.correlation_strength(0.9) == "strong"
        assert lv.correlation_strength(0.5) == "moderate"
        assert lv.correlation_strength(0.2) == "weak"
        assert lv.correlation_strength(0.05) == "none"

    def test_sign_does_not_change_the_strength(self) -> None:
        assert lv.correlation_strength(-0.9) == lv.correlation_strength(0.9)

    def test_a_missing_correlation_is_not_a_finding(self) -> None:
        assert lv.correlation_strength(float("nan")) == "none"

    def test_the_measured_value_is_not_called_strong(self) -> None:
        """+0.44 on the production grid: real, and not the whole story."""
        assert lv.correlation_strength(0.44) == "moderate"


class TestGridEdge:
    """An optimum on the boundary of its own grid is a truncation. The
    project has `at_grid_edge` for exactly this and Section 34 shipped
    without calling it: the rate grid stopped at 6%, the
    percentage-of-balance rules were still improving there, and the figure
    showed a corner as though it were a peak."""

    @staticmethod
    def _frame(rates, best_at):
        rows = []
        for r in rates:
            rows.append({
                "equity": 1.0, "domestic": 0.1, "rule": "constant_percent",
                "rule_label": "constant_percent", "rate": r, "has_rate": True,
                "label": f"cp-{r}", lv.FIXED: 1.0,
                lv.MORTALITY: 1.0 - abs(r - best_at),
                "ruin_fixed": 0.0, "ruin_mortality": 0.0,
                "mean_consumption": 1.0, "front_load": r,
                "reads_mortality": False})
        return pd.DataFrame(rows)

    def test_a_peak_on_the_top_edge_is_flagged(self) -> None:
        frame = self._frame([0.03, 0.04, 0.05, 0.06], best_at=0.06)
        found = lv.verdict(frame, lv.ranking_shift(frame), lv.ablation(frame))
        assert found["best_rated_at_edge"]
        assert not found["rate_optimum_interior"]

    def test_a_peak_on_the_bottom_edge_is_flagged(self) -> None:
        frame = self._frame([0.03, 0.04, 0.05, 0.06], best_at=0.03)
        found = lv.verdict(frame, lv.ranking_shift(frame), lv.ablation(frame))
        assert found["best_rated_at_edge"]

    def test_an_interior_peak_is_not_flagged(self) -> None:
        frame = self._frame([0.03, 0.05, 0.08, 0.10, 0.12], best_at=0.08)
        found = lv.verdict(frame, lv.ranking_shift(frame), lv.ablation(frame))
        assert not found["best_rated_at_edge"]
        assert found["rate_optimum_interior"]

    def test_the_grid_bounds_are_reported(self) -> None:
        frame = self._frame([0.03, 0.05, 0.08], best_at=0.05)
        found = lv.verdict(frame, lv.ranking_shift(frame), lv.ablation(frame))
        assert found["rate_grid_low"] == pytest.approx(0.03)
        assert found["rate_grid_high"] == pytest.approx(0.08)

    def test_the_production_grid_reaches_past_the_old_edge(self) -> None:
        """The config must offer rates above 6%, or the fix is only in the
        prose."""
        from src import data_loader as dl

        grid = dl.load_config("config.yaml")["longevity"]["rate_grid"]
        assert max(grid) > 0.06
        assert max(grid) >= 0.10


class TestRateCurveFigure:
    """Two dials set a spending level -- a withdrawal rate and an assumed
    real return -- and they are different measures, so the figure gives each
    its own panel rather than a shared axis."""

    def test_the_assumed_return_is_swept_not_pinned(self) -> None:
        """The dial the earlier version never swept. Three hand-picked
        values in `spending.rules` is a menu, not a grid."""
        from src import data_loader as dl

        block = dl.load_config("config.yaml")["longevity"]
        assert "assumed_return_grid" in block
        assert len(block["assumed_return_grid"]) >= 6

    def test_the_return_grid_reaches_past_any_plausible_return(self) -> None:
        """What is being located is where over-assuming starts to cost, and
        that cannot be found from inside the range of sensible returns."""
        from src import data_loader as dl

        block = dl.load_config("config.yaml")["longevity"]
        assert max(block["assumed_return_grid"]) >= 0.10
        assert min(block["assumed_return_grid"]) <= 0.0

    def test_the_curve_rules_span_both_families(self) -> None:
        """A chart drawn only from rules that cannot run out would show the
        high optimum and hide the reason for it."""
        from src import plan as pl
        from src import plots

        families = {bool(pl.CAN_DEPLETE.get(r.split(" (")[0], True))
                    for r in plots.RATE_CURVE_RULES}
        assert families == {True, False}

    def test_the_curve_shows_no_more_than_the_palette_allows(self) -> None:
        """Nine rules would cycle the categorical hues, which is never
        allowed; the rest live in the tables."""
        from src import plots

        assert len(plots.RATE_CURVE_RULES) <= 6

    def test_every_named_rule_is_in_the_sweep(self, tmp_path) -> None:
        """A renamed rule would silently drop a line from the figure."""
        from src import data_loader as dl
        from src import plots
        from src import spending as spg

        cfg = dl.load_config("config.yaml")
        known = set()
        for spec in cfg["spending"]["rules"]:
            suffix = str(spec.get("suffix", "") or "")
            key = str(spec["key"])
            known.add(f"{key} ({suffix})" if suffix else key)
        missing = [r for r in plots.RATE_CURVE_RULES if r not in known]
        assert not missing, f"not produced by the sweep: {missing}"
        assert all(r.split(" (")[0] in spg.REGISTRY
                   for r in plots.RATE_CURVE_RULES)

    def test_it_draws_from_the_preference_frame_it_is_handed(self, tmp_path
                                                             ) -> None:
        """The bars are `rate_preference`'s answer, not a second one worked
        out inside the plotting layer, so the chart and the prose that reads
        it cannot disagree. This renders the real call to catch the
        signature drifting away from `main`."""
        from src import plan as pl
        from src import plots

        rows = [(1.0, 0.3, rule, rate, 1.0, 1.4 - 8.0 * (rate - peak) ** 2,
                 0.1, 0.05)
                for rule, peak in (("constant_real", 0.045),
                                   ("constant_percent", 0.08))
                for rate in (0.03, 0.045, 0.06, 0.08, 0.12)]
        swept = _frame(rows)
        preference = lv.rate_preference(swept, pl.CAN_DEPLETE)
        out = plots.plot_rate_curve(swept, preference, tmp_path,
                                    name="rate_curve")
        assert out.exists() and out.stat().st_size > 1_000

    def test_it_renders_the_assumed_return_panel_too(self, tmp_path) -> None:
        """The second dial's panel is empty until the sweep carries the
        column, so the render is checked with it present."""
        from src import plan as pl
        from src import plots

        rated = _frame([
            (1.0, 0.3, "constant_real", rate, 1.0,
             1.4 - 8.0 * (rate - 0.045) ** 2, 0.1, 0.05)
            for rate in (0.03, 0.045, 0.06, 0.08)])
        dialled = _dialled([
            (1.0, 0.3, "amortisation", ret, 1.0,
             1.4 - 6.0 * (ret - 0.05) ** 2, 0.0, 0.0)
            for ret in (0.0, 0.02, 0.05, 0.08, 0.12)])
        swept = pd.concat([rated, dialled], ignore_index=True)
        preference = lv.rate_preference(swept, pl.CAN_DEPLETE)
        out = plots.plot_rate_curve(swept, preference, tmp_path,
                                    name="rate_curve_dialled")
        assert out.exists() and out.stat().st_size > 1_000

    def test_the_two_curve_panels_share_a_vertical_scale(self, tmp_path,
                                                         monkeypatch) -> None:
        """Both panels plot the same quantity. Letting each rescale to its
        own range makes the higher peak unreadable, which is the one
        question putting them side by side is meant to answer.

        `_save` closes the figure, so the close is stubbed out to keep the
        axes alive long enough to read their limits -- the alternative is a
        test that asserts a file exists and calls that a shared scale.
        """
        import matplotlib.pyplot as plt

        from src import plan as pl
        from src import plots

        rated = _frame([
            (1.0, 0.3, "constant_real", rate, 1.0,
             1.0 - 8.0 * (rate - 0.045) ** 2, 0.1, 0.05)
            for rate in (0.03, 0.045, 0.06, 0.08)])
        # Deliberately on a different level from the rated rules, so a
        # per-panel rescale would show up as two different ranges.
        dialled = _dialled([
            (1.0, 0.3, "amortisation", ret, 1.0,
             1.9 - 6.0 * (ret - 0.05) ** 2, 0.0, 0.0)
            for ret in (0.0, 0.02, 0.05, 0.08, 0.12)])
        swept = pd.concat([rated, dialled], ignore_index=True)
        preference = lv.rate_preference(swept, pl.CAN_DEPLETE)

        kept = []
        monkeypatch.setattr(plt, "close", lambda fig: kept.append(fig))
        plots.plot_rate_curve(swept, preference, tmp_path,
                              name="shared_scale")
        assert kept, "the figure was never handed to plt.close"
        axes = kept[0].get_axes()
        assert axes[0].get_ylim() == axes[1].get_ylim()
        # And the shared range actually contains both curves, so sharing it
        # has not clipped one of them out of view.
        low, high = axes[0].get_ylim()
        assert low <= swept["cec_mortality"].min()
        assert high >= swept["cec_mortality"].max()

    def test_the_preference_frame_holds_only_rate_setting_rules(self) -> None:
        """The bar panel is a withdrawal-rate axis. A rule dialled by an
        assumed return has no place on it, and would be read as wanting a
        withdrawal rate it never sets."""
        from src import plan as pl

        rated = _frame([(1.0, 0.3, "constant_real", 0.04, 1.0, 1.2,
                         0.1, 0.05)])
        dialled = _dialled([(1.0, 0.3, "amortisation", 0.04, 1.0, 1.9,
                             0.0, 0.0)])
        swept = pd.concat([rated, dialled], ignore_index=True)
        pref = lv.rate_preference(swept, pl.CAN_DEPLETE)
        assert list(pref["rule_label"]) == ["constant_real"]


class TestRatePreference:
    """The rate each rule wants, and whether "can run out" explains it.

    The tempting story -- rules that cannot deplete want the high rates --
    is nearly right on the real sweep and not exactly right, so it is
    classified rather than told.
    """

    @staticmethod
    def _pref(rows, deplete):
        return lv.rate_preference(_frame(rows), deplete)

    def test_each_rule_appears_once_at_its_own_best_rate(self) -> None:
        pref = self._pref(
            [(1.0, 0.5, "a", 0.04, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "a", 0.08, 1.2, 1.3, 0.0, 0.0),
             (0.6, 0.5, "a", 0.08, 1.1, 1.1, 0.0, 0.0),
             (1.0, 0.5, "b", 0.04, 1.0, 1.4, 0.0, 0.0)],
            {"a": False, "b": True})
        assert list(pref["rule_label"]) == ["a", "b"]
        # The best allocation for the rule is taken, not the first seen.
        assert float(
            pref.loc[pref["rule_label"] == "a", "rate"].iloc[0]) == 0.08

    def test_rules_without_a_rate_are_absent_not_nan(self) -> None:
        pref = self._pref(
            [(1.0, 0.5, "a", 0.05, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "gompertz", float("nan"), 1.0, 2.0, 0.0, 0.0)],
            {"a": True, "gompertz": False})
        assert list(pref["rule_label"]) == ["a"]

    def test_the_bracketed_variant_inherits_the_family_flag(self) -> None:
        pref = self._pref(
            [(1.0, 0.5, "endowment (light smoothing)", 0.08,
              1.0, 1.0, 0.0, 0.0)],
            {"endowment": True})
        assert bool(pref.iloc[0]["can_deplete"])
        assert pref.iloc[0]["family"] == "endowment"

    def test_rows_are_ordered_by_the_rate_wanted(self) -> None:
        pref = self._pref(
            [(1.0, 0.5, "low", 0.03, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "high", 0.09, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "mid", 0.06, 1.0, 1.0, 0.0, 0.0)],
            {})
        assert list(pref["rule_label"]) == ["high", "mid", "low"]

    def test_an_empty_sweep_gives_an_empty_frame(self) -> None:
        assert not len(lv.rate_preference(_frame([]), {}))


class TestRateSplitVerdict:
    @staticmethod
    def _split(rows, deplete):
        return lv.rate_split_verdict(lv.rate_preference(_frame(rows),
                                                        deplete))

    def test_a_clean_split_is_reported_as_separating(self) -> None:
        found = self._split(
            [(1.0, 0.5, "pct", 0.08, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "real", 0.04, 1.0, 1.0, 0.0, 0.0)],
            {"pct": False, "real": True})
        assert found["separates"]
        assert "crossing_rule" not in found

    def test_a_depleting_rule_at_the_top_breaks_the_split(self) -> None:
        found = self._split(
            [(1.0, 0.5, "pct", 0.08, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "endowment", 0.08, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "real", 0.04, 1.0, 1.0, 0.0, 0.0)],
            {"pct": False, "endowment": True, "real": True})
        assert not found["separates"]
        assert found["crossing_rule"] == "endowment"
        assert found["crossing_ties_top"]

    def test_a_crossing_below_the_top_is_flagged_without_a_tie(self) -> None:
        found = self._split(
            [(1.0, 0.5, "pct", 0.09, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "pct2", 0.05, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "endowment", 0.07, 1.0, 1.0, 0.0, 0.0)],
            {"pct": False, "pct2": False, "endowment": True})
        assert not found["separates"]
        assert found["crossing_rule"] == "endowment"
        assert not found["crossing_ties_top"]

    def test_the_spread_is_the_top_minus_the_bottom_in_points(self) -> None:
        found = self._split(
            [(1.0, 0.5, "a", 0.080, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "b", 0.045, 1.0, 1.0, 0.0, 0.0)],
            {})
        assert found["spread_pp"] == pytest.approx(3.5)
        assert found["top_rule"] == "a" and found["bottom_rule"] == "b"

    def test_one_sided_evidence_cannot_claim_a_split(self) -> None:
        # Every rule can deplete, so there is nothing to separate from.
        found = self._split(
            [(1.0, 0.5, "a", 0.08, 1.0, 1.0, 0.0, 0.0),
             (1.0, 0.5, "b", 0.04, 1.0, 1.0, 0.0, 0.0)],
            {"a": True, "b": True})
        assert not found["separates"]

    def test_an_empty_sweep_is_not_measured(self) -> None:
        assert not lv.rate_split_verdict(
            lv.rate_preference(_frame([]), {})).get("measured", False)


class TestRatePreferenceTieBreak:
    def test_equal_scores_resolve_to_the_lower_rate(self) -> None:
        """Two rates that score identically are not a coin toss: the lower
        one is reported, matching the peak the figure rings."""
        pref = lv.rate_preference(
            _frame([(1.0, 0.5, "a", 0.04, 1.0, 1.3, 0.0, 0.0),
                    (1.0, 0.5, "a", 0.09, 1.0, 1.3, 0.0, 0.0)]),
            {"a": True})
        assert float(pref.iloc[0]["rate"]) == 0.04


class TestReturnDialledPlans:
    """An amortisation rule takes no withdrawal rate; its level comes from
    the real return it assumes. Sweeping the rate grid and leaving that one
    pinned to whatever the config listed is how a corner hides."""

    SPECS = [{"key": "constant_real"},
             {"key": "amortisation", "params": {"assumed_return": 0.0},
              "suffix": "0% assumed return"},
             {"key": "amortisation", "params": {"assumed_return": 0.02},
              "suffix": "2% assumed return"},
             {"key": "amortisation", "params": {"assumed_return": 0.04},
              "suffix": "4% assumed return"},
             {"key": "gompertz"}]

    def test_the_grid_overrides_the_pinned_variants(self) -> None:
        plans = lv.plan_grid(self.SPECS, [0.04], [0.0, 0.03, 0.06])
        amort = [(p["assumed_return"], sfx)
                 for k, _, p, sfx in plans if k == "amortisation"]
        assert sorted(v for v, _ in amort) == [0.0, 0.03, 0.06]

    def test_configured_variants_collapse_onto_the_grid(self) -> None:
        """Three specs crossed with a three-point grid is three policies,
        not nine: they are the same family dialled differently."""
        plans = lv.plan_grid(self.SPECS, [0.04], [0.0, 0.03, 0.06])
        assert sum(k == "amortisation" for k, _, _, _ in plans) == 3

    def test_without_a_grid_the_configured_variants_survive(self) -> None:
        """A caller that wants the configured menu rather than a sweep --
        section #spending's rule comparison -- keeps it."""
        plans = lv.plan_grid(self.SPECS, [0.04])
        amort = sorted(p["assumed_return"]
                       for k, _, p, _ in plans if k == "amortisation")
        assert amort == [0.0, 0.02, 0.04]

    def test_the_suffix_names_the_dial(self) -> None:
        plans = lv.plan_grid(self.SPECS, [0.04], [0.025])
        sfx = [s for k, _, _, s in plans if k == "amortisation"]
        assert sfx == ["2.5% assumed return"]

    def test_a_rule_with_no_dial_at_all_still_appears_once(self) -> None:
        plans = lv.plan_grid(self.SPECS, [0.04], [0.0, 0.03])
        assert sum(k == "gompertz" for k, _, _, _ in plans) == 1

    def test_the_combination_reports_the_return_it_amortises_at(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.3, rule="amortisation",
                               params={"assumed_return": 0.05})
        assert combo.assumed_return == pytest.approx(0.05)

    def test_a_rate_setting_rule_reports_no_assumed_return(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.3,
                               rule="constant_real", rate=0.04)
        assert combo.assumed_return is None


class TestReturnGridEdge:
    """The same corner check as the withdrawal rate, on the other dial."""

    @staticmethod
    def _found(returns, peak):
        rated = _frame([(1.0, 0.3, "constant_real", r, 1.0,
                         1.2 - 4.0 * (r - 0.045) ** 2, 0.1, 0.05)
                        for r in (0.03, 0.045, 0.06)])
        dialled = _dialled([(1.0, 0.3, "amortisation", r, 1.0,
                             1.5 - 6.0 * (r - peak) ** 2, 0.0, 0.0)
                            for r in returns])
        swept = pd.concat([rated, dialled], ignore_index=True)
        return lv.verdict(swept, lv.ranking_shift(swept), lv.ablation(swept))

    def test_a_monotone_grid_is_reported_as_a_corner(self) -> None:
        """The 0/2/4% menu the section used to carry, where the best of the
        three was the top of the three."""
        found = self._found([0.0, 0.02, 0.04], peak=0.09)
        assert found["best_return_at_edge"]
        assert not found["return_optimum_interior"]
        assert found["best_return"] == pytest.approx(0.04)

    def test_a_wide_grid_finds_the_peak_inside(self) -> None:
        found = self._found([0.0, 0.02, 0.05, 0.08, 0.12], peak=0.05)
        assert not found["best_return_at_edge"]
        assert found["return_optimum_interior"]
        assert found["best_return"] == pytest.approx(0.05)

    def test_the_grid_bounds_are_recorded_for_the_prose(self) -> None:
        found = self._found([0.0, 0.02, 0.05, 0.08, 0.12], peak=0.05)
        assert found["return_grid_low"] == pytest.approx(0.0)
        assert found["return_grid_high"] == pytest.approx(0.12)

    def test_it_says_whether_the_winner_is_the_dialled_rule(self) -> None:
        """Whether the section's headline rests on that dial is the reason
        the check matters."""
        found = self._found([0.0, 0.02, 0.04], peak=0.09)
        assert found["winner_is_return_dialled"]

    def test_a_sweep_without_the_dial_reports_nothing_about_it(self) -> None:
        swept = _frame([(1.0, 0.3, "constant_real", r, 1.0,
                         1.2 - 4.0 * (r - 0.045) ** 2, 0.1, 0.05)
                        for r in (0.03, 0.045, 0.06)])
        found = lv.verdict(swept, lv.ranking_shift(swept),
                           lv.ablation(swept))
        assert "best_return_at_edge" not in found


class TestTheHorizonNumbersReachTheProse:
    """Three NaNs shipped in the body text of both papers.

    `verdict` never carried `expected_age_at_death`, `life_expectancy` or
    `fixed_horizon_years`; the section that quotes them read them off the
    verdict with a NaN default and printed "the expected age at death is
    nan". The values were being computed in the pipeline all along and
    simply not passed along.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame.from_records([{
            "objective": "fixed", "rule": "gompertz", "rule_label": "gompertz",
            "rate": float("nan"), "equity": 1.0, "domestic": 0.1,
            lv.FIXED: 1.0, lv.MORTALITY: 1.1, "ruin_fixed": 0.1,
            "ruin_mortality": 0.05, "has_rate": False,
            "assumed_return": float("nan"), "has_assumed_return": False,
            "label": "gompertz-1.0", "mean_consumption": 1.0}])

    def test_the_numbers_are_carried_through(self) -> None:
        found = lv.verdict(self._frame(), pd.DataFrame(), pd.DataFrame(),
                           {"expected_age_at_death": 81.9,
                            "life_expectancy": 18.9,
                            "fixed_horizon_years": 30.0})
        assert found["expected_age_at_death"] == pytest.approx(81.9)
        assert found["life_expectancy"] == pytest.approx(18.9)
        assert found["fixed_horizon_years"] == pytest.approx(30.0)

    def test_none_of_them_is_ever_a_nan(self) -> None:
        found = lv.verdict(self._frame(), pd.DataFrame(), pd.DataFrame(),
                           {"expected_age_at_death": 81.9,
                            "life_expectancy": 18.9,
                            "fixed_horizon_years": 30.0})
        assert not any(isinstance(v, float) and np.isnan(v)
                       for k, v in found.items()
                       if k.endswith(("_age_at_death", "_expectancy",
                                      "_horizon_years")))

    def test_omitting_them_adds_no_keys(self) -> None:
        """A caller that does not own a survival curve must not be given
        placeholder keys the prose would then print."""
        found = lv.verdict(self._frame(), pd.DataFrame(), pd.DataFrame())
        for key in ("expected_age_at_death", "life_expectancy",
                    "fixed_horizon_years"):
            assert key not in found

    def test_the_pipeline_supplies_all_three(self) -> None:
        """A source check: the section prints all three, so all three have
        to be passed at the one call site that feeds it."""
        import re
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "main.py").read_text()
        # The lookbehind matters: `slv.verdict(comparison)` (the sleeve
        # study) also ends in "lv.verdict(" and matched first.
        call = re.search(r"(?<![A-Za-z_])lv\.verdict\("
                         r"(?:[^()]|\([^()]*\))*\)", source)
        assert call, "no lv.verdict call found in main.py"
        for key in ("expected_age_at_death", "life_expectancy",
                    "fixed_horizon_years"):
            assert key in call.group(0), key


class TestTheRuinOverstatement:
    """The size of the fixed-horizon distortion, computed once.

    Section #baseline quotes fixed-horizon ruin because that is the
    replicated study's convention, and points forward to this section for
    the size of what it is living with. Both sections therefore read the
    same helper: a second hand-rolled median would be free to drift.
    """

    ROWS = [
        # ruin_fixed, ruin_mortality: ratios 2.0, 4.0, and a rule that
        # cannot deplete at all.
        (1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.20, 0.10),
        (1.0, 0.1, "constant_real", 0.05, 0.95, 0.85, 0.40, 0.10),
        (1.0, 0.1, "gompertz", np.nan, 0.90, 0.95, 0.00, 0.00),
    ]

    def test_it_measures_only_the_rules_that_can_run_out(self) -> None:
        over = lv.ruin_overstatement(_frame(self.ROWS))
        assert over["measured"]
        assert over["combinations"] == 2          # the gompertz row is out
        assert over["median_ratio"] == pytest.approx(3.0)
        assert over["worst_ratio"] == pytest.approx(4.0)

    def test_a_grid_that_cannot_deplete_reports_nothing(self) -> None:
        """Rather than a ratio of zero over zero, which would read as a
        finding that the horizon costs nothing."""
        rows = [(1.0, 0.1, "gompertz", np.nan, 0.90, 0.95, 0.0, 0.0)]
        assert lv.ruin_overstatement(_frame(rows)) == {"measured": False}

    def test_a_frame_without_the_columns_reports_nothing(self) -> None:
        frame = pd.DataFrame({"rule_label": ["constant_real"]})
        assert lv.ruin_overstatement(frame) == {"measured": False}

    def test_a_zero_survival_ruin_is_dropped_not_infinite(self) -> None:
        """A combination that fails only after the survival curve has run
        out divides by zero; the median must not become infinity."""
        rows = [(1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.20, 0.10),
                (1.0, 0.1, "constant_real", 0.05, 0.95, 0.85, 0.30, 0.00)]
        over = lv.ruin_overstatement(_frame(rows))
        assert over["combinations"] == 2          # both can deplete
        assert np.isfinite(over["median_ratio"])
        assert over["median_ratio"] == pytest.approx(2.0)

    def test_the_verdict_quotes_the_helper_rather_than_its_own_median(self
                                                                      ) -> None:
        frame = _frame(self.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        over = lv.ruin_overstatement(frame)
        assert found["median_ruin_ratio"] == pytest.approx(
            over["median_ratio"])
        assert found["depleting_combinations"] == over["combinations"]

    def test_the_overstatement_never_runs_the_wrong_way(self) -> None:
        """Survival-weighted ruin is the same event intersected with being
        alive, so the fixed horizon can only report more of it. On the real
        grid the claim the paper makes is that the ratio exceeds one."""
        import pathlib

        path = pathlib.Path("results/tables/longevity_sweep.csv")
        if not path.exists():
            pytest.skip("pipeline results not present")
        over = lv.ruin_overstatement(pd.read_csv(path))
        assert over["measured"], "the real grid no longer feeds the caveat"
        assert over["median_ratio"] >= 1.0
        assert over["worst_ratio"] >= over["median_ratio"]
