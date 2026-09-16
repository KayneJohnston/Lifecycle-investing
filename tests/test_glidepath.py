"""Tests for the batched evaluator and the glide-path optimisers."""

from __future__ import annotations

import numpy as np
import pytest

from src import bootstrap as bs
from src import glidepath as gp
from src import lifecycle as lc
from src import spending as sp
from src import utility as ut


@pytest.fixture()
def setup(toy_panel, toy_config):
    spec = lc.spec_from_config(toy_config)
    strategies = lc.build_strategies(toy_config, spec)
    sampler = bs.from_config(toy_panel, toy_config, horizon_years=spec.horizon)
    paths = sampler.sample(500, chunk_size=250)
    income = lc.simulate_income(spec, 500, np.random.default_rng(3))
    return toy_config, spec, strategies, paths, income


class TestWeightsFromShares:
    def test_rows_sum_to_one(self) -> None:
        w = gp.weights_from_shares(np.array([0.0, 0.5, 1.0]),
                                   np.array([0.2, 0.5, 0.8]))
        np.testing.assert_allclose(w.sum(axis=1), 1.0)

    def test_decomposes_the_sleeves(self) -> None:
        w = gp.weights_from_shares(np.array([0.6]), np.array([0.25]),
                                   bond_share=0.7)
        np.testing.assert_allclose(w[0], [0.15, 0.45, 0.28, 0.12])

    def test_clips_out_of_range_shares(self) -> None:
        w = gp.weights_from_shares(np.array([1.7, -0.4]), np.array([2.0, -1.0]))
        np.testing.assert_allclose(w[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(w[1], [0.0, 0.0, 0.7, 0.3])


class TestBatchSimulate:
    def test_matches_the_reference_simulator(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        keys = list(strategies)
        weights = np.stack([strategies[k].weights for k in keys])
        consumption, bequest, ruin = gp.batch_simulate(paths, weights, spec,
                                                       income)
        for i, key in enumerate(keys):
            single = lc.simulate(paths, strategies[key], spec, income)
            # Not bit-identical: the batched path accumulates the portfolio
            # return through a BLAS matmul and the reference through einsum,
            # so summation order differs in the last couple of digits.
            np.testing.assert_allclose(consumption[i], single.consumption,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(bequest[i], single.bequest,
                                       rtol=1e-12, atol=1e-10)
            np.testing.assert_array_equal(ruin[i], single.ruin)

    def test_honours_a_custom_spending_rule(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        rule = sp.ConstantPercentRule(rate=0.06)
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        consumption, bequest, ruin = gp.batch_simulate(paths, weights, spec,
                                                       income, rule)
        single = lc.simulate(paths, strategies[key], spec, income, rule)
        np.testing.assert_allclose(consumption[0], single.consumption,
                                   rtol=1e-12, atol=1e-12)
        assert not ruin.any()

    def test_consumption_from_trims_the_leading_years(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        weights = strategies[list(strategies)[0]].weights[None]
        full, _, _ = gp.batch_simulate(paths, weights, spec, income)
        trimmed, _, _ = gp.batch_simulate(paths, weights, spec, income,
                                          consumption_from=spec.n_working)
        assert trimmed.shape[2] == spec.n_retired
        np.testing.assert_allclose(trimmed[0], full[0][:, spec.n_working:])

    def test_the_balance_dial_reaches_the_batched_path(self, setup) -> None:
        """`retirement_balance_scale` is applied by `lifecycle.simulate` at
        the retirement boundary. The batched recursion ignored it, so a
        study that scaled the balance and evaluated in batch scored every
        row at the household's own wealth and could not see the dial at
        all."""
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        scaled = dataclasses.replace(spec, retirement_balance_scale=0.1)
        weights = strategies[list(strategies)[0]].weights[None]
        batched, _, _ = gp.batch_simulate(paths, weights, scaled, income)
        single = lc.simulate(paths, strategies[list(strategies)[0]], scaled,
                             income)
        np.testing.assert_allclose(batched[0], single.consumption,
                                   rtol=1e-10, atol=1e-10)

    def test_the_balance_dial_changes_the_batched_answer(self, setup) -> None:
        """Guards the guard: if the scale were silently dropped again, the
        comparison above would still pass against a reference that had also
        dropped it."""
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        weights = strategies[list(strategies)[0]].weights[None]
        plain, _, _ = gp.batch_simulate(paths, weights, spec, income)
        scaled, _, _ = gp.batch_simulate(
            paths, weights, dataclasses.replace(
                spec, retirement_balance_scale=0.1), income)
        assert not np.allclose(plain[:, :, spec.n_working:],
                               scaled[:, :, spec.n_working:])

    def test_rejects_a_bad_weight_shape(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        with pytest.raises(ValueError, match="weights must be"):
            gp.batch_simulate(paths, np.ones((2, spec.horizon)), spec, income)

    def test_rejects_a_mismatched_horizon(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        with pytest.raises(ValueError, match="weights horizon"):
            gp.batch_simulate(paths, np.ones((1, spec.horizon + 3, 4)), spec,
                              income)


class TestBatchCec:
    def test_matches_the_reference_certainty_equivalent(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        keys = list(strategies)
        weights = np.stack([strategies[k].weights for k in keys])
        got = gp.BatchEvaluator(paths, spec, income, cfg).cec(weights, 5.0)
        for i, key in enumerate(keys):
            outcome = lc.simulate(paths, strategies[key], spec, income)
            bundle = ut.bundle_from_outcome(outcome, cfg, spec)
            want = ut.crra_certainty_equivalent(
                bundle, 5.0, float(cfg["utility"]["discount_factor"]),
                float(cfg["utility"]["bequest_weight"]),
                bool(cfg["utility"]["bequest_enabled"]))
            assert got[i] == pytest.approx(want, rel=1e-10)

    @pytest.mark.parametrize("gamma", [2.0, 5.0, 10.0])
    def test_agrees_across_risk_aversions(self, setup, gamma) -> None:
        cfg, spec, strategies, paths, income = setup
        key = list(strategies)[0]
        got = gp.BatchEvaluator(paths, spec, income, cfg).cec(
            strategies[key].weights[None], gamma)[0]
        outcome = lc.simulate(paths, strategies[key], spec, income)
        want = ut.crra_certainty_equivalent(
            ut.bundle_from_outcome(outcome, cfg, spec), gamma, 0.96, 2.0, True)
        assert got == pytest.approx(want, rel=1e-10)

    def test_more_equity_beats_bills_on_this_panel(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        equity = gp.weights_from_shares(np.ones(spec.horizon),
                                        np.full(spec.horizon, 0.5))
        bills = gp.weights_from_shares(np.zeros(spec.horizon),
                                       np.zeros(spec.horizon), bond_share=0.0)
        scores = evaluator.cec(np.stack([equity, bills]), 5.0)
        assert scores[0] > scores[1]


class TestGlideParameterisation:
    def test_interpolates_between_knots(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        param = gp.GlideParameterisation(
            knot_ages=(spec.age_start, spec.age_death - 1))
        weights = param.build([1.0, 0.0], spec)
        equity = weights[:, 0] + weights[:, 1]
        assert equity[0] == pytest.approx(1.0)
        assert equity[-1] == pytest.approx(0.0)
        assert np.all(np.diff(equity) <= 1e-12)

    def test_rejects_the_wrong_number_of_knots(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        param = gp.GlideParameterisation(knot_ages=(25, 50, 90))
        with pytest.raises(ValueError, match="expected 3 knot values"):
            param.build([1.0, 0.0], spec)

    def test_builds_a_valid_strategy(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        param = gp.GlideParameterisation(knot_ages=(spec.age_start,
                                                    spec.age_death - 1))
        strategy = param.strategy([0.9, 0.4], spec)
        np.testing.assert_allclose(strategy.weights.sum(axis=1), 1.0)


class TestOptimisers:
    def test_free_form_improves_on_its_starting_point(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        grid = [0.0, 0.5, 1.0]
        start = gp.weights_from_shares(np.full(spec.horizon, 0.5),
                                       np.full(spec.horizon, 0.5))
        before = float(evaluator.cec(start[None], 5.0)[0])
        equity, domestic, after = gp.optimise_free_form_banded(
            evaluator, 5.0, grid, grid, start_equity=0.5, start_domestic=0.5,
            n_sweeps=1)
        assert after >= before

    def test_free_form_respects_the_grid(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        grid = [0.0, 0.5, 1.0]
        equity, domestic, _ = gp.optimise_free_form_banded(
            evaluator, 5.0, grid, grid, n_sweeps=1)
        assert set(np.unique(equity)).issubset(set(grid))
        assert set(np.unique(domestic)).issubset(set(grid))

    def test_off_grid_start_is_snapped_onto_the_grid(self, setup) -> None:
        # A coordinate that finds nothing better keeps its starting value, so
        # an unsnapped start would leak an off-grid number into the schedule.
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        grid = [0.0, 0.5, 1.0]
        equity, domestic, _ = gp.optimise_free_form_banded(
            evaluator, 5.0, grid, grid, start_equity=0.83,
            start_domestic=0.11, n_sweeps=1)
        assert set(np.unique(equity)).issubset(set(grid))
        assert set(np.unique(domestic)).issubset(set(grid))

    def test_min_improvement_suppresses_negligible_moves(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        grid = [0.0, 0.5, 1.0]
        loose, _, _ = gp.optimise_free_form_banded(
            evaluator, 5.0, grid, grid, n_sweeps=1, min_improvement=0.0)
        strict, _, _ = gp.optimise_free_form_banded(
            evaluator, 5.0, grid, grid, n_sweeps=1, min_improvement=1.0)
        # An impossible threshold means no coordinate ever moves.
        assert len(np.unique(strict)) == 1

    def test_free_form_reports_a_consistent_certainty_equivalent(self, setup
                                                                 ) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        equity, domestic, cec = gp.optimise_free_form_banded(
            evaluator, 5.0, [0.0, 0.5, 1.0], [0.0, 1.0], n_sweeps=1)
        rebuilt = evaluator.cec(
            gp.weights_from_shares(equity, domestic, 0.7)[None], 5.0)[0]
        assert rebuilt == pytest.approx(cec, rel=1e-12)

    def test_parametric_improves_and_records_a_trace(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        param = gp.GlideParameterisation(knot_ages=(spec.age_start,
                                                    spec.age_death - 1))
        trace = gp.OptimisationTrace()
        values, cec = gp.optimise_parametric(
            evaluator, 5.0, param, [0.0, 0.5, 1.0], start=[0.5, 0.5],
            n_sweeps=2, trace=trace)
        assert np.isfinite(cec)
        rebuilt = evaluator.cec(param.build(values, spec)[None], 5.0)[0]
        assert rebuilt == pytest.approx(cec, rel=1e-12)
        assert len(trace.frame()) > 0

    def test_band_index_groups_ages(self) -> None:
        bands = gp._band_index(11, 5)
        np.testing.assert_array_equal(bands, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])


class TestReporting:
    def test_schedule_frame_has_one_row_per_age(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        frame = gp.schedule_frame(np.ones(spec.horizon),
                                  np.full(spec.horizon, 0.3), spec, 5.0, "x")
        assert len(frame) == spec.horizon
        assert list(frame["age"]) == list(spec.ages)

    def test_comparison_ranks_and_reports_gaps(self, setup) -> None:
        cfg, spec, strategies, paths, income = setup
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
        solved = {"solved": gp.weights_from_shares(np.ones(spec.horizon),
                                                   np.full(spec.horizon, 0.5))}
        frame = gp.compare_to_benchmarks(evaluator, solved, strategies, 5.0)
        assert len(frame) == 1 + len(strategies)
        assert frame["cec"].is_monotonic_decreasing
        assert frame["gap_to_best_pct"].iloc[0] == pytest.approx(0.0)
        assert (frame["gap_to_best_pct"] <= 1e-12).all()


class TestTheFastPathCarriesTheMeansTest:
    """The batched evaluator is what the joint search of `src.plan` calls
    tens of thousands of times, and it settled the pension once at
    retirement. For an earnings-related schedule that is right. For a means
    test it is not: the benefit reads the balance every year, and paying
    the full rate unconditionally is the *untested* control rather than the
    treatment -- so a joint search run under a means-tested spec would have
    been solving the wrong problem and reporting the answer as the right
    one.

    These are the guards that stop the two implementations diverging on the
    one spec the paper's argument lives in.
    """

    @staticmethod
    def _tested(spec):
        import dataclasses

        return dataclasses.replace(
            spec, social_security_formula="means_tested",
            pension_full_rate=0.293, pension_free_area=3.014,
            pension_taper=0.078)

    def test_it_agrees_with_the_reference_under_a_means_test(self,
                                                             setup) -> None:
        cfg, spec, strategies, paths, income = setup
        tested = self._tested(spec)
        keys = list(strategies)
        weights = np.stack([strategies[k].weights for k in keys])
        got = gp.BatchEvaluator(paths, tested, income, cfg).cec(weights, 5.0)
        for i, key in enumerate(keys):
            outcome = lc.simulate(paths, strategies[key], tested, income)
            want = ut.crra_certainty_equivalent(
                ut.bundle_from_outcome(outcome, cfg, tested), 5.0,
                float(cfg["utility"]["discount_factor"]),
                float(cfg["utility"]["bequest_weight"]),
                bool(cfg["utility"]["bequest_enabled"]))
            assert got[i] == pytest.approx(want, rel=1e-10), key

    def test_the_test_actually_bites_in_the_fast_path(self, setup) -> None:
        """An agreement test passes trivially if the means test never
        changes anything, so this checks the two specs give different
        answers before the agreement above is worth having."""
        cfg, spec, strategies, paths, income = setup
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        untested = gp.BatchEvaluator(paths, spec, income, cfg).cec(weights, 5.0)
        tested = gp.BatchEvaluator(paths, self._tested(spec), income,
                                   cfg).cec(weights, 5.0)
        assert not np.isclose(untested[0], tested[0], rtol=1e-6)

    def test_a_higher_balance_is_paid_a_smaller_pension(self, setup) -> None:
        """The taper, seen through the evaluator rather than asserted of it.

        The fixture's household retires on well under the free area, where
        every schedule collects the full rate and the test is asleep, so the
        balance is scaled until it straddles the threshold. A comparison run
        below the free area would pass whatever the code did.
        """
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        tested = self._tested(spec)
        key = list(strategies)[0]
        paid = []
        for scale in (1.0, 8.0):
            at = dataclasses.replace(tested, retirement_balance_scale=scale)
            outcome = lc.simulate(paths, strategies[key], at, income)
            paid.append((float(np.mean(outcome.wealth_at_retirement)),
                         float(np.mean(outcome.social_security))))
        (poor_w, poor_b), (rich_w, rich_b) = paid
        assert rich_w > poor_w
        assert rich_b < poor_b, paid

    def test_the_compulsory_contribution_is_carried_too(self, setup) -> None:
        """The other half of the same divergence, and the one that shipped.

        The fast path charged only the voluntary savings rate, so a regime
        mandating a second contribution stream arrived at the pension age
        with the wrong balance -- and because the omission was silent, two
        regimes differing only in that rate scored bit-identically and the
        duplicate was printed as a finding.
        """
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        mandated = dataclasses.replace(spec, super_guarantee_rate=0.12,
                                       super_contributions_tax=0.15)
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        got = gp.BatchEvaluator(paths, mandated, income, cfg).cec(weights,
                                                                  5.0)[0]
        outcome = lc.simulate(paths, strategies[key], mandated, income)
        want = ut.crra_certainty_equivalent(
            ut.bundle_from_outcome(outcome, cfg, mandated), 5.0,
            float(cfg["utility"]["discount_factor"]),
            float(cfg["utility"]["bequest_weight"]),
            bool(cfg["utility"]["bequest_enabled"]))
        assert got == pytest.approx(want, rel=1e-10)

    def test_mandating_a_contribution_moves_the_fast_path_at_all(
            self, setup) -> None:
        """An agreement test passes trivially if the mandate does nothing."""
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        mandated = dataclasses.replace(spec, super_guarantee_rate=0.12,
                                       super_contributions_tax=0.15)
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        voluntary = gp.BatchEvaluator(paths, spec, income, cfg).cec(weights,
                                                                    5.0)[0]
        both = gp.BatchEvaluator(paths, mandated, income, cfg).cec(weights,
                                                                   5.0)[0]
        assert not np.isclose(voluntary, both, rtol=1e-6)

    @staticmethod
    def _binding(spec):
        """A spec on which every field below has something to move.

        The means test has to bite (the fixture's household retires far
        below the free area) and the eligibility gate has to be open a few
        years (or the pre-eligibility share is a parameter of an empty
        window), or the "it does something" half of the guard is vacuous.
        """
        import dataclasses

        return dataclasses.replace(
            spec, social_security_formula="means_tested",
            pension_full_rate=0.293, pension_free_area=3.014,
            pension_taper=0.078, retirement_balance_scale=8.0,
            super_guarantee_rate=0.12, super_contributions_tax=0.15,
            benefit_start_age=spec.age_retire + 4,
            pre_eligibility_benefit_share=0.6)

    @pytest.mark.parametrize("field,value", [
        ("savings_rate", 0.15),
        ("super_guarantee_rate", 0.0),
        ("super_contributions_tax", 0.30),
        ("retirement_balance_scale", 4.0),
        ("pension_full_rate", 0.40),
        ("pension_free_area", 1.0),
        ("pension_taper", 0.05),
        ("benefit_start_age", 0),
        ("pre_eligibility_benefit_share", 0.0),
    ])
    def test_every_spec_field_the_fast_path_reads_is_read_the_same_way(
            self, setup, field, value) -> None:
        """Stated over the spec rather than over one attribute of it.

        Both defects found here were the same shape -- a field the reference
        simulator honours and the batched one silently drops -- and a guard
        naming one field would not have caught the next. This walks the
        fields that move a balance or a benefit and holds the two
        implementations to the same answer on each.
        """
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        base = self._binding(spec)
        if field == "benefit_start_age":
            value = int(base.age_retire)
        moved = dataclasses.replace(base, **{field: value})
        key = list(strategies)[0]
        weights = strategies[key].weights[None]

        def both(at):
            fast = float(gp.BatchEvaluator(paths, at, income,
                                           cfg).cec(weights, 5.0)[0])
            outcome = lc.simulate(paths, strategies[key], at, income)
            slow = float(ut.crra_certainty_equivalent(
                ut.bundle_from_outcome(outcome, cfg, at), 5.0,
                float(cfg["utility"]["discount_factor"]),
                float(cfg["utility"]["bequest_weight"]),
                bool(cfg["utility"]["bequest_enabled"])))
            return fast, slow

        fast_base, slow_base = both(base)
        fast_moved, slow_moved = both(moved)
        assert fast_base == pytest.approx(slow_base, rel=1e-10), field
        assert fast_moved == pytest.approx(slow_moved, rel=1e-10), field
        # And the field has to do something, or the agreement above is empty.
        assert not np.isclose(slow_base, slow_moved, rtol=1e-6), field

    def test_incidence_moves_neither_implementation(self, setup) -> None:
        """The paper's identity, held against both simulators at once.

        Charging the compulsory contribution to wages changes what the
        household gave up and not what it arrives with, so a
        retirement-window certainty equivalent cannot move. That is asserted
        in the paper as arithmetic; here it is checked on the fast path as
        well, because an evaluator that dropped the contribution entirely
        would also satisfy it -- and did.
        """
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        base = dataclasses.replace(self._binding(spec),
                                   super_guarantee_rate=0.12,
                                   super_contributions_tax=0.15)
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        scores = []
        for incidence in (0.0, 1.0):
            at = dataclasses.replace(base, super_incidence=incidence)
            scores.append(float(gp.BatchEvaluator(paths, at, income,
                                                  cfg).cec(weights, 5.0)[0]))
        assert scores[0] == pytest.approx(scores[1], rel=1e-12)
        # ...and the balance it produces is the one the mandate paid for,
        # which is what distinguishes the identity from the bug.
        assert not np.isclose(
            scores[0],
            float(gp.BatchEvaluator(paths,
                                    dataclasses.replace(
                                        base, super_guarantee_rate=0.0),
                                    income, cfg).cec(weights, 5.0)[0]),
            rtol=1e-6)

    def test_the_eligibility_gate_is_carried_too(self, setup) -> None:
        """Nothing is paid before the claiming age, and the fast path used
        to pay from the retirement date whatever the spec said."""
        import dataclasses

        cfg, spec, strategies, paths, income = setup
        gated = dataclasses.replace(spec, benefit_start_age=spec.age_retire + 4,
                                    pre_eligibility_benefit_share=0.0)
        key = list(strategies)[0]
        weights = strategies[key].weights[None]
        got = gp.BatchEvaluator(paths, gated, income, cfg).cec(weights, 5.0)[0]
        outcome = lc.simulate(paths, strategies[key], gated, income)
        want = ut.crra_certainty_equivalent(
            ut.bundle_from_outcome(outcome, cfg, gated), 5.0,
            float(cfg["utility"]["discount_factor"]),
            float(cfg["utility"]["bequest_weight"]),
            bool(cfg["utility"]["bequest_enabled"]))
        assert got == pytest.approx(want, rel=1e-10)
