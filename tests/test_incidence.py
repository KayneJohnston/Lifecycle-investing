"""Tests for the two dials of the incidence study.

The first draft of `src/incidence` assumed the two referee objections it
answers were the same objection -- that charging the guarantee would move
the household onto the assets test. It does not, and the failure was only
caught by running it. So most of these tests pin the *distinction*: that
incidence moves the working years and the balance dial moves the retiree's
problem, and that neither quietly does the other's job.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

from src import incidence as ic
from src import lifecycle as lc


def _spec(**over: object) -> lc.LifecycleSpec:
    return dataclasses.replace(
        lc.LifecycleSpec(social_security_formula="means_tested"), **over)


def _outcome(wealth: np.ndarray) -> types.SimpleNamespace:
    return types.SimpleNamespace(wealth_at_retirement=np.asarray(wealth,
                                                                 float))


class TestThresholds:
    def test_the_cutoff_is_where_the_taper_has_taken_the_whole_pension(
            self) -> None:
        spec = _spec()
        free, cut = ic.thresholds(spec)
        economy = float(spec.deterministic_income().mean())
        # At the cut-off the taper has withdrawn exactly the full rate.
        withdrawn = spec.pension_taper * (cut - free)
        assert withdrawn == pytest.approx(spec.pension_full_rate * economy)

    def test_a_zero_taper_never_cuts_anyone_off(self) -> None:
        _, cut = ic.thresholds(_spec(pension_taper=0.0))
        assert np.isinf(cut)

    def test_the_free_area_is_a_multiple_of_average_earnings(self) -> None:
        spec = _spec(pension_free_area=3.01)
        free, _ = ic.thresholds(spec)
        economy = float(spec.deterministic_income().mean())
        assert free / economy == pytest.approx(3.01)


class TestBands:
    def test_each_side_of_the_test_is_named(self) -> None:
        bands = ic.band_of(np.array([1.0, 5.0, 50.0]), 3.0, 7.0)
        assert list(bands) == list(ic.BANDS)

    def test_the_boundaries_belong_to_the_outer_bands(self) -> None:
        """A household exactly on the free area is not yet being tapered,
        and one exactly on the cut-off has nothing left to taper."""
        bands = ic.band_of(np.array([3.0, 7.0]), 3.0, 7.0)
        assert list(bands) == [ic.BANDS[0], ic.BANDS[2]]

    def test_bands_are_in_increasing_wealth_order(self) -> None:
        """`shape_verdict` reads monotonicity off this order, so it is not
        a presentational choice."""
        assert ic.BANDS[0].endswith("free area")
        assert "band" in ic.BANDS[1]
        assert ic.BANDS[2].endswith("cut-off")


class TestIncidenceIsNotTheBalance:
    """The finding the module exists to state, pinned at the simulator."""

    def test_charging_the_guarantee_leaves_the_balance_untouched(self) -> None:
        spec = _spec(super_guarantee_rate=0.12)
        free_arm = dataclasses.replace(spec, super_incidence=0.0)
        paid_arm = dataclasses.replace(spec, super_incidence=1.0)
        assert free_arm.super_net_rate == paid_arm.super_net_rate

    def test_charging_the_guarantee_lowers_working_consumption(self) -> None:
        spec = _spec(super_guarantee_rate=0.12)
        income = spec.deterministic_income()[:spec.n_working]
        free_cut = 0.0 * income
        paid_cut = spec.super_net_rate * income
        assert paid_cut.sum() > free_cut.sum() > -1.0

    def test_the_incidence_bounds_are_enforced(self) -> None:
        with pytest.raises(ValueError):
            _spec(super_incidence=1.5)
        with pytest.raises(ValueError):
            _spec(super_incidence=-0.1)

    def test_the_balance_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            _spec(retirement_balance_scale=0.0)
        with pytest.raises(ValueError):
            _spec(retirement_balance_scale=-1.0)


class TestSweeps:
    def _score(self, outcome: object) -> dict:
        return {"cec": 1.0, "cec_lifetime": 1.0}

    def test_the_incidence_sweep_is_the_cross_product(self) -> None:
        frame = ic.sweep(
            lambda a, e: _outcome(np.full(4, 10.0)), lambda a: _spec(),
            [0.0, 0.5, 1.0], [0.0, 1.0], self._score, log_every=0)
        assert len(frame) == 6
        assert set(frame["incidence"]) == {0.0, 0.5, 1.0}

    def test_the_balance_sweep_is_the_cross_product(self) -> None:
        frame = ic.balance_sweep(
            lambda s, e: _outcome(np.full(4, 10.0)), lambda s: _spec(),
            [0.1, 1.0], [0.0, 0.5, 1.0], self._score, log_every=0)
        assert len(frame) == 6
        assert set(frame["scale"]) == {0.1, 1.0}

    def test_the_band_shares_sum_to_one(self) -> None:
        frame = ic.sweep(
            lambda a, e: _outcome(np.array([1.0, 5.0, 50.0, 90.0])),
            lambda a: _spec(), [0.0], [1.0], self._score, log_every=0)
        shares = [c for c in frame.columns if c.startswith("share_")]
        assert len(shares) == 3
        assert frame[shares].sum(axis=1).iloc[0] == pytest.approx(1.0)

    def test_an_empty_grid_gives_an_empty_frame(self) -> None:
        frame = ic.sweep(lambda a, e: _outcome(np.zeros(1)), lambda a: _spec(),
                         [], [1.0], self._score, log_every=0)
        assert not len(frame)


class TestOptima:
    def _frame(self) -> pd.DataFrame:
        rows = []
        for scale, wealth in ((0.1, 1.0), (0.5, 5.0), (1.0, 50.0)):
            for equity, cec in ((0.0, 1.0), (0.5, 2.0), (1.0, 1.5)):
                rows.append({"scale": scale, "equity": equity, "cec": cec,
                             "median_wealth": wealth, "free_area": 3.0,
                             "cutoff": 7.0,
                             "share_inside_the_taper_band": 0.2})
        return pd.DataFrame.from_records(rows)

    def test_it_picks_the_best_equity_at_each_balance(self) -> None:
        best = ic.optimum_by_balance(self._frame())
        assert list(best["equity"]) == [0.5, 0.5, 0.5]

    def test_it_classifies_the_position_from_the_winning_row(self) -> None:
        best = ic.optimum_by_balance(self._frame())
        assert list(best["position"]) == list(ic.BANDS)

    def test_the_optimum_by_incidence_defaults_to_the_lifetime_measure(
            self) -> None:
        """Selecting on the retirement window would give a flat line by
        construction, because that window cannot see a working-life
        transfer at all."""
        frame = pd.DataFrame.from_records([
            {"incidence": 0.0, "equity": 0.2, "cec": 9.0, "cec_lifetime": 1.0,
             "median_wealth": 10.0, "free_area": 3.0, "cutoff": 7.0},
            {"incidence": 0.0, "equity": 0.8, "cec": 1.0, "cec_lifetime": 9.0,
             "median_wealth": 10.0, "free_area": 3.0, "cutoff": 7.0},
        ])
        assert float(ic.optimum_by_incidence(frame)["equity"].iloc[0]) == 0.8
        assert float(
            ic.optimum_by_incidence(frame, "cec")["equity"].iloc[0]) == 0.2

    def test_the_optimum_keeps_no_duplicate_columns(self) -> None:
        """`cec` appears both as the scoring column and in the keep list."""
        frame = pd.DataFrame.from_records([
            {"incidence": 0.0, "equity": 0.5, "cec": 1.0, "cec_lifetime": 1.0,
             "median_wealth": 10.0, "free_area": 3.0, "cutoff": 7.0}])
        best = ic.optimum_by_incidence(frame, "cec")
        assert len(best.columns) == len(set(best.columns))


class TestVerdict:
    def _optima(self, free_equity: float, paid_equity: float,
                paid_cec: float = 0.9) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"incidence": 0.0, "equity": free_equity, "cec": 1.0,
             "cec_lifetime": 1.0, "mean_working_consumption": 1.0,
             "median_wealth": 40.0, "free_area": 3.0, "cutoff": 7.0,
             "share_inside_the_taper_band": 0.0,
             "share_above_the_cut-off": 1.0},
            {"incidence": 1.0, "equity": paid_equity, "cec": 1.0,
             "cec_lifetime": paid_cec, "mean_working_consumption": 0.9,
             "median_wealth": 40.0, "free_area": 3.0, "cutoff": 7.0,
             "share_inside_the_taper_band": 0.0,
             "share_above_the_cut-off": 1.0},
        ])

    def test_it_reports_that_the_balance_did_not_move(self) -> None:
        found = ic.verdict(self._optima(1.0, 1.0))
        assert found["balance_unchanged"]
        assert not found["reaches_the_test"]

    def test_it_reports_the_retirement_measure_as_invariant(self) -> None:
        found = ic.verdict(self._optima(1.0, 1.0))
        assert found["retirement_cec_invariant"]
        assert found["cec_lifetime_fall_pct"] < 0.0

    def test_it_sees_the_allocation_move(self) -> None:
        found = ic.verdict(self._optima(1.0, 0.6))
        assert found["equity_falls_when_charged"]
        assert not found["equity_unchanged"]

    def test_it_sees_the_household_reach_the_test(self) -> None:
        optima = self._optima(1.0, 1.0)
        optima.loc[1, "median_wealth"] = 5.0
        optima.loc[1, "share_inside_the_taper_band"] = 0.6
        found = ic.verdict(optima)
        assert found["reaches_the_test"]
        assert not found["balance_unchanged"]

    def test_an_empty_frame_is_not_measured(self) -> None:
        assert ic.verdict(pd.DataFrame()) == {"measured": False}


class TestShapeVerdict:
    def _profile(self, below: float, band: float, above: float,
                 ) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"scale": 0.1, "position": ic.BANDS[0], "equity": below,
             "median_wealth": 1.0, "cec": 1.0},
            {"scale": 0.5, "position": ic.BANDS[1], "equity": band,
             "median_wealth": 5.0, "cec": 1.0},
            {"scale": 1.0, "position": ic.BANDS[2], "equity": above,
             "median_wealth": 50.0, "cec": 1.0},
        ])

    def test_the_prediction_holds_when_the_band_wants_most_equity(
            self) -> None:
        found = ic.shape_verdict(self._profile(0.6, 0.9, 0.4),
                                 [0.0, 0.4, 0.6, 0.9, 1.0])
        assert found["prediction_holds"]
        assert found["minimum_above_the_cutoff"]
        assert found["band_beats_above"]

    def test_the_prediction_fails_when_the_band_wants_least(self) -> None:
        found = ic.shape_verdict(self._profile(0.9, 0.3, 0.7),
                                 [0.0, 0.3, 0.7, 0.9, 1.0])
        assert not found["prediction_holds"]
        assert found["above_beats_band"]
        assert found["lowest_band"] == ic.BANDS[1]

    def test_a_gap_inside_the_tolerance_is_not_a_gap(self) -> None:
        found = ic.shape_verdict(
            self._profile(0.6, 0.62, 0.60), [0.6, 0.62, 0.64],
            tolerance=0.05)
        assert not found["band_beats_above"]
        assert not found["above_beats_band"]
        assert not found["prediction_holds"]

    def test_it_notices_when_every_optimum_is_on_the_ceiling(self) -> None:
        found = ic.shape_verdict(self._profile(1.0, 1.0, 1.0),
                                 [0.0, 0.5, 1.0])
        assert found["all_at_ceiling"]
        assert found["all_at_grid_edge"]

    def test_it_flags_a_single_optimum_on_the_grid_edge(self) -> None:
        found = ic.shape_verdict(self._profile(0.5, 1.0, 0.5),
                                 [0.0, 0.5, 1.0])
        assert found["any_at_grid_edge"]
        assert not found["all_at_grid_edge"]
        assert not found["all_at_ceiling"]

    def test_an_interior_profile_is_flagged_as_interior(self) -> None:
        found = ic.shape_verdict(self._profile(0.5, 0.7, 0.4),
                                 [0.0, 0.4, 0.5, 0.7, 1.0])
        assert not found["any_at_grid_edge"]

    def test_one_band_alone_cannot_have_a_shape(self) -> None:
        one = pd.DataFrame.from_records([
            {"scale": 1.0, "position": ic.BANDS[2], "equity": 1.0,
             "median_wealth": 50.0, "cec": 1.0}])
        found = ic.shape_verdict(one, [0.0, 1.0])
        assert found["measured"]
        assert "prediction_holds" not in found

    def test_an_empty_profile_is_not_measured(self) -> None:
        assert ic.shape_verdict(pd.DataFrame())["measured"] is False


class TestBandProfile:
    def test_it_is_ordered_by_wealth(self) -> None:
        rows = []
        for scale, wealth in ((1.0, 50.0), (0.1, 1.0), (0.5, 5.0)):
            rows.append({"scale": scale, "equity": 0.5, "cec": 1.0,
                         "median_wealth": wealth, "free_area": 3.0,
                         "cutoff": 7.0})
        profile = ic.band_profile(pd.DataFrame.from_records(rows))
        assert list(profile["median_wealth"]) == [1.0, 5.0, 50.0]

    def test_position_is_reported_as_a_multiple_of_the_cutoff(self) -> None:
        rows = [{"scale": 1.0, "equity": 0.5, "cec": 1.0,
                 "median_wealth": 14.0, "free_area": 3.0, "cutoff": 7.0}]
        profile = ic.band_profile(pd.DataFrame.from_records(rows))
        assert float(profile["over_cutoff"].iloc[0]) == pytest.approx(2.0)


class TestBridgeVerdict:
    """The control that stops a hole in the floor being called a taper."""

    @staticmethod
    def _profile(equities: list) -> pd.DataFrame:
        rows = []
        for (scale, wealth, position), equity in zip(
                ((0.1, 1.0, ic.BANDS[0]), (0.5, 5.0, ic.BANDS[1]),
                 (1.0, 50.0, ic.BANDS[2])), equities):
            rows.append({"scale": scale, "median_wealth": wealth,
                         "position": position, "equity": equity, "cec": 1.0})
        return pd.DataFrame.from_records(rows)

    def _pair(self, aligned: list, bridged: list) -> tuple:
        profiles = {ic.ARMS[0]: self._profile(aligned),
                    ic.ARMS[1]: self._profile(bridged)}
        grid = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        shapes = {a: ic.shape_verdict(b, grid) for a, b in profiles.items()}
        return profiles, shapes

    def test_it_measures_the_average_gap(self) -> None:
        found = ic.bridge_verdict(*self._pair([0.6, 0.9, 0.4],
                                              [0.2, 0.4, 0.2]))
        assert found["bridge_lowers_equity"]
        assert found["mean_equity_gap"] == pytest.approx(
            ((0.6 - 0.2) + (0.9 - 0.4) + (0.4 - 0.2)) / 3)

    def test_it_names_where_the_gap_is_widest(self) -> None:
        found = ic.bridge_verdict(*self._pair([0.6, 0.9, 0.4],
                                              [0.6, 0.0, 0.4]))
        assert found["worst_wealth"] == pytest.approx(5.0)
        assert found["worst_position"] == ic.BANDS[1]
        assert found["max_equity_gap"] == pytest.approx(0.9)

    def test_a_bridge_that_raises_equity_is_reported_that_way(self) -> None:
        found = ic.bridge_verdict(*self._pair([0.2, 0.4, 0.2],
                                              [0.6, 0.9, 0.4]))
        assert found["bridge_raises_equity"]
        assert not found["bridge_lowers_equity"]

    def test_an_identical_pair_moves_nothing(self) -> None:
        found = ic.bridge_verdict(*self._pair([0.6, 0.9, 0.4],
                                              [0.6, 0.9, 0.4]))
        assert not found["bridge_lowers_equity"]
        assert not found["bridge_raises_equity"]
        assert found["mean_equity_gap"] == pytest.approx(0.0)
        assert not found["the_bridge_changes_the_verdict"]

    def test_it_notices_when_the_bridge_flips_the_shape_test(self) -> None:
        # Aligned: the prediction holds (band highest, cut-off lowest).
        # Bridged: it does not (the band wants least).
        found = ic.bridge_verdict(*self._pair([0.6, 0.9, 0.4],
                                              [0.9, 0.2, 0.6]))
        assert found["aligned_prediction_holds"]
        assert not found["bridged_prediction_holds"]
        assert found["the_bridge_changes_the_verdict"]

    def test_one_arm_alone_cannot_be_compared(self) -> None:
        profiles, shapes = self._pair([0.6, 0.9, 0.4], [0.6, 0.9, 0.4])
        del profiles[ic.ARMS[1]]
        assert ic.bridge_verdict(profiles, shapes) == {"measured": False}

    def test_an_empty_arm_is_not_measured(self) -> None:
        profiles, shapes = self._pair([0.6, 0.9, 0.4], [0.6, 0.9, 0.4])
        profiles[ic.ARMS[1]] = profiles[ic.ARMS[1]].iloc[0:0]
        assert ic.bridge_verdict(profiles, shapes) == {"measured": False}


class TestBalanceSweepArms:
    def test_the_arm_is_carried_into_every_row(self) -> None:
        frame = ic.balance_sweep(
            lambda s, e: _outcome(np.full(4, 10.0)), lambda s: _spec(),
            [0.1, 1.0], [0.0, 1.0], lambda o: {"cec": 1.0},
            log_every=0, arm=ic.ARMS[1])
        assert set(frame["arm"]) == {ic.ARMS[1]}

    def test_no_arm_column_appears_when_none_is_given(self) -> None:
        frame = ic.balance_sweep(
            lambda s, e: _outcome(np.full(4, 10.0)), lambda s: _spec(),
            [0.1], [1.0], lambda o: {"cec": 1.0}, log_every=0)
        assert "arm" not in frame.columns

    def test_the_profile_keeps_the_arm(self) -> None:
        frame = ic.balance_sweep(
            lambda s, e: _outcome(np.full(4, 10.0 * s)), lambda s: _spec(),
            [0.1, 1.0], [0.0, 1.0], lambda o: {"cec": 1.0},
            log_every=0, arm=ic.ARMS[0])
        assert "arm" in ic.band_profile(frame).columns


class TestAControlThatMovesEverythingIsNotAgreement:
    """The bug this guards against reported a hundred-point swing as "no
    difference", because both arms happened to carry the same shape label:
    one because the prediction failed, the other because every optimum sat
    on the grid ceiling and there was no shape left to test."""

    @staticmethod
    def _profiles(left: list, right: list) -> tuple:
        def frame(equities: list) -> pd.DataFrame:
            return pd.DataFrame.from_records([
                {"scale": s, "median_wealth": w, "position": pos,
                 "equity": e, "cec": 1.0}
                for (s, w, pos), e in zip(
                    ((0.1, 1.0, ic.BANDS[0]), (0.5, 5.0, ic.BANDS[1]),
                     (1.0, 50.0, ic.BANDS[2])), equities)])
        profiles = {ic.ARMS[0]: frame(left), ic.ARMS[2]: frame(right)}
        grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        shapes = {a: ic.shape_verdict(b, grid) for a, b in profiles.items()}
        return profiles, shapes

    def test_a_hundred_point_swing_changes_the_verdict(self) -> None:
        profiles, shapes = self._profiles([0.0, 0.0, 0.4], [1.0, 1.0, 1.0])
        found = ic.rule_verdict(profiles, shapes)
        assert not found["shape_label_flips"]     # both read "fails"
        assert found["moves_the_allocation"]
        assert found["changes_the_verdict"]

    def test_the_ceiling_arm_is_flagged_as_untestable(self) -> None:
        profiles, shapes = self._profiles([0.0, 0.0, 0.4], [1.0, 1.0, 1.0])
        found = ic.rule_verdict(profiles, shapes)
        assert found["bridged_untestable"]
        assert not found["aligned_untestable"]

    def test_arms_that_truly_agree_change_nothing(self) -> None:
        profiles, shapes = self._profiles([0.0, 0.0, 0.4], [0.0, 0.0, 0.4])
        found = ic.rule_verdict(profiles, shapes)
        assert not found["moves_the_allocation"]
        assert not found["shape_label_flips"]
        assert not found["changes_the_verdict"]


class TestABandIsSummarisedByItsTypicalHousehold:
    """One rich household must not speak for a whole band.

    Summarising each band by its maximum let a single point at the top of
    the wealth grid decide the shape test: with wanted equity 0% at six
    balances past the cut-off and 55% at the seventh, the band reported
    55%, and the comparison against the taper band was a comparison
    against an outlier.
    """

    @staticmethod
    def _profile(above: list) -> pd.DataFrame:
        rows = [{"scale": 0.1, "median_wealth": 1.0,
                 "position": ic.BANDS[0], "equity": 0.0, "cec": 1.0},
                {"scale": 0.5, "median_wealth": 5.0,
                 "position": ic.BANDS[1], "equity": 0.0, "cec": 1.0}]
        for i, equity in enumerate(above):
            rows.append({"scale": 1.0 + i, "median_wealth": 10.0 + i,
                         "position": ic.BANDS[2], "equity": equity,
                         "cec": 1.0})
        return pd.DataFrame.from_records(rows)

    def test_one_outlier_does_not_speak_for_its_band(self) -> None:
        found = ic.shape_verdict(
            self._profile([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.55]),
            [0.0, 0.1, 0.55, 1.0])
        assert found["equity_above_the_cut_off"] == pytest.approx(0.0)

    def test_the_range_is_kept_alongside_the_median(self) -> None:
        found = ic.shape_verdict(
            self._profile([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.55]),
            [0.0, 0.1, 0.55, 1.0])
        assert found["equity_above_the_cut_off_max"] == pytest.approx(0.55)
        assert found["equity_above_the_cut_off_min"] == pytest.approx(0.0)
        assert found["points_above_the_cut_off"] == 7

    def test_a_band_that_really_wants_equity_still_says_so(self) -> None:
        found = ic.shape_verdict(self._profile([0.8, 0.9, 1.0]),
                                 [0.0, 0.8, 0.9, 1.0])
        assert found["equity_above_the_cut_off"] == pytest.approx(0.9)


class TestAFlatProfileHasNoLowestBand:
    """`min` on equal values invents a winner, and the prose then states it.

    "The optimum is lowest below the free area" is a claim about the shape
    of the answer. When every band wants the same share it is a false one,
    and it is exactly the sentence a referee would check first.
    """

    @staticmethod
    def _profile(below: float, band: float, above: float) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"scale": 0.1, "median_wealth": 1.0, "position": ic.BANDS[0],
             "equity": below, "cec": 1.0},
            {"scale": 0.5, "median_wealth": 5.0, "position": ic.BANDS[1],
             "equity": band, "cec": 1.0},
            {"scale": 1.0, "median_wealth": 50.0, "position": ic.BANDS[2],
             "equity": above, "cec": 1.0}])

    def test_an_all_zero_profile_is_flat(self) -> None:
        found = ic.shape_verdict(self._profile(0.0, 0.0, 0.0), [0.0, 0.5, 1.0])
        assert found["flat_across_bands"]
        assert found["lowest_band"] is None
        assert not found["minimum_above_the_cutoff"]
        assert found["band_spread"] == pytest.approx(0.0)
        assert found["common_equity"] == pytest.approx(0.0)

    def test_a_spread_inside_the_tolerance_is_still_flat(self) -> None:
        found = ic.shape_verdict(self._profile(0.40, 0.42, 0.44),
                                 [0.4, 0.42, 0.44], tolerance=0.05)
        assert found["flat_across_bands"]
        assert found["lowest_band"] is None

    def test_a_real_difference_still_names_the_lowest_band(self) -> None:
        found = ic.shape_verdict(self._profile(0.6, 0.9, 0.2),
                                 [0.2, 0.6, 0.9, 1.0])
        assert not found["flat_across_bands"]
        assert found["lowest_band"] == ic.BANDS[2]
        assert found["minimum_above_the_cutoff"]
