"""Whether the rule divide is a property of the mechanism or of one household.

Section 8 of the paper finds a corner at zero equity under a rule that
never reads the balance, and the whole portfolio under one that does, on a
single renter scaled onto the assets test. Corollary 1 says that should
hold for any household the test operates on, because no threshold appears
in the derivation. This module is the test, and these are the tests of the
test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import typology as ty


def _divide(**cells: float) -> pd.DataFrame:
    """A divide table from ``label_position=gap`` shorthand."""
    rows = []
    for key, (blind, reading) in cells.items():
        household, position = key.split("__")
        rows.append({
            "household": household,
            "household_label": household.replace("_", " "),
            "position": position.replace("_", " "),
            "balances": 3,
            f"equity_{ty.BLIND}": blind,
            f"equity_{ty.READING}": reading,
        })
    return pd.DataFrame.from_records(rows)


class TestTheHouseholds:
    """Four types, built from the single homeowner's sourced figures."""

    KW = dict(single_free_area=3.01, renter_free_area=5.63,
              single_full_rate=0.293, couple_free_area_ratio=1.50,
              couple_renter_free_area_ratio=1.23, couple_rate_ratio=1.51)

    def test_there_are_four_of_them(self) -> None:
        assert len(ty.households(**self.KW)) == 4

    def test_a_renter_faces_a_higher_free_area_than_a_homeowner(self) -> None:
        """The family home is exempt, so a renter holding the same wealth in
        the portfolio is assessed later. This is the largest single feature
        of the real system and the model's household owns no home."""
        by = {w.key: w for w in ty.households(**self.KW)}
        assert by["single_renter"].free_area > by["single_homeowner"].free_area

    def test_a_couple_faces_a_higher_free_area_and_a_higher_rate(self
                                                                 ) -> None:
        by = {w.key: w for w in ty.households(**self.KW)}
        assert by["couple_homeowner"].free_area \
            > by["single_homeowner"].free_area
        assert by["couple_homeowner"].full_rate \
            > by["single_homeowner"].full_rate

    def test_the_cut_out_follows_the_taper(self) -> None:
        """Where the pension reaches zero is the free area plus the rate
        divided by the taper, so a steeper taper cuts out sooner."""
        who = ty.households(**self.KW)[0]
        assert who.cut_out(0.078) == pytest.approx(3.01 + 0.293 / 0.078)
        assert who.cut_out(0.156) < who.cut_out(0.078)

    def test_a_ratio_of_one_makes_the_couple_the_single(self) -> None:
        """The sweep's lower bracket: a couple assessed on the single
        thresholds. It has to be reachable, because it is the case that
        says how much of the couple result is the ratio."""
        flat = dict(self.KW, couple_free_area_ratio=1.0,
                    couple_renter_free_area_ratio=1.0, couple_rate_ratio=1.0)
        by = {w.key: w for w in ty.households(**flat)}
        assert by["couple_homeowner"].free_area \
            == pytest.approx(by["single_homeowner"].free_area)


class TestWhichBandABalanceSitsIn:
    """The same balance is in different regimes for a renter and a
    homeowner, which is the reason the types are crossed rather than
    pooled."""

    def test_it_reads_each_row_against_its_own_thresholds(self) -> None:
        wealth = np.array([1.0, 5.0, 20.0])
        free = np.array([3.0, 3.0, 3.0])
        cut = np.array([7.0, 7.0, 7.0])
        assert list(ty.band_of(wealth, free, cut)) == [
            "below the free area", "inside the taper band",
            "above the cut-off"]

    def test_the_same_balance_lands_differently_for_two_types(self) -> None:
        wealth = np.array([5.0, 5.0])
        free = np.array([3.0, 6.0])            # homeowner, renter
        cut = np.array([7.0, 10.0])
        assert list(ty.band_of(wealth, free, cut)) == [
            "inside the taper band", "below the free area"]

    def test_the_free_area_itself_is_below_not_inside(self) -> None:
        assert ty.band_of(np.array([3.0]), np.array([3.0]),
                          np.array([7.0]))[0] == "below the free area"


class TestTheVerdict:
    """Two questions with different answers: whether the *sign* of the gap
    survives the household, which is Corollary 1's claim, and whether the
    *level* does, which is weaker and which the thresholds move."""

    def test_a_divide_that_holds_everywhere_reports_as_holding(self) -> None:
        found = ty.verdict(_divide(
            single__inside_the_taper_band=(0.0, 1.0),
            couple__inside_the_taper_band=(0.0, 1.0),
            single__above_the_cut_off=(0.0, 1.0)))
        assert found["divide_holds_everywhere"]
        assert found["cells_holding"] == found["cells"] == 3
        assert found["narrowest_gap"] == pytest.approx(1.0)

    def test_one_cell_going_the_other_way_breaks_it(self) -> None:
        """The point of running the sweep is that it can come out this way,
        and a verdict that could not say so would not be a test."""
        found = ty.verdict(_divide(
            single__inside_the_taper_band=(0.0, 1.0),
            couple__inside_the_taper_band=(0.8, 0.3)))
        assert not found["divide_holds_everywhere"]
        assert found["cells_holding"] == 1
        assert "couple" in found["worst_cell"]

    def test_a_tie_does_not_count_as_holding(self) -> None:
        """Two rules wanting the same share is not a divide."""
        found = ty.verdict(_divide(
            single__inside_the_taper_band=(0.5, 0.5)))
        assert not found["divide_holds_everywhere"]

    def test_the_corner_is_read_where_the_test_operates(self) -> None:
        """Corollary 1 predicts a corner inside the band and above the
        cut-off, where consumption is falling or flat in the return. Below
        the free area the pension is paid in full and it predicts nothing,
        so reading the corner there would test a claim the corollary does
        not make -- which is what the first version of this verdict did."""
        found = ty.verdict(_divide(
            single__below_the_free_area=(0.6, 1.0),
            single__inside_the_taper_band=(0.0, 1.0),
            single__above_the_cut_off=(0.0, 1.0)))
        assert found["blind_at_zero_where_it_binds"]
        assert not found["blind_always_at_zero"]

    def test_a_corner_that_fails_where_it_binds_is_reported(self) -> None:
        found = ty.verdict(_divide(
            single__inside_the_taper_band=(0.4, 1.0)))
        assert not found["blind_at_zero_where_it_binds"]
        assert found["blind_highest"] == pytest.approx(0.4)

    def test_an_empty_table_is_not_measured(self) -> None:
        assert ty.verdict(pd.DataFrame()) == {"measured": False}

    def test_a_table_without_both_rules_is_not_measured(self) -> None:
        assert ty.verdict(pd.DataFrame([{"household": "a",
                                         "position": "b"}])) \
            == {"measured": False}


class TestTheSweep:
    """One row per household, rule, balance and share, each carrying the
    thresholds it was scored against."""

    KW = TestTheHouseholds.KW

    @staticmethod
    def _score(outcome):
        return {"cec": float(outcome), "median_wealth": float(outcome)}

    def test_every_combination_is_scored_once(self) -> None:
        types = ty.households(**self.KW)[:2]
        seen = []

        def _sim(who, rule, scale, equity):
            seen.append((who.key, rule, scale, equity))
            return scale * 10.0

        frame = ty.sweep(_sim, self._score, types,
                         [ty.BLIND, ty.READING], [0.5, 1.0], [0.0, 1.0],
                         taper=0.078, log_every=0)
        assert len(frame) == 2 * 2 * 2 * 2 == len(seen)
        assert len(set(seen)) == len(seen)

    def test_each_row_carries_its_own_thresholds(self) -> None:
        types = ty.households(**self.KW)
        frame = ty.sweep(lambda w, r, s, e: s, self._score, types,
                         [ty.BLIND], [1.0], [1.0], taper=0.078, log_every=0)
        by = frame.set_index("household")["free_area"].to_dict()
        assert by["single_renter"] > by["single_homeowner"]
        assert (frame["cutoff"] > frame["free_area"]).all()

    def test_the_position_is_computed_against_those_thresholds(self) -> None:
        types = ty.households(**self.KW)[:1]
        frame = ty.sweep(lambda w, r, s, e: s, self._score, types,
                         [ty.BLIND], [1.0, 100.0], [1.0],
                         taper=0.078, log_every=0)
        assert set(frame["position"]) == {"below the free area",
                                          "above the cut-off"}


class TestWhatEachTypeWants:
    def test_the_best_share_is_picked_per_type_rule_and_balance(self) -> None:
        frame = pd.DataFrame.from_records([
            {"household": "a", "household_label": "a", "rule": ty.BLIND,
             "scale": 1.0, "equity": e, "cec": c, "position": "x"}
            for e, c in ((0.0, 1.0), (0.5, 0.9), (1.0, 0.8))])
        best = ty.wanted_by_type(frame)
        assert len(best) == 1
        assert float(best["equity"].iloc[0]) == pytest.approx(0.0)

    def test_the_divide_table_puts_the_two_rules_side_by_side(self) -> None:
        best = pd.DataFrame.from_records([
            {"household": "a", "household_label": "a", "rule": ty.BLIND,
             "scale": 1.0, "equity": 0.0, "position": "inside the taper band"},
            {"household": "a", "household_label": "a", "rule": ty.READING,
             "scale": 1.0, "equity": 1.0, "position": "inside the taper band"},
        ])
        table = ty.divide_table(best)
        assert len(table) == 1
        assert float(table[f"equity_{ty.BLIND}"].iloc[0]) == 0.0
        assert float(table[f"equity_{ty.READING}"].iloc[0]) == 1.0


class TestTheThresholdSweep:
    """The couple thresholds are ratios of the single figures rather than
    separately sourced dollars, so the answer has to be shown not to depend
    on the ratio -- the treatment Section 7.1 gives the pre-eligibility
    payment."""

    def test_it_reports_one_row_per_ratio(self) -> None:
        table = ty.threshold_sensitivity(
            lambda r: _divide(a__inside_the_taper_band=(0.0, 1.0)),
            [1.0, 1.5, 2.0])
        assert list(table["free_area_ratio"]) == [1.0, 1.5, 2.0]
        assert table["divide_holds_everywhere"].all()

    def test_a_ratio_that_breaks_the_divide_is_visible(self) -> None:
        def _run(ratio: float):
            gap = (0.0, 1.0) if ratio < 2.0 else (0.9, 0.2)
            return _divide(a__inside_the_taper_band=gap)

        table = ty.threshold_sensitivity(_run, [1.0, 2.0])
        assert list(table["divide_holds_everywhere"]) == [True, False]
