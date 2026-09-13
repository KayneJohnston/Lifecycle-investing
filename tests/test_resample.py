"""Resampling the panel, which is the interval Section 10.3 says it owes.

That section reports a delete-one-country jackknife and then concedes two
things about it: the markets are not independent draws, and the deletions
overlap in construction because the international sleeve is a
leave-one-out average. Both make its standard error too narrow. These
tests are of the machinery that prices the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import resample as rs


def _cells(**gaps: float) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{"system": k.split("__")[0], "rule": k.split("__")[1],
          "gap_pct": v} for k, v in gaps.items()])


class TestTheDraw:
    MARKETS = tuple("ABCDEFGHIJKLMNOP")          # sixteen, as the panel is

    def test_it_draws_as_many_as_it_was_given(self) -> None:
        got = rs.draw(self.MARKETS, np.random.default_rng(0))
        assert len(got) == len(self.MARKETS)

    def test_every_pick_is_from_the_panel(self) -> None:
        got = rs.draw(self.MARKETS, np.random.default_rng(1))
        assert set(got) <= set(self.MARKETS)

    def test_it_draws_with_replacement(self) -> None:
        """The point of the exercise. A draw that could not repeat a market
        would be a permutation, which carries no information about
        sampling a population of markets at all."""
        seen_repeat = False
        for seed in range(20):
            got = rs.draw(self.MARKETS, np.random.default_rng(seed))
            if len(set(got)) < len(got):
                seen_repeat = True
                break
        assert seen_repeat

    def test_it_is_reproducible_from_the_seed(self) -> None:
        a = rs.draw(self.MARKETS, np.random.default_rng(7))
        b = rs.draw(self.MARKETS, np.random.default_rng(7))
        assert a == b


class TestTheBootstrap:
    def test_it_records_one_block_per_replicate(self) -> None:
        seen = []

        def _gaps(markets):
            seen.append(list(markets))
            return _cells(us__fixed=1.0, au__fixed=-1.0)

        frame = rs.bootstrap(_gaps, list("ABCD"), replicates=5, seed=3,
                             log_every=0)
        assert len(seen) == 5
        assert sorted(frame["replicate"].unique()) == [0, 1, 2, 3, 4]
        assert len(frame) == 5 * 2

    def test_the_draws_are_not_all_the_same_panel(self) -> None:
        """If they were, the spread would be Monte Carlo error and the
        document would be measuring the wrong thing."""
        frame = rs.bootstrap(lambda m: _cells(us__fixed=1.0),
                             list("ABCDEFGHIJKLMNOP"), replicates=12,
                             seed=5, log_every=0)
        assert frame["draw"].nunique() > 1

    def test_it_carries_how_many_distinct_markets_a_draw_held(self) -> None:
        frame = rs.bootstrap(lambda m: _cells(us__fixed=1.0),
                             list("ABCDEFGHIJKLMNOP"), replicates=8,
                             seed=11, log_every=0)
        assert (frame["distinct_markets"] <= 16).all()
        assert (frame["distinct_markets"] >= 1).all()

    def test_no_replicates_gives_an_empty_frame(self) -> None:
        assert not len(rs.bootstrap(lambda m: _cells(a__b=1.0), list("AB"),
                                    replicates=0, seed=1, log_every=0))


class TestTheInterval:
    @staticmethod
    def _spread(values, system="au", rule="fixed"):
        return pd.DataFrame.from_records(
            [{"replicate": i, "system": system, "rule": rule,
              "gap_pct": v} for i, v in enumerate(values)])

    def test_it_reports_a_percentile_interval(self) -> None:
        """Percentile and not normal-approximation: the reason for running
        this is that the section distrusts the normal approximation its own
        deletions support."""
        values = np.linspace(-20.0, 0.0, 101)
        band = rs.interval(self._spread(values), _cells(au__fixed=-10.0))
        row = band.iloc[0]
        assert row["ci_low"] == pytest.approx(np.quantile(values, 0.025))
        assert row["ci_high"] == pytest.approx(np.quantile(values, 0.975))

    def test_it_keeps_the_point_estimate_beside_the_spread(self) -> None:
        band = rs.interval(self._spread([-9.0, -11.0]),
                           _cells(au__fixed=-10.0))
        assert float(band["gap_pct"].iloc[0]) == pytest.approx(-10.0)
        assert float(band["boot_mean"].iloc[0]) == pytest.approx(-10.0)

    def test_it_counts_the_panels_that_keep_the_sign(self) -> None:
        """The statistic the paper leads with elsewhere, for the same
        reason: it assumes nothing about how the replicates scatter."""
        band = rs.interval(self._spread([-3.0, -2.0, 1.0, -1.0]),
                           _cells(au__fixed=-2.0))
        assert float(band["share_keeping_sign"].iloc[0]) == pytest.approx(0.75)

    def test_an_interval_straddling_zero_is_not_excluded(self) -> None:
        band = rs.interval(self._spread(np.linspace(-5.0, 5.0, 101)),
                           _cells(au__fixed=0.5))
        assert not bool(band["excludes_zero"].iloc[0])

    def test_an_interval_clear_of_zero_is_excluded(self) -> None:
        band = rs.interval(self._spread(np.linspace(-20.0, -8.0, 101)),
                           _cells(au__fixed=-13.0))
        assert bool(band["excludes_zero"].iloc[0])


class TestAgainstTheJackknife:
    """The number Section 10.3 owes a reader: it concedes its standard
    error is too narrow and does not say by how much."""

    BAND = pd.DataFrame.from_records([
        {"system": "au", "rule": "fixed", "gap_pct": -13.0,
         "boot_se": 9.0, "excludes_zero": True},
        {"system": "us", "rule": "fixed", "gap_pct": 11.0,
         "boot_se": 4.5, "excludes_zero": True}])
    JACK = pd.DataFrame.from_records([
        {"system": "au", "rule": "fixed", "standard_error": 4.5,
         "ci_excludes_zero": True},
        {"system": "us", "rule": "fixed", "standard_error": 2.25,
         "ci_excludes_zero": True}])

    def test_it_reports_the_ratio_cell_by_cell(self) -> None:
        table = rs.compare_to_jackknife(self.BAND, self.JACK)
        assert list(table["se_ratio"]) == pytest.approx([2.0, 2.0])

    def test_a_sign_the_wider_interval_loses_is_named(self) -> None:
        band = self.BAND.copy()
        band.loc[0, "excludes_zero"] = False
        found = rs.verdict(rs.compare_to_jackknife(band, self.JACK))
        assert not found["every_sign_survives"]
        assert found["signs_lost"] == ["au/fixed"]

    def test_every_sign_surviving_is_reported_as_such(self) -> None:
        found = rs.verdict(rs.compare_to_jackknife(self.BAND, self.JACK))
        assert found["every_sign_survives"]
        assert found["signs_lost"] == []
        assert found["median_se_ratio"] == pytest.approx(2.0)

    def test_the_direction_is_measured_and_not_assumed(self) -> None:
        """Section 10.3's concession says the resampled interval should be
        wider. If it came out narrower the concession would be the wrong
        way round, so the verdict has to be able to say so."""
        band = self.BAND.copy()
        band["boot_se"] = [2.0, 1.0]
        found = rs.verdict(rs.compare_to_jackknife(band, self.JACK))
        assert not found["bootstrap_wider_everywhere"]

    def test_the_headline_cell_is_reported_in_full(self) -> None:
        found = rs.verdict(rs.compare_to_jackknife(self.BAND, self.JACK),
                           headline=("au", "fixed"))
        head = found["headline"]
        assert head["gap_pct"] == pytest.approx(-13.0)
        assert head["se_ratio"] == pytest.approx(2.0)
        assert head["still_signed"]

    def test_an_empty_comparison_is_not_measured(self) -> None:
        assert rs.verdict(pd.DataFrame()) == {"measured": False}


class TestTheConstructionSweep:
    """Two choices the panel's design makes that the paper had been taking
    on the companion study's word."""

    @staticmethod
    def _run(key, setting):
        shift = {"baseline": 0.0, "uniform": 0.4, "gdp": -0.9}[key]
        return _cells(au__fixed=-13.0 + shift, us__fixed=11.0 + shift)

    ARMS = (("baseline", "as configured", None),
            ("uniform", "uniform countries", None),
            ("gdp", "GDP sleeve", None))

    def test_every_arm_is_run_and_labelled(self) -> None:
        frame = rs.construction_sweep(self._run, self.ARMS)
        assert set(frame["arm"]) == {"baseline", "uniform", "gdp"}
        assert set(frame["arm_label"]) == {"as configured",
                                           "uniform countries", "GDP sleeve"}

    def test_a_sign_that_holds_everywhere_reports_as_holding(self) -> None:
        found = rs.construction_verdict(
            rs.construction_sweep(self._run, self.ARMS), "baseline")
        assert found["every_sign_holds"]
        assert found["largest_move_pp"] == pytest.approx(0.9)

    def test_a_construction_choice_that_moves_a_sign_is_caught(self) -> None:
        def _run(key, setting):
            gap = {"baseline": -1.0, "uniform": -1.2, "gdp": +2.0}[key]
            return _cells(au__fixed=gap)

        found = rs.construction_verdict(
            rs.construction_sweep(_run, self.ARMS), "baseline")
        assert not found["every_sign_holds"]
        assert "GDP sleeve" in found["worst"]

    def test_the_baseline_arm_is_not_compared_with_itself(self) -> None:
        found = rs.construction_verdict(
            rs.construction_sweep(self._run, self.ARMS), "baseline")
        assert found["cells"] == 4                # two arms x two cells

    def test_a_missing_baseline_is_not_measured(self) -> None:
        found = rs.construction_verdict(
            rs.construction_sweep(self._run, self.ARMS), "nonexistent")
        assert found == {"measured": False}


class TestWhoseSignAConstructionChoiceMoved:
    """A count cannot tell the two cases apart. A weighting scheme that
    flips only a cell whose interval already straddles zero has agreed
    with the paper; one that flips a cell the paper leads with is a result
    about the panel's construction and belongs in the open."""

    JACK = pd.DataFrame.from_records([
        {"system": "au", "rule": "fixed", "ci_excludes_zero": False},
        {"system": "us", "rule": "fixed", "ci_excludes_zero": True}])

    def test_moving_only_an_unsigned_cell_is_a_check_that_agreed(self
                                                                 ) -> None:
        found = rs.moved_signs_were_already_unsigned(
            {"moved_cells": [("au", "fixed")]}, self.JACK)
        assert found["all_already_unsigned"]
        assert found["signed_cells_moved"] == []
        assert found["unsigned_cells_moved"] == ["au/fixed"]

    def test_moving_a_reported_cell_is_a_finding(self) -> None:
        found = rs.moved_signs_were_already_unsigned(
            {"moved_cells": [("us", "fixed")]}, self.JACK)
        assert not found["all_already_unsigned"]
        assert found["signed_cells_moved"] == ["us/fixed"]

    def test_moving_nothing_needs_no_jackknife(self) -> None:
        found = rs.moved_signs_were_already_unsigned({"moved_cells": []},
                                                     pd.DataFrame())
        assert found["measured"] and found["all_already_unsigned"]

    def test_without_a_jackknife_it_declines_to_judge(self) -> None:
        """Silently calling an unknown cell "already unsigned" would let a
        missing table turn a finding into a reassurance."""
        found = rs.moved_signs_were_already_unsigned(
            {"moved_cells": [("au", "fixed")]}, pd.DataFrame())
        assert found == {"measured": False}

    def test_the_worst_cell_is_reported_in_parts(self) -> None:
        """A page wants the system and rule relabelled, and a caller cannot
        relabel a string that has already been concatenated -- which is how
        two registry keys reached a printed page."""
        def _run(key, setting):
            gap = {"baseline": -1.0, "uniform": -1.1, "gdp": 3.0}[key]
            return _cells(au__fixed=gap)

        found = rs.construction_verdict(
            rs.construction_sweep(_run, TestTheConstructionSweep.ARMS),
            "baseline")
        assert found["worst_system"] == "au"
        assert found["worst_rule"] == "fixed"
        assert found["worst_arm_label"] == "GDP sleeve"
