"""Tests for the levered-ceiling study.

`src.incidence` reports that a means-tested retiree spending a share of the
balance wants 100% equity at every position against the assets test. That is
the top of the grid it was chosen from, so it is consistent both with a
household that wants exactly the whole portfolio in equity and with one that
wants more and could not say so. These tests drive `src.ceiling` with
arithmetic scores rather than a bootstrap, so each claim is checked against a
case whose answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ceiling as cl


def _sweep(best, scales=(0.1, 0.5, 1.0), leverages=(1.0, 1.25, 1.5, 1.75),
           spreads=(0.0,)):
    """A sweep whose optimum sits at ``best(scale, spread)`` by construction.

    The score is a downward parabola in leverage centred on the wanted
    value, so the argmax is exactly that value when it is on the grid and
    the nearest grid point when it is not.
    """
    def score(scale, leverage, spread):
        want = best(scale, spread)
        return {"cec": 1.0 - (leverage - want) ** 2}
    return cl.sweep(score, scales, leverages, spreads, log_every=0)


class TestTheSweep:
    def test_it_covers_the_whole_grid(self) -> None:
        frame = _sweep(lambda s, c: 1.0, spreads=(0.0, 0.01))
        assert len(frame) == 3 * 4 * 2
        assert set(frame.columns) >= {"scale", "leverage", "spread", "cec"}

    def test_an_empty_grid_is_an_empty_frame(self) -> None:
        assert not len(cl.sweep(lambda *a: {"cec": 1.0}, (), (1.0,), (0.0,)))


class TestTheOptimum:
    def test_it_finds_the_wanted_leverage_at_each_balance(self) -> None:
        wanted = {0.1: 1.0, 0.5: 1.25, 1.0: 1.75}
        best = cl.optimum(_sweep(lambda s, c: wanted[s]))
        assert dict(zip(best["scale"], best["leverage"])) == wanted

    def test_it_flags_an_optimum_at_the_top_of_the_grid(self) -> None:
        best = cl.optimum(_sweep(lambda s, c: 1.75))
        assert best["at_ceiling"].all()

    def test_the_unlevered_optimum_is_not_levered(self) -> None:
        best = cl.optimum(_sweep(lambda s, c: 1.0))
        assert not best["levered"].any()
        assert not best["at_ceiling"].any()


class TestWhetherTheCornerWasCensored:
    def test_a_household_that_wants_to_borrow_censors_the_corner(self) -> None:
        frame = _sweep(lambda s, c: 1.5)
        found = cl.verdict(cl.optimum(frame), frame)
        assert found["censored"]
        assert found["max_leverage"] == pytest.approx(1.5)

    def test_a_household_that_does_not_confirms_it(self) -> None:
        frame = _sweep(lambda s, c: 1.0)
        found = cl.verdict(cl.optimum(frame), frame)
        assert not found["censored"]

    def test_the_verdict_is_read_at_the_cheapest_spread(self) -> None:
        """Borrowing is most attractive where it is cheapest, so that is
        where the corner is hardest to defend and where the question is
        settled."""
        frame = _sweep(lambda s, c: 1.0 if c > 0 else 1.5,
                       spreads=(0.0, 0.05))
        found = cl.verdict(cl.optimum(frame), frame)
        assert found["cheapest_spread"] == 0.0
        assert found["censored"]

    def test_it_reports_what_the_borrowing_is_worth(self) -> None:
        frame = _sweep(lambda s, c: 1.5)
        found = cl.verdict(cl.optimum(frame), frame)
        # 1 - 0 against 1 - 0.25 at the unlevered point.
        assert found["median_gain_pct"] == pytest.approx(100.0 / 3.0, rel=1e-6)

    def test_nothing_measured_is_said_so(self) -> None:
        assert cl.verdict(pd.DataFrame(), pd.DataFrame()) == {
            "measured": False}


class TestTheBreakEvenSpread:
    def test_it_is_the_cheapest_spread_nobody_borrows_at(self) -> None:
        frame = _sweep(lambda s, c: 1.0 if c >= 0.03 else 1.5,
                       spreads=(0.0, 0.01, 0.03, 0.05))
        assert cl.break_even_spread(cl.optimum(frame)) == pytest.approx(0.03)

    def test_a_sweep_that_never_reaches_it_says_so(self) -> None:
        frame = _sweep(lambda s, c: 1.5, spreads=(0.0, 0.01))
        assert np.isnan(cl.break_even_spread(cl.optimum(frame)))

    def test_it_is_zero_when_nobody_borrows_at_any_price(self) -> None:
        frame = _sweep(lambda s, c: 1.0, spreads=(0.0, 0.01))
        assert cl.break_even_spread(cl.optimum(frame)) == pytest.approx(0.0)


class TestTheShapeOnceThereIsRoomForIt:
    POSITIONS = {0.1: "below the free area", 0.5: "inside the taper band",
                 1.0: "above the cut-off"}

    def test_it_groups_the_optima_by_where_the_balance_lands(self) -> None:
        frame = _sweep(lambda s, c: {0.1: 1.0, 0.5: 1.75, 1.0: 1.25}[s])
        bands = cl.by_band(cl.optimum(frame), self.POSITIONS)
        assert set(bands["position"]) == set(self.POSITIONS.values())

    def test_the_predicted_ordering_is_recognised(self) -> None:
        """Highest inside the band, lower below the free area, lowest past
        the cut-off -- the prediction Section 2 of the paper makes."""
        frame = _sweep(lambda s, c: {0.1: 1.5, 0.5: 1.75, 1.0: 1.0}[s])
        found = cl.shape_verdict(cl.by_band(cl.optimum(frame),
                                            self.POSITIONS))
        assert found["prediction_holds"]
        assert found["band_beats_below"] and found["band_beats_above"]

    def test_a_flat_answer_is_not_a_shape(self) -> None:
        frame = _sweep(lambda s, c: 1.5)
        found = cl.shape_verdict(cl.by_band(cl.optimum(frame),
                                            self.POSITIONS))
        assert not found["differentiated"]
        assert not found["prediction_holds"]

    def test_the_wrong_ordering_is_not_the_prediction(self) -> None:
        frame = _sweep(lambda s, c: {0.1: 1.0, 0.5: 1.25, 1.0: 1.75}[s])
        found = cl.shape_verdict(cl.by_band(cl.optimum(frame),
                                            self.POSITIONS))
        assert found["differentiated"]
        assert not found["prediction_holds"]

    def test_a_balance_with_no_position_is_dropped(self) -> None:
        frame = _sweep(lambda s, c: 1.5)
        bands = cl.by_band(cl.optimum(frame), {0.1: "below the free area"})
        assert list(bands["position"]) == ["below the free area"]

    def test_nothing_measured_is_said_so(self) -> None:
        assert cl.shape_verdict(pd.DataFrame()) == {"measured": False}
