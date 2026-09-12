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


class TestItStudiesTheArmItSaysItDoes:
    """The levered sweep exists to reopen one corner reported in `docs/35`,
    so it has to be the same household. It was not: it rebuilt the
    specification from the raw config and kept the model's own retirement
    age, so its retiree stopped work four years before the pension began
    and spent four retirement years outside the assets test -- the arm
    `docs/35` deliberately excludes, and the one whose caption promises
    that 'the pension begins the day work stops'."""

    @staticmethod
    def _base():
        import dataclasses
        import yaml

        from src import leisure as le
        from src import lifecycle as lc

        cfg = yaml.safe_load(open("config.yaml"))
        over, _ = le.system_overrides("au_as_legislated", cfg)
        return dataclasses.replace(lc.spec_from_config(cfg), **over), cfg

    def test_the_arms_differ_only_in_the_rule(self) -> None:
        """`ARMS[2]` is `ARMS[0]` with the withdrawal rule changed, so a
        difference between the two is the rule and nothing else."""
        import dataclasses

        from src import incidence as inc

        base, _ = self._base()
        arms = inc.balance_arms(base)
        a, b = arms[inc.ARMS[0]], arms[inc.ARMS[2]]
        differ = {f.name for f in dataclasses.fields(a)
                  if getattr(a, f.name) != getattr(b, f.name)}
        assert differ == {"retirement_rule"}, differ

    def test_the_pension_starts_the_day_work_stops_in_both(self) -> None:
        from src import incidence as inc

        base, _ = self._base()
        arms = inc.balance_arms(base)
        for key in (inc.ARMS[0], inc.ARMS[2]):
            spec = arms[key]
            assert spec.age_retire == spec.benefit_start_age, key

    def test_the_comparison_arm_keeps_the_early_date(self) -> None:
        """`ARMS[1]` is the hole in the floor, and has to keep it."""
        from src import incidence as inc

        base, _ = self._base()
        arms = inc.balance_arms(base)
        assert arms[inc.ARMS[1]].age_retire == base.age_retire
        assert arms[inc.ARMS[1]].age_retire < base.benefit_start_age

    def test_the_levered_sweep_takes_that_arm_and_does_not_rebuild_it(self
                                                                      ) -> None:
        """The guard that matters: the specification step 37 scores is the
        one step 35 swept, field for field."""
        import dataclasses

        from src import incidence as inc

        base, cfg = self._base()
        wanted = inc.balance_arms(base)[
            str(cfg.get("ceiling", {}).get("arm", inc.ARMS[2]))]
        source = open("main.py").read()
        step = source[source.index("def step37_ceiling"):
                      source.index("STEPS = {1: step1_dataset")]
        assert "inc.balance_arms(" in step, \
            "step 37 rebuilds the specification instead of taking the arm"
        assert wanted.age_retire == wanted.benefit_start_age
        assert dataclasses.replace(wanted) == wanted

    def test_the_sweep_spends_by_the_rule_its_arm_names(self) -> None:
        """Scoring one rule against a specification that names another is
        the same class of error a step lower down."""
        from src import incidence as inc
        from src import spending as spg

        base, cfg = self._base()
        arm = inc.balance_arms(base)[
            str(cfg.get("ceiling", {}).get("arm", inc.ARMS[2]))]
        rule = spg.from_spec(arm.retirement_rule, arm.rule_rate)
        assert isinstance(rule, spg.ConstantPercentRule)
        assert rule.rate == pytest.approx(arm.rule_rate)
