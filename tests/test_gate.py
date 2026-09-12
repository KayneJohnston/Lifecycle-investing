"""Tests for the pension-gate study.

The bug this section exists to catch is not a crash. It is a decomposition
that reports zero for a feature at the one date where the feature cannot
operate, and reads the zero as a finding. The tests below are mostly about
telling that apart from a genuine zero, because the two look identical in
the output and only the construction distinguishes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import gate as gt


def _square(rows) -> pd.DataFrame:
    """One row per arm and date, the shape :func:`held_still` consumes."""
    return pd.DataFrame([
        {"arm": arm, "retire_age": age, "cec": value}
        for age, per_arm in rows.items()
        for arm, value in per_arm.items()])


#: A gate at 67. Below it the gated arms lose something; at and above it
#: they are identical to their ungated partners, which is the whole point.
GATED = {
    60: {"baseline": 1.00, "timing": 0.90, "formula": 0.70, "both": 0.62},
    63: {"baseline": 1.02, "timing": 0.97, "formula": 0.72, "both": 0.68},
    67: {"baseline": 1.05, "timing": 1.05, "formula": 0.75, "both": 0.75},
    70: {"baseline": 1.04, "timing": 1.04, "formula": 0.74, "both": 0.74},
}


class TestTheSquareAtEveryDate:
    def test_it_reads_the_two_singles_and_the_interaction(self) -> None:
        table = gt.held_still(_square(GATED))
        row = table[table["retire_age"] == 60].iloc[0]
        assert row["timing_effect"] == pytest.approx(-0.10)
        assert row["formula_effect"] == pytest.approx(-0.30)
        assert row["joint_effect"] == pytest.approx(-0.38)
        assert row["interaction"] == pytest.approx(0.02)

    def test_it_flags_the_dates_where_the_arms_cannot_differ(self) -> None:
        """At and past the gate the joint arm *is* the formula arm, so the
        date carries no information about the gate however it comes out."""
        table = gt.held_still(_square(GATED)).set_index("retire_age")
        assert not table.loc[60, "arms_coincide"]
        assert not table.loc[63, "arms_coincide"]
        assert table.loc[67, "arms_coincide"]
        assert table.loc[70, "arms_coincide"]

    def test_a_date_missing_an_arm_is_dropped_not_guessed(self) -> None:
        rows = {60: {"baseline": 1.0, "timing": 0.9, "formula": 0.7},
                63: dict(GATED[63])}
        table = gt.held_still(_square(rows))
        assert list(table["retire_age"]) == [63]

    def test_an_empty_frame_is_an_empty_table(self) -> None:
        assert not len(gt.held_still(pd.DataFrame()))


class TestWhetherTheArgmaxWasBlind:
    """The finding. A decomposition read at the joint arm's own optimum
    reports the gate as worthless when the optimum sits past the gate --
    and would report the same for any gate whatsoever."""

    def test_it_catches_a_comparison_read_past_the_gate(self) -> None:
        table = gt.held_still(_square(GATED))
        found = gt.gate_verdict(table, gate_age=67, joint_best_age=67)
        assert found["argmax_was_blind"]
        assert found["coincide_at_joint_best"]
        assert found["timing_at_joint_best"] == pytest.approx(0.0)

    def test_it_clears_a_comparison_read_where_the_gate_bites(self) -> None:
        table = gt.held_still(_square(GATED))
        found = gt.gate_verdict(table, gate_age=67, joint_best_age=60)
        assert not found["argmax_was_blind"]
        assert found["timing_at_joint_best"] == pytest.approx(-0.10)

    def test_it_counts_the_dates_that_could_say_anything(self) -> None:
        found = gt.gate_verdict(gt.held_still(_square(GATED)), gate_age=67)
        assert found["dates"] == 4
        assert found["informative_dates"] == 2
        assert found["slack_dates"] == 2
        assert found["coincide_past_gate"]

    def test_it_still_reports_which_feature_dominates(self) -> None:
        """Auditing the comparison is not the same as overturning it. Where
        both features can operate the formula is still much the larger, and
        the section has to say so rather than only that the old number was
        badly measured."""
        found = gt.gate_verdict(gt.held_still(_square(GATED)), gate_age=67)
        assert found["formula_dominates_where_it_bites"]
        assert found["timing_where_it_bites"] == pytest.approx(-0.10)
        assert found["formula_over_timing"] == pytest.approx(4.0, rel=1e-6)

    def test_a_gate_below_every_date_leaves_nothing_to_measure(self) -> None:
        found = gt.gate_verdict(gt.held_still(_square(GATED)), gate_age=50)
        assert found["informative_dates"] == 0
        assert "timing_where_it_bites" not in found

    def test_an_empty_table_is_not_measured(self) -> None:
        assert gt.gate_verdict(pd.DataFrame(), 67) == {"measured": False}


def _bridge(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"bridge": share, "arm": arm, "retire_age": age, "cec": value}
        for (share, age), per_arm in rows.items()
        for arm, value in per_arm.items()])


class TestTheBridge:
    """The undisclosed floor. A partial payment before the eligibility age
    is a consumption floor in a paper about consumption floors, and the
    timing arm is made of it."""

    #: At age 60 the gate binds. With no bridge the arm loses a lot; with a
    #: full bridge the gate is off and a small gain survives -- the absent
    #: actuarial penalty. The two therefore cross.
    ROWS = {
        (0.0, 60): {"baseline": 1.00, "timing": 0.70},
        (0.5, 60): {"baseline": 1.00, "timing": 0.95},
        (1.0, 60): {"baseline": 1.00, "timing": 1.05},
    }

    def test_it_measures_the_effect_at_each_share(self) -> None:
        table = gt.bridge_table(_bridge(self.ROWS))
        assert list(table["timing_effect"].round(6)) == [-0.30, -0.05, 0.05]

    def test_the_ends_separate_the_bridge_from_the_penalty(self) -> None:
        """A pension paying the full rate on both sides of its eligibility
        age is not gated, so what survives there is the other half of the
        feature and the difference is the bridge."""
        found = gt.bridge_verdict(gt.bridge_table(_bridge(self.ROWS)),
                                  gate_age=67, reference_age=60)
        assert found["isolates_penalty"]
        assert found["penalty_only"] == pytest.approx(0.05)
        assert found["bridge_only"] == pytest.approx(-0.35)
        assert found["dominant"] == "bridge"

    def test_it_finds_the_share_at_which_the_two_cancel(self) -> None:
        """Which is the point: a calibration sitting near that share reports
        'the start date does not matter' when neither half is small."""
        found = gt.bridge_verdict(gt.bridge_table(_bridge(self.ROWS)),
                                  gate_age=67, reference_age=60)
        assert found["ends_disagree"]
        assert found["cancels_at_share"] == pytest.approx(0.75)

    def test_ends_that_agree_have_no_crossing(self) -> None:
        rows = {(0.0, 60): {"baseline": 1.0, "timing": 0.7},
                (1.0, 60): {"baseline": 1.0, "timing": 0.9}}
        found = gt.bridge_verdict(gt.bridge_table(_bridge(rows)),
                                  gate_age=67, reference_age=60)
        assert not found["ends_disagree"]
        assert "cancels_at_share" not in found

    def test_a_grid_that_stops_short_of_a_full_bridge_says_so(self) -> None:
        """Without the full-bridge end there is nothing to separate the
        penalty from the gate, and the verdict must not pretend otherwise."""
        rows = {(0.0, 60): {"baseline": 1.0, "timing": 0.7},
                (0.6, 60): {"baseline": 1.0, "timing": 0.98}}
        found = gt.bridge_verdict(gt.bridge_table(_bridge(rows)),
                                  gate_age=67, reference_age=60)
        assert found["measured"] and not found["isolates_penalty"]
        assert "penalty_only" not in found

    def test_it_reports_whether_the_movement_is_monotone(self) -> None:
        assert gt.bridge_verdict(gt.bridge_table(_bridge(self.ROWS)),
                                 gate_age=67, reference_age=60)["monotone"]
        rows = {(0.0, 60): {"baseline": 1.0, "timing": 0.7},
                (0.5, 60): {"baseline": 1.0, "timing": 1.2},
                (1.0, 60): {"baseline": 1.0, "timing": 0.9}}
        assert not gt.bridge_verdict(gt.bridge_table(_bridge(rows)),
                                     gate_age=67,
                                     reference_age=60)["monotone"]

    def test_dates_past_the_gate_are_not_read(self) -> None:
        """The bridge cannot touch a household that retires after the
        eligibility age, so including one would dilute the measurement with
        rows that are zero by construction."""
        rows = dict(self.ROWS)
        rows[(0.0, 70)] = {"baseline": 1.0, "timing": 1.0}
        rows[(1.0, 70)] = {"baseline": 1.0, "timing": 1.0}
        found = gt.bridge_verdict(gt.bridge_table(_bridge(rows)),
                                  gate_age=67)
        assert found["at_lowest_share"] == pytest.approx(-0.30)

    def test_an_empty_table_is_not_measured(self) -> None:
        assert gt.bridge_verdict(pd.DataFrame(), 67) == {"measured": False}
        assert not len(gt.bridge_table(pd.DataFrame()))


class TestItAuditsTheSectionItSaysItAudits:
    def test_the_arms_match_the_leisure_study(self) -> None:
        """This section re-scores that section's own 2x2. If the arms drift
        apart the two are no longer comparable and every sentence setting a
        row here beside a row there is wrong."""
        from src import leisure as le

        assert gt.ARMS == le.FEATURE_ARMS

    def test_the_real_square_was_read_past_the_gate(self) -> None:
        """The finding itself, on the live tables."""
        import pathlib
        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / "results/tables/gate_held_still.csv"
        if not path.exists():
            pytest.skip("the gate study has not been run")
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        gate_age = int(cfg.get("leisure", {}).get("age_pension_age", 67))
        table = pd.read_csv(path)
        best = int(table.loc[table["cec_both"].idxmax(), "retire_age"])
        found = gt.gate_verdict(table, gate_age, best)
        assert found["measured"]
        assert found["argmax_was_blind"], (
            "the leisure study's decomposition now lands where the gate "
            "binds; this section's premise has changed")

    def test_every_date_past_the_gate_is_an_identity(self) -> None:
        """Not approximately. Past the eligibility age the joint arm and
        the formula arm run the same simulation, so a difference of any
        size would mean the gate is doing something it cannot do.

        The gate arm against the baseline is *not* an identity there, and
        the reason is worth pinning down rather than tolerating: that arm
        also drops the actuarial adjustment, so past the reference age it
        forgoes a late-claiming bonus the baseline collects. The residual
        is that bonus and nothing to do with the gate -- which is exactly
        the decomposition the bridge sweep exploits at the other end.
        """
        import pathlib
        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / "results/tables/gate_held_still.csv"
        if not path.exists():
            pytest.skip("the gate study has not been run")
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        gate_age = int(cfg.get("leisure", {}).get("age_pension_age", 67))
        table = pd.read_csv(path)
        past = table[table["retire_age"] >= gate_age]
        assert len(past), "no date on the grid reaches the eligibility age"
        # The identity the section's claim rests on.
        assert np.allclose(past["cec_both"], past["cec_formula"], atol=1e-12)
        assert bool(past["arms_coincide"].all())
        # And the residual that is not it, bounded so a real gate effect
        # leaking in past the gate would still be caught.
        assert np.allclose(past["timing_effect"], 0.0, atol=5e-3), (
            "the gate arm differs from the baseline past the eligibility "
            "age by more than the claiming adjustment can explain")
