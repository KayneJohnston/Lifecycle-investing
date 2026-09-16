"""The statutory rule, and the prediction put to it.

The value of this section is entirely in the rates being someone else's.
A schedule mistyped into the repository, or quietly tuned until the
prediction passed, would turn an out-of-sample test into a calibration
exercise that happens to agree with itself -- so most of what is tested
here is that the schedule in the code is the schedule in the statute, and
that the classifier would report a failure if one came.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anchor as anc  # noqa: E402
from src import lifecycle as lc  # noqa: E402
from src import spending as spg  # noqa: E402

#: Schedule 7 of the Superannuation Industry (Supervision) Regulations
#: 1994, written out here independently of the module under test so that a
#: single edit cannot move both.
STATUTORY = {60: 0.04, 64: 0.04, 65: 0.05, 70: 0.05, 74: 0.05, 75: 0.06,
             79: 0.06, 80: 0.07, 84: 0.07, 85: 0.09, 89: 0.09, 90: 0.11,
             94: 0.11, 95: 0.14, 100: 0.14}


class TestTheScheduleIsTheStatute:
    """The rates are the whole of the claim. If they were ours, the section
    would be a calibration agreeing with itself."""

    @pytest.mark.parametrize("age,rate", sorted(STATUTORY.items()))
    def test_each_age_draws_the_legislated_share(self, age, rate) -> None:
        got = spg.legislated_minimum_rate(np.array([age]))[0]
        assert got == pytest.approx(rate), age

    def test_it_never_falls_with_age(self) -> None:
        ages = np.arange(55, 106)
        rates = spg.legislated_minimum_rate(ages)
        assert np.all(np.diff(rates) >= -1e-12)

    def test_the_config_carries_the_same_schedule_as_the_code(self) -> None:
        """The paper prints the config's copy and simulates the module's.
        Two copies of a statute is one too many unless they are checked."""
        import yaml

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        block = cfg.get("anchor", {})
        if not block.get("minimum_schedule"):
            pytest.skip("the anchor study is not configured")
        listed = {int(a): float(r) for a, r in block["minimum_schedule"]}
        assert listed == {a: r for a, r in spg.LEGISLATED_MINIMUM}

    def test_the_source_is_named_where_the_paper_quotes_it(self) -> None:
        assert "Superannuation Industry" in anc.STATUTE
        assert "Schedule 7" in anc.STATUTE


class TestTheRuleBehavesLikeAPercentageOfBalance:
    """It is the family the mechanism predicts an asset-tested system
    needs, so it has to actually be in that family."""

    def test_it_draws_a_share_of_the_current_balance(self) -> None:
        rule = spg.build("legislated_minimum")
        state = spg.SpendingState(
            year=0, age=70, years_remaining=20,
            wealth=np.array([100.0, 200.0]),
            prev_withdrawal=np.zeros(2), initial_withdrawal=np.zeros(2),
            wealth_at_retirement=np.array([100.0, 200.0]),
            last_return=np.zeros(2), last_inflation=np.zeros(2))
        assert np.allclose(rule.desired(state), [5.0, 10.0])

    def test_it_cannot_deplete_the_portfolio(self) -> None:
        from src import plan as pl

        assert pl.CAN_DEPLETE["legislated_minimum"] is False

    def test_a_multiple_scales_the_draw(self) -> None:
        state = spg.SpendingState(
            year=0, age=70, years_remaining=20, wealth=np.array([100.0]),
            prev_withdrawal=np.zeros(1), initial_withdrawal=np.zeros(1),
            wealth_at_retirement=np.array([100.0]),
            last_return=np.zeros(1), last_inflation=np.zeros(1))
        one = spg.build("legislated_minimum").desired(state)
        two = spg.build("legislated_minimum", multiple=2.0).desired(state)
        assert np.allclose(two, 2.0 * one)

    def test_the_draw_rises_as_the_retiree_ages(self) -> None:
        rule = spg.build("legislated_minimum")

        def at(age: int) -> float:
            state = spg.SpendingState(
                year=0, age=age, years_remaining=5,
                wealth=np.array([100.0]), prev_withdrawal=np.zeros(1),
                initial_withdrawal=np.zeros(1),
                wealth_at_retirement=np.array([100.0]),
                last_return=np.zeros(1), last_inflation=np.zeros(1))
            return float(rule.desired(state)[0])

        assert at(70) < at(80) < at(90) < at(96)


class TestTheVerdictWouldReportAFailure:
    """A prediction that cannot fail is not one, so the classifier is
    driven both ways with synthetic grids."""

    @staticmethod
    def _grid(law: float, lit: float) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"system": "age_pension_matched", "rule": "legislated minimum",
             "gap_pct": law, "winner": "x"},
            {"system": "age_pension_matched", "rule": "fixed_real_rule",
             "gap_pct": lit, "winner": "y"}])

    def test_the_predicted_pattern_is_reported_as_holding(self) -> None:
        got = anc.verdict(self._grid(+12.2, -11.4), "age_pension_matched",
                          "legislated minimum", "fixed_real_rule")
        assert got["measured"]
        assert got["prediction_holds"]
        assert got["swing_pp"] == pytest.approx(23.6)

    def test_a_legislated_rule_that_also_loses_is_a_failure(self) -> None:
        got = anc.verdict(self._grid(-4.0, -11.4), "age_pension_matched",
                          "legislated minimum", "fixed_real_rule")
        assert not got["prediction_holds"]
        assert not got["legislated_rule_keeps_the_lead"]

    def test_a_literature_rule_that_also_wins_is_a_failure(self) -> None:
        """If the fixed real rule kept the lead there would be no reversal
        to explain, and the anchor would be confirming nothing."""
        got = anc.verdict(self._grid(+12.2, +3.0), "age_pension_matched",
                          "legislated minimum", "fixed_real_rule")
        assert not got["prediction_holds"]
        assert not got["assumed_rule_loses_it"]

    def test_a_missing_rule_is_not_measured(self) -> None:
        frame = self._grid(+12.2, -11.4)
        got = anc.verdict(frame[frame["rule"] != "fixed_real_rule"],
                          "age_pension_matched", "legislated minimum",
                          "fixed_real_rule")
        assert not got.get("measured")


class TestTheBehaviouralClaimIsWrittenDown:
    """A design fact is not a holdings fact. The paper says so; this is
    the guard that the falsifiable version is carried rather than
    implied."""

    def test_it_names_what_would_settle_it(self) -> None:
        frame = pd.DataFrame.from_records([
            {"system": "age_pension_matched", "rule": "legislated minimum",
             "gap_pct": 12.2, "winner": "balanced_all_equity"}])
        got = anc.observable_prediction(frame, "age_pension_matched",
                                        "legislated minimum")
        assert got["measured"]
        assert "high-equity" in got["claim"]
        assert "allocation" in got["falsified_by"]


class TestTheShippedRun:
    """What the statutory rule actually did, so the paper's prose cannot
    outlive a rerun that moved it."""

    @staticmethod
    def _gaps() -> pd.DataFrame:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / "anchor_gaps.csv"),
                         recursive=True)
        if not hits:
            pytest.skip("the anchor study has not been run")
        return pd.read_csv(hits[0])

    def test_the_prediction_holds_in_the_shipped_run(self) -> None:
        got = anc.verdict(self._gaps(), "age_pension_matched",
                          "legislated minimum", "fixed_real_rule")
        assert got["measured"]
        assert got["prediction_holds"], got

    def test_it_holds_at_every_multiple_of_the_minimum(self) -> None:
        """The statute sets a floor, not a rate. If the result needed the
        floor exactly it would be a coincidence rather than a property of
        the rule's shape."""
        gaps = self._gaps()
        block = gaps[(gaps["system"] == "age_pension_matched")
                     & (gaps["rule"].str.startswith("legislated minimum"))]
        assert len(block) >= 2
        assert (block["gap_pct"] > 0.0).all(), block

    def test_the_comparator_is_the_cell_the_paper_leads_with(self) -> None:
        gaps = self._gaps()
        hit = gaps[(gaps["system"] == "age_pension_matched")
                   & (gaps["rule"] == "fixed_real_rule")]
        assert len(hit) == 1
        assert float(hit["gap_pct"].iloc[0]) < 0.0
