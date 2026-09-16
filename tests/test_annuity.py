"""The floor a market sells, and what it does to the paper's finding.

The paper's mechanism is that a means test reverses the portfolio ordering
by removing an unconditional floor. That invites one objection above all
others -- annuities exist -- and until this module the paper conceded it in
the limitations instead of answering it.

What is tested here is mostly arithmetic, because the arithmetic is what
makes the answer trustworthy either way. A life annuity that is priced too
cheaply would rescue the all-equity portfolio for a reason that has nothing
to do with pensions, and a fixed thirty-year horizon prices one too cheaply
by construction: it pays for thirty years an income the insurer costed over
twenty. That trap is the subject of half the tests below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import annuity as anu  # noqa: E402
from src import lifecycle as lc  # noqa: E402
from src import mortality as mrt  # noqa: E402


def _survival(spec: lc.LifecycleSpec | None = None) -> np.ndarray:
    spec = spec or lc.LifecycleSpec()
    full = mrt.survival(spec, 88.0, 10.0)
    retired = full[-spec.n_retired:]
    return retired / retired[0]


class TestThePrice:
    """A premium is an expected present value and a load. Both halves have
    a closed form to check against, so neither is taken on trust."""

    def test_a_certain_stream_at_a_zero_rate_is_its_own_length(self) -> None:
        assert anu.price(np.ones(20), 0.0, 1.0) == pytest.approx(20.0)

    def test_discounting_a_certain_stream_is_the_annuity_formula(self) -> None:
        n, rate = 25, 0.03
        want = (1.0 - (1.0 + rate) ** -n) / rate * (1.0 + rate)
        assert anu.price(np.ones(n), rate, 1.0) == pytest.approx(want)

    def test_the_first_payment_is_certain_and_undiscounted(self) -> None:
        """Bought on the retirement date and paid at the start of each year,
        so the annuitant cannot die before collecting once."""
        assert anu.price(np.array([1.0]), 0.5, 1.0) == pytest.approx(1.0)

    def test_a_load_raises_the_premium(self) -> None:
        fair = anu.price(_survival(), 0.02, 1.0)
        loaded = anu.price(_survival(), 0.02, 0.8)
        assert loaded == pytest.approx(fair / 0.8)
        assert loaded > fair

    def test_mortality_makes_a_life_annuity_cheaper_than_a_term_certain(
            self) -> None:
        """The mortality credit, which is the whole reason the instrument
        exists and the whole reason a fixed horizon must not price it."""
        spec = lc.LifecycleSpec()
        life = anu.price(_survival(spec), 0.02, 1.0)
        certain = anu.price(np.ones(spec.n_retired), 0.02, 1.0)
        assert life < certain

    def test_an_impossible_money_worth_is_refused(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                anu.price(_survival(), 0.02, bad)

    def test_an_empty_curve_is_refused_rather_than_priced(self) -> None:
        with pytest.raises(ValueError):
            anu.price(np.array([]), 0.02, 1.0)


class TestTheAssessedValue:
    """What an assets test would put on an income stream it can see."""

    def test_it_starts_at_the_actuarial_value_of_every_payment(self) -> None:
        alive = _survival()
        path = anu.assessable_path(alive, 0.02, income=1.0)
        assert path[0] == pytest.approx(anu.price(alive, 0.02, 1.0))

    def test_it_ends_at_one_year_of_income(self) -> None:
        path = anu.assessable_path(_survival(), 0.02, income=3.0)
        assert path[-1] == pytest.approx(3.0)

    def test_it_never_rises(self) -> None:
        """A declining purchase price is the shape a real schedule uses,
        and a value that rose would let a retiree be pushed off the pension
        by an asset they are consuming."""
        path = anu.assessable_path(_survival(), 0.02, income=1.0)
        assert np.all(np.diff(path) <= 1e-9)

    def test_it_scales_with_the_income(self) -> None:
        one = anu.assessable_path(_survival(), 0.02, income=1.0)
        two = anu.assessable_path(_survival(), 0.02, income=2.0)
        assert np.allclose(two, 2.0 * one)


class TestTheLayer:
    """What the simulator is handed, per unit of the balance at retirement."""

    def test_buying_nothing_costs_nothing_and_pays_nothing(self) -> None:
        spec = lc.LifecycleSpec()
        got = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                        0.02, 0.9, 0.0)
        assert got["premium_share"] == 0.0
        assert got["income_per_wealth"] == 0.0

    def test_the_premium_share_buys_that_share_of_the_balance(self) -> None:
        spec = lc.LifecycleSpec()
        got = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                        0.02, 1.0, 0.5)
        assert got["premium_share"] == pytest.approx(0.5)
        assert got["income_per_wealth"] * got["unit_price"] == \
            pytest.approx(0.5)

    def test_a_load_buys_less_income_for_the_same_share(self) -> None:
        spec = lc.LifecycleSpec()
        fair = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                         0.02, 1.0, 0.5)
        loaded = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                           0.02, 0.8, 0.5)
        assert loaded["income_per_wealth"] < fair["income_per_wealth"]

    def test_the_curve_is_conditioned_on_reaching_the_pension_age(
            self) -> None:
        """A retiree buying at 63 is quoted a price for someone alive at 63,
        not for a 25-year-old's chance of getting there."""
        spec = lc.LifecycleSpec()
        got = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                        0.0, 1.0, 1.0)
        # At a zero discount rate the premium is expected years of payment.
        assert got["unit_price"] == pytest.approx(
            got["life_expectancy_years"])
        assert 10.0 < got["life_expectancy_years"] < spec.n_retired

    def test_a_share_outside_the_unit_interval_is_clipped(self) -> None:
        spec = lc.LifecycleSpec()
        for asked, want in ((-0.5, 0.0), (1.7, 1.0)):
            got = anu.layer(mrt.survival(spec, 88.0, 10.0), spec.n_retired,
                            0.02, 1.0, asked)
            assert got["premium_share"] == pytest.approx(want)


class TestTheSimulatorCarriesIt:
    """The spec fields, and the promise that they change nothing at zero."""

    def test_the_default_spec_buys_no_annuity(self) -> None:
        spec = lc.LifecycleSpec()
        assert spec.annuity_fraction == 0.0
        assert spec.annuity_moneys_worth == 1.0

    def test_an_impossible_purchase_is_refused_at_construction(self) -> None:
        import dataclasses

        spec = lc.LifecycleSpec()
        for field, bad in (("annuity_fraction", 1.5),
                           ("annuity_fraction", -0.1),
                           ("annuity_moneys_worth", 0.0),
                           ("annuity_moneys_worth", 1.4),
                           ("annuity_assessed", 2.0)):
            with pytest.raises(ValueError):
                dataclasses.replace(spec, **{field: bad})

    def test_the_config_reader_defaults_it_off(self) -> None:
        """Every config that predates this study omits the block, and those
        runs have to stay bit-identical."""
        import yaml

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        cfg["lifecycle"].pop("annuity", None)
        spec = lc.spec_from_config(cfg)
        assert spec.annuity_fraction == 0.0


class TestTheVerdictReportsWhicheverWayItComesOut:
    """The classifier has to be able to say the finding failed, or it is
    not a classifier. Both directions are driven with synthetic grids."""

    @staticmethod
    def _grid(gaps_by_fraction: dict) -> pd.DataFrame:
        rows = []
        for fraction, gap in gaps_by_fraction.items():
            rows.append({"system": "age_pension_matched",
                         "rule": "fixed_real_rule",
                         "treatment": "exempt from the assets test",
                         "assessed": 0.0, "fraction": float(fraction),
                         "gap_pct": float(gap)})
        return pd.DataFrame.from_records(rows)

    def test_a_sign_that_holds_at_every_share_is_reported_as_holding(
            self) -> None:
        got = anu.reversal(self._grid({0.0: -13.0, 0.5: -9.0, 1.0: -4.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert got["measured"]
        assert got["reverses_without_annuity"]
        assert got["sign_survives_every_share"]
        assert got["sign_survives_every_interior_share"]
        assert np.isnan(got["first_share_that_flips_it"])

    def test_a_sign_the_annuity_overturns_is_reported_as_overturned(
            self) -> None:
        got = anu.reversal(self._grid({0.0: -13.0, 0.5: +2.0, 1.0: +8.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert not got["sign_survives_every_share"]
        assert not got["sign_survives_every_interior_share"]
        assert got["first_interior_share_that_flips_it"] == pytest.approx(0.5)
        assert not got["flip_needs_the_corner"]

    def test_a_flip_that_needs_the_empty_portfolio_is_labelled_as_one(
            self) -> None:
        """At a full share both arms hold nothing, so the gap there is a
        comparison of two accumulation histories. Reading it as the
        reversal being cured is the mistake this flag exists to stop."""
        got = anu.reversal(self._grid({0.0: -13.0, 0.5: -14.0, 0.75: -6.0,
                                       1.0: +10.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert not got["sign_survives_every_share"]
        assert got["sign_survives_every_interior_share"]
        assert got["flip_needs_the_corner"]
        assert got["corner_gap_pct"] == pytest.approx(10.0)

    def test_a_purchase_that_widens_the_gap_is_reported_as_widening(
            self) -> None:
        got = anu.reversal(self._grid({0.0: -11.4, 0.5: -14.0, 1.0: +10.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert got["annuity_deepens_it"]
        assert got["deepest_at_share"] == pytest.approx(0.5)
        assert got["deepest_interior_gap_pct"] == pytest.approx(-14.0)

    def test_a_purchase_that_only_helps_is_not_reported_as_widening(
            self) -> None:
        got = anu.reversal(self._grid({0.0: -13.0, 0.5: -9.0, 1.0: -4.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert not got["annuity_deepens_it"]

    def test_a_cell_with_no_control_is_not_measured(self) -> None:
        """Every claim here is a difference against annuitising nothing, so
        a grid without that row has nothing to say."""
        got = anu.reversal(self._grid({0.5: -9.0, 1.0: -4.0}),
                           "age_pension_matched", "fixed_real_rule",
                           "exempt from the assets test")
        assert not got["measured"]

    def test_the_wanted_share_is_the_maximiser_and_knows_the_corner(
            self) -> None:
        frame = pd.DataFrame.from_records([
            {"system": "s", "rule": "r", "treatment": "t", "assessed": 0.0,
             "strategy": "balanced_all_equity", "fraction": f,
             "cec_survival": v}
            for f, v in ((0.0, 1.0), (0.5, 1.2), (1.0, 1.1))])
        got = anu.wanted(frame, "balanced_all_equity")
        assert len(got) == 1
        row = got.iloc[0]
        assert row["best_fraction"] == pytest.approx(0.5)
        assert row["gain_pct"] == pytest.approx(20.0)
        assert not bool(row["at_the_corner"])

    def test_a_maximum_at_the_top_of_the_grid_is_flagged(self) -> None:
        """A corner is not an optimum, and a section that reported one as
        an optimum would be reporting the grid rather than the household."""
        frame = pd.DataFrame.from_records([
            {"system": "s", "rule": "r", "treatment": "t", "assessed": 0.0,
             "strategy": "balanced_all_equity", "fraction": f,
             "cec_survival": v}
            for f, v in ((0.0, 1.0), (0.5, 1.2), (1.0, 1.4))])
        assert bool(anu.wanted(frame, "balanced_all_equity")
                    .iloc[0]["at_the_corner"])


class TestTheHorizonTrap:
    """The reason every headline in this section is survival-weighted."""

    @staticmethod
    def _frame(fixed_full: float, lived_full: float) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"fraction": 0.0, "cec": 1.0, "cec_survival": 1.0},
            {"fraction": 1.0, "cec": fixed_full,
             "cec_survival": lived_full}])

    def test_a_fixed_horizon_that_flatters_the_annuity_is_flagged(
            self) -> None:
        got = anu.horizon_distortion(self._frame(1.40, 1.10))
        assert got["measured"]
        assert got["fixed_horizon_flatters_it"]
        assert got["overpaid_pp"] == pytest.approx(30.0)

    def test_a_fixed_horizon_that_does_not_is_not_flagged(self) -> None:
        got = anu.horizon_distortion(self._frame(1.05, 1.20))
        assert not got["fixed_horizon_flatters_it"]
        assert got["overpaid_pp"] < 0.0

    def test_a_frame_without_both_objectives_is_not_measured(self) -> None:
        frame = pd.DataFrame.from_records([{"fraction": 0.0, "cec": 1.0}])
        assert not anu.horizon_distortion(frame).get("measured")


class TestTheShippedRun:
    """What the run in the repository actually found, so a rerun that moved
    it cannot leave the paper's prose behind."""

    @staticmethod
    def _table(name: str) -> pd.DataFrame:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / f"{name}.csv"),
                         recursive=True)
        if not hits:
            pytest.skip(f"{name} has not been generated")
        return pd.read_csv(hits[0])

    def test_the_grid_carries_its_own_control(self) -> None:
        """Annuitising nothing has to be in the sweep, or the comparison is
        against a number quoted from another section at another path
        count."""
        frame = self._table("annuity_sweep")
        assert float(frame["fraction"].min()) == pytest.approx(0.0)

    def test_both_readings_of_the_assets_test_are_run(self) -> None:
        frame = self._table("annuity_sweep")
        assert len(set(frame["treatment"])) == len(anu.TREATMENTS)

    def test_the_control_reproduces_the_ordering_grid_sign(self) -> None:
        """At a zero share this is the paper's own cell, at a different
        path count. The point estimates need not agree to the basis point;
        the sign has to."""
        gaps = self._table("annuity_gaps")
        ordering = self._table("ordering_gaps")
        rule = "fixed_real_rule"
        for system in ("age_pension_matched", "us_social_security"):
            here = gaps[(gaps["system"] == system) & (gaps["rule"] == rule)
                        & (gaps["fraction"] <= 0.0)]
            there = ordering[(ordering["system"] == system)
                             & (ordering["rule"] == rule)]
            if not (len(here) and len(there)):
                continue
            assert np.sign(float(here["gap_pct"].iloc[0])) == \
                np.sign(float(there["gap_pct"].iloc[0])), system


class TestTheMechanismIsMeasuredRatherThanAsserted:
    """The section's first draft explained the widening by asserting that
    sheltering wealth from an assets test is worth more to the portfolio
    holding less of it. It is a plausible sentence, it was not derived from
    the model, and the split below does not support it -- the mean ratio
    moves against the all-equity portfolio under the test and barely moves
    at all without one, which says the annuity acts on the interaction and
    says nothing about which side of the taper does the work.

    So the discriminating fact gets a guard: an instrument effect would
    show up under both pensions, an interaction shows up under one.
    """

    @staticmethod
    def _frame(tested_span: float, untested_span: float) -> pd.DataFrame:
        rows = []
        for system, span in (("age_pension_matched", tested_span),
                             ("us_social_security", untested_span)):
            rows.append({
                "system": system,
                "mean_ratio_at_zero": 1.2, "mean_ratio_lowest": 1.2 - span,
                "mean_ratio_span": span,
                "mean_moves_against_the_challenger": span > 0.0,
                "tail_ratio_at_zero": 1.0, "tail_ratio_lowest": 1.0 - span,
                "tail_ratio_span": span,
                "tail_moves_against_the_challenger": span > 0.0})
        return pd.DataFrame.from_records(rows)

    def test_an_effect_only_under_the_test_is_called_an_interaction(
            self) -> None:
        got = anu.channel_verdict(self._frame(0.10, 0.003),
                                  "age_pension_matched", "us_social_security")
        assert got["measured"]
        assert got["it_is_the_test_and_not_the_instrument"]
        assert got["ratio_of_spans"] > 30.0

    def test_an_effect_under_both_pensions_is_not(self) -> None:
        """If annuitising moved the portfolio comparison by itself it would
        move it under an earnings-related pension too, and then the section
        would be reporting something about annuities rather than about
        means tests."""
        got = anu.channel_verdict(self._frame(0.10, 0.09),
                                  "age_pension_matched", "us_social_security")
        assert not got["it_is_the_test_and_not_the_instrument"]

    def test_a_frame_missing_the_control_is_not_measured(self) -> None:
        frame = self._frame(0.10, 0.003)
        got = anu.channel_verdict(frame[frame["system"] != "us_social_security"],
                                  "age_pension_matched", "us_social_security")
        assert not got.get("measured")

    def test_the_shipped_run_finds_the_interaction(self) -> None:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / "annuity_channel.csv"),
                         recursive=True)
        if not hits:
            pytest.skip("the channel split has not been generated")
        got = anu.channel_verdict(pd.read_csv(hits[0]),
                                  "age_pension_matched", "us_social_security")
        assert got["measured"]
        assert got["it_is_the_test_and_not_the_instrument"], got
        assert got["mean_moves_against_it_under_the_test"], got


class TestTheCrowdingInIsNotTheExemption:
    """Our sign is the opposite of the annuitisation literature's, and the
    first explanation to reach for -- that annuitised wealth is exempt from
    the assets test here and assessed in their settings -- is wrong. The
    grid carries both treatments, so it is checkable, and the paper's prose
    is generated from the check rather than from the guess."""

    @staticmethod
    def _wanted() -> pd.DataFrame:
        import glob

        hits = glob.glob(str(ROOT / "results" / "**" / "annuity_wanted.csv"),
                         recursive=True)
        if not hits:
            pytest.skip("the annuity study has not been run")
        return pd.read_csv(hits[0])

    def test_the_appetite_survives_the_annuity_being_assessed(self) -> None:
        want = self._wanted()
        block = want[want["rule"] == "fixed_real_rule"]
        tested = block[block["system"] == "age_pension_matched"]
        control = block[(block["system"] == "us_social_security")
                        & (block["assessed"] == 0.0)]
        if not len(tested) or not len(control):
            pytest.skip("the treatments are not both in the shipped run")
        assessed = float(
            tested[tested["assessed"] == 1.0]["gain_pct"].iloc[0])
        untested = float(control["gain_pct"].iloc[0])
        assert assessed > untested, (assessed, untested)

    def test_the_exemption_still_adds_to_it(self) -> None:
        """If exempting made no difference at all the paper would be
        claiming a distinction the data does not draw."""
        want = self._wanted()
        block = want[(want["rule"] == "fixed_real_rule")
                     & (want["system"] == "age_pension_matched")]
        if len(block) < 2:
            pytest.skip("the treatments are not both in the shipped run")
        exempt = float(block[block["assessed"] == 0.0]["gain_pct"].iloc[0])
        assessed = float(block[block["assessed"] == 1.0]["gain_pct"].iloc[0])
        assert exempt > assessed
