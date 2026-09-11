"""Tests for the figure layer.

Two failures matter here and neither shows up in a numeric assertion. A tick
label can run off the edge of the canvas, in which case the reader sees
"...ternational Equity" and has to guess; or a label can be a raw config key,
in which case they see ``intl_eq``. Both were present before these tests.

The crop test measures ink at the image border rather than eyeballing the
result, so it fails if anyone reintroduces the original bug -- calling
``_save`` outside the ``rc_context`` that set ``savefig.bbox``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import data_loader as dl
from src import plots


LONG = "100% International Equity"


def _ink_on_the_border(path: Path, depth: int = 2) -> int:
    """Dark pixels within ``depth`` rows/columns of each edge.

    Anything here is text or a line that the crop cut through.
    """
    image = plt.imread(path)
    grey = image[..., :3].mean(axis=-1) if image.ndim == 3 else image
    dark = grey < 0.6
    return int(dark[:depth].sum() + dark[-depth:].sum()
               + dark[:, :depth].sum() + dark[:, -depth:].sum())


class TestSaveCrops:
    def test_a_long_tick_label_is_not_cut_off(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([0, 1], [1.0, 2.0])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LONG, LONG], rotation=30, ha="right")
        assert _ink_on_the_border(plots._save(fig, tmp_path, "long")) == 0

    def test_a_long_y_label_is_not_cut_off(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh([0, 1], [1.0, 2.0])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([LONG, "60/40 Domestic Equity/Domestic Bonds"])
        assert _ink_on_the_border(plots._save(fig, tmp_path, "wide")) == 0

    def test_the_saved_resolution_is_the_configured_one(self,
                                                        tmp_path: Path) -> None:
        # The bug this guards: `savefig.dpi` in STYLE never reached savefig,
        # so every figure came out at `figure.dpi` instead.
        fig, ax = plt.subplots(figsize=(4.0, 2.0))
        ax.plot([0, 1], [0, 1])
        image = plt.imread(plots._save(fig, tmp_path, "dpi"))
        assert image.shape[0] >= 2.0 * plots.STYLE["savefig.dpi"] * 0.9

    def test_the_directory_is_created(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        out = plots._save(fig, tmp_path / "nested" / "deeper", "f")
        assert out.exists()

    def test_the_figure_is_closed(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        plots._save(fig, tmp_path, "closed")
        assert not plt.fignum_exists(fig.number)


class TestLabels:
    def test_every_configured_strategy_has_a_short_form(
            self, real_config_or_skip) -> None:
        for key, spec in real_config_or_skip["strategies"].items():
            assert key in plots.STRATEGY_LABEL, key
            assert spec["label"] in plots.STRATEGY_LABEL, spec["label"]

    def test_the_key_and_the_label_give_the_same_short_form(self) -> None:
        for key, label in plots.STRATEGY_KEYS.items():
            assert plots._flat(key) == plots._flat(label)

    def test_a_short_form_is_shorter_than_what_it_replaces(self) -> None:
        for long, short in plots.STRATEGY_LABEL.items():
            if " " not in long:            # a bare key, not a display label
                continue
            assert len(short) <= len(long)

    def test_every_panel_series_has_a_display_name(self) -> None:
        for series in dl.CORE_SERIES:
            for table in (plots.SERIES_LABEL, plots.SERIES_ABBR):
                assert series in table
                assert "_" not in table[series]

    def test_an_unknown_label_still_gets_a_readable_form(self) -> None:
        assert plots._flat("some_new_strategy", 40) == "some new strategy"

    def test_nothing_ever_renders_as_an_empty_label(self) -> None:
        for value in ("", "   ", "x"):
            assert plots._flat(value) is not None
            assert plots._abbr(value) is not None

    def test_wrapping_keeps_the_breaks_a_short_form_chose(self) -> None:
        assert plots._wrap("Dom.\nequity", 40) == "Dom.\nequity"

    def test_wrapping_still_breaks_a_long_run_of_words(self) -> None:
        assert "\n" in plots._wrap("one two three four five six", 10)

    def test_a_strategy_label_never_stacks_three_lines(self) -> None:
        # A y-axis tick has one row of height; three lines collide with the
        # bars either side of it.
        for key in plots.STRATEGY_LABEL:
            assert plots._flat(key).count("\n") <= 1, key

    def test_the_short_form_keeps_the_words(self) -> None:
        assert plots._flat("international_equity").replace("\n", " ") \
            == "100% international equity"

    def test_no_short_form_leaves_a_gap_where_a_break_was(self) -> None:
        for value in plots.STRATEGY_LABEL.values():
            assert "/ " not in value
            assert "  " not in value

    def test_a_variant_name_is_compressed_but_still_identifies_itself(self
                                                                      ) -> None:
        assert plots._variant("Wealth trigger 20x income") == "Trigger 20x"
        assert plots._variant("Fixed age 63 (baseline)") == "Age 63 (base)"
        assert plots._variant("Flexible +/-3 years, 25x income") \
            == "Flex \u00b13y, 25x"


class TestGrid:
    def test_a_row_of_four_wraps_into_a_grid(self) -> None:
        fig, axes = plots._grid(4, 2.5)
        assert len(axes) == 4
        # Two columns, two rows: as wide as the page and twice the panel high.
        assert fig.get_figwidth() == pytest.approx(plots.PAGE_WIDTH_IN)
        assert fig.get_figheight() == pytest.approx(5.0)
        plt.close(fig)

    def test_an_odd_panel_spans_the_hole(self) -> None:
        fig, axes = plots._grid(3, 2.5)
        widths = [ax.get_position().width for ax in axes]
        assert widths[2] > widths[0] * 1.5
        plt.close(fig)

    def test_the_hole_can_be_left_for_a_legend(self) -> None:
        fig, axes, holes = plots._grid(5, 2.5, span_last=False, spare=True)
        assert len(axes) == 5 and len(holes) == 1
        plt.close(fig)

    def test_no_figure_is_authored_wider_than_the_text_column(self) -> None:
        for n in (1, 2, 3, 4, 5, 6):
            fig, _ = plots._grid(n, 2.5)
            assert fig.get_figwidth() <= plots.PAGE_WIDTH_IN + 1e-9
            plt.close(fig)


class TestEveryFigureIsAuthoredForThePage:
    """A source check, because the failure it guards against is invisible.

    A figure drawn twenty inches wide still looks fine on its own; it only
    falls apart once the paper scales it into a 16.2 cm column, which no
    other test in this suite exercises.
    """

    @staticmethod
    def _sources() -> str:
        return (Path(__file__).resolve().parents[1]
                / "src" / "plots.py").read_text()

    def test_no_call_sets_its_own_figure_width(self) -> None:
        import re
        bad = [m.group(0) for m
               in re.finditer(r"plt\.subplots\(figsize=\([^)]*\)", self._sources())
               if "PAGE_WIDTH_IN" not in m.group(0)]
        assert bad == []

    def test_the_page_width_matches_the_paper(self) -> None:
        # paper/style.py: A4 less 2.4 cm of margin either side.
        column_cm = 21.0 - 2 * 2.4
        assert plots.PAGE_WIDTH_IN == pytest.approx(column_cm / 2.54, abs=0.06)

    def test_the_style_saves_at_print_resolution(self) -> None:
        assert plots.STYLE["savefig.dpi"] >= 300
        assert plots.STYLE["savefig.bbox"] == "tight"


class TestIncidenceFigure:
    """The four-panel incidence figure, and the title it classifies.

    The title on panel four states the paper's one falsifiable prediction,
    so a bug that always printed the confirming title would be the worst
    kind: invisible, and exactly the claim a referee would check.
    """

    @staticmethod
    def _swept() -> "pd.DataFrame":
        import pandas as pd

        rows = []
        for alpha in (0.0, 0.5, 1.0):
            for equity in (0.0, 0.5, 1.0):
                rows.append({
                    "incidence": alpha, "equity": equity,
                    "cec": 1.0 + 0.1 * equity,
                    "cec_lifetime": 1.0 + 0.1 * equity - 0.05 * alpha,
                    "mean_working_consumption": 1.0 - 0.1 * alpha,
                    "median_wealth": 40.0, "free_area": 3.0, "cutoff": 7.0,
                    "share_below_the_free_area": 0.0,
                    "share_inside_the_taper_band": 0.0,
                    "share_above_the_cut-off": 1.0})
        return pd.DataFrame.from_records(rows)

    @staticmethod
    def _balances(band_equity: float = 0.9, above_equity: float = 0.4,
                  below_equity: float = 0.6) -> "pd.DataFrame":
        import pandas as pd

        rows = []
        wanted = {1.0: below_equity, 5.0: band_equity, 50.0: above_equity}
        for scale, wealth in ((0.1, 1.0), (0.5, 5.0), (1.0, 50.0)):
            for equity in (0.0, 0.4, 0.6, 0.9, 1.0):
                rows.append({
                    "scale": scale, "equity": equity,
                    "cec": 1.0 - abs(equity - wanted[wealth]),
                    "cec_lifetime": 1.0 - abs(equity - wanted[wealth]),
                    "median_wealth": wealth, "free_area": 3.0, "cutoff": 7.0,
                    "share_below_the_free_area": 1.0 if wealth < 3 else 0.0,
                    "share_inside_the_taper_band": 1.0 if 3 <= wealth < 7
                    else 0.0,
                    "share_above_the_cut-off": 1.0 if wealth >= 7 else 0.0})
        return pd.DataFrame.from_records(rows)

    def _render(self, tmp_path: Path, band: float, above: float,
                bridge_gap: float = 0.3) -> Path:
        import pandas as pd
        from src import incidence as ic

        grid = [0.0, 0.4, 0.6, 0.9, 1.0]
        swept = self._swept()
        frames = {}
        for arm, shift in zip(ic.ARMS, (0.0, bridge_gap, -bridge_gap)):
            block = self._balances(max(0.0, band - shift),
                                   max(0.0, above - shift))
            block["arm"] = arm
            frames[arm] = block
        balances = pd.concat(frames.values(), ignore_index=True)
        profiles = {a: ic.band_profile(b) for a, b in frames.items()}
        shapes = {a: ic.shape_verdict(b, grid) for a, b in profiles.items()}
        profile = pd.concat([profiles[a] for a in ic.ARMS],
                            ignore_index=True)
        return plots.plot_incidence(
            swept, balances, profile, ic.verdict(
                ic.optimum_by_incidence(swept)),
            shapes[ic.ARMS[0]], ic.bridge_verdict(profiles, shapes),
            tmp_path)

    def test_it_renders(self, tmp_path: Path) -> None:
        out = self._render(tmp_path, 0.9, 0.4)
        assert out.exists() and out.stat().st_size > 0

    def test_nothing_is_cropped(self, tmp_path: Path) -> None:
        assert _ink_on_the_border(self._render(tmp_path, 0.9, 0.4)) == 0

    def test_the_confirming_title_needs_the_confirming_data(self) -> None:
        from src import incidence as ic

        holds = ic.shape_verdict(ic.band_profile(self._balances(0.9, 0.4)),
                                 [0.0, 0.4, 0.6, 0.9, 1.0])
        fails = ic.shape_verdict(ic.band_profile(self._balances(0.4, 0.9)),
                                 [0.0, 0.4, 0.6, 0.9, 1.0])
        assert "most inside the band" in plots._shape_title(holds)
        assert "against the prediction" in plots._shape_title(fails)
        assert plots._shape_title(holds) != plots._shape_title(fails)

    def test_a_ceiling_optimum_is_named_a_ceiling(self) -> None:
        from src import incidence as ic

        ceiling = ic.shape_verdict(
            ic.band_profile(self._balances(1.0, 1.0, 1.0)),
            [0.0, 0.4, 0.6, 0.9, 1.0])
        assert "all equity" in plots._shape_title(ceiling)

    def test_an_unmeasured_shape_makes_no_claim(self) -> None:
        import pandas as pd

        neutral = plots._shape_title({"measured": False})
        assert "prediction" not in neutral
        assert plots._shape_title(
            {"measured": False}) == plots._shape_title({})

    def test_the_title_flags_a_verdict_the_bridge_flips(self) -> None:
        """Panel four draws both arms, so a title true of only one of them
        would be a mis-titled panel."""
        holds = {"measured": True, "prediction_holds": True}
        flips = {"measured": True, "the_bridge_changes_the_verdict": True}
        steady = {"measured": True, "the_bridge_changes_the_verdict": False}
        assert "unfunded" in plots._shape_title(holds, flips)
        assert "unfunded" not in plots._shape_title(holds, steady)
        assert "unfunded" not in plots._shape_title(holds, None)

    def test_the_bridge_note_names_the_direction(self) -> None:
        assert "cost" in plots._bridge_note(
            {"measured": True, "bridge_lowers_equity": True,
             "mean_equity_gap": 0.3})
        assert "add" in plots._bridge_note(
            {"measured": True, "bridge_raises_equity": True,
             "mean_equity_gap": -0.3})
        assert "barely" in plots._bridge_note({"measured": True})
        assert plots._bridge_note({"measured": False}) == ""

    def test_both_arms_are_drawn(self, tmp_path: Path) -> None:
        out = self._render(tmp_path, 0.9, 0.4, bridge_gap=0.5)
        assert out.exists() and out.stat().st_size > 0
        assert _ink_on_the_border(out) == 0

    def test_a_flat_profile_is_not_called_a_minimum(self) -> None:
        flat = {"measured": True, "flat_across_bands": True,
                "lowest_band": None}
        title = plots._shape_title(flat)
        assert "lowest" not in title
        assert "does not decide the portfolio" in title
