"""Is the retiree's corner a corner, or the edge of the grid it was chosen from?

``src.incidence`` sweeps the equity share a means-tested retiree wants and
finds two corners: 0% everywhere under a fixed real withdrawal, and 100%
everywhere under a percentage-of-balance rule. The first is interior to the
grid in the only sense that matters -- there is nothing below zero to want --
and the balance sweep resolves it against every deletion of the panel.

The second is not. A grid that stops at 100% cannot report an optimum above
it, so "100% at every balance" is consistent with two very different worlds:
one where the retiree wants exactly the whole portfolio in equity, and one
where they want half as much again and the grid could not say so. The
prediction ``src.incidence`` is testing is about the *shape* of wanted equity
across the assets test -- highest inside the taper band, lower below the free
area, lowest above the cut-off -- and a shape read off a censored optimum is
not a shape at all. It is a ceiling.

This module lifts the ceiling. The retiree may borrow against the equity
sleeve at the realised bill rate plus a spread, which is
:class:`src.leverage.LeveredEvaluator`'s existing arithmetic applied to the
retirement window alone: leverage is one through the working years, so the
balance the household arrives with is the same balance ``src.incidence``
scaled, and only the retiree's own choice changes.

Three things come out of it, and the third is the one to report.

**Whether the corner was censored.** If the optimum stays at 1.0 with
borrowing available, the corner is real and ``src.incidence`` may state it as
one. If it moves above 1.0, the earlier answer was the grid's and not the
household's.

**Whether the shape appears once there is room for it.** The band, the free
area and the region past the cut-off can now be ranked against each other,
which at a shared ceiling they could not be.

**What borrowing has to cost before the corner comes back.** A break-even
spread is a number a reader can weigh against a real margin loan, where
"the optimum is levered" on its own is not.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Leverage of exactly one is the unlevered portfolio, and the grid must
#: contain it: the whole question is whether the optimum sits there.
UNLEVERED = 1.0

#: How far above the unlevered portfolio an optimum has to sit before the
#: earlier corner is called censored rather than confirmed. One grid step
#: would do; this is deliberately looser so a numerical tie is not a finding.
CENSORED_TOLERANCE = 1e-6


def sweep(score: Callable[[float, float, float], Mapping[str, Any]],
          scales: Sequence[float], leverages: Sequence[float],
          spreads: Sequence[float], log_every: int = 25) -> pd.DataFrame:
    """Score every (balance, leverage, borrowing spread) the study asks for.

    ``score(scale, leverage, spread)`` returns whatever the caller wants
    recorded -- at minimum a certainty equivalent under the key ``cec`` and
    the household's position against the assets test. Keeping the simulation
    behind a callable is what lets the tests drive this with arithmetic
    instead of a bootstrap.
    """
    rows: List[Dict[str, Any]] = []
    total = len(scales) * len(leverages) * len(spreads)
    for scale in scales:
        for spread in spreads:
            for leverage in leverages:
                row = {"scale": float(scale), "leverage": float(leverage),
                       "spread": float(spread)}
                row.update(score(float(scale), float(leverage),
                                 float(spread)))
                rows.append(row)
                if log_every and len(rows) % log_every == 0:
                    LOGGER.info("scored %d of %d", len(rows), total)
    return pd.DataFrame.from_records(rows)


def optimum(frame: pd.DataFrame, column: str = "cec") -> pd.DataFrame:
    """The best leverage at each balance and borrowing spread."""
    if not len(frame):
        return pd.DataFrame()
    keep = frame.loc[frame.groupby(["scale", "spread"])[column].idxmax()]
    grid = np.asarray(sorted(set(frame["leverage"])), dtype=float)
    keep = keep.copy()
    keep["at_ceiling"] = keep["leverage"] >= grid.max() - CENSORED_TOLERANCE
    keep["levered"] = keep["leverage"] > UNLEVERED + CENSORED_TOLERANCE
    return keep.sort_values(["spread", "scale"]).reset_index(drop=True)


def break_even_spread(optima: pd.DataFrame) -> float:
    """The lowest swept spread at which no balance wants to borrow.

    Returns NaN when every swept spread still leaves some balance levered,
    which is a finding rather than a failure: it says the sweep did not
    reach the price at which the unlevered corner comes back, and the
    caller should widen the grid or report the bound.
    """
    if not len(optima):
        return float("nan")
    for spread in sorted(set(optima["spread"])):
        block = optima[optima["spread"] == spread]
        if not bool(block["levered"].any()):
            return float(spread)
    return float("nan")


def verdict(optima: pd.DataFrame, frame: pd.DataFrame,
            column: str = "cec") -> Dict[str, Any]:
    """Whether the unlevered corner survives having room above it.

    ``censored`` is the headline: at a borrowing spread of zero -- the most
    generous price there is -- does the retiree still want exactly the
    unlevered portfolio? Anything else and the 100% reported elsewhere is
    the grid speaking.
    """
    if not len(optima):
        return {"measured": False}
    grid = np.asarray(sorted(set(frame["leverage"])), dtype=float)
    spreads = sorted(set(optima["spread"]))
    free = optima[optima["spread"] == min(spreads)]
    found: Dict[str, Any] = {
        "measured": True,
        "leverage_grid_low": float(grid.min()),
        "leverage_grid_high": float(grid.max()),
        "spreads": [float(s) for s in spreads],
        "balances": int(free["scale"].nunique()),
        "cheapest_spread": float(min(spreads)),
        "censored": bool(free["levered"].any()),
        "any_at_ceiling": bool(free["at_ceiling"].any()),
        "all_at_ceiling": bool(free["at_ceiling"].all()),
        "median_leverage": float(free["leverage"].median()),
        "max_leverage": float(free["leverage"].max()),
        "break_even_spread": break_even_spread(optima),
    }
    # What the earlier corner was worth against what the grid now offers:
    # if borrowing buys almost nothing, the censoring is a technicality.
    unlevered = frame[np.isclose(frame["leverage"], UNLEVERED)]
    at_cheapest = unlevered[unlevered["spread"] == found["cheapest_spread"]]
    if len(at_cheapest):
        merged = free.merge(at_cheapest[["scale", column]], on="scale",
                            suffixes=("", "_unlevered"))
        gain = (merged[column] / merged[f"{column}_unlevered"] - 1.0) * 100.0
        found["median_gain_pct"] = float(np.median(gain))
        found["max_gain_pct"] = float(np.max(gain))
    return found


def by_band(optima: pd.DataFrame, positions: Mapping[float, str],
            spread: float | None = None) -> pd.DataFrame:
    """The wanted leverage in each region of the assets test.

    ``positions`` maps a balance scale to the band it lands in, which the
    incidence sweep already works out; repeating that classification here
    would be a second place for it to be wrong.
    """
    if not len(optima):
        return pd.DataFrame()
    block = optima if spread is None else \
        optima[np.isclose(optima["spread"], float(spread))]
    if not len(block):
        return pd.DataFrame()
    named = block.assign(
        position=[positions.get(float(s), "") for s in block["scale"]])
    named = named[named["position"] != ""]
    if not len(named):
        return pd.DataFrame()
    return (named.groupby("position")
            .agg(balances=("scale", "size"),
                 median_leverage=("leverage", "median"),
                 low=("leverage", "min"), high=("leverage", "max"))
            .reset_index())


def shape_verdict(bands: pd.DataFrame,
                  tolerance: float = 0.05) -> Dict[str, Any]:
    """Whether the predicted ordering across the assets test appears.

    Section 2 of the paper predicts wanted equity highest inside the taper
    band, lower below the free area and lowest above the cut-off. At a
    shared ceiling that ordering is untestable; with room above it, it is
    not. ``differentiated`` is the weaker and more important claim: that the
    three regions want *different* amounts at all.
    """
    if not len(bands) or "position" not in bands:
        return {"measured": False}
    wanted = {str(r["position"]): float(r["median_leverage"])
              for _, r in bands.iterrows()}
    found: Dict[str, Any] = {"measured": True, "wanted": wanted,
                             "regions": int(len(wanted))}
    spread = max(wanted.values()) - min(wanted.values())
    found["spread"] = float(spread)
    found["differentiated"] = bool(spread > tolerance)
    band = next((v for k, v in wanted.items() if "band" in k), None)
    below = next((v for k, v in wanted.items() if "free area" in k), None)
    above = next((v for k, v in wanted.items() if "cut-off" in k), None)
    if band is not None and below is not None:
        found["band_beats_below"] = bool(band > below + tolerance)
    if band is not None and above is not None:
        found["band_beats_above"] = bool(band > above + tolerance)
    if None not in (band, below, above):
        found["prediction_holds"] = bool(
            band > below + tolerance and below > above + tolerance)
    return found
