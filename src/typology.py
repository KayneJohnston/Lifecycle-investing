"""Which households the interaction reaches, and whether it survives them.

Section #incidence establishes the paper's sharpest result on one
household: scale the balance until the assets test binds and the wanted
equity share is 0% under a fixed real withdrawal and at or above the whole
portfolio under a rule that spends the balance.  The obvious objection is
that it is *one* household -- a single non-homeowner, on one earnings
profile, facing one pair of thresholds -- and that the households a real
means test binds differ from it in more than their balance.

That objection is right and it is answerable, because the three things
that separate one Australian retiree from another under the assets test
are all parameters of the schedule rather than of the model:

``home``
    the family home is exempt, so a homeowner is assessed against a lower
    free area than a renter holding the same wealth.  This is the largest
    single feature of the real system and the model's household owns
    nothing outside the portfolio, so both thresholds are run rather than
    one chosen.
``partner``
    a couple is assessed jointly, against a higher free area and for a
    higher combined payment.  The model has one earner, so a couple here
    is one portfolio meeting a couple's schedule -- which is the right
    comparison for the question being asked (does the threshold move the
    answer?) and the wrong one for a couple's actual retirement problem.
``earnings``
    the balance a career produces, which the sweep already moves directly.

Crossing the first two with the third asks whether the finding is a
property of the household it was found on.  Nothing about the mechanism
predicts that it should be -- Corollary 1 of the model section turns on
whether the withdrawal rule reads the balance, and no threshold appears
in it -- so this section is a test the paper can fail, and reporting it
as one is the point.

What this is not
----------------
It is not a calibration to the Australian wealth distribution.  Doing that
properly needs household-level data this study does not carry, and would
change what the balance grid *means*: here every type is run across the
same grid of positions against its own test, which answers "does the
result hold for this kind of household?" and not "how many households of
this kind are there?".  The second question is the more useful one for
policy and the first is the one a referee should ask of a result.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Household", "households", "sweep", "wanted_by_type", "verdict",
    "divide_table", "threshold_sensitivity",
]

#: The positions at which the assets test actually operates on the return.
#: Below the free area the pension is paid in full and above the cut-off it
#: is gone; in both, consumption under a balance-blind rule is flat in the
#: return rather than falling, so Corollary 1 predicts no corner there.
_BINDING: Tuple[str, ...] = ("inside the taper band", "above the cut-off")

#: The rule keys the two sides of the divide are run under. Named here so
#: the section cannot silently end up comparing two rules that fall on the
#: same side of Corollary 1, which is what makes the comparison a test.
BLIND: str = "constant_real"
READING: str = "constant_percent"


@dataclasses.dataclass(frozen=True)
class Household:
    """One kind of retiree, as the assets test sees them.

    ``free_area`` and ``full_rate`` are in multiples of economy-wide
    average earnings, as everywhere in this project, so the schedule
    travels across the panel's currencies unchanged.
    """

    key: str
    label: str
    free_area: float
    full_rate: float
    homeowner: bool
    partnered: bool

    @property
    def cutoff(self) -> float:
        """Where the pension reaches zero, given the taper it is run at."""
        return float("nan")

    def cut_out(self, taper: float) -> float:
        return self.free_area + self.full_rate / float(taper)


def households(single_free_area: float,
               renter_free_area: float,
               single_full_rate: float,
               couple_free_area_ratio: float,
               couple_renter_free_area_ratio: float,
               couple_rate_ratio: float) -> Tuple[Household, ...]:
    """The four types, built from the single homeowner's sourced figures.

    The single figures are statutory and cited in :mod:`src.pension`.  The
    couple figures are *ratios* applied to them rather than separately
    sourced dollars, and the paper says so: a ratio is a weaker claim than
    a rate, and :func:`threshold_sensitivity` sweeps it rather than
    trusting it.
    """
    return (
        Household("single_homeowner", "single, homeowner",
                  single_free_area, single_full_rate, True, False),
        Household("single_renter", "single, renter",
                  renter_free_area, single_full_rate, False, False),
        Household("couple_homeowner", "couple, homeowners",
                  single_free_area * float(couple_free_area_ratio),
                  single_full_rate * float(couple_rate_ratio), True, True),
        Household("couple_renter", "couple, renters",
                  renter_free_area * float(couple_renter_free_area_ratio),
                  single_full_rate * float(couple_rate_ratio), False, True),
    )


def sweep(simulate: Callable[[Household, str, float, float], Any],
          score: Callable[[Any], Dict[str, Any]],
          types: Sequence[Household],
          rules: Sequence[str],
          scales: Sequence[float],
          equities: Sequence[float],
          taper: float,
          log_every: int = 25) -> pd.DataFrame:
    """Every household type, rule, arriving balance and equity share.

    One row per combination, carrying the household's position against its
    *own* thresholds -- which is the whole point of crossing the types,
    since the same balance sits in different regimes for a renter and a
    homeowner.
    """
    rows: List[Dict[str, Any]] = []
    total = len(types) * len(rules) * len(scales) * len(equities)
    n = 0
    for who in types:
        cut = who.cut_out(taper)
        for rule in rules:
            for scale in scales:
                for share in equities:
                    outcome = simulate(who, rule, float(scale), float(share))
                    row: Dict[str, Any] = {
                        "household": who.key,
                        "household_label": who.label,
                        "homeowner": bool(who.homeowner),
                        "partnered": bool(who.partnered),
                        "rule": str(rule),
                        "scale": float(scale),
                        "equity": float(share),
                        "free_area": float(who.free_area),
                        "cutoff": float(cut),
                        "full_rate": float(who.full_rate),
                    }
                    row.update(score(outcome))
                    rows.append(row)
                    n += 1
                    if log_every and n % int(log_every) == 0:
                        LOGGER.info("  scored %d of %d", n, total)
    frame = pd.DataFrame.from_records(rows)
    if len(frame):
        frame["position"] = band_of(frame["median_wealth"].to_numpy(),
                                    frame["free_area"].to_numpy(),
                                    frame["cutoff"].to_numpy())
    return frame


def band_of(wealth: np.ndarray, free: np.ndarray,
            cut: np.ndarray) -> np.ndarray:
    """Where each row's median balance sits against *its own* thresholds."""
    out = np.full(len(wealth), "above the cut-off", dtype=object)
    out[wealth <= free] = "below the free area"
    inside = (wealth > free) & (wealth < cut)
    out[inside] = "inside the taper band"
    return out


def wanted_by_type(frame: pd.DataFrame,
                   column: str = "cec") -> pd.DataFrame:
    """The equity share each (type, rule, balance) prefers."""
    if not len(frame):
        return frame
    keys = ["household", "household_label", "rule", "scale"]
    best = frame.loc[frame.groupby(keys, sort=False)[column].idxmax()].copy()
    return best.sort_values(keys).reset_index(drop=True)


def divide_table(best: pd.DataFrame) -> pd.DataFrame:
    """One row per household type and position: what each rule wants there.

    The shape the finding is stated in.  A reader should be able to run a
    finger down the two rule columns and see whether the divide holds for
    every kind of retiree or only for the one it was found on.
    """
    if not len(best):
        return best
    rows: List[Dict[str, Any]] = []
    order = ["below the free area", "inside the taper band",
             "above the cut-off"]
    for (key, label), block in best.groupby(["household", "household_label"],
                                            sort=False):
        for band in order:
            here = block[block["position"] == band]
            if not len(here):
                continue
            row: Dict[str, Any] = {
                "household": key, "household_label": label,
                "position": band, "balances": int(len(here) // 2 or 1),
            }
            for rule in (BLIND, READING):
                arm = here[here["rule"] == rule]
                row[f"equity_{rule}"] = (float(arm["equity"].median())
                                         if len(arm) else float("nan"))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def verdict(table: pd.DataFrame,
            tolerance: float = 1e-9) -> Dict[str, Any]:
    """Does the divide survive the household, or was it one household's?

    Two questions, and the section has to answer both. Whether the *sign*
    of the gap between the two rules is the same everywhere is the claim
    Corollary 1 makes and the one the paper leans on. Whether the *level*
    each rule wants is the same everywhere is a different and weaker
    claim, and the thresholds move it.
    """
    if not len(table):
        return {"measured": False}
    blind, reading = f"equity_{BLIND}", f"equity_{READING}"
    if blind not in table or reading not in table:
        return {"measured": False}
    gaps = table[reading].to_numpy() - table[blind].to_numpy()
    finite = np.isfinite(gaps)
    if not finite.any():
        return {"measured": False}
    gaps = gaps[finite]
    rows = table[finite]
    found: Dict[str, Any] = {
        "measured": True,
        "cells": int(len(gaps)),
        "households": int(rows["household"].nunique()),
        "positions": int(rows["position"].nunique()),
        "divide_holds_everywhere": bool((gaps > tolerance).all()),
        "cells_holding": int((gaps > tolerance).sum()),
        "narrowest_gap": float(np.min(gaps)),
        "widest_gap": float(np.max(gaps)),
        "median_gap": float(np.median(gaps)),
        # The blind rule's corner, which Corollary 1 says is a property of
        # the budget line rather than of the household -- but only where
        # the test operates. The corollary distinguishes three regimes:
        # below the free area the pension is paid in full and consumption
        # is *flat* in the return, above the cut-off there is no pension
        # and consumption is flat again, and only inside the band does
        # consumption *fall*. So a corner at zero is what the corollary
        # predicts inside the band; outside it the return contributes
        # nothing either way and the optimum is not pinned. Reading the
        # corner across all three regimes tests a claim the corollary does
        # not make, which is how the first version of this verdict
        # reported a failure that was its own.
        "binding": _BINDING,
        "blind_at_zero_where_it_binds": bool(np.allclose(
            rows.loc[rows["position"].isin(_BINDING), blind].to_numpy(),
            0.0, atol=tolerance)),
        "blind_always_at_zero": bool(
            np.allclose(rows[blind].to_numpy(), 0.0, atol=tolerance)),
        "blind_highest": float(np.nanmax(rows[blind].to_numpy())),
        "reading_lowest": float(np.nanmin(rows[reading].to_numpy())),
    }
    if not found["divide_holds_everywhere"]:
        worst = rows.iloc[int(np.argmin(gaps))]
        found["worst_cell"] = (f"{worst['household_label']}, "
                               f"{worst['position']}")
    return found


def threshold_sensitivity(run: Callable[[float], pd.DataFrame],
                          ratios: Sequence[float]) -> pd.DataFrame:
    """The couple case re-run at several threshold ratios.

    The couple thresholds are ratios of the sourced single figures rather
    than separately verified statutory dollars, so the honest treatment is
    the one Section #leisure.1 gives the pre-eligibility payment: sweep the
    parameter and report where the answer stops depending on it.
    """
    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        table = run(float(ratio))
        found = verdict(table)
        rows.append({
            "free_area_ratio": float(ratio),
            "measured": bool(found.get("measured", False)),
            "divide_holds_everywhere": bool(
                found.get("divide_holds_everywhere", False)),
            "cells_holding": int(found.get("cells_holding", 0)),
            "cells": int(found.get("cells", 0)),
            "narrowest_gap": float(found.get("narrowest_gap", float("nan"))),
        })
    return pd.DataFrame.from_records(rows)
