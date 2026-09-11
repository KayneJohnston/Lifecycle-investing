"""Who pays for the guarantee, and where the household sits against the test.

Two objections travel together in the referee's letter, and it is worth
saying at the outset that they are *not* the same objection, because the
first attempt at this module assumed they were and the sweep said otherwise.

**The first.** The Superannuation Guarantee has been free. Its statutory
incidence is on the employer, so the model has never charged the worker for
it, and every Australia-versus-America comparison has handed the Australian
arm a tenth of income the American arm does not get. Most of the empirical
incidence literature puts the economic incidence on wages instead.

**The second.** The household this project simulates is nowhere near the
assets test. It retires with roughly thirty-eight times average earnings
against a cut-off under seven, so the taper never operates on it, and a
paper whose subject is a means test has been demonstrating that means test
on the one household it cannot bind.

The tempting move is to answer both at once: charge the guarantee, watch
wealth fall, and let the household walk down onto the test. It does not
work, and the reason is instructive. Under full economic incidence the
worker funds the contribution out of wages -- take-home pay falls and
working-life consumption with it -- but *the contribution is still made*.
The balance at the pension age is identical to the last cent. Incidence
moves what the household gave up, not what it arrives with.

So the two objections need two dials, and this module has two.

* :func:`sweep` charges the guarantee. It answers what the free guarantee
  was worth to the Australian arm, and whether the allocation results
  elsewhere in the project were resting on it.
* :func:`balance_sweep` moves the balance. Same worker, same career, same
  forty years of take-home pay; only the amount reaching the pension age
  changes, so a difference in the retiree's allocation can be read as a
  response to the balance rather than to what was given up for it. This is
  the dial that puts a household inside the taper band, and the only one
  that can.

**What the theory predicts.** Write ``A`` for the free area, ``b`` for the
full rate and ``tau`` for the taper. Consumption in retirement is wealth
plus benefit, and the benefit is a kinked function of wealth:

* below ``A``      -- ``c = W R + b``: a floor of ``b`` under every outcome;
* inside the band  -- ``c = (1 - tau) W R + (b + tau A)``: risky exposure
  scaled *down* by ``1 - tau`` and the floor raised to ``b + tau A``;
* above the cut-off -- ``c = W R``: no floor at all.

Both middle terms push the same way. Inside the band the taper is
*insurance*: it damps the portfolio's contribution to consumption and lifts
the guaranteed part. So the optimal equity share should be high below the
free area, **higher again inside the band**, and lowest above the cut-off --
non-monotone in wealth, with its minimum where the pension has been tapered
away entirely rather than where it is being withdrawn.

That is the prediction :func:`shape_verdict` tests, and it is a prediction
that can fail: the same table implies a household deep inside the band
holds a smaller portfolio against an unchanged floor, and past some point
that portfolio's own volatility is what it eats.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Where a household sits against the assets test, in reading order.
BANDS: Tuple[str, ...] = ("below the free area", "inside the taper band",
                          "above the cut-off")

#: How much higher the optimal equity share must be inside the band than
#: above the cut-off before the difference is called a difference. One
#: step of a twenty-point grid: smaller than that and the sweep cannot
#: tell a real gap from the resolution it was run at.
SHAPE_TOLERANCE: float = 0.05


def thresholds(spec: Any) -> Tuple[float, float]:
    """``(free area, cut-off)`` in the same units as wealth at retirement."""
    economy = float(spec.deterministic_income().mean())
    free = spec.pension_free_area * economy
    full = spec.pension_full_rate * economy
    cut = free + full / spec.pension_taper if spec.pension_taper else np.inf
    return float(free), float(cut)


def band_of(wealth: np.ndarray, free: float, cut: float) -> np.ndarray:
    """Which side of the test each path falls on."""
    w = np.asarray(wealth, dtype=float)
    return np.where(w <= free, BANDS[0],
                    np.where(w >= cut, BANDS[2], BANDS[1]))


def _row(spec: Any, outcome: Any) -> Dict[str, Any]:
    """The position of one simulated household against the test."""
    free, cut = thresholds(spec)
    economy = float(spec.deterministic_income().mean())
    wealth = np.asarray(outcome.wealth_at_retirement, dtype=float)
    bands = band_of(wealth, free, cut)
    row: Dict[str, Any] = {
        "median_wealth": float(np.median(wealth)) / economy,
        "free_area": free / economy, "cutoff": cut / economy,
    }
    for name in BANDS:
        row[f"share_{name.replace(' ', '_')}"] = float(np.mean(bands == name))
    return row


def sweep(simulate: Callable[[float, float], Any],
          spec_for: Callable[[float], Any],
          incidences: Sequence[float], equities: Sequence[float],
          score: Callable[[Any], Dict[str, Any]],
          log_every: int = 5) -> pd.DataFrame:
    """Every share of the guarantee charged, crossed with every allocation.

    ``simulate(incidence, equity)`` returns one outcome; ``spec_for`` gives
    the spec at that incidence so the thresholds can be read from the same
    household the outcome describes.
    """
    rows: List[Dict[str, Any]] = []
    n = 0
    for alpha in incidences:
        spec = spec_for(float(alpha))
        for share in equities:
            outcome = simulate(float(alpha), float(share))
            row: Dict[str, Any] = {"incidence": float(alpha),
                                   "equity": float(share)}
            row.update(_row(spec, outcome))
            row.update(score(outcome))
            rows.append(row)
            n += 1
            if log_every and n % int(log_every) == 0:
                LOGGER.info("  scored %d of %d", n,
                            len(incidences) * len(equities))
    return pd.DataFrame.from_records(rows)


#: The three arms the balance dial is run at, and what each one controls
#: for. The first is the clean test: retiring when the pension starts puts
#: every retirement year under the means test, which is the only
#: arrangement in which the taper prediction can be asked on its own.
#:
#: The other two are the controls, and each rules out a different way of
#: being wrong about the first. Retiring four years early -- the date every
#: other section of this project uses -- leaves four years standing on a
#: partial benefit, and at this risk aversion a few thin years at the start
#: of retirement can dominate the certainty equivalent of a small balance;
#: without that arm, a hole in the floor could be reported as a taper
#: effect. Spending a share of the balance rather than a fixed real amount
#: rules out the other: under a rule that never spends what the portfolio
#: earns, a good equity outcome raises assessable assets, withdraws the
#: pension, and raises consumption not at all -- so the taper would look
#: like a pure loss whatever the household's risk preferences were.
ARMS: Tuple[str, ...] = ("pension from the day work stops",
                         "four years before the pension starts",
                         "spending a share of the balance")


def balance_sweep(simulate: Callable[[float, float], Any],
                  spec_for: Callable[[float], Any],
                  scales: Sequence[float], equities: Sequence[float],
                  score: Callable[[Any], Dict[str, Any]],
                  log_every: int = 5, arm: str | None = None) -> pd.DataFrame:
    """Every arriving balance, crossed with every allocation.

    The dial is a multiple of the balance this model's own contribution
    assumptions produce, applied at the retirement boundary. Working life is
    identical across the grid by construction, so the retiree's problem is
    the only thing that differs.
    """
    rows: List[Dict[str, Any]] = []
    n = 0
    for scale in scales:
        spec = spec_for(float(scale))
        for share in equities:
            outcome = simulate(float(scale), float(share))
            row: Dict[str, Any] = {"scale": float(scale),
                                   "equity": float(share)}
            if arm is not None:
                row["arm"] = str(arm)
            row.update(_row(spec, outcome))
            row.update(score(outcome))
            rows.append(row)
            n += 1
            if log_every and n % int(log_every) == 0:
                LOGGER.info("  scored %d of %d", n,
                            len(scales) * len(equities))
    return pd.DataFrame.from_records(rows)


def _best_by(frame: pd.DataFrame, key: str, column: str) -> pd.DataFrame:
    """The row each level of ``key`` prefers, sorted by ``key``."""
    if not len(frame):
        return frame
    best = frame.loc[frame.groupby(key)[column].idxmax()].copy()
    return best.sort_values(key).reset_index(drop=True)


def optimum_by_incidence(frame: pd.DataFrame,
                         column: str = "cec_lifetime") -> pd.DataFrame:
    """The equity share each incidence wants, and where it leaves the household.

    Scored on the *lifetime* measure by default. The project's usual
    certainty equivalent covers the retirement window only, and the
    incidence of a working-life contribution is exactly invisible to it --
    so an optimum selected on that measure would be flat by construction,
    and reporting the flat line as a result would be an artefact.
    """
    best = _best_by(frame, "incidence", column)
    if not len(best):
        return best
    keep = (["incidence", "equity", column, "cec", "cec_lifetime",
             "mean_working_consumption", "median_wealth", "cutoff",
             "free_area"]
            + [c for c in frame.columns if c.startswith("share_")])
    seen: List[str] = []
    for c in keep:
        if c in best.columns and c not in seen:
            seen.append(c)
    return best[seen]


def optimum_by_balance(frame: pd.DataFrame,
                       column: str = "cec") -> pd.DataFrame:
    """The equity share each arriving balance wants, with its band position."""
    best = _best_by(frame, "scale", column)
    if not len(best):
        return best
    cut = float(best["cutoff"].iloc[0])
    best["over_cutoff"] = best["median_wealth"] / cut if cut else np.nan
    best["position"] = np.where(
        best["median_wealth"] <= best["free_area"], BANDS[0],
        np.where(best["median_wealth"] >= best["cutoff"], BANDS[2], BANDS[1]))
    keep = (["arm", "scale", "position", "median_wealth", "over_cutoff",
             "equity", column, "prob_ruin", "cutoff", "free_area"]
            + [c for c in frame.columns if c.startswith("share_")])
    seen: List[str] = []
    for c in keep:
        if c in best.columns and c not in seen:
            seen.append(c)
    return best[seen]


def verdict(optima: pd.DataFrame) -> Dict[str, Any]:
    """What charging the guarantee does, and what it does not do.

    Classified rather than asserted, because the first version of this
    module asserted that charging the guarantee would move the household
    onto the test, and it does not: incidence changes what the worker gave
    up, not the balance they arrive with.
    """
    if not len(optima):
        return {"measured": False}
    free_end = optima.iloc[0]
    paid_end = optima.iloc[-1]
    in_band = "share_inside_the_taper_band"
    above = "share_above_the_cut-off"
    found: Dict[str, Any] = {
        "measured": True,
        "free_incidence": float(free_end["incidence"]),
        "paid_incidence": float(paid_end["incidence"]),
        "free_wealth": float(free_end["median_wealth"]),
        "paid_wealth": float(paid_end["median_wealth"]),
        "cutoff": float(free_end["cutoff"]),
        "free_area": float(free_end["free_area"]),
        "free_equity": float(free_end["equity"]),
        "paid_equity": float(paid_end["equity"]),
        "wealth_fall_pct": 100.0 * (paid_end["median_wealth"]
                                    / free_end["median_wealth"] - 1.0),
    }
    for measure in ("cec", "cec_lifetime"):
        if measure not in optima:
            continue
        found[f"free_{measure}"] = float(free_end[measure])
        found[f"paid_{measure}"] = float(paid_end[measure])
        found[f"{measure}_fall_pct"] = 100.0 * (
            paid_end[measure] / free_end[measure] - 1.0)
    if "mean_working_consumption" in optima:
        free_work = float(free_end["mean_working_consumption"])
        paid_work = float(paid_end["mean_working_consumption"])
        found["free_working_consumption"] = free_work
        found["paid_working_consumption"] = paid_work
        found["working_fall_pct"] = 100.0 * (paid_work / free_work - 1.0)
    # The retirement-window measure *cannot* move: the retiree's problem is
    # identical at every incidence. Recorded so the section can report the
    # invariance as the check it is rather than as a null result.
    if "cec" in optima:
        found["retirement_cec_invariant"] = bool(
            abs(found.get("cec_fall_pct", 0.0)) < 1e-9)
    if in_band in optima:
        found["free_in_band"] = float(free_end[in_band])
        found["paid_in_band"] = float(paid_end[in_band])
    if above in optima:
        found["free_above"] = float(free_end[above])
        found["paid_above"] = float(paid_end[above])
    # Does charging the guarantee reach the test at all? The answer on this
    # calibration is no, and the section is written round that.
    found["reaches_the_test"] = bool(
        found["paid_wealth"] < found["cutoff"]
        or found.get("paid_in_band", 0.0) > 0.10)
    # Incidence is a pure transfer within the worker's own budget: the
    # contribution is made either way, so the balance cannot move. Recorded
    # as a check on the simulator rather than as a finding.
    found["balance_unchanged"] = bool(
        abs(found["wealth_fall_pct"]) < 1e-6)
    found["equity_falls_when_charged"] = bool(
        found["paid_equity"] < found["free_equity"])
    found["equity_unchanged"] = bool(
        found["paid_equity"] == found["free_equity"])
    return found


def shape_verdict(optima: pd.DataFrame,
                  equities: Sequence[float] | None = None,
                  tolerance: float = SHAPE_TOLERANCE) -> Dict[str, Any]:
    """Whether the optimum has the shape the kinked budget line predicts.

    The prediction is specific and falsifiable: equity highest inside the
    band, lowest above the cut-off. Anything else -- monotone in wealth, or
    a minimum in the band -- is the theory being wrong, and is reported that
    way.
    """
    if not len(optima) or "position" not in optima:
        return {"measured": False}
    by_band = {band: block for band, block in optima.groupby("position")}
    found: Dict[str, Any] = {
        "measured": True,
        "bands_reached": [b for b in BANDS if b in by_band],
        "scales": int(len(optima)),
    }
    for band in BANDS:
        key = band.replace(" ", "_").replace("-", "_")
        if band not in by_band:
            continue
        wanted = by_band[band]["equity"]
        # The median, not the maximum. A band summarised by its most
        # equity-hungry member lets one household at the top of the wealth
        # grid speak for every household in the band, and the comparison
        # the shape test makes is between typical households.
        found[f"equity_{key}"] = float(wanted.median())
        found[f"equity_{key}_min"] = float(wanted.min())
        found[f"equity_{key}_max"] = float(wanted.max())
        found[f"points_{key}"] = int(len(wanted))
    band_key = "equity_inside_the_taper_band"
    above_key = "equity_above_the_cut_off"
    below_key = "equity_below_the_free_area"
    if band_key in found and above_key in found:
        gap = found[band_key] - found[above_key]
        found["band_above_gap"] = float(gap)
        found["band_beats_above"] = bool(gap > tolerance)
        found["above_beats_band"] = bool(gap < -tolerance)
    if band_key in found and below_key in found:
        found["band_beats_below"] = bool(
            found[band_key] - found[below_key] > tolerance)
    # The headline: is the minimum where the theory says it is?
    reached = [b for b in BANDS if f"equity_{b.replace(' ', '_').replace('-', '_')}"
               in found]
    if len(reached) > 1:
        shares = {b: found[f"equity_{b.replace(' ', '_').replace('-', '_')}"]
                  for b in reached}
        # A flat profile has no lowest band, and ``min`` would invent one by
        # picking whichever happens to come first. Said explicitly, because
        # "the optimum is lowest below the free area" is a claim, and when
        # every band wants the same thing it is a false one.
        found["flat_across_bands"] = bool(
            max(shares.values()) - min(shares.values()) <= tolerance)
        found["band_spread"] = float(max(shares.values())
                                     - min(shares.values()))
        found["common_equity"] = float(np.median(list(shares.values())))
        lowest = min(shares, key=lambda b: shares[b])
        found["lowest_band"] = None if found["flat_across_bands"] else lowest
        found["minimum_above_the_cutoff"] = bool(
            not found["flat_across_bands"] and lowest == BANDS[2])
        found["monotone_in_wealth"] = bool(
            list(shares.values()) == sorted(shares.values(), reverse=True)
            or list(shares.values()) == sorted(shares.values()))
        found["prediction_holds"] = bool(
            found.get("minimum_above_the_cutoff")
            and found.get("band_beats_above", False))
    # A corner is a corner whatever the theory says about it: an optimum
    # sitting on the edge of the grid it was chosen from is a truncation,
    # and a *shape* read off truncated optima is worth less still.
    grid = np.asarray(list(equities) if equities is not None
                      else optima["equity"].unique(), dtype=float)
    if grid.size:
        from .accumulation import at_grid_edge

        chosen = optima["equity"].to_numpy(dtype=float)
        found["equity_grid_low"] = float(grid.min())
        found["equity_grid_high"] = float(grid.max())
        found["at_grid_edge"] = [bool(at_grid_edge(grid, x)) for x in chosen]
        found["any_at_grid_edge"] = bool(any(found["at_grid_edge"]))
        found["all_at_grid_edge"] = bool(all(found["at_grid_edge"]))
        found["all_at_ceiling"] = bool(
            np.all(chosen >= grid.max() - 1e-9))
    return found


def band_profile(frame: pd.DataFrame, column: str = "cec") -> pd.DataFrame:
    """The optimum at each balance, ordered by where it leaves the household.

    This is the panel the theory speaks to: plotted against position, the
    optimum should dip only once the pension has been tapered away, not
    while it is being withdrawn.
    """
    best = optimum_by_balance(frame, column)
    if not len(best):
        return best
    return best.sort_values("median_wealth").reset_index(drop=True)


def contrast(profiles: Mapping[str, pd.DataFrame],
             shapes: Mapping[str, Mapping[str, Any]], left: str, right: str,
             tolerance: float = SHAPE_TOLERANCE) -> Dict[str, Any]:
    """How far two arms of the balance dial disagree about the allocation.

    Both controls are read the same way, so they are computed the same way:
    the average and worst gap in wanted equity, where the worst one falls,
    and whether the two arms come to different verdicts on the shape test.
    That last field is the one that matters -- a control only earns its
    runtime if it can overturn the result it is controlling for.
    """
    if left not in profiles or right not in profiles:
        return {"measured": False}
    a, b = profiles[left], profiles[right]
    if not len(a) or not len(b):
        return {"measured": False}
    merged = a.merge(b, on="scale", suffixes=("_aligned", "_bridged"))
    if not len(merged):
        return {"measured": False}
    gap = (merged["equity_aligned"] - merged["equity_bridged"]).to_numpy(
        dtype=float)
    found: Dict[str, Any] = {
        "measured": True,
        "scales": int(len(merged)),
        "mean_equity_gap": float(np.mean(gap)),
        "max_equity_gap": float(np.max(np.abs(gap))),
        "bridge_lowers_equity": bool(np.mean(gap) > tolerance),
        "bridge_raises_equity": bool(np.mean(gap) < -tolerance),
        "aligned_mean_equity": float(merged["equity_aligned"].mean()),
        "bridged_mean_equity": float(merged["equity_bridged"].mean()),
    }
    # Where the two arms disagree most, in wealth terms: the claim is that
    # the gap bites hardest on the households with least to bridge with.
    worst = int(np.argmax(np.abs(gap)))
    found["worst_wealth"] = float(merged["median_wealth_aligned"].iloc[worst])
    found["worst_position"] = str(merged["position_aligned"].iloc[worst])
    found["left"], found["right"] = left, right
    for key, shape in (("aligned", shapes.get(left, {})),
                       ("bridged", shapes.get(right, {}))):
        found[f"{key}_prediction_holds"] = bool(shape.get("prediction_holds"))
        found[f"{key}_lowest_band"] = shape.get("lowest_band")
    found["shape_label_flips"] = bool(
        found["aligned_prediction_holds"] != found["bridged_prediction_holds"])
    # Two arms can carry the same shape label and still disagree completely
    # about the allocation -- one because the prediction fails, the other
    # because every optimum sits on the grid ceiling and there is no shape
    # to test. So a control counts as mattering if it moves the allocation
    # *or* flips the label; reading only the label would let a hundred-point
    # swing be reported as "no difference".
    found["moves_the_allocation"] = bool(
        found["bridge_lowers_equity"] or found["bridge_raises_equity"])
    found["changes_the_verdict"] = bool(
        found["shape_label_flips"] or found["moves_the_allocation"])
    found["the_bridge_changes_the_verdict"] = found["changes_the_verdict"]
    for key, arm in (("aligned", left), ("bridged", right)):
        found[f"{key}_untestable"] = bool(
            shapes.get(arm, {}).get("all_at_ceiling"))
    return found


def bridge_verdict(profiles: Mapping[str, pd.DataFrame],
                   shapes: Mapping[str, Mapping[str, Any]],
                   tolerance: float = SHAPE_TOLERANCE) -> Dict[str, Any]:
    """What four years without the full pension do to the allocation.

    This project's headline mechanism is that an asset-tested pension
    reverses the all-equity prescription by *removing a floor*. A retiree
    who stops four years before the pension begins stands on part of a
    floor for those four years, so this comparison is that mechanism
    measured rather than argued for -- and it is the control that stops a
    hole in the floor being reported as a taper effect.
    """
    return contrast(profiles, shapes, ARMS[0], ARMS[1], tolerance)


def rule_verdict(profiles: Mapping[str, pd.DataFrame],
                 shapes: Mapping[str, Mapping[str, Any]],
                 tolerance: float = SHAPE_TOLERANCE) -> Dict[str, Any]:
    """What the withdrawal rule does to the allocation near the test.

    A fixed real withdrawal never spends what the portfolio earns, so a
    household near the assets test that does well in equities converts the
    gain entirely into assessable assets: the pension is withdrawn and
    consumption does not rise. Under that rule the taper is a pure loss and
    equity is priced accordingly, whatever the shape of the budget line.
    A percentage-of-balance rule spends the gain instead. The difference
    between the two arms says how much of the result is the means test and
    how much is the rule it was measured under.
    """
    if len(ARMS) < 3:
        return {"measured": False}
    return contrast(profiles, shapes, ARMS[0], ARMS[2], tolerance)


def robustness(frame: pd.DataFrame, columns: Sequence[str],
               equities: Sequence[float] | None = None,
               tolerance: float = SHAPE_TOLERANCE) -> pd.DataFrame:
    """The balance profile re-read under each scoring column.

    The headline of the balance dial is a corner: no equity anywhere the
    test reaches under one withdrawal rule, all of it under another. A
    corner at zero is exactly where a constant-relative-risk-aversion
    objective is least trustworthy -- with a consumption floor near zero,
    utility is unbounded below, and a handful of near-starvation years can
    decide the optimum on their own. So the same simulated outcomes are
    scored at several risk aversions and several floors, and the profile is
    re-read under each.

    Nothing is re-simulated: the scoring columns are all computed from one
    set of outcomes, which is what makes this cheap enough to be routine.
    """
    rows: List[Dict[str, Any]] = []
    for column in columns:
        if column not in frame:
            continue
        profile = optimum_by_balance(frame, column)
        shape = shape_verdict(profile, equities, tolerance)
        row: Dict[str, Any] = {"scoring": str(column)}
        for band in BANDS:
            key = "equity_" + band.replace(" ", "_").replace("-", "_")
            row[band] = float(shape.get(key, np.nan))
        bound = profile[profile["median_wealth"] <= profile["cutoff"]] \
            if "cutoff" in profile else profile.iloc[0:0]
        row["equity_where_the_test_binds"] = (
            float(bound[ "equity"].median()) if len(bound) else np.nan)
        row["equity_overall"] = float(profile["equity"].median()) \
            if len(profile) else np.nan
        row["flat_across_bands"] = bool(shape.get("flat_across_bands", False))
        row["all_at_ceiling"] = bool(shape.get("all_at_ceiling", False))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def robustness_verdict(table: pd.DataFrame, baseline: str = "cec",
                       tolerance: float = SHAPE_TOLERANCE) -> Dict[str, Any]:
    """Whether the corner survives being scored a different way.

    ``survives`` is the field the prose turns on, and it is deliberately
    demanding: every alternative scoring must put the household within
    ``tolerance`` of where the baseline put it. A result that holds at one
    risk aversion and one floor is a result about that specification.
    """
    if not len(table) or baseline not in set(table["scoring"]):
        return {"measured": False}
    base = table[table["scoring"] == baseline].iloc[0]
    others = table[table["scoring"] != baseline]
    anchor = float(base["equity_where_the_test_binds"])
    found: Dict[str, Any] = {
        "measured": True,
        "baseline": str(baseline),
        "baseline_equity": anchor,
        "specifications": int(len(table)),
    }
    if not len(others):
        found["survives"] = True
        return found
    gaps = (others["equity_where_the_test_binds"].to_numpy(dtype=float)
            - anchor)
    found["max_departure"] = float(np.nanmax(np.abs(gaps)))
    found["survives"] = bool(found["max_departure"] <= tolerance)
    found["range_low"] = float(np.nanmin(
        table["equity_where_the_test_binds"].to_numpy(dtype=float)))
    found["range_high"] = float(np.nanmax(
        table["equity_where_the_test_binds"].to_numpy(dtype=float)))
    worst = int(np.nanargmax(np.abs(gaps)))
    found["worst_scoring"] = str(others["scoring"].iloc[worst])
    found["worst_equity"] = float(
        others["equity_where_the_test_binds"].iloc[worst])
    return found
