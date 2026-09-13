"""The interval the paper's own argument says it should be quoting.

Section #ordering.3 reports a delete-one-country jackknife and then, in
terms, says two things that undercut it.  The sixteen markets are not
sixteen independent draws -- they share the century's wars and
depressions -- and the sub-panels overlap in *construction* as well as in
history, because the international sleeve is a leave-one-out average over
whatever markets remain, so dropping one rebuilds every other country's
foreign leg.  Both make a standard error built on those deletions
narrower than the evidence warrants.

A section that argues its own interval is understated and then prints the
interval is in an unstable position.  The obvious question is *by how
much*, and the machinery to answer it is the same machinery that produced
the deletions: resample the panel and rebuild it.

What this does
--------------
Draw sixteen markets **with replacement** from the sixteen, rebuild the
panel from the draw -- which rebuilds the international leg from the same
draw, so the overlap the jackknife cannot see is inside the resampling
rather than outside it -- and re-run the cells the paper's headlines come
from.  The spread of the gap across replicates is a sampling distribution
that assumes nothing about how the deletions scatter and nothing about
their independence.

A resample can draw the same market several times and omit others, which
is the point: it asks what this study would have concluded had the
recorded evidence been a different sixteen markets from the same
population, and that is the uncertainty a reader should weigh a sign
against.

What it cannot do
-----------------
It cannot manufacture information the panel does not carry.  Sixteen
developed markets resampled sixteen at a time is still sixteen markets,
and a bootstrap over them inherits every selection effect in what was
recorded and licensed.  It is a better-calibrated statement of the same
evidence, not more evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "draw", "bootstrap", "interval", "verdict", "compare_to_jackknife",
    "construction_sweep", "construction_verdict",
    "moved_signs_were_already_unsigned",
]


def draw(countries: Sequence[str], rng: Any) -> List[str]:
    """One resample of the panel: ``n`` markets drawn from ``n`` with
    replacement."""
    picks = rng.integers(0, len(countries), size=len(countries))
    return [str(countries[int(i)]) for i in picks]


def bootstrap(gaps_for: Callable[[Sequence[str]], pd.DataFrame],
              countries: Sequence[str],
              replicates: int,
              seed: int,
              log_every: int = 20) -> pd.DataFrame:
    """``replicates`` resampled panels, each re-run and recorded.

    ``gaps_for(markets)`` returns a gap table built on a panel holding
    exactly those markets, repeats included.  One row per replicate per
    cell, carrying the draw so a reader can see that the resamples really
    do differ.
    """
    rng = np.random.default_rng(int(seed))
    rows: List[pd.DataFrame] = []
    for i in range(int(replicates)):
        markets = draw(countries, rng)
        block = gaps_for(markets).copy()
        block.insert(0, "replicate", i)
        block["distinct_markets"] = len(set(markets))
        block["draw"] = ",".join(sorted(markets))
        rows.append(block)
        if log_every and (i + 1) % int(log_every) == 0:
            LOGGER.info("  resampled %d of %d", i + 1, int(replicates))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame())


def interval(frame: pd.DataFrame, point: pd.DataFrame,
             column: str = "gap_pct",
             level: float = 0.95) -> pd.DataFrame:
    """A percentile interval per cell, beside the point estimate.

    Percentile rather than normal-approximation, because the whole reason
    for running this is that the jackknife's normal approximation is the
    part Section #ordering.3 distrusts: its deletions are skewed enough
    that a standard error built on them is the weakest of the statistics
    the section reports.  A percentile interval assumes no shape.
    """
    if not len(frame):
        return pd.DataFrame()
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    anchor = point.set_index(["system", "rule"])[column].to_dict()
    rows: List[Dict[str, Any]] = []
    for (system, rule), block in frame.groupby(["system", "rule"],
                                               sort=False):
        values = block[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        measured = float(anchor.get((system, rule), np.nan))
        rows.append({
            "system": system, "rule": rule,
            "gap_pct": measured,
            "replicates": int(len(values)),
            "boot_mean": float(values.mean()),
            "boot_se": float(values.std(ddof=1)) if len(values) > 1
            else float("nan"),
            "ci_low": float(np.quantile(values, lo_q)),
            "ci_high": float(np.quantile(values, hi_q)),
            "share_keeping_sign": (
                float(np.mean(np.sign(values) == np.sign(measured)))
                if np.isfinite(measured) and measured != 0.0
                else float("nan")),
            "excludes_zero": bool(
                np.quantile(values, lo_q) > 0.0
                or np.quantile(values, hi_q) < 0.0),
        })
    return pd.DataFrame.from_records(rows)


def compare_to_jackknife(boot: pd.DataFrame,
                         jackknife: pd.DataFrame) -> pd.DataFrame:
    """How much wider the resampled interval is than the delete-one one.

    The number Section #ordering.3 owes a reader.  It concedes that its
    standard error is too narrow and does not say by how much; this is by
    how much, cell by cell.
    """
    if not len(boot) or not len(jackknife):
        return pd.DataFrame()
    jack = jackknife.set_index(["system", "rule"])
    rows: List[Dict[str, Any]] = []
    for _, row in boot.iterrows():
        key = (row["system"], row["rule"])
        if key not in jack.index:
            continue
        other = jack.loc[key]
        jack_se = float(other["standard_error"])
        rows.append({
            "system": row["system"], "rule": row["rule"],
            "gap_pct": float(row["gap_pct"]),
            "jackknife_se": jack_se,
            "bootstrap_se": float(row["boot_se"]),
            "se_ratio": (float(row["boot_se"]) / jack_se
                         if jack_se else float("nan")),
            "jackknife_excludes_zero": bool(other["ci_excludes_zero"]),
            "bootstrap_excludes_zero": bool(row["excludes_zero"]),
            "agree": bool(other["ci_excludes_zero"]
                          == row["excludes_zero"]),
        })
    return pd.DataFrame.from_records(rows)


def verdict(compared: pd.DataFrame,
            headline: Tuple[str, str] | None = None) -> Dict[str, Any]:
    """Whether the paper's signs survive the wider interval.

    Two things a reader needs and one the paper needs.  Whether every cell
    the jackknife signs is still signed; by what factor the jackknife
    understated; and, for the cell the paper leads with, the answer in
    full.
    """
    if not len(compared):
        return {"measured": False}
    ratios = compared["se_ratio"].to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    found: Dict[str, Any] = {
        "measured": True,
        "cells": int(len(compared)),
        "median_se_ratio": float(np.median(ratios)) if len(ratios)
        else float("nan"),
        "widest_se_ratio": float(np.max(ratios)) if len(ratios)
        else float("nan"),
        "narrowest_se_ratio": float(np.min(ratios)) if len(ratios)
        else float("nan"),
        # The bootstrap is wider wherever the jackknife's independence
        # assumption was doing work. If it came out narrower the concession
        # in Section #ordering.3 would be the wrong way round, so this is
        # measured rather than assumed.
        "bootstrap_wider_everywhere": bool((ratios > 1.0).all())
        if len(ratios) else False,
        "signs_agree": int(compared["agree"].sum()),
        "signs_lost": [
            f"{r['system']}/{r['rule']}"
            for _, r in compared.iterrows()
            if bool(r["jackknife_excludes_zero"])
            and not bool(r["bootstrap_excludes_zero"])],
        "signs_gained": [
            f"{r['system']}/{r['rule']}"
            for _, r in compared.iterrows()
            if not bool(r["jackknife_excludes_zero"])
            and bool(r["bootstrap_excludes_zero"])],
    }
    found["every_sign_survives"] = not found["signs_lost"]
    if headline is not None:
        hit = compared[(compared["system"] == headline[0])
                       & (compared["rule"] == headline[1])]
        if len(hit):
            row = hit.iloc[0]
            found["headline"] = {
                "system": headline[0], "rule": headline[1],
                "gap_pct": float(row["gap_pct"]),
                "jackknife_se": float(row["jackknife_se"]),
                "bootstrap_se": float(row["bootstrap_se"]),
                "se_ratio": float(row["se_ratio"]),
                "still_signed": bool(row["bootstrap_excludes_zero"]),
            }
    return found


def construction_sweep(run: Callable[[str, Any], pd.DataFrame],
                       arms: Sequence[Tuple[str, str, Any]],
                       column: str = "gap_pct") -> pd.DataFrame:
    """The headline cells re-run under each construction choice.

    Two choices the panel's design makes and the paper has been taking on
    the companion's word: how countries are weighted when a lifetime draws
    one, and how the international sleeve is built.  A headline this paper
    carries should be checked in this paper.
    """
    rows: List[pd.DataFrame] = []
    for key, label, setting in arms:
        block = run(key, setting).copy()
        block.insert(0, "arm", key)
        block.insert(1, "arm_label", label)
        rows.append(block)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


def construction_verdict(frame: pd.DataFrame,
                         baseline_arm: str,
                         column: str = "gap_pct") -> Dict[str, Any]:
    """Whether either construction choice moves a sign or only a level."""
    if not len(frame):
        return {"measured": False}
    base = frame[frame["arm"] == baseline_arm]
    if not len(base):
        return {"measured": False}
    anchor = base.set_index(["system", "rule"])[column].to_dict()
    moves: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        if row["arm"] == baseline_arm:
            continue
        key = (row["system"], row["rule"])
        if key not in anchor:
            continue
        was, now = float(anchor[key]), float(row[column])
        moves.append({"arm": row["arm"], "arm_label": row["arm_label"],
                      "system": row["system"], "rule": row["rule"],
                      "was": was, "now": now, "moved_pp": now - was,
                      "sign_held": bool(np.sign(was) == np.sign(now))})
    if not moves:
        return {"measured": False}
    table = pd.DataFrame.from_records(moves)
    moved = table[~table["sign_held"]]
    return {
        "measured": True,
        "arms": int(table["arm"].nunique()),
        "cells": int(len(table)),
        "every_sign_holds": bool(table["sign_held"].all()),
        "signs_held": int(table["sign_held"].sum()),
        "largest_move_pp": float(table["moved_pp"].abs().max()),
        "median_move_pp": float(table["moved_pp"].abs().median()),
        # The worst cell, in parts as well as joined. The joined form is
        # for a log line; a page wants the arm's label beside a *relabelled*
        # system and rule, and a caller cannot relabel a string that has
        # already been concatenated -- which is how two registry keys
        # reached a printed page.
        "worst": (f"{table.loc[table['moved_pp'].abs().idxmax(), 'arm_label']}"
                  f", {table.loc[table['moved_pp'].abs().idxmax(), 'system']}"
                  f"/{table.loc[table['moved_pp'].abs().idxmax(), 'rule']}"),
        "worst_arm_label": str(
            table.loc[table["moved_pp"].abs().idxmax(), "arm_label"]),
        "worst_system": str(
            table.loc[table["moved_pp"].abs().idxmax(), "system"]),
        "worst_rule": str(
            table.loc[table["moved_pp"].abs().idxmax(), "rule"]),
        # Which cells the choice moves the sign of, as (system, rule)
        # pairs. A construction choice that only flips a cell the paper
        # already declines to sign is a consistency check that passed; one
        # that flips a cell the paper leads with is a finding about the
        # panel's construction. The two read very differently and the
        # caller cannot tell them apart from a count.
        "moved_cells": [(str(r["system"]), str(r["rule"]))
                        for _, r in moved.iterrows()],
        "table": table,
    }


def moved_signs_were_already_unsigned(
        build: Mapping[str, Any],
        jackknife: pd.DataFrame) -> Dict[str, Any]:
    """Whether a construction choice only moves cells nobody was signing.

    The distinction the count cannot make.  If the only sign a weighting
    scheme flips belongs to a cell whose interval already straddles zero,
    the sweep has confirmed the paper's own refusal to sign it rather than
    overturned anything; if it flips a cell the paper leads with, that is a
    result about the panel's construction and belongs in the open.
    """
    cells = list(build.get("moved_cells", ()))
    if not cells:
        return {"measured": True, "moved": 0, "all_already_unsigned": True,
                "signed_cells_moved": []}
    if not len(jackknife):
        return {"measured": False}
    signed = {(str(r["system"]), str(r["rule"]))
              for _, r in jackknife.iterrows()
              if bool(r.get("ci_excludes_zero", False))}
    hit = [c for c in cells if c in signed]
    missed = [c for c in cells if c not in signed]
    return {
        "measured": True,
        "moved": len(cells),
        "all_already_unsigned": not hit,
        "signed_cells_moved": [f"{a}/{b}" for a, b in hit],
        "unsigned_cells_moved": [f"{a}/{b}" for a, b in missed],
    }
