"""The pension gate, held still.

Section #leisure asks which of two features separates Australia's Age
Pension from the American schedule: **how the benefit is worked out** (a
means test rather than a career of earnings) or **when it starts** (a fixed
birthday rather than the day work stops).  It answers by running a 2x2 of
feature arms, solving each one for its own best retirement date, and
comparing the four optima.

That design has a hole in it, and the hole is not visible in the output --
it is visible in the fact that two of the four rows come out identical to
six significant figures.  The joint arm and the formula-only arm both solve
to the pension age.  At that date the eligibility gate is exactly slack: a
household retiring on the birthday the pension arrives on is in the same
position whether or not a gate exists.  The two arms are therefore *the same
simulation*, and their agreement is an identity rather than a finding.  The
interaction term, being the joint arm minus the two singles, is then forced
to equal minus the timing effect and carries no information either.

So the conclusion "it is how the benefit is worked out, not when it starts"
was measured at the one date where when-it-starts cannot matter.

This section measures it where it can.  Two things are swept:

``the date``
    The same four arms are scored at *every* date on the grid, and the 2x2
    is read off at each one.  Holding the date still is what lets the
    timing feature operate: below the pension age the gate binds and the
    two arms separate; at or above it they coincide by construction.  The
    section reports the effect at each date and says which dates could have
    seen the feature at all.

``the bridge``
    A household that stops work before the pension age is not left with
    nothing -- no country arranges that, and a constant-relative-risk-
    aversion objective with a floor at zero is unbounded below, so a model
    that assumed nothing would be reporting the unboundedness rather than
    the pension.  Section #leisure pays a partial benefit over those years.
    The share is a free parameter that no data pins down, it is a
    *consumption floor* in a paper whose whole mechanism is consumption
    floors, and it sits in the arm whose subject is timing.  It is swept
    from nothing to the full rate rather than set.

Neither sweep changes what Section #leisure found about the formula.  What
they change is the standing of what it found about the gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ARMS", "BASELINE", "TIMING", "FORMULA", "BOTH",
    "held_still", "gate_verdict", "bridge_table", "bridge_verdict",
    "COINCIDENCE_TOLERANCE",
]

#: The four feature arms, in the order an ablation reads them. These mirror
#: :data:`src.leisure.FEATURE_ARMS` deliberately: this section re-scores
#: that section's own 2x2 rather than a lookalike of it, so the two can be
#: set side by side.
BASELINE: str = "baseline"
TIMING: str = "timing"
FORMULA: str = "formula"
BOTH: str = "both"
ARMS: Tuple[str, ...] = (BASELINE, TIMING, FORMULA, BOTH)

#: Two arms count as coincident when their certainty equivalents agree to
#: within this, relative. The point of the threshold is not precision: it is
#: that a *difference* of zero to six digits, on a hundred thousand paths
#: with common random numbers, is the signature of an identity rather than
#: of a small effect, and the section should say which it has found.
COINCIDENCE_TOLERANCE: float = 1e-9


def held_still(frame: pd.DataFrame, column: str = "cec") -> pd.DataFrame:
    """The 2x2 read at every fixed retirement date, rather than at each best.

    An ablation compares four arms.  Solving each one for its own optimum
    first and comparing the optima confounds two things: what the feature
    does, and where the feature moves the argmax.  When the joint arm's
    argmax happens to land on the pension age, the gate it carries is slack
    there and the comparison is blind to it.

    Reading the square at a common date is the standard fix, and it costs
    nothing here because the arms are already scored on the same grid.

    :param frame: one row per ``arm`` and ``retire_age``, carrying ``column``.
    :returns: one row per date with the four arms' levels, the two single
        effects against the baseline, the joint effect, and the interaction
        -- plus ``arms_coincide``, true where the formula and joint arms
        agree to :data:`COINCIDENCE_TOLERANCE` and the date can therefore
        say nothing about the gate.
    """
    if not len(frame):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for age, block in frame.groupby("retire_age", sort=True):
        at = {str(r["arm"]): float(r[column]) for _, r in block.iterrows()}
        if any(a not in at for a in ARMS):
            continue
        base = at[BASELINE]
        timing = at[TIMING] - base
        formula = at[FORMULA] - base
        joint = at[BOTH] - base
        rows.append({
            "retire_age": int(age),
            **{f"cec_{a}": at[a] for a in ARMS},
            "timing_effect": timing,
            "formula_effect": formula,
            "joint_effect": joint,
            "interaction": joint - timing - formula,
            # The two facts that decide whether this date is informative.
            "arms_coincide": bool(
                abs(at[BOTH] - at[FORMULA])
                <= COINCIDENCE_TOLERANCE * max(abs(at[FORMULA]), 1.0)),
            "timing_bites": bool(abs(timing) > COINCIDENCE_TOLERANCE
                                 * max(abs(base), 1.0)),
        })
    return pd.DataFrame.from_records(rows)


def gate_verdict(table: pd.DataFrame, gate_age: int,
                 joint_best_age: int | None = None,
                 column: str = "cec") -> Dict[str, Any]:
    """Whether the argmax comparison could have seen the gate at all.

    The claim under audit is Section #leisure's, that the benefit formula
    does the work and the start date does not.  It is worth separating into
    two questions, because only one of them is in doubt:

    * *Is the formula effect large?*  Yes, and nothing here disturbs it --
      it is present at every date on the grid.
    * *Is the timing effect small?*  At the joint arm's own optimum it is
      exactly zero, and it is exactly zero there **whatever the gate does**,
      because the household retires on the pension's own birthday.  That is
      not evidence.

    ``informative_dates`` counts the dates at which the gate binds, and
    ``timing_where_it_bites`` is the largest effect among them.  If that
    number is small the original conclusion stands on its own feet; if it is
    not, the conclusion was an artefact of where the argmax landed.

    :param table: the output of :func:`held_still`.
    :param gate_age: the pension's eligibility age.
    :param joint_best_age: the date the joint arm solves to, when known.
    """
    if not len(table):
        return {"measured": False}
    binds = table[table["retire_age"] < int(gate_age)]
    slack = table[table["retire_age"] >= int(gate_age)]
    found: Dict[str, Any] = {
        "measured": True,
        "gate_age": int(gate_age),
        "dates": int(len(table)),
        "informative_dates": int(len(binds)),
        "slack_dates": int(len(slack)),
    }
    if len(binds):
        widest = binds.loc[binds["timing_effect"].abs().idxmax()]
        found["timing_where_it_bites"] = float(widest["timing_effect"])
        found["timing_widest_age"] = int(widest["retire_age"])
        found["timing_median_where_it_bites"] = float(
            binds["timing_effect"].median())
        found["formula_median_where_it_bites"] = float(
            binds["formula_effect"].median())
        found["interaction_median_where_it_bites"] = float(
            binds["interaction"].median())
        # The comparative claim, made where both features can operate.
        found["formula_dominates_where_it_bites"] = bool(
            abs(found["formula_median_where_it_bites"])
            > abs(found["timing_median_where_it_bites"]))
        if found["timing_median_where_it_bites"]:
            found["formula_over_timing"] = float(
                abs(found["formula_median_where_it_bites"])
                / abs(found["timing_median_where_it_bites"]))
    if len(slack):
        # Everything at or past the gate is an identity, and saying so is
        # the point: an arm that cannot differ is not a measurement.
        found["coincide_past_gate"] = bool(slack["arms_coincide"].all())
        found["timing_max_past_gate"] = float(
            slack["timing_effect"].abs().max())
    if joint_best_age is not None:
        found["joint_best_age"] = int(joint_best_age)
        found["joint_best_is_past_gate"] = bool(
            int(joint_best_age) >= int(gate_age))
        hit = table[table["retire_age"] == int(joint_best_age)]
        if len(hit):
            found["timing_at_joint_best"] = float(
                hit.iloc[0]["timing_effect"])
            found["coincide_at_joint_best"] = bool(
                hit.iloc[0]["arms_coincide"])
    # The headline of the section: the original decomposition was read at a
    # date where the gate is slack, so its timing row was zero by
    # construction rather than by measurement.
    found["argmax_was_blind"] = bool(
        found.get("joint_best_is_past_gate", False)
        and found.get("coincide_at_joint_best", False))
    return found


def bridge_table(frame: pd.DataFrame, column: str = "cec") -> pd.DataFrame:
    """What the pre-eligibility payment does to the timing feature.

    The years between stopping work and the pension age are bridged in this
    model by a partial payment, and the share is a judgement rather than a
    measurement.  It is also the wrong parameter to leave undisclosed: it is
    a floor under consumption in the years the gate creates, so it is
    precisely the thing the timing arm is measuring, and a
    constant-relative-risk-aversion objective is acutely sensitive to floors
    near zero.

    :param frame: one row per ``bridge``, ``arm`` and ``retire_age``.
    :returns: one row per bridge share and date, with the timing effect at
        that share, so the reader can see whether the feature survives the
        parameter or is made of it.
    """
    if not len(frame):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (share, age), block in frame.groupby(["bridge", "retire_age"],
                                             sort=True):
        at = {str(r["arm"]): float(r[column]) for _, r in block.iterrows()}
        if BASELINE not in at or TIMING not in at:
            continue
        row: Dict[str, Any] = {
            "bridge": float(share), "retire_age": int(age),
            "cec_baseline": at[BASELINE], "cec_timing": at[TIMING],
            "timing_effect": at[TIMING] - at[BASELINE],
        }
        if FORMULA in at and BOTH in at:
            row["cec_formula"] = at[FORMULA]
            row["cec_both"] = at[BOTH]
            row["joint_effect"] = at[BOTH] - at[BASELINE]
            row["interaction"] = ((at[BOTH] - at[BASELINE])
                                  - (at[TIMING] - at[BASELINE])
                                  - (at[FORMULA] - at[BASELINE]))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def bridge_verdict(table: pd.DataFrame, gate_age: int,
                   reference_age: int | None = None) -> Dict[str, Any]:
    """Whether the timing conclusion is a finding or a calibration.

    The timing arm carries two things at once, and Section #leisure says so:
    a pension that arrives on a fixed birthday is not reduced for stopping
    work early, whereas an actuarially adjusted one is, so the arm removes
    *a bridge and a penalty together*.  Sweeping the bridge separates them,
    which the argmax comparison could not do:

    ``at a full bridge``
        the gate is switched off -- the household is paid the same rate on
        both sides of the eligibility age -- so whatever effect survives is
        the absent actuarial penalty and nothing else.
    ``at no bridge``
        the arm carries the whole of both.

    The difference between the two ends is the bridge's own contribution,
    and if the two run in opposite directions there is a share at which they
    cancel.  Section #leisure's calibration sits between them, so "the start
    date does not matter" may be reporting that cancellation rather than a
    property of pension design -- which is a question about a parameter no
    data pins down, and therefore one to answer by sweeping it.
    """
    if not len(table):
        return {"measured": False}
    binds = table[table["retire_age"] < int(gate_age)]
    if reference_age is not None:
        binds = binds[binds["retire_age"] == int(reference_age)]
    if not len(binds):
        return {"measured": False}
    shares = sorted(float(x) for x in table["bridge"].unique())
    by_share = binds.groupby("bridge")["timing_effect"].median()
    lowest, highest = float(by_share.index.min()), float(by_share.index.max())
    found: Dict[str, Any] = {
        "measured": True,
        "gate_age": int(gate_age),
        "shares": shares,
        "reference_age": (int(reference_age) if reference_age is not None
                          else None),
        "at_lowest_share": float(by_share.loc[lowest]),
        "at_highest_share": float(by_share.loc[highest]),
        "lowest_share": lowest, "highest_share": highest,
        "span": float(by_share.max() - by_share.min()),
    }
    # Monotone in the bridge is what the mechanism predicts; if it is not,
    # the section has found something it did not go looking for.
    ordered = by_share.reindex(sorted(by_share.index)).to_numpy(dtype=float)
    steps = np.diff(ordered)
    found["monotone"] = bool(np.all(steps >= -1e-12)
                             or np.all(steps <= 1e-12))
    # A full bridge switches the gate off and leaves the absent actuarial
    # penalty behind, so the top of the grid isolates the second half of
    # the feature and the difference isolates the first.
    if abs(highest - 1.0) < 1e-9:
        found["isolates_penalty"] = True
        found["penalty_only"] = found["at_highest_share"]
        found["bridge_only"] = (found["at_lowest_share"]
                                - found["at_highest_share"])
        found["dominant"] = ("bridge"
                             if abs(found["bridge_only"])
                             > abs(found["penalty_only"]) else "penalty")
        # The two pulling opposite ways is what lets a share between them
        # read as "the start date does not matter" when neither piece is
        # small. Linear interpolation on the swept shares, which is enough
        # to say whether such a share is inside the grid at all.
        signs = np.sign([found["at_lowest_share"], found["at_highest_share"]])
        found["ends_disagree"] = bool(signs[0] != signs[1]
                                      and 0.0 not in signs)
        if found["ends_disagree"]:
            grid = np.asarray(sorted(by_share.index), dtype=float)
            values = by_share.reindex(grid).to_numpy(dtype=float)
            crossing = np.flatnonzero(np.sign(values[:-1])
                                      != np.sign(values[1:]))
            if crossing.size:
                i = int(crossing[0])
                lo, hi = values[i], values[i + 1]
                found["cancels_at_share"] = float(
                    grid[i] + (grid[i + 1] - grid[i]) * (-lo) / (hi - lo)
                    if hi != lo else grid[i])
    else:
        found["isolates_penalty"] = False
    return found
