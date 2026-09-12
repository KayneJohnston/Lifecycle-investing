"""Tests for the paper's section numbering.

Numbering used to be written by hand in ninety-odd places, and twice in this
project's history a cross-reference survived a renumber while quietly pointing
at the wrong heading -- a failure the PDF-level reference check cannot see,
because the reference still resolves to *a* section. The `#key` tokens remove
the failure mode by construction; these tests keep it removed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))

content = pytest.importorskip("content")

_RAW = (PAPER / "content.py").read_text()

#: The registry block documents the token syntax with literal ``#key``
#: examples, so the scans below start after it.
SOURCE = _RAW[_RAW.index("return SECTION_TOKEN.sub(swap, text)"):]

#: Adjacent string literals are concatenated by the parser, so a reference
#: split over two source lines is invisible until they are joined.
FLAT = re.sub(r'"\s*\n\s*(f?)"', "", SOURCE)


class TestTheRegistry:
    def test_every_key_is_unique(self) -> None:
        assert len(set(content.SECTION_ORDER)) == len(content.SECTION_ORDER)

    def test_numbers_run_from_one_without_gaps(self) -> None:
        numbers = [content.section_number(k) for k in content.SECTION_ORDER]
        assert numbers == list(range(1, len(content.SECTION_ORDER) + 1))

    def test_an_unknown_key_names_the_valid_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown section"):
            content.section_number("nope")


class TestTokenResolution:
    def test_a_bare_token_becomes_the_section_number(self) -> None:
        n = content.section_number("housing")
        assert content.resolve_sections("Section #housing") == f"Section {n}"

    def test_a_subsection_token_keeps_its_suffix(self) -> None:
        n = content.section_number("leverage")
        assert content.resolve_sections("#leverage.4.1") == f"{n}.4.1"

    def test_a_heading_dot_is_not_swallowed(self) -> None:
        # "#glide. Solving for..." must not read the ". S" as a subsection.
        n = content.section_number("glide")
        assert content.resolve_sections("#glide. Solving") == f"{n}. Solving"

    def test_resolution_is_idempotent_on_resolved_text(self) -> None:
        once = content.resolve_sections("Section #data.6.2 and Section #methods")
        assert content.resolve_sections(once) == once


class TestTheSourceUsesTokensThroughout:
    def test_no_heading_carries_a_hand_written_number(self) -> None:
        literal = re.findall(r'ctx\.h[123]\(\s*f?"(\d)', SOURCE)
        assert not literal, (
            f"{len(literal)} heading(s) still numbered by hand; use a "
            "#key token so the number follows SECTION_ORDER")

    def test_no_cross_reference_carries_a_hand_written_number(self) -> None:
        literal = sorted(set(re.findall(r"Sections? \d[\d.]*", FLAT)))
        assert not literal, (
            f"hand-written cross-references remain: {literal}")

    def test_every_token_names_a_real_section(self) -> None:
        used = set(re.findall(r"#([a-z_]+)", FLAT))
        unknown = used - set(content.SECTION_ORDER)
        assert not unknown, f"tokens with no section: {sorted(unknown)}"

    def test_every_section_is_referred_to_or_at_least_titled(self) -> None:
        used = set(re.findall(r"#([a-z_]+)", FLAT))
        missing = set(content.SECTION_ORDER) - used
        assert not missing, (
            f"sections with no heading token, so they cannot be numbered: "
            f"{sorted(missing)}")


class TestTheOrderGuard:
    @staticmethod
    def _heading(text: str):
        style = type("S", (), {"name": "h1"})()
        return type("P", (), {"text": text, "style": style})()

    def test_a_correct_sequence_passes(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER) + 1)]
        content._check_section_order(parts)          # must not raise

    def test_an_out_of_order_sequence_fails(self) -> None:
        parts = [self._heading("1. A"), self._heading("3. B"),
                 self._heading("2. C")]
        with pytest.raises(AssertionError, match="does not match"):
            content._check_section_order(parts)

    def test_a_missing_section_fails(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER))]
        with pytest.raises(AssertionError, match="does not match"):
            content._check_section_order(parts)

    def test_appendix_headings_are_ignored(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER) + 1)]
        parts.append(self._heading("Appendix A. Model Parameters"))
        content._check_section_order(parts)          # must not raise


class TestReadingOrder:
    """The groupings the roadmap promises the reader."""

    def test_the_headline_is_followed_by_its_robustness_checks(self) -> None:
        for key in ("sensitivity", "sleeve", "hedging"):
            assert content.section_number(key) \
                > content.section_number("baseline")
        # ...and they come before the searches for a better portfolio.
        assert content.section_number("hedging") \
            < content.section_number("glide")

    def test_the_portfolio_searches_relax_constraints_in_order(self) -> None:
        assert (content.section_number("glide")
                < content.section_number("allocation")
                < content.section_number("leverage"))

    def test_the_asset_is_priced_before_it_is_mortgaged(self) -> None:
        assert content.section_number("housing") \
            < content.section_number("mortgage")

    def test_the_non_portfolio_levers_run_in_lifecycle_order(self) -> None:
        # Where you start, what you save, what that responds to, when you
        # stop, how you draw down.
        order = ["valuation", "saving", "accumulation", "retirement",
                 "spending"]
        numbers = [content.section_number(k) for k in order]
        assert numbers == sorted(numbers)

    def test_the_closing_sections_come_last(self) -> None:
        assert list(content.SECTION_ORDER[-3:]) == [
            "discussion", "limitations", "conclusion"]


class TestExtensionGroups:
    """The abstract announces a count and then describes that many studies.

    It has gone stale before. Deriving the counts from the reading order only
    helps if the groups and the order cannot drift apart, which is what these
    check.
    """

    def test_the_groups_partition_the_extensions(self) -> None:
        flat = [key for _, members in content.EXTENSION_GROUPS
                for key in members]
        assert sorted(flat) == sorted(content.EXTENSION_SECTIONS)

    def test_no_study_is_in_two_groups(self) -> None:
        flat = [key for _, members in content.EXTENSION_GROUPS
                for key in members]
        assert len(flat) == len(set(flat))

    def test_each_group_is_contiguous_in_the_reading_order(self) -> None:
        # A group the abstract introduces as "four ask whether..." has to be
        # four consecutive sections, or the prose walks the paper out of
        # order.
        for name, members in content.EXTENSION_GROUPS:
            numbers = [content.section_number(k) for k in members]
            assert numbers == list(range(min(numbers), max(numbers) + 1)), name

    def test_the_groups_run_in_reading_order(self) -> None:
        firsts = [content.section_number(members[0])
                  for _, members in content.EXTENSION_GROUPS]
        assert firsts == sorted(firsts)

    def test_the_count_word_matches_the_membership(self) -> None:
        for name, members in content.EXTENSION_GROUPS:
            word = content.group_count_word(name)
            assert content.NUMBER_WORDS.get(len(members), str(len(members))) \
                == word

    def test_an_unknown_group_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            content.group_count_word("nonexistent")

    def test_every_new_study_has_a_section(self) -> None:
        for key in ("cohorts", "out_of_sample", "human_capital", "mortality"):
            assert key in content.SECTION_ORDER
            assert hasattr(content, f"section_{key}")


class TestNewSectionsAreWiredIn:
    """The two sections added to answer reviewer objections."""

    def test_both_appear_in_the_reading_order(self) -> None:
        assert "pension" in content.SECTION_ORDER
        assert "turnover" in content.SECTION_ORDER

    def test_pension_sits_with_the_robustness_studies(self) -> None:
        """It relaxes a modelling assumption, so it belongs before the
        portfolio movement rather than among the searches."""
        order = list(content.SECTION_ORDER)
        assert order.index("mortality") < order.index("pension")
        assert order.index("pension") < order.index("glide")

    def test_turnover_sits_with_the_other_audit(self) -> None:
        """Costs and unseen data are the same question asked twice."""
        order = list(content.SECTION_ORDER)
        assert order.index("leverage") < order.index("turnover")
        assert order.index("turnover") < order.index("out_of_sample")


class TestSequenceIsWiredIn:
    """The permutation study, and where it has to sit to make sense."""

    def test_it_appears_in_the_reading_order(self) -> None:
        assert "sequence" in content.SECTION_ORDER
        assert hasattr(content, "section_sequence")

    def test_it_follows_the_symptoms_it_explains(self) -> None:
        """It opens by naming the retirement-date result as a symptom.

        Placed before that section the opening sentence refers forward, and
        the reader meets the explanation before the thing being explained.
        """
        order = list(content.SECTION_ORDER)
        assert order.index("retirement") < order.index("sequence")
        assert order.index("inflation") < order.index("sequence")

    def test_it_precedes_the_spending_rules_it_reframes(self) -> None:
        """Its closing claim is about what the rules in #spending trade."""
        order = list(content.SECTION_ORDER)
        assert order.index("sequence") < order.index("spending")

    def test_it_belongs_to_the_plan_group(self) -> None:
        groups = dict(content.EXTENSION_GROUPS)
        assert "sequence" in groups["plan"]

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_the_counts_the_abstract_quotes_are_derived(self) -> None:
        for name, members in content.EXTENSION_GROUPS:
            assert content.group_count_word(name) == \
                content.NUMBER_WORDS[len(members)]
        assert content.extension_count_word() == \
            content.NUMBER_WORDS[len(content.EXTENSION_SECTIONS)]


class TestLimitationsIsNotStale:
    """A Limitations section that contradicts the paper is worse than none.

    Three bullets survived past the sections that answered them, which is the
    single most damaging thing a careful reader can find. These pin the
    wording so it cannot drift back.
    """

    @staticmethod
    def _source() -> str:
        import inspect
        return inspect.getsource(content.section_limitations)

    def test_does_not_claim_the_paper_omits_fees(self) -> None:
        assert "<b>No fees.</b>" not in self._source()

    def test_does_not_claim_mortality_is_deterministic(self) -> None:
        assert "<b>Deterministic mortality.</b>" not in self._source()

    def test_does_not_call_housing_the_single_largest_omission(self) -> None:
        assert "single largest omission" not in self._source()

    def test_names_the_sections_that_answer_each_omission(self) -> None:
        source = self._source()
        for key in ("#fees", "#mortality", "#housing", "#pension",
                    "#cohorts", "#out_of_sample", "#turnover",
                    "#withholding"):
            assert key in source, f"limitations never mentions {key}"


class TestPensionSectionIsDerived:
    """No number in the pension section may be typed rather than computed.

    The section quotes two replacement rates against each other, and the
    comparison only means anything if both come out of the specs the sweep
    actually ran. An earlier draft hardcoded one of them.
    """

    @staticmethod
    def _source() -> str:
        import inspect
        return inspect.getsource(content.section_pension)

    def test_no_hardcoded_replacement_rate(self) -> None:
        source = self._source()
        for literal in ("0.442", "44.2%", "0.293", "29.3%"):
            assert literal not in source, \
                f"{literal!r} is typed into the pension section"

    def test_the_us_rate_is_evaluated_from_the_spec(self) -> None:
        assert "social_security_benefit" in self._source()

    def test_the_australian_rate_comes_from_the_config(self) -> None:
        assert "pension_full_rate" in self._source()


class TestInflationSectionIsWiredIn:
    """The companion to the valuation study, and placed as one."""

    def test_appears_in_the_reading_order(self) -> None:
        assert "inflation" in content.SECTION_ORDER

    def test_sits_beside_the_study_it_mirrors(self) -> None:
        """Both condition a lifetime on a state variable observable at its
        start, so they belong next to each other rather than pages apart."""
        order = list(content.SECTION_ORDER)
        assert order.index("inflation") == order.index("valuation") + 1

    def test_is_counted_among_the_robustness_studies(self) -> None:
        groups = dict(content.EXTENSION_GROUPS)
        assert "inflation" in groups["robustness"]

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_no_hardcoded_correlation_or_gap(self) -> None:
        """Every number in the section comes from the pipeline's own tables."""
        import inspect
        source = inspect.getsource(content.section_inflation)
        for literal in ("0.58", "-5.19", "-2.32", "+3.54", "10%", "100%"):
            assert f'"{literal}"' not in source, \
                f"{literal!r} is typed into the inflation section"


class TestWithholdingSectionIsWiredIn:
    """The concrete instance of the fee experiment, placed beside it."""

    def test_appears_in_the_reading_order(self) -> None:
        assert "withholding" in content.SECTION_ORDER

    def test_follows_the_fee_study_it_sharpens(self) -> None:
        order = list(content.SECTION_ORDER)
        assert order.index("withholding") == order.index("fees") + 1

    def test_is_counted_among_the_robustness_studies(self) -> None:
        assert "withholding" in dict(content.EXTENSION_GROUPS)["robustness"]

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_the_fee_break_even_is_read_not_typed(self) -> None:
        """The section's whole point is a comparison with Section #fees'
        break-even, so that number has to come from the fee module."""
        import inspect
        source = inspect.getsource(content.section_withholding)
        assert "break_even_differential_bp" in source
        for literal in ("114", "115", "29.2", "112"):
            assert f'"{literal}"' not in source


class TestCompactStrategyLabels:
    """The one table wide enough that the configured labels wrap mid-word."""

    def test_every_strategy_that_table_prints_has_a_compact_form(self) -> None:
        """The invariant is about the franking table's own columns.

        A strategy that appears there without a compact form falls back to the
        configured label and wraps mid-word, which is the defect this map
        exists to fix. Strategies that never reach that table need no entry.
        """
        import yaml

        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
        printed = ([str(cfg["franking"]["challenger"])]
                   + [str(r) for r in cfg["franking"]["rivals"]])
        for key in printed:
            assert key in content.COMPACT_STRATEGY, key

    def test_the_compact_forms_fit_a_column_header(self) -> None:
        for key, label in content.COMPACT_STRATEGY.items():
            # Eleven characters wrapped mid-word in the rendered table.
            assert len(label) <= 10, (key, label)

    def test_an_unknown_key_falls_back_rather_than_failing(self) -> None:
        assert content._compact_strategy("solved_schedule") == "solved schedule"


class TestGlideAnchorDescribesItsOwnTable:
    """The subsection whose prose drifted away from its table once already.

    It printed the spending-rule re-solve under a heading, a caption and a
    note that all described an anchored-equity-level sweep the pipeline no
    longer runs -- so the paper contained the result and reported it as
    something else. These pin the prose to the columns it is standing on.
    """

    @staticmethod
    def _source() -> str:
        import inspect
        src = inspect.getsource(content.section_glide)
        return src[src.index("#glide.2"):]

    def test_it_names_the_columns_the_table_carries(self) -> None:
        needed = ("rule", "min_equity_share_at_retirement",
                  "mean_equity_share_elsewhere", "dip_size_pp",
                  "mean_domestic_working", "mean_domestic_retired",
                  "solved_cec")
        src = self._source()
        for column in needed:
            assert column in src, column

    def test_it_does_not_describe_an_anchored_level_sweep(self) -> None:
        """No such sweep exists: `anchor_check` in config.yaml lists rules."""
        import yaml

        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
        entries = cfg["glide_path"]["anchor_check"]["rules"]
        assert all("key" in e for e in entries), entries
        src = self._source().lower()
        for phrase in ("fixed at a series of levels",
                       "anchored equity share",
                       "cost of the anchor"):
            assert phrase not in src, phrase

    def test_the_heading_names_the_finding(self) -> None:
        assert "withdrawal rule" in self._source().split("\n")[0].lower()


class TestMovements:
    """The four movements a reader is told the paper runs in.

    This lived only as a comment above `SECTION_ORDER`, which is to say it
    existed for whoever edits the file and not for whoever reads the paper.
    Now Section 1.3 renders it, so it has to stay true.
    """

    def test_the_movements_partition_the_paper_exactly(self) -> None:
        """A section with no home would vanish from the reader's map while
        still appearing in the paper -- which is how the old hand-written
        roadmap came to describe 21 of 37 sections."""
        flat = [k for _, _, members in content.MOVEMENTS for k in members]
        assert flat == list(content.SECTION_ORDER)

    def test_every_section_appears_once(self) -> None:
        flat = [k for _, _, members in content.MOVEMENTS for k in members]
        assert len(flat) == len(set(flat))

    def test_each_movement_is_a_contiguous_run(self) -> None:
        """The spans are printed as ranges, so a movement whose members are
        scattered would print a range covering sections it does not hold."""
        for name, _, members in content.MOVEMENTS:
            numbers = sorted(content.section_number(k) for k in members)
            assert numbers == list(range(numbers[0], numbers[-1] + 1)), name

    def test_the_span_is_derived_not_typed(self) -> None:
        _, _, members = content.MOVEMENTS[0]
        span = content.movement_span(members)
        first = content.section_number(members[0])
        last = content.section_number(members[-1])
        assert str(first) in span and str(last) in span

    def test_a_one_section_movement_reads_singular(self) -> None:
        assert content.movement_span(("introduction",)) == "Section 1"


class TestThePhraseTheNextSectionIsChecked:
    """"The next section" is a claim about the reading order, not a turn
    of phrase, and the same prose serves two cuts of this study.

    Section `longevity` and Section `ordering` are adjacent in the short
    paper and four apart in the long one, so a sentence calling `ordering`
    "the next section" shipped true in one document and false in the
    other. :func:`content.adjacency` is what the prose goes through now.
    """

    def test_it_says_the_next_section_only_when_it_is_the_next_one(self
                                                                   ) -> None:
        order = list(content.SECTION_ORDER)
        first, second = order[0], order[1]
        assert content.adjacency(first, second) == "the next section"

    def test_it_names_a_section_that_is_further_off(self) -> None:
        order = list(content.SECTION_ORDER)
        assert content.adjacency(order[0], order[3]) == f"Section #{order[3]}"

    def test_a_caller_may_supply_its_own_distant_wording(self) -> None:
        order = list(content.SECTION_ORDER)
        assert content.adjacency(order[0], order[3],
                                 far="the section that uses it") \
            == "the section that uses it"

    def test_a_section_the_cut_does_not_carry_is_not_called_adjacent(self
                                                                     ) -> None:
        """A key resolved through the companion has no number in this
        reading order, so it cannot be the next section here."""
        order = list(content.SECTION_ORDER)
        assert content.adjacency(order[0], "not_a_section_key") \
            == "Section #not_a_section_key"

    def test_the_longevity_pointer_is_right_in_both_cuts(self) -> None:
        """The bug this exists to stop: the long paper called Section 36
        "the next section" from Section 32."""
        near = content.section_number("ordering") \
            == content.section_number("longevity") + 1
        phrase = content.adjacency("longevity", "ordering",
                                   far="the section that uses it")
        assert phrase == ("the next section" if near
                          else "the section that uses it")


class TestReadingOrderDependencies:
    """A section may point forward to say "we return to this", but it must
    not depend on a result the reader has not been given yet."""

    #: Sections whose job is to preview or summarise the whole paper.
    SIGNPOSTS = frozenset({"introduction", "background", "data", "methods",
                           "discussion", "limitations", "conclusion"})

    @staticmethod
    def _bodies() -> dict:
        """Each section's prose, including the private helpers it calls.

        Sliced at the next top-level ``def`` of any kind rather than at the
        next ``def section_``: a private helper written between two sections
        was otherwise counted as part of the one above it, which attributed
        a subsection's cross-references to whichever section happened to
        precede it in the file. Helpers are then folded into every section
        that calls them, so the prose is still checked -- it just gets
        checked against the reader's position in the right section.
        """
        import re
        from pathlib import Path

        src = Path("paper/content.py").read_text()
        spans = [(m.group(1), m.start()) for m in
                 re.finditer(r"\ndef ([A-Za-z_][A-Za-z0-9_]*)\(", src)]
        chunks = {}
        for i, (name, start) in enumerate(spans):
            end = spans[i + 1][1] if i + 1 < len(spans) else len(src)
            chunks[name] = src[start:end]

        helpers = {n: b for n, b in chunks.items()
                   if n.startswith("_") and not n.startswith("__")}
        out = {}
        for key in content.SECTION_ORDER:
            body = chunks.get(f"section_{key}")
            if body is None:
                continue
            # One pass of inlining is enough: the helpers here are leaves or
            # call other helpers the same section also names directly.
            called = [b for n, b in helpers.items()
                      if re.search(rf"\b{re.escape(n)}\s*\(", body)]
            out[key] = body + "".join(called)
        return out

    def test_the_spending_sections_do_not_lean_on_later_ones(self) -> None:
        """`longevity` reads `spending` and `plan`; `leisure` reads
        `longevity`. Ordered the other way -- as they were -- `leisure`
        cited a finding the reader had not reached."""
        import re

        bodies = self._bodies()
        for key in ("plan", "longevity", "leisure", "tax"):
            here = content.section_number(key)
            cited = {r for r in re.findall(r"#([a-z_]+)(?:\.\d+)?",
                                           bodies[key])
                     if r in set(content.SECTION_ORDER) and r != key}
            forward = {r for r in cited
                       if content.section_number(r) > here
                       and r not in self.SIGNPOSTS}
            # `plan` may point on to the institutional pair as signposting,
            # and `longevity` at the section that consumes its pick -- the
            # reason its closing subsection exists is that a later section
            # takes the rule chosen here, so the pointer is the content.
            # Nothing else may reach past itself. This allows signposting
            # and leaning alike; the distinction is not one a regex can
            # draw, so the two allowances are kept narrow and named.
            allowed = {"plan": {"leisure", "tax"},
                       "longevity": {"ordering"}}.get(key, set())
            assert not (forward - allowed), f"{key} depends on {forward}"

    def test_longevity_sits_with_the_spending_sections(self) -> None:
        assert (content.section_number("longevity")
                == content.section_number("plan") + 1)


class TestStoryEmissionOrder:
    """`story()` must call the sections in the order `SECTION_ORDER` numbers
    them.

    `content._check_section_order` already enforces this, but only once the
    whole paper has been assembled -- minutes into a build, after every
    table and figure has been read. Reordering `SECTION_ORDER` without
    reordering the calls is a one-line mistake that deserves a one-second
    failure, so the call order is read straight out of the source.
    """

    @staticmethod
    def _emitted() -> list:
        import re
        from pathlib import Path

        src = Path("paper/content.py").read_text()
        m = re.search(r"\ndef story\(.*?\n(?=\ndef |\Z)", src, re.S)
        assert m, "story() not found"
        called = re.findall(r"parts \+= section_([a-z_]+)\(ctx\)",
                            m.group(0))
        # `references` is emitted as a section but carries no number, so it
        # is not part of the ordering contract.
        known = set(content.SECTION_ORDER)
        return [k for k in called if k in known]

    def test_every_section_is_emitted_exactly_once(self) -> None:
        emitted = self._emitted()
        assert len(emitted) == len(set(emitted))
        assert set(emitted) == set(content.SECTION_ORDER)

    def test_the_call_order_matches_the_numbering(self) -> None:
        assert self._emitted() == list(content.SECTION_ORDER)


class TestRoadmapClaims:
    """Section 1.3 makes claims about where things sit. They have to stay
    true when sections move, and the old roadmap is the reason to check:
    it went stale describing 21 of 37 sections and put `#fees` inside a
    range that ended six sections before it."""

    def test_the_reversing_result_is_where_the_roadmap_says(self) -> None:
        """The roadmap tells a reader the one result that does not survive
        sits at the end of the second movement."""
        name, _, members = content.MOVEMENTS[1]
        assert "pension" in members
        assert content.section_number("pension") == max(
            content.section_number(k) for k in members)

    def test_the_sections_that_explain_it_come_after_it(self) -> None:
        """The roadmap sends the reader on to `longevity` and `leisure` for
        the mechanism, so both must follow the result they explain."""
        anchor = content.section_number("pension")
        assert content.section_number("longevity") > anchor
        assert content.section_number("leisure") > anchor

    def test_the_roadmap_names_them_in_reading_order(self) -> None:
        """Printed the other way round it rendered as "Sections 33 and 32"."""
        assert (content.section_number("longevity")
                < content.section_number("leisure"))


class TestClaimsMatchTheSectionsThatExist:
    """Prose that describes what the paper does, checked against what it
    does. These are the failures a first-time reader actually trips on:
    the Discussion denied four things the paper spends twenty-five pages
    doing, because the sentence predates the sections."""

    @staticmethod
    def _text(name: str) -> str:
        import re
        from pathlib import Path

        src = Path("paper/content.py").read_text()
        m = re.search(rf"\ndef section_{name}\(", src)
        start = m.start()
        nxt = re.search(r"\ndef section_", src[start + 1:])
        end = start + 1 + (nxt.start() if nxt else len(src) - start - 1)
        return src[start:end]

    #: A blanket denial, and the section that makes it false. Checked across
    #: the whole paper rather than one section: the Discussion carried three
    #: of these and the Limitations carried two more, and a test scoped to
    #: the Discussion found only the first three.
    #: The exact unqualified phrasings, not substrings: "No disutility of
    #: labour in the baseline" is a true statement and must keep passing,
    #: while "<b>No disutility of labour.</b>" is the claim that was false.
    DENIALS = (
        ("has no disutility of labour, no taxes", "leisure"),
        ("contains no disutility of labour:", "leisure"),
        ("<b>no disutility of labour.</b>", "leisure"),
        ("no owner-occupied housing and no", "housing"),
        ("nothing here prices an annuity, or a spending rule that adapts",
         "longevity"),
        ("a fixed horizon everywhere but one section", "longevity"),
    )

    def test_no_section_denies_a_capability_the_paper_has(self) -> None:
        from pathlib import Path

        src = Path("paper/content.py").read_text().lower()
        for denial, provider in self.DENIALS:
            assert denial not in src, (
                f"{denial!r} is contradicted by section "
                f"{content.section_number(provider)} ({provider})")

    def test_the_discussion_does_not_deny_its_own_sections(self) -> None:
        body = self._text("discussion").lower()
        for denial, _ in self.DENIALS:
            assert denial not in body, denial

    def test_the_priced_frictions_are_named_where_they_are_priced(self) -> None:
        body = self._text("discussion")
        for key in ("fees", "turnover", "withholding", "tax", "housing",
                    "mortgage", "leisure"):
            assert f"#{key}" in body, key

    def test_no_cross_reference_range_runs_backwards(self) -> None:
        """`§#retirement–§#accumulation` rendered as "§28–§27"."""
        import re
        from pathlib import Path

        src = Path("paper/content.py").read_text()
        bad = []
        for a, b in re.findall(r"#([a-z_]+)\s*[–-]\s*(?:§)?#([a-z_]+)", src):
            known = set(content.SECTION_ORDER)
            if a in known and b in known:
                if content.section_number(b) <= content.section_number(a):
                    bad.append((a, b))
        assert not bad, f"backward ranges: {bad}"


class TestReferencesResolve:
    def test_the_reference_list_carries_no_raw_tokens(self) -> None:
        """One entry cites a section. Rendered without the resolver it
        printed `#accumulation.4` on the page."""
        import re

        from paper.build_paper import Context

        for entry in content.REFERENCES:
            resolved = Context.resolve(entry)
            assert not re.search(r"#[a-z_]+", resolved), entry[:60]

    def test_the_resolver_is_reachable_without_touching_a_private(self) -> None:
        from paper.build_paper import Context

        assert Context.resolve("Section #tax") == "Section 34"


class TestShortPaper:
    """The single-thesis paper cut from the long study.

    It reuses the long paper's section writers, so the two can never
    disagree about a number -- but that reuse is only safe if this module
    and the renderer are looking at the same `content` module, which they
    were not at first.
    """

    def test_it_patches_the_module_the_renderer_actually_uses(self) -> None:
        """`build_paper` puts its own directory on sys.path and does a
        top-level `import content`, so `content` and `paper.content` are two
        objects. Patching the wrong one changed nothing and shipped a paper
        numbered for a different document."""
        import sys

        from paper import short as sh

        assert sh.ct is sys.modules["content"]

    def test_every_inherited_section_exists_in_the_long_order(self) -> None:
        """A typo in an inherited key would silently drop a section."""
        from paper import short as sh

        unknown = [k for k in sh.SHORT_ORDER
                   if k not in content.SECTION_ORDER and k not in sh.OWN]
        assert not unknown

    def test_a_section_it_writes_itself_resolves_to_its_own_number(
            self) -> None:
        """Several keys name a section in both documents -- this paper
        writes its own version. A reference to one must resolve to the
        number it has *here*, never to the companion's."""
        from paper import short as sh

        sh.ct.COMPANION = {"name": "the companion study",
                           "numbers": sh.LONG_NUMBER_ALL}
        try:
            with sh.renumbered():
                for key in sh.OWN:
                    out = sh.ct.resolve_sections(f"Section #{key}")
                    assert out == (
                        f"Section {sh.SHORT_ORDER.index(key) + 1}"), key
                    assert "companion" not in out, key
        finally:
            sh.ct.COMPANION = {}

    def test_a_key_only_this_paper_has_is_not_in_the_long_order(self) -> None:
        """`model` has no companion section, so a reference to it could not
        fall back even if the numbering were wrong."""
        from paper import short as sh

        assert "model" in sh.OWN
        assert "model" not in content.SECTION_ORDER

    def test_the_ordering_section_follows_the_rule_it_depends_on(self
                                                                 ) -> None:
        """It asks what the winning withdrawal rule does to the portfolio
        ranking, so it cannot precede the section that finds that rule."""
        from paper import short as sh

        assert "ordering" in sh.SHORT_ORDER
        assert sh.SHORT_ORDER.index("longevity") < \
            sh.SHORT_ORDER.index("ordering")
        assert content.SECTION_ORDER.index("longevity") < \
            content.SECTION_ORDER.index("ordering")

    def test_every_trimmed_anchor_names_a_trimmed_subsection(self) -> None:
        """`TRIMMED` removes subsections and `TRIMMED_ANCHORS` redirects the
        references into them. The two going out of step is invisible: a
        stale anchor sends a live reference to the companion, and a missing
        one leaves a dangling pointer the build only catches by luck."""
        from paper import short as sh

        keys = {a.split(".")[0] for a in sh.TRIMMED_ANCHORS}
        assert keys == set(sh.TRIMMED), (keys, set(sh.TRIMMED))
        for key, phrases in sh.TRIMMED.items():
            anchors = [a for a in sh.TRIMMED_ANCHORS
                       if a.split(".")[0] == key]
            assert len(anchors) == len(phrases), key

    def test_every_own_section_is_in_the_reading_order(self) -> None:
        from paper import short as sh

        assert set(sh.OWN) <= set(sh.SHORT_ORDER)

    def test_the_model_and_incidence_sections_are_carried(self) -> None:
        """The two additions the referee report asked for."""
        from paper import short as sh

        assert "model" in sh.SHORT_ORDER
        assert "incidence" in sh.SHORT_ORDER
        assert sh.SHORT_ORDER.index("model") < sh.SHORT_ORDER.index("data")

    def test_every_retitled_and_reopened_key_is_inherited(self) -> None:
        """Retitling or reopening a section this paper writes itself would
        do nothing, silently."""
        from paper import short as sh

        inherited = set(sh.SHORT_ORDER) - set(sh.OWN)
        assert set(sh.RETITLED) <= inherited
        assert set(sh.REOPENING) <= inherited
        assert set(sh.DROPPED) <= inherited

    def test_the_argument_sections_are_all_present(self) -> None:
        """The chain the paper's thesis runs along: the pension reverses
        the ordering, one of its two features does the work, a rule can
        restore it, and the last section says which portfolio wins where."""
        from paper import short as sh

        for key in ("pension", "leisure", "longevity", "ordering"):
            assert key in sh.SHORT_ORDER, key

    def test_a_section_it_drops_still_resolves_to_the_companion(self
                                                                ) -> None:
        """The tax section was cut, and the roadmap still points a reader
        at it. That pointer must land in the companion rather than dangle."""
        from paper import short as sh

        assert "tax" not in sh.SHORT_ORDER
        sh.ct.COMPANION = {"name": "the companion study",
                           "numbers": sh.LONG_NUMBER_ALL}
        try:
            with sh.renumbered():
                out = sh.ct.resolve_sections("Section #tax")
        finally:
            sh.ct.COMPANION = {}
        assert out.endswith("of the companion study")

    def test_it_is_materially_shorter(self) -> None:
        from paper import short as sh

        assert len(sh.SHORT_ORDER) <= len(content.SECTION_ORDER) // 3

    def test_renumbering_restores_the_long_order(self) -> None:
        from paper import short as sh

        before = dict(sh.ct._SECTION_NUMBER)
        with sh.renumbered():
            assert sh.ct._SECTION_NUMBER["pension"] == (
                sh.SHORT_ORDER.index("pension") + 1)
        assert sh.ct._SECTION_NUMBER == before

    def test_a_section_it_does_not_carry_points_at_the_companion(self) -> None:
        from paper import short as sh

        sh.ct.COMPANION = {"name": "the companion study",
                           "numbers": sh.LONG_NUMBER_ALL}
        try:
            with sh.renumbered():
                out = sh.ct.resolve_sections("Section #mortality")
        finally:
            sh.ct.COMPANION = {}
        assert out == (f"Section {content.section_number('mortality')} "
                       f"of the companion study")

    def test_the_added_references_cover_the_referee_list(self) -> None:
        from paper import short as sh

        joined = " ".join(sh.EXTRA_REFERENCES)
        for name in ("Cocco", "Gomes", "Viceira", "Campbell", "Benzoni",
                     "Dahlquist", "Hubbard", "Milevsky", "Dimson", "Yaari"):
            assert name in joined, name

    def test_the_calibration_source_is_now_cited(self) -> None:
        """The long paper names Cocco-Gomes-Maenhout as the source of its
        income profile and does not list it."""
        from paper import short as sh

        assert any("Cocco" in r and "Maenhout" in r
                   for r in sh.EXTRA_REFERENCES)


class TestSubsectionRenumbering:
    """Trimming subsections out of an inherited section leaves the survivors
    at their original numbers, so a section whose first heading is 7.5 tells
    the reader four subsections went missing. The remap lives in the
    resolver so the heading and every reference to it move together."""

    def test_a_renumbered_heading_and_its_references_agree(self) -> None:
        from paper import short as sh

        original = dict(sh.ct.RENUMBERED)
        sh.ct.RENUMBERED = {"leisure": {5: 1, 6: 2}}
        try:
            with sh.renumbered():
                n = sh.SHORT_ORDER.index("leisure") + 1
                assert sh.ct.resolve_sections("#leisure.5") == f"{n}.1"
                assert sh.ct.resolve_sections("#leisure.6") == f"{n}.2"
                assert sh.ct.resolve_sections("#leisure") == str(n)
        finally:
            sh.ct.RENUMBERED = original

    def test_an_unmapped_subsection_is_left_alone(self) -> None:
        from paper import short as sh

        original = dict(sh.ct.RENUMBERED)
        sh.ct.RENUMBERED = {"leisure": {5: 1}}
        try:
            with sh.renumbered():
                n = sh.SHORT_ORDER.index("leisure") + 1
                assert sh.ct.resolve_sections("#leisure.9") == f"{n}.9"
        finally:
            sh.ct.RENUMBERED = original

    def test_the_map_is_restored_after_the_story_is_built(self) -> None:
        import content

        assert content.RENUMBERED == {}

    def test_every_renumbered_key_is_an_inherited_section(self) -> None:
        from paper import short as sh

        assert set(sh.RENUMBERED) <= set(sh.SHORT_ORDER) - set(sh.OWN)

    def test_no_renumbered_target_collides_with_a_kept_subsection(self
                                                                  ) -> None:
        """Mapping 5 -> 1 while a real 1 survives would give two headings
        the same number."""
        from paper import short as sh

        for key, moved in sh.RENUMBERED.items():
            trimmed = len(sh.TRIMMED.get(key, ()))
            assert len(set(moved.values())) == len(moved), key
            assert max(moved.values()) <= trimmed + len(moved), key


class TestTheJackknifeDiagnosticIsCarried:
    """The interval on the contested cell reads as imprecision. The
    pseudo-values say something more specific -- that the average
    fifteen-country panel does not reproduce the sign -- and that sentence
    is the one a referee asked for, so it should not be able to fall out of
    the paper silently."""

    def test_the_short_paper_reads_the_pseudo_values(self) -> None:
        source = (PAPER / "short.py").read_text()
        assert "ordering_pseudo" in source

    def test_it_reports_the_count_and_the_bias(self) -> None:
        """Both, because either alone is easy to wave away: a lopsided
        count could be chance, and a bias without the count is a number
        with no intuition attached."""
        source = re.sub(r'"\s*\n\s*(f?)"', "",
                        (PAPER / "short.py").read_text())
        assert "below_point" in source
        assert "bias_estimate" in source

    def test_it_says_the_other_cells_are_well_behaved(self) -> None:
        """The claim is comparative. Without the contrast the diagnostic
        reads as a caveat about jackknives rather than about these cells.

        The wording is not pinned -- an earlier version of this test froze
        a sentence and failed when the section was rewritten to cover two
        reversals rather than one. What has to survive is the contrast
        itself: the section must read the cells that do *not* reverse and
        report how they behave.
        """
        source = re.sub(r'"\s*\n\s*(f?)"', "",
                        (PAPER / "short.py").read_text())
        body = source[source.index("def _both_reversals("):
                      source.index("def _sign_split(")]
        assert "does not reverse" in body, (
            "the diagnostic no longer contrasts the reversing cells with "
            "the rest of the table")
        assert "below_point" in body and "bias_estimate" in body
        # And on the same two statistics, not a different pair.
        assert body.count("bias_estimate") >= 2

    def test_the_verdict_names_the_contested_cell(self) -> None:
        """`bias_verdict` is only meaningful on the cell whose sign is in
        dispute; run on a well-behaved one it should report nothing
        unusual."""
        import pandas as pd

        from src import ordering as odr

        table = pd.DataFrame.from_records([
            {"system": "au", "rule": "fixed", "point": -2.0, "deletions": 16,
             "loo_mean": 0.04, "loo_sd": 2.8, "below_point": 4,
             "bias_estimate": 31.8, "bias_over_point": 15.3},
            {"system": "au", "rule": "amort", "point": 26.6, "deletions": 16,
             "loo_mean": 26.4, "loo_sd": 1.1, "below_point": 7,
             "bias_estimate": -3.3, "bias_over_point": 0.12},
        ])
        assert odr.bias_verdict(table, "fixed", "au")["isolated"]
        assert not odr.bias_verdict(table, "amort", "au")["isolated"]


class TestTheObjectiveIsNotSwitchedSilently:
    """Section 9 rejects a fixed retirement horizon as not neutral between
    withdrawal rules and re-solves against a survival curve. Section 10
    then scored its whole grid on the horizon Section 9 rejected, and said
    nothing about it -- so a reader met a section that opened "the previous
    section found the withdrawal rule a retiree should use" and had no way
    to know the objective had changed underneath them."""

    def test_the_grid_is_scored_on_both_objectives(self) -> None:
        source = open("main.py").read()
        step = source[source.index("def step36_ordering"):
                      source.index("def step37_ceiling")]
        assert "cec_survival" in step
        assert "mrt.certainty_equivalent(" in step

    def test_the_paper_reports_what_the_second_objective_does(self) -> None:
        source = (PAPER / "short.py").read_text()
        assert "ordering_by_objective" in source

    def test_the_paper_says_which_objective_its_tables_use(self) -> None:
        """Reporting the comparison is not enough if the reader cannot
        tell which of the two the headline tables are scored on."""
        flat = re.sub(r'"\s*\n\s*(f?)"', "",
                      (PAPER / "short.py").read_text())
        assert "scored twice" in flat

    def test_the_insulation_claim_is_measured_not_asserted(self) -> None:
        """`objective_verdict` has to be able to come back negative, or
        the sentence it supports is decoration."""
        import pandas as pd

        from src import ordering as odr

        swept = pd.DataFrame.from_records([
            {"system": "au", "rule": "r", "strategy": s,
             "cec": c, "cec_survival": v}
            for s, c, v in [("balanced_all_equity", 1.05, 0.95),
                            ("target_date_fund", 1.00, 1.00)]])
        found = odr.objective_verdict(odr.by_objective(swept), "r", "au")
        assert found["measured"] and not found["insulated"]


class TestTheContributionMarginNamesItsWindow:
    """Every certainty equivalent in this paper is over the retirement
    window, which is right for a comparison between portfolios and makes
    the 2x2's *contribution* margin a gross quantity.

    Doubling the saving rate is forty years of consumption the household
    did not have, and on the retirement window none of that is in the
    number. The paper prices it -- Section 8's incidence dial reports the
    lifetime cost -- but Section 10.1 reported the margin without saying
    which measure it was on, which is where a reader would take it for a
    welfare gain.
    """

    def test_the_window_is_what_the_pipeline_actually_uses(self) -> None:
        """The premise. If this ever changed to the full lifetime the
        paragraph below would be wrong rather than merely unnecessary."""
        import yaml

        cfg = yaml.safe_load(open("config.yaml"))
        assert cfg["utility"]["consumption_window"] == "retirement"

    def test_the_factorial_says_which_measure_its_margin_is_on(self
                                                               ) -> None:
        source = (PAPER / "short.py").read_text()
        block = source[source.index("def _contribution_has_a_cost_side"):
                       source.index("def _matched_factorial")]
        assert "retirement window" in block
        assert "#incidence" in block

    def test_it_is_printed_beside_the_factorial_and_not_orphaned(self
                                                                 ) -> None:
        source = (PAPER / "short.py").read_text()
        block = source[source.index("def _matched_factorial"):]
        assert "_contribution_has_a_cost_side(ctx, f)" in block

    def test_the_lifetime_cost_is_read_and_not_asserted(self) -> None:
        """The two numbers the paragraph quotes come off the incidence
        sweep, so they cannot drift from Section 8's own."""
        source = (PAPER / "short.py").read_text()
        block = source[source.index("def _contribution_has_a_cost_side"):
                       source.index("def _matched_factorial")]
        assert 'f.table("incidence_optimum")' in block
        assert "cec_lifetime" in block
        assert "mean_working_consumption" in block

    def test_the_ordering_table_note_names_the_window_in_both_cuts(self
                                                                   ) -> None:
        """A reader meets the numbers at the table, not at the paragraph
        three pages on, and the long paper has the table without the
        paragraph."""
        for name in ("short.py", "content.py"):
            source = (PAPER / name).read_text()
            head = source.index(
                "The all-equity portfolio's lead over the target-date "
                "fund, in ")
            note = source[head:head + 1800]
            assert "retirement window" in note, name


class TestRuinIsReportedOnBothMeasures:
    """Section 9 objects to counting a portfolio exhausted at ninety-one as
    a failed retirement for a household that most likely died before then.
    Section 10 then established dominance in exactly that measure, and the
    corrected one sat unread in a published table."""

    def test_the_paper_reads_the_survival_weighted_ruin(self) -> None:
        assert "prob_ruin_survival" in (PAPER / "short.py").read_text()

    def test_the_pipeline_computes_it(self) -> None:
        source = open("main.py").read()
        step = source[source.index("def step36_ordering"):
                      source.index("def step37_ceiling")]
        assert "mrt.probability_of_ruin(" in step

    def test_no_results_column_goes_unread(self) -> None:
        """The wart this fixed: a column shipped in the archive that
        nothing explains. Every column the ordering sweep writes should be
        read by the paper, a verdict, or a generated document."""
        import pandas as pd

        path = (PAPER.parent / "results" / "tables" / "ordering_sweep.csv")
        if not path.exists():
            pytest.skip("the ordering sweep has not been run")
        readers = ((PAPER / "short.py").read_text()
                   + (PAPER / "content.py").read_text()
                   + open("main.py").read()
                   + (PAPER.parent / "src" / "report.py").read_text()
                   + (PAPER.parent / "src" / "ordering.py").read_text())
        columns = list(pd.read_csv(path, nrows=1).columns)

        def cited(name: str) -> bool:
            """Whether the source refers to this column *as a column*.

            A bare substring search is too weak to be a guard: `cec`
            occurs inside `cec_survival`, `prob_ruin` inside
            `prob_ruin_survival`, and a short name would pass whatever
            happened to it. Columns are addressed as quoted strings --
            `row["cec"]`, `"cec": lambda v: ...` -- so the quotes are what
            distinguishes a reference from a coincidence. A name built by
            f-string over a grid (`cec_gamma2`) is cited by its stem.
            """
            for candidate in (name, re.sub(r"[\d.]+$", "", name)):
                if not candidate:
                    continue
                if re.search(rf"""['"]{re.escape(candidate)}['"]""", readers):
                    return True
                # `f"cec_gamma{g:g}"` -- the stem is followed by a brace.
                if re.search(rf"""['"]{re.escape(candidate)}\{{""", readers):
                    return True
            return False

        unread = [c for c in columns if not cited(c)]
        assert not unread, unread

    def test_the_unread_check_can_actually_fail(self) -> None:
        """Guards the guard. A column nothing mentions must be caught, or
        the check above is decoration."""
        readers = 'row["cec"] and frame["prob_ruin"]'
        pattern = r"""['"]{}['"]"""
        assert re.search(pattern.format("cec"), readers)
        assert not re.search(pattern.format("nobody_reads_this"), readers)


class TestFloatNumbering:
    """Figure and table numbers are issued by a counter on the context and
    the short paper trims whole subsections *after* that counter has run, so
    the numbering used to arrive with holes in it: Figure 1 then Figure 5,
    Table 3 then Table 8 then Table 23. The renumbering pass closes them on
    the assembled story, where what survived is known."""

    @staticmethod
    def _story(numbers):
        """A story of caption heads at the given numbers, half of them
        nested inside a ``KeepTogether`` the way the real builder nests them.
        """
        import build_paper as bp
        from reportlab.platypus import KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        head = ParagraphStyle("caption_head")
        out = []
        for i, (kind, n) in enumerate(numbers):
            caption = Paragraph(f"{kind} {n}. A caption", head)
            out.append(KeepTogether([caption]) if i % 2 else caption)
        return out, bp

    def test_the_two_walkers_stay_distinct(self) -> None:
        """`_walk` flattens the story for reading and `_slots` yields
        mutable positions for rewriting. A second `def _walk` shadowed the
        first, and the only symptom was the missing-glyph check quietly
        matching nothing -- it filters for Paragraphs, and the generator
        hands back tuples."""
        import build_paper as bp
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import KeepTogether, Paragraph

        inner = Paragraph("leaf", ParagraphStyle("body"))
        story = [KeepTogether([inner])]
        assert inner in bp._walk(story)
        assert list(bp._slots(story)) == [(story, 0), ([inner], 0)]

    def test_gaps_are_closed_in_document_order(self) -> None:
        story, bp = self._story([("Figure", 1), ("Figure", 5), ("Figure", 6)])
        mapping = bp.renumber_floats(story)
        assert mapping["Figure"] == {1: 1, 5: 2, 6: 3}

    def test_the_printed_captions_are_rewritten(self) -> None:
        story, bp = self._story([("Table", 3), ("Table", 8), ("Table", 23)])
        bp.renumber_floats(story)
        printed = [fl.getPlainText() if hasattr(fl, "getPlainText")
                   else fl._content[0].getPlainText() for fl in story]
        assert [p.split(".")[0] for p in printed] == [
            "Table 1", "Table 2", "Table 3"]

    def test_figures_and_tables_are_counted_separately(self) -> None:
        story, bp = self._story([("Figure", 2), ("Table", 7), ("Figure", 9)])
        mapping = bp.renumber_floats(story)
        assert mapping == {"Figure": {2: 1, 9: 2}, "Table": {7: 1}}

    def test_a_document_with_nothing_trimmed_is_left_alone(self) -> None:
        story, bp = self._story([("Figure", 1), ("Figure", 2)])
        before = [id(fl) for fl in story]
        bp.renumber_floats(story)
        assert [id(fl) for fl in story] == before

    def test_a_named_float_resolves_to_its_final_number(self) -> None:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        story, bp = self._story([("Table", 3), ("Table", 8)])
        story.append(Paragraph("see @table:fidelity for the check",
                               ParagraphStyle("body")))
        bp.renumber_floats(story, {"fidelity": ("Table", 8)})
        assert story[-1].getPlainText() == "see Table 2 for the check"

    def test_a_resolved_anchor_is_not_remapped_a_second_time(self) -> None:
        """An anchor resolves to the number the document will print. If
        the literal-number pass runs afterwards it sees that final number
        as an old one and remaps it again -- which moved a reference from
        Table 16 to Table 11, two unrelated tables, and did so silently
        because both numbers exist."""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        # Old 30 prints as 2; old 2 prints as 1. An anchor on old 30 must
        # come out as "Table 2" and must not then be remapped to "Table 1".
        story, bp = self._story([("Table", 2), ("Table", 30)])
        story.append(Paragraph("see @table:late", ParagraphStyle("body")))
        bp.renumber_floats(story, {"late": ("Table", 30)})
        assert story[-1].getPlainText() == "see Table 2"

    def test_a_literal_number_beside_an_anchor_still_remaps(self) -> None:
        """The two passes must both still happen, in the right order."""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        story, bp = self._story([("Table", 2), ("Table", 30)])
        story.append(Paragraph("Table 30 and @table:early",
                               ParagraphStyle("body")))
        bp.renumber_floats(story, {"early": ("Table", 2)})
        assert story[-1].getPlainText() == "Table 2 and Table 1"

    def test_a_name_no_float_claims_stops_the_build(self) -> None:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        story, bp = self._story([("Table", 3)])
        story.append(Paragraph("see @table:missing", ParagraphStyle("body")))
        with pytest.raises(SystemExit, match="does not print"):
            bp.renumber_floats(story, {})

    def test_a_name_on_a_trimmed_float_stops_the_build(self) -> None:
        """The anchor was registered, but the float it named was trimmed out
        of this paper -- so the reference has nothing to point at."""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        story, bp = self._story([("Table", 3)])
        story.append(Paragraph("see @table:gone", ParagraphStyle("body")))
        with pytest.raises(SystemExit, match="does not print"):
            bp.renumber_floats(story, {"gone": ("Table", 8)})

    def test_two_floats_cannot_share_a_name(self) -> None:
        import build_paper as bp

        ctx = bp.Context.__new__(bp.Context)
        ctx.float_anchors = {}
        ctx._anchor("dup", "Table", 1)
        with pytest.raises(SystemExit, match="claimed twice"):
            ctx._anchor("dup", "Figure", 2)

    def test_an_unnamed_float_registers_nothing(self) -> None:
        import build_paper as bp

        ctx = bp.Context.__new__(bp.Context)
        ctx.float_anchors = {}
        ctx._anchor("", "Table", 1)
        assert ctx.float_anchors == {}


class TestNoFloatIsCitedByANumber:
    """A literal "Table 4" in the prose is a pointer nothing checks. One had
    been pointing at the wrong table since the numbering last moved, in both
    papers at once. Named anchors are checked; bare numbers are not, so the
    prose is not allowed to contain them."""

    #: The caption heads the builder writes are the legitimate occurrences,
    #: and they are built from the counter rather than typed.
    CITATION = re.compile(r'"[^"]*\b(?:Figure|Table) \d+\b')

    #: A float number built by arithmetic on the live counter, which is how
    #: one wrong reference survived the switch to named anchors: the number
    #: never appears as a digit in the source, so a scan for digits cannot
    #: see it. ``ctx._table_no + 2`` was off by one, in both papers.
    COMPUTED = re.compile(r"(?:Figure|Table) \{[^}]*_(?:table|figure)_no")

    def test_no_float_number_is_computed_from_the_counter(self) -> None:
        for name in ("content.py", "short.py"):
            source = (PAPER / name).read_text()
            found = self.COMPUTED.findall(re.sub(r'"\s*\n\s*(f?)"', "",
                                                 source))
            assert not found, (name, found)

    def test_the_shared_sections_cite_no_float_by_number(self) -> None:
        found = self.CITATION.findall(FLAT)
        assert not found, found

    def test_the_short_paper_cites_no_float_by_number(self) -> None:
        source = (PAPER / "short.py").read_text()
        flat = re.sub(r'"\s*\n\s*(f?)"', "", source)
        found = self.CITATION.findall(flat)
        assert not found, found

    def test_every_anchor_the_prose_names_is_registered_somewhere(self
                                                                  ) -> None:
        """The reference and the ``anchor=`` that satisfies it are written in
        two different places, and a build only catches the mismatch for the
        paper it builds."""
        import build_paper as bp

        source = _RAW + (PAPER / "short.py").read_text()
        named = {m.group(2)
                 for m in bp._FLOAT_ANCHOR_REFERENCE.finditer(source)}
        declared = set(re.findall(r'anchor="([a-z0-9_]+)"', source))
        assert named <= declared, sorted(named - declared)


class TestTheProseUsesOneDash:
    """Both documents set an em dash 190-odd times and set `--` thirteen
    times, all of them in prose written as a docstring-style ASCII dash and
    never converted. It is the same mark doing the same job, printed two
    ways on facing pages.

    Source comments and docstrings keep `--`; only what reaches a page is
    checked, so this reads the built documents.
    """

    def test_neither_document_prints_a_double_hyphen_dash(self) -> None:
        import re

        from pypdf import PdfReader

        pattern = re.compile(r"\s--\s")
        for name in ("floor_beneath_the_portfolio.pdf",
                     "lifecycle_asset_allocation.pdf"):
            path = PAPER / name
            if not path.exists():
                continue
            text = "\n".join((page.extract_text() or "")
                              for page in PdfReader(str(path)).pages)
            flat = re.sub(r"\s+", " ", text)
            found = [flat[max(0, m.start() - 60):m.end() + 40]
                     for m in pattern.finditer(flat)]
            assert not found, f"{name}: " + "\n".join(found)


class TestAStrategyIsNamedForProse:
    """A table label is a heading -- it opens with a capital and takes no
    article -- and three sentences in Section 6 dropped one into running
    text: "the best strategy becomes Target-date fund", "with 100% bills
    (cash) taking first place"."""

    def test_a_label_becomes_a_noun_phrase(self) -> None:
        assert content.strategy_in_prose("target_date_fund") \
            == "the target-date fund"

    def test_a_label_that_reads_badly_with_an_article_is_given_a_name(
            self) -> None:
        assert content.strategy_in_prose("bills_only") == "cash"

    def test_an_unmapped_strategy_still_gets_an_article(self) -> None:
        assert content.strategy_in_prose("some_new_strategy") \
            == "the some new strategy"

    def test_the_table_label_is_left_alone(self) -> None:
        """The prose form is additional to the heading form, not a
        replacement: a column still wants the capital and no article."""
        assert content._pretty_strategy("target_date_fund") \
            == "Target-date fund"


class TestNoSentenceOpensInLowerCase:
    """Rule labels, country names and spelled counts are common nouns and
    numbers, so they are right in lower case inside a table and wrong at
    the start of a sentence.

    Three sentences in each built document opened with one -- "amortisation
    (6% assumed return) wants 6%", "constant percent wants 8.0%", "seven of
    eight rules in the menu return the lead" -- because the label was
    interpolated straight after a full stop. :func:`content.opens` is what
    those go through now, and this reads the built documents rather than
    the source, because the leak was never in one place.

    Table cells flatten into the same text stream and legitimately carry
    lower-case fragments after an abbreviated header ("Dom. equity", "Geo.
    mean", "S.d. annualised"), as does an abbreviation in prose ("i.i.d.
    returns", "p.a."). Those are matched on the token before the stop and
    skipped, so what is left is prose.
    """

    #: The token before the full stop, where a lower-case word after it is
    #: not a sentence opening: an abbreviation, or a table column header
    #: that the PDF text layer runs into the next cell.
    NOT_A_SENTENCE_END = frozenset({
        "i.i.d", "e.g", "i.e", "cf", "p.a", "s.d", "dom", "intl", "geo",
        "excl", "eq", "no", "vs", "approx", "est", "pct", "ann", "corr",
        "cons", "med", "avg", "min", "max", "yr", "yrs", "fig", "tbl",
        "unle", "leverag",
    })

    def test_neither_document_opens_a_sentence_in_lower_case(self) -> None:
        import re

        from pypdf import PdfReader

        for name in ("floor_beneath_the_portfolio.pdf",
                     "lifecycle_asset_allocation.pdf"):
            path = PAPER / name
            if not path.exists():
                continue
            text = "\n".join((page.extract_text() or "")
                              for page in PdfReader(str(path)).pages)
            flat = re.sub(r"\s+", " ", text)
            bad = []
            for m in re.finditer(
                    r"([A-Za-z.]+)\.\s+([a-z][a-z'\-]+)", flat):
                if m.group(1).lower().strip(".") in self.NOT_A_SENTENCE_END:
                    continue
                bad.append(f"...{flat[max(0, m.start() - 60):m.end()]}")
            assert not bad, f"{name}: " + "\n".join(bad)


class TestRuleLabels:
    """Withdrawal rules reach the results tables under the key the pipeline
    runs them by, and three of them are Python identifiers. They were
    printing as identifiers in table columns, in a parameter appendix and
    once in the middle of a sentence, in both papers."""

    def test_the_baseline_rule_is_written_for_a_reader(self) -> None:
        assert content.rule_label("fixed_real_rule") == "fixed real"

    def test_a_rate_survives_the_relabelling(self) -> None:
        assert content.rule_label("constant_percent at 4%") == \
            "percentage of balance at 4%"

    def test_a_label_that_is_already_prose_is_left_alone(self) -> None:
        assert content.rule_label("amortisation at 6%") == "amortisation at 6%"

    def test_a_plan_label_keeps_its_retirement_age(self) -> None:
        """`plan.Plan.label` appends the retirement age to the rule key, so
        the relabelling has to be a prefix swap and not a lookup."""
        assert content.rule_label("constant_percent at 7.0%, retire at 63") \
            == "percentage of balance at 7.0%, retire at 63"

    def test_an_unknown_rule_loses_its_underscores(self) -> None:
        assert content.rule_label("some_new_rule") == "some new rule"

    def test_one_rule_under_two_keys_gets_one_name(self) -> None:
        """`spending.from_spec` maps `fixed_real_rule` onto the same
        `ConstantRealRule` the spending module keys as `constant_real`, so
        a document calling one "fixed real" and the other "constant real"
        is giving one policy two names. Both papers did."""
        from src import spending as spg

        assert isinstance(spg.from_spec("fixed_real_rule", 0.04),
                          spg.REGISTRY["constant_real"])
        assert content.rule_label("constant_real") \
            == content.rule_label("fixed_real_rule")

    def test_both_papers_print_no_rule_key(self) -> None:
        """The check that matters: no identifier reaches a page. Read off
        the built documents, because the leak was never in one place."""
        import re

        from pypdf import PdfReader

        keys = sorted(content.RULE_LABELS)
        pattern = re.compile(r"\b(" + "|".join(keys) + r")\b")
        for name in ("floor_beneath_the_portfolio.pdf",
                     "lifecycle_asset_allocation.pdf"):
            path = PAPER / name
            if not path.exists():
                pytest.skip(f"{name} has not been built")
            text = "\n".join((page.extract_text() or "")
                             for page in PdfReader(str(path)).pages)
            # "amortisation" is a word as well as a key, so only the keys
            # that are not English are a leak.
            found = {m.group(0) for m in pattern.finditer(text)
                     if "_" in m.group(0)}
            assert not found, (name, sorted(found))


class TestTheReadmeIndexesEveryDocument:
    """The README's table is the repository's index. Four sections were
    added without rows, and the sentence under the table still said
    "thirty-two" -- so the two newest studies, which the paper leans on,
    were invisible to anyone reading the front page."""

    ROOT = PAPER.parent

    @classmethod
    def _readme(cls) -> str:
        return (cls.ROOT / "README.md").read_text()

    @classmethod
    def _documents(cls) -> list:
        return sorted(p.name for p in (cls.ROOT / "docs").glob("*.md"))

    def test_every_document_has_a_row(self) -> None:
        import re

        linked = set(re.findall(r"\(docs/([0-9]+_[a-z_]+\.md)\)",
                                self._readme()))
        missing = sorted(set(self._documents()) - linked)
        assert not missing, missing

    def test_no_row_points_at_a_document_that_is_gone(self) -> None:
        import re

        linked = set(re.findall(r"\(docs/([0-9]+_[a-z_]+\.md)\)",
                                self._readme()))
        assert not sorted(linked - set(self._documents()))

    def test_the_layout_block_counts_the_documents(self) -> None:
        import re

        claimed = re.search(r"generated analysis documents \((\d+) files\)",
                            self._readme())
        assert claimed, "the layout block has been reworded"
        assert int(claimed.group(1)) == len(self._documents())

    def test_the_newest_step_has_a_worked_example(self) -> None:
        """The quick-start block is a selection, not a catalogue, so most
        steps need no line. The newest one does: four studies were added
        without one, and the list stopped advertising the work that the
        paper's last two sections rest on."""
        import re
        import sys

        sys.path.insert(0, str(self.ROOT))
        main = pytest.importorskip("main")
        listed = {int(n) for line in self._readme().split("\n")
                  if line.startswith("python main.py --steps ")
                  for n in re.findall(r"\d+", line.split("#")[0])}
        assert max(main.STEPS) in listed, sorted(listed)[-4:]

    def test_the_advertised_test_count_is_the_real_one(self) -> None:
        """Quoted in the quick-start block, where a reader checks their
        checkout is complete. It had been stale by eight hundred."""
        import re
        import subprocess

        claimed = re.search(r"pytest tests/ -q\s+# ([\d,]+) tests",
                            self._readme())
        assert claimed, "the quick-start block has been reworded"
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q",
             "--collect-only", "-p", "no:cacheprovider"],
            cwd=self.ROOT, capture_output=True, text=True)
        found = re.search(r"(\d+) tests collected", run.stdout)
        assert found, run.stdout[-400:]
        assert int(claimed.group(1).replace(",", "")) == int(found.group(1))

    def test_the_count_under_the_table_is_right(self) -> None:
        """A number written as a word, so it cannot be updated by the
        pipeline and has to be checked."""
        import re

        words = {30: "thirty", 31: "thirty-one", 32: "thirty-two",
                 33: "thirty-three", 34: "thirty-four", 35: "thirty-five",
                 36: "thirty-six", 37: "thirty-seven", 38: "thirty-eight",
                 39: "thirty-nine", 40: "forty"}
        n = len(self._documents())
        claimed = re.search(r"All ([a-z-]+) are \*\*generated\*\*",
                            self._readme())
        assert claimed, "the sentence under the table has been reworded"
        assert claimed.group(1) == words[n], (claimed.group(1), n)


class TestTheRuinConventionIsDeclaredWhereItIsUsed:
    """Section 5 leads on a fixed-horizon ruin probability; Section 9 argues
    the measure overstates the level. Both are right -- the comparison in
    Section 5 is between two portfolios measured alike, and the number is
    the replicated study's convention -- but a reader who meets Section 9
    first has to reconcile them unaided. The caveat does the reconciling,
    and takes the size of the distortion from Section 9's own grid rather
    than describing it."""

    def test_the_baseline_section_carries_the_caveat(self) -> None:
        source = (PAPER / "content.py").read_text()
        body = source[source.index("def section_baseline("):
                      source.index("def section_pension(")]
        assert "_ruin_convention(ctx, f)" in body, (
            "section 5 quotes a fixed-horizon ruin probability with no "
            "pointer to the section that revisits the measure")

    def test_the_caveat_points_at_the_section_that_revisits_it(self) -> None:
        # Adjacent literals are joined first: `"Section "` and
        # `"#longevity keeps"` sit on separate source lines, and an
        # unjoined scan would read the second as a bare reference.
        body = self._prose()
        assert "Section #longevity" in body
        # A bare `#longevity` resolves to a naked numeral mid-sentence,
        # which reads as a typo -- the first draft of this caveat printed
        # "every section but 9 keeps". Every reference is spelled out.
        assert "#longevity" not in body.replace("Section #longevity", "")

    @staticmethod
    def _prose() -> str:
        """The helper's f-strings with adjacent literals concatenated and
        the docstring dropped, which is what the reader actually sees."""
        source = (PAPER / "content.py").read_text()
        body = source[source.index("def _ruin_convention("):
                      source.index("def section_baseline(")]
        body = body.split('"""', 2)[-1]
        return re.sub(r'"\s*\n\s*(f?)"', "", body)

    def test_it_reads_the_size_rather_than_asserting_it(self) -> None:
        source = (PAPER / "content.py").read_text()
        body = source[source.index("def _ruin_convention("):
                      source.index("def section_baseline(")]
        assert "lng.ruin_overstatement(" in body
        assert 'f.table("longevity_sweep")' in body

    def test_the_caveat_does_not_disown_the_comparison(self) -> None:
        """The referee's point was that the number is sound and the
        measure is the question. A caveat that retracted the comparison
        would be the wrong fix."""
        source = (PAPER / "content.py").read_text()
        body = source[source.index("def _ruin_convention("):
                      source.index("def section_baseline(")]
        assert "difference" in body and "measured the same way" in body

    def test_the_built_paper_prints_it(self) -> None:
        pdf = PAPER / "floor_beneath_the_portfolio.pdf"
        reader = pytest.importorskip("pypdf")
        if not pdf.exists():
            pytest.skip("the short paper has not been built")
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.PdfReader(str(pdf)).pages)
        assert "One caveat on the measure, not the comparison" in text
        # And the forward pointer resolved to a number, not to the token.
        assert "#longevity" not in text


class TestTheTwoReversalsAreTreatedAlike:
    """Holding the contribution rate still added a second negative cell,
    and it is the one the paper now leads with. Several passages went on
    describing a single contested cell, and one of them said the reversal
    could not be separated from zero eight pages after the paper had
    separated it. These check the asymmetry has not come back."""

    @staticmethod
    def _gaps():
        import pandas as pd

        path = (PAPER.parent / "results" / "tables" / "ordering_gaps.csv")
        if not path.exists():
            pytest.skip("the ordering study has not been run")
        return pd.read_csv(path)

    def test_the_grid_still_has_two_reversals(self) -> None:
        """The premise. If a rerun leaves one, the prose below is wrong in
        the other direction and this is where that surfaces."""
        gaps = self._gaps()
        assert int((gaps["gap_pct"] < 0).sum()) == 2, (
            "the number of negative cells has changed; every sentence "
            "counting them is now suspect")

    def test_the_negative_count_is_counted_and_not_typed(self) -> None:
        """A draft said 'the only negative entry in 40 cells' while two
        cells were negative: the 40 was generated and the 'only' was not."""
        source = (PAPER / "short.py").read_text()
        assert "only negative entry" not in source
        assert "gap_pct'] < 0).sum()" in source.replace('"', "'"), (
            "the count of negative cells is typed rather than counted")

    def test_both_reversals_get_the_same_three_statistics(self) -> None:
        source = (PAPER / "short.py").read_text()
        body = source[source.index("def _both_reversals("):
                      source.index("def _sign_split(")]
        for field in ("ci_low", "below_point", "bias_estimate",
                      "sign_holds"):
            assert field in body, field
        # Both systems, not one.
        assert "means_tested/voluntary" in body
        assert "australia_as_legislated" in body

    def test_the_diagnostic_is_not_applied_to_one_cell_only(self) -> None:
        """The substantive point: the bias diagnostic fires on the matched
        cell too, and a section that reported it for the legislated cell
        alone would be choosing where to be sceptical."""
        import pandas as pd

        path = (PAPER.parent / "results" / "tables" / "ordering_pseudo.csv")
        if not path.exists():
            pytest.skip("the ordering study has not been run")
        pseudo = pd.read_csv(path)
        rule = "fixed_real_rule"
        cells = pseudo[(pseudo["rule"] == rule)
                       & pseudo["system"].isin(["age_pension_matched",
                                                "australia_as_legislated"])]
        assert len(cells) == 2
        # Both are skewed, which is why both must be reported.
        assert (cells["below_point"] < 6).all(), (
            "the matched cell's pseudo-values are now well behaved; the "
            "paper's even-handedness paragraph needs rechecking")

    def test_no_passage_calls_the_reversal_unresolvable_outright(self) -> None:
        """Three did, after Section 10.1 had resolved it at matched
        contributions. Each now has to name which cell it means."""
        source = (PAPER / "short.py").read_text()
        for dead in ("state the reversal as what it is",
                     "the rule effect is resolved and the reversal is not"):
            assert dead not in source, dead

    def test_the_built_paper_signs_the_matched_cell(self) -> None:
        pdf = PAPER / "floor_beneath_the_portfolio.pdf"
        reader = pytest.importorskip("pypdf")
        if not pdf.exists():
            pytest.skip("the short paper has not been built")
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.PdfReader(str(pdf)).pages)
        assert "all sixteen" in text or "16 of 16" in text


class TestTheTwoHouseholdsAreLabelled:
    """The headline moved from the household that carries the
    Superannuation Guarantee to the one that does not, and the framing
    prose did not move with it -- so the introduction quoted a 32x balance
    beside a headline computed on a 14x one."""

    @staticmethod
    def _bite():
        import pandas as pd

        path = (PAPER.parent / "results" / "tables"
                / "leisure_means_test_bite.csv")
        if not path.exists():
            pytest.skip("the leisure study has not been run")
        return pd.read_csv(path).set_index("household")

    def test_the_two_households_are_genuinely_different(self) -> None:
        """If they ever converge the labelling below is harmless noise;
        while they differ by a factor of two it is not."""
        bite = self._bite()
        matched = float(bite.loc["pension schedule only",
                                 "median_wealth_multiple"])
        legislated = float(bite.loc["as legislated", "median_wealth_multiple"])
        assert legislated / matched > 1.5

    def test_the_introduction_reads_both_rows(self) -> None:
        source = (PAPER / "short.py").read_text()
        body = source[source.index("def introduction("):
                      source.index("def model(")]
        assert "pension schedule only" in body, (
            "the introduction reads only the legislated household while "
            "the headline is computed on the matched one")
        assert "sched[" in body and "legis[" in body

    def test_the_headline_household_is_the_one_the_headline_uses(self) -> None:
        """The matched cell of the ordering grid and the 'pension schedule
        only' row of the means-test table are the same household: 10%
        saving, Australian pension. If the config ever renames one the
        introduction's arithmetic silently describes someone else."""
        import yaml

        cfg = yaml.safe_load((PAPER.parent / "config.yaml").read_text())
        factorial = cfg["ordering"]["factorial"]
        assert factorial["means_tested"]["voluntary"] == "age_pension_matched"
        assert factorial["earnings_related"]["voluntary"] == \
            "us_social_security"


class TestTheFigureAndTheTableNameTheArmsAlike:
    """Section 10's table and its figure both label five pension regimes.
    They did it with two vocabularies: the table said "United States, 20.2%
    saving" and "Means-tested, 10% saving" where the figure said "United
    States, matched saving" and "Age Pension, matched saving" -- the same
    phrase for the two arms whose contribution rates are opposite."""

    def test_every_ordering_arm_has_one_name(self) -> None:
        import re

        from src import plots

        source = (PAPER / "short.py").read_text()
        block = source[source.index('    label = {"us_social_security"'):]
        block = block[:block.index("}")]
        paper_names = dict(re.findall(r'"([a-z_]+)":\s*"([^"]+)"', block))
        assert paper_names, "the ordering section's label map moved"
        for key, name in paper_names.items():
            assert plots.SYSTEM_LABEL.get(key) == name, (
                f"{key} is '{name}' in the table and "
                f"'{plots.SYSTEM_LABEL.get(key)}' in the figure")

    def test_no_two_arms_share_a_name(self) -> None:
        """The failure was not a mismatch but a collision: two different
        contribution rates under one phrase."""
        from src import plots

        keys = ("us_social_security", "us_matched_saving",
                "age_pension_untested", "age_pension_matched",
                "australia_as_legislated")
        names = [plots.SYSTEM_LABEL[k] for k in keys]
        assert len(set(names)) == len(names), names
        assert not any(n.count("matched saving") for n in names), (
            "'matched' describes a relation between two arms, not a "
            "contribution rate; naming an arm by it hides the rate")
