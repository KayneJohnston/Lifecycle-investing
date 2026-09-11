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


class TestReadingOrderDependencies:
    """A section may point forward to say "we return to this", but it must
    not depend on a result the reader has not been given yet."""

    #: Sections whose job is to preview or summarise the whole paper.
    SIGNPOSTS = frozenset({"introduction", "background", "data", "methods",
                           "discussion", "limitations", "conclusion"})

    @staticmethod
    def _bodies() -> dict:
        import re
        from pathlib import Path

        src = Path("paper/content.py").read_text()
        out = {}
        for key in content.SECTION_ORDER:
            m = re.search(rf"\ndef section_{re.escape(key)}\(", src)
            if not m:
                continue
            start = m.start()
            nxt = re.search(r"\ndef section_", src[start + 1:])
            end = start + 1 + (nxt.start() if nxt else len(src) - start - 1)
            out[key] = src[start:end]
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
            # `plan` may point on to the institutional pair as signposting;
            # nothing here may reach past it.
            allowed = {"leisure", "tax"} if key == "plan" else set()
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

    def test_every_short_section_exists_in_the_long_order(self) -> None:
        from paper import short as sh

        unknown = [k for k in sh.SHORT_ORDER if k not in content.SECTION_ORDER]
        assert not unknown

    def test_the_argument_sections_are_all_present(self) -> None:
        from paper import short as sh

        for key in ("pension", "leisure", "longevity", "tax"):
            assert key in sh.SHORT_ORDER, key

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
