"""
Pytest suite for the standalone ``arabic_prosody_feedback.py``.

The module embeds its own copy of the prosody data model and analysis
helpers (formerly in ``arabic_prosody_helpers.py``), so this test file
imports everything directly from ``arabic_prosody_feedback``.

Two layers of tests are provided:

1. Pure unit tests for lookup tables, pattern conversion, scoring,
   zihāf identification, meter-name resolution, and the report-formatting
   functions. These build :class:`FootResult` / :class:`HemistichResult` /
   :class:`VerseResult` / :class:`PoemResult` objects by hand, so they run
   with **no external dependency** (pyarud not required).

2. Integration tests that call :func:`analyze_poem` / :func:`analyze_verse`
   / :func:`analyze_and_report` through the real **pyarud** library. These
   are skipped automatically if pyarud is not installed, and a small
   complementary test confirms the clean ``RuntimeError`` when it's absent.
"""

import pytest

import arabic_prosody_feedback as apf
from arabic_prosody_feedback import (
    CANONICAL_PATTERNS,
    METER_ARABIC_NAMES,
    METER_TEMPLATES,
    FootResult,
    HemistichResult,
    PoemResult,
    VerseResult,
    _enrich_foot,
    _enrich_hemistich,
    _mora_diff,
    _pattern_to_class,
    _render_diff,
    _resolve_key,
    _tafeela_label,
    analyze_and_report,
    analyze_poem,
    analyze_verse,
    binary_to_ux,
    foot_health,
    generate_diagnostics,
    generate_poem_correction_report,
    generate_verse_correction,
    get_canonical_pattern,
    get_tafeela_mnemonic,
    identify_zihaf,
    resolve_meter_key,
    similarity,
    to_pyarud_meter_key,
    ux_to_binary,
)

PYARUD_AVAILABLE = apf._PYARUD_AVAILABLE

needs_pyarud = pytest.mark.skipif(
    not PYARUD_AVAILABLE, reason="pyarud is not installed"
)


# ===========================================================================
# Pattern conversion (binary_to_ux / ux_to_binary / similarity)
# ===========================================================================


class TestPatternConversion:
    def test_binary_to_ux_basic(self):
        assert binary_to_ux("11010") == "UU_U_"
        assert binary_to_ux("1011010") == "U_UU_U_"

    def test_ux_to_binary_basic(self):
        assert ux_to_binary("UU_U_") == "11010"
        assert ux_to_binary("U_UU_U_") == "1011010"

    def test_roundtrip_for_all_canonical_patterns(self):
        for pattern in CANONICAL_PATTERNS.values():
            ux = binary_to_ux(pattern)
            assert ux_to_binary(ux) == pattern

    def test_non_binary_characters_pass_through(self):
        # Separator/space characters are untouched by the substitution.
        assert binary_to_ux("1 0|1") == "U _|U"
        assert ux_to_binary("U _|U") == "1 0|1"


class TestSimilarity:
    def test_identical_strings_score_one(self):
        assert similarity("1010110", "1010110") == 1.0

    def test_score_in_unit_interval(self):
        score = similarity("1010110", "0101101")
        assert 0.0 <= score <= 1.0

    def test_closer_strings_score_higher(self):
        close = similarity("1010110", "1010111")  # one bit different
        far = similarity("1010110", "0101001")  # many bits different
        assert close > far


# ===========================================================================
# Zihāf identification & foot health
# ===========================================================================


class TestIdentifyZihaf:
    def test_identical_patterns_are_salim(self):
        assert identify_zihaf("11010", "11010") == "Salim"

    def test_known_zihaf_lookup(self):
        assert identify_zihaf("1010110", "110110") == "Khaban"
        assert identify_zihaf("11010", "110") == "Hadhf"
        assert identify_zihaf("11010", "10") == "Batr"

    def test_unknown_dropped_bits_plural(self):
        assert identify_zihaf("11010", "11") == "Unknown (3 bits dropped)"

    def test_unknown_dropped_bits_singular(self):
        assert identify_zihaf("110", "11") == "Unknown (1 bit dropped)"

    def test_unknown_added_bits_plural(self):
        assert identify_zihaf("11", "11010") == "Unknown (3 bits added)"

    def test_unknown_added_bits_singular(self):
        assert identify_zihaf("11", "110") == "Unknown (1 bit added)"

    def test_unknown_taskeen_same_length_plural(self):
        assert identify_zihaf("1010", "1100") == "Unknown (Taskeen, 2 bits changed)"

    def test_unknown_taskeen_same_length_singular(self):
        assert identify_zihaf("1010", "1011") == "Unknown (Taskeen, 1 bit changed)"

    @pytest.mark.parametrize(
        "canonical,actual,expected_zihaf",
        [
            ("11010", "1101", "Qabadh"),      # Fawlon Qabadh
            ("1011010", "101101", "Kaff"),     # Faelaton Kaff
            ("1011010", "10110", "Hadhf"),     # Faelaton Hadhf
            ("1011010", "11101", "Shakal"),    # Faelaton Shakal
            ("1011010", "1011", "Waqf"),       # Faelaton Waqf
            ("1101010", "110101", "Kaff"),     # Mafaeelon Kaff
            ("1101010", "11010", "Hadhf"),     # Mafaeelon Hadhf
            ("1101010", "11011", "Shakl_alt"), # Mafaeelon Shakl_alt
            ("1010110", "101110", "Tay"),      # Mustafelon Tay
            ("1010110", "11110", "Khabal"),    # Mustafelon Khabal
            ("1010110", "101010", "Kasf"),     # Mustafelon Kasf
            ("1110110", "1010110", "Edmaar"),  # Mutafaelon Edmaar
            ("1110110", "110110", "Waqas"),    # Mutafaelon Waqas
            ("1110110", "101110", "Khazal"),   # Mutafaelon Khazal
            ("1101110", "110110", "Akal"),     # Mafaelaton Akal
            ("1101110", "1101010", "Asab"),    # Mafaelaton Asab
            ("1101110", "11010", "Qatf"),      # Mafaelaton Qatf
            ("1010101", "110101", "Khaban"),   # Mafoolato Khaban
            ("1010101", "101101", "Tay"),      # Mafoolato Tay
            ("1010101", "10101", "Kasf"),      # Mafoolato Kasf
        ]
    )
    def test_zihaf_map_broad_coverage(self, canonical, actual, expected_zihaf):
        assert identify_zihaf(canonical, actual) == expected_zihaf


class TestFootHealth:
    @pytest.mark.parametrize(
        "status,zihaf_name,expected",
        [
            ("missing", None, "severe"),
            ("extra_bits", None, "severe"),
            ("broken", None, "broken"),
            ("ok", "Salim", "perfect"),
            ("ok", "Khaban", "valid_zihaf"),
            ("ok", None, "valid_zihaf"),
        ],
    )
    def test_health_levels(self, status, zihaf_name, expected):
        assert foot_health(status, 1.0, zihaf_name) == expected


class TestCanonicalPattern:
    def test_known_foot_classes(self):
        assert get_canonical_pattern("Fawlon") == "11010"
        assert get_canonical_pattern("Mustafelon") == "1010110"

    def test_unknown_foot_class_returns_none(self):
        assert get_canonical_pattern("NotAFootClass") is None


# ===========================================================================
# Meter-name resolution
# ===========================================================================


class TestResolveMeterKey:
    @pytest.mark.parametrize(
        "variant", ["khafif", "al-khafif", "khafīf", "خفيف", "الخفيف"]
    )
    def test_khafif_aliases_resolve(self, variant):
        assert resolve_meter_key(variant) == "khafif"

    def test_strips_whitespace_and_lowercases(self):
        assert resolve_meter_key("  KHAFIF  ") == "khafif"

    def test_unknown_meter_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown meter"):
            resolve_meter_key("not-a-real-meter")

    def test_private_resolve_key_matches_public_wrapper(self):
        assert _resolve_key("basit") == resolve_meter_key("basit")


class TestToPyarudMeterKey:
    def test_none_passes_through(self):
        assert to_pyarud_meter_key(None) is None

    @pytest.mark.parametrize("native_key", ["baseet", "khafeef", "ramal", "rajaz"])
    def test_pyarud_native_keys_pass_through_unchanged(self, native_key):
        assert to_pyarud_meter_key(native_key) == native_key

    @pytest.mark.parametrize(
        "variant,expected",
        [
            ("khafif", "khafeef"),
            ("الخفيف", "khafeef"),
            ("basit", "baseet"),
            ("kamil", "kamel"),
            ("wafir", "wafer"),
            ("tawil", "taweel"),
            ("mutaqarib", "mutakareb"),
        ],
    )
    def test_alias_variants_translate_to_pyarud_keys(self, variant, expected):
        assert to_pyarud_meter_key(variant) == expected

    def test_unknown_meter_raises_value_error(self):
        with pytest.raises(ValueError):
            to_pyarud_meter_key("totally-unknown-meter")


# ===========================================================================
# Tafʿīla mnemonics & standalone labels
# ===========================================================================


class TestTafeelaMnemonic:
    def test_known_combination(self):
        assert get_tafeela_mnemonic("Mustafelon", "Khaban") == "مُتَفْعِلُنْ"

    def test_none_zihaf_defaults_to_salim(self):
        assert get_tafeela_mnemonic("Mustafelon", None) == get_tafeela_mnemonic(
            "Mustafelon", "Salim"
        )

    def test_unknown_zihaf_falls_back_with_suffix(self):
        result = get_tafeela_mnemonic("Mustafelon", "NoSuchZihaf")
        salim = get_tafeela_mnemonic("Mustafelon", "Salim")
        assert result == f"{salim} (NoSuchZihaf)"

    def test_unknown_foot_class_with_salim_returns_class_name(self):
        assert get_tafeela_mnemonic("NotAClass", None) == "NotAClass"
        assert get_tafeela_mnemonic("NotAClass", "Salim") == "NotAClass"

    def test_unknown_foot_class_with_zihaf_appends_name(self):
        assert get_tafeela_mnemonic("NotAClass", "Khaban") == "NotAClass (Khaban)"


class TestPatternToClass:
    def test_known_pattern_maps_to_class(self):
        assert _pattern_to_class("UU_U_") == "Fawlon"  # binary 11010

    def test_unknown_pattern_returns_none(self):
        assert _pattern_to_class("XXXXX") is None


class TestTafeelaLabel:
    def test_tafeela_label_salim(self):
        foot = FootResult(
            foot_index=0,
            expected_pattern="U_U_UU_",  # Mustafelon binary: 1010110
            actual_segment="U_U_UU_",
            canonical_pattern="U_U_UU_",
            score=1.0,
            status="ok",
            zihaf_name="Salim",
            health="perfect",
            position_label="Hashw"
        )
        assert _tafeela_label(foot) == "مُسْتَفْعِلُنْ"

    def test_tafeela_label_zihaf(self):
        foot = FootResult(
            foot_index=0,
            expected_pattern="U_U_UU_",
            actual_segment="UUU_UU_",  # Khaban
            canonical_pattern="U_U_UU_",
            score=0.95,
            status="ok",
            zihaf_name="Khaban",
            health="valid_zihaf",
            position_label="Hashw"
        )
        assert _tafeela_label(foot) == "مُتَفْعِلُنْ"

    def test_tafeela_label_unknown(self):
        foot = FootResult(
            foot_index=0,
            expected_pattern="XXXXX",
            actual_segment="XXXXX",
            canonical_pattern="XXXXX",
            score=0.0,
            status="broken",
            zihaf_name=None,
            health="broken",
            position_label="Hashw"
        )
        assert _tafeela_label(foot) == "XXXXX"


# ===========================================================================
# Mora-diff & character-level diff rendering
# ===========================================================================


class TestMoraDiff:
    def test_exact_match(self):
        result = _mora_diff("UU_U_", "UU_U_")
        assert result == {
            "len_diff": 0,
            "direction": "match",
            "first_div": -1,
            "suggestion": "Pattern matches exactly.",
        }

    def test_too_short(self):
        result = _mora_diff("UU_U_U_", "UU_U_")
        assert result["direction"] == "too_short"
        assert result["len_diff"] == 2
        assert result["first_div"] == 5
        assert "too short" in result["suggestion"]
        assert "UU_U_U_" in result["suggestion"]

    def test_too_long(self):
        result = _mora_diff("UU_U_", "UU_U_U_")
        assert result["direction"] == "too_long"
        assert result["len_diff"] == -2
        assert "too long" in result["suggestion"]
        assert "UU_U_" in result["suggestion"]

    def test_wrong_weight_same_length(self):
        result = _mora_diff("UU_U_", "U__U_")
        assert result["direction"] == "wrong_weight"
        assert result["len_diff"] == 0
        assert result["first_div"] == 1
        assert "pos 2: long (_) → short (U)" in result["suggestion"]


class TestMoraDiffSingular:
    def test_too_short_singular(self):
        result = _mora_diff("UU_U_U", "UU_U_")
        assert result["direction"] == "too_short"
        assert result["len_diff"] == 1
        assert "1 mora(s)" in result["suggestion"]

    def test_too_long_singular(self):
        result = _mora_diff("UU_U_", "UU_U_U")
        assert result["direction"] == "too_long"
        assert result["len_diff"] == -1
        assert "1 mora(s)" in result["suggestion"]


class TestRenderDiff:
    def test_perfect_match_uses_only_match_markers(self):
        diff = _render_diff("UU_U_", "UU_U_")
        lines = diff.splitlines()
        assert lines[0].strip().startswith("Expected:")
        assert lines[1].strip().startswith("Actual:")
        assert lines[2].strip().startswith("Diff:")
        # Separate the legend on same line to check the pure diff part cleanly
        diff_part = lines[2].split("    (|")[0]
        assert "×" not in diff_part and "^" not in diff_part and "v" not in diff_part
        assert diff_part.count("|") == 5

    def test_missing_tail_uses_caret_markers(self):
        diff = _render_diff("UU_U_U_", "UU_U_")
        lines = diff.splitlines()
        diff_part = lines[2].split("    (|")[0]
        assert "^" in diff_part
        assert "×" not in diff_part

    def test_wrong_weight_uses_cross_markers(self):
        diff = _render_diff("UU_U_", "U__U_")
        lines = diff.splitlines()
        diff_part = lines[2].split("    (|")[0]
        assert "×" in diff_part

    def test_legend_always_present(self):
        diff = _render_diff("UU_U_", "UU_U_")
        assert "match" in diff and "wrong weight" in diff
        assert "missing" in diff and "extra" in diff


# ===========================================================================
# _enrich_foot / _enrich_hemistich (pure dict -> dataclass enrichment)
# ===========================================================================


class TestEnrichFoot:
    def test_salim_foot(self):
        raw = {
            "foot_index": 2,
            "expected_pattern": "1010110",
            "actual_segment": "1010110",
            "status": "ok",
            "score": 1.0,
        }
        foot = _enrich_foot(raw, "Hashw", 4)
        assert foot.zihaf_name == "Salim"
        assert foot.health == "perfect"
        assert foot.expected_pattern == "U_U_UU_"
        assert foot.actual_segment == "U_U_UU_"
        assert foot.canonical_pattern == "U_U_UU_"
        assert foot.position_label == "Hashw"

    def test_zihaf_foot_resolves_canonical_via_zihaf_map(self):
        # "110110" is the Khaban-modified form of Mustafelon's "1010110"
        raw = {
            "foot_index": 0,
            "expected_pattern": "1010110",
            "actual_segment": "110110",
            "status": "ok",
            "score": 0.9,
        }
        foot = _enrich_foot(raw, "Hashw", 4)
        assert foot.zihaf_name == "Khaban"
        assert foot.health == "valid_zihaf"
        assert foot.canonical_pattern == binary_to_ux("1010110")

    def test_broken_foot_has_no_zihaf(self):
        raw = {
            "foot_index": 0,
            "expected_pattern": "1010110",
            "actual_segment": "111111",
            "status": "broken",
            "score": 0.1,
        }
        foot = _enrich_foot(raw, "Hashw", 4)
        assert foot.zihaf_name is None
        assert foot.health == "broken"

    def test_missing_foot(self):
        raw = {
            "foot_index": 3,
            "expected_pattern": "1010110",
            "actual_segment": "",
            "status": "missing",
            "score": 0.0,
        }
        foot = _enrich_foot(raw, "ʿArūḍ", 4)
        assert foot.status == "missing"
        assert foot.health == "severe"
        assert foot.zihaf_name is None


class TestEnrichHemistich:
    def test_hashw_and_arud_position_labels(self):
        raw_feet = [
            {
                "foot_index": 0,
                "expected_pattern": "1010110",
                "actual_segment": "1010110",
                "status": "ok",
                "score": 1.0,
            },
            {
                "foot_index": 1,
                "expected_pattern": "10110",
                "actual_segment": "1110",
                "status": "ok",
                "score": 0.9,
            },
            {
                "foot_index": 2,
                "expected_pattern": "1010110",
                "actual_segment": "111111",
                "status": "broken",
                "score": 0.1,
            },
        ]
        h = _enrich_hemistich("text", "1010110111010111111", raw_feet, is_ajuz=False)

        assert [f.position_label for f in h.feet] == ["Hashw", "Hashw", "ʿArūḍ"]
        assert h.broken_foot_indices == [2]
        assert h.missing_foot_count == 0
        assert h.is_sound is False
        assert h.extra_bits is None
        assert h.pattern == binary_to_ux("1010110111010111111")

    def test_darb_label_used_for_ajuz_final_foot(self):
        raw_feet = [
            {
                "foot_index": 0,
                "expected_pattern": "1010110",
                "actual_segment": "1010110",
                "status": "ok",
                "score": 1.0,
            },
        ]
        h = _enrich_hemistich("text", "1010110", raw_feet, is_ajuz=True)
        assert h.feet[0].position_label == "Ḍarb"
        assert h.is_sound is True

    def test_extra_bits_label_and_field(self):
        raw_feet = [
            {
                "foot_index": 0,
                "expected_pattern": "1010110",
                "actual_segment": "1010110",
                "status": "ok",
                "score": 1.0,
            },
            {
                "foot_index": 1,
                "expected_pattern": "",
                "actual_segment": "11",
                "status": "extra_bits",
                "score": 0.0,
            },
        ]
        h = _enrich_hemistich("text", "101011011", raw_feet, is_ajuz=False)
        assert h.feet[1].position_label == "Extra"
        assert h.extra_bits == "UU"
        assert h.is_sound is False

    def test_missing_foot_count_is_tracked(self):
        raw_feet = [
            {
                "foot_index": 0,
                "expected_pattern": "1010110",
                "actual_segment": "1010110",
                "status": "ok",
                "score": 1.0,
            },
            {
                "foot_index": 1,
                "expected_pattern": "10110",
                "actual_segment": "",
                "status": "missing",
                "score": 0.0,
            },
        ]
        h = _enrich_hemistich("text", "1010110", raw_feet, is_ajuz=False)
        assert h.missing_foot_count == 1
        assert h.is_sound is False

    def test_average_score_excludes_missing_and_extra_bits(self):
        raw_feet = [
            {
                "foot_index": 0,
                "expected_pattern": "1010110",
                "actual_segment": "1010110",
                "status": "ok",
                "score": 1.0,
            },
            {
                "foot_index": 1,
                "expected_pattern": "10110",
                "actual_segment": "10110",
                "status": "ok",
                "score": 0.5,
            },
            {
                "foot_index": 2,
                "expected_pattern": "10110",
                "actual_segment": "",
                "status": "missing",
                "score": 0.0,
            },
        ]
        h = _enrich_hemistich("text", "101011011010110", raw_feet, is_ajuz=False)
        assert h.score == pytest.approx(0.75)


# ===========================================================================
# generate_diagnostics (pure, built from hand-crafted HemistichResults)
# ===========================================================================


def _make_hemistich(feet, **overrides):
    defaults = dict(
        text="x",
        pattern="x",
        score=1.0,
        is_sound=True,
        broken_foot_indices=[],
        missing_foot_count=0,
        extra_bits=None,
    )
    defaults.update(overrides)
    return HemistichResult(feet=feet, **defaults)


class TestGenerateDiagnostics:
    def test_sound_verse_message(self):
        foot = FootResult(
            0, "UU_U_", "UU_U_", "UU_U_", 1.0, "ok", "Salim", "perfect", "Hashw"
        )
        h = _make_hemistich([foot])
        msgs = generate_diagnostics(h, None, "baseet")
        assert len(msgs) == 1
        assert "metrically sound" in msgs[0]
        assert "البسيط" in msgs[0]

    def test_broken_foot_message(self):
        foot = FootResult(
            0, "UU_U_", "UUU__", "UU_U_", 0.1, "broken", None, "broken", "Hashw"
        )
        h = _make_hemistich([foot], broken_foot_indices=[0], is_sound=False)
        msgs = generate_diagnostics(h, None, "baseet")
        assert any("broken" in m and "foot 1 (Hashw)" in m for m in msgs)

    def test_missing_foot_message_and_summary(self):
        foot = FootResult(
            1, "UU_U_", "", "UU_U_", 0.0, "missing", None, "severe", "ʿArūḍ"
        )
        h = _make_hemistich([foot], missing_foot_count=1, is_sound=False)
        msgs = generate_diagnostics(h, None, "baseet")
        assert any("missing" in m and "foot 2" in m for m in msgs)
        assert any("missing foot(s)" in m for m in msgs)

    def test_extra_bits_message(self):
        foot = FootResult(2, "", "UU", "", 0.0, "extra_bits", None, "severe", "Extra")
        h = _make_hemistich([foot], extra_bits="UU", is_sound=False)
        msgs = generate_diagnostics(h, None, "baseet")
        assert any("extra bits" in m and "UU" in m for m in msgs)

    def test_valid_zihaf_message(self):
        foot = FootResult(
            0,
            "U_UU_U_",
            "UUU_U_",
            "U_UU_U_",
            0.95,
            "ok",
            "Khaban",
            "valid_zihaf",
            "Hashw",
        )
        h = _make_hemistich([foot])
        msgs = generate_diagnostics(h, None, "khafeef")
        assert any("Khaban" in m for m in msgs)

    def test_ajuz_messages_use_ajuz_label(self):
        sadr_foot = FootResult(
            0, "UU_U_", "UU_U_", "UU_U_", 1.0, "ok", "Salim", "perfect", "Hashw"
        )
        ajuz_foot = FootResult(
            0, "UU_U_", "UUU__", "UU_U_", 0.1, "broken", None, "broken", "Hashw"
        )
        sadr = _make_hemistich([sadr_foot])
        ajuz = _make_hemistich([ajuz_foot], broken_foot_indices=[0], is_sound=False)
        msgs = generate_diagnostics(sadr, ajuz, "baseet")
        assert any(m.startswith("[ʿAjuz]") for m in msgs)


class TestGenerateDiagnosticsSimultaneousBroken:
    def test_both_sadr_and_ajuz_broken(self):
        sadr_broken = FootResult(
            0, "UU_U_", "UUU__", "UU_U_", 0.1, "broken", None, "broken", "Hashw"
        )
        ajuz_broken = FootResult(
            0, "UU_U_", "U_U_U", "UU_U_", 0.1, "broken", None, "broken", "Hashw"
        )
        sadr = _make_hemistich([sadr_broken], broken_foot_indices=[0], is_sound=False)
        ajuz = _make_hemistich([ajuz_broken], broken_foot_indices=[0], is_sound=False)
        msgs = generate_diagnostics(sadr, ajuz, "baseet")
        assert any("[Ṣadr]" in m for m in msgs)
        assert any("[ʿAjuz]" in m for m in msgs)


# ===========================================================================
# generate_verse_correction (pure, built from hand-crafted VerseResults)
# ===========================================================================


def _sound_foot(foot_index, expected, position_label):
    return FootResult(
        foot_index,
        expected,
        expected,
        expected,
        1.0,
        "ok",
        "Salim",
        "perfect",
        position_label,
    )


def _sound_verse(combined_score=1.0, meter="baseet"):
    sadr = HemistichResult(
        text="صدر",
        pattern="x",
        feet=[_sound_foot(0, "U_U_UU_", "Hashw"), _sound_foot(1, "U_UU_", "ʿArūḍ")],
        score=1.0,
        is_sound=True,
        broken_foot_indices=[],
        missing_foot_count=0,
        extra_bits=None,
    )
    return VerseResult(
        verse_index=0,
        sadr=sadr,
        ajuz=None,
        combined_score=combined_score,
        meter=meter,
        issues=[f"Verse is metrically sound ({METER_ARABIC_NAMES[meter]})."],
    )


class TestGenerateVerseCorrection:
    def test_sound_verse_header_and_body(self):
        report = generate_verse_correction(_sound_verse())
        assert "VERSE 1" in report
        assert "✓ SOUND" in report
        assert "Score: 100%" in report
        assert "All feet parsed correctly" in report
        assert "DETAILED DIAGNOSIS" not in report
        assert "CORRECTION PRESCRIPTION" not in report

    @pytest.mark.parametrize(
        "score,tag",
        [
            (1.0, "✓ SOUND"),
            (0.95, "~ NEAR-PERFECT"),
            (0.80, "⚠ IRREGULAR"),
            (0.40, "✗ BROKEN"),
        ],
    )
    def test_score_tags(self, score, tag):
        report = generate_verse_correction(_sound_verse(combined_score=score))
        assert tag in report
        assert f"Score: {score * 100:.0f}%" in report

    def test_broken_foot_produces_diagnosis_and_prescription(self):
        broken = FootResult(
            0, "U_UU_", "UU_U_", "U_UU_", 0.1, "broken", None, "broken", "Hashw"
        )
        sound = _sound_foot(1, "U_U_UU_", "ʿArūḍ")
        sadr = HemistichResult("صدر", "x", [broken, sound], 0.55, False, [0], 0, None)
        verse = VerseResult(0, sadr, None, 0.55, "baseet", ["..."])

        report = generate_verse_correction(verse)
        assert "✗ BROKEN" in report
        assert "DETAILED DIAGNOSIS" in report
        assert "CORRECTION PRESCRIPTION" in report
        assert "Foot 1" in report
        assert (
            "Adjust syllable weights" in report
            or "more mora" in report
            or "remove" in report
        )

    def test_missing_foot_prescription(self):
        missing = FootResult(
            1, "U_UU_", "", "U_UU_", 0.0, "missing", None, "severe", "ʿArūḍ"
        )
        sound = _sound_foot(0, "U_U_UU_", "Hashw")
        sadr = HemistichResult("صدر", "x", [sound, missing], 0.5, False, [], 1, None)
        verse = VerseResult(0, sadr, None, 0.5, "baseet", ["..."])

        report = generate_verse_correction(verse)
        assert "? MISSING" in report
        assert "MISSING" in report
        assert "Add text supplying" in report
        assert "U_UU_" in report

    def test_extra_bits_prescription(self):
        extra = FootResult(1, "", "UU", "", 0.0, "extra_bits", None, "severe", "Extra")
        sound = _sound_foot(0, "U_U_UU_", "Hashw")
        sadr = HemistichResult("صدر", "x", [sound, extra], 0.5, False, [], 0, "UU")
        verse = VerseResult(0, sadr, None, 0.5, "baseet", ["..."])

        report = generate_verse_correction(verse)
        assert "! EXTRA" in report
        assert "Remove word(s)" in report
        assert "UU" in report

    def test_meter_schema_toggle(self):
        verse = _sound_verse()
        with_schema = generate_verse_correction(verse, include_meter_schema=True)
        without_schema = generate_verse_correction(verse, include_meter_schema=False)

        assert "METER REFERENCE" in with_schema
        assert METER_TEMPLATES["baseet"] in with_schema
        assert "METER REFERENCE" not in without_schema

    def test_ajuz_grid_rendered_when_present(self):
        ajuz = HemistichResult(
            text="عجز",
            pattern="x",
            feet=[_sound_foot(0, "U_U_UU_", "Hashw"), _sound_foot(1, "U_UU_", "Ḍarb")],
            score=1.0,
            is_sound=True,
            broken_foot_indices=[],
            missing_foot_count=0,
            extra_bits=None,
        )
        verse = _sound_verse()
        verse = VerseResult(
            verse.verse_index,
            verse.sadr,
            ajuz,
            verse.combined_score,
            verse.meter,
            verse.issues,
        )

        report = generate_verse_correction(verse)
        assert "ʿAJUZ (عَجُز)" in report
        assert "عجز" in report


class TestGenerateVerseCorrectionPrescriptions:
    def test_too_short_foot_prescription(self):
        # expected length = 5, actual length = 4 (len_diff = 1 -> too_short -> ADD prescription)
        broken_foot = FootResult(
            foot_index=0,
            expected_pattern="UU_U_",
            actual_segment="UU_U",
            canonical_pattern="UU_U_",
            score=0.8,
            status="broken",
            zihaf_name=None,
            health="broken",
            position_label="Hashw"
        )
        sadr = HemistichResult("صَدْر", "x", [broken_foot], 0.8, False, [0], 0, None)
        verse = VerseResult(0, sadr, None, 0.8, "baseet", ["..."])
        report = generate_verse_correction(verse)
        assert "DETAILED DIAGNOSIS" in report
        assert "CORRECTION PRESCRIPTION" in report
        assert "Replace word(s) giving «UU_U» with word(s) giving «UU_U_» — need 1 more mora(s)" in report

    def test_too_long_foot_prescription(self):
        # expected length = 5, actual length = 6 (len_diff = -1 -> too_long -> TRIM prescription)
        broken_foot = FootResult(
            foot_index=0,
            expected_pattern="UU_U_",
            actual_segment="UU_U_U",
            canonical_pattern="UU_U_",
            score=0.8,
            status="broken",
            zihaf_name=None,
            health="broken",
            position_label="Hashw"
        )
        sadr = HemistichResult("صَدْر", "x", [broken_foot], 0.8, False, [0], 0, None)
        verse = VerseResult(0, sadr, None, 0.8, "baseet", ["..."])
        report = generate_verse_correction(verse)
        assert "Shorten word(s) giving «UU_U_U» to produce «UU_U_» — remove 1 mora(s)" in report


# ===========================================================================
# generate_poem_correction_report (pure, built from hand-crafted PoemResults)
# ===========================================================================


def _verse(idx, score, sound=True):
    if sound:
        feet = [_sound_foot(0, "U_UU_", "ʿArūḍ")]
        sadr = HemistichResult(f"verse {idx}", "x", feet, 1.0, True, [], 0, None)
        issues = [f"Verse is metrically sound ({METER_ARABIC_NAMES['baseet']})."]
    else:
        foot = FootResult(
            0, "U_UU_", "UU_U_", "U_UU_", 0.1, "broken", None, "broken", "ʿArūḍ"
        )
        sadr = HemistichResult(f"verse {idx}", "x", [foot], 0.1, False, [0], 0, None)
        issues = ["..."]
    return VerseResult(idx, sadr, None, score, "baseet", issues)


class TestGeneratePoemCorrectionReport:
    def test_all_sound_poem_skips_detail_sections(self):
        verses = [_verse(0, 1.0, True), _verse(1, 1.0, True)]
        poem = PoemResult("baseet", "البسيط", 2, verses, 1.0, True, [("baseet", 1.0)])

        report = generate_poem_correction_report(poem)
        assert "All verses are metrically sound" in report
        assert "CONSOLIDATED FIX LIST" not in report
        assert "Broken / total: 0 / 2" in report

    def test_only_broken_default_skips_sound_verse_detail(self):
        verses = [_verse(0, 1.0, True), _verse(1, 0.1, False)]
        poem = PoemResult(
            "baseet", "البسيط", 2, verses, 0.55, False, [("baseet", 0.55)]
        )

        report = generate_poem_correction_report(poem)
        assert "VERSE 1" not in report  # detail for the sound verse is skipped
        assert "VERSE 2" in report
        assert "CONSOLIDATED FIX LIST" in report
        assert "Broken / total: 1 / 2" in report

    def test_only_broken_false_includes_all_verse_details(self):
        verses = [_verse(0, 1.0, True), _verse(1, 0.1, False)]
        poem = PoemResult(
            "baseet", "البسيط", 2, verses, 0.55, False, [("baseet", 0.55)]
        )

        report = generate_poem_correction_report(poem, only_broken=False)
        assert "VERSE 1" in report
        assert "VERSE 2" in report

    def test_score_threshold_flags_near_perfect_as_broken(self):
        verses = [_verse(0, 0.95, True)]
        poem = PoemResult("baseet", "البسيط", 1, verses, 0.95, True, [("baseet", 0.95)])

        # Default threshold is 0.99, so a 0.95 verse counts as "broken".
        report = generate_poem_correction_report(poem)
        assert "Broken / total: 1 / 1" in report

    def test_meter_schema_shown_only_once(self):
        verses = [_verse(0, 0.1, False), _verse(1, 0.1, False)]
        poem = PoemResult("baseet", "البسيط", 2, verses, 0.1, False, [("baseet", 0.1)])

        report = generate_poem_correction_report(poem)
        assert report.count("METER REFERENCE") == 1

    def test_consolidated_fix_list_entries(self):
        verses = [_verse(0, 0.1, False)]
        poem = PoemResult("baseet", "البسيط", 1, verses, 0.1, False, [("baseet", 0.1)])

        report = generate_poem_correction_report(poem)
        assert "1. Verse 1 [Ṣadr, Foot 1 (ʿArūḍ)]" in report


class TestGeneratePoemCorrectionReportStatusCombinations:
    def test_consolidated_fix_list_missing_and_extra(self):
        missing_foot = FootResult(
            foot_index=0,
            expected_pattern="UU_U_",
            actual_segment="",
            canonical_pattern="UU_U_",
            score=0.0,
            status="missing",
            zihaf_name=None,
            health="severe",
            position_label="Hashw"
        )
        extra_foot = FootResult(
            foot_index=1,
            expected_pattern="",
            actual_segment="UU",
            canonical_pattern="",
            score=0.0,
            status="extra_bits",
            zihaf_name=None,
            health="severe",
            position_label="Extra"
        )
        sadr = HemistichResult("صَدْر", "x", [missing_foot, extra_foot], 0.0, False, [], 1, "UU")
        verse = VerseResult(0, sadr, None, 0.0, "baseet", ["..."])
        poem = PoemResult("baseet", "البسيط", 1, [verse], 0.0, False, [("baseet", 0.0)])
        report = generate_poem_correction_report(poem)
        assert "CONSOLIDATED FIX LIST" in report
        assert "1. Verse 1 [Ṣadr, Foot 1]  ADD text for missing foot «UU_U_»" in report
        assert "2. Verse 1 [Ṣadr]  REMOVE extra «UU»" in report


# ===========================================================================
# Dataclass model basics
# ===========================================================================


class TestDataModel:
    def test_verse_result_issues_default_factory(self):
        sadr = HemistichResult("x", "x", [], 1.0, True, [], 0, None)
        vr = VerseResult(0, sadr, None, 1.0, "baseet")
        assert vr.issues == []

    def test_independent_default_lists(self):
        sadr1 = HemistichResult("a", "x", [], 1.0, True, [], 0, None)
        sadr2 = HemistichResult("b", "x", [], 1.0, True, [], 0, None)
        vr1 = VerseResult(0, sadr1, None, 1.0, "baseet")
        vr2 = VerseResult(1, sadr2, None, 1.0, "baseet")
        vr1.issues.append("only vr1")
        assert vr2.issues == []


class TestStaticTables:
    def test_meter_arabic_names_and_templates_have_same_keys(self):
        assert set(METER_ARABIC_NAMES.keys()) == set(METER_TEMPLATES.keys())

    def test_canonical_patterns_are_binary_strings(self):
        for name, pattern in CANONICAL_PATTERNS.items():
            assert pattern and set(pattern) <= {"0", "1"}, name


# ===========================================================================
# Integration tests requiring the real pyarud library
# ===========================================================================

SADR = "أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا"
AJUZ = "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ"

BROKEN_SADR = "هَذَا بَيْتٌ مَكْسُورٌ تَمَامًا"
BROKEN_AJUZ = "لَا يُطَابِقُ أَيَّ وَزْنٍ مَعْرُوفٍ"


@needs_pyarud
class TestAnalyzePoemIntegration:
    def test_analyze_verse_sound(self):
        vr = analyze_verse(SADR, AJUZ, meter_name="baseet")
        assert vr.meter == "baseet"
        assert vr.combined_score == pytest.approx(1.0)
        assert vr.sadr.is_sound
        assert vr.ajuz.is_sound
        assert vr.issues  # at least the zihāf diagnostics

    def test_analyze_poem_overall_score_and_flags(self):
        poem = analyze_poem([(SADR, AJUZ)], meter_name="baseet")
        assert poem.meter == "baseet"
        assert poem.meter_arabic == METER_ARABIC_NAMES["baseet"]
        assert poem.total_verses == 1
        assert poem.overall_score == pytest.approx(1.0)
        assert poem.is_metrically_sound is True

    def test_analyze_poem_empty_list_raises(self):
        with pytest.raises(ValueError):
            analyze_poem([])

    def test_analyze_poem_broken_verse_is_not_sound(self):
        poem = analyze_poem([(BROKEN_SADR, BROKEN_AJUZ)], meter_name="baseet")
        assert poem.is_metrically_sound is False
        assert poem.overall_score < 1.0
        v = poem.verses[0]
        assert v.sadr.broken_foot_indices

    def test_generate_verse_correction_real_sound(self):
        vr = analyze_verse(SADR, AJUZ, meter_name="baseet")
        report = generate_verse_correction(vr)
        assert "✓ SOUND" in report
        assert "البسيط" in report
        assert SADR in report
        assert AJUZ in report

    def test_generate_verse_correction_real_broken(self):
        vr = analyze_verse(BROKEN_SADR, BROKEN_AJUZ, meter_name="baseet")
        report = generate_verse_correction(vr)
        assert "✗ BROKEN" in report
        assert "DETAILED DIAGNOSIS" in report
        assert "CORRECTION PRESCRIPTION" in report

    def test_analyze_and_report_sound_poem(self, capsys):
        result, report = analyze_and_report(
            [(SADR, AJUZ)], meter_name="baseet", print_summary=True
        )
        captured = capsys.readouterr()
        assert "Accuracy: 100.00%" in captured.out
        assert isinstance(result, dict)
        assert result["meter"] == "baseet"
        assert "All verses are metrically sound" in report

    def test_analyze_and_report_broken_poem(self):
        result, report = analyze_and_report(
            [(BROKEN_SADR, BROKEN_AJUZ)],
            meter_name="baseet",
            print_summary=False,
            only_broken=True,
        )
        assert "CORRECTION PRESCRIPTION" in report
        assert "CONSOLIDATED FIX LIST" in report
        assert result["verses"][0]["combined_score"] < 1.0

    @pytest.mark.parametrize("variant", ["khafif", "الخفيف", "khafeef"])
    def test_to_pyarud_meter_key_accepted_by_analyze_poem(self, variant):
        key = to_pyarud_meter_key(variant)
        # Should not raise: confirms the resolved key is valid for pyarud.
        poem = analyze_poem([(SADR, AJUZ)], meter_name=key)
        assert poem.meter  # any detected/forced meter string is returned


# ===========================================================================
# Regression tests for bugs documented in the Bug Report
# ===========================================================================


class TestEnrichHemistichEmptyFeetVacuousTruth:
    """
    Regression for Bug 1b.

    Python's ``all()`` returns ``True`` for an empty iterable (vacuous truth).
    Before the fix, ``_enrich_hemistich`` with an empty foot list would set
    ``is_sound=True``, silently hiding the fact that pyarud parsed nothing.
    After the fix, an empty foot list must produce ``is_sound=False``.
    """

    def test_empty_feet_is_not_sound(self):
        h = _enrich_hemistich("text", "", [], is_ajuz=False)
        assert h.is_sound is False, (
            "Empty foot list must not be treated as metrically sound "
            "(was a vacuous-truth bug: all(... for f in []) == True)"
        )

    def test_empty_feet_score_is_zero(self):
        h = _enrich_hemistich("text", "", [], is_ajuz=False)
        assert h.score == pytest.approx(0.0)

    def test_empty_feet_broken_indices_empty(self):
        h = _enrich_hemistich("text", "", [], is_ajuz=False)
        assert h.broken_foot_indices == []

    def test_non_empty_all_ok_feet_still_sound(self):
        """Ensure the guard doesn't break the happy path."""
        raw = [
            {"foot_index": 0, "expected_pattern": "1010110",
             "actual_segment": "1010110", "status": "ok", "score": 1.0},
        ]
        h = _enrich_hemistich("text", "1010110", raw, is_ajuz=False)
        assert h.is_sound is True


@needs_pyarud
class TestAnalyzePoemMeterNameResolution:
    """
    Regression for Bug 1a.

    Passing a raw Arabic meter name (e.g. ``'الطويل'``) directly to
    ``analyze_poem`` used to reach pyarud unchanged, causing it to return a
    per-verse ``{"error": "Meter data not found"}`` dict.  The wrapper silently
    swallowed that, producing ``feet=[], is_sound=True, score=0.0`` — a
    misleading false-positive.

    After the fix:
    * Arabic names / alias variants are auto-resolved to the pyarud key before
      the call.
    * Per-verse error dicts from pyarud raise ``ValueError`` immediately.
    * Truly unknown names raise ``ValueError`` via ``to_pyarud_meter_key``.
    """

    # The literal foot-name strings used in the original notebook experiments.
    SALIM_SADR = "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ"
    QABDH_SADR = "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُنْ"
    QABDH_AJUZ = "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُنْ"

    @pytest.mark.parametrize("arabic_variant", ["الطويل", "طويل"])
    def test_arabic_name_auto_resolves_and_returns_real_results(self, arabic_variant):
        """
        Arabic meter names must be silently resolved, not silently swallowed.
        Result must show actual feet (not empty) and a real score (not 0.0).
        """
        poem = analyze_poem(
            [(self.SALIM_SADR, self.SALIM_SADR)],
            meter_name=arabic_variant,
        )
        assert poem.meter == "taweel", (
            f"Expected auto-resolved meter 'taweel', got {poem.meter!r}"
        )
        v = poem.verses[0]
        assert len(v.sadr.feet) > 0, "feet must not be empty after auto-resolution"
        assert v.combined_score > 0.0, "score must not be 0.0 after auto-resolution"

    def test_arabic_name_does_not_produce_false_positive_sound(self):
        """
        The original bug: Arabic name → empty feet → vacuous is_sound=True.
        After both fixes combined, is_metrically_sound must reflect reality.
        """
        poem = analyze_poem(
            [(self.SALIM_SADR, self.SALIM_SADR)],
            meter_name="الطويل",
        )
        # Salim form at Arud position is metrically forbidden (Al-Qabdh rule),
        # so this verse is NOT sound.
        assert poem.is_metrically_sound is False

    def test_truly_unknown_meter_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown meter"):
            analyze_poem(
                [(self.SALIM_SADR, self.SALIM_SADR)],
                meter_name="not-a-real-meter",
            )

    def test_analyze_verse_arabic_name_auto_resolves(self):
        """analyze_verse is a thin wrapper; same resolution must apply."""
        vr = analyze_verse(self.SALIM_SADR, self.SALIM_SADR, meter_name="الطويل")
        assert vr.meter == "taweel"
        assert len(vr.sadr.feet) > 0

    def test_salim_form_at_arud_correctly_flagged_as_broken(self):
        """
        Regression for Bug 2 / Issue 2 (correct engine behaviour, not a code
        bug).  The Salim form ``مَفَاعِيلُنْ`` at the ʿArūḍ position is
        classically forbidden in Al-Taweel; it must be replaced by the Qabdh
        form ``مَفَاعِلُنْ``.  The engine should flag Foot 4 as broken and
        report a score < 1.0.
        """
        poem = analyze_poem(
            [(self.SALIM_SADR, self.SALIM_SADR)],
            meter_name="taweel",
        )
        v = poem.verses[0]
        assert v.combined_score < 1.0, "Salim Arud must score < 1.0"
        assert v.sadr.broken_foot_indices, "Foot 4 (Arud) must be flagged broken"

    def test_qabdh_form_at_arud_scores_perfectly(self):
        """
        Using the classical Qabdh-modified ending (مَفَاعِلُنْ) at the ʿArūḍ
        position should satisfy Al-Taweel and yield a perfect score.
        """
        poem = analyze_poem(
            [(self.QABDH_SADR, self.QABDH_AJUZ)],
            meter_name="taweel",
        )
        v = poem.verses[0]
        assert v.combined_score == pytest.approx(1.0), (
            "Qabdh form at Arud should score 1.0"
        )
        assert not v.sadr.broken_foot_indices, "No broken feet expected"


@pytest.mark.skipif(
    PYARUD_AVAILABLE,
    reason="pyarud is installed; cannot exercise the missing-dependency path",
)
class TestPyarudMissing:
    def test_analyze_poem_raises_clean_runtime_error(self):
        with pytest.raises(RuntimeError, match="pyarud is not installed"):
            analyze_poem([(SADR, AJUZ)])

    def test_analyze_verse_raises_clean_runtime_error(self):
        with pytest.raises(RuntimeError, match="pyarud is not installed"):
            analyze_verse(SADR, AJUZ)