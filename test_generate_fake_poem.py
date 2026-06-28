# test_generate_fake_poem.py
"""
Test suite for generate_fake_poem.py and arabic_prosody_feedback.py.

This suite implements parameterized and combinatorial testing patterns to verify
that fake poem generation produces structurally correct and metrically sound
classical Arabic poetry ONLY for the 9 meters defined in the _ALIASES mapping.

Source Acknowledgment:
---------------------
- Dynamic pair generation relies on `_MNEMONIC_TO_FOOT` and `_VALID_HASHW_ZIHAFS`
  defined in Phase 1 of `generate_fake_poem.py`.
- Structural templates are sourced from `METER_TEMPLATES` and filtered using
  `_ALIASES` and `to_pyarud_meter_key` inside `arabic_prosody_feedback.py`.
"""

from __future__ import annotations

import pytest
from generate_fake_poem import (
    generate_poem,
    verify_poem,
    _MNEMONIC_TO_FOOT,
    _VALID_HASHW_ZIHAFS,
)
from arabic_prosody_feedback import (
    METER_TEMPLATES,
    _ALIASES,
    _PYARUD_AVAILABLE,
    _TAFEELA_MNEMONIC_MAP,
    to_pyarud_meter_key,
)


def get_supported_pyarud_keys() -> set[str]:
    """
    Extracts the unique targets from _ALIASES and resolves them
    to their standard pyarud-native meter keys.
    """
    supported_keys: set[str] = set()
    # Collect unique targets in _ALIASES (e.g. 'tawil', 'basit', etc.)
    unique_alias_targets = set(_ALIASES.values())

    for target in unique_alias_targets:
        resolved = to_pyarud_meter_key(target)
        if resolved:
            supported_keys.add(resolved)

    return supported_keys


def get_valid_meter_zihaf_pairs() -> list[tuple[str, str]]:
    """
    Programmatically builds a list of valid (meter_key, zihaf_name) pairs,
    restricted ONLY to the meters present in _ALIASES.
    """
    pairs: list[tuple[str, str]] = []
    supported_meters = get_supported_pyarud_keys()

    for meter_key, template in METER_TEMPLATES.items():
        # Restrict the test run to only the selected 9 meters
        if meter_key not in supported_meters:
            continue

        # Salim is the pristine default and always valid for all meters
        pairs.append((meter_key, "Salim"))

        feet = template.split()
        if not feet:
            continue

        # Non-terminal (Hashw) feet exclude the final foot of the hemistich
        for foot in feet[:-1]:
            foot_info = _MNEMONIC_TO_FOOT.get(foot)
            if foot_info is None:
                continue

            foot_class, _ = foot_info
            valid_zihafs = _VALID_HASHW_ZIHAFS.get(foot_class, set())
            for zihaf in valid_zihafs:
                # Confirm a mnemonic mapping exists for this specific combination
                if (foot_class, zihaf) in _TAFEELA_MNEMONIC_MAP:
                    pairs.append((meter_key, zihaf))

    # De-duplicate and sort to guarantee a deterministic test execution order
    return sorted(list(set(pairs)))


def assert_offline_structural_integrity(meter_key: str, verses: list[tuple[str, str]]):
    """
    Performs static structural assertions that do not depend on the external
    pyarud library. This ensures test coverage even in local or CI pipelines
    where pyarud is not installed.
    """
    template = METER_TEMPLATES[meter_key]
    expected_feet = template.split()
    n_expected_words = len(expected_feet)

    for sadr, ajuz in verses:
        sadr_words = sadr.split()
        ajuz_words = ajuz.split()

        # 1. Word Count Preservation: Hemistiches must preserve the template's word count
        assert len(sadr_words) == n_expected_words, (
            f"Word count mismatch in Ṣadr for {meter_key}. "
            f"Expected {n_expected_words}, got {len(sadr_words)}: {sadr_words}"
        )
        assert len(ajuz_words) == n_expected_words, (
            f"Word count mismatch in ʿAjuz for {meter_key}. "
            f"Expected {n_expected_words}, got {len(ajuz_words)}: {ajuz_words}"
        )

        # 2. Terminal Foot Guard: The last foot (ʿArūḍ / Ḍarb) must remain entirely unmodified
        assert sadr_words[-1] == expected_feet[-1], (
            f"Ṣadr terminal foot was modified in {meter_key}. "
            f"Expected '{expected_feet[-1]}', got '{sadr_words[-1]}'"
        )
        assert ajuz_words[-1] == expected_feet[-1], (
            f"ʿAjuz terminal foot was modified in {meter_key}. "
            f"Expected '{expected_feet[-1]}', got '{ajuz_words[-1]}'"
        )


# ===========================================================================
# Test Cases
# ===========================================================================


@pytest.mark.parametrize("meter_key, zihaf", get_valid_meter_zihaf_pairs())
def test_valid_zihafs_across_restricted_meters(meter_key: str, zihaf: str):
    """
    Test that every valid non-terminal zihāf applied to the restricted set of 9
    classical meters produces structurally sound verses. If pyarud is installed,
    it also performs full runtime metrical verification.
    """
    # Generate 2 verses using a fixed seed
    verses = generate_poem(meter=meter_key, n_verses=2, zihaf=zihaf, seed=42)

    # 1. Fallback / Offline checks (always runs)
    assert len(verses) == 2, "Expected exactly 2 verses to be generated"
    assert_offline_structural_integrity(meter_key, verses)

    # 2. Real-time metrical analysis (runs only when pyarud is present)
    if _PYARUD_AVAILABLE:
        result = verify_poem(meter_key, verses)
        assert result is not None, f"Metrical feedback failed to analyze {meter_key}"
        assert result.is_metrically_sound, (
            f"Unsound verses generated for {meter_key} with {zihaf}.\n"
            f"Score: {result.overall_score * 100:.1f}%\n"
            f"Verses:\n"
            f"  Ṣadr : {verses[0][0]}\n"
            f"  ʿAjuz: {verses[0][1]}"
        )


def test_random_zihaf_determinism_and_integrity():
    """
    Test that the random zihāf generation mode operates deterministically
    when provided a seed, and produces metrically sound verses.
    """
    meter_key = "taweel"  # Included in the restricted 9 meters (Tawil)

    # Generate two separate pools using the identical seed
    verses_run_1 = generate_poem(meter=meter_key, n_verses=3, zihaf="random", seed=101)
    verses_run_2 = generate_poem(meter=meter_key, n_verses=3, zihaf="random", seed=101)

    # 1. Determinism assertion
    assert (
        verses_run_1 == verses_run_2
    ), "Randomized generation with the same seed must be deterministic"

    # 2. Structural verification
    assert_offline_structural_integrity(meter_key, verses_run_1)

    # 3. Metrical verification
    if _PYARUD_AVAILABLE:
        result = verify_poem(meter_key, verses_run_1)
        assert result is not None
        assert result.is_metrically_sound, (
            f"Random zihāf generation failed metrical check.\n"
            f"Score: {result.overall_score * 100:.1f}%"
        )
