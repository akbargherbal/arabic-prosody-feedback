"""
generate_fake_poem.py
=====================
Generate metrically sound "fake" Arabic poetry using classical foot-name
mnemonics (tafāʿīl) as placeholder words.

Because Arabic foot names scan phonetically as the metrical feet they name
(e.g. مَفَاعِيلُنْ literally scans as Mafaeelon), replacing every content word
in a verse with its corresponding mnemonic produces fully-diacritised, metrically
flawless mock verses — ideal for metrical drills, LLM fine-tuning data, and
prosody verification pipelines.

Public API
----------
    generate_poem(meter, n_verses, zihaf, seed) → list[tuple[str, str]]
    verify_poem(meter_key, verse_list)          → PoemResult | None
    format_poem(meter_key, verse_list, ...)     → str
    list_meters()                               → str
    list_zihafs(meter_key)                      → str

CLI Usage
---------
    python generate_fake_poem.py --meter taweel
    python generate_fake_poem.py --meter baseet --zihaf Khaban --verses 3
    python generate_fake_poem.py --meter rajaz  --zihaf random --seed 42
    python generate_fake_poem.py --list-meters
    python generate_fake_poem.py --list-zihafs --meter taweel
    python generate_fake_poem.py --meter taweel --compact --no-verify

Supported Meters
----------------
    All 16 classical Arabic meters: taweel, madeed, baseet, wafer, kamel,
    hazaj, rajaz, ramal, saree, munsareh, khafeef, mudhare, muqtadheb,
    mujtath, mutakareb, mutadarak.

    Meter names may be passed as pyarud keys (e.g. "taweel"), English aliases
    (e.g. "tawil"), or Arabic (e.g. "طويل") where aliases exist.  All 16 are
    always accessible via their pyarud key.

Supported Zihāfāt (non-terminal positions only)
------------------------------------------------
    Salim (default), random, Khaban, Qabadh, Kaff, Tay, Edmaar, Idmar,
    Waqas, Akal, Asab, Hadhf, Batr, Shakal, Khabal, Kasf, Qatf, Khazal,
    Waqf, Shakl_alt.

Design Guardrails
-----------------
- Terminal Foot Guard: the last foot of every hemistich (ʿArūḍ / Ḍarb) is
  always taken verbatim from the meter template and is never modified.
- Zihaf Validity: _try_apply_zihaf() rejects modifications that are not
  defined for a given foot class and returns the original mnemonic unchanged.
- Double-Application Prevention: zihaf application always re-roots via the
  foot class's Salim form, preventing stacked modifications.
- Bug 1b Regression Guard: verify_poem() calls the patched analyze_poem()
  which guards against empty foot lists silently passing as metrically sound.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Import the standalone prosody module (all data tables + analysis API)
# ---------------------------------------------------------------------------
from arabic_prosody_feedback import (
    CANONICAL_PATTERNS,
    METER_ARABIC_NAMES,
    METER_TEMPLATES,
    PoemResult,
    _PYARUD_AVAILABLE,
    _TAFEELA_MNEMONIC_MAP,
    _ZIHAF_MAP,
    analyze_poem,
    to_pyarud_meter_key,
)


# ===========================================================================
# Phase 1 — Architecture & Internal Mapping Tables
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1  Invert _TAFEELA_MNEMONIC_MAP → arabic_mnemonic: (foot_class, zihaf)
#
# Two-pass builder to handle conflicting duplicates.  Two known collisions:
#   "فَعُولُنْ" → (Fawlon, Salim)    AND  (Mafaelaton, Qatf)   [Qatf collapses to Fawlon]
#   "فَاعِلُنْ" → (Faelon,  Salim)    AND  (Faelaton,  Hadhf)   [Hadhf of Faelaton]
#
# Pass 1: map ALL Salim forms first — they are canonical baselines.
# Pass 2: fill non-Salim forms only where the mnemonic is not yet present.
# This guarantees that secondary modifications never overwrite canonical roots.
# ---------------------------------------------------------------------------

#: Reverse of _TAFEELA_MNEMONIC_MAP: arabic_mnemonic → (foot_class, zihaf_name).
_MNEMONIC_TO_FOOT: dict[str, tuple[str, str]] = {}

# Pass 1 — Salim baselines
for (_fc, _zn), _mn in _TAFEELA_MNEMONIC_MAP.items():
    if _zn == "Salim":
        _MNEMONIC_TO_FOOT[_mn] = (_fc, "Salim")

# Pass 2 — non-Salim forms (only if the mnemonic is not already mapped)
for (_fc, _zn), _mn in _TAFEELA_MNEMONIC_MAP.items():
    if _zn != "Salim" and _mn not in _MNEMONIC_TO_FOOT:
        _MNEMONIC_TO_FOOT[_mn] = (_fc, _zn)


# ---------------------------------------------------------------------------
# Task 1.2  Extract valid zihaf modifications per foot class from _ZIHAF_MAP
#
# Structure: foot_class (str) → frozenset of zihaf name strings
# ---------------------------------------------------------------------------

# Pre-compute binary → foot class for fast reverse lookup.
#
# IMPORTANT: Mustafelon and Mustafe_lon share binary pattern "1010110",
# and Faelaton and Fae_laton share "1011010".  The split variants
# (Mustafe_lon / Fae_laton) are context-specific (Khafif, Muḍāriʿ) and
# are listed *after* the primary classes in CANONICAL_PATTERNS.
# We use first-occurrence-wins so the primary class always wins:
#   "1010110" → "Mustafelon"  (not "Mustafe_lon")
#   "1011010" → "Faelaton"    (not "Fae_laton")
_BINARY_TO_FOOT_CLASS: dict[str, str] = {}
for _fc_b, _pat_b in CANONICAL_PATTERNS.items():
    if _pat_b not in _BINARY_TO_FOOT_CLASS:      # first occurrence wins
        _BINARY_TO_FOOT_CLASS[_pat_b] = _fc_b

#: (foot_class, zihaf_name) pairs that are classically *terminal-only*
#: modifications (ʿilal), never valid on a non-terminal (Hashw) foot.
#:
#: _ZIHAF_MAP is keyed purely by binary-pattern deltas because that table's
#: real job is identifying *which* modification produced a given pattern
#: (used by analyze_poem/identify_zihaf) — it intentionally contains every
#: transformation pyarud's zihaf.py knows how to perform on a foot, terminal
#: or not. The naive builder below used to treat "appears in _ZIHAF_MAP" as
#: "therefore valid in Hashw", which is wrong: pyarud's own Tafeela classes
#: (pyarud/tafeela.py, `allowed_zehafs`) only license a subset of these per
#: class for non-terminal use; the rest are only reachable via a meter's
#: `arod_dharbs_map` (i.e. ʿArūḍ/Ḍarb-only ʿilal). Applying one of these to
#: a Hashw foot silently truncates it, which then shifts the scan offset of
#: every later foot in the hemistich and cascades into spurious failures
#: there too.
#:
#: This list was verified against pyarud 0.1.10's `Tafeela.allowed_zehafs`
#: for every foot class used by the 9 generatively-supported meters (plus
#: Mafoolato, used by Saree/Munsareh/Muqtadheb, for completeness):
#:   Fawlon      : {Qabadh, Thalm, Tharm}        → Hadhf, Batr are ʿilal
#:   Faelaton    : {Khaban, Kaff, Shakal}        → Hadhf, Waqf are ʿilal
#:   Mafaeelon   : {Qabadh, Kaff}                → Hadhf is an ʿilla;
#:                                                  "Shakl_alt" isn't a
#:                                                  recognised Mafaeelon
#:                                                  zihaf at all
#:   Mustafelon  : {Khaban, Tay, Khabal}         → Kasf is an ʿilla here
#:                                                  (it IS valid Hashw for
#:                                                  Mafoolato, hence this is
#:                                                  scoped per *class*, not
#:                                                  a flat zihaf-name ban)
#:   Mafaelaton  : {Asab, Akal, Nakas}           → Qatf is an ʿilla
_HASHW_INVALID_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("Fawlon", "Hadhf"),
    ("Fawlon", "Batr"),
    ("Faelaton", "Hadhf"),
    ("Faelaton", "Waqf"),
    ("Mafaeelon", "Hadhf"),
    ("Mafaeelon", "Shakl_alt"),
    ("Mustafelon", "Kasf"),
    ("Mafaelaton", "Qatf"),
})

#: Meter-specific Hashw exceptions layered *on top of* the generic,
#: foot-class-based `_VALID_HASHW_ZIHAFS` table below.
#:
#: A handful of zihāfāt are legitimate Hashw modifications for a foot class
#: in most meters but are classically excluded for that class in one
#: specific meter — pyarud encodes this directly via each Bahr subclass's
#: `disallowed_zehafs_for_hashw`. Khafeef is the one case among our 9
#: supported meters where this matters:
#:
#:   Khafeef = Faelaton, Mustafe_lon, Faelaton(terminal)
#:
#:   - Foot 0 (Faelaton): Kaff/Shakal are ordinary Faelaton Hashw zihāfāt
#:     elsewhere (e.g. Ramal) but pyarud's Khafeef class explicitly forbids
#:     them in Hashw (`disallowed_zehafs_for_hashw = {0: ([Kaff, Shakal], ...)}`).
#:   - Foot 1: Khafeef's middle foot is actually "Mustafe_lon", a distinct
#:     pyarud Tafeela class that merely *shares* Mustafelon's Salim binary
#:     pattern (1010110). Mustafe_lon's real allowed zihafs are {Khaban,
#:     Kaff, Tay, Shakal} — notably NOT Khabal, unlike genuine Mustafelon
#:     (used in Rajaz/Baseet/Saree). Since this module's binary→foot-class
#:     lookup collapses both classes into "Mustafelon" (see comment above),
#:     Khabal must be vetoed here specifically for Khafeef's foot 1.
#:
#: Keyed by meter_key → {non-terminal foot index → zihafs to additionally
#: forbid at that exact position}.
_METER_HASHW_OVERRIDES: dict[str, dict[int, frozenset[str]]] = {
    "khafeef": {
        0: frozenset({"Kaff", "Shakal"}),
        1: frozenset({"Khabal"}),
    },
}

#: Maps foot_class → set of zihaf names applicable to that foot class.
#: Derived from _ZIHAF_MAP keys, excluding terminal-only ʿilal — see
#: _HASHW_INVALID_PAIRS above. (Meter-specific exceptions on top of this
#: are handled separately via _METER_HASHW_OVERRIDES, since the same foot
#: class can be Hashw-valid for a zihaf in one meter and not another.)
_VALID_HASHW_ZIHAFS: dict[str, set[str]] = {}

for (_can_bin, _mod_bin), _zn in _ZIHAF_MAP.items():
    _fc = _BINARY_TO_FOOT_CLASS.get(_can_bin)
    if _fc and (_fc, _zn) not in _HASHW_INVALID_PAIRS:
        _VALID_HASHW_ZIHAFS.setdefault(_fc, set()).add(_zn)


def _valid_hashw_zihafs(foot_class: str) -> set[str]:
    """
    Return the set of applicable zihāf names for the given foot class.

    Examples
    --------
    >>> "Khaban" in _valid_hashw_zihafs("Mustafelon")
    True
    >>> "Qabadh" in _valid_hashw_zihafs("Fawlon")
    True
    """
    return _VALID_HASHW_ZIHAFS.get(foot_class, set())


# ===========================================================================
# Phase 2 — Core Generation Engine
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1  _try_apply_zihaf(word, zihaf)
# ---------------------------------------------------------------------------


def _try_apply_zihaf(
    word: str,
    zihaf: str,
    *,
    extra_invalid: Optional[frozenset[str]] = None,
) -> str:
    """
    Apply a zihāf modification to an Arabic mnemonic foot name.

    Algorithm
    ---------
    1. Look up ``word`` in ``_MNEMONIC_TO_FOOT`` to identify ``foot_class``.
    2. Return ``word`` unmodified if the foot class has no known Salim form
       (treats the word as a terminal/compound that cannot be modified).
    3. Return ``word`` unmodified if ``zihaf`` is not listed in
       ``_VALID_HASHW_ZIHAFS[foot_class]``, or is vetoed by ``extra_invalid``
       (rejects invalid modifications — see ``extra_invalid`` below).
    4. Re-resolve from the Salim root via ``(foot_class, zihaf)`` — this
       prevents double-application when ``word`` is already a modified form.
    5. Return the mnemonic for ``(foot_class, zihaf)``.

    Parameters
    ----------
    word:
        Diacritised Arabic mnemonic, e.g. ``"فَعُولُنْ"``.
    zihaf:
        Target modification name, e.g. ``"Qabadh"`` or ``"Salim"``.
    extra_invalid:
        Optional set of zihaf names to reject *in addition to* whatever
        ``_VALID_HASHW_ZIHAFS[foot_class]`` already excludes. Used by
        :func:`_build_hemistich` to layer a meter-specific Hashw exception
        (``_METER_HASHW_OVERRIDES``) on top of the generic, foot-class-level
        validity table — the same foot class can be Hashw-valid for a given
        zihaf in one meter and not another (e.g. Khafeef vs. Ramal/Rajaz).

    Returns
    -------
    str
        Modified mnemonic if the transformation is valid; ``word`` otherwise.

    Examples
    --------
    >>> _try_apply_zihaf("فَعُولُنْ", "Qabadh")
    'فَعُولُ'
    >>> _try_apply_zihaf("مَفَاعِيلُنْ", "Qabadh")
    'مَفَاعِلُنْ'
    >>> _try_apply_zihaf("مَفَاعِلُ", "Qabadh")   # terminal form — not in map
    'مَفَاعِلُ'
    """
    if word not in _MNEMONIC_TO_FOOT:
        # Unknown mnemonic (e.g. a terminal form not listed in the map) — pass through
        return word

    foot_class, _ = _MNEMONIC_TO_FOOT[word]

    # Guard: ensure a Salim root exists for this foot class
    salim_mnemonic = _TAFEELA_MNEMONIC_MAP.get((foot_class, "Salim"))
    if salim_mnemonic is None:
        return word  # No Salim root — treat as terminal; do not modify

    # "Salim" always resolves to the canonical Salim form
    if zihaf == "Salim":
        return salim_mnemonic

    # Reject modifications not defined for this foot class, or vetoed for
    # this specific meter/position by the caller.
    valid = _VALID_HASHW_ZIHAFS.get(foot_class, set())
    if zihaf not in valid or (extra_invalid and zihaf in extra_invalid):
        return word  # Invalid zihaf for this foot class (or this meter/position)

    # Re-resolve from the foot class root to prevent double-application
    target = _TAFEELA_MNEMONIC_MAP.get((foot_class, zihaf))
    if target is None:
        return word  # No mnemonic for this (foot_class, zihaf) pair

    return target


# ---------------------------------------------------------------------------
# Task 2.2  _build_hemistich(meter_key, zihaf, rng)
# ---------------------------------------------------------------------------


def _build_hemistich(meter_key: str, zihaf: str, rng: random.Random) -> str:
    """
    Build one hemistich by applying ``zihaf`` to all non-terminal feet.

    Terminal Foot Guard
    -------------------
    The last foot (index ``n-1``) is *always* kept verbatim from the meter
    template.  The template already encodes the metrically mandatory form for
    the ʿArūḍ / Ḍarb position (e.g. for al-Taweel, the terminal is
    ``مَفَاعِلُ`` — a Qabdh-terminal — not the Salim ``مَفَاعِيلُنْ``).
    Overriding it would break the meter even when applying a globally valid
    zihāf such as ``"Salim"``.

    Random Mode
    -----------
    When ``zihaf == "random"``, each non-terminal foot independently draws
    from the intersection of:
      - classical modifications defined in ``_ZIHAF_MAP`` for that foot class
      - modifications that have a known mnemonic in ``_TAFEELA_MNEMONIC_MAP``
      - "Salim" (always included)

    Parameters
    ----------
    meter_key:
        pyarud meter key, e.g. ``"taweel"``.
    zihaf:
        ``"Salim"``, a named modification (e.g. ``"Qabadh"``), or
        ``"random"`` for per-foot randomisation.
    rng:
        Seeded :class:`random.Random` for reproducible output.

    Returns
    -------
    str
        Space-joined diacritised hemistich string.
    """
    template = METER_TEMPLATES.get(meter_key, "")
    feet = template.split()

    if not feet:
        return ""

    result: list[str] = []
    n = len(feet)

    for i, foot in enumerate(feet):
        # Terminal foot guard — always reproduce the template's own terminal form
        if i == n - 1:
            result.append(foot)
            continue

        # Meter-specific Hashw exception for this exact position, if any
        # (see _METER_HASHW_OVERRIDES) — layered on top of the generic,
        # foot-class-level _VALID_HASHW_ZIHAFS table.
        overrides = _METER_HASHW_OVERRIDES.get(meter_key, {}).get(i)

        if zihaf == "random":
            foot_info = _MNEMONIC_TO_FOOT.get(foot)
            if foot_info is None:
                result.append(foot)
                continue

            foot_class, _ = foot_info
            valid = _VALID_HASHW_ZIHAFS.get(foot_class, set())
            if overrides:
                valid = valid - overrides

            # Candidate list: "Salim" always included; add mods that have a mnemonic
            candidates: list[str] = ["Salim"]
            for mod in sorted(valid):          # sorted → reproducible given same seed
                if (foot_class, mod) in _TAFEELA_MNEMONIC_MAP:
                    candidates.append(mod)

            chosen = rng.choice(candidates)
            result.append(_try_apply_zihaf(foot, chosen, extra_invalid=overrides))
        else:
            result.append(_try_apply_zihaf(foot, zihaf, extra_invalid=overrides))

    return " ".join(result)


# ---------------------------------------------------------------------------
# Task 2.3  Public entry point: generate_poem
# ---------------------------------------------------------------------------

#: All zihaf names recognised by the CLI --zihaf option.
ALL_KNOWN_ZIHAFS: tuple[str, ...] = (
    "Salim",
    "random",
    "Khaban",
    "Qabadh",
    "Kaff",
    "Tay",
    "Edmaar",
    "Idmar",
    "Waqas",
    "Akal",
    "Asab",
    "Hadhf",
    "Batr",
    "Shakal",
    "Khabal",
    "Kasf",
    "Qatf",
    "Khazal",
    "Waqf",
    "Shakl_alt",
)


def generate_poem(
    meter: str,
    n_verses: int = 4,
    zihaf: str = "Salim",
    seed: Optional[int] = None,
) -> list[tuple[str, str]]:
    """
    Generate ``n_verses`` of fake Arabic poetry in the specified meter.

    Every word in every verse is the Arabic mnemonic (tafʿīla name) for the
    prosodic foot at that position.  Because foot names scan phonetically as
    the feet they name, the output is metrically flawless by construction.

    Parameters
    ----------
    meter:
        Meter name in any supported form — pyarud key (``"taweel"``),
        English alias (``"tawil"``), or Arabic (``"طويل"``).
        Use :func:`list_meters` to enumerate all 16 supported meters.
    n_verses:
        Number of verses (bayts) to generate.  Default ``4``.
    zihaf:
        Modification applied to all non-terminal (*Hashw*) foot positions:

        - ``"Salim"`` (default) — canonical, unmodified forms.
        - A named zihāf (``"Khaban"``, ``"Qabadh"``, ``"Kaff"``, etc.) —
          applied where valid for the foot class; feet where it is invalid
          are silently left as Salim.
        - ``"random"`` — independently randomise each non-terminal foot
          per verse, drawing from that foot's valid modification set.

    seed:
        Integer seed for the ``"random"`` zihāf mode.  ``None`` →
        non-deterministic.  Has no effect for named or ``"Salim"`` modes.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(sadr, ʿajuz)`` pairs of fully-diacritised Arabic strings.

    Raises
    ------
    ValueError
        If ``meter`` is not recognised or has no template.

    Examples
    --------
    >>> verses = generate_poem("taweel", n_verses=2, zihaf="Qabadh")
    >>> len(verses)
    2
    >>> isinstance(verses[0], tuple)
    True
    >>> print(verses[0][0])   # Ṣadr
    فَعُولُ مَفَاعِلُنْ فَعُولُ مَفَاعِلُ
    """
    # Resolve any meter name variant → pyarud key (raises ValueError if unknown)
    meter_key = to_pyarud_meter_key(meter)
    if meter_key is None or meter_key not in METER_TEMPLATES:
        raise ValueError(
            f"Meter {meter!r} (resolved: {meter_key!r}) has no hemistich template.\n"
            f"Valid meter keys: {sorted(METER_TEMPLATES)}"
        )

    rng = random.Random(seed)
    verses: list[tuple[str, str]] = []

    for _ in range(n_verses):
        sadr = _build_hemistich(meter_key, zihaf, rng)
        ajuz = _build_hemistich(meter_key, zihaf, rng)
        verses.append((sadr, ajuz))

    return verses


# ===========================================================================
# Phase 3 — Reporting, Verification & CLI Shell
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1  verify_poem
# ---------------------------------------------------------------------------


def verify_poem(
    meter_key: str,
    verse_list: list[tuple[str, str]],
) -> Optional[PoemResult]:
    """
    Run metrical verification via :func:`~arabic_prosody_feedback.analyze_poem`.

    Calls the real-time analyzer from ``arabic_prosody_feedback``.  All verses
    must report genuine accuracy scores — the Bug 1b patch in
    ``_enrich_hemistich`` ensures empty foot lists (which arise when pyarud
    silently rejects an unrecognised meter key) never pass as sound.

    Parameters
    ----------
    meter_key:
        pyarud key, e.g. ``"taweel"``.
    verse_list:
        ``(sadr, ʿajuz)`` pairs as returned by :func:`generate_poem`.

    Returns
    -------
    PoemResult or None
        ``None`` is returned gracefully when **pyarud** is not installed, so
        callers do not need to guard for the import separately.
    """
    if not _PYARUD_AVAILABLE:
        return None

    try:
        return analyze_poem(verse_list, meter_name=meter_key)
    except Exception as exc:                               # pragma: no cover
        print(f"⚠  Verification warning: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Task 3.2  format_poem — styled report and compact output
# ---------------------------------------------------------------------------


def _score_tag(score: float) -> str:
    """Map a combined_score float to a short human-readable tag."""
    if score >= 1.0:
        return "✓ SOUND"
    if score >= 0.90:
        return "~ NEAR-PERFECT"
    if score >= 0.70:
        return "⚠ IRREGULAR"
    return "✗ BROKEN"


def format_poem(
    meter_key: str,
    verse_list: list[tuple[str, str]],
    zihaf: str = "Salim",
    verification: Optional[PoemResult] = None,
    compact: bool = False,
) -> str:
    """
    Format a generated poem for display or pipe consumption.

    Parameters
    ----------
    meter_key:
        pyarud key used to retrieve the Arabic name and template.
    verse_list:
        ``(sadr, ʿajuz)`` pairs from :func:`generate_poem`.
    zihaf:
        The modification used during generation (shown in the header).
    verification:
        Optional :class:`PoemResult` from :func:`verify_poem`.  When
        provided, per-verse accuracy tags and an overall score are added.
    compact:
        ``True`` → bare ``sadr *** ajuz`` lines, pipe-/LLM-friendly.
        ``False`` (default) → styled, verbose report.

    Returns
    -------
    str
    """
    if compact:
        return "\n".join(f"{s} *** {a}" for s, a in verse_list)

    # ── Styled verbose report ──────────────────────────────────────────────
    meter_ar = METER_ARABIC_NAMES.get(meter_key, meter_key)
    template  = METER_TEMPLATES.get(meter_key, "")
    W = "═" * 72

    lines: list[str] = []

    # Header box
    lines.append("╔" + W + "╗")
    lines.append(f"║  FAKE ARABIC POEM  ·  {meter_ar}  ({meter_key})")
    lines.append(f"║  Zihāf : {zihaf:<20}  Verses : {len(verse_list)}")
    if template:
        lines.append(f"║  Template : {template}")

    # Verification availability notice
    if verification is not None:
        pct = verification.overall_score * 100
        sound_tag = (
            "✓ METRICALLY SOUND"
            if verification.is_metrically_sound
            else "⚠ ISSUES DETECTED"
        )
        lines.append(f"║  Verification : {sound_tag}  ·  Overall {pct:.1f}%")
    elif not _PYARUD_AVAILABLE:
        lines.append("║  Verification : unavailable  (pip install pyarud)")
    else:
        lines.append("║  Verification : skipped  (pass --no-verify to suppress)")

    lines.append("╚" + W + "╝")
    lines.append("")

    # ── Per-verse section ──────────────────────────────────────────────────
    for idx, (sadr, ajuz) in enumerate(verse_list, 1):
        ver_fragment = ""
        if verification and idx - 1 < len(verification.verses):
            vr  = verification.verses[idx - 1]
            pct = vr.combined_score * 100
            ver_fragment = f"  [{_score_tag(vr.combined_score)}  {pct:.0f}%]"

        lines.append(f"  ── Verse {idx}{ver_fragment}")
        lines.append(f"  Ṣadr  : {sadr}")
        lines.append(f"  ʿAjuz : {ajuz}")
        lines.append("")

    lines.append("─" * 74)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_meters() -> str:
    """
    Return a formatted table of all 16 supported meters.

    Columns: pyarud key · Arabic name · hemistich template.
    """
    COL = 16
    lines = [
        "  Supported Classical Arabic Meters",
        "  " + "─" * 76,
        f"  {'Key':<{COL}}  {'Arabic':<12}  Template",
        "  " + "─" * 76,
    ]
    for key in sorted(METER_ARABIC_NAMES):
        arabic   = METER_ARABIC_NAMES.get(key, "")
        template = METER_TEMPLATES.get(key, "(no template)")
        lines.append(f"  {key:<{COL}}  {arabic:<12}  {template}")
    lines.append("  " + "─" * 76)
    lines.append(
        "  Tip: pass any key above as --meter, or use aliases where available\n"
        "       (e.g. tawil / الطويل → taweel,  basit / البسيط → baseet)."
    )
    return "\n".join(lines)


def list_zihafs(meter_key: str) -> str:
    """
    Return a formatted list of valid Hashw zihāfāt for ``meter_key``.

    Shows, per unique foot class appearing in the meter's non-terminal
    positions, the Salim form and all applicable named modifications together
    with their Arabic mnemonics.
    """
    template = METER_TEMPLATES.get(meter_key, "")
    if not template:
        return f"  No template found for meter {meter_key!r}."

    meter_ar = METER_ARABIC_NAMES.get(meter_key, meter_key)
    feet      = template.split()

    lines = [
        f"  Hashw Zihāfāt available for  {meter_ar}  ({meter_key})",
        f"  Template : {template}",
        "  " + "─" * 76,
        f"  {'Foot class':<18}  {'Salim form':<20}  Available modifications",
        "  " + "─" * 76,
    ]

    seen_classes: set[str] = set()
    for i, foot in enumerate(feet[:-1]):        # skip terminal foot
        info = _MNEMONIC_TO_FOOT.get(foot)
        if info is None:
            if foot not in seen_classes:         # de-duplicate unknown feet
                lines.append(f"  {'(unknown)':<18}  {foot:<20}  —")
                seen_classes.add(foot)
            continue

        foot_class, _ = info
        if foot_class in seen_classes:
            continue
        seen_classes.add(foot_class)

        salim   = _TAFEELA_MNEMONIC_MAP.get((foot_class, "Salim"), foot)
        valid   = _VALID_HASHW_ZIHAFS.get(foot_class, set())
        overrides = _METER_HASHW_OVERRIDES.get(meter_key, {}).get(i)
        if overrides:
            valid = valid - overrides

        mod_parts: list[str] = []
        for mod in sorted(valid):
            mnemonic = _TAFEELA_MNEMONIC_MAP.get((foot_class, mod))
            if mnemonic:
                mod_parts.append(f"{mod} ({mnemonic})")

        mod_str = ",  ".join(mod_parts) if mod_parts else "(none)"
        lines.append(f"  {foot_class:<18}  {salim:<20}  {mod_str}")

    lines.append("  " + "─" * 76)
    lines.append("  Terminal foot is ALWAYS fixed — it is never modified.")
    lines.append(
        f"  Terminal : {feet[-1]}"
        if feet
        else ""
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 3.3  argparse CLI shell
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_fake_poem",
        description=(
            "Generate metrically sound fake Arabic poetry.\n"
            "Every word is the tafʿīla mnemonic for the foot it occupies,\n"
            "producing fully-diacritised, metrically flawless mock verses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_fake_poem.py --meter taweel\n"
            "  python generate_fake_poem.py --meter baseet --zihaf Khaban --verses 3\n"
            "  python generate_fake_poem.py --meter rajaz  --zihaf random  --seed 42\n"
            "  python generate_fake_poem.py --list-meters\n"
            "  python generate_fake_poem.py --list-zihafs  --meter taweel\n"
            "  python generate_fake_poem.py --meter kamel  --compact --no-verify"
        ),
    )

    p.add_argument(
        "-m", "--meter",
        metavar="METER",
        help=(
            "Meter name — pyarud key (taweel, baseet, …), English alias "
            "(tawil, basit, …), or Arabic where aliases exist.  "
            "Required unless --list-meters is used."
        ),
    )
    p.add_argument(
        "-n", "--verses",
        type=int,
        default=4,
        metavar="N",
        help="Number of verses to generate (default: 4).",
    )
    p.add_argument(
        "-z", "--zihaf",
        default="Salim",
        metavar="ZIHAF",
        help=(
            "Zihāf applied to non-terminal (Hashw) feet.\n"
            "  Salim   — canonical, pristine forms (default)\n"
            "  random  — randomise per foot; use --seed for reproducibility\n"
            "  named   — Khaban, Qabadh, Kaff, Tay, Edmaar, Waqas, Akal,\n"
            "            Asab, Hadhf, Batr, Shakal, Khabal, Kasf, Qatf, …\n"
            "Use --list-zihafs --meter METER to see valid options per meter."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Integer random seed for --zihaf random mode (default: non-deterministic).",
    )
    p.add_argument(
        "--list-meters",
        action="store_true",
        help="Print a table of all 16 supported classical meters and exit.",
    )
    p.add_argument(
        "--list-zihafs",
        action="store_true",
        help="Print valid zihāfāt for --meter and exit.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip real-time metrical verification.  Faster; pyarud not required. "
            "Verification is silently skipped anyway when pyarud is absent."
        ),
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Bare output mode: one «sadr *** ajuz» line per verse.  "
            "Pipe-friendly; disables headers and verification tags."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point.  Returns an exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── --list-meters ────────────────────────────────────────────────────────
    if args.list_meters:
        print(list_meters())
        return 0

    # ── --list-zihafs ────────────────────────────────────────────────────────
    if args.list_zihafs:
        if not args.meter:
            parser.error("--list-zihafs requires --meter METER")
        try:
            meter_key = to_pyarud_meter_key(args.meter)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if meter_key is None:
            print("Error: could not resolve meter name.", file=sys.stderr)
            return 1
        print(list_zihafs(meter_key))
        return 0

    # ── Poem generation ──────────────────────────────────────────────────────
    if not args.meter:
        parser.error(
            "--meter METER is required for poem generation.  "
            "Use --list-meters to see available meters."
        )

    # Resolve meter name → pyarud key
    try:
        meter_key = to_pyarud_meter_key(args.meter)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if meter_key is None:
        print("Error: could not resolve meter name.", file=sys.stderr)
        return 1

    # Generate
    try:
        verses = generate_poem(
            meter    = meter_key,
            n_verses = args.verses,
            zihaf    = args.zihaf,
            seed     = args.seed,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Verify (unless suppressed)
    verification: Optional[PoemResult] = None
    if not args.no_verify and not args.compact:
        verification = verify_poem(meter_key, verses)

    # Format & print
    print(
        format_poem(
            meter_key    = meter_key,
            verse_list   = verses,
            zihaf        = args.zihaf,
            verification = verification,
            compact      = args.compact,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
