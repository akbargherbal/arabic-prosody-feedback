# Arabic Prosody Feedback (`arudy_feedback`)

An engine for generating structured, highly detailed, and LLM-actionable metrical correction feedback for Arabic poetry.

This module acts as a correction layer on top of **pyarud**. Rather than just reporting accuracy percentages, it isolates prosodic deviations down to the syllable level and translates them into explicit correction prescriptions (e.g., adding, removing, or reweighting morae). This text-based feedback is specifically designed to be piped directly back into Large Language Models (LLMs) to guide iterative, multi-turn poetic generation and correction.

---

## Features

- **Syallable-Level Alignment Grid:** Displays expected vs. actual patterns side-by-side using standardized prosodic notation (`U` for short/mutaharrik, `_` for long/sākin) mapped directly to classical Arabic Taf'īla mnemonics (such as `مُسْتَفْعِلُنْ`, `فَاعِلُنْ`, etc.).
- **Character-Level Diffs:** Pinpoints the exact position of metrical errors using precise visual markers:
  - `|` : Match
  - `×` : Syllable weight mismatch (long vs. short)
  - `^` : Missing mora (under-weight)
  - `v` : Extra mora (over-weight)
- **Concrete Correction Prescriptions:** Generates explicit, numbered instructions detailing how many morae need to be added, trimmed, or reweighted for each faulty foot.
- **Poem-Level Consolidated Reports:** Summarizes the health of a whole poem, omitting perfect verses by default while generating detailed correction grids for broken lines.
- **Fully Standalone Module:** Integrates all required helper data structures, static lookup tables (such as Zihāf maps and templates for 16 classical meters), and metrics calculations. Only requires the external package `pyarud`.

---

## Installation

This module is designed to run on Python 3.12+.

1. Clone or copy this repository into your project directory.
2. Install the necessary dependency, `pyarud`:
   ```bash
   pip install pyarud
   ```

---

## Quick Start

The simplest way to analyze a poem and generate an LLM-ready correction report is via the convenience function `analyze_and_report`.

```python
from arabic_prosody_feedback import analyze_and_report

# Provide a list of (sadr, ajuz) tuples representing fully-diacritized Arabic verses
verses = [
    (
        "أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا",
        "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ"
    )
]

# Analyze using a specific meter (supports English, Arabic, or transliterated names)
result_dict, correction_report = analyze_and_report(verses, meter_name="baseet")

print(correction_report)
```

---

## Detailed Report Output Example

When a verse contains metrical issues, the generated output looks like this:

```text
==================================================================
  VERSE 1  ·  ✗ BROKEN  ·  Score: 78%
==================================================================
  Meter : البسيط (baseet)
  Ṣadr  : أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا
  ʿAjuz : وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ

  ┌─ ṢADR (صَدْر) ────────────────────────────────────────────────────────┐
  │  [_ _ U _]  [U U _]  [_ _ U _]  [U U _]   ← Expected
  │  [_ U U _]  [U U _]  [_ _ U _]  [U U _]   ← Actual
  │  [✗ BROKEN]  [  ✓  ]  [  ✓  ]  [  ✓  ]
  │  [ BROKEN ]  [فَعِلُنْ]  [مُسْتَفْعِلُنْ]  [فَعِلُنْ]
  │  Morae: expected 14, actual 14
  └──────────────────────────────────────────────────────────────┘

  ── DETAILED DIAGNOSIS ──────────────────────────────────────────

  ▸ [Ṣadr  |  Foot 1  |  Hashw]
    Expected : __U_  (4 morae)
    Actual   : _UU_  (4 morae)

    Expected:  _ _ U _
    Actual:    _ U U _
    Diff:      | × | |    (| match  × wrong weight  ^ missing  v extra)

    → Wrong syllable weight(s): pos 2: short (U) → long (_).

  ── CORRECTION PRESCRIPTION ─────────────────────────────────────
  1. [Ṣadr, Foot 1 (Hashw)]  Adjust syllable weights in «_UU_» to match «__U_» (same length, wrong weight pattern).

  ── METER REFERENCE ─────────────────────────────────────────────
  البحر  : البسيط (baseet)
  Tafāʿīl: مُسْتَفْعِلُنْ فَاعِلُنْ مُسْتَفْعِلُنْ فَعِلُ
  Ṣadr   : _ _ U _ | U U _ | _ _ U _ | U U _
  ʿAjuz  : _ _ U _ | U U _ | _ _ U _ | U U _

==================================================================
```

---

## API Reference

The engine exposes several high-level analysis and reporting functions:

### 1. `analyze_and_report`

```python
def analyze_and_report(
    verses: list[tuple[str, str]],
    meter_name: str | None = None,
    *,
    only_broken: bool = True,
    score_threshold: float = 0.99,
    print_summary: bool = True,
) -> tuple[dict, str]:
```

Runs a complete analysis and outputs a nested results dictionary and a multi-line string report. Ideal for single-call processing.

### 2. `analyze_poem`

```python
def analyze_poem(
    verses: list[tuple[str, str]],
    *,
    meter_name: str | None = None,
    top_n: int = 3,
) -> PoemResult:
```

Processes an entire poem using `pyarud` and transforms the output into an enriched, structured `PoemResult` data model container.

### 3. `generate_poem_correction_report`

```python
def generate_poem_correction_report(
    poem: PoemResult,
    *,
    only_broken: bool = True,
    score_threshold: float = 0.99,
    include_meter_schema: bool = True,
) -> str:
```

Compiles a consolidated, top-down report of all verses in a poem, with a unified summary list of fixes.

### 4. `generate_verse_correction`

```python
def generate_verse_correction(
    verse: VerseResult,
    *,
    include_meter_schema: bool = True,
) -> str:
```

Generates the side-by-side alignment grid, detailed character-level diff, and numbered prescriptions for a single verse.

---

## Run Tests

A complete test suite is provided in `test_arabic_prosody_feedback.py` covering static table sanity, data structures, pattern conversions, zihāf identification, and integration paths.

To run the test suite:

```bash
pytest test_arabic_prosody_feedback.py -v
```
