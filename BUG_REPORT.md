# ⚑ Bug Report: Generative Metrical Alignment Drift & Terminal 'Ilal Leakage in `generate_fake_poem`

- **Date:** June 28, 2026
- **Environment:** Windows 32-bit (`win32`), Python 3.12.2, `pytest-8.4.2`
- **Target Module:** `generate_fake_poem.py` (integrated with `arabic_prosody_feedback.py`)
- **Failure Rate:** 14 / 190 items failed (~7.37%) [4]

---

## 1. Summary of Failures

The test suite encountered 14 failures during the execution of metrical verification tests on the 9 restricted classical meters [4]. The failures fall into two distinct mechanical categories:

1. **Terminal-Only modifications (_ʿIlal_) applied to non-terminal (_Hashw_) feet:**
   - **Meters affected:** `baseet` (Kasf) [4], `khafeef` (Hadhf, Kasf, Khabal) [4], `mutakareb` (Batr, Hadhf) [4], `rajaz` (Kasf) [4], `ramal` (Hadhf) [4], `taweel` (Batr, Hadhf) [4], `wafer` (Qatf) [4].
2. **Word-boundary phonetic merging (consonant liaison) misalignment:**
   - **Meters affected:** `khafeef` (Kaff, Shakal) [4].
3. **Randomized test failure cascading from the above:**
   - **Test affected:** `test_random_zihaf_determinism_and_integrity` [4].

---

## 2. Root Cause Analysis (Technical Deep-Dive)

### Issue A: Terminal-Only Modifications (_ʿIlal_) Leaking into Hashw (Non-Terminal) Positions

In `generate_fake_poem.py`, the set of valid modifications for non-terminal feet (`_VALID_HASHW_ZIHAFS`) is computed dynamically by scanning the global `_ZIHAF_MAP` [2]:

```python
for (_can_bin, _mod_bin), _zn in _ZIHAF_MAP.items():
    _fc = _BINARY_TO_FOOT_CLASS.get(_can_bin)
    if _fc:
        _VALID_HASHW_ZIHAFS.setdefault(_fc, set()).add(_zn)
```

**The Bug:** `_ZIHAF_MAP` contains both classical non-terminal zihāfāt (e.g., _Khaban_, _Tay_, _Qabadh_) and terminal-only modifications known as **ʿIlal** (e.g., _Hadhf_, _Batr_, _Kasf_, _Qatf_) [2].

Because the dynamic builder does not distinguish between them, terminal-only modifications are treated as valid for non-terminal (_Hashw_) positions [2]. When applied to a non-terminal foot, they cause severe structural issues:

- **Foot Truncation:** Modifications like _Batr_ (which reduces `Fawlon` from 5 syllables to 2 syllables: `فَعْ` [4]) or _Hadhf_ (reducing `Faelaton` to `فَاعِلُنْ` [4]) shorten the generated foot by 2 to 3 morae.
- **Metrical Drift:** Because the generative parser (`pyarud`) scans the entire generated line continuously, a truncated non-terminal foot shifts the starting offset of _every subsequent foot_ in that hemistich. This causes a cascading failure where the rest of the feet are marked as "broken" or "missing" [4].

_Example from test log (baseet with Kasf) [4]:_

- **Generated Text:** `مُسْتَفْعِلْ فَاعِلُنْ مُسْتَفْعِلْ فَعِلُ` [4]
- **Expected Sadr Feet:** `[Mustafelon (Hashw)] [Faelon (Hashw)] [Mustafelon (Hashw)] [Faelon (ʿArūḍ)]`
- **Observed Parser Slices:** Because `مُسْتَفْعِلْ` (Foot 0) is truncated by _Kasf_ [4], the parser shifts its alignment window. Foot 1 (`فَاعِلُنْ`) is evaluated against the wrong phonetic offset, causing it and all subsequent feet to fail verification [4].

---

### Issue B: Word-Boundary Phonetic Merging (Liaison)

Certain non-terminal zihāfāt are classically valid, but when applied generatively with space-separated words, they trigger phonetic boundaries that the underlying parsing engine cannot resolve [4].

_Example from test log (khafeef with Kaff) [4]:_

- **Generated Text:** `فَاعِلَاتُ مُسْتَفْعِلُنْ فَاعِلَاتُ` [4]
- **The Mechanism:** `Kaff` on `Faelaton` yields `فَاعِلَاتُ` [4], which ends in a moving vowel (**mutaharrik** `تُ`) [4]. The next foot (`مُسْتَفْعِلُنْ`) starts with another moving vowel (`مُ`) [4].
- **The Parser Behavior:** In classical Arabic prosody, two adjacent moving letters across a word boundary merge/flow phonetically. The parser's phonetic segmenter merges the end of Foot 0 with the start of Foot 1, producing the slice `U_UU_UU` instead of `U_UU_U_` [4]. This boundary drift breaks the clean space-separated foot mapping, causing a multi-foot failure [4].

---

## 3. Impact

While the generated verses are structurally coherent on a word-by-word basis, they violate the continuous phonetic expectations of the classical Arabic metrical parser (`pyarud`). This discrepancies breaks the reliability of generated datasets for downstream tasks (like LLM fine-tuning or prosody pipeline verification) where strict metrical verification is mandatory.

---

## 4. Proposed Remediation

### Fix 1: Filter out Terminal-Only _ʿIlal_ from `_VALID_HASHW_ZIHAFS`

We must strictly restrict `_VALID_HASHW_ZIHAFS` to classical, non-terminal zihāfāt [2]. This can be achieved by introducing a static exclusion blacklist (or an explicit whitelist) of true _Hashw_ zihāfāt [2].

**Implementation Change (in `generate_fake_poem.py`):**

```python
# Define terminal-only modifications (ʿIlal) that must never be applied to Hashw (non-terminal) positions
TERMINAL_ONLY_ILAL = frozenset({"Kasf", "Hadhf", "Batr", "Qatf", "Waqf"})

# Filter the dynamic zihaf collector
_VALID_HASHW_ZIHAFS: dict[str, set[str]] = {}

for (_can_bin, _mod_bin), _zn in _ZIHAF_MAP.items():
    if _zn in TERMINAL_ONLY_ILAL:
        continue  # Prevent terminal-only modifications from leaking into non-terminal positions

    _fc = _BINARY_TO_FOOT_CLASS.get(_can_bin)
    if _fc:
        _VALID_HASHW_ZIHAFS.setdefault(_fc, set()).add(_zn)
```

---

### Fix 2: Handle Word-Boundary Syllable Liaison (Optional/Heuristic)

For zihāfāt that end in a mutaharrik vowel (like _Kaff_ or _Shakal_), we can either:

1. **Blacklist them** for generative purposes to ensure 100% test compatibility with continuous phonetic parsers.
2. **Inject trailing quiet characters** (like a silent alif or nunation) to terminate the vowel, though this risks changing the classical representation of the mnemonic.

The safest, highest-reliability solution is to exclude `Kaff` and `Shakal` from the non-terminal generative pool when doing strict phonetic space-separated validation.

---

## 5. Proactivity Observation: Phonetic Continuity vs. Whitespace Tokenization

⚑ **Observation: The Discrete Word Boundary Fallacy**
This debugging session exposes a fundamental architectural assumption: _the generator assumes Arabic prosody is discrete and bounded by whitespace, while the parsing engine handles Arabic prosody as a continuous stream of phonemes._

Applying modifications that leave words ending in mutaharrik (moving/short-vowel) syllables breaks the discrete-word assumption because of phonetic liaison. Moving forward, any generative systems for classical Arabic prosody should represent lines as continuous phonetic streams rather than tokenized word lists.
