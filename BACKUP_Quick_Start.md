# Arabic Prosody Feedback: Feature Demonstration

This notebook demonstrates the core features of the `arabic_prosody_feedback` library. We will analyze a classical poem, generate human-readable correction reports, inspect structured prosodic metadata, and resolve classical Arabic meter names and mnemonics.

---

## 1. Setup and Definition of the Poem

First, we will import the necessary components from `arabic_prosody_feedback` and define our dataset. The dataset consists of the first 15 verses of the famous Mu'allaqa of *Al-Harith ibn Hilliza*, which is composed in the **Khafif (الخفيف)** meter.

```python
from arabic_prosody_feedback import (
    analyze_and_report,
    analyze_verse,
    get_tafeela_mnemonic,
    to_pyarud_meter_key
)

# 15 verses from the Mu'allaqa of Al-Harith ibn Hilliza
poem = [
    ("آذَنَتْنَا بِبَيْنِهَا أَسْمَاءُ", "رُبَّ ثَاوٍ يُمَلُّ مِنْهُ الثَّوَاءُ"),
    ("آذَنَتْنَا بِبَيْنِهَا ثُمَّ وَلَّتْ", "لَيْتَ شِعْرِي مَتَى يَكُونُ اللِّقَاءُ"),
    ("بَعْدَ عَهْدٍ لَنَا بِبُرْقَةِ شَمَّاءَ", "فَأَدْنَى دِيَارِهَا الْخَلْصَاءُ"),
    ("فَالْمُحَيَّاةُ فَالصِّفَاحُ فَأَعْنَا", "قُ فِتَاقٍ فَعَاذِبٌ فَالْوَفَاءُ"),
    ("فَرِيَاضُ الْقَطَا فَأَوْدِيَةُ الشُّرْ", "بَبِ فَالشُّعْبَتَانِ فَالْأَبْلَاءُ"),
    ("لَا أَرَى مَنْ عَهِدْتُ فِيهَا فَأَبْكِي", "الْيَوْمَ دَلْهًا وَمَا يَرُدُّ الْبُكَاءُ"),
    ("وَبِعَيْنَيْكَ أَوْقَدَتْ هِنْدٌ النَّا", "رَ أَخِيرًا تُلْوِي بِهَا الْعَلْيَاءُ"),
    ("أَوْقَدَتْهَا بَيْنَ الْعَقِيقِ فَشَخْصَيْ", "نِ بِعُودٍ كَمَا يَلُوحُ الضِّيَاءُ"),
    ("فَتَنَوَّرْتُ نَارَهَا مِنْ بَعِيدٍ", "بِخَزَازٍ هَيْهَاتَ مِنْكَ الصِّلَاءُ"),
    ("غَيْرَ أَنِّي قَدْ أَسْتَعِينُ عَلَى الْهَمْ", "مِ إِذَا خَفَّ بِالثَّوِيِّ النَّجَاءُ"),
    ("بِزَفُوفٍ كَأَنَّهَا هِقْلَةٌ أُمْ", "مُ رِئَالٍ دَوِيَّةٌ سَقْفَاءُ"),
    ("آنَسَتْ نَبْأَةً وَأَفْزَعَهَا الْقَنْ", "نَاصُ عَصْرًا وَقَدْ دَنَا الْإِمْسَاءُ"),
    ("فَتَرَى خَلْفَهَا مِنَ الرَّجْعِ وَالْوَقْ", "عِ مَنِينًا كَأَنَّهُ إِهْبَاءُ"),
    ("وَطِرَاقًا مِنْ خَلْفِهِنَّ طِرَاقٌ", "سَاقِطَاتٌ أَلْوَتْ بِهَا الصَّحْرَاءُ"),
    ("أَتَلَهَّى بِهَا الْهَوَاجِرَ إِذْ كُلْ", "لُ ابْنِ هَمٍّ بَلِيَّةٌ عَمْيَاءُ")
]

print(f"Loaded {len(poem)} verses for analysis.")
```

---

## 2. Generating a Consolidated Poem Correction Report

Using `analyze_and_report`, we can run a full prosodic pass over the entire poem.

This generates:
- A structured Python dictionary containing nested raw statistics.
- A ready-to-use string report. This report is formatted as an LLM prompt, highlighting metrical alignment, syllable weight mismatches, and corrections.

```python
# Analyze the poem. We set score_threshold=0.95 to skip detailing mostly sound verses.
analysis_dict, correction_report = analyze_and_report(
    poem,
    meter_name="khafif",
    score_threshold=0.95,
    print_summary=True
)

# Output the complete textual correction report
print("\n" + "="*50 + " GENERATED REPORT " + "="*50 + "\n")
print(correction_report)
```

---

## 3. Granular Analysis of an Individual Verse

If you want to integrate prosody data into a custom application interface or database, you can use `analyze_verse`. This parses the target verse and returns structured objects (`VerseResult`, `HemistichResult`, and `FootResult`) with explicit metadata.

We will analyze the first verse and inspect its underlying structural data.

```python
# Extract the first verse
sadr, ajuz = poem[0]

# Pre-resolve key to ensure lower-level functions receive native pyarud keys (e.g. "khafeef")
pyarud_key = to_pyarud_meter_key("khafif")

# Perform detailed parsing
verse_result = analyze_verse(sadr, ajuz, meter_name=pyarud_key)

print(f"Verse Index : {verse_result.verse_index}")
print(f"Meter Key   : {verse_result.meter}")
print(f"Joint Score : {verse_result.combined_score * 100:.2f}%")
print("-" * 50)

# Inspect the Ṣadr (First Hemistich)
print(f"Ṣadr Text    : {verse_result.sadr.text}")
print(f"Ṣadr Pattern : {verse_result.sadr.pattern} (U = short, _ = long)")
print(f"Is Sound?    : {verse_result.sadr.is_sound}")
print("\nFoot-by-Foot breakdown:")

for foot in verse_result.sadr.feet:
    print(f"  - Foot {foot.foot_index + 1} ({foot.position_label}):")
    print(f"    Expected pattern  : {foot.expected_pattern}")
    print(f"    Observed segment  : {foot.actual_segment}")
    print(f"    Syllable Score    : {foot.score * 100:.1f}%")
    print(f"    Status & Health   : {foot.status} ({foot.health})")
    print(f"    Identified Zihāf  : {foot.zihaf_name}")
```

---

## 4. Resolving Metrical Keys and Classical Mnemonics

The utility functions help normalize user inputs and resolve classical poetic terminology.

- `to_pyarud_meter_key`: Converts standard, transliterated, or Arabic names into the exact internal key expected by the processing engine.
- `get_tafeela_mnemonic`: Returns the correct classical mnemonic (Tafʿīla) for a given foot and its active metrical deviation (Zihāf).

```python
# Resolve various ways of writing the same meter
# We use lowercase "khafeef" (native key) and capitalized "Khafif" (alias key) to demonstrate successful resolution paths.
inputs = ["الخفيف", "Khafif", "khafeef"]
for raw_input in inputs:
    resolved = to_pyarud_meter_key(raw_input)
    print(f"Resolved: '{raw_input}' -> '{resolved}'")

print("-" * 50)

# Look up Taf'ila Mnemonics under different Zihāf conditions
scenarios = [
    ("Mustafelon", "Salim"),   # Pristine / healthy
    ("Mustafelon", "Khaban"),  # Dropping the second quiet letter
    ("Fawlon", "Qabadh"),      # Dropping the fifth quiet letter
    ("Faelaton", "Kaff")       # Dropping the seventh quiet letter
]

for foot_class, zihaf in scenarios:
    mnemonic = get_tafeela_mnemonic(foot_class, zihaf)
    print(f"Foot: {foot_class:<12} | Zihāf: {zihaf:<8} -> Mnemonic: {mnemonic}")
```
