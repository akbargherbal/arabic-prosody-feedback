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

    Loaded 15 verses for analysis.
    


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

    Verse 1: «آذَنَتْنَا بِبَيْنِهَا أَسْمَاءُ | رُبَّ ثَاوٍ يُمَلُّ مِنْهُ الثَّوَاءُ» — Accuracy: 86.00%
    Verse 2: «آذَنَتْنَا بِبَيْنِهَا ثُمَّ وَلَّتْ | لَيْتَ شِعْرِي مَتَى يَكُونُ اللِّقَاءُ» — Accuracy: 93.00%
    Verse 3: «بَعْدَ عَهْدٍ لَنَا بِبُرْقَةِ شَمَّاءَ | فَأَدْنَى دِيَارِهَا الْخَلْصَاءُ» — Accuracy: 85.00%
    Verse 4: «فَالْمُحَيَّاةُ فَالصِّفَاحُ فَأَعْنَا | قُ فِتَاقٍ فَعَاذِبٌ فَالْوَفَاءُ» — Accuracy: 100.00%
    Verse 5: «فَرِيَاضُ الْقَطَا فَأَوْدِيَةُ الشُّرْ | بَبِ فَالشُّعْبَتَانِ فَالْأَبْلَاءُ» — Accuracy: 100.00%
    Verse 6: «لَا أَرَى مَنْ عَهِدْتُ فِيهَا فَأَبْكِي | الْيَوْمَ دَلْهًا وَمَا يَرُدُّ الْبُكَاءُ» — Accuracy: 87.00%
    Verse 7: «وَبِعَيْنَيْكَ أَوْقَدَتْ هِنْدٌ النَّا | رَ أَخِيرًا تُلْوِي بِهَا الْعَلْيَاءُ» — Accuracy: 92.00%
    Verse 8: «أَوْقَدَتْهَا بَيْنَ الْعَقِيقِ فَشَخْصَيْ | نِ بِعُودٍ كَمَا يَلُوحُ الضِّيَاءُ» — Accuracy: 100.00%
    Verse 9: «فَتَنَوَّرْتُ نَارَهَا مِنْ بَعِيدٍ | بِخَزَازٍ هَيْهَاتَ مِنْكَ الصِّلَاءُ» — Accuracy: 100.00%
    Verse 10: «غَيْرَ أَنِّي قَدْ أَسْتَعِينُ عَلَى الْهَمْ | مِ إِذَا خَفَّ بِالثَّوِيِّ النَّجَاءُ» — Accuracy: 100.00%
    Verse 11: «بِزَفُوفٍ كَأَنَّهَا هِقْلَةٌ أُمْ | مُ رِئَالٍ دَوِيَّةٌ سَقْفَاءُ» — Accuracy: 100.00%
    Verse 12: «آنَسَتْ نَبْأَةً وَأَفْزَعَهَا الْقَنْ | نَاصُ عَصْرًا وَقَدْ دَنَا الْإِمْسَاءُ» — Accuracy: 92.00%
    Verse 13: «فَتَرَى خَلْفَهَا مِنَ الرَّجْعِ وَالْوَقْ | عِ مَنِينًا كَأَنَّهُ إِهْبَاءُ» — Accuracy: 100.00%
    Verse 14: «وَطِرَاقًا مِنْ خَلْفِهِنَّ طِرَاقٌ | سَاقِطَاتٌ أَلْوَتْ بِهَا الصَّحْرَاءُ» — Accuracy: 100.00%
    Verse 15: «أَتَلَهَّى بِهَا الْهَوَاجِرَ إِذْ كُلْ | لُ ابْنِ هَمٍّ بَلِيَّةٌ عَمْيَاءُ» — Accuracy: 100.00%
    
    ================================================== GENERATED REPORT ==================================================
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║  POEM CORRECTION REPORT  ·  الخفيف (khafeef)
    ║  15 verses  ·  Overall score: 95.7%
    ╚══════════════════════════════════════════════════════════════════╝
    
      VERSE SUMMARY
      ────────────────────────────────────────────────────────────
      ✗  Verse  1  [ 86.0%]  آذَنَتْنَا بِبَيْنِهَا أَسْمَاءُ
      ✗  Verse  2  [ 93.0%]  آذَنَتْنَا بِبَيْنِهَا ثُمَّ وَلَّت…
      ✗  Verse  3  [ 85.0%]  بَعْدَ عَهْدٍ لَنَا بِبُرْقَةِ شَمَ…
      ✓  Verse  4  [100.0%]  فَالْمُحَيَّاةُ فَالصِّفَاحُ فَأَعْ…
      ✓  Verse  5  [100.0%]  فَرِيَاضُ الْقَطَا فَأَوْدِيَةُ الش…
      ✗  Verse  6  [ 87.0%]  لَا أَرَى مَنْ عَهِدْتُ فِيهَا فَأَ…
      ✗  Verse  7  [ 92.0%]  وَبِعَيْنَيْكَ أَوْقَدَتْ هِنْدٌ ال…
      ✓  Verse  8  [100.0%]  أَوْقَدَتْهَا بَيْنَ الْعَقِيقِ فَش…
      ✓  Verse  9  [100.0%]  فَتَنَوَّرْتُ نَارَهَا مِنْ بَعِيدٍ
      ✓  Verse 10  [100.0%]  غَيْرَ أَنِّي قَدْ أَسْتَعِينُ عَلَ…
      ✓  Verse 11  [100.0%]  بِزَفُوفٍ كَأَنَّهَا هِقْلَةٌ أُمْ
      ✗  Verse 12  [ 92.0%]  آنَسَتْ نَبْأَةً وَأَفْزَعَهَا الْق…
      ✓  Verse 13  [100.0%]  فَتَرَى خَلْفَهَا مِنَ الرَّجْعِ وَ…
      ✓  Verse 14  [100.0%]  وَطِرَاقًا مِنْ خَلْفِهِنَّ طِرَاقٌ
      ✓  Verse 15  [100.0%]  أَتَلَهَّى بِهَا الْهَوَاجِرَ إِذْ …
      ────────────────────────────────────────────────────────────
      Broken / total: 6 / 15
    
    ══════════════════════════════════════════════════════════════════
      VERSE 1  ·  ⚠ IRREGULAR  ·  Score: 86%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : آذَنَتْنَا بِبَيْنِهَا أَسْمَاءُ
      ʿAjuz : رُبَّ ثَاوٍ يُمَلُّ مِنْهُ الثَّوَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_UU_ ]  [      ]   ← Expected
      │  [UUUU_U_]  [UU_UU_]  [U_U_U ]  [  _   ]   ← Actual
      │  [✗ BROKEN]  [~Qabadh]  [✗ BROKEN]  [! EXTRA]
      │  [ BROKEN]  [مَفَاعِلُنْ]  [BROKEN]  [EXTRA ]
      │  Morae: expected 18, actual 19  ⚠ hemistich is 1 mora(s) too long
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Expected
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Actual
      │  [   ✓   ]  [~Qabadh]  [   ✓   ]
      │  [فَاعِلَاتُنْ]  [مَفَاعِلُنْ]  [فَاعِلَاتُنْ]
      │  Morae: expected 20, actual 20
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ṢADR (صَدْر)  |  Foot 1  |  Hashw]
        Expected : U_UU_U_  (7 morae)
        Actual   : UUUU_U_  (7 morae)
    
        Expected:  U _ U U _ U _
        Actual:    U U U U _ U _
        Diff:      | × | | | | |    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 2: short (U) → long (_).
    
      ▸ [ṢADR (صَدْر)  |  Foot 3  |  Hashw]
        Expected : U_UU_  (5 morae)
        Actual   : U_U_U  (5 morae)
    
        Expected:  U _ U U _
        Actual:    U _ U _ U
        Diff:      | | | × ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 4: long (_) → short (U); pos 5: short (U) → long (_).
    
      ▸ [ṢADR (صَدْر)  |  Foot 4  |  Extra]
        ✗ Extra material after all expected feet consumed.
        Extra bits: _  (1 morae)
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ṢADR (صَدْر), Foot 1 (Hashw)]  Adjust syllable weights in «UUUU_U_» to match «U_UU_U_» (same length, wrong weight pattern).
      2. [ṢADR (صَدْر), Foot 3 (Hashw)]  Adjust syllable weights in «U_U_U» to match «U_UU_» (same length, wrong weight pattern).
      3. [ṢADR (صَدْر)]  Remove word(s) producing the trailing «_» (1 extra morae).
    
      ── METER REFERENCE ─────────────────────────────────────────────
      البحر  : الخفيف (khafeef)
      Tafāʿīl: فَاعِلَاتُنْ مُسْتَفْعِلُنْ فَاعِلَاتُ
      Ṣadr   : U_UU_U_ | UU_UU_ | U_UU_
      ʿAjuz  : U_UU_U_ | UU_UU_ | U_UU_U_
    
    ══════════════════════════════════════════════════════════════════
    
    ══════════════════════════════════════════════════════════════════
      VERSE 2  ·  ~ NEAR-PERFECT  ·  Score: 93%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : آذَنَتْنَا بِبَيْنِهَا ثُمَّ وَلَّتْ
      ʿAjuz : لَيْتَ شِعْرِي مَتَى يَكُونُ اللِّقَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Expected
      │  [UUUU_U_]  [UU_UU_]  [U_UU_U_]   ← Actual
      │  [✗ BROKEN]  [~Qabadh]  [   ✓   ]
      │  [ BROKEN]  [مَفَاعِلُنْ]  [فَاعِلَاتُنْ]
      │  Morae: expected 20, actual 20
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Expected
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Actual
      │  [   ✓   ]  [~Qabadh]  [   ✓   ]
      │  [فَاعِلَاتُنْ]  [مَفَاعِلُنْ]  [فَاعِلَاتُنْ]
      │  Morae: expected 20, actual 20
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ṢADR (صَدْر)  |  Foot 1  |  Hashw]
        Expected : U_UU_U_  (7 morae)
        Actual   : UUUU_U_  (7 morae)
    
        Expected:  U _ U U _ U _
        Actual:    U U U U _ U _
        Diff:      | × | | | | |    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 2: short (U) → long (_).
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ṢADR (صَدْر), Foot 1 (Hashw)]  Adjust syllable weights in «UUUU_U_» to match «U_UU_U_» (same length, wrong weight pattern).
    
    ══════════════════════════════════════════════════════════════════
    
    ══════════════════════════════════════════════════════════════════
      VERSE 3  ·  ⚠ IRREGULAR  ·  Score: 85%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : بَعْدَ عَهْدٍ لَنَا بِبُرْقَةِ شَمَّاءَ
      ʿAjuz : فَأَدْنَى دِيَارِهَا الْخَلْصَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [UUU_U_]  [      ]   ← Expected
      │  [U_UU_U_]  [UU_UU_]  [UUU_U_]  [  U   ]   ← Actual
      │  [   ✓   ]  [~Qabadh]  [~Khaban]  [! EXTRA]
      │  [فَاعِلَاتُنْ]  [مَفَاعِلُنْ]  [فَعِلَاتُنْ]  [EXTRA ]
      │  Morae: expected 19, actual 20  ⚠ hemistich is 1 mora(s) too long
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [UUU_U_]  [UU_UU_]  [U_U_U_]   ← Expected
      │  [UU_U_U]  [U_UU_U]  [_U_U_ ]   ← Actual
      │  [✗ BROKEN]  [✗ BROKEN]  [✗ BROKEN]
      │  [BROKEN]  [BROKEN]  [BROKEN]
      │  Morae: expected 18, actual 17  ⚠ hemistich is 1 mora(s) too short
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ṢADR (صَدْر)  |  Foot 4  |  Extra]
        ✗ Extra material after all expected feet consumed.
        Extra bits: U  (1 morae)
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 1  |  Hashw]
        Expected : UUU_U_  (6 morae)
        Actual   : UU_U_U  (6 morae)
    
        Expected:  U U U _ U _
        Actual:    U U _ U _ U
        Diff:      | | × × × ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 3: long (_) → short (U); pos 4: short (U) → long (_); pos 5: long (_) → short (U); pos 6: short (U) → long (_).
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 2  |  Hashw]
        Expected : UU_UU_  (6 morae)
        Actual   : U_UU_U  (6 morae)
    
        Expected:  U U _ U U _
        Actual:    U _ U U _ U
        Diff:      | × × | × ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 2: long (_) → short (U); pos 3: short (U) → long (_); pos 5: long (_) → short (U); pos 6: short (U) → long (_).
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 3  |  Ḍarb]
        Expected : U_U_U_  (6 morae)
        Actual   : _U_U_  (5 morae)
    
        Expected:  U _ U _ U _
        Actual:    _ U _ U _ ·
        Diff:      × × × × × ^    (| match  × wrong weight  ^ missing  v extra)
    
        → Foot is 1 mora(s) too short. Extend with: long (_)  →  target «U_U_U_».
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ṢADR (صَدْر)]  Remove word(s) producing the trailing «U» (1 extra morae).
      2. [ʿAJUZ (عَجُز), Foot 1 (Hashw)]  Adjust syllable weights in «UU_U_U» to match «UUU_U_» (same length, wrong weight pattern).
      3. [ʿAJUZ (عَجُز), Foot 2 (Hashw)]  Adjust syllable weights in «U_UU_U» to match «UU_UU_» (same length, wrong weight pattern).
      4. [ʿAJUZ (عَجُز), Foot 3 (Ḍarb)]  Replace word(s) giving «_U_U_» with word(s) giving «U_U_U_» — need 1 more mora(s).
    
    ══════════════════════════════════════════════════════════════════
    
    ══════════════════════════════════════════════════════════════════
      VERSE 6  ·  ⚠ IRREGULAR  ·  Score: 87%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : لَا أَرَى مَنْ عَهِدْتُ فِيهَا فَأَبْكِي
      ʿAjuz : الْيَوْمَ دَلْهًا وَمَا يَرُدُّ الْبُكَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Expected
      │  [U_UU_U_]  [UU_UU_]  [U_UU_U_]   ← Actual
      │  [   ✓   ]  [~Qabadh]  [   ✓   ]
      │  [فَاعِلَاتُنْ]  [مَفَاعِلُنْ]  [فَاعِلَاتُنْ]
      │  Morae: expected 20, actual 20
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_U_U_]  [      ]   ← Expected
      │  [U_U_UU_]  [U_UU_U]  [U_U_UU]  [ _U_  ]   ← Actual
      │  [✗ BROKEN]  [✗ BROKEN]  [✗ BROKEN]  [! EXTRA]
      │  [ BROKEN]  [BROKEN]  [BROKEN]  [EXTRA ]
      │  Morae: expected 19, actual 22  ⚠ hemistich is 3 mora(s) too long
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 1  |  Hashw]
        Expected : U_UU_U_  (7 morae)
        Actual   : U_U_UU_  (7 morae)
    
        Expected:  U _ U U _ U _
        Actual:    U _ U _ U U _
        Diff:      | | | × × | |    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 4: long (_) → short (U); pos 5: short (U) → long (_).
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 2  |  Hashw]
        Expected : UU_UU_  (6 morae)
        Actual   : U_UU_U  (6 morae)
    
        Expected:  U U _ U U _
        Actual:    U _ U U _ U
        Diff:      | × × | × ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 2: long (_) → short (U); pos 3: short (U) → long (_); pos 5: long (_) → short (U); pos 6: short (U) → long (_).
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 3  |  Hashw]
        Expected : U_U_U_  (6 morae)
        Actual   : U_U_UU  (6 morae)
    
        Expected:  U _ U _ U _
        Actual:    U _ U _ U U
        Diff:      | | | | | ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 6: short (U) → long (_).
    
      ▸ [ʿAJUZ (عَجُز)  |  Foot 4  |  Extra]
        ✗ Extra material after all expected feet consumed.
        Extra bits: _U_  (3 morae)
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ʿAJUZ (عَجُز), Foot 1 (Hashw)]  Adjust syllable weights in «U_U_UU_» to match «U_UU_U_» (same length, wrong weight pattern).
      2. [ʿAJUZ (عَجُز), Foot 2 (Hashw)]  Adjust syllable weights in «U_UU_U» to match «UU_UU_» (same length, wrong weight pattern).
      3. [ʿAJUZ (عَجُز), Foot 3 (Hashw)]  Adjust syllable weights in «U_U_UU» to match «U_U_U_» (same length, wrong weight pattern).
      4. [ʿAJUZ (عَجُز)]  Remove word(s) producing the trailing «_U_» (3 extra morae).
    
    ══════════════════════════════════════════════════════════════════
    
    ══════════════════════════════════════════════════════════════════
      VERSE 7  ·  ~ NEAR-PERFECT  ·  Score: 92%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : وَبِعَيْنَيْكَ أَوْقَدَتْ هِنْدٌ النَّا
      ʿAjuz : رَ أَخِيرًا تُلْوِي بِهَا الْعَلْيَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [UUU_U_]  [UU_UU_]  [U_UU_ ]  [      ]   ← Expected
      │  [UUU_U_]  [UU_UU_]  [U_U_U ]  [  _   ]   ← Actual
      │  [~Khaban]  [~Qabadh]  [✗ BROKEN]  [! EXTRA]
      │  [فَعِلَاتُنْ]  [مَفَاعِلُنْ]  [BROKEN]  [EXTRA ]
      │  Morae: expected 17, actual 18  ⚠ hemistich is 1 mora(s) too long
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [UUU_U_]  [U_U_UU_]  [U_U_U_]   ← Expected
      │  [UUU_U_]  [U_U_UU_]  [U_U_U_]   ← Actual
      │  [~Khaban]  [   ✓   ]  [~Kasf ]
      │  [فَعِلَاتُنْ]  [مُسْتَفْعِلُنْ]  [مُسْتَفْعِلْ]
      │  Morae: expected 19, actual 19
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ṢADR (صَدْر)  |  Foot 3  |  Hashw]
        Expected : U_UU_  (5 morae)
        Actual   : U_U_U  (5 morae)
    
        Expected:  U _ U U _
        Actual:    U _ U _ U
        Diff:      | | | × ×    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 4: long (_) → short (U); pos 5: short (U) → long (_).
    
      ▸ [ṢADR (صَدْر)  |  Foot 4  |  Extra]
        ✗ Extra material after all expected feet consumed.
        Extra bits: _  (1 morae)
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ṢADR (صَدْر), Foot 3 (Hashw)]  Adjust syllable weights in «U_U_U» to match «U_UU_» (same length, wrong weight pattern).
      2. [ṢADR (صَدْر)]  Remove word(s) producing the trailing «_» (1 extra morae).
    
    ══════════════════════════════════════════════════════════════════
    
    ══════════════════════════════════════════════════════════════════
      VERSE 12  ·  ~ NEAR-PERFECT  ·  Score: 92%
    ══════════════════════════════════════════════════════════════════
      Meter : الخفيف (khafeef)
      Ṣadr  : آنَسَتْ نَبْأَةً وَأَفْزَعَهَا الْقَنْ
      ʿAjuz : نَاصُ عَصْرًا وَقَدْ دَنَا الْإِمْسَاءُ
    
      ┌─ ṢADR (صَدْر) ────────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [UUU_U_]   ← Expected
      │  [UUUU_U_]  [UU_UU_]  [UUU_U_]   ← Actual
      │  [✗ BROKEN]  [~Qabadh]  [~Khaban]
      │  [ BROKEN]  [مَفَاعِلُنْ]  [فَعِلَاتُنْ]
      │  Morae: expected 19, actual 19
      └──────────────────────────────────────────────────────────────┘
    
      ┌─ ʿAJUZ (عَجُز) ───────────────────────────────────────────┐
      │  [U_UU_U_]  [UU_UU_]  [U_U_U_]   ← Expected
      │  [U_UU_U_]  [UU_UU_]  [U_U_U_]   ← Actual
      │  [   ✓   ]  [~Qabadh]  [~Kasf ]
      │  [فَاعِلَاتُنْ]  [مَفَاعِلُنْ]  [مُسْتَفْعِلْ]
      │  Morae: expected 19, actual 19
      └──────────────────────────────────────────────────────────────┘
    
      ── DETAILED DIAGNOSIS ──────────────────────────────────────────
    
      ▸ [ṢADR (صَدْر)  |  Foot 1  |  Hashw]
        Expected : U_UU_U_  (7 morae)
        Actual   : UUUU_U_  (7 morae)
    
        Expected:  U _ U U _ U _
        Actual:    U U U U _ U _
        Diff:      | × | | | | |    (| match  × wrong weight  ^ missing  v extra)
    
        → Wrong syllable weight(s): pos 2: short (U) → long (_).
    
      ── CORRECTION PRESCRIPTION ─────────────────────────────────────
      1. [ṢADR (صَدْر), Foot 1 (Hashw)]  Adjust syllable weights in «UUUU_U_» to match «U_UU_U_» (same length, wrong weight pattern).
    
    ══════════════════════════════════════════════════════════════════
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONSOLIDATED FIX LIST
    ╚══════════════════════════════════════════════════════════════════╝
      1. Verse 1 [Ṣadr, Foot 1 (Hashw)]  REWEIGHT  «UUUU_U_»  →  «U_UU_U_».
      2. Verse 1 [Ṣadr, Foot 3 (Hashw)]  REWEIGHT  «U_U_U»  →  «U_UU_».
      3. Verse 1 [Ṣadr]  REMOVE extra «_».
      4. Verse 2 [Ṣadr, Foot 1 (Hashw)]  REWEIGHT  «UUUU_U_»  →  «U_UU_U_».
      5. Verse 3 [Ṣadr]  REMOVE extra «U».
      6. Verse 3 [ʿAjuz, Foot 1 (Hashw)]  REWEIGHT  «UU_U_U»  →  «UUU_U_».
      7. Verse 3 [ʿAjuz, Foot 2 (Hashw)]  REWEIGHT  «U_UU_U»  →  «UU_UU_».
      8. Verse 3 [ʿAjuz, Foot 3 (Ḍarb)]  ADD 1 mora(s) to  «_U_U_»  →  «U_U_U_».
      9. Verse 6 [ʿAjuz, Foot 1 (Hashw)]  REWEIGHT  «U_U_UU_»  →  «U_UU_U_».
      10. Verse 6 [ʿAjuz, Foot 2 (Hashw)]  REWEIGHT  «U_UU_U»  →  «UU_UU_».
      11. Verse 6 [ʿAjuz, Foot 3 (Hashw)]  REWEIGHT  «U_U_UU»  →  «U_U_U_».
      12. Verse 6 [ʿAjuz]  REMOVE extra «_U_».
      13. Verse 7 [Ṣadr, Foot 3 (Hashw)]  REWEIGHT  «U_U_U»  →  «U_UU_».
      14. Verse 7 [Ṣadr]  REMOVE extra «_».
      15. Verse 12 [Ṣadr, Foot 1 (Hashw)]  REWEIGHT  «UUUU_U_»  →  «U_UU_U_».
    
    


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

    Verse Index : 0
    Meter Key   : khafeef
    Joint Score : 86.00%
    --------------------------------------------------
    Ṣadr Text    : آذَنَتْنَا بِبَيْنِهَا أَسْمَاءُ
    Ṣadr Pattern : UUUU_U_UU_UU_U_U_U_ (U = short, _ = long)
    Is Sound?    : False
    
    Foot-by-Foot breakdown:
      - Foot 1 (Hashw):
        Expected pattern  : U_UU_U_
        Observed segment  : UUUU_U_
        Syllable Score    : 40.0%
        Status & Health   : broken (broken)
        Identified Zihāf  : None
      - Foot 2 (Hashw):
        Expected pattern  : UU_UU_
        Observed segment  : UU_UU_
        Syllable Score    : 100.0%
        Status & Health   : ok (valid_zihaf)
        Identified Zihāf  : Qabadh
      - Foot 3 (Hashw):
        Expected pattern  : U_UU_
        Observed segment  : U_U_U
        Syllable Score    : 26.0%
        Status & Health   : broken (broken)
        Identified Zihāf  : None
      - Foot 4 (Extra):
        Expected pattern  : 
        Observed segment  : _
        Syllable Score    : 0.0%
        Status & Health   : extra_bits (severe)
        Identified Zihāf  : None
    


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

    Resolved: 'الخفيف' -> 'khafeef'
    Resolved: 'Khafif' -> 'khafeef'
    Resolved: 'khafeef' -> 'khafeef'
    --------------------------------------------------
    Foot: Mustafelon   | Zihāf: Salim    -> Mnemonic: مُسْتَفْعِلُنْ
    Foot: Mustafelon   | Zihāf: Khaban   -> Mnemonic: مُتَفْعِلُنْ
    Foot: Fawlon       | Zihāf: Qabadh   -> Mnemonic: فَعُولُ
    Foot: Faelaton     | Zihāf: Kaff     -> Mnemonic: فَاعِلَاتُ
    
