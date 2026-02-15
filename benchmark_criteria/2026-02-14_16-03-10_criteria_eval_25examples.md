# Criteria Evaluation Report

**Date:** 2026-02-14 16:03
**Agent model:** claude-sonnet-4-5-20250929
**Judge model:** gpt-4o-mini
**Document:** BZ_VR1
**Entries evaluated:** 25
**Total time:** 33.9 min

## Summary Metrics

| Metric | Value |
|--------|-------|
| Accuracy (excl. abstentions) | **30.8%** |
| Precision (yes) | 1.000 |
| Recall (yes) | 0.308 |
| F1 Score | **0.471** |
| Abstention rate | 48.0% |
| Correct | 4 |
| Incorrect | 9 |
| Abstained | 12 |

## Confusion Matrix

```
                 Predicted YES    Predicted NO
Expected YES     TP = 4             FN = 9
Expected NO      FP = 0             TN = 0
```

## Positive vs Negated Split

| Split | Total | Correct | Incorrect | Abstained | Accuracy |
|-------|-------|---------|-----------|-----------|----------|
| Positive | 25 | 4 | 9 | 12 | 30.8% |
| Negated | 0 | 0 | 0 | 0 | 0.0% |

## Binding Split

| Binding | Total | Correct | Incorrect | Abstained | Accuracy |
|---------|-------|---------|-----------|-----------|----------|
| Povinné | 22 | 3 | 8 | 11 | 27.3% |
| Doporučené | 3 | 1 | 1 | 1 | 50.0% |

## Top 10 Worst-Performing Criteria

| Criterion ID | Total | Correct | Incorrect | Abstained |
|-------------|-------|---------|-----------|-----------|
| 499 | 5 | 0 | 2 | 3 |
| 552 | 5 | 1 | 2 | 2 |
| 233 | 3 | 0 | 2 | 1 |
| 533 | 2 | 0 | 2 | 0 |
| 444 | 3 | 1 | 1 | 1 |
| 537 | 3 | 1 | 0 | 2 |
| 116 | 2 | 0 | 0 | 2 |
| 351 | 1 | 1 | 0 | 0 |
| 625 | 1 | 0 | 0 | 1 |

## Sample Incorrect Answers (first 10)

### Criterion 233 (expected=yes, classified=no)

**Q:** Obsahuje bezpečnostní zpráva v kapitole 10.5 dostatečně podrobné informace o hlavních projektových charakteristikách a subsystémech spojených s provoz...

**Agent:** Na základě důkladného prohledání dokumentu BZ_VR1 mohu nyní poskytnout odpověď:

## Kapitola 10.5 v dokumentu BZ_VR1

Kapitola **10.5 Další pomocné systémy** \cite{BZ_VR1_p008}\cite{BZ_VR1_p108} v dokumentu BZ_VR1 **neobsahuje informace o turbíně a kondenzátoru**. 

### Obsah kapitoly 10.5

Podle ob...

**Judge:** Kapitola 10.5 neobsahuje informace o turbíně a kondenzátoru, což znamená, že dokument nesplňuje požadavek na podrobné informace o těchto systémech.

### Criterion 233 (expected=yes, classified=no)

**Q:** Obsahuje bezpečnostní zpráva v kapitole 10.5 dostatečně podrobné informace o kondenzátoru a pomocných systémech turbíny, včetně požadavků na provoz tu...

**Agent:** Na základě provedených vyhledávání mohu nyní poskytnout odpověď:

## Analýza kapitoly 10.5 - Další pomocné systémy

Na základě důkladného prohledání dokumentu BZ_VR1 mohu konstatovat, že **kapitola 10.5 "Další pomocné systémy" neobsahuje informace o turbínových systémech**. 

### Co kapitola 10.5 sk...

**Judge:** The agent explicitly states that chapter 10.5 does not contain the required information about the turbine systems, indicating that the document does not satisfy the requirement.

### Criterion 499 (expected=yes, classified=no)

**Q:** Obsahuje BZ v kapitole 13.3.1 dostatečně podrobný popis systému provozních kontrol vybraných zařízení, včetně požadavků na kalibraci a ověření funkce ...

**Agent:** # Odpověď na dotaz o kapitole 13.3.1

Na základě důkladného prohledání dokumentu BZ_VR1 mohu potvrdit následující:

## Existence kapitoly 13.3

**Kapitola 13.3 existuje** a nese název **"Příprava pracovníků reaktoru"** \cite{BZ_VR1_p008}\cite{BZ_VR1_p149}. Tato kapitola začíná na straně 149 dokument...

**Judge:** Podkapitola 13.3.1 neexistuje, což znamená, že dokument nemůže obsahovat požadovaný popis systému provozních kontrol a dalších specifikovaných informací.

### Criterion 444 (expected=yes, classified=no)

**Q:** Vyplývá z bezpečnostní zprávy, že držitel povolení je povinen zajistit, aby pracoviště, kde se vykonává radiační činnost, bylo navrženo a uvedeno do p...

**Agent:** Na základě nalezených informací mohu nyní odpovědět na dotaz. Z bezpečnostní zprávy vyplývá, že držitel povolení má povinnosti související s provozem pracoviště, ale konkrétní formulace o návrhu a uvedení pracoviště do provozu není v dokumentu explicitně uvedena v přesném znění, které je uvedeno v d...

**Judge:** Agent explicitly states that the document does not contain the specific wording required by the question regarding the obligation of the permit holder to ensure the workplace is designed and operated safely.

### Criterion 533 (expected=yes, classified=no)

**Q:** Obsahuje PrBZ v kapitole 13.4.5 popis návodů pro zvládání těžkých havárií, včetně obecného popisu těchto návodů, jejich účelu, rozsahu událostí, které...

**Agent:** Podívám se ještě na detailní obsah kapitoly 13 na straně 8 a 9, abych si ověřil strukturu kapitoly 13.4.

Z obsahu na straně 8 a 9 vidím:
- 13.4 Řízení údržby, odstávek a provozních kontrol ... 153
- 13.5 Řízení změn na reaktoru ... 154

Kapitola 13.4 **nemá žádné podkapitoly** (13.4.1, 13.4.2, atd....

**Judge:** Kapitola 13.4.5 v dokumentu neexistuje, což znamená, že požadované informace o návodě pro zvládání těžkých havárií nejsou obsaženy.

### Criterion 552 (expected=yes, classified=no)

**Q:** Obsahuje bezpečnostní zpráva v podkapitole 17.2 popis specifických aspektů systému řízení, včetně toho, jak jsou naplňovány požadavky prováděcího práv...

**Agent:** Rozumím, že chcete informace výhradně z dokumentu BZ_VR1. Bohužel jsem nenašel podkapitolu 17.2 v tomto dokumentu. Dokument obsahuje kapitolu 17 "Limity a podmínky" a podkapitolu 17.2 "Obsah limitů a podmínek", ale neobsahuje podkapitolu 17.2 týkající se systému řízení podle vyhlášky 408/2016.

Mohu...

**Judge:** The agent explicitly states that section 17.2 does not pertain to the management system as required by the regulation, indicating that the document does not satisfy the requirement.

### Criterion 533 (expected=yes, classified=no)

**Q:** Obsahuje PrBZ v kapitole 13.4.5 popis návodů pro zvládání těžkých havárií, které umožňují reagovat na vzniklou událost na základě příznaků, jako jsou ...

**Agent:** ## Odpověď na dotaz o podkapitole 13.4

Na základě důkladného prohledání dokumentu BZ_VR1 mohu potvrdit, že **podkapitola 13.4 v dokumentu existuje**.

### Nalezená podkapitola 13.4

Podkapitola **13.4 Řízení údržby, odstávek a provozních kontrol** se nachází na straně 152 dokumentu \cite{BZ_VR1_p15...

**Judge:** The agent confirms the existence of section 13.4, but the content described focuses on maintenance and operational controls, not on the management of severe accidents as required by the question.

### Criterion 499 (expected=yes, classified=no)

**Q:** Obsahuje bezpečnostní zpráva v kapitole 13.3.1 dostatečně podrobný popis systému provozních kontrol vybraných zařízení, včetně zdůvodnění naplnění leg...

**Agent:** Na základě provedených vyhledávání mohu nyní poskytnout odpověď. Z obsahu dokumentu (strana 9) a z kapitoly 13 je zřejmé, že **podkapitola 13.3.1 v dokumentu BZ_VR1 neexistuje**. 

Kapitola 13.3 "Příprava pracovníků reaktoru" začíná na straně 149, ale není dále členěna na podkapitoly 13.3.1, 13.3.2 ...

**Judge:** The agent explicitly states that subsection 13.3.1 does not exist in the document, indicating that the requirement described in the question is not met.

### Criterion 552 (expected=yes, classified=no)

**Q:** Obsahuje BZ v podkapitole 17.2 popis specifických aspektů systému řízení DP, včetně toho, jakým způsobem je zajištěno, že pokud je v dokumentaci proce...

**Agent:** Na základě získaných informací z dokumentu BZ_VR1 nyní mohu odpovědět na otázku. Našel jsem kapitolu 17.2 "Obsah limitů a podmínek" na straně 180, která obsahuje informace o kontrolních požadavcích. Také jsem našel kapitolu 18.2 "Odstupňovaný přístup" na straně 189-190, která popisuje aplikaci odstu...

**Judge:** Agent explicitly states that there is no mention of the requirement for control points to require the consent of a designated worker before proceeding, nor that this must be documented, indicating that the document does not satisfy the requirement.

## Cost & Timing

- **Total agent cost:** $13.03
- **Avg cost per query:** $0.5214
- **Avg time per query:** 81.3s
- **Total elapsed:** 33.9 min

