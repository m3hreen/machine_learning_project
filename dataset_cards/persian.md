# Persian dataset card — provisional

- Name: Farsi Mental State and Emotional Needs Dataset for Psychological Text Mining.
- Source: https://data.mendeley.com/datasets/ksmcbg69rr/1
- Contributor: Zahra Jamalou. Version 1; published 7 July 2025.
- Licence: CC BY 4.0, as listed by publisher.
- Original text field: `prompt` in supplied CSV.
- Annotation fields: `mental_state__001` to `mental_state__006`, behavior-pattern and emotional-need columns.
- Publisher-reported annotation: manual annotation by a native Farsi expert with psychology and AI background.
- Publisher-reported source: Persian online mental-health forums and social discussion platforms; detailed collection sources and reproducible annotation guidelines are incomplete.
- Clinical diagnostic validation: not established.

## Verified local counts
1,022 rows; 993 nonblank text rows; 990 unique nonblank texts; 29 rows missing text; three repeated text values with differing mental-state annotations.
72 unique texts have the exact depression label `افسردگی`; 125 unique texts contain a depression-related label, including qualified and uncertain variants. These are candidate counts, not final positive class counts.

## Proposed binary target
Depression annotation present versus a reviewed comparison class. Exact-label positives are the initial candidate pool. Absence of the label does not establish a clinically negative case. Label mapping and comparison selection must be documented before sampling.

## Pilot feasibility and expansion
50 positive candidates appear numerically feasible, subject to age-edit suitability and review. A final 50-post comparison pool is not yet selected. Other states include anxiety, hopelessness, anger, loneliness and suicidal thoughts; these are annotations rather than independently validated disorder datasets.

## Open issues
Resolve duplicates and blank rows; review label variants; check age cues, privacy and post suitability; record negative-class criteria and language-resource framework.
