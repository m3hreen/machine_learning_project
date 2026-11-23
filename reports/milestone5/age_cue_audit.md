# Milestone 5: Original-record age-cue audit

## Method

A reproducible regex screening pass was applied to the six cleaned datasets.
The audit counts records containing explicit numerical age expressions,
direct age/life-stage vocabulary, or either type of cue.

No original text or identifying examples are included in this report.

## Results

| Language | Records checked | Explicit age | Life-stage terms | Either cue | No detected cue |
|---|---:|---:|---:|---:|---:|
| Portuguese | 3004 | 19 | 26 | 42 | 2962 |
| Turkish | 55394 | 171 | 846 | 992 | 54402 |
| Chinese | 233 | 99 | 200 | 206 | 27 |
| Persian | 929 | 102 | 47 | 136 | 793 |
| Tamil | 1250 | 0 | 5 | 5 | 1245 |
| Japanese | 987 | 8 | 10 | 17 | 970 |

## Limitations

- Regex screening can miss indirect, misspelled, or grammatically varied age references.
- Some terms are ambiguous without context and may produce false positives.
- The audit does not establish a person's actual age.
- Chinese records are complete user histories, not individual posts; counts are at the history level.
- The results require careful interpretation and, where feasible, native-speaker review.
- A record with no detected cue is not proof that it contains no age information.

This is an exploratory screening audit, not a clinical or demographic assessment.
