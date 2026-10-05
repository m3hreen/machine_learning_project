# Portuguese and Turkish initial EDA

Counts obtained from local files using trimmed exact-text matching.
Raw files remain unchanged.

## Portuguese
- File: dataset_ideacao (3).csv
- Raw rows: 3,788
- Labels: 1 = suicidal ideation; 0 = comparison.
- Blank texts: 0
- Unique nonblank texts: 3,012
- Excess duplicate rows: 776
- Unique texts with conflicting labels: 8
- After excluding conflicts and deduplicating: 3,004 texts.
- Remaining labels: 863 positive; 2,141 comparison.
- Counts match the documented version 2; version confirmation pending.

## Turkish
- Train rows: 40,317; test rows: 17,280.
- Labels: Depression and Normal.
- Missing or blank texts: 6 train; 4 test.
- Unique texts shared across train and test: 572.
- Combined unique texts with conflicting labels: 5.
- Combined deduplicated texts excluding conflicts: 55,394.
- Remaining labels: 27,547 Depression; 27,847 Normal.
- Combined counts are for inspection, not a finalized data split.
- Annotation provenance and clinical validity remain unconfirmed.

## Sampling and splitting
Exclude blank texts and conflicting-label text groups.
Deduplicate before sampling.
Keep matching texts and known authors in the same partition.
Review symptom evidence and age-edit suitability before selecting
50 positive and 50 comparison posts per language.

## Pending access
French: request email sent; awaiting access approval.
Tamil: no usable raw dataset obtained yet.
