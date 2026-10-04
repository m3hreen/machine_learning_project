# Age Bias in Multilingual Mental-Health Text Classification

CSCI 3052U — Milestone 2 project workspace.

## Research question
Does changing age information change a model's mental-health classification when the underlying symptom evidence stays the same?

## Current progress
- Persian: Farsi Mental State and Emotional Needs dataset downloaded and inspected.
- Chinese: CNSocialDepress downloaded and inspected.
- Four additional languages: awaiting team selections and datasets.
- Planned total: two high-resource, two mid-resource, and two low-resource languages under one explicitly cited classification framework.
- Balanced pilot samples, age variants, model training, and evaluation are not yet completed.

## Setup
Use Python 3.10 or later. The initial inspection uses only the standard library.

```bash
python3 src/inspect_datasets.py
```

Raw files are already included in this local package under `data/raw/`. Git ignores these files. The ZIP also includes prior assignment materials under `references/`, which Git ignores.

## Data and labels
See `dataset_cards/` and `data/README.md`. These are research annotations, not confirmed clinical diagnoses. Chinese labels apply to user collections, while Persian labels are free-form mental-state annotations; post-level binary targets require documented review.

## Planned pilot
Aim for 50 positive and 50 reviewed comparison posts per language, subject to suitability. Include a no-age control and six age conditions. Review pre-existing age and life-stage cues; do not change symptoms. Keep every variant of a source post, and posts from the same known author, in the same train/validation/test split.

## Project layout
- `data/raw/`: original downloaded files (local only)
- `data/processed/`: reviewed samples and variants (local only until redistribution review)
- `dataset_cards/`: source, labels, counts, access and limitations
- `src/`: reproducible inspection and later processing
- `reports/`: initial findings and milestone checklist
- `references/`: original proposal and supplied assignment materials (local only)

## Repository
https://github.com/m3hreen/machine_learning_project

No files have been pushed by this package. Do not upload the whole ZIP to the public repository: extract it and use Git so `.gitignore` applies.
