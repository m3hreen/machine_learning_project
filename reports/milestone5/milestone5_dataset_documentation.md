# Milestone 5 dataset documentation

## Sources, labels, and cleaning

Label 1 is the positive class for each dataset. Label 0 is its source-specific comparison class. The labels represent different targets and do not establish clinical diagnoses.

| Language | Source and input unit | Binary labels | Cleaning and exclusions |
|---|---|---|---|
| Portuguese | `dataset_ideacao (3).csv`, one text per row. Local notes suggest version 2. | 1 = suicidal ideation, 0 = comparison. | 3,788 raw rows. Removed 16 rows from eight conflicting-label text groups and 768 further duplicates. |
| Turkish | Publisher train and test Parquet files, one text per row. | 1 = Depression, 0 = Normal. | 57,597 raw rows. Removed 10 blank rows, 11 rows with conflicting labels, and 2,182 further duplicates. For texts present in both source partitions, retained the training copy where labels agreed. |
| Chinese | CNSocialDepress, `CNSD_dataset.json`, one complete user history per row. | 1 = user-level positive annotation, 0 = user-level comparison annotation. | Kept all 233 histories. Excluded user IDs and annotation outputs from classifier inputs. Individual posts do not inherit the user label. |
| Persian | Farsi Mental State and Emotional Needs Dataset v1, `result.csv`, one text per row. | 1 = exact depression annotation (`افسردگی`). 0 = nonempty annotations with no depression stem (`افسرد`). | 1,022 raw rows. Removed 29 blank rows, six conflicting rows, and 58 texts excluded by the provisional label rule. |
| Tamil | `tamil_audio_dataset.csv`, one automatic transcript of a speech utterance per row. | 1 = released recording label `dep`, 0 = `ndep`. | 1,375 raw rows. Removed two invalid labels, eight blank transcripts, and 115 exact duplicates after trimming. |
| Japanese | `twitter_Japanese.csv`, one tweet per row. | 1 = depression-related under the source lexicon rule, 0 = comparison under that rule. | Removed 13 exact duplicates after trimming from 1,000 raw rows. |

The local project notes identify the [Chinese source repository](https://github.com/jytal/CNSocialDepress) and the [Persian dataset record](https://data.mendeley.com/datasets/ksmcbg69rr/1). They record CC BY 4.0 for Persian and leave Chinese redistribution terms unconfirmed. The Portuguese release version and the upstream citation and reuse terms for Portuguese, Turkish, Tamil, and Japanese still need confirmation.

## Splits and class counts

The existing split manifests use seed 42. Portuguese, Chinese, Persian, Tamil, and Japanese use stratified splits of approximately 60% train, 20% validation, and 20% test. Turkish retains its cleaned publisher test partition and splits the cleaned publisher training partition 80/20 into stratified train and validation sets. Milestone 5 used training for feature fitting and validation for tuning. Test rows remained reserved. Each count pair below is **label 0 / label 1**.

| Language | Clean total | All classes | Train | Validation | Reserved test |
|---|---:|---:|---:|---:|---:|
| Portuguese | 3,004 | 2,141 / 863 | 1,284 / 518 | 428 / 173 | 429 / 172 |
| Turkish | 55,394 | 27,847 / 27,547 | 15,761 / 15,467 | 3,940 / 3,867 | 8,146 / 8,213 |
| Chinese | 233 | 117 / 116 | 70 / 69 | 23 / 24 | 24 / 23 |
| Persian | 929 | 858 / 71 | 514 / 43 | 172 / 14 | 172 / 14 |
| Tamil | 1,250 | 800 / 450 | 480 / 270 | 160 / 90 | 160 / 90 |
| Japanese | 987 | 500 / 487 | 300 / 292 | 100 / 97 | 100 / 98 |

## Scope and limitations

Every cleaned record matched exactly one manifest entry, and no exact trimmed text appeared across different splits. This check used whole input records, so it does not establish that individual Chinese posts, near-duplicate texts, authors, or speakers are disjoint. The Tamil CSV lacks speaker IDs and its automatic transcripts have not been manually reviewed. Japanese lexicon labels may reward vocabulary overlap, and Portuguese suicidal ideation is distinct from the other datasets' depression-related targets. The Persian comparison class indicates absence of a particular annotation pattern. Review age cues and symptom evidence separately before selecting posts for the paired age-sensitivity experiment.

Counts were calculated from `data/processed/*_clean.csv` and `data/processed/*_split_manifest.csv`. Cleaning attrition comes from the summaries in `reports/` and the preprocessing code in `src/`.
