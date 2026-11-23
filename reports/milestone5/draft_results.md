# Milestone 5 draft results

## Validation model comparison

We compared a majority-class benchmark, logistic regression, and a linear SVM on the existing validation splits. Both learned models used the same character TF-IDF vocabulary, fitted only on training text. For each language and model, we evaluated C values of 0.1, 1, and 10 and selected the highest validation macro-F1, breaking ties with the smallest C. Macro-F1 weights the two classes equally and exposes performance losses on smaller classes. The table reports validation scores after tuning. Final test scores for Chinese and Persian are reported in the held-out test section below; test scores for the other four languages are pending.

| Language | Validation rows | Majority | Logistic regression | Linear SVM |
|---|---:|---:|---:|---:|
| Portuguese | 601 | 0.416 | 0.830 | 0.842 |
| Turkish | 7,807 | 0.335 | 0.974 | 0.976 |
| Chinese | 47 | 0.329 | 0.785 | 0.807 |
| Persian | 186 | 0.480 | 0.480 | 0.480 |
| Tamil | 250 | 0.390 | 0.996 | 0.996 |
| Japanese | 197 | 0.337 | 0.853 | 0.878 |

The linear SVM scored higher on four validation sets and tied on two. For Chinese, the SVM's row-bootstrap 95% interval was 0.680–0.914, reflecting uncertainty on only 47 validation rows. All reported intervals condition on fitted models and ignore tuning and any author, speaker, or near-duplicate dependence. They describe validation sampling variability under row independence and cannot establish that one model is reliably better.

## Held-out test evaluation

We evaluated the majority-class benchmark, logistic regression and the linear SVM once on the reserved test splits for Chinese (47 records) and Persian (186 records), using the settings selected on validation and no retuning. The vocabulary and weights were fitted on training text only. The selected C values were 10 (logistic regression) and 1 (SVM) for Chinese, and 0.1 for both models for Persian. Portuguese, Turkish, Tamil and Japanese were not evaluated at this stage, so these results do not cover all six languages.

| Language | Test rows | Majority | Logistic regression | Linear SVM |
|---|---:|---:|---:|---:|
| Chinese | 47 | 0.338 | 0.892 | 0.892 |
| Persian | 186 | 0.480 | 0.480 | 0.480 |

The table reports test macro-F1. On Chinese, both models scored 0.892 (row-bootstrap 95% interval 0.786–0.973). Each model made 5 false negatives and no false positives, giving positive precision of 1.00 and recall of 0.78 (18 of 23 positives found). With only 47 test records, the interval is wide and the two models cannot be distinguished. On Persian, both learned models predicted the comparison class for every record, missing all 14 positives and making no false positives. Their predictions matched the majority baseline, and the 0.925 accuracy reflects class imbalance rather than learned signal (balanced accuracy 0.50). Intervals condition on the fitted models and ignore tuning and any author or near-duplicate dependence. The Chinese records are complete user histories, and the Persian labels are provisional annotation-based labels, so the scores describe these datasets and are not clinical performance estimates.

## Error analysis and limitations

Both selected Persian models missed all 14 positive rows and made no false positives, matching the majority baseline. On Portuguese, the SVM missed 48 positives versus 60 for logistic regression, while false positives increased from 17 to 27. Both Tamil models missed one positive and made zero false positives. Tamil's near-ceiling score warrants checking whether speaker, dataset provenance, or repeated text provides shortcuts before interpreting it as strong generalization. Labels and source populations differ by language, so the scores characterize the individual datasets.

## Exploratory age sensitivity experiment

We conducted an exploratory paired age-prefix sensitivity experiment using the existing validation splits across all six languages. Each validation record was evaluated with an unspecified-age prefix and localized age prefixes for ages 18, 25, 35, 45, 60, and 75. Logistic regression and linear SVM predictions were compared against the unspecified-age condition. The models and TF-IDF vocabulary were fitted using training data only. The reserved test sets were not evaluated.

Prediction changes were generally uncommon, but varied across languages and age conditions. The largest flip rates against the unspecified-age condition were:

- Portuguese: 1.16% for logistic regression and 1.33% for the SVM.
- Turkish: 0.58% for logistic regression and 0.60% for the SVM.
- Chinese: 0% for logistic regression and 2.13% for the SVM.
- Persian: 0% for both models.
- Tamil: 0% for both models.
- Japanese: 1.52% for logistic regression and 2.54% for the SVM.

These findings are exploratory, not evidence that the models are free from age bias. The experiment measures prediction sensitivity to added age prefixes, not real-world fairness across age groups. Original records were not manually screened for existing age cues, and localized prefixes were not independently reviewed for linguistic equivalence. The Chinese dataset contains complete user histories rather than individual posts, and its small validation sample limits interpretation. Zero observed prediction changes do not establish invariance.

The age-cue audit and prefix-vocabulary coverage checks provide additional context and are documented in `age_cue_audit.md` and the age-sensitivity experiment report. A stronger confirmatory study would require manually reviewed examples, validated equivalent age wording, and a prespecified evaluation design.

See `age_sensitivity_results.md` and `age_sensitivity_experiment_log.json` for detailed results and experiment settings.

## Original age-cue audit and localized prefixes

An automated exploratory audit searched the original cleaned records for explicit numeric ages and life-stage expressions. Detected cues appeared in 1.40% of Portuguese records, 1.79% of Turkish records, 88.41% of Chinese records, 14.64% of Persian records, 0.40% of Tamil records, and 1.72% of Japanese records.

The Chinese result represents complete user histories, not individual posts. The audit used language-specific regular expressions and was not manually validated. It may miss indirect age references, spelling variations, and local numeral forms; matches may also include false positives. A record with no detected cue should not be assumed to contain no age information.

The age-prefix experiment used these localized labels: Portuguese `Idade`, Turkish `Yaş`, Chinese `年龄`, Persian `سن`, Tamil `வயது`, and Japanese `年齢`. A training-vocabulary check found some prefix n-grams represented in every language, but coverage differed. This check does not establish that the phrases are equally natural, culturally equivalent, or linguistically validated. Human review is needed before treating the age conditions as equivalent.

## Reproduction

From the project root, run `python src/run_milestone5_validation.py` with the six cleaned CSVs and existing split manifests in `data/processed/`. The experiment log records input hashes, versions, and settings. The validation results file contains confidence intervals and confusion counts. The tuning file records all 36 model settings. The held-out test results were produced with `python src/run_milestone5_test.py --languages chinese persian`, which refuses to run on a language whose test results already exist and logs the selected C values, input hashes and library versions.



