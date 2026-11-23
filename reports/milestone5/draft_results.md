# Milestone 5 draft results

## Validation model comparison

We compared a majority-class benchmark, logistic regression, and a linear SVM on the existing validation splits. Both learned models used the same character TF-IDF vocabulary, fitted only on training text. For each language and model, we evaluated C values of 0.1, 1, and 10 and selected the highest validation macro-F1, breaking ties with the smallest C. Macro-F1 weights the two classes equally and exposes performance losses on smaller classes. The table reports validation scores after tuning. Final test scores are pending.

| Language | Validation rows | Majority | Logistic regression | Linear SVM |
|---|---:|---:|---:|---:|
| Portuguese | 601 | 0.416 | 0.830 | 0.842 |
| Turkish | 7,807 | 0.335 | 0.974 | 0.976 |
| Chinese | 47 | 0.329 | 0.785 | 0.807 |
| Persian | 186 | 0.480 | 0.480 | 0.480 |
| Tamil | 250 | 0.390 | 0.996 | 0.996 |
| Japanese | 197 | 0.337 | 0.853 | 0.878 |

The linear SVM scored higher on four validation sets and tied on two. For Chinese, the SVM's row-bootstrap 95% interval was 0.680–0.914, reflecting uncertainty on only 47 validation rows. All reported intervals condition on fitted models and ignore tuning and any author, speaker, or near-duplicate dependence. They describe validation sampling variability under row independence and cannot establish that one model is reliably better.

## Error analysis and limitations

Both selected Persian models missed all 14 positive rows and made no false positives, matching the majority baseline. On Portuguese, the SVM missed 48 positives versus 60 for logistic regression, while false positives increased from 17 to 27. Both Tamil models missed one positive and made zero false positives. Tamil's near-ceiling score warrants checking whether speaker, dataset provenance, or repeated text provides shortcuts before interpreting it as strong generalization. Labels and source populations differ by language, so the scores characterize the individual datasets.

## Age sensitivity experiment pending

These validation results establish a classification comparison. The age-bias question requires a paired experiment using held-out source posts with reviewed age variants and an age-neutral control. Preserve symptom wording, keep all variants of each source post together, predeclare age conditions and the primary paired measure, and report per-language label flip rates with uncertainty resampled by source post. Review existing age cues and whether translated age statements have equivalent meanings before running the experiment. Final test evaluation, source-level leakage checks, and qualitative review of misclassifications remain pending.

## Reproduction

From the project root, run `python src/run_milestone5_validation.py` with the six cleaned CSVs and existing split manifests in `data/processed/`. The experiment log records input hashes, versions, and settings. The validation results file contains confidence intervals and confusion counts. The tuning file records all 36 model settings.
