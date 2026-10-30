# Validation diagnostics

These diagnostics refit the existing character TF-IDF + Logistic Regression baseline.
Full-training validation metrics must match the saved baseline within 0.00005.
No reserved test predictions or test metrics are computed. Baseline reports are not overwritten.

- Learning curves show training/validation log loss versus training sample count,
  not epochs or solver iterations. Each point refits TF-IDF on its training subset.
  One seed and a fixed validation set are used; no uncertainty intervals are claimed.
- Confusion matrices use released/provisional labels, not confirmed clinical diagnoses.
- ROC AUC and average precision measure rankings; the baseline's default decision
  threshold is unchanged. Average precision is not trapezoidal PR area.
- Feature coefficients show associations, not causal importance. Feature IDs map
  to native character n-grams in each top_features.csv file. Weights should not
  be compared directly across languages.
- Up to two high-confidence examples per available TN/FP/FN/TP group are saved
  only in ignored data/processed/diagnostics/. These are illustrative, selected
  examples, not a representative error sample. Empty groups are not invented.
  Do not publish identifiable or sensitive text without appropriate review.

These are classification diagnostics, not evidence of age bias. Existing label,
speaker/author overlap, near-duplicate and provenance limitations still apply.
