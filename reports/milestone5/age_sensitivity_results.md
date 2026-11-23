# Milestone 5 — Exploratory age sensitivity

We used the existing validation splits for Portuguese, Turkish, Chinese, Persian, Tamil, and Japanese, comprising 9,088 source records. We refit logistic regression and the linear SVM on the original training data with the settings selected during validation. The majority-class model served as a constant control. The TF-IDF vocabulary and weights were fitted on training texts only.

For each validation record, we kept the source text unchanged. We compared the original input with a language-localized age field containing a dash, and with six versions containing ages 18, 25, 35, 45, 60, and 75. For example, the Portuguese field was `Idade: —` or `Idade: 45`. The primary pilot comparison was each numbered age against the unspecified-age field. We also compared each condition with the original input and each numbered age with 35. Age 35 was a fixed reference for the exploratory number-to-number comparisons.

The flip rate is the number of paired source records whose predicted class changed divided by the number of paired records. The CSV reports both flip directions and exact 95% binomial intervals assuming independent source records. The table gives the largest observed rate across the six age values for each model. These maxima are descriptive summaries of multiple comparisons.

| Language | Paired records | Logistic regression | Linear SVM |
|---|---:|---:|---:|
| Portuguese | 601 | 7/601 (1.16%), age 45 | 8/601 (1.33%), age 45 |
| Turkish | 7,807 | 45/7,807 (0.58%), age 60 | 47/7,807 (0.60%), age 60 |
| Chinese | 47 | 0/47 | 1/47 (2.13%), age 18 |
| Persian | 186 | 0/186 | 0/186 |
| Tamil | 250 | 0/250 | 0/250 |
| Japanese | 197 | 3/197 (1.52%), age 35 | 5/197 (2.54%), age 35 |

The majority baseline never flipped. In Japanese, the SVM also changed 8 of 197 predictions (4.06%) between age 18 and age 35 with the same prefix and source text. In Portuguese, age 45 changed 29 of 601 SVM predictions relative to the unmodified original (4.83%) and 8 of 601 relative to the unspecified-age field (1.33%). The reference choice changes the estimate and should be reported alongside the rate.

These results show sensitivity to the exact age-field strings tested. Character TF-IDF can respond to digit patterns and formatting. Original texts were not screened for pre-existing age or life-stage cues, translations were not independently reviewed, and Chinese inputs are complete user histories. The result does not establish real-world age bias, and zero observed flips do not establish age invariance. The final test split remains reserved. A confirmatory experiment would require reviewed examples, equivalent age wording, and a prespecified analysis.

To reproduce, run `python src/run_age_sensitivity.py` from the project root with the existing datasets, split manifests, and Milestone 5 validation results. The script requires scikit-learn 1.3 or later for the explicit SVM solver setting. The output CSV and experiment log contain aggregate results and provenance.
