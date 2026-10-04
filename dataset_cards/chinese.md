# Chinese dataset card — provisional

- Name: CNSocialDepress (CNSD).
- Source: https://github.com/jytal/CNSocialDepress
- Original source corpus: Sina Weibo Depression Dataset (SWDD).
- File: `CNSD_dataset.json`.
- Language: inspected samples use standard written Chinese consistent with Mandarin, primarily simplified characters. Authors' spoken dialects are not established by text.
- Unit of released binary annotation: user-level collection of posts.
- Fields: `inpput` (released spelling) and `output` (label plus structured analysis).
- Labels: `是` means depression-risk user; `否` means comparison/non-depression-risk user.
- Publisher-reported annotation: four senior researchers with psychology and depression-scale expertise; blinded to original SWDD binary labels, with cross-checks and guideline revision.
- Clinical diagnosis: not established; labels are research-oriented depression-risk annotations.
- Access: supplied file successfully downloaded and inspected.
- Licence: academic research intended; explicit redistribution permission has not been confirmed.

## Verified local counts
233 users: 116 positive and 117 comparison. 44,178 numbered post markers. No empty input collections. Output labels agree with user-ID prefixes. 224 user annotations contain numbered evidence references.

## Proposed post pilot
Use annotation evidence references to locate candidate symptom-relevant posts, then review individual posts and suitable comparisons. Do not infer that all posts by a positive user are symptom-positive. There are enough users for a 50 + 50 user-level sample, but a suitable post-level sample is not yet verified.

## Expansion and limitations
Six analysis dimensions cover emotions, psychological state, symptoms, external causes, medical expressions and language patterns. These are not separate binary diagnosis datasets. User grouping and age-cue checks are required; avoid feeding annotation outputs into the model.
