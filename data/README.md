# Dataset access and handling

## Persian
Source: https://data.mendeley.com/datasets/ksmcbg69rr/1
Local original: `raw/persian/result.csv`
Dataset: Farsi Mental State and Emotional Needs Dataset for Psychological Text Mining, version 1, Zahra Jamalou, 7 July 2025.
DOI: 10.17632/ksmcbg69rr.1. Publisher licence: CC BY 4.0.
Retain attribution and document modifications if redistributing permitted derivatives. Review texts for identifying information before any publication.

## Chinese
Source: https://github.com/jytal/CNSocialDepress
Download: https://drive.google.com/file/d/1QCRZJAJsESg9WB6pmLqnDtbmASNhHaaN/view?usp=sharing
Local original: `raw/chinese/CNSD_dataset.json`
Released for academic research; an explicit redistribution licence has not been confirmed. Raw data and derivatives remain excluded from Git pending clarification.

## Sample handling
Original files are preserved. Do not assume missing depression labels mean absence of depression. Do not copy user-level Chinese labels onto every individual post. Annotation outputs must never be included in classifier inputs. Group related source posts and their variants when splitting data.
