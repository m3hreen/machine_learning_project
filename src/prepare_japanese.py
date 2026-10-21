import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/japanese/twitter_Japanese.csv"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
if not {"tweet", "label"}.issubset(df.columns):
    raise ValueError("Expected tweet and label columns.")

raw_rows = len(df)
df["label"] = df["label"].str.strip()
if not df["label"].isin(["0", "1"]).all():
    raise ValueError("Unexpected labels; inspect before continuing.")

df["text"] = df["tweet"]
df["_key"] = df["text"].str.strip()

blank_count = int(df["_key"].eq("").sum())
df = df[df["_key"].ne("")].copy()

label_counts = df.groupby("_key")["label"].nunique()
conflicts = set(label_counts[label_counts.gt(1)].index)
conflicting_rows = int(df["_key"].isin(conflicts).sum())
df = df[~df["_key"].isin(conflicts)].copy()

duplicate_count = int(df["_key"].duplicated().sum())
df = df.drop_duplicates("_key").copy()

df["text_id"] = df["_key"].map(
    lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
)
df["label"] = df["label"].astype(int)
df["source_split"] = "unsplit"

df[["text_id", "text", "label", "source_split"]].to_csv(
    PROCESSED / "japanese_clean.csv", index=False
)

summary = {
    "raw_rows": raw_rows,
    "blank_rows_excluded": blank_count,
    "conflicting_unique_texts": len(conflicts),
    "conflicting_rows_excluded": conflicting_rows,
    "duplicate_rows_excluded_after_conflicts": duplicate_count,
    "remaining_rows": len(df),
    "remaining_label_counts": {
        str(label): int(count)
        for label, count in df["label"].value_counts().sort_index().items()
    },
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "unit": "One Japanese tweet",
    "target": "Released lexicon-derived depression-related label",
    "label_1": "Depression-related according to the source lexicon rule",
    "label_0": "Non-depression-related according to the source lexicon rule",
    "limitations": [
        "Automatic keyword labels are not clinical diagnoses.",
        "Author-independent splitting cannot be verified from this CSV.",
        "Near-duplicates may remain.",
        "Human review for symptom evidence and age-edit suitability is pending."
    ]
}

(REPORTS / "japanese_cleaning_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nSaved processed Japanese data; raw file unchanged.")
