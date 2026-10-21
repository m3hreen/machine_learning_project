from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/tamil/tamil_audio_dataset.csv"
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(SOURCE, keep_default_na=False)
summary = {"raw_rows": len(frame)}

frame["text"] = frame["audio_text"].astype(str)
frame["_key"] = frame["text"].str.strip()
frame["_label"] = frame["label"].astype(str).str.strip().str.lower()

# Only accept the documented source labels; do not guess corrections.
valid_labels = frame["_label"].isin(["dep", "ndep"])
summary["invalid_label_rows_excluded"] = int((~valid_labels).sum())
frame = frame.loc[valid_labels].copy()

blank = frame["_key"].eq("")
summary["blank_rows_excluded_after_label_check"] = int(blank.sum())
frame = frame.loc[~blank].copy()
frame["label"] = frame["_label"].map({"dep": 1, "ndep": 0})

groups = frame.groupby("_key")["label"].nunique()
conflicts = set(groups[groups.gt(1)].index)
conflict_rows = frame["_key"].isin(conflicts)

summary["conflicting_unique_texts"] = len(conflicts)
summary["conflicting_rows_excluded"] = int(conflict_rows.sum())
frame = frame.loc[~conflict_rows].copy()

duplicates = frame["_key"].duplicated()
summary["duplicate_rows_excluded_after_conflicts"] = int(duplicates.sum())
frame = frame.loc[~duplicates].copy()

frame["text_id"] = frame["_key"].map(
    lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
)
frame["source_split"] = "unsplit"

summary["remaining_rows"] = len(frame)
summary["remaining_label_counts"] = {
    str(k): int(v)
    for k, v in frame["label"].value_counts().sort_index().items()
}
summary["source_sha256"] = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()
summary["unit"] = "Automatic transcript of a speech utterance"
summary["target"] = "Released dep/ndep recording label"
summary["speaker_ids_available"] = False
summary["limitations"] = [
    "Whisper transcription errors have not been manually reviewed.",
    "Recording labels are not independently validated text diagnoses.",
    "Speaker-independent splitting cannot be verified from this CSV.",
    "Repeated sentences and near-duplicates may remain.",
    "Dataset-use and derivative permissions require confirmation.",
    "Not yet a reviewed sample for the age-intervention study.",
]

frame[["text_id", "text", "label", "source_split"]].to_csv(
    OUT / "tamil_clean.csv", index=False
)
(REPORTS / "tamil_cleaning_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print("\nSaved processed Tamil candidates; raw files unchanged.")
