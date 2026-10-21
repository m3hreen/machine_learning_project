from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/persian/result.csv"
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(SOURCE, keep_default_na=False)
columns = [c for c in frame.columns if c.startswith("mental_state")]

if not columns:
    raise ValueError("No mental-state annotation columns found.")

frame["_key"] = frame["prompt"].astype(str).str.strip()
frame["_annotations"] = frame[columns].apply(
    lambda row: tuple(sorted({
        str(value).strip()
        for value in row
        if str(value).strip()
    })),
    axis=1,
)

summary = {"raw_rows": len(frame)}
blank = frame["_key"].eq("")
summary["blank_text_rows_excluded"] = int(blank.sum())
frame = frame.loc[~blank].copy()

# Exclude a repeated text if its mental-state annotation sets disagree.
sets_per_text = frame.groupby("_key")["_annotations"].nunique()
conflicts = set(sets_per_text[sets_per_text.gt(1)].index)
conflict_rows = frame["_key"].isin(conflicts)

summary["conflicting_unique_texts"] = len(conflicts)
summary["conflicting_rows_excluded"] = int(conflict_rows.sum())
frame = frame.loc[~conflict_rows].copy()

duplicates = frame["_key"].duplicated()
summary["duplicate_rows_excluded_after_conflicts"] = int(duplicates.sum())
frame = frame.loc[~duplicates].copy()


def provisional_label(annotations):
    if not annotations:
        return -1
    if "افسردگی" in annotations:
        return 1
    if any("افسرد" in value for value in annotations):
        return -1
    return 0


frame["label"] = frame["_annotations"].map(provisional_label)
summary["unique_texts_excluded_by_label_rule"] = int(
    frame["label"].eq(-1).sum()
)
frame = frame.loc[frame["label"].isin([0, 1])].copy()

if frame["label"].nunique() != 2:
    raise ValueError("The provisional rule did not produce both classes.")

frame["text"] = frame["prompt"].astype(str)
frame["text_id"] = frame["_key"].map(
    lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
)
frame["source_split"] = "unsplit"

summary["remaining_rows"] = len(frame)
summary["remaining_label_counts"] = {
    str(k): int(v)
    for k, v in frame["label"].value_counts().sort_index().items()
}
summary["target"] = (
    "Provisional exact depression-annotation presence, "
    "not clinical depression status"
)
summary["label_1"] = "Exact mental-state annotation: افسردگی"
summary["label_0"] = (
    "Nonempty mental-state annotations with no depression stem افسرد"
)
summary["source_sha256"] = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()
summary["review_status"] = (
    "Rule-derived labels; human review pending. "
    "Not the final age-study sample."
)

frame[["text_id", "text", "label", "source_split"]].to_csv(
    OUT / "persian_clean.csv", index=False
)
(REPORTS / "persian_cleaning_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nSaved provisional Persian data; raw file unchanged.")
