from pathlib import Path
import hashlib
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/chinese/CNSD_dataset.json"
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

users = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
records = []

for user_id, item in users.items():
    annotation = item.get("output", "")
    match = re.search(
        r'答案\s*[:：]\s*[“"\s]*([是否])',
        annotation,
    )
    if not match:
        raise ValueError("An annotation could not be parsed; review required.")

    history = item.get("inpput")
    if not isinstance(history, str):
        raise ValueError("A user history is missing or is not text.")

    # Remove numbered post markers, preserving the post content.
    history = re.sub(r"text_\d+\s*[:：]", "", history).strip()

    records.append({
        "text": history,
        "label": 1 if match.group(1) == "是" else 0,
        "source_split": "unsplit",
    })

frame = pd.DataFrame(records)
blank = frame["text"].eq("")
frame = frame.loc[~blank].copy()

groups = frame.groupby("text")["label"].nunique()
conflicts = set(groups[groups.gt(1)].index)
conflict_rows = frame["text"].isin(conflicts)

summary = {
    "unit": "One complete user history",
    "raw_users": len(users),
    "blank_histories_excluded": int(blank.sum()),
    "conflicting_unique_histories": len(conflicts),
    "conflicting_rows_excluded": int(conflict_rows.sum()),
}

frame = frame.loc[~conflict_rows].copy()
duplicates = frame["text"].duplicated()
summary["duplicate_histories_excluded_after_conflicts"] = int(
    duplicates.sum()
)
frame = frame.loc[~duplicates].copy()

frame["text_id"] = frame["text"].map(
    lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
)

summary["remaining_users"] = len(frame)
summary["remaining_label_counts"] = {
    str(k): int(v)
    for k, v in frame["label"].value_counts().sort_index().items()
}
summary["source_sha256"] = hashlib.sha256(
    SOURCE.read_bytes()
).hexdigest()
summary["model_input"] = (
    "User history only; annotation output and user IDs excluded"
)

frame[["text_id", "text", "label", "source_split"]].to_csv(
    OUT / "chinese_clean.csv", index=False
)
(REPORTS / "chinese_cleaning_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print("\nSaved local Chinese histories and aggregate cleaning report.")
