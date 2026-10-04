import csv
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

# Persian
with (ROOT / "data/raw/persian/result.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

fields = [key for key in rows[0] if key.startswith("mental_state")]
texts = {}

for row in rows:
    text = (row.get("prompt") or "").strip()
    if text:
        texts.setdefault(text, []).append(row)

def has_depression(row, exact=True):
    labels = [(row.get(key) or "").strip() for key in fields]
    if exact:
        return "افسردگی" in labels
    return any("افسرد" in label for label in labels)

persian = {
    "total_rows": len(rows),
    "nonblank_text_rows": sum(
        bool((row.get("prompt") or "").strip()) for row in rows
    ),
    "unique_nonblank_texts": len(texts),
    "exact_depression_unique_texts": sum(
        any(has_depression(row) for row in group)
        for group in texts.values()
    ),
    "depression_related_unique_texts": sum(
        any(has_depression(row, exact=False) for row in group)
        for group in texts.values()
    ),
}

# Chinese
with (ROOT / "data/raw/chinese/CNSD_dataset.json").open(
    encoding="utf-8-sig"
) as f:
    users = json.load(f)

labels = Counter()
posts = 0

for item in users.values():
    match = re.search(
        r'答案\s*[:：]\s*[“"\s]*([是否])',
        item["output"],
    )
    labels[match.group(1) if match else "unparsed"] += 1
    posts += len(re.findall(r"text_\d+\s*[:：]", item["inpput"]))

chinese = {
    "users": len(users),
    "user_labels": dict(labels),
    "numbered_post_markers": posts,
}

print(json.dumps(
    {"persian": persian, "chinese": chinese},
    ensure_ascii=False,
    indent=2,
))
