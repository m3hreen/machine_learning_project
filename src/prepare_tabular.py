from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(frame, label_column, mapping):
    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str)

    # Match duplicates using trimmed text; preserve original text as input.
    frame["_key"] = frame["text"].str.strip()
    frame["label"] = frame[label_column].map(mapping)

    if frame["label"].isna().any():
        raise ValueError("Unexpected or missing label; check the source file.")

    raw_rows = len(frame)
    blank = frame["_key"].eq("")
    frame = frame.loc[~blank].copy()

    label_counts = frame.groupby("_key")["label"].nunique()
    conflicts = set(label_counts[label_counts.gt(1)].index)
    conflict_rows = frame["_key"].isin(conflicts)

    stats = {
        "raw_rows": raw_rows,
        "blank_rows_excluded": int(blank.sum()),
        "conflicting_unique_texts": len(conflicts),
        "conflicting_rows_excluded": int(conflict_rows.sum()),
    }

    frame = frame.loc[~conflict_rows].copy()

    # Train is loaded before test, so shared Turkish texts stay in train.
    duplicate = frame["_key"].duplicated(keep="first")
    stats["duplicate_rows_excluded_after_conflicts"] = int(duplicate.sum())
    frame = frame.loc[~duplicate].copy()

    frame["label"] = frame["label"].astype(int)
    frame["text_id"] = frame["_key"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    stats["remaining_rows"] = len(frame)
    stats["remaining_label_counts"] = {
        str(k): int(v)
        for k, v in frame["label"].value_counts().sort_index().items()
    }

    return frame[["text_id", "text", "label", "source_split"]], stats


def main():
    summary = {}

    # Portuguese: require exactly one CSV to avoid mixing releases.
    paths = sorted((ROOT / "data/raw/portuguese").glob("*.csv"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one Portuguese CSV.")

    path = paths[0]
    frame = pd.read_csv(path, sep=None, engine="python")
    frame["source_split"] = "unsplit"
    clean, stats = prepare(frame, "target", {0: 0, 1: 1})
    clean.to_csv(OUT / "portuguese_clean.csv", index=False)
    stats["source_sha256"] = {path.name: file_hash(path)}
    summary["portuguese"] = stats

    # Turkish: preserve the supplied train/test distinction.
    frames = []
    hashes = {}
    text_sets = {}

    for split in ["train", "test"]:
        path = ROOT / f"data/raw/turkish/{split}-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame["source_split"] = split
        texts = frame["text"].fillna("").astype(str).str.strip()
        text_sets[split] = set(texts[texts.ne("")])
        frames.append(frame)
        hashes[path.name] = file_hash(path)

    combined = pd.concat(frames, ignore_index=True)
    clean, stats = prepare(
        combined, "labels", {"Normal": 0, "Depression": 1}
    )

    train_ids = set(clean.loc[clean["source_split"].eq("train"), "text_id"])
    test_ids = set(clean.loc[clean["source_split"].eq("test"), "text_id"])
    assert train_ids.isdisjoint(test_ids), "Train/test text overlap remains."

    stats["original_shared_unique_texts"] = len(
        text_sets["train"] & text_sets["test"]
    )
    stats["remaining_source_split_counts"] = {
        str(k): int(v) for k, v in clean["source_split"].value_counts().items()
    }
    stats["source_sha256"] = hashes
    stats["overlap_policy"] = (
        "Exclude globally conflicting texts. Keep one copy of other texts; "
        "retain train copies and remove matching test copies."
    )

    clean.to_csv(OUT / "turkish_clean.csv", index=False)
    summary["turkish"] = stats

    path = REPORTS / "tabular_cleaning_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved local processed files and aggregate cleaning report.")


if __name__ == "__main__":
    main()
