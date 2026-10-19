
import pandas as pd
from pathlib import Path

# Load the Persian dataset
df = pd.read_csv("data/raw/result.csv")

# Display basic dataset information
print("FIRST FIVE ROWS:")
print(df.head().to_string())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET SIZE:")
print(df.shape)

# Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Check for duplicate and empty texts
texts = df["prompt"].dropna().astype(str).str.strip()
texts = texts[texts != ""]

print("\nNONEMPTY TEXTS:", len(texts))
print("DUPLICATE TEXTS:", texts.duplicated().sum())

# Identify all mental-state annotation columns
mental_cols = [
    col for col in df.columns
    if col.startswith("mental_state__")
]

# Create a working copy
clean_df = df.copy()

# Remove missing and blank texts
clean_df = clean_df.dropna(subset=["prompt"])

clean_df = clean_df[
    clean_df["prompt"].astype(str).str.strip() != ""
].copy()

# Create cleaned text for duplicate detection
clean_df["text_clean"] = (
    clean_df["prompt"].astype(str).str.strip()
)

# Remove exact duplicate texts
clean_df = clean_df.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).copy()

# Extract all mental-state annotations
def extract_labels(row):
    labels = []

    for col in mental_cols:
        value = row[col]

        if pd.notna(value):
            label = str(value).strip()

            if label and label not in labels:
                labels.append(label)

    return labels

clean_df["mental_health_labels"] = clean_df.apply(
    extract_labels,
    axis=1
)

# Count posts with mental-state annotations
clean_df["has_mental_state_label"] = (
    clean_df["mental_health_labels"].apply(
        lambda labels: len(labels) > 0
    )
)

print("\nCLEANED DATASET SIZE:")
print(clean_df.shape)

print("\nPOSTS WITH MENTAL-STATE LABELS:")
print(clean_df["has_mental_state_label"].sum())

print("\nPOSTS WITHOUT MENTAL-STATE LABELS:")
print((~clean_df["has_mental_state_label"]).sum())

# Count all mental-state labels
all_labels = clean_df["mental_health_labels"].explode()
all_labels = all_labels.dropna()

print("\nTOTAL UNIQUE MENTAL-STATE LABELS:")
print(all_labels.nunique())

print("\nTOP 30 MOST COMMON MENTAL-STATE LABELS:")
print(all_labels.value_counts().head(30).to_string())

# Count how many labels each post has
clean_df["label_count"] = (
    clean_df["mental_health_labels"].apply(len)
)

print("\nNUMBER OF LABELS PER POST:")
print(clean_df["label_count"].value_counts().sort_index())

# Save cleaned dataset
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

clean_df.to_csv(
    output_path / "persian_mental_health.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved: data/processed/persian_mental_health.csv")
