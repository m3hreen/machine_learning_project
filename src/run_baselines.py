from pathlib import Path
import hashlib
import json
import platform
import time

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
SEED = 42


def split_data(frame, language):
    if language == "turkish":
        # Preserve the cleaned publisher test partition.
        pool = frame.loc[frame["source_split"].eq("train")].copy()
        test = frame.loc[frame["source_split"].eq("test")].copy()
        train, validation = train_test_split(
            pool, test_size=0.20, random_state=SEED,
            stratify=pool["label"],
        )
    else:
        train, remainder = train_test_split(
            frame, test_size=0.40, random_state=SEED,
            stratify=frame["label"],
        )
        validation, test = train_test_split(
            remainder, test_size=0.50, random_state=SEED,
            stratify=remainder["label"],
        )

    parts = {"train": train, "validation": validation, "test": test}
    ids = {name: set(part["text_id"]) for name, part in parts.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])

    # Save assignments locally for reproducibility.
    manifests = []
    for name, part in parts.items():
        assignment = part[["text_id"]].copy()
        assignment["split"] = name
        manifests.append(assignment)
    pd.concat(manifests).to_csv(
        PROCESSED / f"{language}_split_manifest.csv", index=False
    )
    return parts


def scores(truth, prediction):
    return {
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(
            truth, prediction, labels=[0, 1],
            average="macro", zero_division=0,
        ),
        "positive_precision": precision_score(
            truth, prediction, zero_division=0
        ),
        "positive_recall": recall_score(
            truth, prediction, zero_division=0
        ),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = []
    details = {
        "seed": SEED,
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "label_status": "Released labels; individual-text review pending",
        "languages": {},
    }

    for language in ['portuguese', 'turkish', 'chinese', 'persian', 'tamil', 'japanese']:
        path = PROCESSED / f"{language}_clean.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        assert frame["label"].isin([0, 1]).all()
        assert frame["text_id"].is_unique
        assert frame["text"].str.strip().ne("").all()

        parts = split_data(frame, language)
        train = parts["train"]
        validation = parts["validation"]

        language_details = {
            "processed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "split_class_counts": {
                name: {
                    str(k): int(v)
                    for k, v in part["label"].value_counts().items()
                }
                for name, part in parts.items()
            },
            "models": {},
        }

        # Naive benchmark: predict the most common training label.
        started = time.perf_counter()
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit([[0]] * len(train), train["label"])
        prediction = dummy.predict([[0]] * len(validation))
        elapsed = time.perf_counter() - started
        models = [
            ("majority_class", prediction, elapsed, {"strategy": "most_frequent"})
        ]

        # Fit vocabulary and IDF on training text only.
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=50000,
            lowercase=False,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(train["text"])
        x_validation = vectorizer.transform(validation["text"])

        classifier = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=SEED,
        )
        classifier.fit(x_train, train["label"])
        prediction = classifier.predict(x_validation)
        elapsed = time.perf_counter() - started

        if classifier.n_iter_.max() >= classifier.max_iter:
            raise RuntimeError("Logistic regression reached its iteration limit.")

        models.append((
            "char_tfidf_logistic_regression", prediction, elapsed,
            {
                "analyzer": "char",
                "ngram_range": [2, 4],
                "min_df": 2,
                "max_features": 50000,
                "actual_features": len(vectorizer.vocabulary_),
                "lowercase": False,
                "sublinear_tf": True,
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
            },
        ))

        for name, prediction, elapsed, settings in models:
            row = {
                "language": language,
                "model": name,
                "partition": "validation",
                "train_n": len(train),
                "validation_n": len(validation),
                "reserved_test_n": len(parts["test"]),
                **scores(validation["label"], prediction),
                "fit_and_predict_seconds": elapsed,
            }
            rows.append(row)
            language_details["models"][name] = {
                "settings": settings,
                "confusion_matrix_labels": [0, 1],
                "confusion_matrix": confusion_matrix(
                    validation["label"], prediction, labels=[0, 1]
                ).tolist(),
            }

        details["languages"][language] = language_details

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "baseline_validation.csv", index=False)
    (REPORTS / "baseline_run.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    print(table.round(4).to_string(index=False))
    print("\nSaved reports/baseline_validation.csv and reports/baseline_run.json")
    print("Test sets remain unevaluated.")


if __name__ == "__main__":
    main()
