"""Exploratory, paired age-prefix sensitivity on the existing validation split."""
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from run_milestone5_validation import (
    LANGUAGES, PROCESSED, REPORTS, SEED, VECTOR_SETTINGS, TfidfVectorizer,
)

AGES = (18, 25, 35, 45, 60, 75)
PREFIX = {
    "portuguese": "Idade",
    "turkish": "Yaş",
    "chinese": "年龄",
    "persian": "سن",
    "tamil": "வயது",
    "japanese": "年齢",
}
BUILDERS = {
    "logistic_regression": lambda c: LogisticRegression(
        C=c, solver="liblinear", random_state=SEED, max_iter=5000),
    "linear_svm": lambda c: LinearSVC(C=c, dual="auto", random_state=SEED, max_iter=5000),
}


def summarize(language, model, condition, reference, pred, baseline):
    if pred.shape != baseline.shape or len(pred) == 0:
        raise ValueError("Paired predictions must have matching nonempty shapes.")
    flips = int(np.count_nonzero(pred != baseline))
    ci = binomtest(flips, len(pred)).proportion_ci()
    return dict(language=language, model=model, condition=condition, reference=reference,
                n=len(pred), flips=flips, flip_rate=flips / len(pred),
                ci_low=ci.low, ci_high=ci.high,
                flips_0_to_1=int(np.count_nonzero((baseline == 0) & (pred == 1))),
                flips_1_to_0=int(np.count_nonzero((baseline == 1) & (pred == 0))))


def run_language(language, selected):
    path = PROCESSED / f"{language}_clean.csv"
    manifest = PROCESSED / f"{language}_split_manifest.csv"
    frame = pd.read_csv(path, keep_default_na=False).merge(
        pd.read_csv(manifest), on="text_id", validate="one_to_one", how="outer", indicator=True
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError(f"Split manifest does not match {language} data.")
    train = frame.loc[frame["split"].eq("train")]
    texts = frame.loc[frame["split"].eq("validation"), "text"]
    if texts.empty or texts.str.strip().eq("").any():
        raise ValueError(f"Expected nonempty validation texts for {language}.")

    vectorizer = TfidfVectorizer(**VECTOR_SETTINGS)
    train_vectors = vectorizer.fit_transform(train["text"])
    variants = {"original": vectorizer.transform(texts)}
    for condition, value in [("age_unspecified", "—"), *((str(age), str(age)) for age in AGES)]:
        variants[condition] = vectorizer.transform(f"{PREFIX[language]}: {value}\n" + texts)

    models = {"majority": None}
    for name, build in BUILDERS.items():
        c = float(selected.loc[(language, name), "c"])
        model = build(c).fit(train_vectors, train["label"])
        if np.max(model.n_iter_) >= model.max_iter:
            raise RuntimeError(f"{language} {name} reached the iteration limit.")
        models[name] = model

    rows = []
    for name, model in models.items():
        predictions = {
            condition: (model.predict(vectors) if model is not None
                        else np.full(len(texts), train["label"].mode().iloc[0]))
            for condition, vectors in variants.items()
        }
        for condition, pred in predictions.items():
            if condition == "original":
                continue
            references = ("original",) if condition == "age_unspecified" else ("original", "age_unspecified", "35")
            for reference in references:
                if condition != reference:
                    rows.append(summarize(language, name, condition, reference, pred, predictions[reference]))
    print(f"{language}: {len(texts):,} paired source records", flush=True)
    return rows, {
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def main():
    dest = REPORTS / "milestone5"
    selection_path = dest / "validation_results.csv"
    selected = pd.read_csv(selection_path).set_index(["language", "model"], verify_integrity=True)
    rows, inputs = [], {}
    for language in LANGUAGES:
        result, inputs[language] = run_language(language, selected)
        rows.extend(result)
    pd.DataFrame(rows).to_csv(dest / "age_sensitivity_results.csv", index=False)
    log = {
        "partition": "validation", "test_evaluated": False, "exploratory": True,
        "seed": SEED, "ages": AGES, "prefixes": PREFIX,
        "unspecified_value": "—", "primary_pilot_reference": "age_unspecified",
        "interval": "Exact 95% binomial interval for paired flip indicators, assuming independent source records",
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selected_c": {lang: {name: float(selected.loc[(lang, name), "c"]) for name in BUILDERS}
                       for lang in LANGUAGES},
        "svm_dual": "auto", "vectorizer": VECTOR_SETTINGS, "inputs": inputs,
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
    }
    (dest / "age_sensitivity_experiment_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
