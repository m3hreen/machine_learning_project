"""Compare models on the existing validation splits. Leave test data reserved."""
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from run_baselines import PROCESSED, REPORTS, SEED, scores

LANGUAGES = ("portuguese", "turkish", "chinese", "persian", "tamil", "japanese")
C_VALUES = (0.1, 1.0, 10.0)
VECTOR_SETTINGS = dict(analyzer="char", ngram_range=(2, 4), min_df=2,
                       max_features=50000, lowercase=False, sublinear_tf=True)


def bootstrap_macro_f1(truth, prediction):
    """Estimate row-level sampling uncertainty conditional on the fitted model."""
    counts = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    draws = np.random.default_rng(SEED).multinomial(counts.sum(), counts / counts.sum(), 2000)
    tn, fp, fn, tp = draws.T
    neg_denom = 2 * tn + fp + fn
    pos_denom = 2 * tp + fp + fn
    values = tn / np.maximum(neg_denom, 1) + tp / np.maximum(pos_denom, 1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def compare(language):
    path = PROCESSED / f"{language}_clean.csv"
    manifest_path = PROCESSED / f"{language}_split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    frame = pd.read_csv(path, keep_default_na=False).merge(
        manifest, on="text_id", validate="one_to_one", how="outer", indicator=True
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError(f"Split manifest does not match {language} dataset")
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    started = perf_counter()
    vectorizer = TfidfVectorizer(**VECTOR_SETTINGS)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(validation["text"])
    vector_seconds = perf_counter() - started
    truth = validation["label"].to_numpy()
    models = {
        "logistic_regression": lambda c: LogisticRegression(
            C=c, solver="liblinear", random_state=SEED, max_iter=5000),
        "linear_svm": lambda c: LinearSVC(C=c, random_state=SEED, max_iter=5000),
    }
    predictions = {"majority": (None, np.full(len(truth), train["label"].mode().iloc[0]))}
    trials = []
    for name, build in models.items():
        best = -1.0
        for c in C_VALUES:
            started = perf_counter()
            model = build(c).fit(x_train, train["label"])
            pred = model.predict(x_val)
            if np.max(model.n_iter_) >= model.max_iter:
                raise RuntimeError(f"{language} {name} C={c} reached the iteration limit")
            result = scores(truth, pred)
            trials.append(dict(language=language, model=name, c=c, **result,
                               fit_predict_seconds=perf_counter()-started))
            if result["macro_f1"] > best:
                best = result["macro_f1"]
                predictions[name] = (c, pred)
    rows = []
    for name, (c, pred) in predictions.items():
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        lo, hi = bootstrap_macro_f1(truth, pred)
        rows.append(dict(language=language, model=name, c=c, train_n=len(train),
                         validation_n=len(validation), **scores(truth, pred),
                         macro_f1_ci_low=lo, macro_f1_ci_high=hi,
                         tn=tn, fp=fp, fn=fn, tp=tp))
    print(f"{language}: {len(validation)} validation rows", flush=True)
    metadata = {"data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "features": len(vectorizer.vocabulary_), "vectorization_seconds": vector_seconds}
    return rows, trials, metadata


def main():
    destination = REPORTS / "milestone5"
    destination.mkdir(parents=True, exist_ok=True)
    results, trials, inputs = [], [], {}
    for language in LANGUAGES:
        rows, grid, inputs[language] = compare(language)
        results.extend(rows)
        trials.extend(grid)
    pd.DataFrame(results).to_csv(destination / "validation_results.csv", index=False)
    pd.DataFrame(trials).to_csv(destination / "tuning_trials.csv", index=False)
    log = dict(partition="validation", test_evaluated=False, selection_metric="macro_f1",
               c_grid=C_VALUES, tie_break="smallest C", seed=SEED, bootstrap_draws=2000,
               python=platform.python_version(), pandas=pd.__version__,
               numpy=np.__version__, sklearn=sklearn.__version__,
               vectorizer=VECTOR_SETTINGS, inputs=inputs)
    (destination / "experiment_log.json").write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
