"""One-time held-out test evaluation. Uses C values selected on validation; no retuning.

Run from the project root:  python src/run_milestone5_test.py
Refuses to overwrite an existing test result so the test set is not reused by accident.
"""
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from run_baselines import PROCESSED, REPORTS, SEED, scores
from run_milestone5_validation import LANGUAGES, VECTOR_SETTINGS, bootstrap_macro_f1

BUILDERS = {
    "logistic_regression": lambda c: LogisticRegression(
        C=c, solver="liblinear", random_state=SEED, max_iter=5000),
    "linear_svm": lambda c: LinearSVC(C=c, dual="auto", random_state=SEED, max_iter=5000),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(language, selected, recorded):
    path = PROCESSED / f"{language}_clean.csv"
    manifest_path = PROCESSED / f"{language}_split_manifest.csv"
    # Inputs must be identical to those used in validation.
    if sha(path) != recorded["inputs"][language]["data_sha256"]:
        raise ValueError(f"{language}: processed data differs from validation run")
    if sha(manifest_path) != recorded["inputs"][language]["manifest_sha256"]:
        raise ValueError(f"{language}: split manifest differs from validation run")

    frame = pd.read_csv(path, keep_default_na=False).merge(
        pd.read_csv(manifest_path), on="text_id", validate="one_to_one",
        how="outer", indicator=True)
    if not frame["_merge"].eq("both").all():
        raise ValueError(f"Split manifest does not match {language} dataset")
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]

    vectorizer = TfidfVectorizer(**VECTOR_SETTINGS)
    x_train = vectorizer.fit_transform(train["text"])  # training text only
    x_val = vectorizer.transform(validation["text"])
    x_test = vectorizer.transform(test["text"])
    truth = test["label"].to_numpy()

    models = {"majority": None, **{n: None for n in BUILDERS}}
    preds = {"majority": np.full(len(truth), train["label"].mode().iloc[0])}
    chosen = {"majority": None}
    for name, build in BUILDERS.items():
        c = float(selected.loc[(language, name), "c"])
        model = build(c).fit(x_train, train["label"])
        if np.max(model.n_iter_) >= model.max_iter:
            raise RuntimeError(f"{language} {name} reached the iteration limit")
        # Sanity check: refit must reproduce the saved validation score.
        got = scores(validation["label"].to_numpy(), model.predict(x_val))["macro_f1"]
        want = float(selected.loc[(language, name), "macro_f1"])
        if abs(got - want) > 5e-5:
            raise ValueError(f"{language} {name}: validation macro-F1 {got:.6f} "
                             f"does not match saved {want:.6f}; check environment")
        preds[name] = model.predict(x_test)
        chosen[name] = c

    rows = []
    for name, pred in preds.items():
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        lo, hi = bootstrap_macro_f1(truth, pred)
        rows.append(dict(language=language, model=name, c=chosen[name],
                         train_n=len(train), test_n=len(test), **scores(truth, pred),
                         macro_f1_ci_low=lo, macro_f1_ci_high=hi,
                         tn=tn, fp=fp, fn=fn, tp=tp))
    print(f"{language}: {len(test)} test rows", flush=True)
    return rows, {"data_sha256": sha(path), "manifest_sha256": sha(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    args = parser.parse_args()
    dest = REPORTS / "milestone5"
    # One file per language so separate runs never collide and no test set is scored twice.
    for language in args.languages:
        done = dest / f"test_results_{language}.csv"
        if done.exists():
            raise FileExistsError(f"{done} exists. {language} test set already evaluated.")

    selection_path = dest / "validation_results.csv"
    selected = pd.read_csv(selection_path).set_index(["language", "model"], verify_integrity=True)
    recorded = json.loads((dest / "experiment_log.json").read_text(encoding="utf-8"))

    results, inputs = [], {}
    for language in args.languages:
        rows, inputs[language] = evaluate(language, selected, recorded)
        results.append((language, rows))

    for language, rows in results:
        pd.DataFrame(rows).to_csv(dest / f"test_results_{language}.csv", index=False)
    log = dict(partition="test", test_evaluated=True, retuned=False, languages=args.languages,
               selection_source="validation_results.csv",
               selection_sha256=sha(selection_path),
               selected_c={lang: {n: float(selected.loc[(lang, n), "c"]) for n in BUILDERS}
                           for lang in args.languages},
               seed=SEED, bootstrap_draws=2000, svm_dual="auto",
               interval="Row-bootstrap 95% interval conditional on fitted model; "
                        "ignores author/speaker/near-duplicate dependence",
               python=platform.python_version(), pandas=pd.__version__,
               numpy=np.__version__, sklearn=sklearn.__version__,
               vectorizer=VECTOR_SETTINGS, inputs=inputs)
    (dest / f"test_experiment_log_{'_'.join(args.languages)}.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
