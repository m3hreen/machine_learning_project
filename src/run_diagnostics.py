"""Validation-only diagnostics. Run from the project's src directory or root."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

from run_baselines import ROOT, PROCESSED, REPORTS, SEED, scores, split_data

LANGUAGES = ["portuguese", "turkish", "chinese", "persian", "tamil", "japanese"]
OUT = REPORTS / "diagnostics"
LOCAL = PROCESSED / "diagnostics"
MODEL = "char_tfidf_logistic_regression"


def fit_model(train, validation):
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2,
        max_features=50000, lowercase=False, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_valid = vectorizer.transform(validation["text"])
    classifier = LogisticRegression(
        C=1.0, solver="liblinear", max_iter=1000, random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(x_train, train["label"])
    return vectorizer, classifier, x_train, x_valid


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def check_scores(language, name, observed, reference):
    rows = reference.loc[
        reference["language"].eq(language) & reference["model"].eq(name)
    ]
    if len(rows) != 1 or rows.iloc[0]["partition"] != "validation":
        raise ValueError(f"Missing or ambiguous validation reference: {language}/{name}")
    for metric, value in observed.items():
        expected = float(rows.iloc[0][metric])
        if not np.isfinite(expected) or abs(value - expected) > 0.00005:
            raise ValueError(
                f"Baseline mismatch for {language}/{name}/{metric}: "
                f"expected {expected:.6f}, observed {value:.6f}. "
                "Check code, environment and input files before continuing."
            )


def run_language(language, reference, recorded):
    path = PROCESSED / f"{language}_clean.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = recorded["languages"][language]["processed_sha256"]
    if digest != expected_hash:
        raise ValueError(f"Processed file hash differs from baseline: {language}")
    frame = pd.read_csv(path, keep_default_na=False)
    if (not frame["label"].isin([0, 1]).all()
            or not frame["text_id"].is_unique
            or not frame["text"].str.strip().ne("").all()):
        raise ValueError(f"Invalid processed data: {language}")
    parts = split_data(frame, language)
    train, valid = parts["train"], parts["validation"]
    # No test features, predictions, or metrics are computed.
    vectorizer, classifier, x_train, x_valid = fit_model(train, valid)
    predicted = classifier.predict(x_valid)
    probability = classifier.predict_proba(x_valid)[:, 1]
    observed = scores(valid["label"], predicted)
    check_scores(language, MODEL, observed, reference)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train), 1)), train["label"])
    dummy_pred = dummy.predict(np.zeros((len(valid), 1)))
    check_scores(language, "majority_class", scores(valid["label"], dummy_pred), reference)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, label, pred in zip(axes, ["Majority class", "TF-IDF + Logistic Regression"],
                               [dummy_pred, predicted]):
        ConfusionMatrixDisplay.from_predictions(
            valid["label"], pred, labels=[0, 1],
            display_labels=["Comparison (0)", "Positive (1)"],
            ax=ax, colorbar=False, cmap="Blues",
        )
        ax.set_title(label)
    fig.suptitle(f"{language.title()}: validation confusion matrices")
    save_figure(fig, f"{language}_confusion")

    fpr, tpr, _ = roc_curve(valid["label"], probability)
    precision, recall, _ = precision_recall_curve(valid["label"], probability)
    auc = roc_auc_score(valid["label"], probability)
    ap = average_precision_score(valid["label"], probability)
    prevalence = float(valid["label"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance reference")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].step(recall, precision, where="post", label=f"Average precision = {ap:.3f}")
    axes[1].axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence = {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=8)
    fig.suptitle(f"{language.title()}: validation probability rankings")
    save_figure(fig, f"{language}_roc_pr")

    curve_rows = []
    for fraction in [0.2, 0.5, 1.0]:
        if fraction == 1:
            subset, current, xt, xv = train, classifier, x_train, x_valid
        else:
            subset, _ = train_test_split(
                train, train_size=fraction, random_state=SEED, stratify=train["label"],
            )
            _, current, xt, xv = fit_model(subset, valid)
        curve_rows.append({
            "language": language, "requested_fraction": fraction, "train_n": len(subset),
            "training_log_loss": log_loss(subset["label"], current.predict_proba(xt), labels=[0, 1]),
            "validation_log_loss": log_loss(valid["label"], current.predict_proba(xv), labels=[0, 1]),
        })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / f"{language}_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curves["train_n"], curves["training_log_loss"], "o-", label="Training")
    ax.plot(curves["train_n"], curves["validation_log_loss"], "o-", label="Validation")
    ax.set(xlabel="Number of training examples", ylabel="Log loss",
           title=f"{language.title()}: learning curve (one seed)")
    ax.legend()
    save_figure(fig, f"{language}_learning_curve")

    # IDs keep multilingual figures portable when system fonts lack a script.
    coefficients = pd.DataFrame({
        "feature": vectorizer.get_feature_names_out(),
        "coefficient": classifier.coef_[0],
    })
    coefficients["feature_id"] = [f"F{i:05d}" for i in range(len(coefficients))]
    selected = pd.concat([
        coefficients.nsmallest(10, "coefficient"),
        coefficients.nlargest(10, "coefficient"),
    ]).drop_duplicates("feature_id").sort_values("coefficient")
    selected.to_csv(OUT / f"{language}_top_features.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(selected["feature_id"], selected["coefficient"],
            color=["#b24a3a" if c < 0 else "#216c9b" for c in selected["coefficient"]])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Logistic Regression coefficient (positive favors label 1)",
           ylabel="Feature ID (character n-grams in accompanying CSV)",
           title=f"{language.title()}: strongest character associations")
    save_figure(fig, f"{language}_features")

    examples = valid[["text_id", "text", "label"]].copy()
    examples["prediction"] = predicted
    examples["positive_probability"] = probability
    examples["outcome"] = [
        ("TP" if p else "FN") if y else ("FP" if p else "TN")
        for y, p in zip(valid["label"], predicted)
    ]
    examples["prediction_confidence"] = np.where(predicted == 1, probability, 1 - probability)
    picked = examples.sort_values("prediction_confidence", ascending=False).groupby(
        "outcome", sort=True
    ).head(2)
    picked.to_csv(LOCAL / f"{language}_examples.csv", index=False)
    outcome_counts = examples["outcome"].value_counts().reindex(["TN", "FP", "FN", "TP"], fill_value=0)
    print(f"{language}: baseline matched; figures saved; example texts saved locally.", flush=True)
    return {
        "language": language, **observed, "roc_auc": auc, "average_precision": ap,
        "validation_prevalence": prevalence,
        **{key: int(value) for key, value in outcome_counts.items()},
        "processed_sha256": digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    args = parser.parse_args()
    for language in args.languages:
        path = PROCESSED / f"{language}_clean.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run the preparation scripts first: missing {path}")
    reference = pd.read_csv(REPORTS / "baseline_validation.csv")
    recorded = json.loads((REPORTS / "baseline_run.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    # This protection applies even if the repository's existing ignore rules change.
    (LOCAL / ".gitignore").write_text("*\n", encoding="utf-8")
    results = [run_language(language, reference, recorded) for language in args.languages]
    pd.DataFrame(results).to_csv(OUT / "validation_diagnostics.csv", index=False)
    (OUT / "diagnostics_run.json").write_text(json.dumps({
        "seed": SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "languages": args.languages, "test_evaluated": False,
        "learning_curve_fractions": [0.2, 0.5, 1.0],
        "baseline_metrics_checked": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text("""# Validation diagnostics

These diagnostics refit the existing character TF-IDF + Logistic Regression baseline.
Full-training validation metrics must match the saved baseline within 0.00005.
No reserved test predictions or test metrics are computed. Baseline reports are not overwritten.

- Learning curves show training/validation log loss versus training sample count,
  not epochs or solver iterations. Each point refits TF-IDF on its training subset.
  One seed and a fixed validation set are used; no uncertainty intervals are claimed.
- Confusion matrices use released/provisional labels, not confirmed clinical diagnoses.
- ROC AUC and average precision measure rankings; the baseline's default decision
  threshold is unchanged. Average precision is not trapezoidal PR area.
- Feature coefficients show associations, not causal importance. Feature IDs map
  to native character n-grams in each top_features.csv file. Weights should not
  be compared directly across languages.
- Up to two high-confidence examples per available TN/FP/FN/TP group are saved
  only in ignored data/processed/diagnostics/. These are illustrative, selected
  examples, not a representative error sample. Empty groups are not invented.
  Do not publish identifiable or sensitive text without appropriate review.

These are classification diagnostics, not evidence of age bias. Existing label,
speaker/author overlap, near-duplicate and provenance limitations still apply.
""", encoding="utf-8")
    print(f"\nFinished. Figures and aggregate tables: {OUT}")
    print("Reserved test sets remain unevaluated.")


if __name__ == "__main__":
    main()
