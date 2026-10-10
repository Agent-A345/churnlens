"""Train, compare and explain churn models.

Run from the project root:
    python -m src.train

Outputs:
    models/churn_model.joblib   best pipeline (preprocessing + model)
    models/metrics.json         scores for every model
    models/feature_info.json    input ranges and categories for the web app
    reports/figures/*.png       confusion matrix, ROC curves, SHAP plots
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_INFO_PATH,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data_prep import load_clean
from src.explain import clean_feature_names

warnings.filterwarnings("ignore", category=UserWarning)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_models(pos_weight: float) -> dict:
    """Three models from simple to strong. Class imbalance is handled with
    class weights, so no synthetic rows are added to the data."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def tune_xgboost(pipe: Pipeline, X_train, y_train, cv) -> Pipeline:
    """Small randomized search over the most important XGBoost settings."""
    params = {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 8),
        "model__gamma": uniform(0, 2),
    }
    search = RandomizedSearchCV(
        pipe,
        params,
        n_iter=30,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = {k.replace("model__", ""): int(v) if float(v).is_integer() else round(float(v), 4)
            for k, v in search.best_params_.items()}
    print(f"  best CV F1 = {search.best_score_:.4f}")
    print(f"  best params = {best}")
    return search.best_estimator_, float(search.best_score_), best


def save_feature_info(df: pd.DataFrame) -> None:
    """Store input ranges and options so the app builds its form from data."""
    info = {"numeric": {}, "categorical": {}}
    for col in NUMERIC_FEATURES:
        s = df[col].dropna()
        is_int = bool(np.all(np.mod(s, 1) == 0))
        info["numeric"][col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "is_int": is_int,
        }
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts()
        info["categorical"][col] = {
            "options": counts.index.tolist(),
            "mode": counts.index[0],
        }
    FEATURE_INFO_PATH.write_text(json.dumps(info, indent=2))


def plot_confusion(model, X_test, y_test, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Stayed (0)", "Churned (1)"],
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix: {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc(fitted: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_title("ROC Curves on Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict) -> None:
    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ax = df.plot(kind="bar", figsize=(9, 5), rot=0, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()


def explain_with_shap(pipe: Pipeline, X_test: pd.DataFrame, y_test) -> list:
    """Global and single-customer SHAP plots for the final XGBoost model."""
    pre = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    X_t = pd.DataFrame(
        pre.transform(X_test), columns=clean_feature_names(pre.get_feature_names_out())
    )
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_t)

    # Show original (unscaled) values on the plots so they are readable.
    raw_numeric = X_test[NUMERIC_FEATURES].reset_index(drop=True)
    display = X_t.copy()
    for col in NUMERIC_FEATURES:
        display[col] = raw_numeric[col].values
    sv.data = display.values

    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP Summary: What Drives Churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP| Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # One customer the model flags as high risk who actually churned.
    proba = pipe.predict_proba(X_test)[:, 1]
    churners = np.where((np.asarray(y_test) == 1) & (proba >= 0.5))[0]
    idx = int(churners[np.argmax(proba[churners])]) if len(churners) else int(np.argmax(proba))
    shap.plots.waterfall(sv[idx], max_display=12, show=False)
    plt.title(f"Why customer #{idx} is predicted to churn (p = {proba[idx]:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_single_customer.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.Series(np.abs(sv.values).mean(axis=0).astype(float), index=X_t.columns)
    return importance.sort_values(ascending=False).head(10).round(4).to_dict()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    print(f"Data: {len(df):,} rows, churn rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, model in build_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        scores = evaluate(pipe, X_test, y_test)
        scores["cv_f1_mean"] = float(cv_f1.mean())
        scores["cv_f1_std"] = float(cv_f1.std())
        results[name], fitted[name] = scores, pipe
        print(f"{name:22s} CV F1 {cv_f1.mean():.4f} | test F1 {scores['f1']:.4f} "
              f"| ROC-AUC {scores['roc_auc']:.4f}")

    print("Tuning XGBoost...")
    base = Pipeline([("prep", build_preprocessor()),
                     ("model", build_models(pos_weight)["XGBoost"])])
    tuned, tuned_cv_f1, best_params = tune_xgboost(base, X_train, y_train, cv)
    results["XGBoost (Tuned)"] = evaluate(tuned, X_test, y_test)
    results["XGBoost (Tuned)"]["cv_f1_mean"] = tuned_cv_f1
    fitted["XGBoost (Tuned)"] = tuned
    print(f"{'XGBoost (Tuned)':22s} test F1 {results['XGBoost (Tuned)']['f1']:.4f} "
          f"| ROC-AUC {results['XGBoost (Tuned)']['roc_auc']:.4f}")

    # The final model must be XGBoost-based so SHAP TreeExplainer works in the app.
    # It is picked on cross-validation F1, so the test set is never used to choose.
    xgb_names = ["XGBoost", "XGBoost (Tuned)"]
    best_name = max(xgb_names, key=lambda n: results[n]["cv_f1_mean"])
    best = fitted[best_name]
    print(f"\nFinal model: {best_name}")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["Stayed", "Churned"], digits=4))

    plot_confusion(best, X_test, y_test, best_name)
    plot_roc(fitted, X_test, y_test)
    plot_model_comparison(results)
    top_features = explain_with_shap(best, X_test, y_test)

    cm = confusion_matrix(y_test, best.predict(X_test))
    joblib.dump(best, MODEL_PATH)
    save_feature_info(df)
    METRICS_PATH.write_text(json.dumps({
        "final_model": best_name,
        "xgb_tuned_params": best_params,
        "n_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "results": {name: {m: round(float(v), 4) for m, v in r.items()}
                    for name, r in results.items()},
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "top_shap_features": top_features,
    }, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
