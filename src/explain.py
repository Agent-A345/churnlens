"""SHAP helpers shared by training and the web app."""

import numpy as np
import pandas as pd
import shap

from src.config import CATEGORICAL_FEATURES, FEATURE_LABELS, FEATURES


def clean_feature_names(names) -> list:
    """'num__Tenure' -> 'Tenure', 'cat__Gender_Male' -> 'Gender_Male'."""
    return [n.split("__", 1)[-1] for n in names]


def original_feature(encoded_name: str) -> str:
    """Map a one-hot column like 'Gender_Male' back to 'Gender'."""
    for col in CATEGORICAL_FEATURES:
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name


def get_explainer(pipe):
    return shap.TreeExplainer(pipe.named_steps["model"])


def explain_customers(pipe, explainer, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP value per ORIGINAL feature for each row of X.

    One-hot columns of the same categorical feature are summed, so the
    explanation talks about 'Payment mode' instead of 'PaymentMode_UPI'.
    Values are in log-odds: positive pushes toward churn, negative away.
    """
    pre = pipe.named_steps["prep"]
    X_t = pre.transform(X[FEATURES])
    names = clean_feature_names(pre.get_feature_names_out())
    values = explainer.shap_values(X_t)
    df = pd.DataFrame(values, columns=names)
    grouped = df.T.groupby([original_feature(n) for n in names]).sum().T
    return grouped[FEATURES]


def base_value(explainer) -> float:
    ev = explainer.expected_value
    return float(np.ravel(ev)[0])


def describe(feature: str, value) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if pd.isna(value):
        value = "missing"
    if feature == "Complain":
        return "Raised a complaint last month" if value == 1 else "No complaint last month"
    return f"{label}: {value}"


def top_reasons(contrib: pd.Series, row: pd.Series, k: int = 3):
    """Top k features pushing toward churn and top k pushing away."""
    up = contrib[contrib > 0].sort_values(ascending=False).head(k)
    down = contrib[contrib < 0].sort_values().head(k)
    to_churn = [(describe(f, row[f]), float(v)) for f, v in up.items()]
    to_stay = [(describe(f, row[f]), float(v)) for f, v in down.items()]
    return to_churn, to_stay
