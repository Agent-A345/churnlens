"""Load the raw Kaggle file and apply the cleaning steps found during EDA.

Run directly to write the cleaned CSV:
    python -m src.data_prep
"""

from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    FEATURES,
    ID_COL,
    RAW_DATA_PATH,
    TARGET,
)

# The raw data uses different spellings for the same category.
CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw dataset. Accepts the original .xlsx or a .csv export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle "
            "(ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) "
            "and place 'E Commerce Dataset.xlsx' inside the data/ folder."
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    # The Excel file has a data dictionary sheet and the data sheet ("E Comm").
    sheets = pd.read_excel(path, sheet_name=None)
    for df in sheets.values():
        if TARGET in df.columns:
            return df
    raise ValueError(f"No sheet with a '{TARGET}' column found in {path.name}")


def fix_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate category spellings and strip stray whitespace."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    for col, mapping in CATEGORY_FIXES.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning that is safe to do before the train/test split.

    Missing values are NOT filled here. They are imputed inside the model
    pipeline using training-set medians only, so the test set never leaks
    into training.
    """
    df = fix_categories(df)
    df = df.drop(columns=[ID_COL], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df[FEATURES + [TARGET]]


def load_clean(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved {len(data):,} rows to {CLEAN_DATA_PATH}")
