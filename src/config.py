"""Central paths and column definitions used across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "E Commerce Dataset.xlsx"
CLEAN_DATA_PATH = DATA_DIR / "ecommerce_churn_clean.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"

FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "Churn"
ID_COL = "CustomerID"

NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-English labels used in the app and in SHAP explanations.
FEATURE_LABELS = {
    "Tenure": "Months with the company",
    "CityTier": "City tier (1 = metro, 3 = small town)",
    "WarehouseToHome": "Warehouse to home distance (km)",
    "HourSpendOnApp": "Hours spent on app",
    "NumberOfDeviceRegistered": "Devices registered",
    "SatisfactionScore": "Satisfaction score (1 to 5)",
    "NumberOfAddress": "Saved addresses",
    "Complain": "Raised a complaint last month (1 = yes)",
    "OrderAmountHikeFromlastYear": "Order amount increase vs last year (%)",
    "CouponUsed": "Coupons used last month",
    "OrderCount": "Orders last month",
    "DaySinceLastOrder": "Days since last order",
    "CashbackAmount": "Average cashback (Rs)",
    "PreferredLoginDevice": "Preferred login device",
    "PreferredPaymentMode": "Preferred payment mode",
    "Gender": "Gender",
    "PreferedOrderCat": "Preferred order category",
    "MaritalStatus": "Marital status",
}
