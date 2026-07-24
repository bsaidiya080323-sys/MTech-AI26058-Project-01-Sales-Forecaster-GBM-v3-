"""
utils.py
Shared constants, helper functions, and sample-dataset generator
for the Sales Forecaster (GBM) application.
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths / Constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "sales_data.csv")

CATEGORICAL_COLUMNS = ["Category", "Promotion", "Holiday_Indicator"]

FEATURE_COLUMNS = [
    "Store_ID",
    "Category",
    "Selling_Price",
    "Discount",
    "Promotion",
    "Holiday_Indicator",
    "Day_of_Week",
    "Month",
    "Year",
    "Previous_Sales",
]

TARGET_COLUMN = "Sales"

APP_STYLE_DARK = """
QWidget {
    background-color: #1e1e2f;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QMainWindow {
    background-color: #1e1e2f;
}
QTabWidget::pane {
    border: 1px solid #33334d;
    background: #24243a;
}
QTabBar::tab {
    background: #2b2b42;
    color: #cfcfe8;
    padding: 10px 18px;
    margin: 2px;
    border-radius: 6px;
}
QTabBar::tab:selected {
    background: #4c5fd7;
    color: white;
    font-weight: bold;
}
QPushButton {
    background-color: #4c5fd7;
    color: white;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #5f70e0;
}
QPushButton:disabled {
    background-color: #3a3a52;
    color: #888;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #2b2b42;
    border: 1px solid #44446a;
    border-radius: 4px;
    padding: 5px;
    color: #f0f0f0;
}
QTableWidget {
    background-color: #24243a;
    gridline-color: #3a3a52;
    color: #e0e0e0;
}
QHeaderView::section {
    background-color: #33334d;
    color: #e0e0e0;
    padding: 4px;
    border: none;
}
QProgressBar {
    border: 1px solid #44446a;
    border-radius: 5px;
    text-align: center;
    color: white;
    background-color: #2b2b42;
}
QProgressBar::chunk {
    background-color: #4c5fd7;
    border-radius: 5px;
}
QGroupBox {
    border: 1px solid #3a3a52;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLabel#TitleLabel {
    font-size: 22px;
    font-weight: bold;
    color: #8ea2ff;
}
QLabel#SubtitleLabel {
    font-size: 13px;
    color: #a0a0c0;
}
QStatusBar {
    background-color: #16162a;
    color: #a0a0c0;
}
"""


def ensure_dirs():
    """Make sure asset/report directories exist."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ASSETS_DIR, "icons"), exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_sample_dataset(path=DEFAULT_DATASET_PATH, n_rows=1500, seed=42):
    """
    Generates a synthetic retail sales dataset matching the schema
    described in the project prompt, and writes it to `path`.
    Useful for demo / first-run purposes.
    """
    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Grocery", "Clothing", "Home", "Toys"]
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="D")

    df = pd.DataFrame()
    df["Date"] = rng.choice(dates, size=n_rows)
    df["Product_ID"] = rng.integers(1000, 1100, size=n_rows)
    df["Product_Name"] = [f"Product_{i}" for i in rng.integers(1, 60, size=n_rows)]
    df["Store_ID"] = rng.integers(1, 15, size=n_rows)
    df["Category"] = rng.choice(categories, size=n_rows)
    df["Selling_Price"] = np.round(rng.uniform(5, 500, size=n_rows), 2)
    df["Discount"] = np.round(rng.uniform(0, 40, size=n_rows), 1)
    df["Promotion"] = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])
    df["Holiday_Indicator"] = rng.choice(["Yes", "No"], size=n_rows, p=[0.15, 0.85])
    df["Temperature"] = np.round(rng.uniform(-5, 40, size=n_rows), 1)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df["Previous_Sales"] = rng.integers(10, 500, size=n_rows)

    promo_boost = np.where(df["Promotion"] == "Yes", 1.25, 1.0)
    holiday_boost = np.where(df["Holiday_Indicator"] == "Yes", 1.4, 1.0)
    discount_effect = 1 + (df["Discount"] / 100.0)
    category_multiplier = df["Category"].map(
        {"Electronics": 1.3, "Grocery": 1.6, "Clothing": 1.1, "Home": 1.2, "Toys": 0.9}
    )

    base = (
        0.6 * df["Previous_Sales"]
        + 0.05 * (500 - df["Selling_Price"])
        + rng.normal(0, 15, size=n_rows)
    )
    df["Units_Sold"] = np.clip(
        (base * promo_boost * holiday_boost * discount_effect * category_multiplier).astype(int),
        1,
        None,
    )
    df["Sales"] = np.round(df["Units_Sold"] * df["Selling_Price"] * (1 - df["Discount"] / 100), 2)

    df = df[
        [
            "Date",
            "Product_ID",
            "Product_Name",
            "Store_ID",
            "Category",
            "Units_Sold",
            "Selling_Price",
            "Discount",
            "Promotion",
            "Holiday_Indicator",
            "Temperature",
            "Day_of_Week",
            "Month",
            "Year",
            "Previous_Sales",
            "Sales",
        ]
    ]

    df.to_csv(path, index=False)
    return path


def format_currency(value):
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return str(value)
