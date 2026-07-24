"""
sales_forecaster_gbm.py
========================================================================
Sales Forecaster (GBM) v3 — Single-File Edition
------------------------------------------------------------------------
A desktop application (PyQt5 + Scikit-learn + Pandas + NumPy + Matplotlib
+ Joblib) that forecasts retail sales using Gradient Boosting Regression.

Run with:
    pip install -r requirements.txt   (PyQt5, scikit-learn, pandas, numpy,
                                        matplotlib, joblib)
    python sales_forecaster_gbm.py

Tabs:
    1. Dashboard          4. Model Training       7. Reports
    2. Dataset Upload     5. Sales Prediction
    3. Preprocessing      6. Visualization
========================================================================
"""

import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("QT5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QMessageBox, QStatusBar,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont


# ==========================================================================
# SECTION 1: Constants / Utils
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "sales_data.csv")

CATEGORICAL_COLUMNS = ["Category", "Promotion", "Holiday_Indicator"]
FEATURE_COLUMNS = [
    "Store_ID", "Category", "Selling_Price", "Discount", "Promotion",
    "Holiday_Indicator", "Day_of_Week", "Month", "Year", "Previous_Sales",
]
TARGET_COLUMN = "Sales"

APP_STYLE_DARK = """
QWidget { background-color: #1e1e2f; color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
QMainWindow { background-color: #1e1e2f; }
QTabWidget::pane { border: 1px solid #33334d; background: #24243a; }
QTabBar::tab { background: #2b2b42; color: #cfcfe8; padding: 10px 18px;
    margin: 2px; border-radius: 6px; }
QTabBar::tab:selected { background: #4c5fd7; color: white; font-weight: bold; }
QPushButton { background-color: #4c5fd7; color: white; border-radius: 6px;
    padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background-color: #5f70e0; }
QPushButton:disabled { background-color: #3a3a52; color: #888; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2b2b42;
    border: 1px solid #44446a; border-radius: 4px; padding: 5px; color: #f0f0f0; }
QTableWidget { background-color: #24243a; gridline-color: #3a3a52; color: #e0e0e0; }
QHeaderView::section { background-color: #33334d; color: #e0e0e0; padding: 4px; border: none; }
QProgressBar { border: 1px solid #44446a; border-radius: 5px; text-align: center;
    color: white; background-color: #2b2b42; }
QProgressBar::chunk { background-color: #4c5fd7; border-radius: 5px; }
QGroupBox { border: 1px solid #3a3a52; border-radius: 6px; margin-top: 10px;
    padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLabel#TitleLabel { font-size: 22px; font-weight: bold; color: #8ea2ff; }
QLabel#SubtitleLabel { font-size: 13px; color: #a0a0c0; }
QStatusBar { background-color: #16162a; color: #a0a0c0; }
"""

DARK_BG, PANEL_BG, ACCENT, ACCENT2, TEXT_COLOR, GRID_COLOR = (
    "#1e1e2f", "#24243a", "#4c5fd7", "#8ea2ff", "#e0e0e0", "#3a3a52",
)


def ensure_dirs():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ASSETS_DIR, "icons"), exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def format_currency(value):
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def generate_sample_dataset(path=DEFAULT_DATASET_PATH, n_rows=1500, seed=42):
    """Generates a synthetic retail sales dataset for demo purposes."""
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
    base = (0.6 * df["Previous_Sales"] + 0.05 * (500 - df["Selling_Price"])
            + rng.normal(0, 15, size=n_rows))
    df["Units_Sold"] = np.clip(
        (base * promo_boost * holiday_boost * discount_effect * category_multiplier).astype(int),
        1, None,
    )
    df["Sales"] = np.round(df["Units_Sold"] * df["Selling_Price"] * (1 - df["Discount"] / 100), 2)

    df = df[["Date", "Product_ID", "Product_Name", "Store_ID", "Category", "Units_Sold",
             "Selling_Price", "Discount", "Promotion", "Holiday_Indicator", "Temperature",
             "Day_of_Week", "Month", "Year", "Previous_Sales", "Sales"]]
    df.to_csv(path, index=False)
    return path


# ==========================================================================
# SECTION 2: Data Preprocessing
# ==========================================================================
class DataPreprocessor:
    """load -> clean -> engineer -> encode -> split, keeping encoders for reuse."""

    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.label_encoders = {}
        self.summary = {}

    def load_csv(self, path):
        self.raw_df = pd.read_csv(path)
        return self.raw_df

    def clean(self, df=None):
        df = (self.raw_df if df is None else df).copy()
        n_before = len(df)
        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates()
        n_missing_before = int(df.isnull().sum().sum())

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

        self.summary = {
            "rows_before": n_before,
            "rows_after_dedup": len(df),
            "duplicates_removed": n_duplicates,
            "missing_values_filled": n_missing_before,
        }
        return df

    def engineer_features(self, df):
        df = df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if "Day_of_Week" not in df.columns:
                df["Day_of_Week"] = df["Date"].dt.dayofweek
            if "Month" not in df.columns:
                df["Month"] = df["Date"].dt.month
            if "Year" not in df.columns:
                df["Year"] = df["Date"].dt.year

        if TARGET_COLUMN not in df.columns and {"Units_Sold", "Selling_Price"}.issubset(df.columns):
            discount = df["Discount"] if "Discount" in df.columns else 0
            df[TARGET_COLUMN] = df["Units_Sold"] * df["Selling_Price"] * (1 - discount / 100)

        defaults = {
            "Store_ID": 1, "Category": "General",
            "Selling_Price": df["Selling_Price"].mean() if "Selling_Price" in df.columns else 0,
            "Discount": 0, "Promotion": "No", "Holiday_Indicator": "No",
            "Day_of_Week": 0, "Month": 1, "Year": 2024, "Previous_Sales": 0,
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        return df

    def encode(self, df):
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def encode_single_value(self, column, value):
        le = self.label_encoders.get(column)
        if le is None:
            return 0
        value = str(value)
        return int(le.transform([value])[0]) if value in le.classes_ else 0

    def run_pipeline(self, path):
        self.load_csv(path)
        df = self.clean(self.raw_df)
        df = self.engineer_features(df)
        df = self.encode(df)

        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required columns: {missing_features}")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Dataset must contain a '{TARGET_COLUMN}' column, or 'Units_Sold' "
                f"and 'Selling_Price' columns so it can be derived."
            )
        self.processed_df = df
        return df

    def split(self, df=None, test_size=0.2, random_state=42):
        df = self.processed_df if df is None else df
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ==========================================================================
# SECTION 3: Model Training
# ==========================================================================
class ModelTrainer:
    """Wraps GradientBoostingRegressor: training, evaluation, persistence."""

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
        )
        self.metrics = {}
        self.is_trained = False

    def train(self, X_train, y_train, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        self.model.fit(X_train, y_train)
        if progress_callback:
            progress_callback(90)
        self.is_trained = True
        if progress_callback:
            progress_callback(100)
        return self.model

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        self.metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        return self.metrics, y_pred

    def feature_importances(self):
        if not self.is_trained:
            return {}
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def save(self, path=DEFAULT_MODEL_PATH, extra=None):
        payload = {"model": self.model, "is_trained": self.is_trained}
        if extra:
            payload.update(extra)
        joblib.dump(payload, path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        payload = joblib.load(path)
        self.model = payload["model"]
        self.is_trained = payload.get("is_trained", True)
        return payload


# ==========================================================================
# SECTION 4: Predictor (single-row prediction)
# ==========================================================================
class Predictor:
    """Bridges GUI inputs -> encoded feature row -> model prediction + confidence."""

    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def build_feature_row(self, inputs: dict) -> pd.DataFrame:
        row = {}
        for col in FEATURE_COLUMNS:
            value = inputs.get(col)
            if col in self.preprocessor.label_encoders:
                row[col] = self.preprocessor.encode_single_value(col, value)
            else:
                row[col] = value
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, inputs: dict):
        X_row = self.build_feature_row(inputs)
        prediction = float(self.trainer.predict(X_row)[0])
        confidence = self._estimate_confidence(X_row)
        return prediction, confidence

    def _estimate_confidence(self, X_row):
        """Naive confidence proxy from the spread of late boosting stages."""
        try:
            staged = list(self.trainer.model.staged_predict(X_row))
            tail = staged[-10:] if len(staged) >= 10 else staged
            values = [float(v[0]) for v in tail]
            spread = max(values) - min(values) if values else 0.0
            mean_val = sum(values) / len(values) if values else 1.0
            rel_spread = spread / (abs(mean_val) + 1e-6)
            return round(max(0.0, min(100.0, 100 * (1 - rel_spread))), 1)
        except Exception:
            return None


# ==========================================================================
# SECTION 5: Visualization (Matplotlib chart builders)
# ==========================================================================
def _new_figure(figsize=(6, 4)):
    fig = Figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor(DARK_BG)
    return fig


def _style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def chart_historical_sales_trend(df, date_col="Date", sales_col="Sales"):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    if date_col in df.columns:
        temp = df[[date_col, sales_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col])
        temp = temp.sort_values(date_col)
        ax.plot(temp[date_col], temp[sales_col], color=ACCENT2, linewidth=1.2)
        ax.xaxis_date()
        fig.autofmt_xdate()
    else:
        ax.plot(df[sales_col].values, color=ACCENT2, linewidth=1.2)
    ax.set_title("Historical Sales Trend")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    return fig


def chart_actual_vs_predicted(y_test, y_pred):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    y_test_arr, y_pred_arr = np.asarray(y_test), np.asarray(y_pred)
    ax.scatter(y_test_arr, y_pred_arr, s=18, color=ACCENT2, alpha=0.7, edgecolors="none")
    min_val, max_val = min(y_test_arr.min(), y_pred_arr.min()), max(y_test_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#ff6b6b", linewidth=1.5, linestyle="--")
    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    fig.tight_layout()
    return fig


def chart_monthly_sales_forecast(df, month_col="Month", sales_col="Sales"):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    if month_col in df.columns:
        grouped = df.groupby(month_col)[sales_col].mean().sort_index()
        ax.bar(grouped.index.astype(str), grouped.values, color=ACCENT)
    ax.set_title("Monthly Sales Forecast (Avg)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Sales")
    fig.tight_layout()
    return fig


def chart_feature_importance(importances: dict):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    items = sorted(importances.items(), key=lambda x: x[1])
    ax.barh([i[0] for i in items], [i[1] for i in items], color=ACCENT2)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def chart_error_distribution(y_test, y_pred):
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    ax.hist(residuals, bins=30, color=ACCENT, edgecolor=DARK_BG)
    ax.axvline(0, color="#ff6b6b", linewidth=1.5, linestyle="--")
    ax.set_title("Residual Error Distribution")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


# ==========================================================================
# SECTION 6: Background training thread (keeps GUI responsive)
# ==========================================================================
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, trainer, X_train, X_test, y_train, y_test):
        super().__init__()
        self.trainer = trainer
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def run(self):
        try:
            self.trainer.train(self.X_train, self.y_train, progress_callback=self.progress.emit)
            metrics, y_pred = self.trainer.evaluate(self.X_test, self.y_test)
            self.finished_ok.emit({"metrics": metrics, "y_pred": y_pred})
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ==========================================================================
# SECTION 7: Main Window (PyQt5 multi-tab GUI)
# ==========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Sales Forecaster (GBM) v3 — Retail Analytics Suite")
        self.resize(1280, 820)
        self.setStyleSheet(APP_STYLE_DARK)

        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.predictor = None
        self.dataset_path = None
        self.processed_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None
        self.training_worker = None

        self._build_ui()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_upload_tab(), "Dataset Upload")
        self.tabs.addTab(self._build_preprocessing_tab(), "Preprocessing")
        self.tabs.addTab(self._build_training_tab(), "Model Training")
        self.tabs.addTab(self._build_prediction_tab(), "Sales Prediction")
        self.tabs.addTab(self._build_visualization_tab(), "Visualization")
        self.tabs.addTab(self._build_reports_tab(), "Reports")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a dataset to begin.")

    # ---------------------------------------------------- Tab 1: Dashboard
    def _build_dashboard_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Sales Forecaster (GBM) — Retail Analytics & Inventory Planning")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Predict future retail sales using Gradient Boosting Regression.\n"
            "Upload historical data, train a model, and forecast demand to optimize inventory."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(15)

        grid = QGridLayout()
        self.dash_dataset_label = QLabel("Dataset: Not loaded")
        self.dash_rows_label = QLabel("Rows: —")
        self.dash_model_label = QLabel("Model status: Not trained")
        self.dash_r2_label = QLabel("Model R² Score: —")
        for i, lbl in enumerate([self.dash_dataset_label, self.dash_rows_label,
                                  self.dash_model_label, self.dash_r2_label]):
            box = QGroupBox()
            box_layout = QVBoxLayout(box)
            lbl.setFont(QFont("Segoe UI", 11))
            box_layout.addWidget(lbl)
            grid.addWidget(box, i // 2, i % 2)
        layout.addLayout(grid)

        info_box = QGroupBox("System Overview")
        info_layout = QVBoxLayout(info_box)
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(
            "Workflow:\n"
            "  1. Upload historical sales dataset (Dataset Upload tab)\n"
            "  2. Clean & preprocess the data (Preprocessing tab)\n"
            "  3. Train the Gradient Boosting model (Model Training tab)\n"
            "  4. Enter future sales parameters and predict (Sales Prediction tab)\n"
            "  5. Review interactive charts (Visualization tab)\n"
            "  6. Export forecast reports (Reports tab)\n\n"
            "Algorithm: GradientBoostingRegressor (Scikit-learn)\n"
            "Evaluation Metrics: MAE, MSE, RMSE, R²"
        )
        info_layout.addWidget(info_text)
        layout.addWidget(info_box)
        return widget

    def _refresh_dashboard(self):
        if self.dataset_path:
            self.dash_dataset_label.setText(f"Dataset: {os.path.basename(self.dataset_path)}")
        if self.processed_df is not None:
            self.dash_rows_label.setText(f"Rows: {len(self.processed_df)}")
        if self.trainer.is_trained:
            self.dash_model_label.setText("Model status: Trained ✔")
        if self.trainer.metrics:
            self.dash_r2_label.setText(f"Model R² Score: {self.trainer.metrics['R2']:.4f}")

    # ----------------------------------------------- Tab 2: Dataset Upload
    def _build_upload_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse CSV…")
        browse_btn.clicked.connect(self._on_browse_csv)
        sample_btn = QPushButton("Generate Sample Dataset")
        sample_btn.clicked.connect(self._on_generate_sample)
        btn_row.addWidget(browse_btn)
        btn_row.addWidget(sample_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.upload_path_label = QLabel("No dataset loaded.")
        layout.addWidget(self.upload_path_label)
        self.upload_stats_label = QLabel("")
        self.upload_stats_label.setWordWrap(True)
        layout.addWidget(self.upload_stats_label)

        self.upload_table = QTableWidget()
        layout.addWidget(self.upload_table)
        return widget

    def _on_browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Sales CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_dataset(path)

    def _on_generate_sample(self):
        self._load_dataset(generate_sample_dataset())

    def _load_dataset(self, path):
        try:
            df = pd.read_csv(path)
            self.dataset_path = path
            self.upload_path_label.setText(f"Loaded: {path}")
            self.upload_stats_label.setText(
                f"Rows: {len(df)}  |  Columns: {len(df.columns)}  |  "
                f"Missing values: {int(df.isnull().sum().sum())}  |  "
                f"Duplicate rows: {int(df.duplicated().sum())}"
            )
            self._populate_table(self.upload_table, df.head(100))
            self.status_bar.showMessage(f"Dataset loaded: {os.path.basename(path)}")
            self._refresh_dashboard()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error Loading Dataset", str(exc))

    @staticmethod
    def _populate_table(table: QTableWidget, df: pd.DataFrame):
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        table.resizeColumnsToContents()

    # ------------------------------------------------ Tab 3: Preprocessing
    def _build_preprocessing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        run_btn = QPushButton("Run Preprocessing Pipeline")
        run_btn.clicked.connect(self._on_run_preprocessing)
        layout.addWidget(run_btn)

        self.preprocess_summary = QTextEdit()
        self.preprocess_summary.setReadOnly(True)
        layout.addWidget(self.preprocess_summary)

        self.preprocess_table = QTableWidget()
        layout.addWidget(self.preprocess_table)
        return widget

    def _on_run_preprocessing(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first (Dataset Upload tab).")
            return
        try:
            df = self.preprocessor.run_pipeline(self.dataset_path)
            self.processed_df = df
            summary = self.preprocessor.summary
            encoded_cols = list(self.preprocessor.label_encoders.keys())
            text = (
                f"Rows before cleaning: {summary.get('rows_before')}\n"
                f"Duplicates removed: {summary.get('duplicates_removed')}\n"
                f"Rows after dedup: {summary.get('rows_after_dedup')}\n"
                f"Missing values filled: {summary.get('missing_values_filled')}\n"
                f"Encoded categorical columns: {encoded_cols}\n"
                f"Feature columns used: {FEATURE_COLUMNS}\n"
                f"Target column: Sales\n"
                f"Final processed shape: {df.shape}"
            )
            self.preprocess_summary.setPlainText(text)
            self._populate_table(self.preprocess_table, df.head(100))
            self.status_bar.showMessage("Preprocessing complete.")
            self._refresh_dashboard()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preprocessing Error", str(exc))
            traceback.print_exc()

    # -------------------------------------------------- Tab 4: Training
    def _build_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        params_box = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_box)

        params_layout.addWidget(QLabel("Number of Estimators:"), 0, 0)
        self.spin_n_estimators = QSpinBox()
        self.spin_n_estimators.setRange(10, 2000)
        self.spin_n_estimators.setValue(300)
        params_layout.addWidget(self.spin_n_estimators, 0, 1)

        params_layout.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setRange(0.001, 1.0)
        self.spin_learning_rate.setSingleStep(0.01)
        self.spin_learning_rate.setValue(0.05)
        params_layout.addWidget(self.spin_learning_rate, 0, 3)

        params_layout.addWidget(QLabel("Max Depth:"), 1, 0)
        self.spin_max_depth = QSpinBox()
        self.spin_max_depth.setRange(1, 20)
        self.spin_max_depth.setValue(4)
        params_layout.addWidget(self.spin_max_depth, 1, 1)

        params_layout.addWidget(QLabel("Test Split %:"), 1, 2)
        self.spin_test_split = QSpinBox()
        self.spin_test_split.setRange(5, 50)
        self.spin_test_split.setValue(20)
        params_layout.addWidget(self.spin_test_split, 1, 3)

        layout.addWidget(params_box)

        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Train Gradient Boosting Model")
        self.train_btn.clicked.connect(self._on_train_model)
        save_model_btn = QPushButton("Save Model (.pkl)")
        save_model_btn.clicked.connect(self._on_save_model)
        load_model_btn = QPushButton("Load Existing Model")
        load_model_btn.clicked.connect(self._on_load_model)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(save_model_btn)
        btn_row.addWidget(load_model_btn)
        layout.addLayout(btn_row)

        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)

        self.metrics_label = QTextEdit()
        self.metrics_label.setReadOnly(True)
        layout.addWidget(self.metrics_label)
        return widget

    def _on_train_model(self):
        if self.processed_df is None:
            QMessageBox.warning(self, "No Processed Data", "Please run preprocessing first.")
            return

        self.trainer = ModelTrainer(
            n_estimators=self.spin_n_estimators.value(),
            learning_rate=self.spin_learning_rate.value(),
            max_depth=self.spin_max_depth.value(),
        )
        test_size = self.spin_test_split.value() / 100.0
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split(
            self.processed_df, test_size=test_size
        )

        self.train_btn.setEnabled(False)
        self.train_progress.setValue(0)
        self.status_bar.showMessage("Training model…")

        self.training_worker = TrainingWorker(
            self.trainer, self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.training_worker.progress.connect(self.train_progress.setValue)
        self.training_worker.finished_ok.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.start()

    def _on_training_finished(self, result):
        self.train_btn.setEnabled(True)
        metrics = result["metrics"]
        self.y_pred = result["y_pred"]
        self.predictor = Predictor(self.trainer, self.preprocessor)

        text = (
            f"Training complete.\n\n"
            f"MAE:  {metrics['MAE']:.4f}\n"
            f"MSE:  {metrics['MSE']:.4f}\n"
            f"RMSE: {metrics['RMSE']:.4f}\n"
            f"R² Score: {metrics['R2']:.4f}"
        )
        self.metrics_label.setPlainText(text)
        self.status_bar.showMessage("Model trained successfully.")
        self._refresh_dashboard()

    def _on_training_failed(self, error_msg):
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Training Failed", error_msg)
        self.status_bar.showMessage("Training failed.")

    def _on_save_model(self):
        if not self.trainer.is_trained:
            QMessageBox.warning(self, "Model Not Trained", "Train a model before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)")
        if path:
            self.trainer.save(path, extra={"label_encoders": self.preprocessor.label_encoders})
            QMessageBox.information(self, "Model Saved", f"Model saved to:\n{path}")
            self.status_bar.showMessage(f"Model saved to {path}")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)")
        if path:
            try:
                payload = self.trainer.load(path)
                if "label_encoders" in payload:
                    self.preprocessor.label_encoders = payload["label_encoders"]
                self.predictor = Predictor(self.trainer, self.preprocessor)
                QMessageBox.information(self, "Model Loaded", f"Model loaded from:\n{path}")
                self.status_bar.showMessage(f"Model loaded from {path}")
                self._refresh_dashboard()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Load Error", str(exc))

    # ---------------------------------------------- Tab 5: Sales Prediction
    def _build_prediction_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form_box = QGroupBox("Future Sales Parameters")
        form_layout = QGridLayout(form_box)

        self.pred_store_id = QSpinBox()
        self.pred_store_id.setRange(1, 999)
        self.pred_category = QComboBox()
        self.pred_category.addItems(["Electronics", "Grocery", "Clothing", "Home", "Toys", "General"])
        self.pred_price = QDoubleSpinBox()
        self.pred_price.setRange(0, 100000)
        self.pred_price.setValue(100)
        self.pred_discount = QDoubleSpinBox()
        self.pred_discount.setRange(0, 100)
        self.pred_promotion = QComboBox()
        self.pred_promotion.addItems(["No", "Yes"])
        self.pred_holiday = QComboBox()
        self.pred_holiday.addItems(["No", "Yes"])
        self.pred_day_of_week = QSpinBox()
        self.pred_day_of_week.setRange(0, 6)
        self.pred_month = QSpinBox()
        self.pred_month.setRange(1, 12)
        self.pred_year = QSpinBox()
        self.pred_year.setRange(2020, 2035)
        self.pred_year.setValue(datetime.now().year)
        self.pred_prev_sales = QDoubleSpinBox()
        self.pred_prev_sales.setRange(0, 1000000)

        fields = [
            ("Store ID", self.pred_store_id), ("Category", self.pred_category),
            ("Selling Price", self.pred_price), ("Discount %", self.pred_discount),
            ("Promotion", self.pred_promotion), ("Holiday Indicator", self.pred_holiday),
            ("Day of Week (0=Mon)", self.pred_day_of_week), ("Month", self.pred_month),
            ("Year", self.pred_year), ("Previous Sales", self.pred_prev_sales),
        ]
        for i, (label_text, field) in enumerate(fields):
            form_layout.addWidget(QLabel(label_text), i // 2, (i % 2) * 2)
            form_layout.addWidget(field, i // 2, (i % 2) * 2 + 1)

        layout.addWidget(form_box)

        predict_btn = QPushButton("Predict Future Sales")
        predict_btn.clicked.connect(self._on_predict)
        layout.addWidget(predict_btn)

        self.prediction_result_label = QLabel("Predicted Sales: —")
        self.prediction_result_label.setObjectName("TitleLabel")
        layout.addWidget(self.prediction_result_label)

        self.prediction_confidence_label = QLabel("Confidence: —")
        layout.addWidget(self.prediction_confidence_label)
        layout.addStretch()
        return widget

    def _on_predict(self):
        if self.predictor is None:
            QMessageBox.warning(self, "Model Not Ready", "Train or load a model first.")
            return
        inputs = {
            "Store_ID": self.pred_store_id.value(),
            "Category": self.pred_category.currentText(),
            "Selling_Price": self.pred_price.value(),
            "Discount": self.pred_discount.value(),
            "Promotion": self.pred_promotion.currentText(),
            "Holiday_Indicator": self.pred_holiday.currentText(),
            "Day_of_Week": self.pred_day_of_week.value(),
            "Month": self.pred_month.value(),
            "Year": self.pred_year.value(),
            "Previous_Sales": self.pred_prev_sales.value(),
        }
        try:
            prediction, confidence = self.predictor.predict(inputs)
            self.last_prediction = {"inputs": inputs, "prediction": prediction, "confidence": confidence}
            self.prediction_result_label.setText(f"Predicted Sales: {format_currency(prediction)}")
            conf_text = f"{confidence}%" if confidence is not None else "N/A"
            self.prediction_confidence_label.setText(f"Confidence (indicative): {conf_text}")
            self.status_bar.showMessage("Prediction generated.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Prediction Error", str(exc))

    # ------------------------------------------------- Tab 6: Visualization
    def _build_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        self.chart_selector = QComboBox()
        self.chart_selector.addItems([
            "Historical Sales Trend", "Actual vs Predicted Sales",
            "Monthly Sales Forecast", "Feature Importance", "Error Distribution",
        ])
        render_btn = QPushButton("Render Chart")
        render_btn.clicked.connect(self._on_render_chart)
        btn_row.addWidget(self.chart_selector)
        btn_row.addWidget(render_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.chart_container_layout = QVBoxLayout()
        self.chart_widget_placeholder = QLabel("Select a chart and click 'Render Chart'.")
        self.chart_container_layout.addWidget(self.chart_widget_placeholder)
        layout.addLayout(self.chart_container_layout)

        self.current_canvas = None
        return widget

    def _clear_chart_area(self):
        if self.current_canvas is not None:
            self.chart_container_layout.removeWidget(self.current_canvas)
            self.current_canvas.setParent(None)
            self.current_canvas = None
        else:
            self.chart_container_layout.removeWidget(self.chart_widget_placeholder)
            self.chart_widget_placeholder.setParent(None)

    def _on_render_chart(self):
        choice = self.chart_selector.currentText()
        try:
            if choice == "Historical Sales Trend":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_historical_sales_trend(self.processed_df)
            elif choice == "Actual vs Predicted Sales":
                if self.y_pred is None:
                    raise ValueError("Train the model first.")
                fig = chart_actual_vs_predicted(self.y_test, self.y_pred)
            elif choice == "Monthly Sales Forecast":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = chart_monthly_sales_forecast(self.processed_df)
            elif choice == "Feature Importance":
                if not self.trainer.is_trained:
                    raise ValueError("Train the model first.")
                fig = chart_feature_importance(self.trainer.feature_importances())
            elif choice == "Error Distribution":
                if self.y_pred is None:
                    raise ValueError("Train the model first.")
                fig = chart_error_distribution(self.y_test, self.y_pred)
            else:
                return

            self._clear_chart_area()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(450)
            self.chart_container_layout.addWidget(canvas)
            self.current_canvas = canvas
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot Render Chart", str(exc))

    # ------------------------------------------------------ Tab 7: Reports
    def _build_reports_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        export_predictions_btn = QPushButton("Export Predictions (CSV)")
        export_predictions_btn.clicked.connect(self._on_export_predictions)
        export_report_btn = QPushButton("Export Forecast Report (TXT)")
        export_report_btn.clicked.connect(self._on_export_report)
        btn_row.addWidget(export_predictions_btn)
        btn_row.addWidget(export_report_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        layout.addWidget(self.report_preview)
        return widget

    def _on_export_predictions(self):
        if self.y_pred is None:
            QMessageBox.warning(self, "No Predictions", "Train the model to generate test predictions first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Predictions", os.path.join(REPORTS_DIR, "prediction_results.csv"), "CSV Files (*.csv)"
        )
        if path:
            out_df = pd.DataFrame({"Actual_Sales": self.y_test.values, "Predicted_Sales": self.y_pred})
            out_df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Predictions exported to:\n{path}")

    def _on_export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Forecast Report", os.path.join(REPORTS_DIR, "forecast_report.txt"), "Text Files (*.txt)"
        )
        if not path:
            return

        lines = [
            "=" * 60, "SALES FORECASTER (GBM) — FORECAST REPORT", "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        ]
        if self.dataset_path:
            lines.append(f"Dataset: {self.dataset_path}")
        if self.processed_df is not None:
            lines.append(f"Processed rows: {len(self.processed_df)}")
        lines.append("")

        if self.trainer.metrics:
            lines.append("Model Performance:")
            for k, v in self.trainer.metrics.items():
                lines.append(f"  {k}: {v:.4f}")
            lines.append("")

        if self.trainer.is_trained:
            lines.append("Feature Importances:")
            for name, importance in sorted(self.trainer.feature_importances().items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {importance:.4f}")
            lines.append("")

        if hasattr(self, "last_prediction"):
            lines.append("Last Prediction:")
            lines.append(f"  Inputs: {self.last_prediction['inputs']}")
            lines.append(f"  Predicted Sales: {format_currency(self.last_prediction['prediction'])}")
            lines.append(f"  Confidence: {self.last_prediction['confidence']}%")

        report_text = "\n".join(lines)
        with open(path, "w") as f:
            f.write(report_text)

        self.report_preview.setPlainText(report_text)
        QMessageBox.information(self, "Report Exported", f"Report saved to:\n{path}")


# ==========================================================================
# SECTION 8: Entry Point
# ==========================================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
