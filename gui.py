"""
gui.py
PyQt5 multi-tab desktop interface for the Sales Forecaster (GBM) app.

Tabs:
    1. Dashboard
    2. Dataset Upload
    3. Data Preprocessing
    4. Model Training
    5. Sales Prediction
    6. Visualization
    7. Reports
"""

import os
import sys
import traceback
from datetime import datetime

import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QTextEdit,
    QMessageBox,
    QStatusBar,
    QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from utils import (
    APP_STYLE_DARK,
    DEFAULT_MODEL_PATH,
    DEFAULT_DATASET_PATH,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    format_currency,
    ensure_dirs,
    generate_sample_dataset,
)
from preprocessing import DataPreprocessor
from train_model import ModelTrainer
from predictor import Predictor
import visualization as viz


# ==========================================================================
# Background worker thread for model training (keeps GUI responsive)
# ==========================================================================
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, trainer, X_train, X_test, y_train, y_test):
        super().__init__()
        self.trainer = trainer
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        try:
            self.trainer.train(self.X_train, self.y_train, progress_callback=self.progress.emit)
            metrics, y_pred = self.trainer.evaluate(self.X_test, self.y_test)
            self.finished_ok.emit({"metrics": metrics, "y_pred": y_pred})
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ==========================================================================
# Main Window
# ==========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        self.setWindowTitle("Sales Forecaster (GBM) v3 — Retail Analytics Suite")
        self.resize(1280, 820)
        self.setStyleSheet(APP_STYLE_DARK)

        # Shared application state
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.predictor = None
        self.dataset_path = None
        self.processed_df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None
        self.training_worker = None

        self._build_ui()

    # ----------------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Tab 1: Dashboard
    # ------------------------------------------------------------
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

        for i, lbl in enumerate(
            [self.dash_dataset_label, self.dash_rows_label, self.dash_model_label, self.dash_r2_label]
        ):
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

    # ------------------------------------------------------------
    # Tab 2: Dataset Upload
    # ------------------------------------------------------------
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
        path = generate_sample_dataset()
        self._load_dataset(path)

    def _load_dataset(self, path):
        try:
            df = pd.read_csv(path)
            self.dataset_path = path
            self.raw_preview_df = df
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
                item = QTableWidgetItem(str(df.iloc[r, c]))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    # ------------------------------------------------------------
    # Tab 3: Preprocessing
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Tab 4: Model Training
    # ------------------------------------------------------------
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)"
        )
        if path:
            self.trainer.save(path, extra={"label_encoders": self.preprocessor.label_encoders})
            QMessageBox.information(self, "Model Saved", f"Model saved to:\n{path}")
            self.status_bar.showMessage(f"Model saved to {path}")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", DEFAULT_MODEL_PATH, "Pickle Files (*.pkl)"
        )
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

    # ------------------------------------------------------------
    # Tab 5: Sales Prediction
    # ------------------------------------------------------------
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
            ("Store ID", self.pred_store_id),
            ("Category", self.pred_category),
            ("Selling Price", self.pred_price),
            ("Discount %", self.pred_discount),
            ("Promotion", self.pred_promotion),
            ("Holiday Indicator", self.pred_holiday),
            ("Day of Week (0=Mon)", self.pred_day_of_week),
            ("Month", self.pred_month),
            ("Year", self.pred_year),
            ("Previous Sales", self.pred_prev_sales),
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

    # ------------------------------------------------------------
    # Tab 6: Visualization
    # ------------------------------------------------------------
    def _build_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        self.chart_selector = QComboBox()
        self.chart_selector.addItems(
            [
                "Historical Sales Trend",
                "Actual vs Predicted Sales",
                "Monthly Sales Forecast",
                "Feature Importance",
                "Error Distribution",
            ]
        )
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
                fig = viz.historical_sales_trend(self.processed_df)
            elif choice == "Actual vs Predicted Sales":
                if self.y_pred is None:
                    raise ValueError("Train the model first.")
                fig = viz.actual_vs_predicted(self.y_test, self.y_pred)
            elif choice == "Monthly Sales Forecast":
                if self.processed_df is None:
                    raise ValueError("Load and preprocess a dataset first.")
                fig = viz.monthly_sales_forecast(self.processed_df)
            elif choice == "Feature Importance":
                if not self.trainer.is_trained:
                    raise ValueError("Train the model first.")
                fig = viz.feature_importance_chart(self.trainer.feature_importances())
            elif choice == "Error Distribution":
                if self.y_pred is None:
                    raise ValueError("Train the model first.")
                fig = viz.error_distribution(self.y_test, self.y_pred)
            else:
                return

            self._clear_chart_area()
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(450)
            self.chart_container_layout.addWidget(canvas)
            self.current_canvas = canvas
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot Render Chart", str(exc))

    # ------------------------------------------------------------
    # Tab 7: Reports
    # ------------------------------------------------------------
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
            self,
            "Export Predictions",
            os.path.join(REPORTS_DIR, "prediction_results.csv"),
            "CSV Files (*.csv)",
        )
        if path:
            out_df = pd.DataFrame(
                {"Actual_Sales": self.y_test.values, "Predicted_Sales": self.y_pred}
            )
            out_df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Predictions exported to:\n{path}")

    def _on_export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Forecast Report",
            os.path.join(REPORTS_DIR, "forecast_report.txt"),
            "Text Files (*.txt)",
        )
        if not path:
            return

        lines = [
            "=" * 60,
            "SALES FORECASTER (GBM) — FORECAST REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
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
            for name, importance in sorted(
                self.trainer.feature_importances().items(), key=lambda x: -x[1]
            ):
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


def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
