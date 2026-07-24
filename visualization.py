"""
visualization.py
Builds Matplotlib Figure objects for embedding in the PyQt5 GUI via
FigureCanvasQTAgg. Each function returns a ready-to-embed Figure.
"""

import numpy as np
import matplotlib
matplotlib.use("QT5Agg")
from matplotlib.figure import Figure

DARK_BG = "#1e1e2f"
PANEL_BG = "#24243a"
ACCENT = "#4c5fd7"
ACCENT2 = "#8ea2ff"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#3a3a52"


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


def historical_sales_trend(df, date_col="Date", sales_col="Sales"):
    """Line chart of historical sales over time (resampled by day)."""
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)

    if date_col in df.columns:
        temp = df[[date_col, sales_col]].copy()
        temp[date_col] = matplotlib.dates.date2num(
            __import__("pandas").to_datetime(temp[date_col])
        )
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


def actual_vs_predicted(y_test, y_pred):
    """Scatter plot comparing actual vs predicted sales values."""
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    ax.scatter(y_test_arr, y_pred_arr, s=18, color=ACCENT2, alpha=0.7, edgecolors="none")
    min_val = min(y_test_arr.min(), y_pred_arr.min())
    max_val = max(y_test_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#ff6b6b", linewidth=1.5, linestyle="--")

    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    fig.tight_layout()
    return fig


def monthly_sales_forecast(df, month_col="Month", sales_col="Sales"):
    """Bar chart of average sales aggregated by month."""
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


def feature_importance_chart(importances: dict):
    """Horizontal bar chart of feature importances from the model."""
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_axis(ax)

    items = sorted(importances.items(), key=lambda x: x[1])
    names = [i[0] for i in items]
    values = [i[1] for i in items]

    ax.barh(names, values, color=ACCENT2)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def error_distribution(y_test, y_pred):
    """Histogram of residual errors (actual - predicted)."""
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
