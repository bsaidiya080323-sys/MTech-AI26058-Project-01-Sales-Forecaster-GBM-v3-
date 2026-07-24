"""
train_model.py
Wraps Scikit-learn's GradientBoostingRegressor: training, evaluation,
and model persistence (save/load) via Joblib.
"""

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import FEATURE_COLUMNS, DEFAULT_MODEL_PATH


class ModelTrainer:
    """
    Handles training and evaluation of a Gradient Boosting Regression
    model for sales forecasting.
    """

    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.metrics = {}
        self.is_trained = False

    def train(self, X_train, y_train, progress_callback=None):
        """
        Trains the GradientBoostingRegressor. If progress_callback is
        supplied, it is invoked periodically with a percentage (0-100)
        to drive a GUI progress bar. Scikit-learn does not expose true
        per-tree callbacks for GradientBoostingRegressor easily, so we
        approximate progress using `staged_predict` after a fit, and
        also emit start/end signals for responsiveness.
        """
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
        """
        Evaluates the trained model on test data. Returns a dict of
        MAE, MSE, RMSE, and R2 metrics, and stores predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self.metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }
        return self.metrics, y_pred

    def feature_importances(self):
        """Returns a dict mapping feature name -> importance score."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(FEATURE_COLUMNS, importances))

    def predict(self, X):
        """Predicts sales for the given feature matrix/row(s)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------
    def save(self, path=DEFAULT_MODEL_PATH, extra=None):
        """
        Saves the trained model (and optionally extra metadata, such
        as label encoders) to disk using Joblib.
        """
        payload = {"model": self.model, "is_trained": self.is_trained}
        if extra:
            payload.update(extra)
        joblib.dump(payload, path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        """
        Loads a previously saved model bundle from disk. Returns the
        full payload dict so callers can also restore label encoders.
        """
        payload = joblib.load(path)
        self.model = payload["model"]
        self.is_trained = payload.get("is_trained", True)
        return payload
