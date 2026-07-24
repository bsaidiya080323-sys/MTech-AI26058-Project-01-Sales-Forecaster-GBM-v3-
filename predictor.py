"""
predictor.py
Builds a single-row feature vector from user-entered GUI inputs and
runs it through the trained model + preprocessing encoders to
produce a future sales prediction.
"""

import pandas as pd

from utils import FEATURE_COLUMNS


class Predictor:
    """
    Bridges GUI input fields to the trained ModelTrainer + DataPreprocessor,
    producing a single predicted sales value along with a naive
    confidence indication derived from the model's tree-level
    prediction spread.
    """

    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def build_feature_row(self, inputs: dict) -> pd.DataFrame:
        """
        inputs is a dict with raw (human-readable) values, e.g.:
            {
                "Store_ID": 3,
                "Category": "Electronics",
                "Selling_Price": 120.0,
                "Discount": 10.0,
                "Promotion": "Yes",
                "Holiday_Indicator": "No",
                "Day_of_Week": 4,
                "Month": 12,
                "Year": 2026,
                "Previous_Sales": 350,
            }
        Categorical fields are encoded using the fitted LabelEncoders
        from the preprocessor.
        """
        row = {}
        for col in FEATURE_COLUMNS:
            value = inputs.get(col)
            if col in self.preprocessor.label_encoders:
                row[col] = self.preprocessor.encode_single_value(col, value)
            else:
                row[col] = value

        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, inputs: dict):
        """
        Returns (predicted_value, confidence_pct) for the given raw
        input dictionary.
        """
        X_row = self.build_feature_row(inputs)
        prediction = float(self.trainer.predict(X_row)[0])
        confidence = self._estimate_confidence(X_row)
        return prediction, confidence

    def _estimate_confidence(self, X_row):
        """
        Naive confidence estimate: uses the spread of staged
        predictions (across boosting stages) as an inverse proxy for
        uncertainty. Purely indicative, not a statistical guarantee.
        """
        try:
            staged = list(self.trainer.model.staged_predict(X_row))
            tail = staged[-10:] if len(staged) >= 10 else staged
            values = [float(v[0]) for v in tail]
            spread = max(values) - min(values) if values else 0.0
            mean_val = sum(values) / len(values) if values else 1.0
            rel_spread = spread / (abs(mean_val) + 1e-6)
            confidence = max(0.0, min(100.0, 100 * (1 - rel_spread)))
            return round(confidence, 1)
        except Exception:
            return None
