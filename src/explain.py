"""
PredictIQ - Model Explainability Module (SHAP)
Explains individual machine risk predictions using Shapley Additive Explanations.
"""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PredictIQExplainer:
    def __init__(self, model, background_data=None):
        self.model = model
        # Use TreeExplainer for tree ensembles (XGBoost, Random Forest)
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model, background_data)

    def explain_sample(self, sample_df: pd.DataFrame):
        """
        Computes SHAP values for a single sample DataFrame (1 row).
        Returns a tuple of (shap_values_obj, explanation_summary_df).
        """
        shap_values = self.explainer(sample_df)
        
        # If output has multiple classes, pick failure class (index 1)
        if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
            single_explanation = shap_values[0, :, 1]
        else:
            single_explanation = shap_values[0]
            
        features = sample_df.columns.tolist()
        vals = sample_df.iloc[0].values
        shap_contribs = single_explanation.values

        summary_df = pd.DataFrame({
            "Feature": features,
            "Actual Value": vals,
            "SHAP Contribution": shap_contribs,
            "Abs Impact": np.abs(shap_contribs)
        }).sort_values(by="Abs Impact", ascending=False)

        summary_df["Impact Direction"] = summary_df["SHAP Contribution"].apply(
            lambda x: "Pushes Risk UP (Hazard)" if x > 0 else "Lowers Risk (Safe)"
        )

        return single_explanation, summary_df

    def plot_waterfall(self, single_explanation, max_display=8):
        """
        Generates a clean SHAP waterfall plot as a Matplotlib Figure.
        """
        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(single_explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
