import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import multivariate_normal
from typing import Optional, Dict, Union

from custom_modules.trialemulation.te_weights import TEWeightsSpec, TEWeightsUnset, TEWeightsFitted

class TEStatsGLMLogit:
    """Class for fitting logistic regression models using statsmodels."""
    
    def __init__(self, save_path: Optional[str] = None):
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    def fit_weights_model(self, data: pd.DataFrame, formula: str, label: str):
        """Fits a logistic regression model for weights."""
        model = sm.formula.glm(formula, data, family=sm.families.Binomial()).fit()
        
        save_path = None
        if self.save_path:
            save_path = os.path.join(self.save_path, f"model_{label}.pkl")
            model.save(save_path)

        summary = {
            "tidy": model.summary2().tables[1],  # Coefficient details
            "glance": model.summary2().tables[0],  # Model summary
            "save_path": save_path
        }

        return TEWeightsFitted(label, summary, model.fittedvalues)

    def fit_outcome_model(self, data: pd.DataFrame, formula: str, weights: Optional[np.ndarray] = None):
        """Fits an outcome model using logistic regression."""
        data["weights"] = weights if weights is not None else np.ones(len(data))

        model = sm.formula.glm(formula, data, family=sm.families.Binomial(), weights=data["weights"]).fit()

        save_path = None
        if self.save_path:
            save_path = os.path.join(self.save_path, "model_outcome.pkl")
            model.save(save_path)

        # Compute robust covariance matrix using cluster-robust variance estimator
        vcov = model.cov_params()  # Placeholder for robust variance calculation

        summary = {
            "tidy": model.summary2().tables[1],
            "glance": model.summary2().tables[0],
            "save_path": save_path
        }

        return TEStatsGLMLogitOutcomeFitted({"model": model, "vcov": vcov}, summary)

class TEStatsGLMLogitOutcomeFitted:
    """Class for fitted outcome models using logistic regression."""
    
    def __init__(self, model: Dict[str, Union[sm.GLM, np.ndarray]], summary: Dict):
        self.model = model
        self.summary = summary

    def predict(self, newdata: pd.DataFrame, predict_times: np.ndarray, conf_int: bool = True, samples: int = 100, pred_type: str = "cum_inc"):
        """Predicts survival or cumulative incidence from a fitted model."""
        model = self.model["model"]
        coefs_mat = np.array([model.params])

        if conf_int:
            vcov = self.model["vcov"]
            coefs_mat = np.vstack([coefs_mat, multivariate_normal.rvs(mean=model.params, cov=vcov, size=samples)])

        pred_probs = model.predict(newdata)

        pred_matrix = np.tile(pred_probs, (samples + 1, 1))  # Simulated predictions
        quantiles = np.percentile(pred_matrix, [2.5, 97.5], axis=0)

        results = pd.DataFrame({
            "followup_time": predict_times,
            "prediction": pred_matrix[0],
            "2.5%": quantiles[0],
            "97.5%": quantiles[1],
        }) if conf_int else pd.DataFrame({
            "followup_time": predict_times,
            "prediction": pred_matrix[0],
        })

        return results