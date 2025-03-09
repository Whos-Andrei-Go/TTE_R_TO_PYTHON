import os
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

class GLMLogitModelFitter:
    """Fit logistic regression models using statsmodels' GLM with a logit link."""
    
    def __init__(self, save_path=None):
        self.save_path = save_path  # Directory to save models (optional)

    def fit_weights_model(self, data, formula):
        """Fit weight models using logistic regression."""
        model = sm.GLM.from_formula(formula, data, family=sm.families.Binomial()).fit()
        
        # Save model if save_path is specified
        model_file = None
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            model_file = os.path.join(self.save_path, "weights_model.pkl")
            model.save(model_file)
        
        return {
            "model": model,
            "summary": model.summary(),
            "save_path": model_file
        }
    
    def fit_outcome_model(self, data, formula, weights=None):
        """Fit outcome model using logistic regression with robust variance estimation."""
        
        # Assign weights (default to 1 if none provided)
        data["weights"] = 1 if weights is None else weights
        
        # Fit logistic regression model
        model = sm.GLM.from_formula(formula, data, family=sm.families.Binomial(), freq_weights=data["weights"]).fit()
        
        # Compute robust covariance matrix (clustered by `id`)
        robust_vcov = cov_cluster(model, data["id"])
        
        return {
            "model": model,
            "summary": model.summary(),
            "vcov": robust_vcov
        }

def stats_glm_logit(save_path=None):
    """Mimic the behavior of stats_glm_logit() in R."""
    return GLMLogitModelFitter(save_path=save_path)
