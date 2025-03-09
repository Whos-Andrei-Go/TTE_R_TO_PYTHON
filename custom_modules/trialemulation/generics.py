from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
import numpy as np

class TEDataPrep:
    def __init__(self, N, min_period, max_period, censor_models=None, switch_models=None):
        self.N = N
        self.min_period = min_period
        self.max_period = max_period
        self.censor_models = censor_models or {}
        self.switch_models = switch_models or {}

    def summary(self):
        print(f"Number of observations in expanded data: {self.N}")
        print(f"First trial period: {self.min_period}")
        print(f"Last trial period: {self.max_period}\n")

        if self.switch_models or self.censor_models:
            print("-" * 40)
            print("Weight models")
            if self.switch_models:
                print("Treatment switch models:")
                for name, model in self.switch_models.items():
                    print(f"switch_models[{name}]:")
                    print(model.summary() if hasattr(model, "summary") else model)
                    print("-" * 40)
            if self.censor_models:
                print("Censoring models:")
                for name, model in self.censor_models.items():
                    print(f"censor_models[{name}]:")
                    print(model.summary() if hasattr(model, "summary") else model)
                    print("-" * 40)

class TEDatastore(ABC):
    @abstractmethod
    def save_expanded_data(self, data):
        pass

    @abstractmethod
    def read_expanded_data(self, period=None, subset_condition=None):
        pass

class TECSVStore(TEDatastore):
    def __init__(self, directory):
        self.directory = directory

    def save_expanded_data(self, data):
        file_path = f"{self.directory}/expanded_data.csv"
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def read_expanded_data(self, period=None, subset_condition=None):
        print(f"Reading data from {self.directory} (Period: {period}, Condition: {subset_condition})")

class TEOutcomeFitter:
    def __init__(self):
        self.model = None

    def fit_outcome_model(self, data, formula, weights=None):
        X = data.drop(columns=["outcome"])  # Assume 'outcome' is the dependent variable
        y = data["outcome"]
        self.model = LogisticRegression()
        self.model.fit(X, y, sample_weight=weights)
        print("Outcome model fitted.")
        
class TEWeightsFitter:
    def __init__(self):
        self.model = None

    def fit_weights_model(self, data, formula, model_type="logit", weights=None):
        """
        Fits a logistic regression model for treatment or censoring weights.

        Parameters:
        - data (pd.DataFrame): The dataset containing predictors and the outcome.
        - formula (str): The formula string (not used directly here, but could be parsed for Patsy).
        - model_type (str): The type of model ('logit' for logistic regression).
        - weights (array-like): Optional sample weights for weighted regression.
        """
        if "logit" not in model_type.lower():
            raise ValueError("Currently, only 'logit' (logistic regression) is supported.")

        # Assuming the first column is the binary outcome (censoring/treatment)
        outcome_col = data.columns[0]
        X = data.drop(columns=[outcome_col])  # Predictors
        y = data[outcome_col]  # Binary outcome

        # Fit logistic regression with optional weights
        self.model = LogisticRegression()
        self.model.fit(X, y, sample_weight=weights)

        print("Weights model fitted.")

    def predict_prob(self, X):
        """
        Predicts probabilities using the fitted model.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(X)[:, 1]
