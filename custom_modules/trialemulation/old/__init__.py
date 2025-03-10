import pandas as pd
import numpy as np
import statsmodels.api as sm


class TrialSequence:
    def __init__(self, data=None, estimand="Generic"):
        """
        Base class for a trial sequence.

        :param data: Pandas DataFrame containing trial data.
        :param estimand: Type of estimand (e.g., "ITT", "PP", "AT").
        """
        self.data = data if data is not None else pd.DataFrame()
        self.estimand = estimand
        self.switch_weights = None
        self.censor_weights = None
        self.outcome_model = None

    def set_data(self, data):
        """Set the trial data."""
        self.data = data

    def set_switch_weight_model(self, numerator_formula, denominator_formula):
        """Define the treatment switching weight model."""
        self.switch_weights = {
            "numerator": numerator_formula,
            "denominator": denominator_formula
        }

    def set_censor_weight_model(self, numerator_formula, denominator_formula):
        """Define the censoring weight model."""
        self.censor_weights = {
            "numerator": numerator_formula,
            "denominator": denominator_formula
        }

    def calculate_weights(self, switch_weights=True, censor_weights=True):
        """Calculate weights based on censoring and treatment switching models."""
        if self.data.empty:
            raise ValueError("Data must be set before calculating weights.")

        self.data["wt"] = 1  # Initialize weight column

        if switch_weights and self.switch_weights:
            self.calculate_switch_weights()
            self.data["wt"] *= self.data["wtS"]

        if censor_weights and self.censor_weights:
            self.calculate_censor_weights()
            self.data["wt"] *= self.data["wtC"]

        return self

    def calculate_switch_weights(self):
        """Calculate inverse probability of treatment switching weights."""
        if not self.switch_weights:
            raise ValueError("Switch weight models are not defined.")

        self.data["p_n"] = np.nan
        self.data["p_d"] = np.nan

        for treatment_val in [0, 1]:
            subset = self.data[self.data["am_1"] == treatment_val]
            self.data.loc[subset.index, "p_n"] = self.fit_weights_model(subset, self.switch_weights["numerator"])
            self.data.loc[subset.index, "p_d"] = self.fit_weights_model(subset, self.switch_weights["denominator"])

        self.data["wtS"] = np.where(
            self.data["treatment"] == 0, (1.0 - self.data["p_n"]) / (1.0 - self.data["p_d"]),
            self.data["p_n"] / self.data["p_d"]
        )

        self.data["wtS"].fillna(1, inplace=True)

    def calculate_censor_weights(self):
        """Calculate inverse probability of censoring weights."""
        if not self.censor_weights:
            raise ValueError("Censor weight models are not defined.")

        self.data["pC_n"] = self.fit_weights_model(self.data, self.censor_weights["numerator"])
        self.data["pC_d"] = self.fit_weights_model(self.data, self.censor_weights["denominator"])

        self.data["pC_d"].fillna(1, inplace=True)
        self.data["pC_n"].fillna(1, inplace=True)
        self.data["wtC"] = self.data["pC_n"] / self.data["pC_d"]

    def fit_weights_model(self, data, formula):
        """Fit logistic regression model to compute probabilities."""
        y, X = self.extract_formula_data(data, formula)
        model = sm.Logit(y, X).fit(disp=0)
        return model.predict(X)

    def extract_formula_data(self, data, formula):
        """Extract dependent and independent variables from a formula."""
        target_var, predictor_vars = formula.split("~")
        target_var = target_var.strip()
        predictor_vars = [var.strip() for var in predictor_vars.split("+")]

        y = data[target_var].astype(float)
        X = sm.add_constant(data[predictor_vars].astype(float))
        return y, X

    def __repr__(self):
        return f"TrialSequence(estimand={self.estimand}, data_shape={self.data.shape})"


# Subclasses for Specific Estimands
class TrialSequenceITT(TrialSequence):
    def __init__(self, data=None):
        super().__init__(data, estimand="ITT")


class TrialSequencePP(TrialSequence):
    def __init__(self, data=None):
        super().__init__(data, estimand="PP")
        self.switch_weights = None


class TrialSequenceAT(TrialSequence):
    def __init__(self, data=None):
        super().__init__(data, estimand="AT")
        self.switch_weights = None
