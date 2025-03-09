import numpy as np
import pandas as pd
import patsy

from custom_modules.trialemulation.te_weights import TeWeightsSpec, TeWeightsUnset, TeWeightsFitted

class CalculateWeights:
    """Handles calculation of inverse probability of censoring and switching weights."""

    def __init__(self, trial_sequence):
        """Initialize with a TrialSequence object."""
        self.trial_sequence = trial_sequence

    def calculate_weights_trial_seq(self, quiet=False, switch_weights=True, censor_weights=True):
        """Calculate weights for the trial sequence."""
        ts = self.trial_sequence

        if ts.data is None:
            raise ValueError("Data must be set before calculating weights.")

        ts.data["wt"] = 1  # Initialize weights to 1

        if switch_weights and isinstance(ts.switch_weights, TeWeightsSpec):
            self.calculate_switch_weights()
            ts.data["wt"] *= ts.data["wtS"]

        if censor_weights and isinstance(ts.censor_weights, TeWeightsSpec):
            self.calculate_censor_weights()
            ts.data["wt"] *= ts.data["wtC"]

        if not quiet:
            print("Weights calculated successfully.")

        return ts

    def calculate_switch_weights(self):
        """Calculate switch weights using logistic regression models."""
        ts = self.trial_sequence

        if ts.switch_weights is None or ts.switch_weights.model_fitter is None:
            raise ValueError("Switch weight model fitter is missing. Did you call set_switch_weight_model()?")

        ts.data["p_n"] = np.nan
        ts.data["p_d"] = np.nan
        ts.switch_weights.fitted = {}

        # Fit numerator model
        model_1_index = ts.data.query("am_1 == 1").index
        fitted_n1 = ts.switch_weights.model_fitter.fit_weights_model(ts.data.loc[model_1_index], ts.switch_weights.numerator)
        
        ts.switch_weights.fitted["n1"] = TeWeightsFitted(
            label="P(treatment = 1 | previous treatment = 1) for numerator",
            summary=fitted_n1["summary"],
            fitted_model=fitted_n1["model"],
            save_path=fitted_n1.get("save_path")
        )
        
        ts.data.loc[model_1_index, "p_n"] = fitted_n1["model"].predict(ts.data.loc[model_1_index])

        # Fit denominator model
        fitted_d1 = ts.switch_weights.model_fitter.fit_weights_model(ts.data.loc[model_1_index], ts.switch_weights.denominator)
        
        ts.switch_weights.fitted["d1"] = TeWeightsFitted(
            label="P(treatment = 1 | previous treatment = 1) for denominator",
            summary=fitted_d1["summary"],
            fitted_model=fitted_d1["model"],
            save_path=fitted_d1.get("save_path")
        )

        ts.data.loc[model_1_index, "p_d"] = fitted_d1["model"].predict(ts.data.loc[model_1_index])

        # Compute switch weights (wtS)
        ts.data["wtS"] = np.where(
            ts.data["treatment"] == 0,  
            (1.0 - ts.data["p_n"]) / (1.0 - ts.data["p_d"]),  
            ts.data["p_n"] / ts.data["p_d"]
        )

        # Ensure there are no NaN values in `wtS`
        ts.data["wtS"] = ts.data["wtS"].fillna(1)


    def calculate_censor_weights(self):
        """Calculate censoring weights using logistic regression models."""
        ts = self.trial_sequence

        if ts.censor_weights is None:
            raise ValueError("Censor weights have not been set.")

        ts.data["pC_n"] = np.nan
        ts.data["pC_d"] = np.nan

        elig_0_index = ts.data.query("am_1 == 0").index
        elig_1_index = ts.data.query("am_1 == 1").index

        # Fit numerator model
        ts.censor_weights.fitted = {}

        ts.censor_weights.fitted["n0"] = ts.censor_weights.model_fitter.fit_weights_model(
            ts.data.loc[elig_0_index], ts.censor_weights.numerator
        )
        ts.data.loc[elig_0_index, "pC_n"] = ts.censor_weights.fitted["n0"]["model"].predict(ts.data.loc[elig_0_index])

        ts.censor_weights.fitted["n1"] = ts.censor_weights.model_fitter.fit_weights_model(
            ts.data.loc[elig_1_index], ts.censor_weights.numerator
        )
        ts.data.loc[elig_1_index, "pC_n"] = ts.censor_weights.fitted["n1"]["model"].predict(ts.data.loc[elig_1_index])

        # Fit denominator model
        ts.censor_weights.fitted["d0"] = ts.censor_weights.model_fitter.fit_weights_model(
            ts.data.loc[elig_0_index], ts.censor_weights.denominator
        )
        ts.data.loc[elig_0_index, "pC_d"] = ts.censor_weights.fitted["d0"]["model"].predict(ts.data.loc[elig_0_index])

        ts.censor_weights.fitted["d1"] = ts.censor_weights.model_fitter.fit_weights_model(
            ts.data.loc[elig_1_index], ts.censor_weights.denominator
        )
        ts.data.loc[elig_1_index, "pC_d"] = ts.censor_weights.fitted["d1"]["model"].predict(ts.data.loc[elig_1_index])

        # Compute censor weights
        ts.data["wtC"] = ts.data["pC_n"] / ts.data["pC_d"]
        ts.data["wtC"] = ts.data["wtC"].fillna(1)
