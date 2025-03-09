import numpy as np
import pandas as pd
import patsy

from custom_modules.trialemulation.te_weights import TeWeightsSpec, TeWeightsUnset, TeWeightsFitted

class CalculateWeights:
    """Handles calculation of inverse probability of censoring and switching weights."""

    def __init__(self, trial_sequence):
        """Initialize with a TrialSequence object."""
        self.trial_sequence = trial_sequence

    def calculate_weights_trial_seq(self, quiet=False):
        """Calculate weights for the trial sequence using the appropriate weight model."""
        ts = self.trial_sequence

        if ts.data is None:
            raise ValueError("Data must be set before calculating weights.")

        ts.data["wt"] = 1  # Initialize weights to 1

        # Determine if we should calculate switch or censor weights
        has_switch_weights = isinstance(ts.switch_weights, TeWeightsSpec) and ts.switch_weights.model_fitter is not None
        has_censor_weights = isinstance(ts.censor_weights, TeWeightsSpec) and ts.censor_weights.model_fitter is not None

        if has_switch_weights:
            self.calculate_switch_weights()
            ts.data["wt"] *= ts.data["wtS"]  # Apply switch weights

        if has_censor_weights:
            self.calculate_censor_weights()
            ts.data["wt"] *= ts.data["wtC"]  # Apply censor weights

        if not quiet:
            print("Weights calculated successfully.")

        ts.data['wt'] = ts.data['wt'].fillna(1.0)

        return ts  # Return updated trial sequence

    def calculate_switch_weights(self):
        """Calculate switch weights using logistic regression models."""
        ts = self.trial_sequence

        if ts.switch_weights is None or ts.switch_weights.model_fitter is None:
            raise ValueError("Switch weight model fitter is missing. Did you call set_switch_weight_model()?")

        ts.data["p_n"] = np.nan
        ts.data["p_d"] = np.nan
        ts.switch_weights.fitted = {}

        # Switch from Treatment = 1
        if "eligible_wts_1" in ts.data.columns:
            data_1_expr = (ts.data["am_1"] == 1) & (ts.data["eligible_wts_1"] == 1)
        else:
            data_1_expr = ts.data["am_1"] == 1

        model_1_index = ts.data[data_1_expr].index

        # Fit numerator model for treatment = 1
        fitted_n1 = ts.switch_weights.model_fitter.fit_weights_model(
            ts.data.loc[model_1_index], ts.switch_weights.numerator
        )
        
        ts.switch_weights.fitted["n1"] = fitted_n1
        
        ts.data.loc[model_1_index, "p_n"] = fitted_n1["model"].predict(ts.data.loc[model_1_index])

        # Fit denominator model for treatment = 1
        fitted_d1 = ts.switch_weights.model_fitter.fit_weights_model(
            ts.data.loc[model_1_index], ts.switch_weights.denominator
        )
        
        ts.switch_weights.fitted["d1"] = TeWeightsFitted(
            label="P(treatment = 1 | previous treatment = 1) for denominator",
            summary=fitted_d1["summary"],
            fitted_model=fitted_d1["model"],
            save_path=fitted_d1.get("save_path")
        )

        ts.data.loc[model_1_index, "p_d"] = fitted_d1["model"].predict(ts.data.loc[model_1_index])

        # Switch from Treatment = 0
        if "eligible_wts_1" in ts.data.columns:
            data_0_expr = (ts.data["am_1"] == 0) & (ts.data["eligible_wts_1"] == 0)
        else:
            data_0_expr = ts.data["am_1"] == 0

        model_0_index = ts.data[data_0_expr].index

        # Fit numerator model for treatment = 0
        fitted_n0 = ts.switch_weights.model_fitter.fit_weights_model(
            ts.data.loc[model_0_index], ts.switch_weights.numerator
        )
        
        ts.switch_weights.fitted["n0"] = TeWeightsFitted(
            label="P(treatment = 1 | previous treatment = 0) for numerator",
            summary=fitted_n0["summary"],
            fitted_model=fitted_n0["model"],
            save_path=fitted_n0.get("save_path")
        )
        
        ts.data.loc[model_0_index, "p_n"] = fitted_n0["model"].predict(ts.data.loc[model_0_index])

        # Fit denominator model for treatment = 0
        fitted_d0 = ts.switch_weights.model_fitter.fit_weights_model(
            ts.data.loc[model_0_index], ts.switch_weights.denominator
        )
        
        ts.switch_weights.fitted["d0"] = TeWeightsFitted(
            label="P(treatment = 1 | previous treatment = 0) for denominator",
            summary=fitted_d0["summary"],
            fitted_model=fitted_d0["model"],
            save_path=fitted_d0.get("save_path")
        )

        ts.data.loc[model_0_index, "p_d"] = fitted_d0["model"].predict(ts.data.loc[model_0_index])

        # Compute switch weights (wtS)
        if "eligible_wts_0" in ts.data.columns or "eligible_wts_1" in ts.data.columns:
            ts.data.loc[(ts.data["eligible_wts_0"] == 1) | (ts.data["eligible_wts_1"] == 1), "wtS"] = np.where(
                ts.data["treatment"] == 0,
                (1.0 - ts.data["p_n"]) / (1.0 - ts.data["p_d"]),
                ts.data["p_n"] / ts.data["p_d"]
            )
        else:
            ts.data.loc[ts.data["treatment"] == 0, "wtS"] = (1.0 - ts.data["p_n"]) / (1.0 - ts.data["p_d"])
            ts.data.loc[ts.data["treatment"] == 1, "wtS"] = ts.data["p_n"] / ts.data["p_d"]

        # Ensure there are no NaN values in `wtS`
        ts.data["wtS"] = ts.data["wtS"].fillna(1)

        return self

    def calculate_censor_weights(self):
        """Calculate censoring weights using logistic regression models."""
        ts = self.trial_sequence

        if ts.censor_weights is None:
            raise ValueError("Censor weights have not been set.")

        ts.data["pC_n"] = np.nan
        ts.data["pC_d"] = np.nan

        # Determine eligibility indices
        elig_0_index = ts.data.query("am_1 == 0").index
        elig_1_index = ts.data.query("am_1 == 1").index

        # Fit numerator model
        ts.censor_weights.fitted = {}

        if ts.censor_weights.pool_numerator:
            # Fit model for the entire dataset
            ts.censor_weights.fitted["n"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data, ts.censor_weights.numerator
            )
            ts.data["pC_n"] = ts.censor_weights.fitted["n"]["model"].predict(ts.data)
            ts.censor_weights.data_subset_expr["n"] = True
        else:
            # Fit model for eligible group 0
            ts.censor_weights.fitted["n0"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data.loc[elig_0_index], ts.censor_weights.numerator
            )
            ts.data.loc[elig_0_index, "pC_n"] = ts.censor_weights.fitted["n0"]["model"].predict(ts.data.loc[elig_0_index])
            ts.censor_weights.data_subset_expr["n0"] = "am_1 == 0"

            # Fit model for eligible group 1
            ts.censor_weights.fitted["n1"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data.loc[elig_1_index], ts.censor_weights.numerator
            )
            ts.data.loc[elig_1_index, "pC_n"] = ts.censor_weights.fitted["n1"]["model"].predict(ts.data.loc[elig_1_index])
            ts.censor_weights.data_subset_expr["n1"] = "am_1 == 1"

        # Fit denominator model
        if ts.censor_weights.pool_denominator:
            ts.censor_weights.fitted["d"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data, ts.censor_weights.denominator
            )
            ts.data["pC_d"] = ts.censor_weights.fitted["d"]["model"].predict(ts.data)
            ts.censor_weights.data_subset_expr["d"] = True
        else:
            # Fit model for eligible group 0
            ts.censor_weights.fitted["d0"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data.loc[elig_0_index], ts.censor_weights.denominator
            )
            ts.data.loc[elig_0_index, "pC_d"] = ts.censor_weights.fitted["d0"]["model"].predict(ts.data.loc[elig_0_index])
            ts.censor_weights.data_subset_expr["d0"] = "am_1 == 0"

            # Fit model for eligible group 1
            ts.censor_weights.fitted["d1"] = ts.censor_weights.model_fitter.fit_weights_model(
                ts.data.loc[elig_1_index], ts.censor_weights.denominator
            )
            ts.data.loc[elig_1_index, "pC_d"] = ts.censor_weights.fitted["d1"]["model"].predict(ts.data.loc[elig_1_index])
            ts.censor_weights.data_subset_expr["d1"] = "am_1 == 1"

        # Compute censor weights
        ts.data["pC_d"].fillna(1, inplace=True)
        ts.data["pC_n"].fillna(1, inplace=True)
        ts.data["wtC"] = ts.data["pC_n"] / ts.data["pC_d"]

        return self