from .te_weights import TEWeightsSpec, TEWeightsUnset, TEWeightsFitted
from .generics import TEWeightsFitter

import pandas as pd
import numpy as np

def calculate_weights_trial_seq(obj, quiet=False, switch_weights=True, censor_weights=True):
    """Calculate weights for a trial sequence."""
    obj.data["wt"] = 1  # Initialize weight column

    # Calculate switch weights if required
    if switch_weights and not isinstance(obj.switch_weights, TEWeightsUnset):
        obj = calculate_switch_weights(obj)
        obj.data["wt"] *= obj.data["wtS"]

    # Calculate censor weights if required
    if censor_weights and not isinstance(obj.censor_weights, TEWeightsUnset):
        obj = calculate_censor_weights(obj)
        obj.data["wt"] *= obj.data["wtC"]

    return obj


def calculate_switch_weights(obj):
    """Calculate treatment switching weights"""
    data = obj.data
    fitter = obj.switch_weights.model_fitter

    # Treatment = 1
    if "eligible_wts_1" in data.columns:
        data_1_expr = ((data["am_1"] == 1) & (data["eligible_wts_1"] == 1)).astype(bool)
    else:
        data_1_expr = (data["am_1"] == 1).astype(bool)

    model_1_index = data.index[data_1_expr] if not data_1_expr.empty else []

    if not model_1_index.empty:
        try:
            obj.switch_weights.fitted["n1"] = fitter.fit_weights_model(
                data.loc[model_1_index], obj.switch_weights.numerator,
                "P(treatment = 1 | previous treatment = 1) for numerator"
            )
            data.loc[model_1_index, "p_n"] = obj.switch_weights.fitted["n1"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in numerator model for treatment = 1:", e)
            data.loc[model_1_index, "p_n"] = 0.5  # Set a neutral value if perfect separation occurs

        try:
            obj.switch_weights.fitted["d1"] = fitter.fit_weights_model(
                data.loc[model_1_index], obj.switch_weights.denominator,
                "P(treatment = 1 | previous treatment = 1) for denominator"
            )
            data.loc[model_1_index, "p_d"] = obj.switch_weights.fitted["d1"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in denominator model for treatment = 1:", e)
            data.loc[model_1_index, "p_d"] = 0.5  # Set a neutral value if perfect separation occurs

    # Treatment = 0
    if "eligible_wts_1" in data.columns:
        data_0_expr = (data["am_1"] == 0) & (data["eligible_wts_1"] == 0)
    else:
        data_0_expr = data["am_1"] == 0

    model_0_index = data.index[data_0_expr] if not data_0_expr.empty else []

    if not model_0_index.empty:
        try:
            obj.switch_weights.fitted["n0"] = fitter.fit_weights_model(
                data.loc[model_0_index], obj.switch_weights.numerator,
                "P(treatment = 1 | previous treatment = 0) for numerator"
            )
            data.loc[model_0_index, "p_n"] = obj.switch_weights.fitted["n0"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in numerator model for treatment = 0:", e)
            data.loc[model_0_index, "p_n"] = 0.5  # Set a neutral value if perfect separation occurs

        try:
            obj.switch_weights.fitted["d0"] = fitter.fit_weights_model(
                data.loc[model_0_index], obj.switch_weights.denominator,
                "P(treatment = 1 | previous treatment = 0) for denominator"
            )
            data.loc[model_0_index, "p_d"] = obj.switch_weights.fitted["d0"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in denominator model for treatment = 0:", e)
            data.loc[model_0_index, "p_d"] = 0.5  # Set a neutral value if perfect separation occurs

    # Combine Weights
    data["wtS"] = 1.0
    mask_0 = (data["treatment"] == 0)
    mask_1 = (data["treatment"] == 1)

    # Avoid division by zero
    data.loc[mask_0, "wtS"] = (1.0 - data["p_n"].fillna(0)) / (1.0 - data["p_d"].fillna(1))
    data.loc[mask_1, "wtS"] = data["p_n"].fillna(0) / data["p_d"].fillna(1)

    # Fill missing values in wtS
    data["wtS"].fillna(1)

    return obj


def calculate_censor_weights(obj):
    """Calculate censoring weights"""
    data = obj.data
    fitter = obj.censor_weights.model_fitter

    if obj.censor_weights.pool_numerator:
        obj.censor_weights.fitted["n"] = fitter.fit_weights_model(
            data, obj.censor_weights.numerator, "P(censor_event = 0 | X) for numerator"
        )
        data["pC_n"] = obj.censor_weights.fitted["n"].fitted
    else:
        mask_0 = data["am_1"] == 0
        mask_1 = data["am_1"] == 1

        try:
            obj.censor_weights.fitted["n0"] = fitter.fit_weights_model(
                data[mask_0], obj.censor_weights.numerator,
                "P(censor_event = 0 | X, previous treatment = 0) for numerator"
            )
            data.loc[mask_0, "pC_n"] = obj.censor_weights.fitted["n0"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in numerator model for treatment = 0:", e)
            data.loc[mask_0, "pC_n"] = 0.5  # Set a neutral value if perfect separation occurs

        try:
            obj.censor_weights.fitted["n1"] = fitter.fit_weights_model(
                data[mask_1], obj.censor_weights.numerator,
                "P(censor_event = 0 | X, previous treatment = 1) for numerator"
            )
            data.loc[mask_1, "pC_n"] = obj.censor_weights.fitted["n1"].fitted
        except ValueError as e:
            print("Warning: Potential perfect separation issue in numerator model for treatment = 1:", e)
            data.loc[mask_1, "pC_n"] = 0.5  # Set a neutral value if perfect separation occurs

    # Ensure pC_d is created or set a default value
    if "pC_d" not in data.columns:
        data["pC_d"] = 1  # or some other default value

    # Fill missing values for pC_d and pC_n
    data["pC_d"].fillna(1)
    data["pC_n"].fillna(1)

    # Calculate weights
    data["wtC"] = data["pC_n"] / data["pC_d"]

    # Handle potential division by zero in wtC
    data["wtC"].replace([np.inf, -np.inf], 0)  # Replace infinite values with 0

    return obj
