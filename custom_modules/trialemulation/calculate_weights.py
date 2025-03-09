from custom_modules.trialemulation.te_weights import TeWeightsSpec, TeWeightsUnset, TeWeightsFitted

import pandas as pd
import numpy as np

def calculate_weights_trial_seq(obj, switch_weights=True, censor_weights=True):
    """Calculate weights for a trial sequence"""
    obj.data["wt"] = 1

    if switch_weights and not isinstance(obj.switch_weights, TeWeightsUnset):
        obj = calculate_switch_weights(obj)
        obj.data["wt"] *= obj.data["wtS"]

    if censor_weights and not isinstance(obj.censor_weights, TeWeightsUnset):
        obj = calculate_censor_weights(obj)
        obj.data["wt"] *= obj.data["wtC"]

    return obj


def calculate_switch_weights(obj):
    """Calculate treatment switching weights"""
    data = obj.data

    if "eligible_wts_1" in data.columns:
        data_1_expr = (data["am_1"] == 1) & (data["eligible_wts_1"] == 1)
    else:
        data_1_expr = data["am_1"] == 1

    model_1_index = data.index[data_1_expr]

    obj.switch_weights.fitted["n1"] = fit_weights_model(
        data.loc[model_1_index], obj.switch_weights.numerator,
        "P(treatment = 1 | previous treatment = 1) for numerator"
    )
    data.loc[model_1_index, "p_n"] = obj.switch_weights.fitted["n1"].fitted

    obj.switch_weights.fitted["d1"] = fit_weights_model(
        data.loc[model_1_index], obj.switch_weights.denominator,
        "P(treatment = 1 | previous treatment = 1) for denominator"
    )
    data.loc[model_1_index, "p_d"] = obj.switch_weights.fitted["d1"].fitted

    # Repeat for Treatment = 0
    if "eligible_wts_1" in data.columns:
        data_0_expr = (data["am_1"] == 0) & (data["eligible_wts_1"] == 0)
    else:
        data_0_expr = data["am_1"] == 0

    model_0_index = data.index[data_0_expr]

    obj.switch_weights.fitted["n0"] = fit_weights_model(
        data.loc[model_0_index], obj.switch_weights.numerator,
        "P(treatment = 1 | previous treatment = 0) for numerator"
    )
    data.loc[model_0_index, "p_n"] = obj.switch_weights.fitted["n0"].fitted

    obj.switch_weights.fitted["d0"] = fit_weights_model(
        data.loc[model_0_index], obj.switch_weights.denominator,
        "P(treatment = 1 | previous treatment = 0) for denominator"
    )
    data.loc[model_0_index, "p_d"] = obj.switch_weights.fitted["d0"].fitted

    # Combine Weights
    data["wtS"] = 1.0
    mask_0 = (data["treatment"] == 0)
    mask_1 = (data["treatment"] == 1)
    
    data.loc[mask_0, "wtS"] = (1.0 - data["p_n"]) / (1.0 - data["p_d"])
    data.loc[mask_1, "wtS"] = data["p_n"] / data["p_d"]
    data["wtS"].fillna(1, inplace=True)

    return obj



def calculate_censor_weights(obj):
    """Calculate censoring weights"""
    data = obj.data

    if obj.censor_weights.pool_numerator:
        obj.censor_weights.fitted["n"] = fit_weights_model(
            data, obj.censor_weights.numerator, "P(censor_event = 0 | X) for numerator"
        )
        data["pC_n"] = obj.censor_weights.fitted["n"].fitted
    else:
        mask_0 = data["am_1"] == 0
        mask_1 = data["am_1"] == 1

        obj.censor_weights.fitted["n0"] = fit_weights_model(
            data[mask_0], obj.censor_weights.numerator,
            "P(censor_event = 0 | X, previous treatment = 0) for numerator"
        )
        data.loc[mask_0, "pC_n"] = obj.censor_weights.fitted["n0"].fitted

        obj.censor_weights.fitted["n1"] = fit_weights_model(
            data[mask_1], obj.censor_weights.numerator,
            "P(censor_event = 0 | X, previous treatment = 1) for numerator"
        )
        data.loc[mask_1, "pC_n"] = obj.censor_weights.fitted["n1"].fitted

    data["pC_d"].fillna(1, inplace=True)
    data["pC_n"].fillna(1, inplace=True)
    data["wtC"] = data["pC_n"] / data["pC_d"]

    return obj
