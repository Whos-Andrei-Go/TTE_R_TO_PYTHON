import pandas as pd
import numpy as np

from .censor_func import censor_func

def data_manipulation(data: pd.DataFrame, use_censor: bool = True) -> pd.DataFrame:
    """
    Preprocesses the data for weight calculation and extension.

    Parameters:
    - data: pandas DataFrame containing the data.
    - use_censor: Whether to apply censoring due to treatment switch.

    Returns:
    - Preprocessed DataFrame.
    """
    # Ensure 'id' and 'period' are sorted for correct group operations
    data = data.sort_values(by=["id", "period"]).copy()

    # Remove observations before trial eligibility
    data["after_eligibility"] = data.groupby("id")["period"].transform(lambda x: x >= x[data["eligible"] == 1].min() if (data["eligible"] == 1).any() else False)
    if not data["after_eligibility"].all():
        print("Warning: Observations before trial eligibility were removed")
        data = data[data["after_eligibility"]]

    data.drop(columns=["after_eligibility"], inplace=True)

    # Remove observations after the outcome event
    data["after_event"] = data.groupby("id")["period"].transform(lambda x: x > x[data["outcome"] == 1].min() if (data["outcome"] == 1).any() else False)
    if data["after_event"].any():
        print("Warning: Observations after the outcome occurred were removed")
        data = data[~data["after_event"]]

    data.drop(columns=["after_event"], inplace=True)

    # Calculate event time
    event_data = data.groupby("id").last()[["period", "outcome"]].reset_index()
    event_data["time_of_event"] = np.where(event_data["outcome"] == 1, event_data["period"].astype(float), 9999)

    # Merge event data
    sw_data = data.merge(event_data[["id", "time_of_event"]], on="id", how="left")

    # Define first occurrence in each group
    sw_data["first"] = ~sw_data["id"].duplicated(keep="first")

    # Lagged treatment
    sw_data["am_1"] = sw_data.groupby("id")["treatment"].shift(1, fill_value=0)

    # Initialize cumulative treatment, switching, and regime tracking
    sw_data["cumA"] = 0
    sw_data["switch"] = 0
    sw_data["regime_start"] = sw_data["period"]
    sw_data["time_on_regime"] = 0

    # Update based on treatment changes
    mask_not_first = ~sw_data["first"]
    sw_data.loc[mask_not_first & (sw_data["am_1"] != sw_data["treatment"]), "switch"] = 1
    sw_data.loc[mask_not_first & (sw_data["am_1"] == sw_data["treatment"]), "switch"] = 0

    sw_data.loc[mask_not_first & (sw_data["switch"] == 1), "regime_start"] = sw_data["period"]
    sw_data["regime_start"] = sw_data.groupby("id")["regime_start"].ffill()

    # Compute time on regime
    sw_data["regime_start_shift"] = sw_data.groupby("id")["regime_start"].shift(1)
    sw_data.loc[mask_not_first, "time_on_regime"] = sw_data["period"] - sw_data["regime_start_shift"].astype(float)
    sw_data.drop(columns=["regime_start_shift"], inplace=True)

    # Cumulative treatment assignment
    sw_data["cumA"] = sw_data.groupby("id")["treatment"].cumsum()

    # Apply censoring if required
    if use_censor:
        sw_data["started0"] = np.nan
        sw_data["started1"] = np.nan
        sw_data["stop0"] = np.nan
        sw_data["stop1"] = np.nan
        sw_data["eligible0_sw"] = np.nan
        sw_data["eligible1_sw"] = np.nan
        sw_data["delete"] = np.nan

        sw_data = censor_func(sw_data)  # Assuming censor_func is implemented elsewhere

        # Remove censored observations
        sw_data = sw_data[sw_data["delete"] == False]
        sw_data.drop(columns=["delete", "eligible0_sw", "eligible1_sw", "started0", "started1", "stop0", "stop1"], inplace=True)

    # Define eligibility indicators
    sw_data["eligible0"] = (sw_data["am_1"] == 0).astype(int)
    sw_data["eligible1"] = (sw_data["am_1"] == 1).astype(int)

    return sw_data
