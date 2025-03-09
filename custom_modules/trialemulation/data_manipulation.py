import pandas as pd
import numpy as np

def data_manipulation(data: pd.DataFrame, use_censor: bool = True) -> pd.DataFrame:
    """Preprocess data for weight calculation and treatment extension."""
    
    # Ensure required columns exist
    required_columns = {"id", "period", "treatment", "outcome", "eligible"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove observations before eligibility
    eligible_min_period = data.loc[data["eligible"] == 1, "period"].min()
    data["after_eligibility"] = data["period"] >= eligible_min_period if not pd.isna(eligible_min_period) else True

    if data["after_eligibility"].eq(False).any():
        print("Warning: Observations before trial eligibility were removed.")
    data = data[data["after_eligibility"]].drop(columns=["after_eligibility"])

    # Remove observations after the outcome event
    data["after_event"] = data.groupby("id")["period"].transform(
        lambda x: x > x[data["outcome"] == 1].min() if (data["outcome"] == 1).any() else False
    )
    if data["after_event"].eq(True).any():
        print("Warning: Observations after the outcome occurred were removed.")
    data = data[~data["after_event"]].drop(columns=["after_event"])

    # Calculate time_of_event (9999 if no event occurred)
    data["time_of_event"] = data.groupby("id")["outcome"].transform(
        lambda x: x.idxmax() if x.max() == 1 else 9999
    )

    # Compute treatment switch tracking
    data["am_1"] = data.groupby("id")["treatment"].shift(fill_value=0)  # Lagged treatment
    data["switch"] = (data["am_1"] != data["treatment"]).astype(int)

    # Compute regime start (reset when switching occurs)
    data["regime_start"] = np.where(data["switch"] == 1, data["period"], np.nan)
    data["regime_start"] = data.groupby("id")["regime_start"].ffill()
    data["regime_start"] = data["regime_start"].fillna(data["period"])  # ✅ FIXED

    # Compute time_on_regime (reset at switches)
    data["time_on_regime"] = data["period"] - data.groupby("id")["regime_start"].shift(fill_value=data["period"].min())

    # Compute cumulative treatment duration
    data["cumA"] = data.groupby("id")["treatment"].cumsum()

    # Apply censoring if required
    if use_censor:
        # Placeholder for censoring logic, similar to censor_func in R
        data["delete"] = False  # Modify based on actual censoring conditions
        data = data[~data["delete"]].drop(columns=["delete"])

    # Define eligibility for different treatment paths
    data["eligible0"] = (data["am_1"] == 0).astype(int)
    data["eligible1"] = (data["am_1"] == 1).astype(int)

    return data
