import pandas as pd
import numpy as np

def censor_func(sw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Python translation of the Rcpp censor_func function.
    
    Args:
        sw_data (pd.DataFrame): Input DataFrame with required columns.
    
    Returns:
        pd.DataFrame: Modified DataFrame after applying the censoring function.
    """
    n = len(sw_data)
    
    # Extracting columns
    started0 = sw_data["started0"].to_numpy()
    started1 = sw_data["started1"].to_numpy()
    stop0 = sw_data["stop0"].to_numpy()
    stop1 = sw_data["stop1"].to_numpy()
    eligible0_sw = sw_data["eligible0_sw"].to_numpy()
    eligible1_sw = sw_data["eligible1_sw"].to_numpy()
    delete_ = sw_data["delete"].astype(bool).to_numpy()

    t_first = sw_data["first"].to_numpy()
    t_eligible = sw_data["eligible"].to_numpy()
    t_treatment = sw_data["treatment"].to_numpy()
    t_switch = sw_data["switch"].to_numpy()

    # Initializing state variables
    started0_ = 0
    started1_ = 0
    stop0_ = 0
    stop1_ = 0
    eligible0_sw_ = 0
    eligible1_sw_ = 0

    for i in range(n):
        if t_first[i]:
            started0_ = started1_ = stop0_ = stop1_ = eligible0_sw_ = eligible1_sw_ = 0

        if stop0_ == 1 or stop1_ == 1:
            started0_ = started1_ = stop0_ = stop1_ = eligible0_sw_ = eligible1_sw_ = 0

        if started0_ == 0 and started1_ == 0 and t_eligible[i] == 1:
            if t_treatment[i] == 0:
                started0_ = 1
            elif t_treatment[i] == 1:
                started1_ = 1

        if started0_ == 1 and stop0_ == 0:
            eligible0_sw_ = 1
            eligible1_sw_ = 0
        elif started1_ == 1 and stop1_ == 0:
            eligible0_sw_ = 0
            eligible1_sw_ = 1
        else:
            eligible0_sw_ = eligible1_sw_ = 0

        if t_switch[i] == 1:
            if t_eligible[i] == 1:
                if t_treatment[i] == 1:
                    started1_ = 1
                    stop1_ = 0
                    started0_ = stop0_ = 0
                    eligible1_sw_ = 1
                elif t_treatment[i] == 0:
                    started0_ = 1
                    stop0_ = 0
                    started1_ = stop1_ = 0
                    eligible0_sw_ = 1
            else:
                stop0_ = started0_
                stop1_ = started1_

        if eligible0_sw_ == 0 and eligible1_sw_ == 0:
            delete_[i] = True
        else:
            started0[i] = started0_
            started1[i] = started1_
            stop0[i] = stop0_
            stop1[i] = stop1_
            eligible1_sw[i] = eligible1_sw_
            eligible0_sw[i] = eligible0_sw_
            delete_[i] = False

    # Updating DataFrame with modified values
    sw_data["started0"] = started0
    sw_data["started1"] = started1
    sw_data["stop0"] = stop0
    sw_data["stop1"] = stop1
    sw_data["eligible0_sw"] = eligible0_sw
    sw_data["eligible1_sw"] = eligible1_sw
    sw_data["delete"] = delete_

    return sw_data
