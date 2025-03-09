import pandas as pd
import numpy as np
from typing import Optional, Union, List
import os

def assert_directory_exists(directory):
    """Check if a directory exists, raise an error if it does not."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

def data_extension(data: pd.DataFrame,
                  keeplist: List[str],
                  outcome_cov: Optional[str] = None,
                  first_period: Optional[int] = None,
                  last_period: Optional[int] = None,
                  censor_at_switch: bool = False,
                  where_var: Optional[str] = None,
                  data_dir: str = '',
                  separate_files: bool = False,
                  chunk_size: int = 200) -> dict:
    """
    Expands the longitudinal data into a sequence of trials.

    Parameters:
    - data: A DataFrame to be expanded.
    - keeplist: A list containing names of variables used in the final model.
    - outcome_cov: A formula for covariate adjustment of the outcome model.
    - first_period: First period value to start expanding about.
    - last_period: Last period value to expand about.
    - censor_at_switch: Use censoring for per-protocol analysis.
    - where_var: Variables used in where conditions for subsetting the data.
    - data_dir: Directory to save data.
    - separate_files: Save expanded data in separate CSV files for each trial.
    - chunk_size: Number of ids to expand in each chunk.

    Returns:
    A dictionary containing expanded data and metadata.
    """
    if separate_files:
        assert_directory_exists(data_dir)

    if first_period is None:
        first_period = data['period'].min()
    if last_period is None:
        last_period = data['period'].max()

    outcomeCov_var = outcome_cov.split(' + ') if outcome_cov else []

    if separate_files:
        all_ids = data['id'].unique()
        ids_split = np.array_split(all_ids, np.ceil(len(all_ids) / chunk_size))
        N = 0
        for ids in ids_split:
            switch_data = expand(
                sw_data=data[data['id'].isin(ids)],
                outcomeCov_var=outcomeCov_var,
                where_var=where_var,
                use_censor=censor_at_switch,
                minperiod=first_period,
                maxperiod=last_period,
                keeplist=keeplist
            )
            N += len(switch_data)
            for p in switch_data['trial_period'].unique():
                file_p = os.path.join(data_dir, f"trial_{p}.csv")
                switch_data[switch_data['trial_period'] == p].to_csv(file_p, mode='a', index=False)

        files = [os.path.join(data_dir, f"trial_{p}.csv") for p in range(first_period, last_period + 1)]
        return {
            'data': [file for file in files if os.path.exists(file)],
            'min_period': first_period,
            'max_period': last_period,
            'N': N,
            'data_template': switch_data.iloc[0:0]  # Empty DataFrame with the same structure
        }
    else:
        switch_data = expand(
            sw_data=data,
            outcomeCov_var=outcomeCov_var,
            where_var=where_var,
            use_censor=censor_at_switch,
            minperiod=first_period,
            maxperiod=last_period,
            keeplist=keeplist
        )
        return {
            'data': switch_data,
            'min_period': first_period,
            'max_period': last_period,
            'N': len(switch_data),
            'data_template': switch_data.iloc[0:0]  # Empty DataFrame with the same structure
        }

def expand(sw_data: pd.DataFrame,
           outcomeCov_var: List[str],
           where_var: Optional[str],
           use_censor: bool,
           minperiod: int,
           maxperiod: int,
           keeplist: List[str]) -> pd.DataFrame:
    """
    Expand the data based on the specified parameters.

    Parameters:
    - sw_data: DataFrame to expand.
    - outcomeCov_var: A list of individual baseline variables used in the final model.
    - where_var: Variables used in where conditions for subsetting the data.
    - use_censor: Use censoring for per-protocol analysis.
    - maxperiod: Maximum period.
    - minperiod: Minimum period.
    - keeplist: A list containing names of variables used in the final model.

        Returns:
    A DataFrame containing the expanded data.
    """
    # Create a temporary DataFrame for expansion
    temp_data = sw_data.copy()
    temp_data['expand'] = 0.0
    temp_data['wtprod'] = 1.0
    temp_data['elgcount'] = 0.0
    temp_data['treat'] = 0.0
    temp_data['dosesum'] = 0.0

    # Mark eligible entries for expansion
    temp_data.loc[
        (sw_data['eligible'] == 1) & (sw_data['period'].between(minperiod, maxperiod)),
        'expand'
    ] = 1

    # Initialize weights and treatment values
    sw_data['weight0'] = sw_data['wt'].cumprod()
    temp_data['wtprod'] = sw_data['weight0']
    temp_data['treat'] = sw_data['treatment']
    temp_data['dosesum'] = sw_data['cumA']
    temp_data['elgcount'] = sw_data['eligible']

    # Handle initial treatment values
    temp_data.loc[sw_data['eligible'] == 1, 'init'] = sw_data.loc[sw_data['eligible'] == 1, 'treatment']
    temp_data['init_shift'] = temp_data['treatment'].shift()
    temp_data.loc[sw_data['eligible'] == 0, 'init'] = temp_data['init_shift']

    # Copy outcome covariates if provided
    if outcomeCov_var:
        for var in outcomeCov_var:
            temp_data[var] = sw_data[var]

    # Copy where variables if provided
    if where_var:
        temp_data[where_var] = sw_data[where_var]

    # Create expanded index
    expand_index = np.repeat(np.arange(len(sw_data)), sw_data['period'] + 1)

    # Create the switch data DataFrame
    switch_data = pd.DataFrame({
        'id': sw_data.loc[expand_index, 'id'],
        'period_new': sw_data.loc[expand_index, 'period'],
        'cumA_new': sw_data.loc[expand_index, 'cumA'],
        'treatment_new': sw_data.loc[expand_index, 'treatment'],
        'switch_new': np.where(use_censor, sw_data.loc[expand_index, 'switch'], 0),
        'outcome_new': sw_data.loc[expand_index, 'outcome'],
        'time_of_event': sw_data.loc[expand_index, 'time_of_event'],
        'weight0': sw_data.loc[expand_index, 'weight0'],
        'trial_period': np.arange(len(sw_data['period']) + 1)
    })

    # Merge with temp_data to get additional variables
    switch_data = switch_data.merge(temp_data, on=['id', 'trial_period'], how='left')

    # Calculate follow-up time
    switch_data['followup_time'] = switch_data['period_new'] - switch_data['trial_period']

    # Handle dose calculations if included in keeplist
    if 'dose' in keeplist:
        switch_data['dose'] = switch_data['cumA_new'] - switch_data['dosesum'] + switch_data['treat']

    # Set cases based on conditions
    switch_data['case'] = 0
    if not use_censor:
        switch_data.loc[(switch_data['time_of_event'] == switch_data['period_new']) & (switch_data['outcome_new'] == 1), 'case'] = 1
    else:
        switch_data.loc[switch_data['switch_new'] == 1, 'case'] = np.nan
        switch_data.loc[(switch_data['switch_new'] == 0) & (switch_data['time_of_event'] == switch_data['period_new']) & (switch_data['outcome_new'] == 1), 'case'] = 1

    # Rename columns
    switch_data.rename(columns={'case': 'outcome', 'init': 'assigned_treatment', 'treatment_new': 'treatment'}, inplace=True)

    # Filter based on expand flag
    switch_data = switch_data[switch_data['expand'] == 1]

    # Return only the specified columns
    return switch_data[keeplist]

def expand_until_switch(s: np.ndarray, n: int) -> np.ndarray:
    """
    Check if patients have switched treatment in eligible trials and set `expand = 0`.

    Parameters:
    - s: A numeric vector where `1` indicates a treatment switch in that period.
    - n: Length of s.

    Returns:
    A vector of indicator values up until the first switch.
    """
    first_switch = np.where(s == 1)[0]
    if first_switch.size > 0:
        first_switch_index = first_switch[0]
        return np.concatenate([np.ones(first_switch_index), np.zeros(n - first_switch_index)])
    else:
        return np.ones(n)