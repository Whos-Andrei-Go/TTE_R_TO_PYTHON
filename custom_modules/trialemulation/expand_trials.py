import pandas as pd
from typing import List, Optional

from .te_expansion import TEExpansion, TEExpansionUnset
from .te_outcome_model import TEOutcomeModel, TEOutcomeModelUnset
from .data_extension import expand

class ExpandTrials:
    def __init__(self, data: pd.DataFrame, expansion, outcome_model):
        self.data = data  # Assuming data is a pandas DataFrame
        self.expansion = expansion  # Expansion object
        self.outcome_model = outcome_model  # Outcome model object

    def expand_trials(self) -> 'ExpandTrials':
        # Avoid NOTE for data.table
        eligible = None

        # Use the observed min/max periods if they are within the specified limits
        first_period = max(self.expansion.first_period, self.data.loc[self.data['eligible'] == 1, 'period'].min())
        last_period = min(self.expansion.last_period, self.data.loc[self.data['eligible'] == 1, 'period'].max())
        chunk_size = self.expansion.chunk_size
        censor_at_switch = self.expansion.censor_at_switch

        outcome_adj_vars = self.outcome_model.adjustment_vars  # This is a set
        keeplist = list(set([
            "id", "trial_period", "followup_time", "outcome", "weight", "treatment"
        ] + list(outcome_adj_vars) + [self.outcome_model.treatment_var]))  # Convert set to list

        # Initialize weights if not present
        if 'wt' not in self.data.columns:
            self.data['wt'] = 1

        all_ids = self.data['id'].unique()
        if chunk_size == 0:
            ids_split = [all_ids]
        else:
            ids_split = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]

        for ids in ids_split:
            switch_data = expand(
                sw_data=self.data[self.data['id'].isin(ids)],
                outcomeCov_var=outcome_adj_vars,
                where_var=None,
                use_censor=censor_at_switch,
                minperiod=first_period,
                maxperiod=last_period,
                keeplist=keeplist
            )
            self.expansion.datastore = self.save_expanded_data(self.expansion.datastore, switch_data)

        return self

    def save_expanded_data(self, datastore, switch_data: pd.DataFrame):
        """Save the expanded data to the datastore."""
        datastore = pd.concat([datastore, switch_data], ignore_index=True)
        return datastore