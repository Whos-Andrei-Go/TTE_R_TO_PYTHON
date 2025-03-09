import pandas as pd
import statsmodels.api as sm
from custom_modules.trialemulation.te_expansion import TeExpansion, TeExpansionUnset

class TrialSequence:
    def __init__(self, estimand, data=None):
        self.estimand = estimand
        self.data = data
        self.censor_weights = None
        self.switch_weights = None
        self.outcome_model = None
        self.expansion = TeExpansionUnset() 

    def set_expansion_options(self, chunk_size=0, datastore=None, censor_at_switch=False, first_period=0, last_period=float("inf")):
        self.expansion = TeExpansion(chunk_size, datastore, censor_at_switch, first_period, last_period)

    def set_data(self, data, id_col='id', period_col='period', treatment_col='treatment', outcome_col='outcome', eligible_col='eligible'):
        self.data = data.rename(columns={
            id_col: 'id',
            period_col: 'period',
            treatment_col: 'treatment',
            outcome_col: 'outcome',
            eligible_col: 'eligible'
        })
        self.data["time_on_regime"] = self.data.groupby("id")["treatment"].cumsum()

    def set_censor_weight_model(self, censor_event, numerator_formula, denominator_formula):
        self.censor_weights = {'censor_event': censor_event, 'numerator': numerator_formula, 'denominator': denominator_formula}

    def set_switch_weight_model(self, numerator_formula, denominator_formula):
        self.switch_weights = {'numerator': numerator_formula, 'denominator': denominator_formula}

    def set_outcome_model(self, treatment_var, adjustment_terms, model_fitter=sm.Logit):
        self.outcome_model = {'treatment_var': treatment_var, 'adjustment_terms': adjustment_terms, 'model_fitter': model_fitter}

    def show(self):
        print(f"Trial Sequence Object\nEstimand: {self.estimand}\n")

        print("Data:")
        if hasattr(self.data, "data"):
            print(self.data.data.head())  # Print first few rows if it's a DataFrame
        else:
            print(self.data)

        print("\nIPW for informative censoring:")
        print(self.censor_weights if self.censor_weights else "Not set")

        print("\nOutcome model:")
        print(self.outcome_model if self.outcome_model else "Not set")

        print("\nExpansion options:")
        print(self.expansion)  # This will now call `TeExpansion.__str__()` or `TeExpansionUnset.__str__()`

    def __str__(self):
        return (
            f"Trial Sequence Object\n"
            f"Estimand: {self.estimand}\n"
            f"\nData:\n{self.data if isinstance(self.data, pd.DataFrame) else 'No data set.'}"
            f"\n\nExpansion Options:\n{self.expansion}"
            f"\n\nOutcome Model:\n{self.outcome_model}"
            f"\n\nCensor Weights:\n{self.censor_weights}"
            f"\n\nOutcome Data:\n{self.outcome_data}"
        )

    def __repr__(self):
        return self.__str__()  # Makes sure `repr(trial)` also prints a readable format



class TrialSequenceITT(TrialSequence):
    def __init__(self, data=None):
        super().__init__('Intention-to-treat', data)

class TrialSequencePP(TrialSequence):
    def __init__(self, data=None):
        super().__init__('Per-protocol', data)
        self.switch_weights = None

class TrialSequenceAT(TrialSequence):
    def __init__(self, data=None):
        super().__init__('As treated', data)
        self.switch_weights = None