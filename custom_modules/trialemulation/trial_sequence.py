import pandas as pd
import statsmodels.api as sm
import patsy

from custom_modules.trialemulation.data_manipulation import data_manipulation
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

        self.data = data_manipulation(data)

    def set_censor_weight_model(self, censor_event, numerator=None, denominator=None, pool_models="none", model_fitter=None):
        """Set up the censor weight model using logistic regression."""
        
        # Default values if arguments are missing
        if numerator is None:
            numerator = "1"  # Equivalent to `~1` in R
        if denominator is None:
            denominator = "1"  # Equivalent to `~1` in R

        # Validate inputs
        if not isinstance(censor_event, str):
            raise ValueError("censor_event must be a string.")
        if censor_event not in self.data.columns:
            raise ValueError(f"Column '{censor_event}' not found in dataset.")
        
        # Prevent usage of 'time_on_regime' in numerator
        if "time_on_regime" in numerator or "time_on_regime" in denominator:
            raise ValueError("time_on_regime should not be used in numerator or denominator.")
        
        # Validate formulas using patsy
        try:
            patsy.dmatrix(numerator, self.data)
            patsy.dmatrix(denominator, self.data)
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")

        # Ensure model_fitter is provided
        if model_fitter is None:
            model_fitter = stats_glm_logit()

        # Update formulas (R: `1 - censor_event ~ .`)
        numerator = f"1 - {censor_event} {numerator}"
        denominator = f"1 - {censor_event} {denominator}"

        # Store the censor weight specification
        self.censor_weights = {
            "numerator": numerator,
            "denominator": denominator,
            "pool_numerator": pool_models in ["numerator", "both"],
            "pool_denominator": pool_models == "both",
            "model_fitter": model_fitter
        }

        # Call update_outcome_formula (implement separately if needed)
        self.update_outcome_formula()

        return self  # Enable method chaining

    def set_switch_weight_model(self, numerator=None, denominator=None, model_fitter=None):
        """Set up the switch weight model using a provided model fitter."""

        if self.data is None:
            raise ValueError("Please use set_data() to set up the data before setting switch weight models.")

        if model_fitter is None:
            raise ValueError("A model_fitter must be provided (e.g., stats_glm_logit()).")

        # Default formulas if missing (R: `~1` means no covariates)
        if numerator is None:
            numerator = "1"
        if denominator is None:
            denominator = "1"

        # Prevent usage of 'time_on_regime' in formulas
        if "time_on_regime" in numerator or "time_on_regime" in denominator:
            raise ValueError("time_on_regime should not be used in numerator or denominator.")

        # Ensure formulas include `treatment ~ ...`
        numerator = f"treatment ~ {numerator}"
        denominator = f"treatment ~ {denominator}"

        # Validate formulas using patsy.dmatrices() instead of dmatrix()
        try:
            patsy.dmatrices(numerator, self.data)
            patsy.dmatrices(denominator, self.data)
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")

        # Fit numerator and denominator models
        numerator_model = model_fitter.fit_weights_model(self.data, numerator)
        denominator_model = model_fitter.fit_weights_model(self.data, denominator)

        # Store the fitted models
        self.switch_weights = {
            "numerator": numerator_model,
            "denominator": denominator_model,
            "model_fitter": model_fitter
        }

        return self  # Allow method chaining

    def set_outcome_model(self, treatment_var, adjustment_terms, model_fitter=sm.Logit):
        self.outcome_model = {'treatment_var': treatment_var, 'adjustment_terms': adjustment_terms, 'model_fitter': model_fitter}

    def update_outcome_formula(self):
        """Update the outcome formula after setting censor weights."""
        if self.censor_weights is None:
            raise ValueError("Censor weights have not been set.")
        
        # Placeholder: Implement outcome formula update logic
        print("Outcome formula updated based on censor weights.")


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

    def show_switch_weights(self):
        """Display switch weight model details in a formatted way."""
        if self.switch_weights is None:
            print("Switch weights have not been set.")
            return

        # Extract values
        numerator = self.switch_weights["numerator"]["model"].model.formula
        denominator = self.switch_weights["denominator"]["model"].model.formula
        model_fitter = self.switch_weights["model_fitter"]

        # Print formatted output
        print(f"  - Numerator formula: {numerator}")  
        print(f"  - Denominator formula: {denominator}")
        print(f"  - Model fitter type: {type(model_fitter).__name__}")
        print("  - Weight models not fitted. Use calculate_weights()")

    def show_censor_weights(self):
        """Display censor weight model details in a formatted way."""
        if self.censor_weights is None:
            print("Censor weights have not been set.")
            return

        # Extract values
        numerator = self.censor_weights["numerator"]
        denominator = self.censor_weights["denominator"]
        pool_numerator = self.censor_weights["pool_numerator"]
        pool_denominator = self.censor_weights["pool_denominator"]
        model_fitter = self.censor_weights["model_fitter"]

        # Print formatted output
        print(f"  - Numerator formula: {numerator}")
        print(f"  - Denominator formula: {denominator}")

        if pool_numerator and pool_denominator:
            print("  - Both numerator and denominator models are pooled across treatment arms.")
        elif pool_numerator:
            print("  - Numerator model is pooled across treatment arms. Denominator model is not pooled.")
        elif pool_denominator:
            print("  - Denominator model is pooled across treatment arms. Numerator model is not pooled.")
        else:
            print("  - Neither model is pooled.")

        print(f"  - Model fitter type: {type(model_fitter).__name__}")  # Print class name
        print("  - Weight models not fitted. Use calculate_weights()")

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
    def __init__(self, data=None, save_path=None):
        super().__init__('Per-protocol', data)
        self.switch_weights = None

class TrialSequenceAT(TrialSequence):
    def __init__(self, data=None):
        super().__init__('As treated', data)
        self.switch_weights = None