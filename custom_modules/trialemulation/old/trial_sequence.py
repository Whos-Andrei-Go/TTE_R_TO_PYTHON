import pandas as pd
import statsmodels.api as sm
import patsy
import re

from custom_modules.trialemulation.data_manipulation import data_manipulation
from custom_modules.trialemulation.te_expansion import TeExpansion, TeExpansionUnset
from custom_modules.trialemulation.calculate_weights import CalculateWeights
from custom_modules.trialemulation.model_fitting import stats_glm_logit
from custom_modules.trialemulation.te_weights import TeWeightsSpec, TeWeightsUnset, TeWeightsFitted

class OutcomeModel:
    def __init__(self):
        self.stabilised_weights_terms = None
        self.treatment_terms = None
        self.adjustment_terms = None
        self.followup_time_terms = None
        self.trial_period_terms = None
        self.formula = None
        self.adjustment_vars = []

class TrialSequence:
    def __init__(self, estimand, data=None):
        self.estimand = estimand
        self.data = data
        self.censor_weights = TeWeightsUnset()
        self.switch_weights = TeWeightsUnset()
        self.outcome_model = OutcomeModel()
        self.expansion = TeExpansionUnset()
        self.weights_calculator = CalculateWeights(self)

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

        processed_data = data_manipulation(data)

        # Ensure it's still a DataFrame
        if not isinstance(processed_data, pd.DataFrame):
            raise ValueError("data_manipulation did not return a DataFrame!")

        self.data = processed_data  # ✅ Now always assigns processed data correctly

    def set_censor_weight_model(self, censor_event, numerator=None, denominator=None, pool_models="none", model_fitter=None):
        """Set up the censor weight model using logistic regression."""

        if self.data is None:
            raise ValueError("Please use set_data() before setting the censor weight model.")

        # Default formulas if missing (`~1` in R means no covariates)
        numerator = "1" if numerator is None else numerator.lstrip("~").strip()
        denominator = "1" if denominator is None else denominator.lstrip("~").strip()

        # Construct the formulas correctly
        numerator = f"1 - {censor_event} ~ {numerator}"  
        denominator = f"1 - {censor_event} ~ {denominator}"

        # Validate formulas using `patsy`
        try:
            patsy.dmatrices(numerator, self.data)
            patsy.dmatrices(denominator, self.data)
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")

        # Ensure model_fitter is provided
        if model_fitter is None:
            model_fitter = stats_glm_logit()

        # Store the censor weight specification
        self.censor_weights = TeWeightsSpec(
            numerator=numerator,
            denominator=denominator,
            pool_numerator=pool_models in ["numerator", "both"],
            pool_denominator=pool_models == "both",
            model_fitter=model_fitter
        )

        # Call update_outcome_formula
        self.update_outcome_formula()

        return self  # Enable method chaining


    def set_switch_weight_model(self, numerator=None, denominator=None, model_fitter=None):
        """Set up the switch weight model using a provided model fitter."""

        if self.data is None:
            raise ValueError("Please use set_data() before setting switch weight models.")

        if model_fitter is None:
            raise ValueError("A model_fitter must be provided (e.g., stats_glm_logit()).")

        # Default formulas if missing (`~1` in R means no covariates)
        numerator = f"treatment ~ {numerator}" if numerator else "treatment ~ 1"
        denominator = f"treatment ~ {denominator}" if denominator else "treatment ~ 1"

        # Validate formulas using `patsy`
        try:
            patsy.dmatrices(numerator, self.data)
            patsy.dmatrices(denominator, self.data)
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")

        # Store the switch weight specification using `TeWeightsSpec`
        self.switch_weights = TeWeightsSpec(
            numerator=numerator,
            denominator=denominator,
            pool_numerator=False,  # Default behavior
            pool_denominator=False,
            model_fitter=model_fitter
        )

        return self  # Enable method chaining


    def set_outcome_model(self, treatment_var=None, adjustment_terms="1", followup_time_terms="followup_time + I(followup_time**2)", trial_period_terms="trial_period + I(trial_period**2)", model_fitter=None):
        """Set up the outcome model for the trial sequence."""

        # Ensure data is set
        if self.data is None:
            raise ValueError("Use set_data() before calling set_outcome_model().")

        # Default treatment variable if not specified
        if treatment_var is None:
            treatment_var = "treatment"  # Default for general case

        # Default model fitter
        if model_fitter is None:
            model_fitter = stats_glm_logit()

        # Validate formulas using patsy
        try:
            patsy.dmatrices(f"{treatment_var} ~ {adjustment_terms}", self.data)
            patsy.dmatrices(f"{followup_time_terms}", self.data)
            patsy.dmatrices(f"{trial_period_terms}", self.data)
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")

        # Extract variables from formulas
        treatment_vars = patsy.ModelDesc.from_formula(f"{treatment_var} ~ {adjustment_terms}").lhs_termlist
        adjustment_vars = list(patsy.dmatrix(adjustment_terms, self.data, return_type="dataframe").columns)

        # Store the outcome model
        self.outcome_model = {
            "treatment_var": treatment_vars, 
            "adjustment_vars": adjustment_vars,
            "model_fitter": model_fitter,
            "adjustment_terms": adjustment_terms,
            "treatment_terms": treatment_var,
            "followup_time_terms": followup_time_terms,
            "trial_period_terms": trial_period_terms
        }

        # Update outcome formula if needed
        self.update_outcome_formula()

        return self  # Enable method chaining


    def calculate_weights(self, quiet=False):
        """Calculate weights for the trial sequence."""
        if self.data is None:
            raise ValueError("Data must be set before calculating weights.")

        # Create weights calculator if not already present
        if not hasattr(self, "weights_calculator") or self.weights_calculator is None:
            self.weights_calculator = CalculateWeights(self)

        # Call weight calculation logic
        return self.weights_calculator.calculate_weights_trial_seq(quiet)


    # def update_outcome_formula(self):
    #     """Update the outcome formula after setting censor weights."""
    #     if self.censor_weights is None:
    #         raise ValueError("Censor weights have not been set.")
        
    #     # Placeholder: Implement outcome formula update logic
    #     print("Outcome formula updated based on censor weights.")

    def update_outcome_formula(self):
        """Update the outcome formula after setting censor weights."""
        if self.censor_weights is None:
            raise ValueError("Censor weights have not been set.")

        # Ensure stabilised weights are set
        self.outcome_model.stabilised_weights_terms = self.get_stabilised_weights_terms()

        # Create a dictionary of formulas
        formula_list = {
            "intercept": "1",  # Equivalent to ~1 in R
            "treatment_terms": self.outcome_model.treatment_terms.lstrip("~").strip() if self.outcome_model.treatment_terms else "",
            "adjustment_terms": self.outcome_model.adjustment_terms.lstrip("~").strip() if self.outcome_model.adjustment_terms else "",
            "followup_time_terms": self.outcome_model.followup_time_terms.lstrip("~").strip() if self.outcome_model.followup_time_terms else "",
            "trial_period_terms": self.outcome_model.trial_period_terms.lstrip("~").strip() if self.outcome_model.trial_period_terms else "",
            "stabilised_weights_terms": self.outcome_model.stabilised_weights_terms.lstrip("~").strip()
        }

        # Remove empty formulas
        formula_list = {k: v for k, v in formula_list.items() if v}

        # Construct the final formula
        first_term = next(iter(formula_list.values()))  # First formula
        remaining_terms = [v for k, v in list(formula_list.items())[1:]]

        outcome_formula = f"outcome ~ {first_term} + " + " + ".join(remaining_terms) if remaining_terms else f"outcome ~ {first_term}"

        # Update the outcome model's formula
        self.outcome_model.formula = outcome_formula

        # Extract adjustment variables
        adjustment_vars = []
        for term in [self.outcome_model.adjustment_terms, self.outcome_model.stabilised_weights_terms]:
            if term:
                term = term.lstrip("~")  # Ensure no extra ~
                adjustment_vars.extend(patsy.dmatrix(term, self.data, return_type="dataframe").columns)

        self.outcome_model.adjustment_vars = list(set(adjustment_vars))

        return self  # Enable method chaining

        
    def extract_rhs(formula):
        """Extract the right-hand side (RHS) of a formula after ~"""
        if "~" in formula:
            return formula.split("~", 1)[1].strip()  # Get everything after ~
        return formula.strip()  # If no ~, return as is

    def get_stabilised_weights_terms(self):
        """Get the stabilised weights terms based on the current weights."""
        if not isinstance(self, TrialSequence):
            raise ValueError("Object must be of type 'TrialSequence'.")

        stabilised_terms = ["1"]  # Start with "1" (equivalent to ~1 in R)

        # Check for censor weights
        if hasattr(self, 'censor_weights') and not isinstance(self.censor_weights, TeWeightsUnset):
            term = extract_rhs(self.censor_weights.numerator)
            print(f"Extracted Censor Weight Term: {term}")  # Debugging
            if term:
                stabilised_terms.append(term)

        # Check for switch weights
        if hasattr(self, 'switch_weights') and not isinstance(self.switch_weights, TeWeightsUnset):
            term = extract_rhs(self.switch_weights.numerator)
            print(f"Extracted Switch Weight Term: {term}")  # Debugging
            if term:
                stabilised_terms.append(term)

        final_terms = " + ".join(stabilised_terms)
        print(f"Final Stabilised Terms: {final_terms}")  # Debugging

        return final_terms


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

        # Extract values using dot notation
        numerator = self.switch_weights.numerator
        denominator = self.switch_weights.denominator
        model_fitter = self.switch_weights.model_fitter

        print(f"  - Numerator formula: {numerator}")  
        print(f"  - Denominator formula: {denominator}")
        print(f"  - Model fitter type: {type(model_fitter).__name__}")

        if self.switch_weights.fitted:
            print("  - View weight model summaries with `show_weight_models()`")
        else:
            print("  - Weight models not fitted. Use `calculate_weights()`")


    def show_censor_weights(self):
        """Display censor weight model details in a formatted way."""
        if self.censor_weights is None:
            print("Censor weights have not been set.")
            return

        # Extract values using dot notation
        numerator = self.censor_weights.numerator
        denominator = self.censor_weights.denominator
        pool_numerator = self.censor_weights.pool_numerator
        pool_denominator = self.censor_weights.pool_denominator
        model_fitter = self.censor_weights.model_fitter

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

        if self.censor_weights.fitted:
            print("  - View weight model summaries with `show_weight_models()`")
        else:
            print("  - Weight models not fitted. Use `calculate_weights()`")

    def show_weight_models(self):
        """Display summaries of the fitted weight models for censoring and switching."""

        def print_model_section(title, weights):
            """Helper function to print model summaries."""
            if not isinstance(weights, TeWeightsSpec) or not weights.fitted:
                return  # Skip if no fitted models
            
            print(f"\n## {title}\n{'-' * len(title)}")

            for model_name, model_obj in weights.fitted.items():
                print(f"\n[[{model_name}]]")
                if isinstance(model_obj, TeWeightsFitted):
                    model_obj.show()
                else:
                    print(f"Unexpected model type: {type(model_obj)}")

        # Display censor weight models
        print_model_section("Weight Models for Informative Censoring", self.censor_weights)

        # Display switch weight models
        print_model_section("Weight Models for Treatment Switching", self.switch_weights)


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

    def set_outcome_model(self, adjustment_terms="1", followup_time_terms="followup_time + I(followup_time**2)", trial_period_terms="trial_period + I(trial_period**2)", model_fitter=None):
        """Set outcome model for ITT (Intention-to-Treat) trial."""
        return super().set_outcome_model(
            treatment_var="assigned_treatment",
            adjustment_terms=adjustment_terms,
            followup_time_terms=followup_time_terms,
            trial_period_terms=trial_period_terms,
            model_fitter=model_fitter
        )

class TrialSequencePP(TrialSequence):
    def __init__(self, data=None, save_path=None):
        super().__init__('Per-protocol', data)
        self.switch_weights = TeWeightsUnset()

    def set_outcome_model(self, adjustment_terms="1", followup_time_terms="followup_time + I(followup_time**2)", trial_period_terms="trial_period + I(trial_period**2)", model_fitter=None):
        """Set outcome model for PP (Per-Protocol) trial."""
        return super().set_outcome_model(
            treatment_var="assigned_treatment",
            adjustment_terms=adjustment_terms,
            followup_time_terms=followup_time_terms,
            trial_period_terms=trial_period_terms,
            model_fitter=model_fitter
        )

class TrialSequenceAT(TrialSequence):
    def __init__(self, data=None):
        super().__init__('As treated', data)
        self.switch_weights = None

    def set_outcome_model(self, treatment_var="dose", adjustment_terms="1", followup_time_terms="followup_time + I(followup_time**2)", trial_period_terms="trial_period + I(trial_period**2)", model_fitter=None):
        """Set outcome model for AT (As-Treated) trial."""
        return super().set_outcome_model(
            treatment_var=treatment_var,
            adjustment_terms=adjustment_terms,
            followup_time_terms=followup_time_terms,
            trial_period_terms=trial_period_terms,
            model_fitter=model_fitter
        )