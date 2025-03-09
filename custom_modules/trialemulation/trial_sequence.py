from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

# Datastore doesn't work for some reason???
from .te_datastore import TEDatastore, TEDatastoreDataTable
from .te_data import TEData, TEDataUnset, TEOutcomeData
from .data_manipulation import data_manipulation
from .te_weights import TEWeightsSpec, TEWeightsUnset, TEWeightsFitted
from .te_expansion import TEExpansion, TEExpansionUnset
from .te_outcome_model import TEOutcomeModel, TEOutcomeModelUnset
from .censor_func import censor_func
from .te_stats_glm_logit import TEStatsGLMLogit
from .calculate_weights import calculate_weights_trial_seq



@dataclass
class TrialSequence:
    """Base class for trial sequence"""
    data: TEData = field(default_factory=TEDataUnset)
    estimand: str = ""
    expansion: TEExpansion = field(default_factory=TEExpansionUnset)
    outcome_model: TEOutcomeModel = field(default_factory=TEOutcomeModelUnset)
    censor_weights: TEWeightsSpec = field(default_factory=TEWeightsUnset)
    outcome_data: Optional[TEOutcomeData] = None
    switch_weights: Optional[TEWeightsSpec] = None  # Only used for PP

    def show(self):
        """Display trial sequence information"""
        print("Trial Sequence Object")
        print(f"Estimand: {self.estimand}\n")
        print("Data:")
        print(self.data)

        # Only print censor_weights if it's not unset
        if not isinstance(self.censor_weights, TEWeightsUnset):
            print("\nIPW for informative censoring:")
            print(self.censor_weights)

        # Only print switch_weights if it exists and is meaningful
        if self.switch_weights and not isinstance(self.switch_weights, TEWeightsUnset):
            print("\nIPW for treatment switch censoring:")
            print(self.switch_weights)

        if not isinstance(self.outcome_model, TEOutcomeModelUnset):
            print(self.expansion)
            print("\nOutcome model:")
            print(self.outcome_model)

        if hasattr(self.expansion, "datastore") and self.expansion.datastore is not None:
            if hasattr(self.expansion.datastore, "N") and self.expansion.datastore.N > 0:
                print(self.outcome_data)

    def set_data(self, data: pd.DataFrame, 
                 id_col="id", period_col="period", treatment_col="treatment",
                 outcome_col="outcome", eligible_col="eligible"):
        """Sets the trial data for the sequence"""
        required_cols = {id_col, period_col, treatment_col, outcome_col, eligible_col}

        if not required_cols.issubset(data.columns):
            missing_cols = required_cols - set(data.columns)
            raise ValueError(f"Missing columns: {missing_cols}")

        self.data = data.rename(columns={
            id_col: "id",
            period_col: "period",
            treatment_col: "treatment",
            outcome_col: "outcome",
            eligible_col: "eligible"
        })

        self.data = data_manipulation(self.data)

        return self  # Returning `self` allows method chaining

    def set_censor_weight_model(self, censor_event: str,
                                numerator: Optional[str] = None, denominator: Optional[str] = None,
                                pool_models: str = "none", model_fitter=None):
        """Sets the censoring weight model"""

        # Validate inputs
        if numerator is None:
            numerator = "1"
        if denominator is None:
            denominator = "1"

        if censor_event not in self.data.columns:
            raise ValueError(f"{censor_event} not found in dataset")

        if pool_models not in ["none", "both", "numerator"]:
            raise ValueError("pool_models must be one of: 'none', 'both', 'numerator'")

        # Remove leading tilde if present
        numerator = numerator.lstrip('~').strip()
        denominator = denominator.lstrip('~').strip()

        # Modify formulas
        numerator_formula = f"(1 - {censor_event}) ~ {numerator}"
        denominator_formula = f"(1 - {censor_event}) ~ {denominator}"

        # Ensure model_fitter is a valid model fitter instance
        if model_fitter is None:
            model_fitter = TEStatsGLMLogit(os.path.join(PP_PATH, "censor_models"))

        # Store the censoring weights model
        self.censor_weights = TEWeightsSpec(
            numerator=numerator_formula,
            denominator=denominator_formula,
            pool_numerator=(pool_models in ["numerator", "both"]),
            pool_denominator=(pool_models == "both"),
            model_fitter=model_fitter
        )

        # Update outcome formula if an outcome model is set
        if hasattr(self, "outcome_model") and not isinstance(self.outcome_model, TEOutcomeModelUnset):
            self.update_outcome_formula()

        return self

    def set_switch_weight_model(self, numerator: str, denominator: str, model_fitter, 
                            eligible_wts_0=None, eligible_wts_1=None, 
                            pool_numerator=None, pool_denominator=None):
        """Sets the switching weight model without requiring an outcome model"""

        if isinstance(self, TrialSequenceITT):
            raise ValueError("Switching weights are not supported for intention-to-treat analyses")

        if isinstance(self.data, TEDataUnset):
            raise ValueError("Please use set_data() before setting switch weight models")

        # Validate eligibility weight columns
        data_cols = self.data.columns
        if eligible_wts_0 and eligible_wts_0 not in data_cols:
            raise ValueError(f"Column '{eligible_wts_0}' not found in dataset")
        if eligible_wts_1 and eligible_wts_1 not in data_cols:
            raise ValueError(f"Column '{eligible_wts_1}' not found in dataset")

        numerator_formula = f"treatment ~ {numerator}"
        denominator_formula = f"treatment ~ {denominator}"

        # Store the switch weights model
        self.switch_weights = TEWeightsSpec(
            numerator=numerator_formula,
            denominator=denominator_formula,
            model_fitter=model_fitter,
            pool_numerator=pool_numerator,
            pool_denominator=pool_denominator
        )

        # **Only update the outcome formula if an outcome model exists**
        if hasattr(self, "outcome_model") and not isinstance(self.outcome_model, TEOutcomeModelUnset):
            self.update_outcome_formula()

        return self

    def set_outcome_model(self, treatment_var="0", adjustment_terms="",  # Change default to empty string
                        followup_time_terms="followup_time + np.power(followup_time, 2)",
                        trial_period_terms="trial_period + np.power(trial_period, 2)",
                        model_fitter=None):
        if self.data is None:
            raise ValueError("Use set_data() before set_outcome_model()")

        # Create formulas
        formulas = {
            "treatment": treatment_var,
            "adjustment": adjustment_terms,
            "followup": followup_time_terms,
            "period": trial_period_terms,
            "stabilised": self.get_stabilised_weights_terms()  # Ensure this returns variable names
        }

        # Collect adjustment variables
        # Split by '+' and strip whitespace, then filter out any non-variable names
        adjustment_vars = set(var.strip() for var in formulas["adjustment"].split("+")) | \
                        set(var.strip() for var in formulas["stabilised"].split("+"))

        # Validate variables exist in data
        missing_vars = [var for var in adjustment_vars if var not in self.data.columns and var != '']
        if missing_vars:
            raise ValueError(f"Missing variables in adjustment terms: {missing_vars}")

        self.outcome_model = {
            "treatment_var": formulas["treatment"],
            "adjustment_vars": adjustment_vars,
            "model_fitter": model_fitter or LogisticRegression(),
            "adjustment_terms": formulas["adjustment"],
            "treatment_terms": formulas["treatment"],
            "followup_time_terms": formulas["followup"],
            "trial_period_terms": formulas["period"],
            "stabilised_weights_terms": formulas["stabilised"]
        }

        self.update_outcome_formula()

    def get_stabilised_weights_terms(self):
        stabilised_terms = []

        if hasattr(self, "censor_weights") and self.censor_weights:
            # Assuming censor_weights has attributes for numerator and denominator
            stabilised_terms.append(self.censor_weights.numerator)
            stabilised_terms.append(self.censor_weights.denominator)

        if hasattr(self, "switch_weights") and self.switch_weights:
            # Assuming switch_weights has attributes for numerator and denominator
            stabilised_terms.append(self.switch_weights.numerator)
            stabilised_terms.append(self.switch_weights.denominator)

        # Join the terms with ' + ' and return
        return " + ".join(stabilised_terms)

    def update_outcome_formula(self):
        if isinstance(self.outcome_model, TEOutcomeModelUnset) or not self.outcome_model:
            raise ValueError("Outcome model is not set. Use set_outcome_model() before proceeding.")

        terms = [
            "1",
            self.outcome_model.treatment_terms or "",
            self.outcome_model.adjustment_terms or "",
            self.outcome_model.followup_time_terms or "",
            self.outcome_model.trial_period_terms or "",
            self.outcome_model.stabilised_weights_terms or ""
        ]

        # Construct final formula
        outcome_formula = " + ".join(filter(None, terms))
        self.outcome_model.formula = f"outcome ~ {outcome_formula}"

        # Update adjustment vars
        self.outcome_model.adjustment_vars = set(
            self.outcome_model.adjustment_terms.split(" + ")
        ) | set(self.outcome_model.stabilised_weights_terms.split(" + "))

        return self

    def calculate_weights(self, quiet: bool = False):
        """Calculates weights for the trial sequence based on its type."""

        # Check if censor weights are set
        use_censor_weights = not isinstance(self.censor_weights, TEWeightsUnset)

        # Determine trial type and apply correct weight calculations
        if isinstance(self, TrialSequenceITT):
            return calculate_weights_trial_seq(self, quiet, False, use_censor_weights)

        elif isinstance(self, TrialSequenceAT):
            if isinstance(self.switch_weights, TEWeightsUnset):
                raise ValueError("Switch weight models are not specified. Use set_switch_weight_model()")
            return calculate_weights_trial_seq(self, quiet, True, use_censor_weights)

        elif isinstance(self, TrialSequencePP):
            if isinstance(self.switch_weights, TEWeightsUnset):
                raise ValueError("Switch weight models are not specified. Use set_switch_weight_model()")
            return calculate_weights_trial_seq(self, quiet, True, use_censor_weights)

        else:
            raise TypeError("Unknown trial sequence type.")

    def show_weight_models(self):
        """Show Weight Model Summaries"""
        if self.censor_weights is not None:
            if hasattr(self.censor_weights, 'fitted') and isinstance(self.censor_weights.fitted, dict):
                print("Weight Models for Informative Censoring")
                for name, model in self.censor_weights.fitted.items():
                    print(f"[[{name}]]")
                    model.show()  # Assuming model has a show method

        if self.switch_weights is not None:
            if hasattr(self.switch_weights, 'fitted') and isinstance(self.switch_weights.fitted, dict):
                print("Weight Models for Treatment Switching")
                for name, model in self.switch_weights.fitted.items():
                    print(f"[[{name}]]")
                    model.show()  # Assuming model has a show method

    def set_expansion_options(self, output, chunk_size=0, first_period=0, last_period=float('inf'), censor_at_switch=None):
        if not isinstance(output, TEDatastore):
            raise TypeError("Output must be of type te_datastore")
        if not isinstance(chunk_size, int) or chunk_size < 0:
            raise ValueError("chunk_size must be a non-negative integer")
        if first_period != 0 and not isinstance(first_period, int):
            raise ValueError("first_period must be an integer")
        if last_period != float('inf') and not isinstance(last_period, int):
            raise ValueError("last_period must be an integer")

        self.expansion = TEExpansion(
            chunk_size=chunk_size,
            datastore=output,
            first_period=first_period,
            last_period=last_period,
            censor_at_switch=censor_at_switch
        )
        return self

    def predict(self, newdata, predict_times, conf_int=True, samples=100, type='survival'):
        # Placeholder for the actual prediction logic
        # This function should return a DataFrame with the necessary prediction results
        followup_time = np.arange(0, 11)  # Follow-up times from 0 to 10
        survival_diff = np.random.rand(len(followup_time))  # Random survival differences
        lower_bound = survival_diff - 0.1  # 2.5% lower bound
        upper_bound = survival_diff + 0.1  # 97.5% upper bound

        return pd.DataFrame({
            'followup_time': followup_time,
            'survival_diff': survival_diff,
            '2.5%': lower_bound,
            '97.5%': upper_bound
        })

def trial_sequence(estimand: str = "ITT") -> TrialSequence:
    """Factory function to create a trial sequence with the correct estimand"""
    trial = TrialSequence(estimand=estimand)
    
    if estimand == "PP":
        trial.switch_weights = TEWeightsUnset()  # Add switch weights for per-protocol
    
    return trial


@dataclass
class TrialSequencePP(TrialSequence):
    """Per-Protocol Trial Sequence"""
    estimand: str = "Per-protocol"
    switch_weights: TEWeightsSpec = field(default_factory=TEWeightsUnset)

    def set_expansion_options(self, output, chunk_size, first_period=0, last_period=float('inf')):
        return super().set_expansion_options(output, chunk_size, first_period, last_period, censor_at_switch=True)


@dataclass
class TrialSequenceITT(TrialSequence):
    """Intention-To-Treat Trial Sequence"""
    estimand: str = "Intention-to-treat"

    def set_expansion_options(self, output, chunk_size=0, first_period=0, last_period=float('inf')):
        return super().set_expansion_options(output, chunk_size, first_period, last_period, censor_at_switch=False)


@dataclass
class TrialSequenceAT(TrialSequence):
    """As-Treated Trial Sequence"""
    estimand: str = "As treated"
    switch_weights: TEWeightsSpec = field(default_factory=TEWeightsUnset)


def trial_sequence(estimand: str, **kwargs) -> TrialSequence:
    """Factory function for creating trial sequence objects"""
    estimand_classes = {
        "ITT": TrialSequenceITT,
        "PP": TrialSequencePP,
        "AT": TrialSequenceAT
    }

    if estimand in estimand_classes:
        return estimand_classes[estimand](**kwargs)
    else:
        raise ValueError(f"{estimand} does not extend class TrialSequence")