class TEOutcomeFitted:
    """
    Fitted Outcome Model Object

    Attributes:
    - model (dict): Contains fitted model objects.
    - summary (dict): Contains dataframes for model summaries similar to `broom::tidy()` and `glance()`.
    """

    def __init__(self, model=None, summary=None):
        self.model = model if model else {}
        self.summary = summary if summary else {}

    def show(self):
        """Prints the model summary."""
        if self.summary:
            print("Model Summary:\n")
            if "tidy" in self.summary:
                print(self.summary["tidy"].round(2).to_string(index=False))  # Mimicking `print.data.frame`
            print("\n")
            if "glance" in self.summary:
                print(self.summary["glance"].round(3).to_string(index=False))
        else:
            print("Use fit_msm() to fit the outcome model")


class TEOutcomeModel:
    """
    Outcome Model Object

    Attributes:
    - formula (str): Formula for model fitting.
    - adjustment_vars (list): Adjustment variables.
    - treatment_var (str): Variable used for treatment.
    - adjustment_terms (str): User-specified terms for inclusion in the outcome model.
    - treatment_terms (str): Terms defining the treatment effect estimation.
    - followup_time_terms (str): Terms for modeling follow-up time in an emulated trial.
    - trial_period_terms (str): Terms for modeling the start time ("trial_period") of an emulated trial.
    - stabilised_weights_terms (str): Adjustment terms from numerator models of stabilized weights.
    - model_fitter (object): Model fitting object (could be an instance of `TEModelFitter`).
    - fitted (TEOutcomeFitted): Stores the fitted model objects.
    """

    def __init__(
        self, formula, adjustment_vars, treatment_var, adjustment_terms,
        treatment_terms, followup_time_terms, trial_period_terms, stabilised_weights_terms,
        model_fitter, fitted=None
    ):
        self.formula = formula
        self.adjustment_vars = adjustment_vars
        self.treatment_var = treatment_var
        self.adjustment_terms = adjustment_terms
        self.treatment_terms = treatment_terms
        self.followup_time_terms = followup_time_terms
        self.trial_period_terms = trial_period_terms
        self.stabilised_weights_terms = stabilised_weights_terms
        self.model_fitter = model_fitter
        self.fitted = fitted if fitted else TEOutcomeFitted()

    def show(self):
        """Prints the outcome model details and the fitted model summary."""
        print(f"- Formula: {self.formula}")
        print(f"- Treatment variable: {self.treatment_var}")
        print(f"- Adjustment variables: {', '.join(self.adjustment_vars)}")
        print(f"- Model fitter type: {type(self.model_fitter).__name__}\n")

        self.fitted.show()


class TEOutcomeModelUnset(TEOutcomeModel):
    """
    Class for an unset outcome model.
    """

    def __init__(self):
        super().__init__(None, [], None, None, None, None, None, None, None, None)

    def show(self):
        print("- Outcome model not specified. Use set_outcome_model()")


# Example Usage:
# model_fitted = TEOutcomeFitted(model={"glm": some_glm_model}, summary={"tidy": tidy_df, "glance": glance_df})
# outcome_model = TEOutcomeModel("y ~ x + z", ["x", "z"], "treatment", "...", "...", "...", "...", "...", some_fitter, model_fitted)
# outcome_model.show()
