import pandas as pd

class TeWeightsFitted:
    """Represents a fitted weight model with summary information."""

    def __init__(self, label, summary, fitted_model, save_path=None):
        self.label = label  # Model description
        self.summary = summary  # Model summary (tidy + glance)
        self.fitted_model = fitted_model  # Actual model object
        self.save_path = save_path  # File path to saved model (if exists)

    def show(self):
        """Prints the summary of the fitted weight model in an R-style format."""
        print(f"Model: {self.label}\n")

        if isinstance(self.summary, dict) and "summary" in self.summary:
            summary_df = self.summary["summary"]
            if isinstance(summary_df, pd.DataFrame):
                print(summary_df.to_string(index=False))  # Print coefficients table
                print("")

        if hasattr(self.fitted_model, "aic"):  # If statsmodels GLM object
            print(f"null.deviance df.null logLik    AIC      BIC      deviance df.residual nobs")
            print(f"{self.fitted_model.null_deviance:<12.4f} {self.fitted_model.df_model:<7} "
                  f"{self.fitted_model.llf:<8.4f} {self.fitted_model.aic:<8.4f} "
                  f"{self.fitted_model.bic:<8.4f} {self.fitted_model.deviance:<8.4f} "
                  f"{self.fitted_model.df_resid:<11} {self.fitted_model.nobs:<4}")

        if self.save_path:
            print("\npath")
            print(self.save_path)


class TeWeightsSpec:
    """Stores specifications for weight models, including formulas and fitted results."""

    def __init__(self, numerator, denominator, pool_numerator, pool_denominator, model_fitter):
        self.numerator = numerator  # Formula for numerator model
        self.denominator = denominator  # Formula for denominator model
        self.pool_numerator = pool_numerator  # Whether numerator model is pooled
        self.pool_denominator = pool_denominator  # Whether denominator model is pooled
        self.model_fitter = model_fitter  # Model fitter instance
        self.fitted = {}  # Dictionary to store fitted models
        self.data_subset_expr = {}  # Store data filtering expressions

    def show(self):
        """Display weight model specifications."""
        print(f" - Numerator formula: {self.numerator}")
        print(f" - Denominator formula: {self.denominator}")
        if self.pool_numerator:
            if self.pool_denominator:
                print(" - Numerator and denominator models are pooled across treatment arms.")
            else:
                print(" - Numerator model is pooled across treatment arms. Denominator model is not pooled.")
        print(f" - Model fitter type: {type(self.model_fitter).__name__}")
        if self.fitted:
            print(" - View weight model summaries with `show_weight_models()`")
        else:
            print(" - Weight models not fitted. Use `calculate_weights()`")

    def show_weight_models(self):
        """Display summaries of the fitted weight models."""
        if not self.fitted:
            print(" - No weight models fitted. Use `calculate_weights()` first.")
            return
        
        print("\n=== Weight Model Summaries ===")
        for key, model in self.fitted.items():
            print(f"[[{key}]]")
            model.show()

class TeWeightsUnset(TeWeightsSpec):
    """Represents an unset weight model."""
    
    def __init__(self):
        super().__init__(numerator=None, denominator=None, pool_numerator=False, pool_denominator=False, model_fitter=None)

    def show(self):
        """Display message for unset weight model."""
        print(" - No weight model specified")
