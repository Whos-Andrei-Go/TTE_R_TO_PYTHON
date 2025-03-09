import pandas as pd
import statsmodels.api as sm
import numpy as np

from custom_modules.trialemulation.te_data import TEDataUnset

class TEWeightsFitted:
    """Fitted Weights Object"""
    def __init__(self, label, summary=None, fitted=None):
        self.label = label
        self.summary = summary if summary else {}
        
        # Ensure fitted is either a pandas Series or array-like object, not a dict
        if fitted is None:
            self.fitted = None
        elif isinstance(fitted, (np.ndarray, pd.Series)):
            self.fitted = fitted
        else:
            raise ValueError("Fitted values must be a pandas Series or numpy ndarray.")
    
    def show(self):
        print(f"Model: {self.label}")
        for key, df in self.summary.items():
            print(f"Summary [{key}]:\n", df)
        
        # Show fitted values if available
        if self.fitted is not None:
            print(f"Fitted values:\n", self.fitted)
        else:
            print("No fitted values available.")


class TEWeightsSpec:
    """Weight specification"""
    def __init__(self, numerator, denominator, pool_numerator, pool_denominator, model_fitter):
        self.numerator = numerator
        self.denominator = denominator
        self.pool_numerator = pool_numerator
        self.pool_denominator = pool_denominator
        self.model_fitter = model_fitter
        self.fitted = {}
        self.data_subset_expr = {}

    def show(self):
        print(f" - Numerator formula: {self.numerator}")
        print(f" - Denominator formula: {self.denominator}")
        if self.pool_numerator:
            if self.pool_denominator:
                print(" - Numerator and denominator models are pooled across treatment arms.")
            else:
                print(" - Numerator model is pooled. Denominator model is not.")
        print(f" - Model fitter type: {type(self.model_fitter).__name__}")
        if self.fitted:
            print(" - View weight model summaries with show_weight_models()")
        else:
            print(" - Weight models not fitted. Use calculate_weights()")

class TEWeightsSwitch(TEWeightsSpec):
    pass

class TEWeightsCensoring(TEWeightsSpec):
    pass

class TEWeightsUnset(TEWeightsSpec):
    def __init__(self):
        super().__init__(numerator="1", denominator="1", pool_numerator=False, pool_denominator=False, model_fitter=None)