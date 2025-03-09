import pandas as pd
import statsmodels.api as sm

from custom_modules.trialemulation.te_data import TEDataUnset

class TEWeightsFitted:
    """Fitted Weights Object"""
    def __init__(self, label, summary=None, fitted=None):
        self.label = label
        self.summary = summary if summary else {}
        self.fitted = fitted if fitted else {}

    def show(self):
        print(f"Model: {self.label}")
        for key, df in self.summary.items():
            print(f"Summary [{key}]:\n", df)

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