import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from scipy.stats import multivariate_normal
from scipy.special import expit  # Sigmoid function (inverse logit)
import patsy


def predict_TE_msm(model, newdata, predict_times, conf_int=True, samples=100, pred_type="cum_inc"):
    if model.model.family.link.__class__.__name__ != "Logit":
        raise ValueError("Only logistic regression models are supported.")

    coefs = model.params.values.reshape(1, -1)

    if conf_int:
        robust_matrix = model.cov_params()
        coefs = np.vstack([coefs, multivariate_normal.rvs(mean=model.params, cov=robust_matrix, size=samples)])

    newdata = check_newdata(newdata, model, predict_times)

    pred_fun = calculate_survival if pred_type == "survival" else calculate_cum_inc

    treatment_values = {"assigned_treatment_0": 0, "assigned_treatment_1": 1}
    pred_list = calculate_predictions(newdata, model, treatment_values, pred_fun, coefs, len(predict_times))

    pred_list["difference"] = pred_list["assigned_treatment_1"] - pred_list["assigned_treatment_0"]

    result = {}
    for key, pred_matrix in pred_list.items():
        col_name = pred_type if key == "difference" else f"{pred_type}_{key[-1]}"
        if conf_int:
            quantiles = np.percentile(pred_matrix, [2.5, 97.5], axis=1).T
            result[key] = pd.DataFrame({
                "followup_time": predict_times,
                col_name: pred_matrix[:, 0],
                "2.5%": quantiles[:, 0],
                "97.5%": quantiles[:, 1]
            })
        else:
            result[key] = pd.DataFrame({"followup_time": predict_times, col_name: pred_matrix[:, 0]})

    return result

def check_newdata(newdata, model, predict_times):
    required_vars = [var for var in model.model.exog_names if var != "outcome"]

    if newdata is None:
        newdata = model.model.data.frame[required_vars].copy()
        newdata = newdata[newdata["followup_time"] == 0]
    else:
        if not isinstance(newdata, pd.DataFrame):
            raise ValueError("newdata must be a pandas DataFrame.")
        
        missing_vars = set(required_vars) - set(newdata.columns)
        if missing_vars:
            raise ValueError(f"newdata is missing required columns: {missing_vars}")

        newdata = newdata[required_vars].copy()
        newdata = newdata[newdata["followup_time"] == 0]

        # Check attributes match (data types)
        model_dtypes = model.model.data.frame[required_vars].dtypes.to_dict()
        newdata_dtypes = newdata.dtypes.to_dict()

        if model_dtypes != newdata_dtypes:
            warnings.warn("Attributes of newdata do not match data used for fitting. Attempting to fix.")
            newdata = pd.concat([model.model.data.frame[required_vars].iloc[:0], newdata])
            if model.model.data.frame[required_vars].dtypes.to_dict() != newdata.dtypes.to_dict():
                raise ValueError("Failed to fix attributes. Data types do not match.")

    # Expand newdata for each predict_time
    newdata_expanded = pd.concat([newdata] * len(predict_times), ignore_index=True)
    newdata_expanded["followup_time"] = np.repeat(predict_times, len(newdata))

    return newdata_expanded

def calculate_cum_inc(p_mat):
    if not isinstance(p_mat, np.ndarray):
        raise ValueError("p_mat must be a numpy array")
    
    result = 1 - calculate_survival(p_mat)

    if not np.all(np.diff(result, axis=1) >= 0):
        raise ValueError("Result is not monotonically increasing")
    
    return result

def calculate_survival(p_mat):
    if not isinstance(p_mat, np.ndarray):
        raise ValueError("p_mat must be a numpy array")
    
    result = cumprod_matrix(1 - p_mat, by="rows").mean(axis=0)

    if not np.all(np.diff(result) <= 0):
        raise ValueError("Result is not monotonically decreasing")
    
    return result

def cumprod_matrix(x, by="rows"):
    if by not in ["rows", "cols"]:
        raise ValueError("by must be 'rows' or 'cols'")

    y = np.ones_like(x)

    if by == "cols":
        y[0, :] = x[0, :]
        for i in range(1, x.shape[0]):
            y[i, :] = y[i - 1, :] * x[i, :]
    elif by == "rows":
        y[:, 0] = x[:, 0]
        for i in range(1, x.shape[1]):
            y[:, i] = y[:, i - 1] * x[:, i]

    return y

def calculate_predictions(newdata, model, treatment_values, pred_fun, coefs_mat, matrix_n_col):
    """
    Calculate and transform predictions.

    Parameters:
    - newdata (pd.DataFrame): New data to predict outcome.
    - model (sklearn.linear_model or statsmodels GLM): The trained GLM model.
    - treatment_values (dict): Mapping of treatment variable values, e.g., {'assigned_treatment_0': 0, 'assigned_treatment_1': 1}.
    - pred_fun (function): Function to transform the prediction matrix.
    - coefs_mat (np.ndarray): Coefficient matrix (samples x features).
    - matrix_n_col (int): Expected number of columns after prediction.

    Returns:
    - dict: Dictionary with transformed predicted values for different treatment values.
    """
    results = {}

    for treatment_name, treatment_value in treatment_values.items():
        newdata["assigned_treatment"] = treatment_value  # Set treatment value
        
        # Create model matrix (design matrix)
        model_matrix = model_design_matrix(newdata, model)  # Function to build model matrix

        pred_list = []
        for coef_vec in coefs_mat:
            linear_pred = model_matrix @ coef_vec.T  # Linear combination
            predicted_probs = expit(linear_pred)  # Apply inverse logit (sigmoid function)

            pred_list.append(pred_fun(predicted_probs.reshape(-1, matrix_n_col)))

        results[treatment_name] = np.column_stack(pred_list)  # Stack results column-wise

    return results

def model_design_matrix(newdata, model):
    """
    Construct a design matrix for a given model.

    Parameters:
    - newdata (pd.DataFrame): The new data to use for predictions.
    - model (statsmodels GLM or sklearn model): The trained model.

    Returns:
    - np.ndarray: The design matrix.
    """
    formula = model.formula if hasattr(model, "formula") else None

    if formula:
        return patsy.dmatrix(formula, newdata, return_type="dataframe").values
    else:
        return newdata[model.feature_names_in_].values  # Use sklearn-style features