import pandas as pd

class TEData:
    """
    Trial Emulation Data Class

    Attributes:
    - data (pd.DataFrame): Must contain columns ["id", "period", "treatment", "outcome", "eligible"].
    - nobs (int): Number of observations.
    - n (int): Number of unique individuals.
    """

    REQUIRED_COLUMNS = ["id", "period", "treatment", "outcome", "eligible"]

    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.data = data
        self.nobs = len(data)
        self.n = data["id"].nunique()

    def show(self):
        """Displays basic information about the dataset."""
        print(f" - N: {self.nobs} observations from {self.n} patients")
        show_cols = [col for col in self.data.columns if col not in {
            "time_of_event", "first", "am_1", "cumA", "switch", "regime_start", "eligible0", "eligible1",
            "p_n", "p_d", "pC_n", "pC_d"
        }]
        print(self.data[show_cols].head(4))


class TEDataUnset(TEData):
    """
    Class for an unset TEData object.
    """

    def __init__(self):
        super().__init__(pd.DataFrame(columns=TEData.REQUIRED_COLUMNS))
        self.nobs = 0
        self.n = 0

    def show(self):
        print(" - No data has been set. Use set_data()")


class TEOutcomeData:
    """
    Trial Emulation Outcome Data Class

    Attributes:
    - data (pd.DataFrame): Must contain columns ["id", "trial_period", "followup_time", "outcome", "weight"].
    - n_rows (int): Number of rows.
    - n_ids (int): Number of unique patients.
    - periods (list): Unique trial periods.
    - p_control (float): Probability control parameter.
    - subset_condition (str): Subset condition used for filtering.
    """

    REQUIRED_COLUMNS = ["id", "trial_period", "followup_time", "outcome", "weight"]

    def __init__(self, data, p_control=None, subset_condition=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.data = data
        self.n_rows = len(data)
        self.n_ids = data["id"].nunique()
        self.periods = sorted(data["trial_period"].unique())
        self.p_control = p_control if p_control is not None else None
        self.subset_condition = subset_condition if subset_condition is not None else None

        if self.n_rows == 0:
            print("Warning: Outcome data has 0 rows")

    def show(self):
        """Displays outcome data information."""
        if self.n_rows == 0:
            print("No outcome data, use load_expanded_data()")
        else:
            print("Outcome data")
            print(f"N: {self.n_rows} observations from {self.n_ids} patients in {len(self.periods)} trial periods")
            print(f"Periods: {self.periods}")
            if self.subset_condition:
                print(f"Subset condition: {self.subset_condition}")
            if self.p_control:
                print(f"Sampling control observations with probability: {self.p_control}")
            print(self.data.head(4))


# Example Usage:
# data_df = pd.DataFrame({...})  # DataFrame with required columns
# te_data = TEData(data_df)
# te_data.show()
#
# outcome_data_df = pd.DataFrame({...})  # DataFrame with outcome-related columns
# te_outcome_data = TEOutcomeData(outcome_data_df)
# te_outcome_data.show()
