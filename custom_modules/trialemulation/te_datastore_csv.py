import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TEDatastoreCSV:
    """Class for managing a CSV datastore."""
    path: str
    files: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["file", "period"]))
    template: pd.DataFrame = field(default_factory=pd.DataFrame)
    N: int = 0

    def __post_init__(self):
        """Ensure the directory exists and is empty."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        elif os.listdir(self.path):
            raise ValueError(f"{self.path} must be empty")

    def show(self):
        """Display information about the CSV datastore."""
        print("A TE Datastore CSV object")
        print(f"N: {self.N} observations")
        print(f"Periods: {self.files['period'].tolist()}")
        print(f"Path: {self.path}")
        print(f"Columns: {', '.join(self.template.columns)}")

    def save_expanded_data(self, data: pd.DataFrame):
        """Save expanded data to CSV files."""
        trial_periods = data["trial_period"].unique()
        for period in trial_periods:
            file_path = os.path.join(self.path, f"trial_{period}.csv")
            data[data["trial_period"] == period].to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

        self.N += len(data)
        self.files = pd.DataFrame({
            "file": [os.path.join(self.path, f"trial_{period}.csv") for period in trial_periods],
            "period": trial_periods
        })

        if self.template.empty:
            self.template = data.iloc[0:0]  # Create an empty template based on the data structure

    def read_expanded_data(self, period: Optional[List[int]] = None, subset_condition: Optional[str] = None) -> pd.DataFrame:
        """Read expanded data with optional filtering."""
        if period is None:
            files_to_read = self.files["file"].tolist()
        else:
            files_to_read = self.files[self.files["period"].isin(period)]["file"].tolist()

        data_frames = [pd.read_csv(file) for file in files_to_read]
        data_table = pd.concat(data_frames, ignore_index=True)

        if subset_condition is not None:
            data_table = data_table.query(subset_condition)

        return data_table

    def sample_expanded_data(self, p_control: float, period: Optional[List[int]] = None, subset_condition: Optional[str] = None, seed: Optional[int] = None) -> pd.DataFrame:
        """Sample expanded data based on control probability."""
        if seed is not None:
            pd.np.random.seed(seed)  # Set the random seed

        all_periods = self.files["period"].tolist()

        if period is None:
            periods = all_periods
        else:
            periods = [p for p in period if p in all_periods]
            if len(periods) < len(period):
                omitted = set(period) - set(periods)
                print(f"Warning: The following periods don't exist in the data and were omitted: {omitted}")

        sampled_data = pd.concat([
            self.read_expanded_data(p, subset_condition).sample(frac=p_control) for p in periods
        ], ignore_index=True)

        return sampled_data

# Example usage
def save_to_csv(path: str) -> TEDatastoreCSV:
    """Function to create a new TE Datastore CSV object."""
    return TEDatastoreCSV(path=path)

# Example of creating a CSV datastore
csv_dir = "path/to/csv_directory"
csv_datastore = save_to_csv(csv_dir)