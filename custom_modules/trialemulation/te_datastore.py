import pandas as pd

class TEDatastore:
    """Base class for defining how expanded trial data should be stored."""
    
    def __init__(self):
        self.N = 0  # Number of observations

    def __repr__(self):
        return f"TEDatastore(N={self.N})"


class TEDatastoreDataTable(TEDatastore):
    """Class for storing expanded data as a pandas DataFrame."""
    
    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame()  # Initialize an empty DataFrame

    def show(self):
        """Display information about the datastore."""
        print("A TE Datastore Datatable object")
        print(f"N: {self.N} observations")
        print(self.data.head(4))  # Show the first 4 rows of the DataFrame

    def save_expanded_data(self, data):
        """Save expanded data to the datastore."""
        self.data = pd.concat([self.data, data], ignore_index=True)
        self.N = len(self.data)  # Update the number of observations
        return self

    def read_expanded_data(self, period=None, subset_condition=None):
        """Read expanded data with optional filtering."""
        if period is not None:
            if not isinstance(period, (list, range)):
                raise ValueError("period must be a list or range of integers.")
            data_table = self.data[self.data['trial_period'].isin(period)]
        else:
            data_table = self.data

        if subset_condition is not None:
            # Evaluate the subset condition using pandas query
            data_table = data_table.query(subset_condition)

        return data_table


# Example usage
def save_to_datatable():
    """Function to create a new TE Datastore Datatable object."""
    return TEDatastoreDataTable()
