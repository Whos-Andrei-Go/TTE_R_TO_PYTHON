import os
import duckdb
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TEDatastoreDuckDB:
    """Class for managing a DuckDB datastore."""
    path: str
    table: str = "trial_data"
    con: Optional[duckdb.DuckDBPyConnection] = field(init=False, default=None)
    N: int = field(default=0)

    def __post_init__(self):
        """Initialize the DuckDB connection."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_path = os.path.join(self.path, "expanded_data.duckdb")
        self.con = duckdb.connect(database=self.file_path, read_only=False)

    def show(self):
        """Display information about the DuckDB datastore."""
        print("A TE Datastore DuckDB object")
        print(f"N: {self.N} observations")
        print(f"Path: {self.file_path}")

    def save_expanded_data(self, data: pd.DataFrame):
        """Save expanded data to DuckDB."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} does not exist.")
        
        if not duckdb.query(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]:
            duckdb.write_table(self.con, self.table, data)
        else:
            duckdb.append_table(self.con, self.table, data)

        self.N += len(data)

    def read_expanded_data(self, period=None, subset_condition=None):
        """Read expanded data with optional filtering."""
        query = f"SELECT * FROM {self.table}"
        conditions = []

        if period is not None:
            conditions.append(f"trial_period IN ({', '.join(map(str, period))})")
        
        if subset_condition is not None:
            conditions.append(subset_condition)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return pd.read_sql(query, self.con)

    def sample_expanded_data(self, p_control, period=None, subset_condition=None, seed=None):
        """Sample expanded data based on control probability."""
        query = f"SELECT * FROM {self.table} WHERE outcome = 0"
        if period is not None:
            query += f" AND trial_period IN ({', '.join(map(str, period))})"
        if subset_condition is not None:
            query += f" AND {subset_condition}"

        data_0 = pd.read_sql(query, self.con)

        query = f"SELECT * FROM {self.table} WHERE outcome = 1"
        if period is not None:
            query += f" AND trial_period IN ({', '.join(map(str, period))})"
        if subset_condition is not None:
            query += f" AND {subset_condition}"

        data_1 = pd.read_sql(query, self.con)

        # Combine and sample
        combined_data = pd.concat([data_0, data_1])
        if seed is not None:
            np.random.seed(seed)

        sampled_data = combined_data.sample(frac=p_control, random_state=seed)
        sampled_data["sample_weight"] = np.where(sampled_data["outcome"] == 1, 1, 1 / p_control)

        return sampled_data

def translate_to_sql(string: str) -> str:
    """Translate subset_condition to SQL syntax."""
    replacements = {
        "|": "OR",
        "&": "AND",
        "==": "=",
        "%in%": "IN",
        "^c\\(": "("
    }

    for old, new in replacements.items():
        string = string.replace(old, new)

    return string

def translate_num_vec(vec):
    """Translate numerical vectors to SQL syntax."""
    for i in range(len(vec)):
        if ":" in vec[i]:
            start, end = map(int, vec[i].split(":"))
            vec[i] = f"({', '.join(map(str, range(start, end + 1)))})"
    return vec