import os
import time
import pandas as pd
import re


def quiet_print(quiet: bool, x, *args, **kwargs):
    """Conditional printing."""
    if not quiet:
        print(x, *args, **kwargs)


def quiet_msg(quiet: bool, x, *args, **kwargs):
    """Conditional messaging (analogous to R's `message`)."""
    if not quiet:
        print(x, *args, **kwargs)


def quiet_line(quiet: bool):
    """Print a separator line conditionally."""
    quiet_msg(quiet, console_line() + "\n")


def console_line(prop: float = 0.75) -> str:
    """Generate a line separator based on terminal width."""
    return "-" * int(prop * os.get_terminal_size().columns)


def cat_underline(text: str, newlines: int = 2):
    """Print text underlined."""
    print(text)
    print("-" * len(text))
    print("\n" * newlines, end='')


def quiet_msg_time(quiet: bool, msg: str, proc_time: float):
    """Print a message with elapsed time."""
    formatted_time = f"{proc_time:.1f} s" if proc_time < 10 else f"{proc_time:.5g} s"
    quiet_msg(quiet, f"{msg}{formatted_time}")


def assert_monotonic(x, increasing=True):
    """Assert monotonicity of a numeric vector."""
    if increasing and not all(x[i] <= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Not monotonically increasing")
    elif not increasing and not all(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Not monotonically decreasing")


def as_formula(x):
    """Convert a string or list of variables into a formula-like string."""
    if isinstance(x, str) and "~" in x:
        return x  # Assume it's already a formula
    elif isinstance(x, (list, tuple)):
        return f"~ {' + '.join(x)}"
    raise TypeError("Input must be a string with '~' or a list of variable names.")


def add_rhs(f1: str, f2: str) -> str:
    """Combine right-hand sides of two formula-like strings."""
    f1_rhs = f1.split("~")[1].strip() if "~" in f1 else f1
    f2_rhs = f2.split("~")[1].strip() if "~" in f2 else f2
    return f"~ {f1_rhs} + {f2_rhs}"


def rhs_vars(f: str):
    """Extract variables from the right-hand side of a formula-like string."""
    return re.findall(r'\b\w+\b', f.split("~")[1]) if "~" in f else []


def extract_baseline(trial_file: str, baseline_file: str, quiet: bool = True):
    """
    Extract baseline observations from a CSV file.

    Reads `trial_file` and saves rows where `followup_time == 0` to `baseline_file`.
    """
    if os.path.exists(trial_file):
        quiet_msg(quiet, f"Extracting baseline observations from {trial_file}")
        df = pd.read_csv(trial_file)
        baseline_df = df[df["followup_time"] == 0]
        baseline_df.to_csv(baseline_file, index=False)
        return baseline_file
    else:
        raise FileNotFoundError(f"File not found: {trial_file}")


def catn(*args):
    """Print with an extra newline."""
    print(*args, "\n")


def drop_path(snapshot: str) -> str:
    """Remove file paths from snapshot tests."""
    return re.sub(r"Path: \S*", "Path:", snapshot)
