class TeExpansion:
    def __init__(self, chunk_size=0, datastore=None, censor_at_switch=False, first_period=0, last_period=float("inf")):
        self.chunk_size = chunk_size
        self.datastore = datastore  # Should be an instance of TeDatastore or None
        self.censor_at_switch = censor_at_switch
        self.first_period = first_period
        self.last_period = last_period

    def __str__(self):
        output = (
            f"Sequence of Trials Data:\n"
            f"- Chunk size: {self.chunk_size}\n"
            f"- Censor at switch: {self.censor_at_switch}\n"
            f"- First period: {self.first_period} | Last period: {self.last_period}"
        )
        if self.datastore and hasattr(self.datastore, "N") and self.datastore.N > 0:
            output += f"\n\n{self.datastore}"  # Calls the datastore's __str__ method
        else:
            output += "\n- Use expand_trials() to construct the sequence of trials dataset."
        return output

class TeExpansionUnset(TeExpansion):
    def __init__(self):
        """Unset expansion should not have datastore or trial settings."""
        super().__init__()  # Initialize with default values
        self.datastore = None  # Explicitly remove datastore

    def __str__(self):
        return (
            "Sequence of Trials Data:\n"
            "- Use set_expansion_options() and expand_trials() to construct the sequence of trials dataset."
        )
