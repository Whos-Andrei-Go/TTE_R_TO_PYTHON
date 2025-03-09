class TEExpansion:
    """
    Represents the sequence of trials data expansion.

    Attributes:
    - chunk_size (int): Size of data chunks.
    - datastore (TEDatastore): Data storage object.
    - censor_at_switch (bool): Whether to apply censoring at treatment switch.
    - first_period (int): First time period in the data.
    - last_period (int): Last time period in the data.
    """

    def __init__(self, chunk_size: int, datastore, censor_at_switch: bool, first_period: int, last_period: int):
        self.chunk_size = chunk_size
        self.datastore = datastore  # Assumes TEDatastore is implemented elsewhere
        self.censor_at_switch = censor_at_switch
        self.first_period = first_period
        self.last_period = last_period

    def __str__(self):
        output = [
            "Sequence of Trials Data:",
            f"- Chunk size: {self.chunk_size}",
            f"- Censor at switch: {self.censor_at_switch}",
            f"- First period: {self.first_period} | Last period: {self.last_period}",
        ]
        if self.datastore.N > 0:  # Assuming datastore has attribute N
            output.append("\n" + str(self.datastore))
        else:
            output.append("- Use expand_trials() to construct the sequence of trials dataset.")
        return "\n".join(output)

    def show(self):
        """Display the object details."""
        print(self)


class TEExpansionUnset(TEExpansion):
    """
    Represents an unset state of TEExpansion.
    """

    def __init__(self):
        super().__init__(chunk_size=0, datastore=None, censor_at_switch=False, first_period=0, last_period=0)

    def __str__(self):
        return "Sequence of Trials Data:\n- Use set_expansion_options() and expand_trials() to construct the sequence of trials dataset."
