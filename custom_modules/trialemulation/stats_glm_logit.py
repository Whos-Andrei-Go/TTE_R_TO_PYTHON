def calculate_weights_trial_seq(trial, quiet=False, switch_weights=True, censor_weights=True):
    """
    Wrapper function to calculate weights for a trial sequence.

    :param trial: Instance of TrialSequence class.
    :param quiet: Suppress output (not used in Python, but kept for compatibility).
    :param switch_weights: Boolean, whether to calculate switch weights.
    :param censor_weights: Boolean, whether to calculate censor weights.
    """
    if not isinstance(trial, TrialSequence):
        raise TypeError("trial must be an instance of TrialSequence or its subclasses.")

    trial.calculate_weights(switch_weights=switch_weights, censor_weights=censor_weights)
    return trial
