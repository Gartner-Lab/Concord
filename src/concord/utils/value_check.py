

def validate_probability(prob, name):
    if prob is not None:
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"{name} should be between 0 and 1.0, got {prob}")

def validate_probability_dict_compatible(prob, name):
    if isinstance(prob, dict):
        # Validate each probability in the dictionary
        for key, value in prob.items():
            validate_probability(value, f"{name}_[{key}]")
    else:
        validate_probability(prob, name)

def check_dict_condition(dictionary, condition):
    for _, value in dictionary.items():
        if value is not None and condition(value):
            return True
    return False
