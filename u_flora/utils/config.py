def replace_keys(input_dict: dict, match: str = "-", target: str = "_") -> dict:
    """Recursively replace ``match`` with ``target`` in dictionary keys.

    Flower configs use hyphens in keys (e.g. ``learning-rate-max``)
    but OmegaConf / Python attrs use underscores.
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
