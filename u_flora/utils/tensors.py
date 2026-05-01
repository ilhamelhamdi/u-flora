import torch


def cast_state_dict_for_arrayrecord(state_dict: dict[str, torch.Tensor]) -> dict:
    """Cast unsupported dtypes (e.g., bfloat16) to float32 for ArrayRecord."""
    converted: dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            converted[key] = value.float()
        else:
            converted[key] = value
    return converted
