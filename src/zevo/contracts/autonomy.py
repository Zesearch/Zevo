"""Autonomy classification by the number of user-pinned decisions."""


def autonomy_level(dataset: str, base_model: str, training_method: str) -> str:
    """L1 pins all three decisions; L2 any two; L3 any one; L4 none."""
    pinned = sum(bool((value or "").strip()) for value in (dataset, base_model, training_method))
    return f"L{4 - pinned}"
