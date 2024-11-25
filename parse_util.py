def try_float(target):
    """Try converting a string to a float and return None if parsing fails.

    Args:
        target: The string to parse.

    Returns:
        The number parsed.
    """
    try:
        return float(target)
    except ValueError:
        return None


def try_int(target):
    """Try converting a string to an int and return None if parsing fails.

    Args:
        target: The string to parse.

    Returns:
        The number parsed.
    """
    try:
        return int(target)
    except ValueError:
        return round(try_float(target))
