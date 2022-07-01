class ParameterException(Exception):
    """Basic exception handling for missing parameters."""
    def __init__(self, param):
        super().__init__(f'{param} missing from measurement parameters.')


def get_param(p: str, params: dict):
    """
    Get a parameter by key from a parameter dictionary.

    :param p: The parameter name (key).
    :param params: The parameter dictionary.
    :returns: The specified parameter (value).
    :raises ParameterException: If p is not a key in params.
    """
    if p not in params:
        raise ParameterException(p)
    return params[p]

