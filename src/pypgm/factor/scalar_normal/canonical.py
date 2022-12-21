from copy import deepcopy


def multiply_canonical_forms(parameters_1, parameters_2, inplace=False):
    """Multiply two canonical parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing canonical parameters
        parameters_2 (dict) :
            Second dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether multiplication should happen inplace
            Take note that only the first dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing canonical parameters of product
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)

    # Add wmean attributes
    parameters_1['wmean'] = parameters_1['wmean'] + parameters_2['wmean']

    # Add precision attributes
    parameters_1['prec'] = parameters_1['prec'] + parameters_2['prec']

    return parameters_1


def inpl_multiply_canonical_forms(parameters_1, parameters_2):
    return multiply_canonical_forms(parameters_1, parameters_2, inplace=True)


def divide_canonical_forms(parameters_1, parameters_2, inplace=False):
    """Divide two canonical parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing canonical parameters
        parameters_2 (dict) :
            Second dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether division should happen inplace
            Take note that only the first dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing canonical parameters of quotient
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)

    # Subtract wmean attributes
    parameters_1['wmean'] = parameters_1['wmean'] - parameters_2['wmean']

    # Subtract precision attributes
    parameters_1['prec'] = parameters_1['prec'] - parameters_2['prec']

    return parameters_1


def inpl_divide_canonical_forms(parameters_1, parameters_2):
    return divide_canonical_forms(parameters_1, parameters_2, inplace=True)
