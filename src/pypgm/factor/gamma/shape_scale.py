from copy import deepcopy


def multiply_shape_scale_forms(parameters_1, parameters_2, inplace=False):
    """Multiply two shape_scale parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing shape_scale parameters
        parameters_2 (dict) :
            Second dictionary containing shape_scale parameters
        inplace (bool) :
            Boolean value indicating whether multiplication should happen inplace
            Take note that only the first parameter dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing shape_scale parameters of product
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)

    # Add shape attributes
    parameters_1['shape'] = parameters_1['shape'] + parameters_2['shape'] - 1

    # Add scale attributes
    parameters_1['scale'] = 1./(1./parameters_1['scale'] + 1./parameters_2['scale'])

    return parameters_1


def inpl_multiply_shape_scale_forms(parameters_1, parameters_2):
    return multiply_shape_scale_forms(parameters_1, parameters_2, inplace=True)


def divide_shape_scale_forms(parameters_1, parameters_2, inplace=False):
    """Divide two shape_scale parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing shape_scale parameters
        parameters_2 (dict) :
            Second dictionary containing shape_scale parameters
        inplace (bool) :
            Boolean value indicating whether division should happen inplace
            Take note that only the first parameter dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing shape_scale parameters of quotient
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)

    # Subtract shape attributes
    parameters_1['shape'] = parameters_1['shape'] - parameters_2['shape'] + 1

    # Subtract scale attributes
    parameters_1['scale'] = 1./(1./parameters_1['scale'] - 1./parameters_2['scale'])

    return parameters_1


def inpl_divide_shape_scale_forms(parameters_1, parameters_2):
    return divide_shape_scale_forms(parameters_1, parameters_2, inplace=True)
