import numpy as np

from copy import deepcopy


def multiply_parameter_sets(parameter_set_1, parameter_set_2, inplace=False):
    """Multiply two parameter_set dictionaries
    Args :
        parameter_set_1 (dict) :
            First dictionary containing parameter_set components
        parameter_set_2 (dict) :
            Second dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether multiplication should happen inplace
            Take note that only the first dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing parameter_set components of product
    """

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set_1 = deepcopy(parameter_set_1)

    # Set components individually if they are available
    if parameter_set_2.get('input') != None:
        parameter_set_1['input'] = parameter_set_2.get('input')
    if parameter_set_2.get('output') != None:
        parameter_set_1['output'] = parameter_set_2.get('output')

    return parameter_set_1


def inpl_multiply_parameter_sets(parameter_set_1, parameter_set_2, inplace=False):
    return multiply_parameter_sets(parameter_set_1, parameter_set_2, True)
