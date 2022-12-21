from copy import deepcopy


def standard_to_canonical(parameters, inplace=False):
    """Convert standard parameters into canonical parameters
    Args :
        parameters (dict) :
            Dictionary containing standard parameters
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing canonical parameters
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters = deepcopy(parameters)

    # Obtain relevant parameters
    mean = parameters.pop('mean')
    var = parameters.pop('var')

    # Convert canonical parameters to standard parameters
    prec = 1./var
    wmean = prec*mean

    # Update parameter dictionary
    parameters['wmean'] = wmean
    parameters['prec'] = prec

    return parameters


def canonical_to_standard(parameters, inplace=False):
    """Convert canonical parameters into standard parameters
    Args :
        parameters (dict) :
            Dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing standard parameters
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters = deepcopy(parameters)

    # Obtain relevant parameters
    wmean = parameters.pop('wmean')
    prec = parameters.pop('prec')

    # Convert canonical parameters to standard parameters
    var = 1./prec
    mean = var*wmean

    # Update parameter dictionary
    parameters['mean'] = mean
    parameters['var'] = var

    return parameters


def to_canonical(parameters, inplace=False):
    """Ensures given parameters are in canonical form
    Args :
        parameters (dict) :
            Dictionary containing parameters of any form
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing canonical parameters
    """

    from pypgm.factor import scalar_normal

    # If parameters are in canonical form    
    if parameters['par_form'] == scalar_normal.PARAMETER_FORM.CANONICAL:
        if not(inplace):
            parameters = deepcopy(parameters)

    # If parameters are in standard form
    elif parameters['par_form'] == scalar_normal.PARAMETER_FORM.STANDARD:
        parameters = standard_to_canonical(parameters, inplace)
        parameters['par_form'] = scalar_normal.PARAMETER_FORM.CANONICAL
    
    else:
        raise Exception('Parameter dictionary has invalid parameter form attribute')

    return parameters


def inpl_to_canonical(parameters):
    return to_canonical(parameters, inplace=True)


def to_standard(parameters, inplace=False):
    """Ensures given parameters are in standard form
    Args :
        parameters (dict) :
            Dictionary containing parameters of any form
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing standard parameters
    """

    from pypgm.factor import scalar_normal

    # If parameters are in standard form    
    if parameters['par_form'] == scalar_normal.PARAMETER_FORM.STANDARD:
        if not(inplace):
            parameters = deepcopy(parameters)

    # If parameters are in canonical form
    elif parameters['par_form'] == scalar_normal.PARAMETER_FORM.CANONICAL:
        parameters = canonical_to_standard(parameters, inplace)
        parameters['par_form'] = scalar_normal.PARAMETER_FORM.STANDARD
    
    else:
        raise Exception('Parameter dictionary has invalid parameter form attribute')

    return parameters


def inpl_to_standard(parameters):
    return to_standard(parameters, inplace=True)
