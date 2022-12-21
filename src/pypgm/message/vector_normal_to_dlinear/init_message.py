def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from vector_normal factor to dlinear factor
    Args :
        from_factor_parameters (dict) :
            Parameters of vector_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of dlinear factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized vector_normal message
    """

    import numpy as np

    from pypgm.factor import FACTOR_TYPE, Factor

    # Create message from parameters of factor sending message
    if set(domain) == set(to_factor_parameters['input_vars']):
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         input_vars = np.sort(to_factor_parameters['input_vars']).tolist(),
                         input = from_factor_parameters)
        
    elif set(domain) == set(to_factor_parameters['weight_vars']):
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         weight_vars = np.sort(to_factor_parameters['weight_vars']).tolist(),
                         weight = from_factor_parameters)

    else:
        raise Exception("Incompatible domain argument given")
   
    return message.parameters
