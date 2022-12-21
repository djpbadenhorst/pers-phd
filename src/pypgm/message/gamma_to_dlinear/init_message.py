def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from gamma factor to dlinear factor
    Args :
        from_factor_parameters (dict) :
            Parameters of gamma factor sending message
        to_factor_parameters (dict) :
            Parameters of dlinear factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized gamma message
    """

    from pypgm.factor import FACTOR_TYPE, Factor

    # Create message from parameters of factor sending message
    if set(domain) == set(to_factor_parameters['modval_vars']):
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         modval_vars = to_factor_parameters['modval_vars'],
                         modval = from_factor_parameters)

    else:
        raise Exception("Incompatible domain argument given")
   
    return message.parameters
