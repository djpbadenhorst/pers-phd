def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from scalar_normal factor to dsigmoid factor
    Args :
        from_factor_parameters (dict) :
            Parameters of scalar_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of dsigmoid factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized dsigmoid message
    """

    from pypgm.factor import FACTOR_TYPE, Factor

    # Create message from parameters of factor sending message
    if set(domain) == set(to_factor_parameters['input_vars']):
        message = Factor(ftype = FACTOR_TYPE.DSIGMOID,
                         input_vars = to_factor_parameters['input_vars'],
                         input = from_factor_parameters)

    elif set(domain) == set(to_factor_parameters['output_vars']):
        message = Factor(ftype = FACTOR_TYPE.DSIGMOID,
                         output_vars = to_factor_parameters['output_vars'],
                         output = from_factor_parameters)

    else:
        raise Exception("Incompatible domain argument given")
   
    return message.parameters
