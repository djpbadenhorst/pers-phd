def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from dsigmoid factor to scalar_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of dsigmoid factor sending message
        to_factor_parameters (dict) :
            Parameters of scalar_normal factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized scalar_normal message
    """

    from pypgm.factor import Factor

    # Create message from parameters of factor receiving message
    message = Factor(**to_factor_parameters)
        
    return message.parameters
