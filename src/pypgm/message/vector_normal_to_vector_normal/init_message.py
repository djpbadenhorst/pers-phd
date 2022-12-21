def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from vector_normal factor to vector_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of vector_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of vector_normal factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized vector_normal message
    """

    from pypgm.factor import Factor
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Marginalize to relevant subset of variables
    message.marginalize_to(domain)
   
    return message.parameters
