def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from scalar_normal factor to vector_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of scalar_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of vector_normal factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized vector_normal message
    """

    from pypgm.factor import FACTOR_TYPE, Factor, vector_normal
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Update parameter attribute
    message.parameters['ftype'] = FACTOR_TYPE.VECTOR_NORMAL
    message.parameters['par_form'] = vector_normal.PARAMETER_FORM.CANONICAL
    message.parameters['cov_form'] = vector_normal.COVARIANCE_FORM.COMMON
    message.parameters['wmean'] = [message.parameters['wmean']]
   
    return message.parameters
