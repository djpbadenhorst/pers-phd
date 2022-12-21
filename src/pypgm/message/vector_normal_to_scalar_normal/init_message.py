def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from vector_normal factor to scalar_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of vector_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of scalar_normal factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized scalar_normal message
    """

    from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal, vector_normal
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Marginalize to relevant subset of variables
    message.marginalize_to(domain)

    # Update parameter attribute
    message.parameters['ftype'] = FACTOR_TYPE.SCALAR_NORMAL
    message.parameters['par_form'] = scalar_normal.PARAMETER_FORM.CANONICAL
    message.parameters['wmean'] = message.parameters['wmean'][0]
    cov_form = message.parameters.pop('cov_form')
    
    if cov_form == vector_normal.COVARIANCE_FORM.COMMON:
        pass
    elif cov_form == vector_normal.COVARIANCE_FORM.DIAGONAL:
        message.parameters['prec'] = message.parameters['prec'][0]
        
    elif cov_form == vector_normal.COVARIANCE_FORM.FULL:
        message.parameters['prec'] = message.parameters['prec'][0][0]
    
    return message.parameters
