def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from scalar_normal factor to vector_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of scalar_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of vector_normal factor receiving message
        inc_message_parameters (dict) :
            Parameters of incoming scalar_normal factor
        out_message_parameters (dict) :
            Parameters of outgoing vector_normal factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated vector_normal message
    """

    from pypgm.factor import FACTOR_TYPE, Factor, vector_normal
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Divide message by incoming message
    message.divide_by(Factor(**inc_message_parameters))

    # Update parameter attribute
    message.parameters['ftype'] = FACTOR_TYPE.VECTOR_NORMAL
    message.parameters['par_form'] = vector_normal.PARAMETER_FORM.CANONICAL
    message.parameters['cov_form'] = vector_normal.COVARIANCE_FORM.COMMON
    message.parameters['wmean'] = [message.parameters['wmean']]
   
    return message.parameters

