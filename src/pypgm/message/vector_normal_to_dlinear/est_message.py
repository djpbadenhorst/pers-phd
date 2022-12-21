def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from vector_normal factor to dlinear factor
    Args :
        from_factor_parameters (dict) :
            Parameters of vector_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of dlinear factor receiving message
        inc_message_parameters (dict) :
            Parameters of vector_normal factor
        out_message_parameters (dict) :
            Parameters of outgoing dlinear factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated dlinear message
    """

    from pypgm.factor import FACTOR_TYPE, Factor
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Divide message by incoming message
    message.divide_by(Factor(**inc_message_parameters))

    # Create message based on domain of outgoing message
    if out_message_parameters.get('input_vars') != None:
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         input_vars = out_message_parameters['input_vars'],
                         input = message.parameters)
        
    elif out_message_parameters.get('weight_vars') != None:
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         weight_vars = out_message_parameters['weight_vars'],
                         weight = message.parameters)

    else:
        raise Exception("Domains of given factors are incompatible")
   
    return message.parameters

