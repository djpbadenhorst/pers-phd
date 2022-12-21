def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from vector_normal factor to vector_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of vector_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of vector_normal factor receiving message
        inc_message_parameters (dict) :
            Parameters of incoming vector_normal factor
        out_message_parameters (dict) :
            Parameters of outgoing vector_normal factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated vector_normal message
    """

    from pypgm.factor import Factor
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Divide message by incoming message
    message.divide_by(Factor(**inc_message_parameters))
    
    # Marginalize to relevant subset of variables
    message.marginalize_to(out_message_parameters['vars'])
   
    return message.parameters

