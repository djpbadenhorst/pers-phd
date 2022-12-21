def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from scalar_normal factor to scalar_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of scalar_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of scalar_normal factor receiving message
        inc_message_parameters (dict) :
            Parameters of incoming scalar_normal factor
        out_message_parameters (dict) :
            Parameters of outgoing scalar_normal factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated scalar_normal message
    """

    from pypgm.factor import Factor
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Divide message by incoming message
    message.divide_by(Factor(**inc_message_parameters))
   
    return message.parameters

