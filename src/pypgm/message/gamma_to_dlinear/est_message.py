def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from gamma factor to dlinear factor
    Args :
        from_factor_parameters (dict) :
            Parameters of gamma factor sending message
        to_factor_parameters (dict) :
            Parameters of dlinear factor receiving message
        inc_message_parameters (dict) :
            Parameters of gamma factor
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
    if out_message_parameters.get('modval_vars') != None:
        message = Factor(ftype = FACTOR_TYPE.DLINEAR,
                         modval_vars = out_message_parameters['modval_vars'],
                         modval = message.parameters)

    else:
        raise Exception("Domains of given factors are incompatible")
   
    return message.parameters

