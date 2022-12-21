def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from scalar_normal factor to dsigmoid factor
    Args :
        from_factor_parameters (dict) :
            Parameters of scalar_normal factor sending message
        to_factor_parameters (dict) :
            Parameters of dsigmoid factor receiving message
        inc_message_parameters (dict) :
            Parameters of scalar_normal factor
        out_message_parameters (dict) :
            Parameters of outgoing dsigmoid factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated dsigmoid message
    """

    from pypgm.factor import FACTOR_TYPE, Factor
    
    # Create message from parameters of factor sending message
    message = Factor(**from_factor_parameters)

    # Divide message by incoming message
    message.divide_by(Factor(**inc_message_parameters))

    # Create message based on domain of outgoing message
    if out_message_parameters.get('output_vars') != None:
        message = Factor(ftype = FACTOR_TYPE.DSIGMOID,
                         output_vars = out_message_parameters['output_vars'],
                         output = message.parameters)

    elif out_message_parameters.get('input_vars') != None:
        message = Factor(ftype = FACTOR_TYPE.DSIGMOID,
                         input_vars = out_message_parameters['input_vars'],
                         input = message.parameters)

    else:
        raise Exception("Domains of given factors are incompatible")
   
    return message.parameters

