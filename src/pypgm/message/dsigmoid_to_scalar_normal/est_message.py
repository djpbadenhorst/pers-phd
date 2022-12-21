def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from dsigmoid factor to scalar_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of dsigmoid factor sending message
        to_factor_parameters (dict) :
            Parameters of scalar_normal factor receiving message
        inc_message_parameters (dict) :
            Parameters of incoming dsigmoid factor
        out_message_parameters (dict) :
            Parameters of outgoing scalar_normal factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated dsigmoid message
    """

    from pypgm.factor import Factor, dsigmoid

    # Estimate outgoing message with transformation
    if set(out_message_parameters['vars']) == set(from_factor_parameters['input_vars']):
        parameter_set = dsigmoid.estimate_input_component(from_factor_parameters, inplace=True)
        message = Factor(**parameter_set['input'])
        message.divide_by(Factor(**inc_message_parameters['input']))

    elif set(out_message_parameters['vars']) == set(from_factor_parameters['output_vars']):
        parameter_set = dsigmoid.estimate_output_component(from_factor_parameters, inplace=True)
        message = Factor(**parameter_set['output'])
        message.divide_by(Factor(**inc_message_parameters['output']))

    else:
        raise Exception("Domains of given factors are incompatible")
   
    return message.parameters

