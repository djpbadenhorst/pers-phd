def est_message(from_factor_parameters, to_factor_parameters,
                inc_message_parameters, out_message_parameters):
    """Estimate message parameters from dlinear factor to gamma factor
    Args :
        from_factor_parameters (dict) :
            Parameters of dlinear factor sending message
        to_factor_parameters (dict) :
            Parameters of gamma factor receiving message
        inc_message_parameters (dict) :
            Parameters of incoming dlinear factor
        out_message_parameters (dict) :
            Parameters of outgoing gamma factor
    Returns :
        (dict) :
            Dictionary containing parameters of estimated dlinear message
    """

    from pypgm.factor import Factor, dlinear

    # Estimate outgoing message with kl optimization
    if set(out_message_parameters['vars']) == set(from_factor_parameters['modval_vars']):
        #parameter_set = dlinear.kl_optimization_iowm(from_factor_parameters, inplace=True)
        parameter_set = dlinear.kl_optimization_m(from_factor_parameters, inplace=True)
        message = Factor(**parameter_set['modval'])
        message.divide_by(Factor(**inc_message_parameters['modval']))

    else:
        raise Exception("Domains of given factors are incompatible")
   
    return message.parameters

