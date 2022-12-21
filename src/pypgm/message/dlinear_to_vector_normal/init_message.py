def init_message(from_factor_parameters, to_factor_parameters, domain):
    """Initializes message parameters from dlinear factor to vector_normal factor
    Args :
        from_factor_parameters (dict) :
            Parameters of dlinear factor sending message
        to_factor_parameters (dict) :
            Parameters of vector_normal factor receiving message
        domain (list) :
            List of variable names describing domain of sepset
    Returns :
        (dict) :
            Dictionary containing parameters of initialized vector_normal message
    """

    import numpy as np
    from pypgm.factor import Factor, vector_normal

    # Create message from parameters of factor receiving message
    message = Factor(**to_factor_parameters)
        
    return message.parameters
