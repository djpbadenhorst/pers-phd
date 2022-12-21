class Message():
    """Object used to calculate messages"""

    @staticmethod
    def init_message(from_factor_parameters, to_factor_parameters, domain):
        from pypgm.message import FUNCTIONS
        
        # Infer message type and obtain corresponding functions
        mtype = (from_factor_parameters['ftype'] << 8) + to_factor_parameters['ftype']
        functions = FUNCTIONS[mtype]
        
        # Perform error checks on parameters before initialization
        functions['pre_init_message_check'](
            from_factor_parameters, to_factor_parameters, domain)

        # Create parameters for initial message between two Factor objects
        parameters = functions['init_message'](
            from_factor_parameters, to_factor_parameters, domain)

        # Perform error checks on parameters after initialization
        functions['post_init_message_check'](
            parameters)

        return parameters

    
    @staticmethod
    def est_message(from_factor_parameters, to_factor_parameters,
                    inc_message_parameters, out_message_parameters):
        from pypgm.message import FUNCTIONS
        
        # Infer message type and obtain corresponding functions
        mtype = (from_factor_parameters['ftype'] << 8) + to_factor_parameters['ftype']
        functions = FUNCTIONS[mtype]
        
        # Perform error checks on parameters before estimation
        functions['pre_est_message_check'](
            from_factor_parameters, to_factor_parameters,
            inc_message_parameters, out_message_parameters)

        # Create parameters for estimated message between two Factor objects
        parameters = functions['est_message'](
            from_factor_parameters, to_factor_parameters,
            inc_message_parameters, out_message_parameters)
            
        # Perform error checks on parameters after estimation
        functions['post_est_message_check'](
            parameters)

        return parameters
