class Factor(object):
    """Object used to represent Factor"""

    def __init__(self, **factor_parameters):
        from pypgm.factor import FACTOR_FUNCTIONS
        
        # Store function pointers to be used
        self.functions = FACTOR_FUNCTIONS[factor_parameters['ftype']]

        # Store parameters to be used
        self.parameters = factor_parameters

        
    def __str__(self):
        return(self.__indent_str__(1))

    
    def __indent_str__(self, indent_level):
        from utils.sdict import SDict
        
        return SDict(**self.parameters).__indent_str__(indent_level)

    
    def init_parameters(self):
        # Perform error checks on parameters before initialization
        self.functions['pre_init_parameters_check'](
            self.parameters)

        # Initialize parameters and ensure correct format
        self.parameters = self.functions['init_parameters'](
            self.parameters)

        # Perform error checks on parameters after initialization
        self.functions['post_init_parameters_check'](
            self.parameters)

        
    def multiply_by(self, factor):
        # Perform error checks on parameters before multiplication
        self.functions['pre_multiply_by_check'](
            self.parameters, factor.parameters)

        # Multiply Factor object by another
        self.parameters = self.functions['multiply_by'](
            self.parameters, factor.parameters)

        # Perform error checks on parameters after multiplication
        self.functions['post_multiply_by_check'](
            self.parameters)

        
    def divide_by(self, factor):
        # Perform error checks on parameters before division
        self.functions['pre_divide_by_check'](
            self.parameters, factor.parameters)

        # Divide Factor object by another
        self.parameters = self.functions['divide_by'](
            self.parameters, factor.parameters)

        # Perform error checks on parameters after division
        self.functions['post_divide_by_check'](
            self.parameters)

        
    def normalize(self):
        # Perform error checks on parameters before normalization
        self.functions['pre_normalize_check'](
            self.parameters)

        # Normalize Factor object
        self.parameters = self.functions['normalize'](
            self.parameters)

        # Perform error checks on parameters after normalization
        self.functions['post_normalize_check'](
            self.parameters)

        
    def marginalize_to(self, domain):
        # Perform error checks on parameters before marginalization
        self.functions['pre_marginalize_to_check'](
            self.parameters, domain)

        # Marginalize Factor object
        self.parameters = self.functions['marginalize_to'](
            self.parameters, domain)

        # Perform error checks on parameters after marginalization
        self.functions['post_marginalize_to_check'](
            self.parameters)
