import numpy as np

from copy import deepcopy


def align_component_with_variables(component, variables, inplace=False):
    """Aligns component with the given variable set
    Args :
        component (dict) :
            Dictionary containing parameters of single component
        variables (list) :
            List of variables to component is to be aligned with
        inplace (bool) :
            Boolean value indicating whether alignment should happen inplace
    Returns :
        (dict) :
            Dictionary containing aligned component
    """

    from pypgm.factor import FACTOR_TYPE, vector_normal
    
    # Create copy of input parameter_set if necessary
    if not(inplace):
        component = deepcopy(component)

    # Align scalar normal component
    if component['ftype'] == FACTOR_TYPE.SCALAR_NORMAL:
        pass
    
    # Align vector_normal component
    elif component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL:
        var_ind = np.searchsorted(component['vars'], variables)

        # Align component variables vector
        variables = np.array(component['vars'])[var_ind].tolist()
        component['vars'] = variables

        # Align wmean vector
        wmean = np.array(component['wmean'])[var_ind].tolist()
        component['wmean'] = wmean

        # Align precision matrix
        if component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            pass
        
        elif component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            prec = np.array(component['prec'])[var_ind].tolist()
            component['prec'] = prec
        
        elif component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            prec = np.array(component['prec'])
            prec = prec[np.ix_(var_ind, var_ind)].tolist()
            component['prec'] = prec

        else:
            raise Exception('Parameter dictionary has invalid covariance form attribute')

    # Align gamma component
    elif component['ftype'] == FACTOR_TYPE.GAMMA:
        pass

    return component


def multiply_parameter_sets(parameter_set_1, parameter_set_2, inplace=False):
    """Multiply two parameter_set dictionaries
    Args :
        parameter_set_1 (dict) :
            First dictionary containing parameter_set components
        parameter_set_2 (dict) :
            Second dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether multiplication should happen inplace
            Take note that only the first dictionary can be altered
    Returns :
        (dict) :
            Dictionary containing parameter_set components of product
    """

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set_1 = deepcopy(parameter_set_1)

    # Obtain relevant parameters
    input_component = parameter_set_2.get('input')
    output_component = parameter_set_2.get('output')
    weight_component = parameter_set_2.get('weight')
    modval_component = parameter_set_2.get('modval')

    # Set components individually if they are available
    if input_component != None:
        align_component_with_variables(input_component, parameter_set_1['input_vars'], inplace=True)
        parameter_set_1['input'] = input_component
        
    if output_component != None:
        align_component_with_variables(output_component, parameter_set_1['output_vars'], inplace=True)
        parameter_set_1['output'] = output_component
        
    if weight_component != None:
        align_component_with_variables(weight_component, parameter_set_1['weight_vars'], inplace=True)
        parameter_set_1['weight'] = weight_component
        
    if modval_component != None:
        align_component_with_variables(modval_component, parameter_set_1['modval_vars'], inplace=True)
        parameter_set_1['modval'] = modval_component

    return parameter_set_1


def inpl_multiply_parameter_sets(parameter_set_1, parameter_set_2, inplace=False):
    return multiply_parameter_sets(parameter_set_1, parameter_set_2, True)
