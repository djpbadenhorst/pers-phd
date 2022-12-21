import numpy as np

from copy import deepcopy


def marginalize_standard_form(parameters, domain, inplace = False):
    """Marginalize standard parameters to subset of the domain
    Args :
        parameters (dict) :
            Dictionary containing standard parameters
        domain (list) :
            List of variables to marginalize to
        inplace (bool) :
            Boolean value indicating whether marginalization should happen inplace
    Returns :
        (dict) :
            Dictionary containing canonical parameters of marginal
    """

    from pypgm.factor import vector_normal
    
    # Create copy of input parameters if necessary
    if not(inplace):
        parameters = deepcopy(parameters)

    # Obtain relevant parameters
    variables = np.array(parameters['vars'])
    cov_form = parameters['cov_form']
    mean = np.array(parameters['mean'])
    cov = parameters['cov']

    # Determine variables that need to be removed
    sort_ind = variables.argsort()
    var_ind = sort_ind[np.searchsorted(variables, domain, sorter = sort_ind)]

    # Convert standard parameters to canonical parameters
    if cov_form == vector_normal.COVARIANCE_FORM.COMMON:
        variables = variables[var_ind].tolist()
        mean = mean[var_ind].tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.DIAGONAL:
        variables = variables[var_ind].tolist()
        mean = mean[var_ind].tolist()
        cov = np.array(cov)[var_ind].tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.FULL:
        variables = variables[var_ind].tolist()
        mean = mean[var_ind].tolist()
        cov = np.array(cov)[np.ix_(var_ind, var_ind)].tolist()
        
    else:
        raise Exception('Parameter dictionary has invalid covariance form attribute')

    # Update parameter dictionary
    parameters['vars'] = variables
    parameters['mean'] = mean
    parameters['cov'] = cov

    return parameters
