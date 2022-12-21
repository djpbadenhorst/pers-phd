import numpy as np

from copy import deepcopy

    
def align_canonical_forms(parameters_1, parameters_2, inplace=False):
    """Align domains and corresponding attributes of canonical parameter dictionary
    Args :
        parameters_1 (dict) :
            First dictionary containing canonical parameters
        parameters_2 (dict) :
            Second dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether alignment should happen inplace
            Take note that both dictionaries can be altered
    Return:
        list :
            List containing two parameter dictionaries
    """

    from pypgm.factor import vector_normal
    
    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)
        parameters_2 = deepcopy(parameters_2)

    # Obtain index representation of variables
    vars_1 = np.array(parameters_1['vars'])
    vars_2 = np.array(parameters_2['vars'])
    full_vars = np.unique(np.append(vars_1, vars_2))
    full_dim = len(full_vars)
    var_ind_1 = np.searchsorted(full_vars, vars_1)
    var_ind_2 = np.searchsorted(full_vars, vars_2)

    # Align wmean vectors
    wmean_1 = np.zeros(full_dim)
    wmean_2 = np.zeros(full_dim)
    wmean_1[var_ind_1] = parameters_1['wmean']
    wmean_2[var_ind_2] = parameters_2['wmean']
    parameters_1['wmean'] = wmean_1.tolist()
    parameters_2['wmean'] = wmean_2.tolist()

    # Obtain covariance form of parameters
    cov_form = parameters_1['cov_form']

    # Align precision based on covariance form of parameters        
    if cov_form == vector_normal.COVARIANCE_FORM.COMMON:
        pass
        
    elif cov_form == vector_normal.COVARIANCE_FORM.DIAGONAL:
        prec_1 = np.zeros(full_dim)
        prec_2 = np.zeros(full_dim)
        prec_1[var_ind_1] = parameters_1['prec']
        prec_2[var_ind_2] = parameters_2['prec']
        parameters_1['prec'] = prec_1.tolist()
        parameters_2['prec'] = prec_2.tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.FULL:
        prec_1 = np.zeros((full_dim,full_dim))
        prec_2 = np.zeros((full_dim,full_dim))
        prec_1[np.ix_(var_ind_1,var_ind_1)] = parameters_1['prec']
        prec_2[np.ix_(var_ind_2,var_ind_2)] = parameters_2['prec']
        parameters_1['prec'] = prec_1.tolist()
        parameters_2['prec'] = prec_2.tolist()

    else:
        raise Exception('Parameter dictionary has invalid covariance form attribute')

    # Update variables attribute
    parameters_1['vars'] = full_vars.tolist()
    parameters_2['vars'] = full_vars.tolist()

    return [parameters_1, parameters_2]


def multiply_canonical_forms(parameters_1, parameters_2, inplace=False):
    """Multiply two canonical parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing canonical parameters
        parameters_2 (dict) :
            Second dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether multiplication should happen inplace
            Take note that both dictionaries can be altered
    Returns :
        (dict) :
            Dictionary containing canonical parameters of product
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)
        parameters_2 = deepcopy(parameters_2)

    # Ensure parameters are aligned
    align_canonical_forms(parameters_1, parameters_2, inplace=True)

    # Add wmean attributes
    parameters_1['wmean'] = np.array(parameters_1['wmean']) + np.array(parameters_2['wmean'])
    parameters_1['wmean'] = parameters_1['wmean'].tolist()

    # Add precision attributes
    parameters_1['prec'] = np.array(parameters_1['prec']) + np.array(parameters_2['prec'])
    parameters_1['prec'] = parameters_1['prec'].tolist()

    return parameters_1


def inpl_multiply_canonical_forms(parameters_1, parameters_2):
    return multiply_canonical_forms(parameters_1, parameters_2, inplace=True)


def divide_canonical_forms(parameters_1, parameters_2, inplace=False):
    """Divide two canonical parameter dictionaries
    Args :
        parameters_1 (dict) :
            First dictionary containing canonical parameters
        parameters_2 (dict) :
            Second dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether division should happen inplace
            Take note that both dictionaries can be altered
    Returns :
        (dict) :
            Dictionary containing canonical parameters of quotient
    """

    # Create copy of input parameters if necessary
    if not(inplace):
        parameters_1 = deepcopy(parameters_1)
        parameters_2 = deepcopy(parameters_2)

    # Ensure parameters are aligned
    align_canonical_forms(parameters_1, parameters_2, inplace=True)

    # Subtract wmean attributes
    parameters_1['wmean'] = np.array(parameters_1['wmean']) - np.array(parameters_2['wmean'])
    parameters_1['wmean'] = parameters_1['wmean'].tolist()

    # Subtract precision attributes
    parameters_1['prec'] = np.array(parameters_1['prec']) - np.array(parameters_2['prec'])
    parameters_1['prec'] = parameters_1['prec'].tolist()

    return parameters_1


def inpl_divide_canonical_forms(parameters_1, parameters_2):
    return divide_canonical_forms(parameters_1, parameters_2, inplace=True)


def marginalize_canonical_form(parameters, domain, inplace=False):
    """Marginalize canonical parameters to subset of the domain
    Args :
        parameters (dict) :
            Dictionary containing canonical parameters
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
        
    # Covert canonical form to standard form
    vector_normal.canonical_to_standard(parameters, inplace=True)

    # Marginalize standard form
    vector_normal.marginalize_standard_form(parameters, domain, inplace=True)

    # Covert standard form to canonical form
    vector_normal.standard_to_canonical(parameters, inplace=True)

    return parameters


def inpl_marginalize_canonical_form(parameters, domain):
    return marginalize_canonical_form(parameters, domain, inplace=True)
