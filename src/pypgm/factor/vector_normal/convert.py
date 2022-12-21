import numpy as np
import scipy as sci

from copy import deepcopy


def standard_to_canonical(parameters, inplace=False):
    """Convert standard parameters into canonical parameters
    Args :
        parameters (dict) :
            Dictionary containing standard parameters
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing canonical parameters
    """

    from pypgm.factor import vector_normal
    
    # Create copy of input parameters if necessary
    if not(inplace):
        parameters = deepcopy(parameters)

    # Obtain relevant parameters
    cov_form = parameters['cov_form']
    mean = np.array(parameters.pop('mean'))
    cov = parameters.pop('cov')

    # Convert standard parameters to canonical parameters
    if cov_form == vector_normal.COVARIANCE_FORM.COMMON:
        prec = 1./cov
        wmean = (mean*prec).tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.DIAGONAL:
        prec = (1./np.array(cov)).tolist()
        wmean = (mean*np.array(prec)).tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.FULL:
        prec = (sci.linalg.inv(cov)).tolist()
        wmean = (mean.dot(prec)).tolist()
        
    else:
        raise Exception('Parameter dictionary has invalid covariance form attribute')

    # Update parameter dictionary
    parameters['wmean'] = wmean
    parameters['prec'] = prec

    return parameters


def canonical_to_standard(parameters, inplace=False):
    """Convert canonical parameters into standard parameters
    Args :
        parameters (dict) :
            Dictionary containing canonical parameters
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing standard parameters
    """

    from pypgm.factor import vector_normal
    
    # Create copy of input parameters if necessary
    if not(inplace):
        parameters = deepcopy(parameters)

    # Obtain relevant parameters
    cov_form = parameters['cov_form']
    wmean = np.array(parameters.pop('wmean'))
    prec = parameters.pop('prec')
    
    # Convert canonical parameters to standard parameters
    if cov_form == vector_normal.COVARIANCE_FORM.COMMON:
        cov = 1./prec
        mean = (cov*wmean).tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.DIAGONAL:
        cov = (1./np.array(prec)).tolist()
        mean = (wmean*np.array(cov)).tolist()
        
    elif cov_form == vector_normal.COVARIANCE_FORM.FULL:
        cov = (sci.linalg.inv(prec)).tolist()
        mean = (wmean.dot(cov)).tolist()
        
    else:
        raise Exception('Parameter dictionary has invalid covariance form attribute')

    # Update parameter dictionary
    parameters['mean'] = mean
    parameters['cov'] = cov

    return parameters


def to_canonical(parameters, inplace=False):
    """Ensures given parameters are in canonical form
    Args :
        parameters (dict) :
            Dictionary containing parameters of any form
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing canonical parameters
    """

    from pypgm.factor import vector_normal

    # If parameters are in canonical form    
    if parameters['par_form'] == vector_normal.PARAMETER_FORM.CANONICAL:
        if not(inplace):
            parameters = deepcopy(parameters)

    # If parameters are in standard form
    elif parameters['par_form'] == vector_normal.PARAMETER_FORM.STANDARD:
            parameters = standard_to_canonical(parameters, inplace)
            parameters['par_form'] = vector_normal.PARAMETER_FORM.CANONICAL

    else:
        raise Exception('Parameter dictionary has invalid parameter form attribute')

    return parameters


def inpl_to_canonical(parameters):
    return to_canonical(parameters, inplace=True)


def to_standard(parameters, inplace=False):
    """Ensures given parameters are in standard form
    Args :
        parameters (dict) :
            Dictionary containing parameters of any form
        inplace (bool) :
            Boolean value indicating whether conversion should happen inplace
    Returns :
        (dict) :
            Dictionary containing standard parameters
    """

    from pypgm.factor import vector_normal
    
    # If parameters are in standard form    
    if parameters['par_form'] == vector_normal.PARAMETER_FORM.STANDARD:
        if not(inplace):
            parameters = deepcopy(parameters)

    # If parameters are in canonical form
    elif parameters['par_form'] == vector_normal.PARAMETER_FORM.CANONICAL:
            parameters = canonical_to_standard(parameters, inplace)
            parameters['par_form'] = vector_normal.PARAMETER_FORM.STANDARD

    else:
        raise Exception('Parameter dictionary has invalid parameter form attribute')

    return parameters


def inpl_to_standard(parameters):
    return to_standard(parameters, inplace=True)
