import itertools

import numpy as np


def calculate_pdf(eval_values, **factor_parameters):
    """Function to calculate pdf for given parameters
    Args :
        eval_values (list):
            List of values where pdf should be evaluated
        factor_parameters (dict):
            Dictionary containing parameters
    Returns :
        (list) :
            Resulting values describing pdf values
    """

    from pypgm.factor import vector_normal
    
    # Obtain relevant parameters
    wmean = np.array(factor_parameters['wmean'])
    prec = factor_parameters['prec']
    dim = len(wmean)
    
    if factor_parameters['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        prec = np.array(prec)
    elif factor_parameters['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        prec = np.diag(prec)
    elif factor_parameters['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        prec = np.diag([prec]*dim)
    else:
        raise Exception('Factor parameters has invalid covariance form attribute')
        
    cov = np.linalg.inv(prec)
    mean = wmean.dot(cov)

    # Calculate pdf values
    count = 0
    pdf = np.zeros(len(eval_values)**dim)
    for vec in itertools.product(*([eval_values] * dim)):
        vec = np.array(vec)
        tmp_c = vec.dot(prec).dot(vec)
        tmp_c += -2*wmean.dot(vec)
        tmp_c += wmean.dot(mean)
        tmp_c = -0.5*tmp_c
        pdf[count] = tmp_c
        count = count+1
    pdf = np.exp(pdf)
    
    # Reshape and rescale calculated pdf values
    pdf = np.reshape(pdf, [len(eval_values)] * dim)
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)

    return pdf

