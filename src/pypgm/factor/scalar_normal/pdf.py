import numpy as np
import scipy as sci


def calculate_pdf(eval_values, **factor_parameters):
    """Function to calculate pdf for given factor parameters
    Args :
        eval_values (list):
            List of values where pdf should be evaluated
        factor_parameters (dict):
            Dictionary containing parameters
    Returns :
        (list) :
            Resulting values describing pdf values
    """

    # Obtain relevant parameters
    wmean = factor_parameters['wmean']
    prec = factor_parameters['prec']
    var = 1./prec
    mean = wmean*var

    # Calculate pdf values
    pdf = sci.stats.norm.pdf(eval_values, mean, np.sqrt(var))

    # Rescale calculated pdf values
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)

    return pdf

