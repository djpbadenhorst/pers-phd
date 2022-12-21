import numpy as np
import scipy as sci

from scipy import stats


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
    shape = factor_parameters['shape']
    scale = factor_parameters['scale']

    # Calculate pdf values
    pdf =  sci.stats.gamma.pdf(eval_values, shape, scale=scale)

    # Rescale calculated pdf values
    if pdf[0] == np.inf:
        pdf[0] = 1e10
        
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)

    return pdf

