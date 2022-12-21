import numpy as np
import scipy as sci

from copy import deepcopy


def estimate_input_component(parameter_set, number_samples = 10000, inplace=False):
    """Sample from output component and transform into samples from input component
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        number_samples (int) :
            Number of samples to be used to estimate input component
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
            Dictionary containing parameter_set components
    """

    from pypgm.factor import scalar_normal

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set = deepcopy(parameter_set)
        
    # Obtain relevant parameters
    input_component = parameter_set['input']
    i_wmean = input_component.pop('wmean')
    i_prec = input_component.pop('prec')
    output_component = parameter_set['output']    
    scalar_normal.canonical_to_standard(output_component, inplace=True)
    o_mean = output_component.pop('mean')
    o_var = output_component.pop('var')

    # Sample from output component
    samples = sci.stats.norm.rvs(o_mean, np.sqrt(o_var), size=number_samples)
    samples[samples>1-1e-6] = 1-1e-6
    samples[samples<1e-6] = 1e-6

    # Transform to samples from input component
    trans_samples = np.log(samples/(1-samples))

    # Obtain parameters from transformed samples
    trans_mean, trans_var = [np.mean(trans_samples), np.var(trans_samples)]

    # Update parameter dictionary
    parameter_set['input']['wmean'] = trans_mean/trans_var + i_wmean
    parameter_set['input']['prec'] = 1./trans_var + i_prec
    parameter_set.pop('output')
    
    return parameter_set


def estimate_output_component(parameter_set, number_samples = 10000, inplace=False):
    """Sample from input component and transform into samples from output component
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        number_samples (int) :
            Number of samples to be used to estimate output component
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
            Dictionary containing estimated parameter_set components
    """

    from pypgm.factor import scalar_normal

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set = deepcopy(parameter_set)
        
    # Obtain relevant parameters
    input_component = parameter_set['input']
    scalar_normal.canonical_to_standard(input_component, inplace=True)
    i_mean = input_component.pop('mean')
    i_var = input_component.pop('var')
    output_component = parameter_set['output']
    o_wmean = output_component.pop('wmean')
    o_prec = output_component.pop('prec')

    # Sample from input component
    samples = sci.stats.norm.rvs(i_mean, np.sqrt(i_var), size=number_samples)

    # Transform to samples from output component
    trans_samples = 1./(1+np.exp(-samples))
    trans_mean, trans_var = [np.mean(trans_samples), np.var(trans_samples)+1e-6]

    # Update parameter dictionary
    parameter_set['output']['wmean'] = trans_mean/trans_var + o_wmean
    parameter_set['output']['prec'] = 1./trans_var + o_prec
    parameter_set.pop('input')
    
    return parameter_set
