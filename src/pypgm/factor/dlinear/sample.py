import numpy as np
import scipy as sci


def sample_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
        input_component, output_component, weight_component, modval_component,
        number_samples=1000):

    from pykalman.sqrt.unscented import cholupdate
    
    from pypgm.factor import vector_normal
    
    # Obtain relevant input parameters
    i_wmean = np.array(input_component['wmean'])
    i_dim = len(i_wmean)
    if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_prec = np.diag(np.array([input_component['prec']]*i_dim))
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_prec = np.diag(input_component['prec'])
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_prec = np.array(input_component['prec'])
    else:
        raise Exception('Input component has incompatible covariance form attributes')
    i_ilmat = sci.linalg.cholesky(i_prec)

    # Obtain relevant output parameters
    o_wmean = output_component['wmean']
    o_prec = output_component['prec']
    
    # Obtain relevant weight parameters
    w_wmean = np.array(weight_component['wmean'])
    w_dim = len(w_wmean)
    if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_prec = np.diag(np.array([weight_component['prec']]*w_dim))
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_prec = np.diag(weight_component['prec'])
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_prec = np.array(weight_component['prec'])
    else:
        raise Exception('Weight component has incompatible covariance form attributes')
    w_ilmat = sci.linalg.cholesky(w_prec)

    # Obtain relevant modval parameters
    m_shape = modval_component['shape']
    m_scale = modval_component['scale']
    
    # Create dictionary containing original random samples
    samples = {
        'input_samples': sci.stats.norm.rvs(0, 1, size=(number_samples, i_dim)),
        'output_samples': sci.stats.norm.rvs(0, 1, size=number_samples),
        'weight_samples': sci.stats.norm.rvs(0, 1, size=(number_samples, w_dim)),
        'modval_samples': sci.stats.gamma.rvs(m_shape + 0.5, size=number_samples),
    }

    # Initial samples for Gibbs sampling
    i_samp = i_wmean.dot(sci.linalg.inv(i_prec))
    o_samp = o_wmean/o_prec
    w_samp = w_wmean.dot(sci.linalg.inv(w_prec))
    m_samp = 1e-3

    # Functions to be used for sampling
    def sample_input():
        tmp_rand = samples['input_samples'][count]
        new_i_wmean = (m_samp * o_samp * w_samp) + i_wmean
        new_i_ilmat = cholupdate(i_ilmat, w_samp, m_samp)
        new_i_lmat = sci.linalg.solve_triangular(new_i_ilmat, np.identity(i_dim))
        new_i_cov = new_i_lmat.dot(new_i_lmat.transpose())
        #new_i_cov = sci.linalg.inv(i_prec + p_samp*np.outer(w_samp, w_samp))
        #new_i_lmat = sci.linalg.cholesky(new_i_cov, lower=True)
        new_i_mean = new_i_wmean.dot(new_i_cov)
        i_samp = np.array(new_i_lmat).dot(tmp_rand) + new_i_mean
        return i_samp

    def sample_output():
        tmp_rand = samples['output_samples'][count]
        new_o_var = 1. / (m_samp + o_prec)
        new_o_mean = (i_samp.dot(w_samp) * m_samp + o_wmean) * new_o_var
        o_samp = new_o_mean + tmp_rand * np.sqrt(new_o_var)
        return o_samp

    def sample_weight():
        tmp_rand = samples['weight_samples'][count]
        new_w_wmean = (m_samp * o_samp * i_samp) + w_wmean
        new_w_ilmat = cholupdate(w_ilmat, i_samp, m_samp)
        new_w_lmat = sci.linalg.solve_triangular(new_w_ilmat, np.identity(w_dim))
        new_w_cov = new_w_lmat.dot(new_w_lmat.transpose())
        #new_w_cov = sci.linalg.inv(w_prec + p_samp*np.outer(i_samp, i_samp))
        #new_w_lmat = sci.linalg.cholesky(new_w_cov, lower=True)
        new_w_mean = new_w_wmean.dot(new_w_cov)
        w_samp = np.array(new_w_lmat).dot(tmp_rand) + new_w_mean
        return w_samp

    def sample_modval():
        tmp_rand = samples['modval_samples'][count]
        err_samp = (1./m_scale + 0.5 * (o_samp - w_samp.dot(i_samp))**2)
        m_samp = tmp_rand / err_samp
        return m_samp

    
    # Create dictionary of functions to call for sampling of each variable
    variables = ['input','output','weight','modval']
    functions = {
        'input' : sample_input,
        'output' : sample_output,
        'weight' : sample_weight,
        'modval' : sample_modval,
    }
    
    # Run Gibbs sampling
    for count in range(number_samples):
        for i in np.random.permutation(4):
            if variables[i] == 'input':
                i_samp = functions[variables[i]]()
                samples['input_samples'][count] = i_samp
            elif variables[i] == 'output':
                o_samp = functions[variables[i]]()
                samples['output_samples'][count] = o_samp
            elif variables[i] == 'weight':
                w_samp = functions[variables[i]]()
                samples['weight_samples'][count] = w_samp
            elif variables[i] == 'modval':
                m_samp = functions[variables[i]]()
                samples['modval_samples'][count] = m_samp

    return samples
    

def sample_from_posteriors(parameter_set, number_samples=1000):
    """Sample from posterior of relevant variables
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        number_samples (int) :
            Number of samples to be used to estimate input component
    Returns :
        (dict) :
            Dictionary containing samples from relevant posteriors
    """

    from pypgm.factor import FACTOR_TYPE

    # Obtain relevant parameters
    input_component = parameter_set['input']
    output_component = parameter_set['output']
    weight_component = parameter_set['weight']
    modval_component = parameter_set['modval']

    if (input_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        output_component['ftype'] == FACTOR_TYPE.SCALAR_NORMAL and
        weight_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        modval_component['ftype'] == FACTOR_TYPE.GAMMA):

        samples = sample_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
            input_component, output_component, weight_component, modval_component,
            number_samples=1000)
        
    else:
        raise Exception('Given components is of incompatible factor types')
    
    return samples



