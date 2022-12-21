import numpy as np
import scipy as sci

from copy import deepcopy


def _kl_iowm_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(m_scale_s, kl_args):
    # Obtain relevant parameters
    m_scale_s = m_scale_s[0]

    # Calculate kl divergence
    tmp_c1 = 0.5*m_scale_s*kl_args['m_shape']*kl_args['kappa']

    tmp_c2 = -np.real(sci.special.loggamma(kl_args['m_shape']))
    #tmp_c2 = -np.real(sci.special.loggamma(kl_args['m_shape']))
    tmp_c2 += m_scale_s*kl_args['m_shape']/kl_args['m_scale']-kl_args['m_shape']

    tmp_c3 = sci.special.digamma(kl_args['m_shape'])
    #tmp_c3 = sci.special.digamma(kl_args['m_shape'])
    tmp_c3 = tmp_c3*(kl_args['m_shape'] - kl_args['m_shape'] - 1)

    tmp_c4 = -np.log(m_scale_s)
    tmp_c4 = tmp_c4*(kl_args['m_shape'])
    
    kl = tmp_c1+tmp_c2+tmp_c3+tmp_c4

    return kl

def kl_iowm_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(vec, kl_args):
    # Obtain relevant parameters
    m_shape_s, m_scale_s = vec

    # Calculate kl divergence
    tmp_c1 = 0.5*m_scale_s*m_shape_s*kl_args['kappa']

    tmp_c2 = -np.real(sci.special.loggamma(m_shape_s))
    #tmp_c2 = -np.real(sci.special.loggamma(kl_args['m_shape']))
    tmp_c2 += m_scale_s*m_shape_s/kl_args['m_scale']-m_shape_s

    tmp_c3 = sci.special.digamma(m_shape_s)
    #tmp_c3 = sci.special.digamma(kl_args['m_shape'])
    tmp_c3 = tmp_c3*(m_shape_s - kl_args['m_shape'] - 0.5)

    tmp_c4 = -np.log(m_scale_s)
    tmp_c4 = tmp_c4*(kl_args['m_shape'] + 0.5)
    
    kl = tmp_c1+tmp_c2+tmp_c3+tmp_c4

    return kl**2


def kl_optimization_iowm(parameter_set, inplace=False):
    """Estimate parameter_set dictionary by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (list) :
            Dictionary containing estimated parameter_set components
    """

    from pypgm.factor import FACTOR_TYPE, vector_normal, scalar_normal

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set = deepcopy(parameter_set)

    # Obtain relevant parameters
    input_component = parameter_set['input']
    output_component = parameter_set['output']
    weight_component = parameter_set['weight']
    modval_component = parameter_set['modval']

    if (input_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        output_component['ftype'] == FACTOR_TYPE.SCALAR_NORMAL and
        weight_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        modval_component['ftype'] == FACTOR_TYPE.GAMMA):

        # Obtain relevant input parameters
        vector_normal.canonical_to_standard(input_component, inplace=True)
        i_cov = input_component.pop('cov')
        i_mean = np.array(input_component.pop('mean'))
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_cov = np.array(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov = np.diag(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov = np.diag([i_cov]*len(i_mean))
        else:
            raise Exception('Input component has invalid covariance form attribute')

        # Obtain relevant output parameters
        scalar_normal.canonical_to_standard(output_component, inplace=True)
        o_var = output_component.pop('var')
        o_mean = output_component.pop('mean')

        # Obtain relevant weight parameters
        vector_normal.canonical_to_standard(weight_component, inplace=True)
        w_cov = weight_component.pop('cov')
        w_mean = np.array(weight_component.pop('mean'))
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_cov = np.array(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov = np.diag(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov = np.diag([w_cov]*len(w_mean))
        else:
            raise Exception('Weight component has invalid covariance form attribute')

        # Obtain relevant modval parameters
        m_shape = modval_component['shape']
        m_scale = modval_component['scale']

        # Create arguments to be sent to kl divergence
        kappa = o_var + o_mean**2 - 2*o_mean*i_mean.dot(w_mean)
        kappa += sci.trace(w_cov.dot(i_cov))
        kappa += i_mean.dot(w_cov).dot(i_mean)
        kappa += w_mean.dot(i_cov).dot(w_mean)
        kappa += (w_mean.dot(i_mean))**2

        kl_args = {
            'kappa': kappa,
            'm_shape': m_shape+0.5,
            'm_scale': m_scale,
        }
        
        # Find parameters minimizing kl divergence
        bounds = [[1e-10,None]]
        optimizer = sci.optimize.minimize(
            _kl_iowm_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma,
            [m_scale], kl_args, tol=1e-10, bounds=bounds, method = 'L-BFGS-B')
        opt_vec = optimizer.x

        # Update parameter dictionary
        parameter_set['modval']['shape'] = m_shape+0.5
        parameter_set['modval']['scale'] = opt_vec[0]
        parameter_set.pop('input')
        parameter_set.pop('output')
        parameter_set.pop('weight')

    else:
        raise Exception('Given components is of incompatible factor types')

    return parameter_set


def kl_optimization_m(parameter_set, inplace=False):
    """Estimate parameter_set dictionary for modval by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (list) :
            Dictionary containing estimated parameter_set components
    """

    from pypgm.factor import FACTOR_TYPE, vector_normal, scalar_normal

    # Create copy of input parameter_set if necessary
    if not(inplace):
        parameter_set = deepcopy(parameter_set)

    # Obtain relevant parameters
    input_component = parameter_set['input']
    output_component = parameter_set['output']
    weight_component = parameter_set['weight']
    modval_component = parameter_set['modval']

    if (input_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        output_component['ftype'] == FACTOR_TYPE.SCALAR_NORMAL and
        weight_component['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        modval_component['ftype'] == FACTOR_TYPE.GAMMA):

        # Obtain parameters from parameter dictionary
        dim = len(input_component['wmean'])
        mat_ind = np.triu_indices(dim)
        i_wmean = np.array(input_component['wmean'])
        i_prec = np.array(input_component['prec'])
        o_wmean = output_component['wmean']
        o_prec = output_component['prec']
        w_wmean = np.array(weight_component['wmean'])
        w_prec = np.array(weight_component['prec'])
        m_shape = modval_component['shape']
        m_scale = modval_component['scale']
        
        # Obtain relevant input parameters
        vector_normal.canonical_to_standard(input_component, inplace=True)
        i_cov = input_component.pop('cov')
        i_mean = np.array(input_component.pop('mean'))
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_cov = np.array(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov = np.diag(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov = np.diag([i_cov]*len(i_mean))
        else:
            raise Exception('Input component has invalid covariance form attribute')

        # Obtain relevant output parameters
        scalar_normal.canonical_to_standard(output_component, inplace=True)
        o_var = output_component.pop('var')
        o_mean = output_component.pop('mean')

        # Obtain relevant weight parameters
        vector_normal.canonical_to_standard(weight_component, inplace=True)
        w_cov = weight_component.pop('cov')
        w_mean = np.array(weight_component.pop('mean'))
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_cov = np.array(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov = np.diag(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov = np.diag([w_cov]*len(w_mean))
        else:
            raise Exception('Weight component has invalid covariance form attribute')

        # Create arguments to be sent to kl divergence
        kappa = o_var + o_mean**2 - 2*o_mean*i_mean.dot(w_mean)
        kappa += sci.trace(w_cov.dot(i_cov))
        kappa += i_mean.dot(w_cov).dot(i_mean)
        kappa += w_mean.dot(i_cov).dot(w_mean)
        kappa += (w_mean.dot(i_mean))**2
        
        # Find parameters minimizing kl divergence
        m_shape_s = m_shape+0.5
        m_scale_s = 1./m_scale + 0.5*kappa
        m_scale_s = 1./m_scale_s

        # Update parameter dictionary
        parameter_set['modval']['shape'] = m_shape_s
        parameter_set['modval']['scale'] = m_scale_s
        #parameter_set.pop('input')
        #parameter_set.pop('output')
        #parameter_set.pop('weight')

    else:
        raise Exception('Given components is of incompatible factor types')

    return parameter_set
