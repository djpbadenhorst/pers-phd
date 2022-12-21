import itertools

import numpy as np


def _iterative_calculation(vector, eval_values, desired_dim, callback, **callback_parameters):
    if len(vector) == desired_dim:
        tmp_value = callback(vector, **callback_parameters)

        return tmp_value
    else:
        tmp_value = np.sum(
            [_iterative_calculation(np.append(vector, tmp_val),
                                    eval_values, desired_dim,
                                    callback, **callback_parameters)
             for tmp_val in eval_values])

        return tmp_value


def input_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
        i_eval_values, m_eval_values, **factor_parameters):

    from pypgm.factor import vector_normal, scalar_normal
        
    # Obtain relevant input parameters
    input_component = vector_normal.to_standard(factor_parameters['input'])
    dim = len(input_component['mean'])
    i_cov = input_component['cov']
    i_mean = np.array(input_component['mean'])
    if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_cov = np.array(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_cov = np.diag(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_cov = np.diag([i_cov]*len(i_mean))
    else:
        raise Exception('Input component has invalid covariance form attribute')
    i_prec = np.linalg.inv(i_cov)

    # Obtain relevant output parameters
    output_component = scalar_normal.to_standard(factor_parameters['output'])
    o_var = output_component['var']
    o_mean = output_component['mean']

    # Obtain relevant weight parameters
    weight_component = vector_normal.to_standard(factor_parameters['weight'])
    w_cov = weight_component['cov']
    w_mean = np.array(weight_component['mean'])
    if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_cov = np.array(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_cov = np.diag(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_cov = np.diag([w_cov]*len(w_mean))
    else:
        raise Exception('Weight component has invalid covariance form attribute')

    # Obtain relevant modval parameters
    m_shape = np.array(factor_parameters['modval']['shape'])
    m_scale = np.array(factor_parameters['modval']['scale'])

    # Function to calculate relevant posterior
    def _posterior(vec, **pars):
        i_vec = vec[:-1]
        m_val = vec[-1]

        m_prior = ((m_val)**(pars['m_shape'] - 1))
        m_prior = m_prior * (np.exp(-1./pars['m_scale'] * m_val))

        if m_val == 0:
            m_val = 1e-10
            
        tmp_c = (1./m_val + pars['o_var'] +
                     i_vec.dot(pars['w_cov']).dot(i_vec))

        post = (pars['o_mean'] - pars['w_mean'].dot(i_vec))**2
        post = np.exp(-0.5 * post / tmp_c)
        post = post * m_prior * (tmp_c)**(-0.5)

        return post

    
    # Create dictionary of parameters to be sent to _posterior
    pars = {}
    pars.update({'o_mean': o_mean})
    pars.update({'o_var': o_var})
    pars.update({'w_mean': w_mean})
    pars.update({'w_cov': w_cov})
    pars.update({'m_shape': m_shape})
    pars.update({'m_scale': m_scale})

    # Calculate prior and interaction
    count = 0
    prior = np.zeros(len(i_eval_values)**dim)
    interaction = np.zeros(len(i_eval_values)**dim)
    for i_vec in itertools.product(*([i_eval_values] * dim)):
        tmp_c = _iterative_calculation(np.array(i_vec),
                                       m_eval_values,
                                       dim + 1,
                                       _posterior,
                                       **pars)
        interaction[count] = tmp_c
        tmp_c = (i_vec-i_mean).dot(i_prec).dot(i_vec-i_mean)
        tmp_c = np.exp(-0.5 * tmp_c)
        prior[count] = tmp_c
        count = count + 1
        
    # Calculate posterior pdf by multiplying prior by interaction term
    pdf = interaction * prior
    pdf = np.reshape(pdf, [len(i_eval_values)] * dim)
    interaction = np.reshape(interaction, [len(i_eval_values)] * dim)

    # Rescale pdf and interaction
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)
    if np.max(interaction) != 0:
        interaction = interaction/np.max(interaction)

    return [interaction, pdf]


def output_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
        o_eval_values, w_eval_values, m_eval_values, **factor_parameters):

    from pypgm.factor import vector_normal, scalar_normal
    
    # Obtain relevant input parameters
    input_component = vector_normal.to_standard(factor_parameters['input'])
    dim = len(input_component['mean'])
    i_cov = input_component['cov']
    i_mean = np.array(input_component['mean'])
    if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_cov = np.array(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_cov = np.diag(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_cov = np.diag([i_cov]*len(i_mean))
    else:
        raise Exception('Input component has invalid covariance form attribute')

    # Obtain relevant output parameters
    output_component = scalar_normal.to_standard(factor_parameters['output'])
    o_var = output_component['var']
    o_mean = output_component['mean']

    # Obtain relevant weight parameters
    weight_component = vector_normal.to_standard(factor_parameters['weight'])
    w_cov = weight_component['cov']
    w_mean = np.array(weight_component['mean'])
    if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_cov = np.array(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_cov = np.diag(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_cov = np.diag([w_cov]*len(w_mean))
    else:
        raise Exception('Weight component has invalid covariance form attribute')
    w_prec = np.linalg.inv(w_cov)

    # Obtain relevant modval parameters
    m_shape = np.array(factor_parameters['modval']['shape'])
    m_scale = np.array(factor_parameters['modval']['scale'])

    # Function to calculate relevant posterior
    def _posterior(vec, **pars):
        o_val = vec[0]
        m_val = vec[1]
        w_vec = vec[2:]

        m_prior = ((m_val)**(pars['m_shape'] - 1))
        m_prior = m_prior * (np.exp(-1./pars['m_scale'] * m_val))

        if m_val == 0:
            m_val = 1e-10

        w_prior = (w_vec - pars['w_mean']).dot(pars['w_prec'])
        w_prior = w_prior.dot(w_vec - pars['w_mean'])
        w_prior = np.exp(-0.5 * w_prior)
            
        tmp_c = (1./m_val + w_vec.dot(pars['i_cov']).dot(w_vec))

        post = (o_val - pars['i_mean'].dot(w_vec))**2
        post = np.exp(-0.5 * post / tmp_c)
        post = post * m_prior * w_prior * (tmp_c)**(-0.5)

        return post

    
    # Create dictionary of parameters to be sent to _posterior
    pars = {}
    pars.update({'i_mean': i_mean})
    pars.update({'i_cov': i_cov})
    pars.update({'w_mean': w_mean})
    pars.update({'w_prec': w_prec})
    pars.update({'m_shape': m_shape})
    pars.update({'m_scale': m_scale})

    # Calculate prior and interaction
    count_1 = 0
    count_2 = 0
    prior = np.zeros(len(o_eval_values))
    interaction = np.zeros(len(o_eval_values) * len(m_eval_values))
    for o_val in o_eval_values:
        tmp_c = np.exp(-0.5 * (o_val - o_mean)**2 / o_var)
        prior[count_1] = tmp_c
        count_1 = count_1 + 1
        for m_val in m_eval_values:
            tmp_c = _iterative_calculation(np.array(
                [o_val, m_val]), w_eval_values, dim + 2, _posterior, **pars)
            interaction[count_2] = tmp_c
            count_2 = count_2 + 1

    # Calculate posterior pdf by multiplying prior by interaction term
    interaction = np.sum(np.reshape(interaction, (len(o_eval_values), len(m_eval_values))), 1)
    pdf = interaction * prior

    # Rescale pdf and interaction
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)
    if np.max(interaction) != 0:
        interaction = interaction/np.max(interaction)

    return [interaction, pdf]


def weight_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
        w_eval_values, m_eval_values, **factor_parameters):

    from pypgm.factor import vector_normal, scalar_normal
    
    # Obtain relevant input parameters
    input_component = vector_normal.to_standard(factor_parameters['input'])
    dim = len(input_component['mean'])
    i_cov = input_component['cov']
    i_mean = np.array(input_component['mean'])
    if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_cov = np.array(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_cov = np.diag(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_cov = np.diag([i_cov]*len(i_mean))
    else:
        raise Exception('Input component has invalid covariance form attribute')

    # Obtain relevant output parameters
    output_component = scalar_normal.to_standard(factor_parameters['output'])
    o_var = output_component['var']
    o_mean = output_component['mean']

    # Obtain relevant weight parameters
    weight_component = vector_normal.to_standard(factor_parameters['weight'])
    w_cov = weight_component['cov']
    w_mean = np.array(weight_component['mean'])
    if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_cov = np.array(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_cov = np.diag(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_cov = np.diag([w_cov]*len(w_mean))
    else:
        raise Exception('Weight component has invalid covariance form attribute')
    w_prec = np.linalg.inv(w_cov)

    # Obtain relevant modval parameters
    m_shape = np.array(factor_parameters['modval']['shape'])
    m_scale = np.array(factor_parameters['modval']['scale'])

    # Function to calculate relevant posterior
    def _posterior(vec, **pars):
        w_vec = vec[:-1]
        m_val = vec[-1]

        m_prior = ((m_val)**(pars['m_shape'] - 1))
        m_prior = m_prior * (np.exp(-1./pars['m_scale'] * m_val))

        if m_val == 0:
            m_val = 1e-10
            
        tmp_c = (1./m_val + pars['o_var'] +
                 w_vec.dot(pars['i_cov']).dot(w_vec))

        post = (pars['o_mean'] - pars['i_mean'].dot(w_vec))**2
        post = np.exp(-0.5 * post / tmp_c)
        post = post * m_prior * (tmp_c)**(-0.5)

        return post

    
    # Create dictionary of parameters to be sent to _posterior
    pars = {}
    pars.update({'i_mean': i_mean})
    pars.update({'i_cov': i_cov})
    pars.update({'o_mean': o_mean})
    pars.update({'o_var': o_var})
    pars.update({'m_shape': m_shape})
    pars.update({'m_scale': m_scale})

    # Calculate prior and interaction
    count = 0
    prior = np.zeros(len(w_eval_values)**dim)
    interaction = np.zeros(len(w_eval_values)**dim)
    for w_vec in itertools.product(*([w_eval_values] * dim)):
        tmp_c = _iterative_calculation(
            np.array(w_vec), m_eval_values, dim + 1, _posterior, **pars)
        interaction[count] = tmp_c
        tmp_c = np.exp(-0.5 *(w_vec - w_mean).dot(w_prec).dot(w_vec - w_mean))
        prior[count] = tmp_c
        count = count + 1

    # Calculate posterior pdf by multiplying prior by interaction term
    pdf = prior * interaction
    pdf = np.reshape(pdf, [len(w_eval_values)] * dim)
    interaction = np.reshape(interaction, [len(w_eval_values)] * dim)

    # Rescale pdf and interaction
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)
    if np.max(interaction) != 0:
        interaction = interaction/np.max(interaction)

    return [interaction, pdf]


def modval_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
        w_eval_values, m_eval_values, **factor_parameters):
    
    from pypgm.factor import vector_normal, scalar_normal
    
    # Obtain relevant input parameters
    input_component = vector_normal.to_standard(factor_parameters['input'])
    dim = len(input_component['mean'])
    i_cov = input_component['cov']
    i_mean = np.array(input_component['mean'])
    if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_cov = np.array(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_cov = np.diag(i_cov)
    elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_cov = np.diag([i_cov]*len(i_mean))
    else:
        raise Exception('Input component has invalid covariance form attribute')

    # Obtain relevant output parameters
    output_component = scalar_normal.to_standard(factor_parameters['output'])
    o_var = output_component['var']
    o_mean = output_component['mean']

    # Obtain relevant weight parameters
    weight_component = vector_normal.to_standard(factor_parameters['weight'])
    w_cov = weight_component['cov']
    w_mean = np.array(weight_component['mean'])
    if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_cov = np.array(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_cov = np.diag(w_cov)
    elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_cov = np.diag([w_cov]*len(w_mean))
    else:
        raise Exception('Weight component has invalid covariance form attribute')
    w_prec = np.linalg.inv(w_cov)

    # Obtain relevant modval parameters
    m_shape = np.array(factor_parameters['modval']['shape'])
    m_scale = np.array(factor_parameters['modval']['scale'])

    # Function to calculate relevant posterior
    def _posterior(vec, **pars):
        m_val = vec[0]
        w_vec = vec[1:]

        w_prior = np.exp(-0.5 * (w_vec - pars['w_mean']).dot(
            pars['w_prec']).dot(w_vec - pars['w_mean']))

        if m_val == 0:
            m_val = 1e-10
            
        tmp_c = (1./m_val + pars['o_var'] +
                 w_vec.dot(pars['i_cov']).dot(w_vec))

        post = (pars['o_mean'] - pars['i_mean'].dot(w_vec))**2
        post = np.exp(-0.5 * post / tmp_c)
        post = post * w_prior * (tmp_c)**(-0.5)

        return post

    
    # Create dictionary of parameters to be sent to _posterior
    pars = {}
    pars.update({'i_mean': i_mean})
    pars.update({'i_cov': i_cov})
    pars.update({'o_mean': o_mean})
    pars.update({'o_var': o_var})
    pars.update({'w_mean': w_mean})
    pars.update({'w_prec': w_prec})

    # Create empty list to be used to calculate prior and posterior on modval
    count = 0
    prior = np.zeros(len(m_eval_values))
    interaction = np.zeros(len(m_eval_values))
    for m_val in m_eval_values:        
        tmp_c = _iterative_calculation(np.array([m_val]), w_eval_values,
                                       dim + 1, _posterior, **pars)
        interaction[count] = tmp_c
        tmp_c = ((m_val)**(m_shape - 1)) * (np.exp(-1./m_scale * m_val))
        prior[count] = tmp_c
        count = count + 1

    # Calculate posterior pdf by multiplying prior by interaction term
    pdf = interaction * prior
    
    # Rescale pdf and interaction
    if np.max(pdf) != 0:
        pdf = pdf/np.max(pdf)
    if np.max(interaction) != 0:
        interaction = interaction/np.max(interaction)

    return [interaction, pdf]


def calculate_pdf(i_eval_values, o_eval_values, w_eval_values, m_eval_values,
                  component, **factor_parameters):
    """Function to calculate pdf for given parameters
    Args :
        i_eval_values (list):
            List of values at which to evaluate input variable
        o_eval_values (list):
            List of values at which to evaluate modval
        w_eval_values (list):
            List of values at which to evaluate input variable
        m_eval_values (list):
            List of values at which to evaluate modval
        component (str):
            String indicating which component to calculate posterior on
        factor_parameters (dict):
            Dictionary containing parameters
    Returns :
        (list) :
            Resulting values describing pdf values on requested component
    """
    
    from pypgm.factor import FACTOR_TYPE

    if (factor_parameters['input']['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        factor_parameters['output']['ftype'] == FACTOR_TYPE.SCALAR_NORMAL and
        factor_parameters['weight']['ftype'] == FACTOR_TYPE.VECTOR_NORMAL and
        factor_parameters['modval']['ftype'] == FACTOR_TYPE.GAMMA):

        # Calculate pdf on input component
        if component == 'input':
            pdf = input_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
                i_eval_values, m_eval_values, **factor_parameters)

        # Calculate pdf on output component
        elif component == 'output':
            pdf = output_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
                o_eval_values, w_eval_values, m_eval_values, **factor_parameters)

        # Calculate pdf on weight component
        elif component == 'weight':
            pdf = weight_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
                w_eval_values, m_eval_values, **factor_parameters)

        # Calculate pdf on modval component
        elif component == 'modval':
            pdf = modval_pdf_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(
                w_eval_values, m_eval_values, **factor_parameters)
        
        else:
            raise Exception('Given component variable is invalid')
        
    else:
        raise Exception('Given components is of incompatible factor types')

    return pdf
            








