import numpy as np
import scipy as sci

from copy import deepcopy


def kl_iow_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma(vec, kl_args):
    from pypgm.factor import vector_normal

    # Obtain variables to recreate estimated parameters from vector
    dim = kl_args['dim']
    mat_ind = kl_args['mat_ind']

    # Recreate estimated mean parameters from vector
    i_mean_s = vec[0:dim]
    o_mean_s = vec[dim]
    w_mean_s = vec[dim+1:2*dim+1]

    # Recreate estimated input covariance attribute from vector
    ind = 2*dim+1
    if kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        i_lmat_s = np.zeros([dim,dim])
        i_lmat_s[mat_ind] = vec[ind:ind+len(mat_ind[0])]
        i_cov_s = (i_lmat_s.T).dot(i_lmat_s)
        ind += len(mat_ind[0])
    elif kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        i_cov_s = np.diag((vec[ind:ind+dim])**2)
        ind += dim
    elif kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        i_cov_s = np.diag([vec[ind]**2]*dim)
        ind += 1

    # Recreate estimated output variance attribute from vector
    o_var_s = vec[ind]**2
    ind += 1

    # Recreate estimated weight covariance attribute from vector
    if kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
        w_lmat_s = np.zeros([dim,dim])
        w_lmat_s[mat_ind] = vec[ind:ind+len(mat_ind[0])]
        w_cov_s = (w_lmat_s.T).dot(w_lmat_s)
        ind += len(mat_ind[0])
    elif kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
        w_cov_s = np.diag((vec[ind:ind+dim])**2)
        ind += dim
    elif kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
        w_cov_s = np.diag([vec[ind]**2]*dim)
        ind += 1

    # Calculate kl divergence
    tmp_c1 = -np.log(sci.linalg.det(i_cov_s))
    tmp_c1 += -np.log(o_var_s)
    tmp_c1 += -np.log(sci.linalg.det(w_cov_s))

    tmp_c2 = (o_var_s + o_mean_s**2 - 2*o_mean_s*i_mean_s.dot(w_mean_s))
    tmp_c2 += sci.trace(w_cov_s.dot(i_cov_s)) + (i_mean_s.dot(w_mean_s))**2
    tmp_c2 += i_mean_s.dot(w_cov_s).dot(i_mean_s) + w_mean_s.dot(i_cov_s).dot(w_mean_s)
    tmp_c2 = kl_args['m_scale']*(kl_args['m_shape']-1)*tmp_c2

    tmp_c3 = sci.trace(i_cov_s.dot(kl_args['i_prec']))
    tmp_c3 += i_mean_s.dot(kl_args['i_prec']).dot(i_mean_s)
    tmp_c3 += -2*i_mean_s.dot(kl_args['i_wmean'])

    tmp_c4 = (o_var_s + o_mean_s**2)*kl_args['o_prec']
    tmp_c4 += -2*o_mean_s*kl_args['o_wmean']

    tmp_c5 = sci.trace(w_cov_s.dot(kl_args['w_prec']))
    tmp_c5 += w_mean_s.dot(kl_args['w_prec']).dot(w_mean_s)
    tmp_c5 += -2*w_mean_s.dot(kl_args['w_wmean'])

    kl = tmp_c1+tmp_c2+tmp_c3+tmp_c4+tmp_c5

    return kl


def kl_optimization_iow(parameter_set, inplace=False):
    """Estimate parameter_set dictionary by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
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

        # Create arguments to be sent to kl divergence
        kl_args = {
            'dim': len(input_component['wmean']),
            'mat_ind': np.triu_indices(len(input_component['wmean'])),
            'i_cov_form': input_component['cov_form'],
            'i_wmean': np.array(input_component.pop('wmean')),
            'i_prec': np.array(input_component.pop('prec')),
            'o_wmean': np.array(output_component.pop('wmean')),
            'o_prec': np.array(output_component.pop('prec')),
            'w_cov_form': weight_component['cov_form'],
            'w_wmean': np.array(weight_component.pop('wmean')),
            'w_prec': np.array(weight_component.pop('prec')),
            'm_shape': np.array(modval_component['shape']),
            'm_scale': np.array(modval_component['scale']),
        }

        # Obtain initial values from input component
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_cov = sci.linalg.inv(kl_args['i_prec'])
            i_lmat = sci.linalg.cholesky(i_cov)
            i_lmat = i_lmat[kl_args['mat_ind']]
            i_mean = kl_args['i_wmean'].dot(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov = 1./(kl_args['i_prec'])
            i_lmat = np.sqrt(i_cov)
            i_mean = kl_args['i_wmean'].dot(np.diag(i_cov))
            kl_args['i_prec'] = np.diag(kl_args['i_prec'])
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov = 1./(kl_args['i_prec'])
            i_lmat = np.sqrt(i_cov)
            i_mean = kl_args['i_wmean']*i_cov
            kl_args['i_prec'] = np.diag([kl_args['i_prec']]*kl_args['dim'])
        else:
            raise Exception('Input component has invalid covariance form attribute')

        # Obtain initial values from output component
        o_var = 1./kl_args['o_prec']
        o_std = np.sqrt(o_var)
        o_mean = kl_args['o_wmean']*o_var

        # Obtain initial values from weight component
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_cov = sci.linalg.inv(kl_args['w_prec'])
            w_lmat = sci.linalg.cholesky(w_cov)
            w_lmat = w_lmat[kl_args['mat_ind']]
            w_mean = kl_args['w_wmean'].dot(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov = 1./(kl_args['w_prec'])
            w_lmat = np.sqrt(w_cov)
            w_mean = kl_args['w_wmean'].dot(np.diag(w_cov))
            kl_args['w_prec'] = np.diag(kl_args['w_prec'])
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov = 1./(kl_args['w_prec'])
            w_lmat = np.sqrt(w_cov)
            w_mean = kl_args['w_wmean']*w_cov
            kl_args['w_prec'] = np.diag([kl_args['w_prec']]*kl_args['dim'])
        else:
            raise Exception('Weight component has invalid covariance form attribute')

        # Create vector of initial values
        init_vec = np.hstack((i_mean, o_mean, w_mean, i_lmat, o_std, w_lmat))

        # Find parameters minimizing kl divergence
        optimizer = sci.optimize.minimize(
            kl_iow_input_vector_normal_output_scalar_normal_weight_vector_normal_modval_gamma,
            init_vec, kl_args, tol=1e-1000, method='Powell')
        opt_vec = optimizer.x

        # Obtain variables to recreate estimated parameters from vector
        dim = kl_args['dim']
        mat_ind = kl_args['mat_ind']

        # Recreate estimated mean parameters from vector
        i_mean_s = opt_vec[0:dim]
        o_mean_s = opt_vec[dim]
        w_mean_s = opt_vec[dim+1:2*dim+1]

        # Recreate estimated input covariance attribute from vector
        ind = 2*dim+1
        if kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_lmat_s = np.zeros([dim,dim])
            i_lmat_s[mat_ind] = opt_vec[ind:ind+len(mat_ind[0])]
            i_cov_s = ((i_lmat_s.T).dot(i_lmat_s)).tolist()
            ind += len(mat_ind[0])
        elif kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov_s = ((opt_vec[ind:ind+dim])**2).tolist()
            ind += dim
        elif kl_args['i_cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov_s = opt_vec[ind]**2
            ind += 1

        # Recreate estimated output variance attribute from vector
        o_var_s = opt_vec[ind]**2
        ind += 1

        # Recreate estimated weight covariance attribute from vector
        if kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_lmat_s = np.zeros([dim,dim])
            w_lmat_s[mat_ind] = opt_vec[ind:ind+len(mat_ind[0])]
            w_cov_s = ((w_lmat_s.T).dot(w_lmat_s)).tolist()
            ind += len(mat_ind[0])
        elif kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov_s = ((opt_vec[ind:ind+dim])**2).tolist()
            ind += dim
        elif kl_args['w_cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov_s = opt_vec[ind]**2
            ind += 1

        # Update parameter dictionary
        input_component['mean'] = i_mean_s.tolist()
        input_component['cov'] = i_cov_s
        output_component['mean'] = o_mean_s
        output_component['var'] = o_var_s
        weight_component['mean'] = w_mean_s.tolist()
        weight_component['cov'] = w_cov_s
        parameter_set = {
            'input': vector_normal.standard_to_canonical(input_component, inplace=True),
            'output': scalar_normal.standard_to_canonical(output_component, inplace=True),
            'weight': vector_normal.standard_to_canonical(weight_component, inplace=True),
        }

    else:
        raise Exception('Given components is of incompatible factor types')
        
    return parameter_set


def kl_optimization_i(parameter_set, inplace=False):
    """Estimate parameter_set dictionary for input by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
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
     
        # Obtain initial values from output component
        o_mean = o_wmean/o_prec

        # Obtain initial values from weight component
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_cov = sci.linalg.inv(w_prec)
            w_mean = w_wmean.dot(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov = 1./(w_prec)
            w_mean = w_wmean.dot(np.diag(w_cov))
            w_prec = np.diag(w_prec)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov = 1./(w_prec)
            w_mean = w_wmean*w_cov
            w_prec = np.diag([w_prec]*dim)
        else:
            raise Exception('Weight component has invalid covariance form attribute')

        # Calculate new parameters
        tmp_const = m_scale*(m_shape-1)
        i_wmean_s = i_wmean + tmp_const*o_mean*w_mean
        i_prec_s = i_prec + tmp_const*(w_cov + np.outer(w_mean, w_mean))

        # Recreate estimated weight covariance attribute from vector
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            pass
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_prec_s = np.diag(i_prec_s)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_prec_s = np.mean(np.diag(i_prec_s))
        
        # Update parameter dictionary
        input_component['wmean'] = i_wmean_s.tolist()
        input_component['prec'] = i_prec_s.tolist()
        parameter_set['input'] = input_component

    else:
        raise Exception('Given components is of incompatible factor types')
        
    return parameter_set


def kl_optimization_o(parameter_set, inplace=False):
    """Estimate parameter_set dictionary for output by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
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

        # Obtain initial values from input component
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_cov = sci.linalg.inv(i_prec)
            i_mean = i_wmean.dot(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov = 1./(i_prec)
            i_mean = i_wmean.dot(np.diag(i_cov))
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov = 1./(i_prec)
            i_mean = i_wmean*i_cov
        else:
            raise Exception('Input component has invalid covariance form attribute')
        
        # Obtain initial values from weight component
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            w_cov = sci.linalg.inv(w_prec)
            w_mean = w_wmean.dot(w_cov)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_cov = 1./(w_prec)
            w_mean = w_wmean.dot(np.diag(w_cov))
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_cov = 1./(w_prec)
            w_mean = w_wmean*w_cov
        else:
            raise Exception('Weight component has invalid covariance form attribute')

        # Calculate new parameters
        tmp_const = m_scale*(m_shape-1)
        o_wmean_s = o_wmean + tmp_const*i_mean.dot(w_mean)
        o_prec_s = o_prec + tmp_const
        
        # Update parameter dictionary
        output_component['wmean'] = o_wmean_s
        output_component['prec'] = o_prec_s
        parameter_set['output'] = output_component

    else:
        raise Exception('Given components is of incompatible factor types')
        
    return parameter_set


def kl_optimization_w(parameter_set, inplace=False):
    """Estimate parameter_set dictionary for weight by optimizing kl divergence
    Args :
        parameter_set (dict) :
            Dictionary containing parameter_set components
        inplace (bool) :
            Boolean value indicating whether estimation should happen inplace
    Returns :
        (dict) :
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

        # Obtain initial values from input component
        if input_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            i_cov = sci.linalg.inv(i_prec)
            i_mean = i_wmean.dot(i_cov)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            i_cov = 1./(i_prec)
            i_mean = i_wmean.dot(np.diag(i_cov))
            i_prec = np.diag(i_prec)
        elif input_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            i_cov = 1./(i_prec)
            i_mean = i_wmean*i_cov
            i_prec = np.diag([i_prec]*dim)
        else:
            raise Exception('Input component has invalid covariance form attribute')
        
        # Obtain initial values from output component
        o_mean = o_wmean/o_prec

        # Calculate new parameters
        tmp_const = m_scale*(m_shape-1)
        w_wmean_s = w_wmean + tmp_const*o_mean*i_mean
        w_prec_s = w_prec + tmp_const*(i_cov + np.outer(i_mean, i_mean))

        # Recreate estimated weight covariance attribute from vector
        if weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.FULL:
            pass
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.DIAGONAL:
            w_prec_s = np.diag(w_prec_s)
        elif weight_component['cov_form'] == vector_normal.COVARIANCE_FORM.COMMON:
            w_prec_s = np.mean(np.diag(w_prec_s))
        
        # Update parameter dictionary
        weight_component['wmean'] = w_wmean_s.tolist()
        weight_component['prec'] = w_prec_s.tolist()
        parameter_set['weight'] = weight_component

    else:
        raise Exception('Given components is of incompatible factor types')
        
    return parameter_set
