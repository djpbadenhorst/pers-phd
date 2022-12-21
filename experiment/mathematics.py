import ipdb

import numpy as np

EPSILON = 1e-5

def standard_to_canonical(mean, var, **kwargs):
    mean = np.array(mean)
    var = np.array(var)

    if var.ndim == 2:
        wmean = np.zeros(len(mean))
        prec = np.zeros(np.shape(var))

        non_zero_bool = np.diag(var) != 0
        subvec_ind = np.where(non_zero_bool)[0]
        submat_ind = np.ix_(subvec_ind,subvec_ind)

        sub_prec = np.linalg.inv(var[submat_ind])
        sub_wmean = sub_prec.dot(mean[subvec_ind])

        wmean[subvec_ind] = sub_wmean
        prec[submat_ind] = sub_prec

        zero_bool = np.diag(var) == 0
        subvec_ind = np.where(zero_bool)[0]
        wmean[subvec_ind] = mean[subvec_ind]
        prec[subvec_ind, subvec_ind] = -1
        
    else: raise Exception("Invalid variance parameter")

    return wmean.tolist(), prec.tolist()


def canonical_to_standard(wmean, prec, **kwargs):
    wmean = np.array(wmean)
    prec = np.array(prec)
    
    if prec.ndim == 2:
        mean = np.zeros(len(wmean))
        var = np.zeros(np.shape(prec))

        non_inf_bool = np.diag(prec)!=-1
        subvec_ind = np.where(non_inf_bool)[0]
        submat_ind = np.ix_(subvec_ind, subvec_ind)

        sub_var = np.linalg.inv(prec[submat_ind])
        sub_mean = sub_var.dot(wmean[subvec_ind])

        mean[subvec_ind] = sub_mean
        var[submat_ind] = sub_var

        inf_bool = np.diag(prec)==-1
        subvec_ind = np.where(inf_bool)[0]
        mean[subvec_ind] = wmean[subvec_ind]
    else: raise Exception("Invalid variance parameter")

    return mean, var


def multiply_message(factor_parameters, inc_message):
    if factor_parameters['ftype'] == 'dlinear' and inc_message['ftype'] == 'dlinear':
        
        def mult_dlinear_dlinear(factor_parameters, inc_message):
            if inc_message.has_key('modval'):
                factor_parameters.update({'modval': inc_message['modval']})
            elif inc_message.has_key('inputs'):
                factor_parameters.update({'inputs': inc_message['inputs']})
            elif inc_message.has_key('weights'):
                factor_parameters.update({'weights': inc_message['weights']})
            elif inc_message.has_key('output'):
                factor_parameters.update({'output': inc_message['output']})
            else:
                raise Exception("Invalid dlinear parameters")

            return factor_parameters

        
        return mult_dlinear_dlinear(factor_parameters, inc_message)

    elif factor_parameters['ftype'] == 'gaussian' and inc_message['ftype'] == 'gaussian':
        
        def mult_gaussian_gaussian(factor_parameters, inc_message):
            old_wmean = np.array(factor_parameters['wmean'])
            old_prec = np.array(factor_parameters['prec'])

            new_wmean = np.zeros(len(old_wmean))
            new_prec = np.zeros(np.shape(old_prec))

            inc_wmean = np.array(inc_message['wmean'])
            inc_prec = np.array(inc_message['prec'])

            non_inf_bool = np.bitwise_and(np.diag(inc_prec) != -1, np.diag(old_prec) != -1)
            subvec_ind = np.where(non_inf_bool)[0]
            submat_ind = np.ix_(subvec_ind, subvec_ind)

            sub_new_prec = old_prec[submat_ind] + inc_prec[submat_ind]
            sub_new_wmean = old_wmean[subvec_ind] + inc_wmean[subvec_ind]

            new_prec[submat_ind] = sub_new_prec
            new_wmean[subvec_ind] = sub_new_wmean

            subvec_ind = np.where(np.diag(inc_prec) == -1)[0]
            new_wmean[subvec_ind] = inc_wmean[subvec_ind]
            new_prec[subvec_ind, subvec_ind] = -1

            subvec_ind = np.where(np.diag(old_prec) == -1)[0]
            new_wmean[subvec_ind] = old_wmean[subvec_ind]
            new_prec[subvec_ind, subvec_ind] = -1
            
            factor_parameters['wmean'] = new_wmean
            factor_parameters['prec'] = new_prec
            
            return factor_parameters

        
        return mult_gaussian_gaussian(factor_parameters, inc_message)        

    elif factor_parameters['ftype'] == 'gamma' and inc_message['ftype'] == 'gamma':

        def mult_gamma_gamma(factor_parameters, inc_message):
            new_shape = factor_parameters['shape'] + inc_message['shape'] - 1
            new_scale = 1./factor_parameters['scale'] + 1./inc_message['scale']
            new_scale = 1./new_scale
            
            factor_parameters['shape'] = new_shape
            factor_parameters['scale'] = new_scale
            
            return factor_parameters

            
        return mult_gamma_gamma(factor_parameters, inc_message)

    else:
        raise Exception("Invalid belief type")

    return factor_parameters


def approximate_message(belief_from, belief_to, inc_message, out_message):
    if belief_from['ftype'] == 'gaussian' and belief_to['ftype'] == 'dlinear':

        def approx_gaussian_dlinear(belief_from, belief_to, inc_message, out_message):
            old_wmean = np.array(belief_from['wmean'])
            old_prec = np.array(belief_from['prec'])

            new_wmean = np.zeros(len(old_wmean))
            new_prec = np.zeros(np.shape(old_prec))

            inc_wmean = np.array(inc_message['wmean'])
            inc_prec = np.array(inc_message['prec'])

            non_inf_bool = np.bitwise_and(np.diag(inc_prec) != -1, np.diag(old_prec) != -1)
            subvec_ind = np.where(non_inf_bool)[0]
            submat_ind = np.ix_(subvec_ind, subvec_ind)

            sub_new_prec = old_prec[submat_ind] - inc_prec[submat_ind]
            sub_new_wmean = old_wmean[subvec_ind] - inc_wmean[subvec_ind]

            new_prec[submat_ind] = sub_new_prec
            new_wmean[subvec_ind] = sub_new_wmean

            subvec_ind = np.where(np.diag(inc_prec) == -1)[0]
            new_wmean[subvec_ind] = inc_wmean[subvec_ind]
            new_prec[subvec_ind, subvec_ind] = -1

            subvec_ind = np.where(np.diag(old_prec) == -1)[0]
            new_wmean[subvec_ind] = old_wmean[subvec_ind]
            new_prec[subvec_ind, subvec_ind] = -1

            domain = out_message.keys()
            if 'inputs' in domain: domain = 'inputs'
            elif 'output' in domain: domain = 'output'
            elif 'weights' in domain: domain = 'weights'
            else: raise Exception("Invalid domain component")

            out_message[domain]['wmean'] = new_wmean.tolist()
            out_message[domain]['prec'] = new_prec.tolist()

            return out_message


        return approx_gaussian_dlinear(belief_from, belief_to, inc_message, out_message)

    elif belief_from['ftype'] == 'gamma' and belief_to['ftype'] == 'dlinear':

        def approx_gamma_dlinear(belief_from, belief_to, inc_message, out_message):
            old_shape = belief_from['shape']
            old_scale = belief_from['scale']

            inc_shape = inc_message['shape']
            inc_scale = inc_message['scale']

            new_shape = old_shape - inc_shape + 1
            new_scale = 1./old_scale - 1./inc_scale
            new_scale = 1./new_scale

            domain = out_message.keys()
            if 'modval' in domain: domain = 'modval'
            else: raise Exception("Invalid domain component")

            out_message[domain]['shape'] = new_shape
            out_message[domain]['scale'] = new_scale

            return out_message


        return approx_gamma_dlinear(belief_from, belief_to, inc_message, out_message)

    elif belief_from['ftype'] == 'dlinear' and belief_to['ftype'] == 'gaussian':
        
        def approx_dlinear_gaussian(belief_from, belief_to, inc_message, out_message):
            latent = belief_from['latent']
            old_i_wmean = np.array(belief_from['inputs']['wmean'])
            old_i_prec = np.array(belief_from['inputs']['prec'])
            old_o_wmean = belief_from['output']['wmean'][latent]
            old_o_prec = np.array(belief_from['output']['prec'])[latent, latent]
            old_w_wmean = np.array(belief_from['weights']['wmean'])
            old_w_prec = np.array(belief_from['weights']['prec'])
            old_m_shape = belief_from['modval']['shape']
            old_m_scale = belief_from['modval']['scale']

            if old_o_prec == -1:
                old_o_mean = np.log(np.min((np.max((old_o_wmean, EPSILON)), 1-EPSILON))) - np.log(1-np.min((np.max((old_o_wmean, EPSILON)), 1-EPSILON)))
                old_o_var = 0
            else:
                old_o_var = 1./(old_o_prec)
                old_o_mean = old_o_wmean*old_o_var

                '''
                samples = np.array([
                    old_o_mean - np.sqrt(1.5*old_o_var),
                    old_o_mean,
                    old_o_mean + np.sqrt(1.5*old_o_var)])
                '''
                samples = np.random.normal(old_o_mean, np.sqrt(old_o_var),1000)

                    
                if belief_from['act_func'] == 'sigmoid':
                    samples = np.array([np.min((np.max((tmp, EPSILON)), 1-EPSILON)) for tmp in samples])
                    trans_samples = np.log(samples) - np.log(1-samples)
                elif belief_from['act_func'] == 'relu':
                    trans_samples = samples
                    
                else: raise Exception("Invalid activation function")                    

                old_o_mean = np.mean(trans_samples)
                old_o_var = np.var(trans_samples)
            
            domain = inc_message.keys()
            if 'inputs' in domain:
                non_inf_bool = np.diag(old_i_prec)!=-1
                subvec_ind = np.where(non_inf_bool)[0]
                submat_ind = np.ix_(subvec_ind, subvec_ind)

                new_i_wmean = np.zeros(len(old_i_wmean))
                new_i_prec = np.zeros(np.shape(old_i_prec))

                old_w_mean, old_w_var = canonical_to_standard(old_w_wmean, old_w_prec)

                tmp_const = old_m_scale*(old_m_shape-1)
                new_i_full_wmean = (tmp_const*old_o_mean*old_w_mean)[:-1]
                new_i_full_prec = (tmp_const*(old_w_var + np.outer(old_w_mean, old_w_mean)))[:-1,:-1]

                #new_i_full_wmean = (tmp_const*old_o_mean*old_w_mean)[:-1] + old_i_wmean
                #new_i_full_prec = (tmp_const*(old_w_var + np.outer(old_w_mean, old_w_mean)))[:-1,:-1] + old_i_prec
                #new_i_mean = np.linalg.inv(new_i_full_prec).dot(new_i_full_wmean)

                new_i_wmean[subvec_ind] = new_i_full_wmean[subvec_ind]
                new_i_prec[submat_ind] = new_i_full_prec[submat_ind]

                inf_bool = np.diag(old_i_prec)==-1
                subvec_ind = np.where(inf_bool)[0]
                new_i_prec[subvec_ind,subvec_ind] = -1

                out_message['wmean'] = new_i_wmean.tolist()
                out_message['prec'] = new_i_prec.tolist()

                return out_message
            
            elif 'output' in domain:
                new_o_wmean = np.array(belief_from['output']['wmean'])
                new_o_prec = np.array(belief_from['output']['prec'])

                old_i_sub_mean, old_i_sub_var = canonical_to_standard(old_i_wmean, old_i_prec)
                old_i_mean = np.append(old_i_sub_mean, 1)
                old_i_var = np.zeros(np.shape(old_w_prec))
                old_i_var[np.ix_(np.arange(len(old_i_sub_mean)), np.arange(len(old_i_sub_mean)))] = old_i_sub_var

                old_w_mean, old_w_var = canonical_to_standard(old_w_wmean, old_w_prec)

                tmp_const = old_m_scale*(old_m_shape-1)
                if old_o_var == 0:
                    new_o_var = 0.0
                    new_o_mean = old_o_mean
                else:
                    new_o_var = 1./(1./old_o_var + tmp_const)
                    new_o_mean = (tmp_const*old_i_mean.dot(old_w_mean) + old_o_mean/old_o_var)*new_o_var
                
                '''
                samples = np.array([
                    old_o_mean - np.sqrt(1.5*old_o_var),
                    old_o_mean,
                    old_o_mean + np.sqrt(1.5*old_o_var)])
                '''
                samples = np.random.normal(old_o_mean, np.sqrt(old_o_var),1000)

                if belief_from['act_func'] == 'sigmoid':
                    trans_samples = 1./(1+np.exp(-samples))
                elif belief_from['act_func'] == 'relu':
                    trans_samples = samples
                    trans_samples[trans_samples<0] = 0
                else: raise Exception("Invalid activation function")

                trans_o_mean = np.mean(trans_samples)
                trans_o_var = np.var(trans_samples)

                if trans_o_var == 0:
                    new_o_wmean[latent] = trans_o_mean
                    new_o_prec[latent, latent] = -1
                else:
                    new_o_wmean[latent] = trans_o_mean/trans_o_var - old_o_wmean
                    new_o_prec[latent, latent] = np.max((1./trans_o_var - old_o_prec, EPSILON))

                out_message['wmean'] = new_o_wmean.tolist()
                out_message['prec'] = new_o_prec.tolist()

                return out_message
                
            elif 'weights' in domain:
                non_inf_bool = np.diag(old_w_prec)!=-1
                subvec_ind = np.where(non_inf_bool)[0]
                submat_ind = np.ix_(subvec_ind, subvec_ind)

                new_w_wmean = np.zeros(len(old_w_wmean))
                new_w_prec = np.zeros(np.shape(old_w_prec))
                
                old_i_sub_mean, old_i_sub_var = canonical_to_standard(old_i_wmean, old_i_prec)
                old_i_mean = np.append(old_i_sub_mean, 1)
                old_i_var = np.zeros(np.shape(old_w_prec))
                old_i_var[np.ix_(np.arange(len(old_i_sub_mean)), np.arange(len(old_i_sub_mean)))] = old_i_sub_var

                tmp_const = old_m_scale*(old_m_shape-1)
                new_w_full_wmean = tmp_const*old_o_mean*old_i_mean
                new_w_full_prec = tmp_const*(old_i_var + np.outer(old_i_mean, old_i_mean))

                #new_w_full_wmean = tmp_const*old_o_mean*old_i_mean + old_w_wmean
                #new_w_full_prec = tmp_const*(old_i_var + np.outer(old_i_mean, old_i_mean)) + old_w_prec
                #new_w_mean = np.linalg.inv(new_w_full_prec).dot(new_w_full_wmean)
                
                new_w_wmean[subvec_ind] = new_w_full_wmean[subvec_ind]
                new_w_prec[submat_ind] = new_w_full_prec[submat_ind]

                inf_bool = np.diag(old_w_prec)==-1
                subvec_ind = np.where(inf_bool)[0]
                new_w_wmean[subvec_ind] = old_w_wmean[subvec_ind]
                new_w_prec[subvec_ind,subvec_ind] = -1

                out_message['wmean'] = new_w_wmean.tolist()
                out_message['prec'] = new_w_prec.tolist()

                return out_message

            else:
                raise Exception("Invalid component for incoming message")


        return approx_dlinear_gaussian(belief_from, belief_to, inc_message, out_message)
            
    elif belief_from['ftype'] == 'dlinear' and belief_to['ftype'] == 'gamma':
        
        def approx_dlinear_gamma(belief_from, belief_to, inc_message, out_message):
            latent = belief_from['latent']
            old_i_wmean = np.array(belief_from['inputs']['wmean'])
            old_i_prec = np.array(belief_from['inputs']['prec'])
            old_o_wmean = belief_from['output']['wmean'][latent]
            old_o_prec = np.array(belief_from['output']['prec'])[latent, latent]
            old_w_wmean = np.array(belief_from['weights']['wmean'])
            old_w_prec = np.array(belief_from['weights']['prec'])
            old_m_shape = belief_from['modval']['shape']
            old_m_scale = belief_from['modval']['scale']

            if old_o_prec == -1:
                old_o_mean = np.log(np.min((np.max((old_o_wmean, EPSILON)), 1-EPSILON))) - np.log(1-np.min((np.max((old_o_wmean, EPSILON)), 1-EPSILON)))
                old_o_var = 0
            else:
                old_o_var = 1./(old_o_prec)
                old_o_mean = old_o_wmean*old_o_var

                '''
                samples = np.array([
                    old_o_mean - np.sqrt(1.5*old_o_var),
                    old_o_mean,
                    old_o_mean + np.sqrt(1.5*old_o_var)])
                '''
                samples = np.random.normal(old_o_mean, np.sqrt(old_o_var),1000)

                if belief_from['act_func'] == 'sigmoid':
                    samples = np.array([np.min((np.max((tmp, EPSILON)), 1-EPSILON)) for tmp in samples])
                    trans_samples = np.log(samples) - np.log(1-samples)
                elif belief_from['act_func'] == 'relu':
                    trans_samples = samples
                    
                else: raise Exception("Invalid activation function")                    

                old_o_mean = np.mean(trans_samples)
                old_o_var = np.var(trans_samples)

            domain = inc_message.keys()
            if 'modval' in domain:
                old_i_sub_mean, old_i_sub_var = canonical_to_standard(old_i_wmean, old_i_prec)
                old_i_mean = np.append(old_i_sub_mean, 1)
                old_i_var = np.zeros(np.shape(old_w_prec))
                old_i_var[np.ix_(np.arange(len(old_i_sub_mean)), np.arange(len(old_i_sub_mean)))] = old_i_sub_var

                old_w_mean, old_w_var = canonical_to_standard(old_w_wmean, old_w_prec)

                new_shape = 1.5
                new_scale = old_o_var + old_o_mean**2 - 2*old_o_mean*old_i_mean.dot(old_w_mean)
                new_scale += np.sum(np.diag(old_i_var.dot(old_w_var)))
                new_scale += old_i_mean.dot(old_w_var).dot(old_i_mean)
                new_scale += old_w_mean.dot(old_i_var).dot(old_w_mean)
                new_scale += (old_w_mean.dot(old_i_mean))**2
                new_scale = 2./new_scale

                out_message['shape'] = new_shape
                out_message['scale'] = new_scale

                return out_message

            else:
                raise Exception("Invalid component for incoming message")


        return approx_dlinear_gamma(belief_from, belief_to, inc_message, out_message)
    
    else:
        raise Exception("Invalid belief_from type")
