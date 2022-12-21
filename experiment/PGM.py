import ipdb

import sys

import numpy as np

from mathematics import *

from config import config

from utils import *

from messages import *

from IO import *

def fname(*fname_args):
    fname = fname_args[0] + "("
    for vname_args in fname_args[1:]:
        fname += vname(vname_args) + ','
    fname = fname[:-1]
    fname += ")"
    return fname

def vname(*vname_args):
    vname = ''
    for variables in vname_args:
        vname += variables[0]
        if len(variables) > 1:
            vname += "_" + "_".join(str(i).zfill(4) for i in variables[1:])
    return vname

def padstr(integer):
    return str(integer).zfill(5)



class PGM(object):
    def init_factors(self, data_inputs, data_outputs, factor_type):
        self.data_inputs = data_inputs
        self.data_outputs = data_outputs
        
        print "\nStore {} for inputs".format(factor_type)
        input_shape = config.mod_input_dim
        weight_shapes = config.mod_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)
        layer = 'l_0'
        for cnt, sample in enumerate(self.data_inputs):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            factor_name = fname('f', ['x_' + layer,cnt])
            
            par_mean = sample
            par_var = config.pgm_inputs_var_prior
            var_type = config.pgm_inputs_var_type
            
            if var_type == 'full': par_var = np.diag([par_var]*len(sample)).tolist()
            else: raise Exception('Invalid variance type')

            par_wmean, par_prec = standard_to_canonical(par_mean, par_var)

            if factor_type == 'prior': store_priors(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
            elif factor_type == 'belief': store_beliefs(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
            else: raise Exception("Invalid factor_type")
            
        print "\nStore {} for outputs".format(factor_type)
        for layer in range(len(num_nodes)):
            layer_up = 'l_' + str(layer+1)
            layer_down = 'l_' + str(layer)
            print '\n' + layer_down
            
            for cnt, sample in enumerate(self.data_outputs):
                sys.stdout.write('\r' + str(layer))
                sys.stdout.flush()
                
                factor_name = fname('f', ['x_' + layer_up, cnt])

                if layer_down == np.sort(num_nodes.keys())[-1]:
                    par_mean = sample
                else:
                    par_mean = [config.pgm_outputs_mean_prior[layer_down]]*np.prod(num_nodes[layer_down])
                par_var = config.pgm_outputs_var_prior[layer_down]
                var_type = config.pgm_outputs_var_type[layer_down]
                    
                if var_type == 'full': par_var = np.diag([par_var]*np.prod(num_nodes[layer_down])).tolist()
                else: raise Exception('Invalid variance type')

                par_wmean, par_prec = standard_to_canonical(par_mean, par_var)

                if factor_type == 'prior': store_priors(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
                elif factor_type == 'belief': store_beliefs(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
                else: raise Exception("Invalid factor_type")

        print "\nStore {} for weight priors".format(factor_type)
        input_shape = config.mod_input_dim
        for layer in range(len(num_nodes)):
            
            layer = 'l_' + str(layer)
            indices = get_indices(input_shape, weight_shapes[layer])
            for latent in range(np.prod(num_nodes[layer])):
                factor_name = fname('f', ['w_'+layer, latent])

                par_mean = [config.pgm_weights_mean_prior[layer]]*(np.prod(input_shape)+1)
                var_type = config.pgm_weights_var_type[layer]

                if var_type == 'full':
                    par_var = np.zeros((np.prod(input_shape)+1, np.prod(input_shape)+1))
                    tmp_i = np.append(indices[latent], np.prod(input_shape))
                    par_var[tmp_i, tmp_i] = config.pgm_weights_var_prior[layer]
                    par_var[-1, -1] = config.pgm_bias_var_prior[layer]
                else: raise Exception('Invalid variance type')

                par_wmean, par_prec = standard_to_canonical(par_mean, par_var)

                if factor_type == 'prior': store_priors(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
                elif factor_type == 'belief': store_beliefs(factor_name, {'wmean': par_wmean, 'prec': par_prec, 'ftype': 'gaussian'})
                else: raise Exception("Invalid factor_type")

            input_shape = num_nodes[layer]

        print "\nStore {} for modval priors".format(factor_type)
        input_shape = config.mod_input_dim
        for layer in range(len(num_nodes)):
            
            layer = 'l_' + str(layer)
            for latent in range(np.prod(num_nodes[layer])):
            
                factor_name = fname('f', ['p_'+layer, latent])

                par_shape = config.pgm_modval_shape_prior[layer]
                par_scale = config.pgm_modval_scale_prior[layer]

                if factor_type == 'prior': store_priors(factor_name, {'shape': par_shape, 'scale': par_scale, 'ftype': 'gamma'})
                elif factor_type == 'belief': store_beliefs(factor_name, {'shape': par_shape, 'scale': par_scale, 'ftype': 'gamma'})
                else: raise Exception("Invalid factor_type")

            input_shape = num_nodes[layer]

        print "\nStore {} for interaction priors".format(factor_type)
        for cnt, sample in enumerate(self.data_outputs):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            for layer in range(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    if factor_type == 'prior': store_priors(factor_name, {'ftype': 'dlinear', 'latent': latent, 'act_func': config.mod_act_funcs[layer_down]})
                    elif factor_type == 'belief': store_beliefs(factor_name, {'ftype': 'dlinear', 'latent': latent, 'act_func': config.mod_act_funcs[layer_down]})
                    else: raise Exception("Invalid factor_type")

        
    def init_messages(self):
        number_samples = len(self.data_inputs)
        
        print "\nInitialize messages inputs <-> interaction"
        input_shape = config.mod_input_dim
        weight_shapes = config.mod_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)
        for cnt, sample in enumerate(range(number_samples)):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            for layer in range(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name_1 = fname('f', ['x_' + layer_down, cnt])
                    factor_name_2 = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])
                    factor_parameters_1 = read_priors(factor_name_1)

                    message_parameters_1 = {'inputs': factor_parameters_1, 'ftype': 'dlinear'}
                    message_parameters_2 = factor_parameters_1
                    message_parameters_2.update({'ftype': 'gaussian'})

                    store_messages(factor_name_1, factor_name_2, message_parameters_1)
                    store_messages(factor_name_2, factor_name_1, message_parameters_2)

        print "\nInitialize messages outputs <-> interaction"
        for cnt, sample in enumerate(range(number_samples)):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            for layer in range(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name_1 = fname('f', ['x_' + layer_up, cnt])
                    factor_name_2 = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])
                    factor_parameters_1 = read_priors(factor_name_1)

                    message_parameters_1 = {'output': factor_parameters_1, 'ftype': 'dlinear'}
                    message_parameters_2 = factor_parameters_1
                    message_parameters_2.update({'ftype': 'gaussian'})
                    
                    store_messages(factor_name_1, factor_name_2, message_parameters_1)
                    store_messages(factor_name_2, factor_name_1, message_parameters_2)

        print "\nInitialize messages weights <-> interaction"
        for cnt, sample in enumerate(range(number_samples)):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            for layer in range(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name_1 = fname('f', ['w_' + layer_down, latent])
                    factor_name_2 = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])
                    factor_parameters_1 = read_priors(factor_name_1)

                    message_parameters_1 = {'weights': factor_parameters_1, 'ftype': 'dlinear'}
                    message_parameters_2 = factor_parameters_1
                    message_parameters_2.update({'ftype': 'gaussian'})
                    
                    store_messages(factor_name_1, factor_name_2, message_parameters_1)
                    store_messages(factor_name_2, factor_name_1, message_parameters_2)

        print "\nInitialize messages modval <-> interaction"
        for cnt, sample in enumerate(range(number_samples)):
            sys.stdout.write('\r' + str(cnt))
            sys.stdout.flush()
            
            for layer in range(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name_1 = fname('f', ['p_' + layer_down, latent])
                    factor_name_2 = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])
                    factor_parameters_1 = read_priors(factor_name_1)

                    factor_parameters_1['shape'] = 1.5
                    
                    message_parameters_1 = {'modval': factor_parameters_1, 'ftype': 'dlinear'}
                    message_parameters_2 = factor_parameters_1
                    message_parameters_2.update({'ftype': 'gamma'})
                    
                    store_messages(factor_name_1, factor_name_2, message_parameters_1)
                    store_messages(factor_name_2, factor_name_1, message_parameters_2)

    
    def pass_messages_manual(self, batch_size):
        number_samples = len(self.data_inputs)
        
        print "\nUpdate dlinear factors"
        input_shape = config.mod_input_dim
        weight_shapes = config.mod_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)
        for cnt, sample in enumerate(range(number_samples)):
            for layer in np.arange(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                for latent in range(np.prod(num_nodes[layer_down])):
                    factor_name = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])
                    run_instruction(['upd_belief', factor_name])

        print "\nUpdate weights and modval beliefs"
        for layer in np.arange(len(num_nodes))[::-1]:
            layer_down = 'l_' + str(layer)
            layer_up = 'l_' + str(layer+1)
            for latent in np.arange(np.prod(num_nodes[layer_down])):
                factor_name_weights = fname('f', ['w_'+layer_down, latent])
                factor_name_modval = fname('f', ['p_'+layer_down, latent])
                run_instruction(['upd_belief', factor_name_weights])
                run_instruction(['upd_belief', factor_name_modval])
                
                    
        print "\nTrain batch"
        for _ in range(100):
            batch = np.random.choice(np.arange(number_samples), batch_size)
            #batch = np.arange(number_samples)
            
            self.evaluate_accuracy()
            for layer in np.arange(len(num_nodes)):
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                latent = np.random.choice(np.arange(np.prod(num_nodes[layer_down])))

                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_belief', factor_name_inputs])
                    run_instruction(['upd_message', factor_name_inputs, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_weights])
                    
                run_instruction(['upd_belief', factor_name_weights])

                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_message', factor_name_weights, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_modval])

                run_instruction(['upd_belief', factor_name_modval])
                    
                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_message', factor_name_modval, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_outputs])
                    run_instruction(['upd_belief', factor_name_outputs])

            for layer in np.arange(len(num_nodes))[::-1]:
                layer_down = 'l_' + str(layer)
                layer_up = 'l_' + str(layer+1)
                latent = np.random.choice(np.arange(np.prod(num_nodes[layer_down])))

                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_belief', factor_name_outputs])
                    run_instruction(['upd_message', factor_name_outputs, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_weights])
                    
                run_instruction(['upd_belief', factor_name_weights])

                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_message', factor_name_weights, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_modval])

                run_instruction(['upd_belief', factor_name_modval])
                    
                for cnt in batch:
                    factor_name_inputs = fname('f', ['x_' + layer_down, cnt])
                    factor_name_outputs = fname('f', ['x_' + layer_up, cnt])
                    factor_name_weights = fname('f', ['w_'+layer_down, latent])
                    factor_name_modval = fname('f', ['p_'+layer_down, latent])
                    factor_name_dlinear = fname('d', ['x_' + layer_down, cnt], ['x_' + layer_up, cnt], ['p_'+layer_down, latent], ['w_'+layer_down, latent])

                    run_instruction(['upd_message', factor_name_modval, factor_name_dlinear])
                    run_instruction(['upd_belief', factor_name_dlinear])
                    run_instruction(['upd_message', factor_name_dlinear, factor_name_inputs])
                    run_instruction(['upd_belief', factor_name_inputs])

                    
    def evaluate_accuracy(self):
        input_shape = config.mod_input_dim
        weight_shapes = config.mod_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)

        weights = {}
        bias = {}
        for layer in range(len(weight_shapes)):
            layer = 'l_' + str(layer)
            weights.update({layer:[]})
            bias.update({layer:[]})
            for latent in np.arange(np.prod(num_nodes[layer])):
                factor_name = fname('f', ['w_'+layer, latent])
                factor_parameters = read_beliefs(factor_name)
                tmp = canonical_to_standard(factor_parameters['wmean'], factor_parameters['prec'])
                selector = np.diag(tmp[1])!=0
                tmp = tmp[0][selector]
                weights[layer].append(tmp[:-1])
                bias[layer].append(tmp[-1])
            weights[layer] = np.array(weights[layer])

        data_predicted = predict_on_inputs(np.array(self.data_inputs), weights, bias, input_shape, weight_shapes, config.mod_act_funcs, 0)
        errors = np.array(self.data_outputs) - data_predicted
        print np.mean(errors**2)



