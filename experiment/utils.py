import ipdb

import numpy as np


def get_output_shape(input_shape, weight_shape):
    if len(weight_shape) == 1:
        weight_shape = weight_shape[0]
        return (input_shape[0]-weight_shape[0]+1, input_shape[1]-weight_shape[1]+1)
    else:
        
        if not np.all(np.array(weight_shape) == np.reshape(np.array(weight_shape[0]).tolist()*len(weight_shape), (len(weight_shape), -1))):
            raise Exception('Can only allow repeated weight shapes')
        if not np.all(np.array(weight_shape)[0][::-1] == np.array(input_shape)):
            raise Exception('Can only allow multiple weight shapes if weight_shape[::-1] = input_shape')
        return (1, len(weight_shape))


def get_num_nodes(input_shape, weight_shapes):
    num_nodes = {}
    
    for layer in range(len(weight_shapes)):
        layer = 'l_' + str(layer)
        weight_shape = weight_shapes[layer]
        out_shape = get_output_shape(input_shape, weight_shape)
        num_nodes.update({layer:out_shape})
        input_shape = out_shape

    return num_nodes


def get_indices(input_shape, weight_shape):
    y, x = get_output_shape(input_shape, weight_shape)
    if len(weight_shape) == 1:
        x_basis = np.arange(x)
        weight_shape = weight_shape[0]
        start_point = np.hstack([np.arange(j, ((input_shape[0]+1)*input_shape[1])-(input_shape[0]*weight_shape[1]), input_shape[1]) for j in x_basis])
        start_point.sort()
        input_indices = np.array([(np.arange(i,i+weight_shape[0]*input_shape[1],input_shape[1])[:,None]+np.arange(weight_shape[1])).flatten() for i in start_point])
    else:
        input_indices = np.reshape(np.arange(np.prod(input_shape)).tolist()*x, (x,-1))
    return input_indices


def predict_on_inputs(input_data, weights, bias, input_shape, weight_shapes, act_funcs, output_noise):
    num_nodes = get_num_nodes(input_shape, weight_shapes)
    
    for layer in range(len(weights)):
        layer = 'l_' + str(layer)
        indices = get_indices(input_shape, weight_shapes[layer])
        output_data = []
        for ind, w_mat in zip(indices, weights[layer]):
            lat_output = input_data[:,ind].dot(w_mat)
            output_data.append(lat_output)
        bias_mat = np.repeat(bias[layer], len(input_data)).reshape(np.shape(output_data)).T
        input_data = apply_act_func(np.vstack(output_data).T + bias_mat, act_funcs[layer])
        if output_noise != 0:
            input_data = input_data + np.random.normal(0, output_noise[layer], np.shape(input_data))
        input_shape = num_nodes[layer]
        
    return input_data


def apply_act_func(data, function):
    if function == 'sigmoid':
        return 1./(1+np.exp(-data))
    elif function == 'relu':
        data = np.array(data)
        data[data<0]=0
        return data
    else:
        raise Exception('Invalid activation function')

