import ipdb

import tensorflow as tf
import numpy as np

from config import config

from utils import *


class NN(object):
    def init(self):
        print "Set up network with tensorflow"

        # Set up inputs placeholder
        self.inputs = tf.placeholder(tf.float64, [None, np.prod(config.mod_input_dim)])

        # Get relevant constants
        input_shape = config.mod_input_dim
        weight_shapes = config.mod_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)

        print "Create weights tensors"
        self.weights = {}
        self.bias = {}
        for layer in range(len(weight_shapes)):
            layer = 'l_' + str(layer)
            
            weight_init = np.random.uniform(-1,1, [np.prod(input_shape), np.prod(num_nodes[layer])])
            self.weights.update({layer:tf.Variable(weight_init, dtype=tf.float64)})
            
            bias_init = np.random.uniform(-1,1, [np.prod(num_nodes[layer])])
            self.bias.update({layer:tf.Variable(bias_init, dtype=tf.float64)})

            input_shape = num_nodes[layer]

        print "Propagate transformation"
        input_shape = config.mod_input_dim
        id_mat = tf.constant(np.eye(np.prod(config.mod_input_dim)), dtype = tf.float64)
        input_tensor = self.inputs
        for layer in range(len(weight_shapes)):
            layer = 'l_' + str(layer)
            indices = get_indices(input_shape, weight_shapes[layer])
            select_matrix = np.zeros([np.prod(input_shape), np.prod(num_nodes[layer])])
            for i in range(len(indices)): select_matrix[indices[i],i] = 1
            select_matrix = tf.constant(select_matrix, dtype = tf.float64)
            input_tensor = tf.matmul(input_tensor, tf.multiply(select_matrix, self.weights[layer])) + self.bias[layer]
            
            if config.mod_act_funcs[layer] == 'sigmoid': input_tensor = tf.nn.sigmoid(input_tensor)
            elif config.mod_act_funcs[layer] == 'relu': input_tensor = tf.nn.relu(input_tensor)
            else: raise Exception('Invalid activation function')
            
            input_shape = num_nodes[layer]

        # Set up outputs and predicted variables
        self.out_pred = input_tensor
        self.out_true = tf.placeholder(tf.float64, [None, np.prod(num_nodes[layer])])
        
        # Create loss function and optimizer
        self.loss = tf.reduce_mean(tf.square(self.out_pred - self.out_true))
        self.optimizer = tf.train.AdamOptimizer(config.nn_learn_rate).minimize(self.loss)

        
    def train_batch(self, data_inputs, data_outputs, batch_size, number_iterations, test_inputs, test_outputs):
        #batch = np.random.choice(np.arange(len(data_outputs)), batch_size)
        #batch_input = np.array(data_inputs)[batch,:]
        #batch_output = np.array(data_outputs)[batch]
        #batch_input = batch_input[np.argsort(np.argmax(batch_output,1))]
        #batch_output = batch_output[np.argsort(np.argmax(batch_output,1))]
        
        feed_dict_train = {self.inputs:np.array(data_inputs), self.out_true: np.array(data_outputs)}
        feed_dict_test = {self.inputs:np.array(test_inputs), self.out_true: np.array(test_outputs)}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for cnt in range(number_iterations):
                opt, loss_train = sess.run((self.optimizer, self.loss), feed_dict=feed_dict_train)
                loss_test = sess.run((self.loss), feed_dict=feed_dict_test)
                if cnt%10 == 0: print "{} - {}".format(loss_train, loss_test)
