import ipdb

import os
import gzip
import pickle

import numpy as np

from config import config

from utils import *


class data(object):
    def __init__(self):
        pass

    def _pickle_mnist_data(self):
        def get_images_from_file(filename, number_samples):
            images = np.zeros((number_samples, 28, 28))
            with gzip.open(filename, 'rb') as file_handler:
                header = int(file_handler.read(4).encode('hex'), 16)
                number_images = int(file_handler.read(4).encode('hex'), 16)
                number_rows = int(file_handler.read(4).encode('hex'), 16)
                number_cols = int(file_handler.read(4).encode('hex'), 16)
                
                for cnt in range(number_samples):
                    for row in range(28):
                        for col in range(28):
                            images[cnt, row, col] = int(file_handler.read(1).encode('hex'), 16)
            return images
        
        def get_labels_from_file(filename, number_samples):
            labels = np.zeros(number_samples)
            with gzip.open(filename, 'rb') as file_handler:
                header = int(file_handler.read(4).encode('hex'), 16)
                number_labels = int(file_handler.read(4).encode('hex'), 16)
                
                for cnt in range(number_samples):
                    labels[cnt] = int(file_handler.read(1).encode('hex'), 16)
            return labels


        print "Read from raw data files"
        original_train_images = get_images_from_file(config.mnist_train_images_file, 60000)
        original_train_labels = get_labels_from_file(config.mnist_train_labels_file, 60000)
        original_test_images = get_images_from_file(config.mnist_test_images_file, 10000)
        original_test_labels = get_labels_from_file(config.mnist_test_labels_file, 10000)

        # Sort data accoding to label
        original_train_images = original_train_images[np.argsort(original_train_labels)]
        original_train_labels = original_train_labels[np.argsort(original_train_labels)]
        original_test_images = original_test_images[np.argsort(original_test_labels)]
        original_test_labels = original_test_labels[np.argsort(original_test_labels)]

        # Store data in dict
        train_data = {'images':original_train_images.tolist(), 'labels':original_train_labels.tolist()}
        test_data = {'images':original_test_images.tolist(), 'labels':original_test_labels.tolist()}

        print "Store data in pickle files"
        with open("./train_data.obj","wb") as file_handler:
            pickle.dump(train_data, file_handler)

        with open("./test_data.obj","wb") as file_handler:
            pickle.dump(test_data, file_handler)

        # Free memory
        del original_train_images
        del original_train_labels
        del original_test_images
        del original_test_labels
        del train_data
        del test_data
        
    
    def get_mnist_data(self):
        if not (os.path.exists('./train_data.obj') and os.path.exists('./test_data.obj')):
            print "Pickle file does not exist"
            self._pickle_mnist_data()

        print "Obtain train data from pickle file"
        with open("./train_data.obj","r") as file_handler:
            train_data = pickle.load(file_handler)
        train_labels = np.array(train_data['labels'])
        train_images = np.array(train_data['images'])
        chosen_samples = []
        for image_id, number_samples in config.train_classes.iteritems():
            cl_samples = np.random.choice(np.where(train_labels == image_id)[0], number_samples, replace = False)
            chosen_samples += cl_samples.tolist()
        chosen_train_labels = train_labels[chosen_samples]
        chosen_train_images = train_images[chosen_samples]

        print "Obtain test data from pickle file"
        with open("./test_data.obj","r") as file_handler:
            test_data = pickle.load(file_handler)
        test_labels = np.array(test_data['labels'])
        test_images = np.array(test_data['images'])
        chosen_samples = []
        for image_id, number_samples in config.test_classes.iteritems():
            cl_samples = np.random.choice(np.where(test_labels==image_id)[0], number_samples, replace = False)
            chosen_samples += cl_samples.tolist()
        chosen_test_labels = test_labels[chosen_samples]
        chosen_test_images = test_images[chosen_samples]

        # Free memory
        del train_data
        del train_labels
        del train_images
        del test_labels
        del test_images
        del test_data

        # Create output dict and return
        out_dict = {
            'train': {
                'output': (chosen_train_labels[:,None] == np.arange(10)).astype(float).tolist(),
                'input': chosen_train_images.reshape((-1, 28*28)).tolist(),
            },
            'test': {
                'output': (chosen_test_labels[:,None] == np.arange(10)).astype(float).tolist(),
                'input': chosen_test_images.reshape((-1, 28*28)).tolist(),
            }
        }

        return out_dict
        
            
    def get_sim_data(self):
        print "Simulate data"
        train_input = np.random.normal(config.sim_input_mean, config.sim_input_std, config.sim_num_train*np.prod(config.mod_input_dim))
        train_input = train_input.reshape(config.sim_num_train, np.prod(config.mod_input_dim))
        test_input = np.random.normal(config.sim_input_mean, config.sim_input_std, config.sim_num_test*np.prod(config.mod_input_dim))
        test_input = test_input.reshape(config.sim_num_test, np.prod(config.mod_input_dim))
        
        input_shape = config.mod_input_dim
        weight_shapes = config.true_weight_dims
        num_nodes = get_num_nodes(input_shape, weight_shapes)
        
        # Create simulated weights
        weights = {}
        bias = {}
        for layer in range(len(weight_shapes)):
            layer = 'l_' + str(layer)
            weights.update({layer:[]})
            bias.update({layer:[]})
            for lat in range(np.prod(num_nodes[layer])):
                weights[layer].append(np.random.normal(0, config.sim_weight_std[layer], np.prod(weight_shapes[layer][0])))
                bias[layer].append(np.random.normal(0, config.sim_bias_std[layer]))
                
        # Predict using simulated weights
        simulated_train_data = predict_on_inputs(train_input, weights, bias, input_shape, weight_shapes, config.true_act_funcs, config.sim_output_noise)
        #simulated_train_data[simulated_train_data<0.5] = 0
        #simulated_train_data[simulated_train_data>=0.5] = 1
        simulated_test_data = predict_on_inputs(test_input, weights, bias, input_shape, weight_shapes, config.true_act_funcs, config.sim_output_noise)
        #simulated_test_data[simulated_test_data<0.5] = 0
        #simulated_test_data[simulated_test_data>=0.5] = 1
        
        # Create output dict and return
        out_dict = {
            'train': {
                'output': simulated_train_data.tolist(),
                'input': train_input.tolist(),
            },
            'test': {
                'output': simulated_test_data.tolist(),
                'input': test_input.tolist(),
            }
        }
        
        return out_dict


    

