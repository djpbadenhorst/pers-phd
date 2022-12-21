import numpy as np

from utils import *


class config(object):
    # Algorithms to run
    run_nn = False
    run_pgm = True

    use_mnist_data = False
    use_sim_data = True
        
    # MNIST DATA
    mnist_test_images_file = '../data/mnist/t10k-images-idx3-ubyte.gz'
    mnist_test_labels_file = '../data/mnist/t10k-labels-idx1-ubyte.gz'
    mnist_train_images_file = '../data/mnist/train-images-idx3-ubyte.gz'
    mnist_train_labels_file = '../data/mnist/train-labels-idx1-ubyte.gz'
    
    train_classes = { 0:1000, 1:1000, 2:1000, 3:1000, 4:1000, 5:1000, 6:1000, 7:1000, 8:1000, 9:1000}
    #train_classes = { 0:5923, 1:6742, 2:5958, 3:6131, 4:5842, 5:5421, 6:5918, 7:6265, 8:5851, 9:5949}
    test_classes = { 0:100, 1:100, 2:100, 3:100, 4:100, 5:100, 6:100, 7:100, 8:100, 9:100}
    #test_classes = { 0:980, 1:1135, 2:1032, 3:1010, 4:982, 5:892, 6:958, 7:1028, 8:974, 9:1009}
    
    # SIMULATED DATA
    sim_num_train = 50
    sim_num_test = 10
    
    true_weight_dims = {'l_0':[(3,2)], 'l_1': [(3,4)]*3, 'l_2': [(3,1)]*2}
    true_act_funcs = {'l_0':'sigmoid', 'l_1':'sigmoid', 'l_2':'sigmoid'}
    
    sim_input_mean = 0
    sim_input_std = 1
    sim_output_noise = {'l_0':0.0, 'l_1':0.0, 'l_2':1e-5}
    sim_weight_std = {'l_0':5.0, 'l_1':5.0, 'l_2':5.0}
    sim_bias_std = {'l_0':1.0, 'l_1':1.0, 'l_2':1.0}

    # PGM PARAMETERS
    pgm_model_folder = ['./pgm_mnist/', './pgm_sim/'][use_sim_data]
    pgm_init_priors = True
    pgm_init_beliefs = True
    pgm_init_messages = True
    pgm_pass_messages_manual = True
    pgm_inputs_var_type = 'full'
    pgm_inputs_var_prior = 0.0
    pgm_outputs_var_type = {'l_0':'full', 'l_1':'full', 'l_2':'full'}
    pgm_outputs_mean_prior = {'l_0': 0.5, 'l_1': 0.5, 'l_2': None}
    pgm_outputs_var_prior = {'l_0': 1.0, 'l_1': 1.0, 'l_2': 0.0}
    pgm_weights_var_type = {'l_0':'full', 'l_1':'full', 'l_2':'full'}
    pgm_weights_mean_prior = {'l_0': 0.0, 'l_1': 0.0, 'l_2': 0.0}
    pgm_weights_var_prior = {'l_0': 1e3, 'l_1': 1e3, 'l_2': 1e3}
    pgm_bias_var_prior = {'l_0': 1e3, 'l_1': 1e3, 'l_2': 1e3}
    pgm_modval_shape_prior = {'l_0': 1e3, 'l_1': 1e3, 'l_2': 1e3}
    pgm_modval_scale_prior = {'l_0': 1e-3, 'l_1': 1e-3, 'l_2': 1e-3}
    pgm_batch_size = 100
    
    # NN PARAMETERS
    nn_learn_rate = 0.001
    nn_batch_size = 5000
    nn_num_iters = 10000
    
    # MODEL PARAMETERS
    mod_input_dim = [(28,28), (6,4)][use_sim_data]
    if use_sim_data: mod_weight_dims = {'l_0':[(3,2)], 'l_1': [(3,4)]*3, 'l_2': [(3,1)]*2}
    else: mod_weight_dims = {'l_0':[(10,10)], 'l_1': [(19,19)]*20, 'l_2': [(20,1)]*10}
    mod_act_funcs = {'l_0':'sigmoid', 'l_1':'sigmoid', 'l_2':'sigmoid'}
    
