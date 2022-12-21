import ipdb

import numpy as np
np.random.seed(0)

from config import config

from NN import NN
NN = NN()

from PGM import PGM
PGM = PGM()

from data import data
data = data()


if __name__ == '__main__':
    if ((config.use_mnist_data and config.use_sim_data)
        or
        (not(config.use_mnist_data) and not(config.use_sim_data))):
        raise Exception("Please select data to be used")
    
    elif config.use_mnist_data:
        data_dict = data.get_mnist_data()
    
    elif config.use_sim_data:
        data_dict = data.get_sim_data()

    if config.run_nn:
        NN.init()
        NN.train_batch(data_dict['train']['input'], data_dict['train']['output'],
                       config.nn_batch_size, config.nn_num_iters,
                       data_dict['test']['input'], data_dict['test']['output'])

    if config.run_pgm:
        if config.pgm_init_priors:
            PGM.init_factors(data_dict['train']['input'], data_dict['train']['output'], 'prior')
        if config.pgm_init_beliefs:
            PGM.init_factors(data_dict['train']['input'], data_dict['train']['output'], 'belief')
        if config.pgm_init_messages:
            PGM.init_messages()
        if config.pgm_pass_messages_manual:
            PGM.pass_messages_manual(20)
