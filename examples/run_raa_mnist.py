from common import TEST

import unittest

def fname(*fname_args):
    fname = fname_args[0] + "("
    for variables in fname_args[1:]:
        fname += variables[0]
        if len(variables) > 1:
            fname += "_" + "_".join(str(i).zfill(4) for i in variables[1:])
        fname += ','
    fname = fname[:-1]
    fname += ")"
    return fname


def bufstr(integer):
    return str(integer).zfill(4)


class test_raa_mnist(unittest.TestCase):        
    def test(self):
        """Run mnist example"""

        import numpy as np
        np.random.seed(0)

        import pickle
        import itertools
        import progressbar

        # Global variables to alter
        SIZE_OF_BLOCK = [5,5]
        DIGITS_TO_CONSIDER = [1,4,7]
        
        NUM_TRAIN_IMAGES = 100   # Maximum of 60000
        NUM_TEST_IMAGES = 100    # Maximum of 10000

        INIT_W1_STD = 0.1
        INIT_W2_STD = 0.1
        
        # BP global variables to alter
        BP_ITERS = 10000
        LEARNING_RATE = 0.1
        REPORTING_RESOLUTION = 5
        
        # MP global variables to alter
        NUM_MP_ITERS = 50
        [X_VAR, Y1_VAR, W1_VAR, Z1_VAR] = [1e-6, 1e3, 1e3, 1]
        [P1_SHAPE, P1_SCALE] = [1+1e-3, 1e3]
        [Y2_VAR, W2_VAR, Z2_VAR] = [1e3, 1e3, 1e-6]
        [P2_SHAPE, P2_SCALE] = [1+1e-3, 1e3]
        X_VAR_FORM = 'common'
        W1_VAR_FORM = 'common'
        Z1_VAR_FORM = 'common'
        W2_VAR_FORM = 'common'
        
        # Global variables not to change
        DIM_OF_IMAGE = 28
        NUM_BLOCKS = [DIM_OF_IMAGE - SIZE_OF_BLOCK[0] + 1, DIM_OF_IMAGE - SIZE_OF_BLOCK[1] + 1]
        
        OUTPUT_FOLDER = '../results/'
        FILE_TEST_DATA = '../data/mnist/t10k-images-idx3-ubyte.gz'
        FILE_TEST_LABELS = '../data/mnist/t10k-labels-idx1-ubyte.gz'
        FILE_TRAIN_DATA = '../data/mnist/train-images-idx3-ubyte.gz'
        FILE_TRAIN_LABELS = '../data/mnist/train-labels-idx1-ubyte.gz'

        def get_data_from_file(filename, number_samples):
            import gzip
            
            data = np.zeros((number_samples, DIM_OF_IMAGE, DIM_OF_IMAGE))
            with gzip.open(filename, 'rb') as file_handler:
                tmp_1 = int(file_handler.read(4).encode('hex'), 16)
                number_datapoints = int(file_handler.read(4).encode('hex'), 16)
                number_rows = int(file_handler.read(4).encode('hex'), 16)
                number_cols = int(file_handler.read(4).encode('hex'), 16)
                
                for cnt in progressbar.ProgressBar()(range(number_samples)):
                    for row in range(DIM_OF_IMAGE):
                        for col in range(DIM_OF_IMAGE):
                            data[cnt, row, col] = int(file_handler.read(1).encode('hex'), 16)
            return data
        
        def get_labels_from_file(filename, number_samples):
            import gzip
            
            labels = np.zeros(number_samples)
            with gzip.open(filename, 'rb') as file_handler:
                tmp_1 = int(file_handler.read(4).encode('hex'), 16)
                number_labels = int(file_handler.read(4).encode('hex'), 16)
                
                for cnt in progressbar.ProgressBar()(range(number_samples)):
                    labels[cnt] = int(file_handler.read(1).encode('hex'), 16)
            return labels


        # Get all data from files
        print "Obtaining test data from file"
        original_test_data = get_data_from_file(FILE_TEST_DATA, NUM_TEST_IMAGES)
        print "Obtaining test labels from file"
        original_test_labels = get_labels_from_file(FILE_TEST_LABELS, NUM_TEST_IMAGES)
        print "Obtaining training data from file"
        original_train_data = get_data_from_file(FILE_TRAIN_DATA, NUM_TRAIN_IMAGES)
        print "Obtaining training labels from file"
        original_train_labels = get_labels_from_file(FILE_TRAIN_LABELS, NUM_TRAIN_IMAGES)

        # Select only data corresponding to relevant digits
        test_data = []
        test_labels = []
        train_data = []
        train_labels = []
        for digit in DIGITS_TO_CONSIDER:
            tmp_data = original_test_data[original_test_labels == digit].tolist()
            tmp_labels = original_test_labels[original_test_labels == digit].tolist()
            test_data += tmp_data
            test_labels += tmp_labels
            tmp_data = original_train_data[original_train_labels == digit].tolist()
            tmp_labels = original_train_labels[original_train_labels == digit].tolist()
            train_data += tmp_data
            train_labels += tmp_labels

        # Create initial values for all weights to be used for both mp and bp
        W1_INIT = np.random.normal(0,INIT_W1_STD, size=[np.prod(NUM_BLOCKS), np.prod(SIZE_OF_BLOCK)])
        W2_INIT = np.random.uniform(0,INIT_W2_STD, size=[len(DIGITS_TO_CONSIDER), np.prod(NUM_BLOCKS)])
        
        def back_propagation():
            import logging
            logging.getLogger("tensorflow").setLevel(logging.WARNING)

            import tensorflow as tf
            
            # Create variables to be used in function
            W1_VEC = {}
            W2_VEC = {}
            Z1_VALS = []
            test_dict = {}
            train_dict = {}
            test_data_flattened = np.reshape(test_data, [np.shape(test_data)[0], DIM_OF_IMAGE**2])
            train_data_flattened = np.reshape(train_data, [np.shape(train_data)[0], DIM_OF_IMAGE**2])

            print "Set up network for back propagation"
            # Set up variables in first layer
            for lat_cnt in progressbar.ProgressBar()(range(np.prod(NUM_BLOCKS))):
                x_pos = lat_cnt%NUM_BLOCKS[0]
                x_coordinates = range(x_pos, x_pos + SIZE_OF_BLOCK[0])
                y_pos = lat_cnt/NUM_BLOCKS[0]
                y_coordinates = range(y_pos, y_pos + SIZE_OF_BLOCK[1])
                
                coordinates = [list(i) for i in itertools.product(x_coordinates, y_coordinates)]
                flat_coordinates = map(lambda val: val[0]+DIM_OF_IMAGE*val[1], coordinates)
                flat_coordinates.sort()

                tmp_x_block = tf.placeholder(tf.float64, [None, np.prod(SIZE_OF_BLOCK)])
                train_dict.update({tmp_x_block: train_data_flattened[:,np.array(flat_coordinates)]})
                test_dict.update({tmp_x_block: test_data_flattened[:,np.array(flat_coordinates)]})

                tmp_w1 = tf.Variable(np.reshape(W1_INIT[lat_cnt], [len(W1_INIT[lat_cnt]),1]))
                W1_VEC.update({'w1_{}'.format(lat_cnt): tmp_w1})
                tmp_z1 = 1./(1+tf.exp(-tf.matmul(tmp_x_block, tmp_w1)))
                Z1_VALS.append(tmp_z1)

            # Concatenate outputs of first layer into single latent variable vector
            tmp_z1_vec = tf.concat(Z1_VALS,1)

            # Propagate latent variable vector into softmax
            Z2_VALS = []
            for i in range(len(DIGITS_TO_CONSIDER)):
                tmp_w2 = tf.Variable(np.reshape(W2_INIT[i], [np.prod(NUM_BLOCKS),1]))
                W2_VEC.update({'w2_{}'.format(i):tmp_w2})
                tmp_z2 = 1./(1+tf.exp(tf.matmul(tmp_z1_vec, tmp_w2)))
                Z2_VALS.append(tmp_z2)

            # Create loss function
            EST_Z2_VEC = tf.concat(Z2_VALS,1)
            TRUE_Z2_VEC = tf.placeholder(tf.float64, [None, len(DIGITS_TO_CONSIDER)])
            LOSS = tf.reduce_mean(-tf.reduce_sum(TRUE_Z2_VEC * tf.log(EST_Z2_VEC), reduction_indices=[1]))

            # Create dictionary for test target labels
            tmp = np.zeros((len(test_labels), 10))
            for i in range(len(test_labels)):
                tmp[i][int(test_labels[i])] = 1
            test_dict.update({TRUE_Z2_VEC : tmp[:,DIGITS_TO_CONSIDER]})

            # Create dictionary for train target labels
            tmp = np.zeros((len(train_labels), 10))
            for i in range(len(train_labels)):
                tmp[i][int(train_labels[i])] = 1
            train_dict.update({TRUE_Z2_VEC : tmp[:,DIGITS_TO_CONSIDER]})

            # Optimize loss function with steepest descent
            test_loss = []
            train_loss = []
            tf.set_random_seed(0)
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(LOSS)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for cnt in progressbar.ProgressBar()(range(1,BP_ITERS+1)):
                    opt, train_loss_val = sess.run((optimizer, LOSS), feed_dict = train_dict)
                    test_loss_val = sess.run((LOSS), feed_dict = test_dict)

                    if cnt%REPORTING_RESOLUTION == 0:
                        train_loss.append(train_loss_val)
                        test_loss.append(test_loss_val)

                        w1_vec = []
                        for lat_cnt in range(np.prod(NUM_BLOCKS)):
                            w1_vec.append(W1_VEC['w1_{}'.format(lat_cnt)].eval().transpose()[0])
                        pickle.dump(np.array(w1_vec).tolist(), open('w1_vec_{}'.format(str(cnt).zfill(10)),'w'))
                        
                        w2_vec = []
                        for dig_ind in range(len(DIGITS_TO_CONSIDER)):
                            w2_vec.append(W2_VEC['w2_{}'.format(dig_ind)].eval().transpose()[0])
                        pickle.dump(np.array(w2_vec).tolist(), open('w2_vec_{}'.format(str(cnt).zfill(10)),'w'))
            
            # Store output weights and loss function values
            pickle.dump(train_loss, open('train_loss','w'))
            pickle.dump(test_loss, open('test_loss','w'))
                    
        #back_propagation()

        def message_propagation():
            import time
            
            from pypgm import PGM
            
            from pypgm.factor import FACTOR_TYPE, vector_normal, scalar_normal
            
            pgm = PGM('./raa/')
            
            def init_pgm():
                # Ensures correct form of covariance matrices
                if X_VAR_FORM == 'common':
                    x_var = X_VAR
                    x_var_form = vector_normal.COVARIANCE_FORM.COMMON
                elif X_VAR_FORM == 'diagonal':
                    x_var = [X_VAR]*np.prod(SIZE_OF_BLOCK)
                    x_var_form = vector_normal.COVARIANCE_FORM.DIAGONAL
                elif X_VAR_FORM == 'full':
                    x_var = np.diag([X_VAR]*np.prod(SIZE_OF_BLOCK)).tolist()
                    x_var_form = vector_normal.COVARIANCE_FORM.FULL
                else:
                    raise Exception('Invalid covariance form on X1')

                if Z1_VAR_FORM == 'common':
                    z1_var = Z1_VAR
                    z1_var_form = vector_normal.COVARIANCE_FORM.COMMON
                elif Z1_VAR_FORM == 'diagonal':
                    z1_var = [Z1_VAR]*np.prod(SIZE_OF_BLOCK)
                    z1_var_form = vector_normal.COVARIANCE_FORM.DIAGONAL
                elif Z1_VAR_FORM == 'full':
                    z1_var = np.diag([Z1_VAR]*np.prod(SIZE_OF_BLOCK)).tolist()
                    z1_var_form = vector_normal.COVARIANCE_FORM.FULL
                else:
                    raise Exception('Invalid covariance form on Z1')

                if W1_VAR_FORM == 'common':
                    w1_var = W1_VAR
                    w1_var_form = vector_normal.COVARIANCE_FORM.COMMON
                elif W1_VAR_FORM == 'diagonal':
                    w1_var = [W1_VAR]*np.prod(SIZE_OF_BLOCK)
                    w1_var_form = vector_normal.COVARIANCE_FORM.DIAGONAL
                elif W1_VAR_FORM == 'full':
                    w1_var = np.diag([W1_VAR]*np.prod(SIZE_OF_BLOCK)).tolist()
                    w1_var_form = vector_normal.COVARIANCE_FORM.FULL
                else:
                    raise Exception('Invalid covariance form on W1')

                if W2_VAR_FORM == 'common':
                    w2_var = W2_VAR
                    w2_var_form = vector_normal.COVARIANCE_FORM.COMMON
                elif W2_VAR_FORM == 'diagonal':
                    w2_var = [W2_VAR]*np.prod(SIZE_OF_BLOCK)
                    w2_var_form = vector_normal.COVARIANCE_FORM.DIAGONAL
                elif W2_VAR_FORM == 'full':
                    w2_var = np.diag([W2_VAR]*np.prod(SIZE_OF_BLOCK)).tolist()
                    w2_var_form = vector_normal.COVARIANCE_FORM.FULL
                else:
                    raise Exception('Invalid covariance form on W2')
                    
                print "Setup first layer of network for message propagation"
                for lat_cnt in progressbar.ProgressBar()(range(np.prod(NUM_BLOCKS))):
                    x_pos = lat_cnt%NUM_BLOCKS[0]
                    x_coordinates = range(x_pos, x_pos + SIZE_OF_BLOCK[0])
                    y_pos = lat_cnt/NUM_BLOCKS[0]
                    y_coordinates = range(y_pos, y_pos + SIZE_OF_BLOCK[1])

                    coordinates = [list(i) for i in itertools.product(x_coordinates, y_coordinates)]
                    flat_coordinates = map(lambda val: val[0]+DIM_OF_IMAGE*val[1], coordinates)
                    flat_coordinates.sort()

                    fnode_w1_vars = ['w1_{}_{}'.format(bufstr(lat_cnt), bufstr(i)) for i in range(np.prod(SIZE_OF_BLOCK))]
                    fnode_p1_vars = ['p1_{}'.format(bufstr(lat_cnt))]
                    
                    pgm.add_fnode(fname('f',['w1', lat_cnt]),
                                  ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                  par_form=vector_normal.PARAMETER_FORM.STANDARD,
                                  cov_form=w1_var_form,
                                  vars=fnode_w1_vars,
                                  mean=W1_INIT[lat_cnt].T.tolist(),
                                  cov=w1_var)

                    pgm.add_fnode(fname('f',['p1', lat_cnt]),
                                  ftype=FACTOR_TYPE.GAMMA,
                                  vars=fnode_p1_vars,
                                  shape=P1_SHAPE,
                                  scale=P1_SCALE)

                    for cnt in range(len(train_data)):
                        fnode_x_vars = ['x_{}_{}'.format(bufstr(cnt), bufstr(i)) for i in range(DIM_OF_IMAGE**2)]
                        fnode_x_vars = np.array(fnode_x_vars)[flat_coordinates].tolist()
                        fnode_y1_vars = ['y1_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        fnode_z1_vars = ['z1_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        
                        pgm.add_fnode(fname('f',['x_blocks', cnt, lat_cnt]),
                                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                      par_form=vector_normal.PARAMETER_FORM.STANDARD,
                                      cov_form=x_var_form,
                                      vars=fnode_x_vars,
                                      mean=np.array(train_data[cnt]).flatten()[flat_coordinates].tolist(),
                                      cov=x_var)

                        pgm.add_fnode(fname('f', ['x_blocks', cnt, lat_cnt], ['y1', cnt, lat_cnt], ['w1', lat_cnt], ['p1', lat_cnt]),
                                      ftype=FACTOR_TYPE.DLINEAR,
                                      input_vars=fnode_x_vars,
                                      output_vars=fnode_y1_vars,
                                      weight_vars=fnode_w1_vars,
                                      modval_vars=fnode_p1_vars)

                        pgm.add_fnode(fname('f', ['y1', cnt, lat_cnt]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.CANONICAL,
                                      vars=fnode_y1_vars,
                                      wmean=0,
                                      prec=1./Y1_VAR)

                        pgm.add_fnode(fname('f', ['y1', cnt, lat_cnt], ['z1', cnt, lat_cnt]), 
                                      ftype=FACTOR_TYPE.DSIGMOID)

                        pgm.add_fnode(fname('f', ['z1', cnt, lat_cnt]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.CANONICAL,
                                      vars=fnode_z1_vars,
                                      wmean=0,
                                      prec=1./Z1_VAR)

                print "Setup second layer of network for message propagation"
                for lat_cnt in progressbar.ProgressBar()(range(len(DIGITS_TO_CONSIDER))):
                    fnode_w2_vars = ['w2_{}_{}'.format(bufstr(lat_cnt), bufstr(i)) for i in range(len(DIGITS_TO_CONSIDER))]
                    fnode_p2_vars = ['p2_{}'.format(bufstr(lat_cnt))]

                    pgm.add_fnode(fname('f',['w2',lat_cnt]),
                                  ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                  par_form=vector_normal.PARAMETER_FORM.STANDARD,
                                  cov_form=w2_var_form,
                                  vars=fnode_w2_vars,
                                  mean=W2_INIT[lat_cnt].T.tolist()[0],
                                  cov=w2_var)

                    pgm.add_fnode(fname('f',['p2',lat_cnt]),
                                  ftype=FACTOR_TYPE.GAMMA,
                                  vars=fnode_p2_vars,
                                  shape=P2_SHAPE,
                                  scale=P2_SCALE)
                    
                    for cnt in range(len(train_labels)):
                        fnode_z1_vars = ['z1_{}_{}'.format(bufstr(cnt), bufstr(i)) for i in range(np.prod(NUM_BLOCKS))]
                        fnode_y2_vars = ['y2_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        fnode_z2_vars = ['z2_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        
                        pgm.add_fnode(fname('f',['z1',cnt]),
                                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                      par_form=vector_normal.PARAMETER_FORM.CANONICAL,
                                      cov_form=z1_var_form,
                                      vars=fnode_z1_vars,
                                      wmean=[0]*np.prod(NUM_BLOCKS),
                                      prec=z1_var)

                        pgm.add_fnode(fname('f', ['z1', cnt], ['y2', cnt, lat_cnt], ['w2', lat_cnt], ['p2', lat_cnt]),
                                      ftype=FACTOR_TYPE.DLINEAR,
                                      input_vars=fnode_z1_vars,
                                      output_vars=fnode_y2_vars,
                                      weight_vars=fnode_w2_vars,
                                      modval_vars=fnode_p2_vars)

                        pgm.add_fnode(fname('f', ['y2',cnt, lat_cnt]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.CANONICAL,
                                      vars=fnode_y2_vars,
                                      wmean=0,
                                      prec=1./Y2_VAR)

                        pgm.add_fnode(fname('f', ['y2',cnt,lat_cnt], ['z2',cnt,lat_cnt]),
                                      ftype=FACTOR_TYPE.DSIGMOID)
                        
                        pgm.add_fnode(fname('f', ['z2',cnt,lat_cnt]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                                      vars=fnode_z2_vars,
                                      mean=int(train_labels[cnt]==DIGITS_TO_CONSIDER[lat_cnt]),
                                      var=Z2_VAR)

            beg = time.time()
            init_pgm()
            print time.time() - beg

            def connect_nodes():
                print "Connect nodes in first layer of network for message propagation"
                for lat_cnt in progressbar.ProgressBar()(range(NUM_LATENT)):
                    x_pos = lat_cnt%NUM_BLOCKS
                    x_coordinates = range(x_pos, x_pos + SIZE_OF_BLOCK)
                    y_pos = lat_cnt/NUM_BLOCKS
                    y_coordinates = range(y_pos, y_pos + SIZE_OF_BLOCK)
                    coordinates = [list(i) for i in itertools.product(x_coordinates, y_coordinates)]
                    flat_coordinates = map(lambda val: val[0]+DIM_OF_IMAGE*val[1], coordinates)
                    flat_coordinates.sort()
                
                    fnode_w1_vars = ['w1_{}_{}'.format(bufstr(lat_cnt), bufstr(i)) for i in range(SIZE_OF_BLOCK**2)]
                    fnode_p1_vars = ['p1_{}'.format(bufstr(lat_cnt))]
                        
                    for cnt in range(len(train_data)):
                        fnode_x_vars = ['x_{}_{}'.format(bufstr(cnt), bufstr(i)) for i in range(DIM_OF_IMAGE**2)]
                        fnode_x_vars = np.array(fnode_x_vars)[flat_coordinates].tolist()
                        fnode_y1_vars = ['y1_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        fnode_z1_vars = ['z1_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))

                        pgm.connect_fnodes(factor_name, 'f(x_blocks_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt)), fnode_x_vars)
                        pgm.connect_fnodes(factor_name, 'f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_y1_vars)
                        pgm.connect_fnodes(factor_name, 'f(w1_{})'.format(bufstr(lat_cnt)), fnode_w1_vars)
                        pgm.connect_fnodes(factor_name, 'f(p1_{})'.format(bufstr(lat_cnt)), fnode_p1_vars)

                        factor_name = 'f(y1_{}_{},z1_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt),bufstr(lat_cnt))

                        pgm.connect_fnodes(factor_name, 'f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_y1_vars)
                        pgm.connect_fnodes(factor_name, 'f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_y1_vars)

                print "Connect nodes in latent vec"
                for lat_cnt in progressbar.ProgressBar()(range(NUM_LATENT)):
                    for cnt in range(len(train_labels)):
                        pgm.connect_fnodes('f(z1_{})'.format(bufstr(cnt)), 'f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), ['z1_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))])
                
                print "Connect nodes in first layer of network for message propagation"
                for lat_cnt in progressbar.ProgressBar()(range(NUM_DIGITS)):
                    fnode_w2_vars = ['w2_{}_{}'.format(bufstr(lat_cnt), bufstr(i)) for i in range(NUM_LATENT)]
                    fnode_p2_vars = ['p2_{}'.format(bufstr(lat_cnt))]
                    
                    for cnt in range(len(train_labels)):
                        fnode_z1_vars = ['z1_{}_{}'.format(bufstr(cnt), bufstr(i)) for i in range(NUM_LATENT)]
                        fnode_y2_vars = ['y2_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]
                        fnode_z2_vars = ['z2_{}_{}'.format(bufstr(cnt), bufstr(lat_cnt))]

                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))                        

                        pgm.connect_fnodes(factor_name, 'f(z1_{})'.format(bufstr(cnt)), fnode_z1_vars)
                        pgm.connect_fnodes(factor_name, 'f(y2_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_y2_vars)
                        pgm.connect_fnodes(factor_name, 'f(w2_{})'.format(bufstr(lat_cnt)), fnode_w2_vars)
                        pgm.connect_fnodes(factor_name, 'f(p2_{})'.format(bufstr(lat_cnt)), fnode_p2_vars)
                        
                        factor_name = 'f(y2_{}_{},z2_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt),bufstr(lat_cnt))

                        pgm.connect_fnodes(factor_name, 'f(y2_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_y2_vars)
                        pgm.connect_fnodes(factor_name, 'f(z2_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), fnode_z2_vars)
                        
            #connect_nodes()

            def message_passing():
                def init_mp():
                    for lat_cnt in range(NUM_DIGITS):
                        for cnt in range(len(train_data)):
                            factor_name = 'f(y2_{}_{},z2_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt),bufstr(lat_cnt))
                            pgm.update_message('f(z2_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt)), factor_name)
                            pgm.update_belief(factor_name)
                            pgm.update_message(factor_name, 'f(y2_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt)))
                            pgm.update_belief('f(y2_{}_{})'.format(bufstr(cnt),bufstr(lat_cnt)))
                            pgm.update_message('f(y2_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), factor_name)

                def second_layer_dl(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_belief(factor_name)

                def second_layer_weights(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(w2_{})'.format(bufstr(lat_cnt)))
                    pgm.update_belief('f(w2_{})'.format(bufstr(lat_cnt)))
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message('f(w2_{})'.format(bufstr(lat_cnt)), factor_name)

                def second_layer_modval(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(p2_{})'.format(bufstr(lat_cnt)))
                    pgm.update_belief('f(p2_{})'.format(bufstr(lat_cnt)))
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message('f(p2_{})'.format(bufstr(lat_cnt)), factor_name)

                def second_layer_to_latent_vec(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(z1_{})'.format(bufstr(cnt)))

                def update_latent_vec():
                    for cnt in range(len(train_data)):
                        pgm.update_belief('f(z1_{})'.format(bufstr(cnt)))

                def latent_vec_to_first_layer(lat_cnt):
                    for cnt in range(len(train_data)):
                        pgm.update_message('f(z1_{})'.format(bufstr(cnt)), 'f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        pgm.update_belief('f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        factor_name = 'f(z1_{}_{},y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt))
                        pgm.update_message('f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), factor_name)
                        pgm.update_belief(factor_name)
                        pgm.update_message(factor_name, 'f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        pgm.update_belief('f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message('f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), factor_name)

                def first_layer_dl(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_belief(factor_name)

                def first_layer_weights(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(w1_{})'.format(bufstr(lat_cnt)))
                    pgm.update_belief('f(w1_{})'.format(bufstr(lat_cnt)))
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message('f(w1_{})'.format(bufstr(lat_cnt)), factor_name)

                def first_layer_modval(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(p1_{})'.format(bufstr(lat_cnt)))
                    pgm.update_belief('f(p1_{})'.format(bufstr(lat_cnt)))
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message('f(p1_{})'.format(bufstr(lat_cnt)), factor_name)

                def first_layer_to_latent_vec(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(x_blocks_{}_{},y1_{}_{},w1_{},p1_{})'.format(bufstr(cnt),bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        pgm.update_belief('f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))

                        factor_name = 'f(z1_{}_{},y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt), bufstr(cnt), bufstr(lat_cnt))
                        pgm.update_message('f(y1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), factor_name)
                        pgm.update_belief(factor_name)
                        pgm.update_message(factor_name, 'f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))

                        pgm.update_belief('f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)))
                        pgm.update_message('f(z1_{}_{})'.format(bufstr(cnt), bufstr(lat_cnt)), 'f(z1_{})'.format(bufstr(cnt)))

                def latent_vec_to_second_layer(lat_cnt):
                    for cnt in range(len(train_data)):
                        factor_name = 'f(z1_{},y2_{}_{},w2_{},p2_{})'.format(bufstr(cnt), bufstr(cnt), bufstr(lat_cnt), bufstr(lat_cnt), bufstr(lat_cnt))
                        pgm.update_message(factor_name, 'f(z1_{})'.format(bufstr(cnt)))

                
                import time
                init_mp()
                for count in range(NUM_MP_ITERS):
                    print 'ITER : {}'.format(count)
                    beg = time.time()
                    for lat_cnt in range(NUM_DIGITS):
                        print 'a - {}'.format(time.time()-beg)
                        second_layer_dl(lat_cnt)
                        print 'b - {}'.format(time.time()-beg)
                        second_layer_weights(lat_cnt)
                        print 'c - {}'.format(time.time()-beg)
                        second_layer_dl(lat_cnt)
                        print 'd - {}'.format(time.time()-beg)
                        second_layer_modval(lat_cnt)
                        print 'e - {}'.format(time.time()-beg)
                        second_layer_dl(lat_cnt)
                        print 'f - {}'.format(time.time()-beg)
                        second_layer_to_latent_vec(lat_cnt)
                        print 'g - {}'.format(time.time()-beg)
                    update_latent_vec()
                    for lat_cnt in range(NUM_LATENT):
                        print 'h - {}'.format(time.time()-beg)
                        latent_vec_to_first_layer(lat_cnt)
                        print 'i - {}'.format(time.time()-beg)
                        first_layer_dl(lat_cnt)
                        print 'j - {}'.format(time.time()-beg)
                        first_layer_weights(lat_cnt)
                        print 'k - {}'.format(time.time()-beg)
                        first_layer_dl(lat_cnt)
                        print 'l - {}'.format(time.time()-beg)
                        first_layer_modval(lat_cnt)
                        print 'm - {}'.format(time.time()-beg)
                        first_layer_dl(lat_cnt)
                        print 'n - {}'.format(time.time()-beg)
                        first_layer_to_latent_vec(lat_cnt)
                        print 'o - {}'.format(time.time()-beg)
                    update_latent_vec()
                    for lat_cnt in range(NUM_DIGITS):
                        print 'p - {}'.format(time.time()-beg)
                        latent_vec_to_second_layer(lat_cnt)
                        print 'q - {}'.format(time.time()-beg)

            #message_passing()
            
        message_propagation()
        
        TEST.LOG("START ASSERTS", 3)

        TEST.LOG("EXAMPLE COMPLETE", 1)

