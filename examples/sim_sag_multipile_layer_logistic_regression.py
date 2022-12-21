from common import TEST

import unittest

import ipdb

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
    return str(integer).zfill(4)


class test_logistic_regression(unittest.TestCase):
    def tearDown(self):
        """Destroys necessary variables"""

        import shutil

        shutil.rmtree('./sag/')
        
        
    def test(self):
        """Example for basic logistic regression"""

        TEST.LOG("START - EXAMPLE", 1)
        TEST.LOG(self.test.__doc__, 2)

        import numpy as np
        np.random.seed(0)

        # Global variables used in simulations
        NUMBER_SAMPLES = 100
        NUM_ITERATIONS = 1000
        SAMPLE_BATCH_SIZE = 10
        LAYER_BATCH_SIZE = 1
        
        # Simulate samples
        W_TRUE = [
            [[ 1.8, -0.4], [-0.4,  1.3], [0.4,  -0.3]],
            [[-1.2,  1.2, 0.5], [0.1,  0.3, -0.5]],
            [[1.0, 0.4], [-1.1, -0.2]]
        ]
        NOISE = 1e-3
        X_STD = 1
        X_MATRIX = np.random.normal(0.5, X_STD, (NUMBER_SAMPLES, len(W_TRUE[0][0])))
        Z_MATRIX = X_MATRIX
        for layer in range(len(W_TRUE)):
            Y_MATRIX = Z_MATRIX.dot(np.array(W_TRUE[layer]).T)
            Z_MATRIX = 1./(1+np.exp(-Y_MATRIX))
        Z_MATRIX = Z_MATRIX + np.random.normal(0,NOISE,np.shape(Z_MATRIX))

        # Prior parameter values
        XI_COV = [1e-5] + [1.]*(len(W_TRUE)-1)
        XI_COV_FORM = ['FULL', 'FULL', 'FULL']
        W_COV = [1e10]*len(W_TRUE)
        W_COV_FORM = ['FULL', 'FULL', 'FULL']
        Y_VAR = [1e2]*len(W_TRUE)
        XO_VAR = [5.]*(len(W_TRUE)-1) + [1e-5]
        P_SHAPE, P_SCALE = [[1]*len(W_TRUE), [1+1e-3]*len(W_TRUE)]

        # Imports used in simulation
        import time
        import progressbar
                
        from utils.sdict import SDict
        
        from pypgm import PGM
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal, scalar_normal
        
        # Adapt covariance matrices according to given covariance matrix forms
        for layer in range(len(W_TRUE)):
            if XI_COV_FORM[layer] == 'COMMON':
                XI_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.COMMON
            elif XI_COV_FORM[layer] == 'DIAGONAL':
                XI_COV[layer] = [XI_COV[layer]]*len(W_TRUE[layer][0])
                XI_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.DIAGONAL
            elif XI_COV_FORM[layer] == 'FULL':
                XI_COV[layer] = np.diag([XI_COV[layer]]*len(W_TRUE[layer][0])).tolist()
                XI_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.FULL
                
            if W_COV_FORM[layer] == 'COMMON':
                W_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.COMMON
            elif W_COV_FORM[layer] == 'DIAGONAL':
                W_COV[layer] = [W_COV[layer]]*len(W_TRUE[layer][0])
                W_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.DIAGONAL
            elif W_COV_FORM[layer] == 'FULL':
                W_COV[layer] = np.diag([W_COV[layer]]*len(W_TRUE[layer][0])).tolist()
                W_COV_FORM[layer] = vector_normal.COVARIANCE_FORM.FULL
            
        pgm = PGM('./sag/')

        def create_variable_dictionaries():
            fnode_xi_vars, fnode_y_vars, fnode_w_vars, fnode_p_vars, fnode_xo_vars = [{},{},{},{},{}]
            for layer in range(len(W_TRUE)):
                for lat in range(len(W_TRUE[layer])):
                    tmp_vars = [vname(['w', layer, lat, i]) for i in range(len(W_TRUE[layer][lat]))]
                    fnode_w_vars.update({str([layer, lat]):tmp_vars})
                    tmp_vars = [vname(['p', layer, lat])]
                    fnode_p_vars.update({str([layer, lat]):tmp_vars})
                    
                for cnt in range(NUMBER_SAMPLES):
                    tmp_vars = [vname(['x', layer, cnt, i]) for i in range(len(W_TRUE[layer][lat]))]
                    fnode_xi_vars.update({str([layer, cnt]):tmp_vars})
                    for lat in range(len(W_TRUE[layer])):
                        tmp_vars = [vname(['y', layer, cnt, lat])]
                        fnode_y_vars.update({str([layer, cnt, lat]):tmp_vars})
                        tmp_vars = [vname(['x', layer+1, cnt, lat])]
                        fnode_xo_vars.update({str([layer, cnt, lat]):tmp_vars})
                        
            return fnode_xi_vars, fnode_y_vars, fnode_w_vars, fnode_p_vars, fnode_xo_vars

        fnode_xi_vars, fnode_y_vars, fnode_w_vars, fnode_p_vars, fnode_xo_vars = create_variable_dictionaries()
        
        def init_pgm():
            for layer in range(len(W_TRUE)):
                for lat in range(len(W_TRUE[layer])):
                    tmp_mean = [0]*len(W_TRUE[layer][lat])
                    #tmp_mean = np.random.normal(0,1, size = len(W_TRUE[layer][lat])).tolist()
                    pgm.add_fnode(fname('f', ['w', layer, lat]),
                                  ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                  par_form=vector_normal.PARAMETER_FORM.STANDARD,
                                  cov_form=W_COV_FORM[layer],
                                  vars=fnode_w_vars[str([layer, lat])],
                                  mean=tmp_mean,
                                  cov=W_COV[layer])
                    pgm.add_fnode(fname('f', ['p', layer, lat]),
                                  ftype=FACTOR_TYPE.GAMMA,
                                  vars=fnode_p_vars[str([layer,lat])],
                                  shape=P_SHAPE[layer],
                                  scale=P_SCALE[layer])

                for cnt in range(NUMBER_SAMPLES):
                    tmp_mean = [0.0]*len(fnode_xi_vars[str([layer, cnt])])
                    #tmp_mean = np.random.normal(0.0,0, size = len(fnode_xi_vars[str([layer, cnt])])).tolist()
                    if layer == 0:
                        tmp_mean = X_MATRIX[cnt].tolist()
                    pgm.add_fnode(fname('f', ['x', layer, cnt]),
                                  ftype=FACTOR_TYPE.VECTOR_NORMAL,
                                  par_form=vector_normal.PARAMETER_FORM.STANDARD,
                                  cov_form=XI_COV_FORM[layer],
                                  vars=fnode_xi_vars[str([layer, cnt])],
                                  mean=tmp_mean,
                                  cov=XI_COV[layer])
                    
                    for lat in range(len(W_TRUE[layer])):
                        tmp_mean = 0
                        #tmp_mean = np.random.normal(0,1e1)
                        pgm.add_fnode(fname('f', ['y', layer, cnt, lat]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                                      vars=fnode_y_vars[str([layer, cnt, lat])],
                                      mean=tmp_mean,
                                      var=Y_VAR[layer])
                        pgm.add_fnode(fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat]),
                                      ftype=FACTOR_TYPE.DLINEAR,
                                      input_vars=fnode_xi_vars[str([layer, cnt])],
                                      output_vars=fnode_y_vars[str([layer, cnt, lat])],
                                      weight_vars=fnode_w_vars[str([layer, lat])],
                                      modval_vars=fnode_p_vars[str([layer, lat])])
                        pgm.add_fnode(fname('f', ['y',layer, cnt, lat], ['x',layer+1, cnt, lat]),
                                      ftype=FACTOR_TYPE.DSIGMOID,
                                      input_vars=fnode_y_vars[str([layer, cnt, lat])],
                                      output_vars=fnode_xo_vars[str([layer, cnt, lat])])
                        tmp_mean = 0.5
                        #tmp_mean = np.random.normal(0.5,1e2)
                        if layer == len(W_TRUE)-1:
                            tmp_mean = Z_MATRIX[cnt][lat]
                        pgm.add_fnode(fname('f', ['x', layer+1, cnt, lat]),
                                      ftype=FACTOR_TYPE.SCALAR_NORMAL,
                                      par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                                      vars=fnode_xo_vars[str([layer, cnt, lat])],
                                      mean=tmp_mean,
                                      var=XO_VAR[layer])

        print 'init_pgm'
        init_pgm()

        def connect_fnodes():
            for layer in range(len(W_TRUE)):
                for cnt in range(NUMBER_SAMPLES):
                    if layer != 0:
                        for lat in range(len(W_TRUE[layer-1])):
                            pgm.connect_fnodes(fname('f', ['x', layer, cnt]), fname('f', ['x', layer, cnt, lat]), fnode_xo_vars[str([layer-1, cnt, lat])])
                    for lat in range(len(W_TRUE[layer])):
                        dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])
                        dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])
                        pgm.connect_fnodes(fname('f', ['x', layer, cnt]), dlinear_factor_name, fnode_xi_vars[str([layer, cnt])])
                        pgm.connect_fnodes(fname('f', ['y', layer, cnt, lat]), dlinear_factor_name, fnode_y_vars[str([layer, cnt, lat])])
                        pgm.connect_fnodes(fname('f', ['w', layer, lat]), dlinear_factor_name, fnode_w_vars[str([layer, lat])])
                        pgm.connect_fnodes(fname('f', ['p', layer, lat]), dlinear_factor_name, fnode_p_vars[str([layer, lat])])
                        pgm.connect_fnodes(fname('f', ['y', layer, cnt, lat]), dsigmoid_factor_name, fnode_y_vars[str([layer, cnt, lat])])
                        pgm.connect_fnodes(fname('f', ['x', layer+1, cnt, lat]), dsigmoid_factor_name, fnode_xo_vars[str([layer, cnt, lat])])

        print 'connect_fnodes'
        connect_fnodes()

        def message_passing():
            def update_dsigmoid(layer, lat, cnt, CHECK):
                dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])

                try:
                    if CHECK : print "BEFORE INPUT \n{}".format(scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['input']))
                    if CHECK : print "BEFORE OUTPUT \n{}".format(scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['output']))
                except:
                    if CHECK : print "BEFORE \n EMPTY"
                    
                pgm.update_belief(dsigmoid_factor_name)
                
                if CHECK : print "AFTER INPUT \n{}".format(scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['input']))
                if CHECK : print "AFTER OUTPUT \n{}".format(scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['output']))
                pass

            def update_dlinear(layer, lat, cnt, CHECK):
                dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])
                pgm.update_belief(dlinear_factor_name)
                pass
                    
            def output_to_dsigmoid(layer, lat, cnt, CHECK):
                dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])
                pgm.update_message(fname('f', ['x', layer+1, cnt, lat]), dsigmoid_factor_name)
                pass

            def dsigmoid_to_dlinear(layer, lat, cnt, CHECK):
                dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])
                dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])

                pgm.update_message(dsigmoid_factor_name, fname('f', ['y', layer, cnt, lat]))
                
                if CHECK : print "DSIG -> Y \n{}".format(scalar_normal.to_standard(pgm.get_message_parameters(dsigmoid_factor_name, fname('f', ['y', layer, cnt, lat]))))
                
                pgm.update_belief(fname('f', ['y', layer, cnt, lat]))
                
                pgm.update_message(fname('f', ['y', layer, cnt, lat]), dlinear_factor_name)

                if CHECK : print "Y -> DLIN \n{}".format(scalar_normal.to_standard(pgm.get_message_parameters(fname('f', ['y', layer, cnt, lat]), dlinear_factor_name)['output']))
                
                pass

            def update_weights(layer, lat, samples, CHECK):
                for i in samples:
                    dlinear_factor_name = fname('f', ['x', layer, i], ['y', layer, i, lat], ['w', layer, lat], ['p', layer, lat])
                    pgm.update_message(dlinear_factor_name, fname('f', ['w', layer, lat]))
                    
                pgm.update_belief(fname('f', ['w', layer, lat]))
                
                for i in samples:
                    dlinear_factor_name = fname('f', ['x', layer, i], ['y', layer, i, lat], ['w', layer, lat], ['p', layer, lat])
                    pgm.update_message(fname('f', ['w', layer, lat]), dlinear_factor_name)
                    
                pass

            def update_modval(layer, lat, samples, CHECK):
                for i in samples:
                    dlinear_factor_name = fname('f', ['x', layer, i], ['y', layer, i, lat], ['w', layer, lat], ['p', layer, lat])
                    pgm.update_message(dlinear_factor_name, fname('f', ['p', layer, lat]))
                    
                pgm.update_belief(fname('f', ['p', layer, lat]))
                
                for i in samples:
                    dlinear_factor_name = fname('f', ['x', layer, i], ['y', layer, i, lat], ['w', layer, lat], ['p', layer, lat])
                    pgm.update_message(fname('f', ['p', layer, lat]), dlinear_factor_name)
                    
                pass

            def dlinear_to_input_vector(layer, lat, cnt, CHECK):
                dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])
                pgm.update_message(dlinear_factor_name, fname('f', ['x', layer, cnt]))
                pgm.update_belief(fname('f', ['x', layer, cnt]))
                pass

            def input_vector_to_previous_layer(layer, lat, cnt, CHECK):
                pgm.update_message(fname('f', ['x', layer+1, cnt]), fname('f', ['x', layer+1, cnt, lat]))
                pgm.update_belief(fname('f', ['x', layer+1, cnt, lat]))
                pass

            def input_vector_to_dlinear(layer, lat, cnt, CHECK):
                dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])
                pgm.update_message(fname('f', ['x', layer, cnt]), dlinear_factor_name)
                pass

            def dlinear_to_dsigmoid(layer, lat, cnt, CHECK):
                dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])
                dlinear_factor_name = fname('f', ['x', layer, cnt], ['y', layer, cnt, lat], ['w', layer, lat], ['p', layer, lat])
                pgm.update_message(dlinear_factor_name, fname('f', ['y', layer, cnt, lat]))
                pgm.update_belief(fname('f', ['y', layer, cnt, lat]))
                pgm.update_message(fname('f', ['y', layer, cnt, lat]), dsigmoid_factor_name)
                pass

            def dsigmoid_to_next_layer(layer, lat, cnt, CHECK):
                dsigmoid_factor_name = fname('f', ['y',layer, cnt, lat], ['x', layer+1, cnt, lat])
                pgm.update_message(dsigmoid_factor_name, fname('f', ['x', layer+1, cnt, lat]))
                pgm.update_belief(fname('f', ['x', layer+1, cnt, lat]))
                pgm.update_message(fname('f', ['x', layer+1, cnt, lat]), fname('f', ['x', layer+1, cnt]))
                pgm.update_belief(fname('f', ['x', layer+1, cnt]))
                pass

            def show_weights():
                for layer in range(len(W_TRUE))[::-1]:
                    for lat in range(len(W_TRUE[layer])):
                        print vector_normal.to_standard(pgm.get_belief_parameters(fname('f', ['w', layer, lat])), inplace=False)

            def show_modval():
                for layer in range(len(W_TRUE))[::-1]:
                    for lat in range(len(W_TRUE[layer])):
                        print pgm.get_belief_parameters(fname('f', ['p', layer, lat]))

            def show_accuracy():
                W_EST = []
                for layer in range(len(W_TRUE)):
                    layer_weights = []
                    for lat in range(len(W_TRUE[layer])):
                        layer_weights.append(vector_normal.to_standard(pgm.get_belief_parameters(fname('f', ['w', layer, lat])), inplace=False)['mean'])
                    W_EST.append(layer_weights)

                Z_PRED = X_MATRIX
                for layer in range(len(W_EST)):
                    Y_MATRIX = Z_PRED.dot(np.array(W_EST[layer]).T)
                    Z_PRED = 1./(1+np.exp(-Y_MATRIX))
                
                err_sq = np.mean((Z_PRED - Z_MATRIX)**2)
                print "SQUARE ERR = {}".format(err_sq)

                return err_sq

            def show_points():
                W_EST = []
                for layer in range(len(W_TRUE)):
                    layer_weights = []
                    for lat in range(len(W_TRUE[layer])):
                        print vector_normal.to_standard(pgm.get_belief_parameters(fname('f', ['w', layer, lat])), inplace=False)['mean']
                        layer_weights.append(vector_normal.to_standard(pgm.get_belief_parameters(fname('f', ['w', layer, lat])), inplace=False)['mean'])
                    W_EST.append(layer_weights)

                Z_PRED = X_MATRIX
                for layer in range(len(W_EST)):
                    Y_MATRIX = Z_PRED.dot(np.array(W_EST[layer]).T)
                    Z_PRED = 1./(1+np.exp(-Y_MATRIX))

                print "TRUE, PRED, ERROR"
                print np.round(np.column_stack((Z_MATRIX, Z_PRED, Z_MATRIX - Z_PRED)),3)


            def batch_train(sample_batch_size, layer_batch_size):
                CHECK = False
                samples = np.random.choice(np.arange(NUMBER_SAMPLES), sample_batch_size, replace=False)
                samples.sort()
                # BACKWARD
                for layer in range(len(W_TRUE))[::-1]:
                    #for lat in range(len(W_TRUE[layer])):
                    for lat in np.random.choice(range(len(W_TRUE[layer])), layer_batch_size):
                        for i in samples:
                            if layer != len(W_TRUE)-1:
                                if CHECK: print "input_vector_to_previous_layer"
                                input_vector_to_previous_layer(layer, lat, i, False)
                                if CHECK: print "output_to_dsigmoid"
                                output_to_dsigmoid(layer, lat, i, False)

                            if CHECK: print "update_dsigmoid"
                            update_dsigmoid(layer, lat, i, False)
                            if CHECK: print "dsigmoid_to_dlinear"
                            dsigmoid_to_dlinear(layer, lat, i, False)
                            if CHECK: print "update_dlinear"
                            update_dlinear(layer, lat, i, False)
                            
                        if CHECK: print "update_weights"
                        update_weights(layer, lat, samples, False)
                        
                        if CHECK: print "update_dlinear"
                        for i in samples: update_dlinear(layer, lat, i, False)
                        
                        if CHECK: print "update_modval"
                        update_modval(layer, lat, samples, False)

                for layer in range(len(W_TRUE))[::-1]:
                    #for lat in range(len(W_TRUE[layer])):
                    for lat in np.random.choice(range(len(W_TRUE[layer])), 1):
                        for i in samples:                        
                            if CHECK: print "update_dlinear"
                            update_dlinear(layer, lat, i, False)

                            if layer != 0:
                                if CHECK: print "dlinear_to_input_vector"
                                dlinear_to_input_vector(layer, lat, i, False)

                # FORWARD
                for layer in range(len(W_TRUE)):
                    #for lat in range(len(W_TRUE[layer])):
                    for lat in np.random.choice(range(len(W_TRUE[layer])), 1):
                        for i in samples:
                            if layer != 0:
                                if CHECK: print "input_vector_to_dlinear"
                                input_vector_to_dlinear(layer, lat, i, False)

                            if CHECK: print "update_dlinear"
                            update_dlinear(layer, lat, i, False)

                        if CHECK: print "update_weights"
                        update_weights(layer, lat, samples, False)
                        
                        if CHECK: print "update_dlinear"
                        for i in samples: update_dlinear(layer, lat, i, False)
                        
                        if CHECK: print "update_modval"
                        update_modval(layer, lat, samples, False)
                            
                for layer in range(len(W_TRUE)):
                    #for lat in range(len(W_TRUE[layer])):
                    for lat in np.random.choice(range(len(W_TRUE[layer])), 1):
                        for i in samples:
                            if CHECK: print "update_dlinear"
                            update_dlinear(layer, lat, i, False)
                            if CHECK: print "dlinear_to_dsigmoid"
                            dlinear_to_dsigmoid(layer, lat, i, False)
                            if CHECK: print "update_dsigmoid"
                            update_dsigmoid(layer, lat, i, False)

                            if layer != len(W_TRUE)-1:
                                if CHECK: print "dsigmoid_to_next_layer"
                                dsigmoid_to_next_layer(layer, lat, i, False)

            
            err = []            
            for counter in range(NUM_ITERATIONS):
                print "ITER {}".format(counter)
                
                batch_train(SAMPLE_BATCH_SIZE, LAYER_BATCH_SIZE)

                show_points()
                show_weights()
                show_modval()
                err.append(show_accuracy())
            
        message_passing()

        TEST.LOG("EXAMPLE COMPLETE", 1)



        '''
        from pypgm.factor import scalar_normal, vector_normal
        pgm.get_belief_parameters(dsigmoid_factor_name)
        print scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['input'])
        print scalar_normal.to_standard(pgm.get_belief_parameters(dsigmoid_factor_name)['output'])
        print scalar_normal.to_standard(pgm.get_message_parameters(dsigmoid_factor_name)['output'])
        '''
