from common import TEST

import unittest

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
        NUM_ITERATIONS = 100
        
        # Simulate samples
        W1_TRUE = [[0.8, -0.4], [1.1, -1.2]]
        W2_TRUE = [1.1, 0.1]
        NOISE = 0.001
        X_RANGE = 10
        X_TRUE = np.random.normal(0.5, X_RANGE, len(W1_TRUE))
        Y1_TRUE = X_TRUE.dot(W1_TRUE)
        Z1_TRUE = 1./(1+np.exp(-Y1_TRUE))
        Y2_TRUE = Z1_TRUE.dot(W2_TRUE)
        Z2_TRUE = 1./(1+np.exp(-Y2_TRUE))
        Z2_TRUE = Z2_TRUE + np.random.normal(0,Z2_TRUE,(np.shape(Z2_TRUE)))

        # Prior parameter values
        X_COV = [1e-5, 1e-5]
        W1_COV = [100, 100]
        Y1_VAR = 1e5
        Z1_VAR = 2
        Z1_COV = [1e5, 1e5]
        W2_COV = [1000, 1000]
        Y2_VAR = 1e5
        Z2_VAR = 1e-5
        P1_SHAPE, P1_SCALE = [1+1e-3, 1e3]
        P2_SHAPE, P2_SCALE = [1+1e-3, 1e3]
        
        # Imports used in simulation
        from pypgm import PGM
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal, scalar_normal
                    
        pgm = PGM('./sag/')

        def init_pgm():
            pgm.add_fnode('f(x)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                          vars=['x_1','x_2'],
                          mean=X_TRUE.tolist(),
                          cov=X_COV)

            pgm.add_fnode('f(w_1_1)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                          vars=['w_1_1_1','w_1_1_2'],
                          mean=[0]*2,
                          #mean=np.random.normal(0,1e-1,2).tolist(),
                          cov=W1_COV)
            pgm.add_fnode('f(w_1_2)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                          vars=['w_1_2_1','w_1_2_2'],
                          mean=[0]*2,
                          #mean=np.random.normal(0,1e-1,2).tolist(),
                          cov=W1_COV)

            pgm.add_fnode('f(p_1_1)',
                          ftype=FACTOR_TYPE.GAMMA,
                          vars=['p_1_1'],
                          shape=P1_SHAPE,
                          scale=P1_SCALE)
            pgm.add_fnode('f(p_1_2)',
                          ftype=FACTOR_TYPE.GAMMA,
                          vars=['p_1_2'],
                          shape=P1_SHAPE,
                          scale=P1_SCALE)

            pgm.add_fnode('f(y_1_1)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['y_1_1'],
                          mean=0,
                          var=Y1_VAR)
            pgm.add_fnode('f(y_1_2)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['y_1_2'],
                          mean=0,
                          var=Y1_VAR)
            
            pgm.add_fnode('f(x,y_1_1,w_1_1,p_1_1)',
                          ftype=FACTOR_TYPE.DLINEAR,
                          input_vars=['x_1', 'x_2'],
                          output_vars=['y_1_1'],
                          weight_vars=['w_1_1_1', 'w_1_1_2'],
                          modval_vars=['p_1_1'])
            pgm.add_fnode('f(x,y_1_2,w_1_2,p_1_2)',
                          ftype=FACTOR_TYPE.DLINEAR,
                          input_vars=['x_1', 'x_2'],
                          output_vars=['y_1_2'],
                          weight_vars=['w_1_2_1', 'w_1_2_2'],
                          modval_vars=['p_1_2'])

            pgm.add_fnode('f(z_1_1)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['z_1_1'],
                          mean=0.5,
                          var=Z1_VAR)
            pgm.add_fnode('f(z_1_2)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['z_1_2'],
                          mean=0.5,
                          var=Z1_VAR)

            pgm.add_fnode('f(y_1_1,z_1_1)',
                          ftype=FACTOR_TYPE.DSIGMOID,
                          input_vars=['y_1_1'],
                          output_vars=['z_1_1'])
            pgm.add_fnode('f(y_1_2,z_1_2)',
                          ftype=FACTOR_TYPE.DSIGMOID,
                          input_vars=['y_1_2'],
                          output_vars=['z_1_2'])

            pgm.add_fnode('f(z_1)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                          vars=['z_1_1','z_1_2'],
                          mean=[0.5]*2,
                          cov=Z1_COV)

            pgm.add_fnode('f(w_2)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                          vars=['w_2_1','w_2_2'],
                          mean=np.random.normal(0,1e-1,2).tolist(),
                          cov=W2_COV)

            pgm.add_fnode('f(p_2)',
                          ftype=FACTOR_TYPE.GAMMA,
                          vars=['p_2'],
                          shape=P2_SHAPE,
                          scale=P2_SCALE)

            pgm.add_fnode('f(y_2)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['y_2'],
                          mean=0,
                          var=Y2_VAR)

            pgm.add_fnode('f(z_1,y_2,w_2,p_2)',
                          ftype=FACTOR_TYPE.DLINEAR,
                          input_vars=['z_1_1', 'z_1_2'],
                          output_vars=['y_2'],
                          weight_vars=['w_2_1', 'w_2_2'],
                          modval_vars=['p_2'])

            pgm.add_fnode('f(z_2)',
                          ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['z_2'],
                          mean=Z2_TRUE,
                          var=Z2_VAR)

            pgm.add_fnode('f(y_2,z_2)',
                          ftype=FACTOR_TYPE.DSIGMOID,
                          input_vars=['y_2'],
                          output_vars=['z_2'])

        print 'init_pgm'
        init_pgm()

        def connect_fnodes():
            pgm.connect_fnodes('f(x)', 'f(x,y_1_1,w_1_1,p_1_1)', ['x_1','x_2'])
            pgm.connect_fnodes('f(x)', 'f(x,y_1_2,w_1_2,p_1_2)', ['x_1','x_2'])
            pgm.connect_fnodes('f(w_1_1)', 'f(x,y_1_1,w_1_1,p_1_1)', ['w_1_1_1','w_1_1_2'])
            pgm.connect_fnodes('f(w_1_2)', 'f(x,y_1_2,w_1_2,p_1_2)', ['w_1_2_1','w_1_2_2'])
            pgm.connect_fnodes('f(p_1_1)', 'f(x,y_1_1,w_1_1,p_1_1)', ['p_1_1'])
            pgm.connect_fnodes('f(p_1_2)', 'f(x,y_1_2,w_1_2,p_1_2)', ['p_1_2'])
            pgm.connect_fnodes('f(y_1_1)', 'f(x,y_1_1,w_1_1,p_1_1)', ['y_1_1'])
            pgm.connect_fnodes('f(y_1_2)', 'f(x,y_1_2,w_1_2,p_1_2)', ['y_1_2'])
            
            pgm.connect_fnodes('f(y_1_1)', 'f(y_1_1,z_1_1)', ['y_1_1'])
            pgm.connect_fnodes('f(y_1_2)', 'f(y_1_2,z_1_2)', ['y_1_2'])
            pgm.connect_fnodes('f(z_1_1)', 'f(y_1_1,z_1_1)', ['z_1_1'])
            pgm.connect_fnodes('f(z_1_2)', 'f(y_1_2,z_1_2)', ['z_1_2'])

            pgm.connect_fnodes('f(z_1_1)', 'f(z_1)', ['z_1_1'])
            pgm.connect_fnodes('f(z_1_2)', 'f(z_1)', ['z_1_2'])

            pgm.connect_fnodes('f(z_1)', 'f(z_1,y_2,w_2,p_2)', ['z_1_1','z_1_2'])
            pgm.connect_fnodes('f(w_2)', 'f(z_1,y_2,w_2,p_2)', ['w_2_1','w_2_2'])
            pgm.connect_fnodes('f(p_2)', 'f(z_1,y_2,w_2,p_2)', ['p_2'])
            pgm.connect_fnodes('f(y_2)', 'f(z_1,y_2,w_2,p_2)', ['y_2'])

            pgm.connect_fnodes('f(y_2)', 'f(y_2,z_2)', ['y_2'])
            pgm.connect_fnodes('f(z_2)', 'f(y_2,z_2)', ['z_2'])

        print 'connect_fnodes'
        connect_fnodes()

        def message_passing():
            def update_dlinear_l1(var):
                pgm.update_belief('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))

            def update_weights_l1(var):
                pgm.update_message('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var), 'f(w_1_{})'.format(var))
                pgm.update_belief('f(w_1_{})'.format(var))
                pgm.update_message('f(w_1_{})'.format(var), 'f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))

            def update_modval_l1(var):
                pgm.update_message('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var), 'f(p_1_{})'.format(var))
                pgm.update_belief('f(p_1_{})'.format(var))
                pgm.update_message('f(p_1_{})'.format(var), 'f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))

            def l1_to_l2(var):
                pgm.update_message('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var), 'f(y_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var), 'f(y_1_{})'.format(var)))
                pgm.update_belief('f(y_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_belief_parameters('f(y_1_{})'.format(var)))
                pgm.update_message('f(y_1_{})'.format(var), 'f(y_1_{},z_1_{})'.format(var,var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(y_1_{})'.format(var), 'f(y_1_{},z_1_{})'.format(var,var))['input'])
                pgm.update_belief('f(y_1_{},z_1_{})'.format(var,var))
                pgm.update_message('f(y_1_{},z_1_{})'.format(var,var), 'f(z_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(y_1_{},z_1_{})'.format(var,var), 'f(z_1_{})'.format(var)))
                pgm.update_belief('f(z_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_belief_parameters('f(z_1_{})'.format(var)))
                pgm.update_message('f(z_1_{})'.format(var),'f(z_1)')
                print vector_normal.to_standard(pgm.get_message_parameters('f(z_1_{})'.format(var),'f(z_1)'))

            def update_l2_vec():
                pgm.update_belief('f(z_1)')

            def l2_to_l1(var):
                pgm.update_message('f(z_1)','f(z_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(z_1)','f(z_1_{})'.format(var)))
                pgm.update_belief('f(z_1_{})'.format(var))
                pgm.update_message('f(z_1_{})'.format(var), 'f(y_1_{},z_1_{})'.format(var,var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(z_1_{})'.format(var), 'f(y_1_{},z_1_{})'.format(var,var))['output'])
                pgm.update_belief('f(y_1_{},z_1_{})'.format(var,var))
                pgm.update_message('f(y_1_{},z_1_{})'.format(var,var), 'f(y_1_{})'.format(var))
                print scalar_normal.to_standard(pgm.get_message_parameters('f(y_1_{},z_1_{})'.format(var,var), 'f(y_1_{})'.format(var)))
                pgm.update_belief('f(y_1_{})'.format(var))
                pgm.update_message('f(y_1_{})'.format(var), 'f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))

            def update_input_l2():
                pgm.update_message('f(z_1)', 'f(z_1,y_2,w_2,p_2)')

            def update_dlinear_l2():
                pgm.update_belief('f(z_1,y_2,w_2,p_2)')

            def update_weights_l2():
                pgm.update_message('f(z_1,y_2,w_2,p_2)', 'f(w_2)')
                pgm.update_belief('f(w_2)')
                pgm.update_message('f(w_2)', 'f(z_1,y_2,w_2,p_2)')

            def update_modval_l2():
                pgm.update_message('f(z_1,y_2,w_2,p_2)', 'f(p_2)')
                pgm.update_belief('f(p_2)')
                pgm.update_message('f(p_2)', 'f(z_1,y_2,w_2,p_2)')


            def show_dlinear_l1(var):
                print vector_normal.to_standard(pgm.get_belief_parameters('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))['input'])
                print scalar_normal.to_standard(pgm.get_belief_parameters('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))['output'])
                print vector_normal.to_standard(pgm.get_belief_parameters('f(x,y_1_{},w_1_{},p_1_{})'.format(var,var,var))['weight'])

            def show_dlinear_l2():
                print vector_normal.to_standard(pgm.get_belief_parameters('f(z_1,y_2,w_2,p_2)')['input'])
                print scalar_normal.to_standard(pgm.get_belief_parameters('f(z_1,y_2,w_2,p_2)')['output'])
                print vector_normal.to_standard(pgm.get_belief_parameters('f(z_1,y_2,w_2,p_2)')['weight'])


                
            # Initialize output of dlinear
            pgm.update_belief('f(z_2)')
            pgm.update_message('f(z_2)', 'f(y_2,z_2)')
            pgm.update_belief('f(y_2,z_2)')
            pgm.update_message('f(y_2,z_2)', 'f(y_2)')
            pgm.update_belief('f(y_2)')
            pgm.update_message('f(y_2)', 'f(z_1,y_2,w_2,p_2)')
            update_dlinear_l2()
            show_dlinear_l2()

            import ipdb
            ipdb.set_trace()
            for counter in range(NUM_ITERATIONS):
                print "ITER {}".format(counter)

                update_dlinear_l1(1)
                update_dlinear_l1(2)
                
                show_dlinear_l1(1)
                show_dlinear_l1(2)

                update_weights_l1(1)
                update_weights_l1(2)

                update_dlinear_l1(1)
                update_dlinear_l1(2)

                show_dlinear_l1(1)
                show_dlinear_l1(2)

                update_modval_l1(1)
                update_modval_l1(2)

                update_dlinear_l1(1)
                update_dlinear_l1(2)

                l1_to_l2(1)
                l1_to_l2(2)

                update_l2_vec()

                update_input_l2()

                update_dlinear_l2()

                show_dlinear_l2()

                update_weights_l2()
                
                update_dlinear_l2()

                show_dlinear_l2()
                
                update_modval_l2()
                
                update_dlinear_l2()

                show_dlinear_l2()

                l2_to_l1(1)
                l2_to_l1(2)

                update_dlinear_l1(1)
                update_dlinear_l1(2)

                update_weights_l1(1)
                update_weights_l1(2)
                
                update_dlinear_l1(1)
                update_dlinear_l1(2)
                
                update_modval_l1(1)
                update_modval_l1(2)


                W1_EST = [vector_normal.to_standard(pgm.get_belief_parameters('f(w_1_1)'))['mean'],
                          vector_normal.to_standard(pgm.get_belief_parameters('f(w_1_2)'))['mean']]
                W2_EST = vector_normal.to_standard(pgm.get_belief_parameters('f(w_2)'))['mean']
                print "EVALUATE"
                print vector_normal.to_standard(pgm.get_belief_parameters('f(w_1_1)'))
                print vector_normal.to_standard(pgm.get_belief_parameters('f(w_1_2)'))
                print vector_normal.to_standard(pgm.get_belief_parameters('f(w_2)'))
                print W1_TRUE
                print W2_TRUE
                Y1_PRED = X_TRUE.dot(W1_EST)
                Z1_PRED = 1./(1+np.exp(-Y1_PRED))
                Y2_PRED = Z1_PRED.dot(W2_EST)
                Z2_PRED = 1./(1+np.exp(-Y2_PRED))
                print Z2_PRED
                print Z2_TRUE

        message_passing()

        TEST.LOG("EXAMPLE COMPLETE", 1)
