from common import TEST

import unittest

import pylab as plt

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


def padstr(integer):
    return str(integer).zfill(4)


class test_linear_regression(unittest.TestCase):
    def tearDown(self):
        """Destroys necessary variables"""

        import shutil

        shutil.rmtree('./sad/')
        
        
    def test(self):
        """Consider linear regression simulation"""

        TEST.LOG("START - EXAMPLE", 1)
        TEST.LOG(self.test.__doc__, 2)

        import numpy as np
        np.random.seed(0)

        # Global variables used in simulations
        NUMBER_SAMPLES = 10
        NUM_ITERATIONS = 10
        SHOW_IMAGES = True

        # Simulate samples
        W_TRUE = [3, 4]
        Y_NOISE = 1
        X_RANGE = 5
        X_MATRIX = np.random.normal(0, X_RANGE, (NUMBER_SAMPLES, len(W_TRUE)))
        Y_VEC = (X_MATRIX.dot(W_TRUE) + np.random.normal(0, Y_NOISE, NUMBER_SAMPLES))

        # Prior parameter values
        X_COV = 0.01
        X_COV_FORM = 'FULL'
        W_COV = 100
        W_COV_FORM = 'FULL'
        Y_VAR = 0.01
        P_SHAPE, P_SCALE = [1+1e-3, 1]

        # Range and resolution used for generating output
        [W_MID, W_RANGE, W_INCR]  = [0, 10, 1.0]
        [M_START, M_STOP, M_INCR] = [0, 50, 1.0]

        # Values used to evaluate posterior distributions
        W_EVAL_VALUES = np.arange(W_MID - W_RANGE, W_MID + W_RANGE+1e-10, W_INCR)
        M_EVAL_VALUES = np.arange(M_START, M_STOP+1e-10, M_INCR)

        # Imports used in simulation
        import time
                
        from utils.sdict import SDict
        
        from pypgm import PGM
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal, scalar_normal, gamma
        
        # Adapt covariance matrices according to given covariance matrix forms
        if X_COV_FORM == 'COMMON':
            X_COV_FORM = vector_normal.COVARIANCE_FORM.COMMON
        elif X_COV_FORM == 'DIAGONAL':
            X_COV = [X_COV]*len(W_TRUE)
            X_COV_FORM = vector_normal.COVARIANCE_FORM.DIAGONAL
        elif X_COV_FORM == 'FULL':
            X_COV = np.diag([X_COV]*len(W_TRUE)).tolist()
            X_COV_FORM = vector_normal.COVARIANCE_FORM.FULL

        if W_COV_FORM == 'COMMON':
            W_COV_FORM = vector_normal.COVARIANCE_FORM.COMMON
        elif W_COV_FORM == 'DIAGONAL':
            W_COV = [W_COV]*len(W_TRUE)
            W_COV_FORM = vector_normal.COVARIANCE_FORM.DIAGONAL
        elif W_COV_FORM == 'FULL':
            W_COV = np.diag([W_COV]*len(W_TRUE)).tolist()
            W_COV_FORM = vector_normal.COVARIANCE_FORM.FULL
        
        pgm = PGM('./sad/')
        
        def init_pgm():
            fnode_w_vars = ['w_'+padstr(i) for i in range(len(W_TRUE))]
            fnode_p_vars = ['p']
            pgm.add_fnode('f(w)',
                          ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=W_COV_FORM,
                          vars=fnode_w_vars,
                          mean=[0]*len(W_TRUE),
                          cov=W_COV)
            pgm.add_fnode('f(p)',
                          ftype=FACTOR_TYPE.GAMMA,
                          vars=['p'],
                          shape=P_SHAPE,
                          scale=P_SCALE)
            
            for cnt in range(NUMBER_SAMPLES):
                fnode_x_vars = ['x_'+padstr(cnt)+"_"+padstr(i) for i in range(len(W_TRUE))]
                fnode_y_vars = ['y_'+padstr(cnt)]
                pgm.add_fnode(fname('f', ['x',cnt]),
                              ftype=FACTOR_TYPE.VECTOR_NORMAL,
                              par_form=vector_normal.PARAMETER_FORM.STANDARD,
                              cov_form=X_COV_FORM,
                              vars=fnode_x_vars,
                              mean=X_MATRIX[cnt].tolist(),
                              cov=X_COV)
                pgm.add_fnode(fname('f', ['y',cnt]),
                              ftype=FACTOR_TYPE.SCALAR_NORMAL,
                              par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                              vars=fnode_y_vars,
                              mean=Y_VEC[cnt],
                              var=Y_VAR)
                pgm.add_fnode(fname('f', ['x',cnt], ['y',cnt], ['w'], ['p']),
                              ftype=FACTOR_TYPE.DLINEAR,
                              input_vars=fnode_x_vars,
                              output_vars=fnode_y_vars,
                              weight_vars=fnode_w_vars,
                              modval_vars=fnode_p_vars)

        print 'init_pgm'
        init_pgm()

        def connect_fnodes():
            fnode_w_vars = ['w_'+padstr(i) for i in range(len(W_TRUE))]
            fnode_p_vars = ['p']
            for cnt in range(NUMBER_SAMPLES):
                fnode_x_vars = ['x_'+padstr(cnt)+"_"+padstr(i) for i in range(len(W_TRUE))]
                fnode_y_vars = ['y_'+padstr(cnt)]                
                factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                pgm.connect_fnodes(fname('f', ['x',cnt]), factor_name, fnode_x_vars)
                pgm.connect_fnodes(fname('f', ['y',cnt]), factor_name, fnode_y_vars)
                pgm.connect_fnodes('f(w)', factor_name, fnode_w_vars)
                pgm.connect_fnodes('f(p)', factor_name, fnode_p_vars)

        print 'connect_fnodes'
        connect_fnodes()

        def message_passing():
            if SHOW_IMAGES:
                weights_belief = pgm.get_belief_parameters('f(w)')
                plt.title('Prior')
                plt.imshow(vector_normal.calculate_pdf(W_EVAL_VALUES, **weights_belief), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W_EVAL_VALUES)), np.round(np.linspace(min(W_EVAL_VALUES), max(W_EVAL_VALUES), len(W_EVAL_VALUES)),0))
                plt.yticks(np.arange(len(W_EVAL_VALUES)), np.round(np.linspace(min(W_EVAL_VALUES), max(W_EVAL_VALUES), len(W_EVAL_VALUES)),0))
                plt.savefig('./images/examples/sad_w_' + padstr(0))
                plt.close()

                modval_belief = pgm.get_belief_parameters('f(p)')
                plt.title('Prior')
                plt.plot(M_EVAL_VALUES, gamma.calculate_pdf(M_EVAL_VALUES, **modval_belief))
                plt.savefig('./images/examples/sad_p_' + padstr(0))
                plt.close()
            
            for counter in range(NUM_ITERATIONS):
                print "ITER {}".format(counter)

                # Update dlinear
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_belief(factor_name)

                # Messages on W
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_message(factor_name, 'f(w)')
                pgm.update_belief('f(w)')
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_message('f(w)', factor_name)

                # Update dlinear
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_belief(factor_name)

                # Messages on P
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_message(factor_name, 'f(p)')
                pgm.update_belief('f(p)')
                for cnt in range(NUMBER_SAMPLES):
                    factor_name = fname('f', ['x',cnt], ['y',cnt], ['w'], ['p'])
                    pgm.update_message('f(p)', factor_name)

                weights_belief = vector_normal.to_standard(pgm.get_belief_parameters('f(w)'))
                print weights_belief
                print pgm.get_belief_parameters('f(p)')
                print np.mean((Y_VEC - X_MATRIX.dot(weights_belief['mean']))**2)

                if SHOW_IMAGES:
                    weights_belief = pgm.get_belief_parameters('f(w)')
                    plt.title('Belief')
                    plt.imshow(vector_normal.calculate_pdf(W_EVAL_VALUES, **weights_belief), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W_EVAL_VALUES)), np.round(np.linspace(min(W_EVAL_VALUES), max(W_EVAL_VALUES), len(W_EVAL_VALUES)),0))
                    plt.yticks(np.arange(len(W_EVAL_VALUES)), np.round(np.linspace(min(W_EVAL_VALUES), max(W_EVAL_VALUES), len(W_EVAL_VALUES)),0))
                    plt.savefig('./images/examples/sad_w_' + padstr(counter+1))
                    plt.close()

                    modval_belief = pgm.get_belief_parameters('f(p)')
                    plt.title('Belief')
                    plt.plot(M_EVAL_VALUES, gamma.calculate_pdf(M_EVAL_VALUES, **modval_belief))
                    plt.savefig('./images/examples/sad_p_' + padstr(counter+1))
                    plt.close()

        message_passing()
        
        TEST.LOG("EXAMPLE COMPLETE", 1)
