from common import TEST

import unittest


def show_results(pdfs):
    import numpy as np
    import pylab as plt
    import scipy as sci

    for component_key in pdfs.keys():
        print 'Showing results for ' + component_key
        eval_values = pdfs[component_key].pop('eval_values')
        
        plt.figure(figsize=(20,8))
        plt.suptitle(component_key)
        for cnt, method_key in enumerate(pdfs[component_key].keys()):
            if component_key == 'input' or component_key == 'weight':
                plt.subplot(2,3,cnt+1)
                plt.title(method_key)
                image = pdfs[component_key][method_key]
                plt.imshow(image, origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(image)), np.round(np.linspace(min(eval_values), max(eval_values), len(image)),0))
                plt.yticks(np.arange(len(image)), np.round(np.linspace(min(eval_values), max(eval_values), len(image)),0))

            elif component_key == 'output' or component_key == 'modval':
                plt.subplot(2,3,cnt+1)
                plt.title(method_key)
                line = pdfs[component_key][method_key]
                plt.plot(line)
                plt.xticks(np.arange(len(line)), np.round(np.linspace(min(eval_values), max(eval_values), len(line)),0))

        plt.savefig('./images/examples/sac_'+'_'.join([component_key]))
        plt.close()
            
                
class test_dlinear_posteriors(unittest.TestCase):       
    def test(self):
        """Consider estimations of dlinear posteriors
            input: vector_normal
            output: scalar_normal
            weight: vector_normal
            modval: gamma
        """

        TEST.LOG("START - EXAMPLE", 1)
        TEST.LOG(self.test.__doc__, 2)

        import numpy as np
        np.random.seed(0)
        
        # Booleans used to indicate which outputs to genereate
        EXACT = True
        SAMPLE = True
        ESTIMATE = True
        ESTIMATE_ANALYTICAL = True

        # Prior parameter values
        I_COV_FORM = 'FULL'
        I_MEAN, I_COV = [
            [1., 2.],
            0.1
        ]
        O_MEAN, O_VAR = [3, 0.1]
        W_COV_FORM = 'FULL'
        W_MEAN, W_COV = [
            [0., 0.],
            100
        ]
        M_SHAPE, M_SCALE = [1+1e-3,1e3]

        # Range and resolution used for generating output
        [I_MID, I_RANGE, I_INCR]  = [1, 4,  0.5]
        [O_MID, O_RANGE, O_INCR]  = [3, 4,  0.5]
        [W_MID, W_RANGE, W_INCR]  = [0, 10, 1.0]
        [M_START, M_STOP, M_INCR] = [0, 50, 1.0]

        # Number of samples to use to generate output 
        NUMBER_SAMPLES = 1000

        # Values used to evaluate posterior distributions
        I_EVAL_VALUES = np.arange(I_MID - I_RANGE, I_MID + I_RANGE+1e-10, I_INCR)
        O_EVAL_VALUES = np.arange(O_MID - O_RANGE, O_MID + O_RANGE+1e-10, O_INCR)
        W_EVAL_VALUES = np.arange(W_MID - W_RANGE, W_MID + W_RANGE+1e-10, W_INCR)
        M_EVAL_VALUES = np.arange(M_START, M_STOP+1e-10, M_INCR)

        # Imports used in simulation
        from copy import deepcopy
        
        from pypgm.factor import FACTOR_TYPE

        from pypgm.factor import vector_normal, scalar_normal, dlinear, gamma

        # Adapt covariance matrices according to given covariance matrix forms
        if I_COV_FORM == 'COMMON':
            I_COV_FORM = vector_normal.COVARIANCE_FORM.COMMON
        elif I_COV_FORM == 'DIAGONAL':
            I_COV = [I_COV]*2
            I_COV_FORM = vector_normal.COVARIANCE_FORM.DIAGONAL
        elif I_COV_FORM == 'FULL':
            I_COV = np.diag([I_COV]*2).tolist()
            I_COV_FORM = vector_normal.COVARIANCE_FORM.FULL

        if W_COV_FORM == 'COMMON':
            W_COV_FORM = vector_normal.COVARIANCE_FORM.COMMON
        elif W_COV_FORM == 'DIAGONAL':
            W_COV = [W_COV]*2
            W_COV_FORM = vector_normal.COVARIANCE_FORM.DIAGONAL
        elif W_COV_FORM == 'FULL':
            W_COV = np.diag([W_COV]*2).tolist()
            W_COV_FORM = vector_normal.COVARIANCE_FORM.FULL
            
        # Initialization of factor parameter dictionaries
        input_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': I_COV_FORM,
            'vars': ['x1', 'x2'],
            'mean': I_MEAN,
            'cov': I_COV,
        }
        vector_normal.inpl_to_canonical(input_component)
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'vars': ['y'],
            'mean': O_MEAN,
            'var': O_VAR,
        }
        scalar_normal.inpl_to_canonical(output_component)
        weight_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': W_COV_FORM,
            'vars': ['w1', 'w2'],
            'mean': W_MEAN,
            'cov': W_COV,
        }
        vector_normal.inpl_to_canonical(weight_component)
        modval_component = {
            'ftype': FACTOR_TYPE.GAMMA,
            'vars': ['p'],
            'shape': M_SHAPE,
            'scale': M_SCALE,
        }
        dlinear_parameters = {
            'input' : input_component,
            'output' : output_component,
            'weight' : weight_component,
            'modval' : modval_component,
        }

        def create_exact_data_dictionaries():
            print 'Calculate exact posterior for input'
            tmp = dlinear.calculate_pdf(
                I_EVAL_VALUES, O_EVAL_VALUES, W_EVAL_VALUES, M_EVAL_VALUES,
                'input', **dlinear_parameters)
            tmp_input = {
                'prior' : vector_normal.calculate_pdf(I_EVAL_VALUES, **input_component),
                'posterior' : tmp[1],
            }
            
            print 'Calculate exact posterior for output'
            tmp = dlinear.calculate_pdf(
                I_EVAL_VALUES, O_EVAL_VALUES, W_EVAL_VALUES, M_EVAL_VALUES,
                'output', **dlinear_parameters)
            tmp_output = {
                'prior' : scalar_normal.calculate_pdf(O_EVAL_VALUES, **output_component),
                'posterior' : tmp[1],
            }
            
            print 'Calculate exact posterior for weight'
            tmp = dlinear.calculate_pdf(
                I_EVAL_VALUES, O_EVAL_VALUES, W_EVAL_VALUES, M_EVAL_VALUES,
                'weight', **dlinear_parameters)
            tmp_weight = {
                'prior' : vector_normal.calculate_pdf(W_EVAL_VALUES, **weight_component),
                'posterior' : tmp[1],
            }
            
            print 'Calculate exact posterior for modval'
            tmp = dlinear.calculate_pdf(
                I_EVAL_VALUES, O_EVAL_VALUES, W_EVAL_VALUES, M_EVAL_VALUES,
                'modval', **dlinear_parameters)
            tmp_modval = {
                'prior' : gamma.calculate_pdf(M_EVAL_VALUES, **modval_component),
                'posterior' : tmp[1],
            }

            return [tmp_input, tmp_output, tmp_weight, tmp_modval]

        def create_sample_data_dictionaries():
            print 'Sampling from full posterior'
            tmp_samples = dlinear.sample_from_posteriors(deepcopy(dlinear_parameters))
        
            tmp_input = {'sample' :
                         np.histogram2d(tmp_samples['input_samples'][:,0], tmp_samples['input_samples'][:,1],
                                        bins=50, range = [[min(I_EVAL_VALUES), max(I_EVAL_VALUES)]]*2)[0]
            }
            tmp_output = {'sample' :
                          np.histogram(tmp_samples['output_samples'], bins=20, 
                                        range = [min(O_EVAL_VALUES), max(O_EVAL_VALUES)])[0]
            }
            tmp_weight = {'sample' :
                          np.histogram2d(tmp_samples['weight_samples'][:,0], tmp_samples['weight_samples'][:,1],
                                         bins=50, range = [[min(W_EVAL_VALUES), max(W_EVAL_VALUES)]]*2)[0]
            }
            tmp_modval = {'sample' :
                          np.histogram(tmp_samples['modval_samples'], bins=20, 
                                       range = [min(M_EVAL_VALUES), max(M_EVAL_VALUES)])[0]
            }

            return [tmp_input, tmp_output, tmp_weight, tmp_modval]

        def create_estimate_data_dictionaries():
            print 'Estimating parameters by optimizing kl divergence'
            tmp_parameters = dlinear.kl_optimization_iow(deepcopy(dlinear_parameters))
            
            tmp_input = {'kl_est' : vector_normal.calculate_pdf(I_EVAL_VALUES, **tmp_parameters['input'])}
            tmp_output = {'kl_est' : scalar_normal.calculate_pdf(O_EVAL_VALUES, **tmp_parameters['output'])}
            tmp_weight = {'kl_est' : vector_normal.calculate_pdf(W_EVAL_VALUES, **tmp_parameters['weight'])}

            tmp_parameters = dlinear.kl_optimization_iowm(deepcopy(dlinear_parameters))
            tmp_modval = {'kl_est' : gamma.calculate_pdf(M_EVAL_VALUES, **tmp_parameters['modval'])}

            return [tmp_input, tmp_output, tmp_weight, tmp_modval]

        def create_estimate_analytical_data_dictionaries():
            print 'Estimating parameters by optimizing kl divergence analytically'
            
            tmp_parameters = dlinear.kl_optimization_i(deepcopy(dlinear_parameters))            
            tmp_input = {'kl_est_analyt' : vector_normal.calculate_pdf(I_EVAL_VALUES, **tmp_parameters['input'])}

            tmp_parameters = dlinear.kl_optimization_o(deepcopy(dlinear_parameters))
            tmp_output = {'kl_est_analyt' : scalar_normal.calculate_pdf(O_EVAL_VALUES, **tmp_parameters['output'])}

            tmp_parameters = dlinear.kl_optimization_w(deepcopy(dlinear_parameters))
            tmp_weight = {'kl_est_analyt' : vector_normal.calculate_pdf(W_EVAL_VALUES, **tmp_parameters['weight'])}

            tmp_parameters = dlinear.kl_optimization_m(deepcopy(dlinear_parameters))
            tmp_modval = {'kl_est_analyt' : gamma.calculate_pdf(M_EVAL_VALUES, **tmp_parameters['modval'])}

            return [tmp_input, tmp_output, tmp_weight, tmp_modval]


        # Create empty dictionaries to be filled
        pdfs = {
            'input':{'eval_values': I_EVAL_VALUES},
            'output':{'eval_values': O_EVAL_VALUES},
            'weight':{'eval_values': W_EVAL_VALUES},
            'modval':{'eval_values': M_EVAL_VALUES},
        }
        
        if EXACT:
            tmp_input, tmp_output, tmp_weight, tmp_modval = create_exact_data_dictionaries()
            pdfs['input'].update(tmp_input)
            pdfs['output'].update(tmp_output)
            pdfs['weight'].update(tmp_weight)
            pdfs['modval'].update(tmp_modval)
        
        if SAMPLE:
            tmp_input, tmp_output, tmp_weight, tmp_modval = create_sample_data_dictionaries()
            pdfs['input'].update(tmp_input)
            pdfs['output'].update(tmp_output)
            pdfs['weight'].update(tmp_weight)
            pdfs['modval'].update(tmp_modval)
        
        if ESTIMATE:
            tmp_input, tmp_output, tmp_weight, tmp_modval = create_estimate_data_dictionaries()
            pdfs['input'].update(tmp_input)
            pdfs['output'].update(tmp_output)
            pdfs['weight'].update(tmp_weight)
            pdfs['modval'].update(tmp_modval)

        if ESTIMATE_ANALYTICAL:
            tmp_input, tmp_output, tmp_weight, tmp_modval = create_estimate_analytical_data_dictionaries()
            pdfs['input'].update(tmp_input)
            pdfs['output'].update(tmp_output)
            pdfs['weight'].update(tmp_weight)
            pdfs['modval'].update(tmp_modval)

        show_results(pdfs)
        
        TEST.LOG("EXAMPLE COMPLETE", 1)
