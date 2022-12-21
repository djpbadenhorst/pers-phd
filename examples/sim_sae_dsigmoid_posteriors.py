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
            plt.title(method_key)
            line = pdfs[component_key][method_key]
            plt.plot(eval_values, line)
            
        plt.savefig('./images/examples/sae_'+'_'.join([component_key]))
        plt.close()

        
class test_dsigmoid_posterior(unittest.TestCase):       
    def test(self):
        """Consider estimations of dsigmoid posteriors
            input: scalar_normal
            output: scalar_normal
        """

        TEST.LOG("START - EXAMPLE", 1)
        TEST.LOG(self.test.__doc__, 2)
        
        import numpy as np
        np.random.seed(0)

        # Booleans used to indicate which outputs to genereate
        PARAMETER = True

        # Prior parameter values
        I_MEAN, I_VAR = [1, 1.1]
        O_MEAN, O_VAR = [0.6, 0.01]

        # Range and resolution used for generating output
        I_MID = 0
        I_RANGE = 5
        I_INCR = 0.2
        O_MID = 0.5
        O_RANGE = 1
        O_INCR = 0.02

        # Values used to evaluate posterior distributions
        I_EVAL_VALUES = np.arange(I_MID - I_RANGE, I_MID + I_RANGE+1e-10, I_INCR)
        O_EVAL_VALUES = np.arange(O_MID - O_RANGE, O_MID + O_RANGE+1e-10, O_INCR)

        # Imports used in simulation
        from copy import deepcopy
        
        from pypgm.factor import FACTOR_TYPE, scalar_normal, dsigmoid

        # Initialization of factor parameter dictionaries
        input_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'vars': ['y'],
            'mean': I_MEAN,
            'var': I_VAR,
        }
        scalar_normal.inpl_to_canonical(input_component)
        
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'vars': ['z'],
            'mean': O_MEAN,
            'var': O_VAR,
        }
        scalar_normal.inpl_to_canonical(output_component)

        dsigmoid_parameters = {
            'input_vars' : ['y'],
            'input': input_component,
            'output_vars' : ['z'],
            'output': output_component,
        }
        
        def create_estimate_data_dictionaries():
            print 'Estimating parameters by transforming samples'
            tmp_input_parameters = dsigmoid.estimate_input_component(deepcopy(dsigmoid_parameters))
            tmp_output_parameters = dsigmoid.estimate_output_component(deepcopy(dsigmoid_parameters))
            
            tmp_input = {'trans_est' : scalar_normal.calculate_pdf(I_EVAL_VALUES, **tmp_input_parameters['input'])}
            tmp_output = {'trans_est' : scalar_normal.calculate_pdf(O_EVAL_VALUES, **tmp_output_parameters['output'])}
            return [tmp_input, tmp_output]


        # Create empty dictionaries to be filled
        pdfs = {
            'input':{'eval_values': I_EVAL_VALUES},
            'output':{'eval_values': O_EVAL_VALUES},
        }
        
        if PARAMETER:
            tmp_input, tmp_output = create_estimate_data_dictionaries()
            pdfs['input'].update(tmp_input)
            pdfs['output'].update(tmp_output)

        show_results(pdfs)
        
        TEST.LOG("EXAMPLE COMPLETE", 1)
