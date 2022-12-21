from common import TEST

import unittest


class test_sample(unittest.TestCase):
    """Class containing tests for pypgm/factor/dlinear/sample.py"""
    
    def test_01(self):
        """Tests the following :
        1) Functionality of sample_from_posteriors
            input: vector_normal
            output: scalar_normal
            weight: vector_normal
            modval: gamma
        """

        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_SAMPLE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        np.random.seed(0)

        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal, scalar_normal

        input_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [1, 2],
            'prec': [[1, 0.1], [0.1, 1]],
        }
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3,
            'prec': 1,
        }
        weight_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [4, 5],
            'prec': [[1, 0.1], [0.1, 1]],
        }
        modval_component = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6,
            'scale': 1./25,
        }
        parameter_set = {
            'input': input_component,
            'output': output_component,
            'weight': weight_component,
            'modval': modval_component, 
        }

        posterior_samples = dlinear.sample_from_posteriors(parameter_set)
        
        input_mean = np.mean(posterior_samples['input_samples'], 0)
        output_mean = np.mean(posterior_samples['output_samples'])
        weight_mean = np.mean(posterior_samples['weight_samples'], 0)
        modval_mean = np.mean(posterior_samples['modval_samples'])

        input_cov = np.cov(np.array(posterior_samples['input_samples']).transpose())
        output_var = np.var(posterior_samples['output_samples'])
        weight_cov = np.cov(np.array(posterior_samples['weight_samples']).transpose())
        modval_var = np.var(posterior_samples['modval_samples'])
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT MEAN", 3)
        TEST.LOG(input_mean, 3)
        TEST.LOG("INPUT COV", 3)
        TEST.LOG(input_cov, 3)
        TEST.LOG("OUTPUT MEAN", 3)
        TEST.LOG(output_mean, 3)
        TEST.LOG("OUTPUT VAR", 3)
        TEST.LOG(output_var, 3)
        TEST.LOG("WEIGHT MEAN", 3)
        TEST.LOG(weight_mean, 3)
        TEST.LOG("WEIGHT COV", 3)
        TEST.LOG(weight_cov, 3)
        TEST.LOG("MODVAL MEAN", 3)
        TEST.LOG(modval_mean, 3)
        TEST.LOG("MODVAL VAR", 3)
        TEST.LOG(modval_var, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = [0.121777, 0.978102]
        TEST.EQUALS(
            self, input_mean, tmp,
            "Error 01")
        tmp = [[ 0.722959,-0.405983],
               [-0.405983, 0.574698]]
        TEST.EQUALS(
            self, input_cov, tmp,
            "Error 02")
        TEST.EQUALS(
            self, output_mean, 3.164425,
            "Error 03")
        TEST.EQUALS(
            self, output_var, 0.914897,
            "Error 04")
        tmp = [3.419752, 4.352368]
        TEST.EQUALS(
            self, weight_mean, tmp,
            "Error 05")
        tmp = [[ 1.033090,-0.065392],
               [-0.065392, 1.051764]]
        TEST.EQUALS(
            self, weight_cov, tmp,
            "Error 06")
        TEST.EQUALS(
            self, modval_mean, 0.237131,
            "Error 07")
        TEST.EQUALS(
            self, modval_var, 0.009589,
            "Error 08")
        
        TEST.LOG("TEST COMPLETE", 1)

