from common import TEST

import unittest


class test_transform_samples(unittest.TestCase):
    """Class containing tests for pypgm/factor/dsigmoid/transform_samples.py"""
    
    def test_01(self):
        """Tests the following :
        1) Functionality of estimate_input_component
        2) Functionality of estimate_output_component
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DSIGMOID_TRANSFORM_SAMPLES_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        np.random.seed(0)
        
        from pypgm.factor import FACTOR_TYPE, scalar_normal, dsigmoid

        input_component = {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
                'vars': ['y'],
                'mean': 1,
                'var': 1,
        }
        scalar_normal.to_canonical(input_component, inplace=True)
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'vars': ['z'],
            'mean': 0.5,
            'var': 0.1,
        }
        scalar_normal.to_canonical(output_component, inplace=True)
        parameter_set = {
            'input': input_component,
            'output': output_component,
        }

        parameter_set_1 = dsigmoid.estimate_input_component(parameter_set)
        parameter_set_2 = dsigmoid.estimate_output_component(parameter_set)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT POSTERIOR", 3)
        TEST.LOG(parameter_set_1, 3)
        TEST.LOG("OUTPUT POSTERIOR", 3)
        TEST.LOG(parameter_set_2, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'input':{
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 0.997593,
                'prec': 1.043696,
            }
        }
        TEST.EQUALS(
            self, parameter_set_1, tmp,
            "Error 01")
        tmp = {
            'output':{
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['z'],
                'wmean': 123.711739,
                'prec': 200.385707,
            }
        }
        TEST.EQUALS(
            self, parameter_set_2, tmp,
            "Error 02")
        
        TEST.LOG("TEST COMPLETE", 1)

