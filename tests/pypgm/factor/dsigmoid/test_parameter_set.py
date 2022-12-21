from common import TEST

import unittest


class test_parameter_set(unittest.TestCase):
    """Class containing tests for pypgm/factor/dsigmoid/parameter_set.py"""
        
    def test_01(self):
        """Tests the following :
        1) Functionality of multiply_parameter_sets
            input: scalar_normal
            output: scalar_normal
        """

        TEST.LOG("START - PYPGM_FACTOR_DSIGMOID_PARAMETER_SET_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import FACTOR_TYPE, scalar_normal, dsigmoid

        parameter_set = {}
        input_component = {
            'input': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 1.1,
                'prec': 1.2,
            }
        }
        output_component = {
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['z'],
                'wmean': 2.1,
                'prec': 2.2,
            }
        }
        
        parameter_set = dsigmoid.multiply_parameter_sets(parameter_set, input_component)
        parameter_set = dsigmoid.multiply_parameter_sets(parameter_set, output_component)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(SDict(**parameter_set), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'input': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'prec': 1.2,
                'wmean': 1.1,
            },
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['z'],
                'prec': 2.2,
                'wmean': 2.1,
            }

        }
        TEST.EQUALS(
            self, parameter_set, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

