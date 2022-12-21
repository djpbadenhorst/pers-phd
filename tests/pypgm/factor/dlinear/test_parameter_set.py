from common import TEST

import unittest


class test_parameter_set(unittest.TestCase):
    """Class containing tests for pypgm/factor/dlinear/parameter_set.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of align_component_with_variables for vector_normal component
        """

        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_PARAMETER_SET_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal

        vector_normal_component_common = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x1', 'x2', 'x3'],
            'wmean': [1, 2, 3],
            'prec': 10,
        }
        vector_normal_component_diagonal = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x1', 'x2', 'x3'],
            'wmean': [1, 2, 3],
            'prec': [10, 20, 30],
        }
        vector_normal_component_full = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x1', 'x2', 'x3'],
            'wmean': [1, 2, 3],
            'prec': [[10, 1, 2], [1, 20, 3], [2, 3, 30]],
        }

        vector_normal_component_common = dlinear.align_component_with_variables(
            vector_normal_component_common, ['x3','x2','x1'])
        vector_normal_component_diagonal = dlinear.align_component_with_variables(
            vector_normal_component_diagonal, ['x3','x2','x1'])
        vector_normal_component_full = dlinear.align_component_with_variables(
            vector_normal_component_full, ['x3','x2','x1'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("VECTOR NORMAL - COMMON", 3)
        TEST.LOG(SDict(**vector_normal_component_common), 3)
        TEST.LOG("VECTOR NORMAL - DIAGONAL", 3)
        TEST.LOG(SDict(**vector_normal_component_diagonal), 3)
        TEST.LOG("VECTOR NORMAL - FULL", 3)
        TEST.LOG(SDict(**vector_normal_component_full), 3)

        TEST.LOG("START ASSERTS", 3)

        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x3', 'x2', 'x1'],
            'wmean': [3, 2, 1],
            'prec': 10,
        }
        TEST.EQUALS(
            self, vector_normal_component_common, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x3', 'x2', 'x1'],
            'wmean': [3, 2, 1],
            'prec': [30, 20, 10],
        }
        TEST.EQUALS(
            self, vector_normal_component_diagonal, tmp,
            "Error 02")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x3', 'x2', 'x1'],
            'wmean': [3, 2, 1],
            'prec': [[30, 3, 2], [3, 20, 1], [2, 1, 10]],
        }
        TEST.EQUALS(
            self, vector_normal_component_full, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of multiply_parameter_sets
            input: vector_normal
            output: scalar_normal
            weight: vector_normal
            modval: gamma
        """

        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_PARAMETER_SET_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import FACTOR_TYPE, vector_normal, scalar_normal, dlinear

        parameter_set = {
            'input_vars': ['x3', 'x2', 'x1'],
            'output_vars': ['y'],
            'weight_vars': ['w2', 'w1', 'w3'],
            'modval_vars': ['p'],
        }
        input_component = {
            'input': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x1', 'x2', 'x3'],
                'wmean': [0.1, 0.2, 0.3],
                'prec': [1.1, 1.2, 1.3],
            }
        }
        output_component = {
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 2.1,
                'prec': 2.2,
            }
        }
        weight_component = {
            'weight': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['w1', 'w2', 'w3'],
                'wmean': [3.1, 3.2, 3.3],
                'prec': [4.1, 4.2, 4.3],
            }
        }
        modval_component = {
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 5.1,
                'scale': 6.1,
            }
        }
        
        parameter_set = dlinear.multiply_parameter_sets(parameter_set, input_component)
        parameter_set = dlinear.multiply_parameter_sets(parameter_set, output_component)
        parameter_set = dlinear.multiply_parameter_sets(parameter_set, weight_component)
        parameter_set = dlinear.multiply_parameter_sets(parameter_set, modval_component)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(SDict(**parameter_set), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'input_vars': ['x3', 'x2', 'x1'],
            'input': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x3', 'x2', 'x1'],
                'wmean': [0.3, 0.2, 0.1], 
                'prec': [1.3, 1.2, 1.1],
            },
            'output_vars': ['y'],
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 2.1,
                'prec': 2.2, 
            },
            'weight_vars': ['w2', 'w1', 'w3'],
            'weight': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['w2', 'w1', 'w3'],
                'wmean': [3.2, 3.1, 3.3], 
                'prec': [4.2, 4.1, 4.3],
            },
            'modval_vars': ['p'],
            'modval':
            {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 5.1,
                'scale': 6.1
            },
        }
        TEST.EQUALS(
            self, parameter_set, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

