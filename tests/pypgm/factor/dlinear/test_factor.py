from common import TEST

import unittest


class test_factor(unittest.TestCase):
    """Class containing tests for pypgm/factor/factor.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of Factor.multiply_by for dlinear
        """

        TEST.LOG("START - PYPGM_FACTOR_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal

        factor_1 = Factor(ftype=FACTOR_TYPE.DLINEAR,
                          input_vars=['x1', 'x2'],
                          output_vars=['y'],
                          modval_vars=['p'])
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.DLINEAR,
                          input={
                              'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                              'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                              'vars': ['x1', 'x2'],
                              'wmean': [1, 2],
                              'prec': [10,20]
                          })
        factor_2.init_parameters()
        
        factor_3 = Factor(ftype=FACTOR_TYPE.DLINEAR,
                          output={
                              'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                              'vars': ['y'],
                              'wmean': 3,
                              'prec': 10
                          })
        factor_3.init_parameters()

        factor_4 = Factor(ftype=FACTOR_TYPE.DLINEAR,
                          modval={
                              'ftype': FACTOR_TYPE.GAMMA,
                              'vars': ['p'],
                              'shape': 4,
                              'scale': 5
                          })
        factor_4.init_parameters()

        factor_1.multiply_by(factor_2)
        factor_1.multiply_by(factor_3)
        factor_1.multiply_by(factor_4)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'input_vars': ['x1','x2'],
            'input': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x1', 'x2'],
                'prec': [10, 20],
                'wmean': [1, 2],
            },
            'output_vars': ['y'],
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'vars': ['y'],
                'prec': 10,
                'wmean': 3,
            },
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 4,
                'scale': 5,
            },
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
