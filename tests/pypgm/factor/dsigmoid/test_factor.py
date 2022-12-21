from common import TEST

import unittest


class test_factor(unittest.TestCase):
    """Class containing tests for pypgm/factor/factor.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of Factor.multiply_by for dsigmoid
        """

        TEST.LOG("START - PYPGM_FACTOR_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal

        factor_1 = Factor(ftype=FACTOR_TYPE.DSIGMOID)
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.DSIGMOID,
                          input={
                              'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                              'vars': ['y'],
                              'wmean': 3,
                              'prec': 10
                          })
        factor_2.init_parameters()
        
        factor_3 = Factor(ftype=FACTOR_TYPE.DSIGMOID,
                          output={
                              'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                              'vars': ['z'],
                              'wmean': 3,
                              'prec': 10
                          })
        factor_3.init_parameters()

        factor_1.multiply_by(factor_2)
        factor_1.multiply_by(factor_3)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DSIGMOID,
            'input': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'vars': ['y'],
                'wmean': 3,
                'prec': 10,
            },
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'vars': ['z'],
                'wmean': 3,
                'prec': 10,
            },
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
