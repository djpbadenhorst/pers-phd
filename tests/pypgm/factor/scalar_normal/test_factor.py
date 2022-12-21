from common import TEST

import unittest


class test_factor(unittest.TestCase):
    """Class containing tests for pypgm/factor/factor.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of Factor.multiply_by for scalar_normal
        """

        TEST.LOG("START - PYPGM_FACTOR_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal

        factor_1 = Factor(ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['x'],
                          mean=1,
                          var=1)
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['x'],
                          mean=2,
                          var=2)
        factor_2.init_parameters()

        factor_1.multiply_by(factor_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 2,
            'prec': 1.5,
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of Factor.divide_by for scalar_normal
        """

        TEST.LOG("START - PYPGM_FACTOR_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal

        factor_1 = Factor(ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['x'],
                          mean=1,
                          var=1)
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.SCALAR_NORMAL,
                          par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                          vars=['x'],
                          mean=2,
                          var=2)
        factor_2.init_parameters()

        factor_1.divide_by(factor_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("QUOTIENT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 0,
            'prec': 0.5,
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)
