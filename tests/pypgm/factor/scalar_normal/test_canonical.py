from common import TEST

import unittest


class test_canonical(unittest.TestCase):
    """Class containing tests for pypgm/factor/scalar_normal/canonical.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of multiply_canonical_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CANONICAL_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import scalar_normal

        canonical_parameters_1 = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 2,
            'prec': 20,
        }
        canonical_parameters_2 = {
            'form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 1,
            'prec': 10,
        }

        canonical_product = scalar_normal.multiply_canonical_forms(
            canonical_parameters_1, canonical_parameters_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(SDict(**canonical_product), 3)
        
        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 3,
            'prec': 30,
        }
        TEST.EQUALS(
            self, canonical_product, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)


    def test_02(self):
        """Tests the following :
        1) Functionality of divide_canonical_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CANONICAL_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import scalar_normal

        canonical_parameters_1 = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 3,
            'prec': 30,
        }
        canonical_parameters_2 = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 1,
            'prec': 10,
        }

        canonical_quotient = scalar_normal.divide_canonical_forms(
            canonical_parameters_1, canonical_parameters_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("QUOTIENT", 3)
        TEST.LOG(SDict(**canonical_quotient), 3)
        
        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 2,
            'prec': 20,
        }
        TEST.EQUALS(
            self, canonical_quotient, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

        
