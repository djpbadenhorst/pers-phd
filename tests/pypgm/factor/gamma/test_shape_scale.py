from common import TEST

import unittest


class test_shape_scale(unittest.TestCase):
    """Class containing tests for pypgm/factor/gamma/shape_scale.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of multiply_shape_scale_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_GAMMA_SHAPE_SCALE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict
        
        from pypgm.factor import gamma

        shape_scale_1 = {
            'vars': ['p'],
            'shape': 5,
            'scale': 1./4,
        }
        shape_scale_2 = {
            'vars': ['p'],
            'shape': 1,
            'scale': 1./2,
        }

        shape_scale_product = gamma.multiply_shape_scale_forms(shape_scale_1, shape_scale_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(SDict(**shape_scale_product), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'vars': ['p'],
            'shape': 5,
            'scale': 1./6,
        }
        TEST.EQUALS(
            self, shape_scale_product, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of divide_shape_scale_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_GAMMA_SCALE_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)
        
        from utils import SDict
        
        from pypgm.factor import gamma

        shape_scale_1 = {
            'vars': ['p'],
            'shape': 5,
            'scale': 1./4,
        }
        shape_scale_2 = {
            'vars': ['p'],
            'shape': 1,
            'scale': 1./2,
        }

        shape_scale_quotient = gamma.divide_shape_scale_forms(shape_scale_1, shape_scale_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("QUOTIENT", 3)
        TEST.LOG(SDict(**shape_scale_quotient), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'vars': ['p'],
            'shape': 5,
            'scale': 1./2,
        }
        TEST.EQUALS(
            self, shape_scale_quotient, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
        
