from common import TEST

import unittest


class test_convert(unittest.TestCase):
    """Class containing tests for pypgm/factor/scalar_normal/convert.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of canonical_to_standard
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CONVERT_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import scalar_normal

        canonical_parameters = {
            'wmean': 1,
            'prec': 10,
        }

        standard_parameters = scalar_normal.canonical_to_standard(canonical_parameters)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("STANDARD", 3)
        TEST.LOG(SDict(**standard_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'mean': 0.1,
            'var': 0.1,
        }
        TEST.EQUALS(
            self, tmp, standard_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of standard_to_canonical
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CONVERT_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import scalar_normal

        standard_parameters = {
            'mean': 1,
            'var': 10,
        }

        canonical_parameters = scalar_normal.standard_to_canonical(standard_parameters)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL", 3)
        TEST.LOG(SDict(**canonical_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'wmean': 0.1,
            'prec': 0.1,
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)


    def test_03(self):
        """Tests the following :
        1) Functionality of to_standard
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CONVERT_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import scalar_normal

        canonical_parameters = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 1,
            'prec': 10,
        }

        standard_parameters = scalar_normal.to_standard(canonical_parameters)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("STANDARD", 3)
        TEST.LOG(SDict(**standard_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'mean': 0.1,
            'var': 0.1,
        }
        TEST.EQUALS(
            self, tmp, standard_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)


    def test_04(self):
        """Tests the following :
        1) Functionality of to_canonical
        """

        TEST.LOG("START - PYPGM_FACTOR_SCALAR_NORMAL_CONVERT_04", 1)
        TEST.LOG(self.test_04.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import scalar_normal

        standard_parameters = {
            'par_form': scalar_normal.PARAMETER_FORM.STANDARD,
            'mean': 1,
            'var': 10,
        }

        canonical_parameters = scalar_normal.to_canonical(standard_parameters)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL", 3)
        TEST.LOG(SDict(**canonical_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 0.1,
            'prec': 0.1,
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

