from common import TEST

import unittest


class test_convert(unittest.TestCase):
    """Class containing tests for pypgm/factor/vector_normal/convert.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of canonical_to_standard
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CONVERT_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils import SDict

        from pypgm.factor import vector_normal

        canonical_parameters_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'prec': 10,
            'wmean': [1, 2],
        }
        canonical_parameters_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'prec': [10, 20],
            'wmean': [1, 2],
        }
        canonical_parameters_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'prec': [[10, 0.5], [0.5, 20]],
            'wmean': [1, 2],
        }

        standard_parameters_common = vector_normal.canonical_to_standard(canonical_parameters_common)
        standard_parameters_diagonal = vector_normal.canonical_to_standard(canonical_parameters_diagonal)
        standard_parameters_full = vector_normal.canonical_to_standard(canonical_parameters_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("STANDARD - COMMON COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_common), 3)
        TEST.LOG("STANDARD - DIAGONAL COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_diagonal), 3)
        TEST.LOG("STANDARD - FULL COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'cov': 0.1,
            'mean': [0.1, 0.2],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_common,
            "Error 01")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'cov': [0.1, 0.05],
            'mean': [0.1, 0.1],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_diagonal,
            "Error 02")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'cov': [[0.100125, -0.002503], [-0.002503, 0.0500625]],
            'mean': [0.095118, 0.097622],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_full,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of standard_to_canonical
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CONVERT_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import vector_normal

        standard_parameters_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'cov': 10,
            'mean': [1, 2],
        }
        standard_parameters_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'cov': [10, 20],
            'mean': [1, 2],
        }
        standard_parameters_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'cov': [[10, 0.5], [0.5, 20]],
            'mean': [1, 2],
        }

        canonical_parameters_common = vector_normal.standard_to_canonical(standard_parameters_common)
        canonical_parameters_diagonal = vector_normal.standard_to_canonical(standard_parameters_diagonal)
        canonical_parameters_full = vector_normal.standard_to_canonical(standard_parameters_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL - COMMON COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_common), 3)
        TEST.LOG("CANONICAL - DIAGONAL COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_diagonal), 3)
        TEST.LOG("CANONICAL - FULL COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'prec': 0.1,
            'wmean': [0.1, 0.2],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_common,
            "Error 01")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'prec': [0.1, 0.05],
            'wmean': [0.1, 0.1],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_diagonal,
            "Error 02")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'prec': [[0.100125, -0.002503], [-0.002503, 0.0500625]],
            'wmean': [0.095118, 0.097622],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_full,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)


    def test_03(self):
        """Tests the following :
        1) Functionality of to_standard
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CONVERT_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)

        from utils import SDict

        from pypgm.factor import vector_normal

        canonical_parameters_common = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'prec': 10,
            'wmean': [1, 2],
        }
        canonical_parameters_diagonal = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'prec': [10, 20],
            'wmean': [1, 2],
        }
        canonical_parameters_full = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'prec': [[10, 0.5], [0.5, 20]],
            'wmean': [1, 2],
        }
        
        standard_parameters_common = vector_normal.to_standard(canonical_parameters_common)
        standard_parameters_diagonal = vector_normal.to_standard(canonical_parameters_diagonal)
        standard_parameters_full = vector_normal.to_standard(canonical_parameters_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("STANDARD - COMMON COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_common), 3)
        TEST.LOG("STANDARD - DIAGONAL COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_diagonal), 3)
        TEST.LOG("STANDARD - FULL COVARIANCE", 3)
        TEST.LOG(SDict(**standard_parameters_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'cov': 0.1,
            'mean': [0.1, 0.2],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_common,
            "Error 01")
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'cov': [0.1, 0.05],
            'mean': [0.1, 0.1],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_diagonal,
            "Error 02")
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'cov': [[0.100125, -0.002503], [-0.002503, 0.0500625]],
            'mean': [0.095118, 0.097622],
        }
        TEST.EQUALS(
            self, tmp, standard_parameters_full,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)


    def test_04(self):
        """Tests the following :
        1) Functionality of to_canonical
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CONVERT_04", 1)
        TEST.LOG(self.test_04.__doc__, 2)

        from utils import SDict

        from pypgm.factor import vector_normal

        standard_parameters_common = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'cov': 10,
            'mean': [1, 2],
        }
        standard_parameters_diagonal = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'cov': [10, 20],
            'mean': [1, 2],
        }
        standard_parameters_full = {
            'par_form': vector_normal.PARAMETER_FORM.STANDARD,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'cov': [[10, 0.5], [0.5, 20]],
            'mean': [1, 2],
        }

        canonical_parameters_common = vector_normal.to_canonical(standard_parameters_common)
        canonical_parameters_diagonal = vector_normal.to_canonical(standard_parameters_diagonal)
        canonical_parameters_full = vector_normal.to_canonical(standard_parameters_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL - COMMON COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_common), 3)
        TEST.LOG("CANONICAL - DIAGONAL COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_diagonal), 3)
        TEST.LOG("CANONICAL - FULL COVARIANCE", 3)
        TEST.LOG(SDict(**canonical_parameters_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'prec': 0.1,
            'wmean': [0.1, 0.2],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_common,
            "Error 01")
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'prec': [0.1, 0.05],
            'wmean': [0.1, 0.1],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_diagonal,
            "Error 02")
        tmp = {
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'prec': [[0.100125, -0.002503], [-0.002503, 0.0500625]],
            'wmean': [0.095118, 0.097622],
        }
        TEST.EQUALS(
            self, tmp, canonical_parameters_full,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)
