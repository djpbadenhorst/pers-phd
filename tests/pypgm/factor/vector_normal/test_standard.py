from common import TEST

import unittest


class test_standard(unittest.TestCase):
    """Class containing tests for pypgm/factor/vector_normal/standard.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of marginalize_standard_form
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_STANDARD_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict

        from pypgm.factor import vector_normal

        standard_parameters_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x','y','z'],
            'mean': [1., 2., 3.],
            'cov': 10,
        }
        standard_parameters_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x','y','z'],
            'mean': [1., 2., 3.],
            'cov': [10., 20., 30.],
        }
        standard_parameters_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x','y','z'],
            'mean': [1., 2., 3.],
            'cov': [[10., 0.5, -0.5], [0.5, 20., 0.5], [-0.5, 0.5, 30.]],
        }
        
        standard_parameters_common_xz = vector_normal.marginalize_standard_form(
            standard_parameters_common, ['x', 'z'])
        standard_parameters_diagonal_xz = vector_normal.marginalize_standard_form(
            standard_parameters_diagonal, ['x', 'z'])
        standard_parameters_full_xz = vector_normal.marginalize_standard_form(
            standard_parameters_full, ['x', 'z'])

        standard_parameters_common_y = vector_normal.marginalize_standard_form(
            standard_parameters_common, ['y'])
        standard_parameters_diagonal_y = vector_normal.marginalize_standard_form(
            standard_parameters_diagonal, ['y'])
        standard_parameters_full_y = vector_normal.marginalize_standard_form(
            standard_parameters_full, ['y'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("STANDARD XZ - COMMON", 3)
        TEST.LOG(SDict(**standard_parameters_common_xz), 3)
        TEST.LOG("STANDARD XZ - DIAGONAL", 3)
        TEST.LOG(SDict(**standard_parameters_diagonal_xz), 3)
        TEST.LOG("STANDARD XZ - FULL", 3)
        TEST.LOG(SDict(**standard_parameters_full_xz), 3)
        TEST.LOG("STANDARD Y - COMMON", 3)
        TEST.LOG(SDict(**standard_parameters_common_y), 3)
        TEST.LOG("STANDARD Y - DIAGONAL", 3)
        TEST.LOG(SDict(**standard_parameters_diagonal_y), 3)
        TEST.LOG("STANDARD Y - FULL", 3)
        TEST.LOG(SDict(**standard_parameters_full_y), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'z'],
            'mean': [1, 3],
            'cov': 10,
        }
        TEST.EQUALS(
            self, standard_parameters_common_xz, tmp,
            "Error 01")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'z'],
            'mean': [1, 3],
            'cov': [10, 30],
        }
        TEST.EQUALS(
            self, standard_parameters_diagonal_xz, tmp,
            "Error 02")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'z'],
            'mean': [1, 3],
            'cov': [[10,-0.5],[-0.5,30]],
        }
        TEST.EQUALS(
            self, standard_parameters_full_xz, tmp,
            "Error 03")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['y'],
            'mean': [2],
            'cov': 10,
        }
        TEST.EQUALS(
            self, standard_parameters_common_y, tmp,
            "Error 04")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['y'],
            'mean': [2],
            'cov': [20],
        }
        TEST.EQUALS(
            self, standard_parameters_diagonal_y, tmp,
            "Error 05")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['y'],
            'mean': [2],
            'cov': [[20]],
        }
        TEST.EQUALS(
            self, standard_parameters_full_y, tmp,
            "Error 06")
        
        TEST.LOG("TEST COMPLETE", 1)
