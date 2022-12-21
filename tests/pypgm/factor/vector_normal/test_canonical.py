from common import TEST

import unittest


class test_canonical(unittest.TestCase):
    """Class containing tests for pypgm/factor/vector_normal/canonical.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of align_canonical_forms for canonical parameters with diagonal covariance matrix
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CANONICAL_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)
        
        from utils import SDict
        
        from pypgm.factor import vector_normal

        canonical_parameters_xy = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [10, 20],
        }
        canonical_parameters_yx = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [30, 40],
        }
        canonical_parameters_xz = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'z'],
            'wmean': [5, 6],
            'prec': [50, 60],
        }
        canonical_parameters_ab = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['a', 'b'],
            'wmean': [7, 8],
            'prec': [70, 80],
        }

        [canonical_parameters_xy_1, canonical_parameters_xy_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_yx)
        
        [canonical_parameters_xyz_1, canonical_parameters_xyz_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_xz)

        [canonical_parameters_abxy_1, canonical_parameters_abxy_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_ab)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL XY 1", 3)
        TEST.LOG(SDict(**canonical_parameters_xy_1), 3)
        TEST.LOG("CANONICAL XY 2", 3)
        TEST.LOG(SDict(**canonical_parameters_xy_2), 3)
        TEST.LOG("CANONICAL XYZ 1", 3)
        TEST.LOG(SDict(**canonical_parameters_xyz_1), 3)
        TEST.LOG("CANONICAL XYZ 2", 3)
        TEST.LOG(SDict(**canonical_parameters_xyz_2), 3)
        TEST.LOG("CANONICAL ABXY 1", 3)
        TEST.LOG(SDict(**canonical_parameters_abxy_1), 3)
        TEST.LOG("CANONICAL ABXY 2", 3)
        TEST.LOG(SDict(**canonical_parameters_abxy_2), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x', 'y'],
                'wmean': [1, 2],
                'prec': [10, 20],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x', 'y'],
                'wmean': [4, 3],
                'prec': [40, 30],
            }]            
        TEST.EQUALS(
            self, [canonical_parameters_xy_1, canonical_parameters_xy_2], tmp,
            "Error 01")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x', 'y', 'z'],
                'wmean': [1, 2, 0],
                'prec': [10, 20, 0],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x', 'y', 'z'],
                'wmean': [5, 0, 6],
                'prec': [50, 0, 60],
            }]
        TEST.EQUALS(
            self, [canonical_parameters_xyz_1, canonical_parameters_xyz_2], tmp,
            "Error 02")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['a', 'b', 'x', 'y'],
                'wmean': [0, 0, 1, 2],
                'prec': [0, 0, 10, 20],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['a', 'b', 'x', 'y'],
                'wmean': [7,8,0,0],
                'prec': [70 ,80 ,0 ,0],
            }]
        TEST.EQUALS(
            self, [canonical_parameters_abxy_1, canonical_parameters_abxy_2], tmp,
            "Error 03")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_02(self):
        """Tests the following :
        1) Functionality of align_canonical_forms for canonical parameters with full covariance matrix
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CANONICAL_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)
        
        from utils import SDict
        
        from pypgm.factor import vector_normal

        canonical_parameters_xy = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [[10, -5], [-5, 20]],
        }
        canonical_parameters_yx = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [[30, -10], [-10, 40]],
        }
        canonical_parameters_xz = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'z'],
            'wmean': [5, 6],
            'prec': [[50, -15], [-15, 60]],
        }
        canonical_parameters_ab = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['a', 'b'],
            'wmean': [7, 8],
            'prec': [[70, -20], [-20, 80]],
        }

        [canonical_parameters_xy_1, canonical_parameters_xy_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_yx)
        
        [canonical_parameters_xyz_1, canonical_parameters_xyz_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_xz)

        [canonical_parameters_abxy_1, canonical_parameters_abxy_2] = vector_normal.align_canonical_forms(
            canonical_parameters_xy, canonical_parameters_ab)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("CANONICAL XY 1", 3)
        TEST.LOG(SDict(**canonical_parameters_xy_1), 3)
        TEST.LOG("CANONICAL XY 2", 3)
        TEST.LOG(SDict(**canonical_parameters_xy_2), 3)
        TEST.LOG("CANONICAL XYZ 1", 3)
        TEST.LOG(SDict(**canonical_parameters_xyz_1), 3)
        TEST.LOG("CANONICAL XYZ 2", 3)
        TEST.LOG(SDict(**canonical_parameters_xyz_2), 3)
        TEST.LOG("CANONICAL ABXY 1", 3)
        TEST.LOG(SDict(**canonical_parameters_abxy_1), 3)
        TEST.LOG("CANONICAL ABXY 2", 3)
        TEST.LOG(SDict(**canonical_parameters_abxy_2), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['x', 'y'],
                'wmean': [1, 2],
                'prec': [[10, -5], [-5, 20]],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['x', 'y'],
                'wmean': [4, 3],
                'prec': [[40, -10], [-10, 30]],
            }]            
        TEST.EQUALS(
            self, [canonical_parameters_xy_1, canonical_parameters_xy_2], tmp,
            "Error 01")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['x', 'y', 'z'],
                'wmean': [1, 2, 0],
                'prec': [[10, -5, 0], [-5, 20, 0], [0, 0, 0]],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['x', 'y', 'z'],
                'wmean': [5, 0, 6],
                'prec': [[50, 0, -15], [0, 0, 0], [-15, 0, 60]],
            }]
        TEST.EQUALS(
            self, [canonical_parameters_xyz_1, canonical_parameters_xyz_2], tmp,
            "Error 02")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['a', 'b', 'x', 'y'],
                'wmean': [0, 0, 1, 2],
                'prec': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, -5], [0, 0, -5, 20]],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['a', 'b', 'x', 'y'],
                'wmean': [7,8,0,0],
                'prec': [[70, -20, 0, 0], [-20, 80, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            }]
        TEST.EQUALS(
            self, [canonical_parameters_abxy_1, canonical_parameters_abxy_2], tmp,
            "Error 03")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_03(self):
        """Tests the following :
        1) Functionality of multiply_canonical_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CANONICAL_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import vector_normal

        canonical_parameters_xy_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': 10,
        }
        canonical_parameters_yx_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': 10,
        }
        canonical_parameters_xy_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [10,20],
        }
        canonical_parameters_yx_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [30,40],
        }
        canonical_parameters_xy_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [[10, -5], [-5, 20]],
        }
        canonical_parameters_yx_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [[30, -10], [-10, 40]],
        }

        canonical_product_common = vector_normal.multiply_canonical_forms(
            canonical_parameters_xy_common, canonical_parameters_yx_common)
        canonical_product_diagonal = vector_normal.multiply_canonical_forms(
            canonical_parameters_xy_diagonal, canonical_parameters_yx_diagonal)
        canonical_product_full = vector_normal.multiply_canonical_forms(
            canonical_parameters_xy_full, canonical_parameters_yx_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT - COMMON", 3)
        TEST.LOG(SDict(**canonical_product_common), 3)
        TEST.LOG("PRODUCT - DIAGONAL", 3)
        TEST.LOG(SDict(**canonical_product_diagonal), 3)
        TEST.LOG("PRODUCT - FULL", 3)
        TEST.LOG(SDict(**canonical_product_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'y'],
            'wmean': [5,5],
            'prec': 20,
        }
        TEST.EQUALS(
            self, canonical_product_common, tmp,
            "Error 01")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y'],
            'wmean': [5,5],
            'prec': [50,50],
        }
        TEST.EQUALS(
            self, canonical_product_diagonal, tmp,
            "Error 02")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [5,5],
            'prec': [[50,-15],[-15,50]],
        }
        TEST.EQUALS(
            self, canonical_product_full, tmp,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_04(self):
        """Tests the following :
        1) Functionality of divide_canonical_forms
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CANONICAL_04", 1)
        TEST.LOG(self.test_04.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import vector_normal

        canonical_parameters_xy_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': 10,
        }
        canonical_parameters_yx_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': 10,
        }
        canonical_parameters_xy_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [10,20],
        }
        canonical_parameters_yx_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [30,40],
        }
        canonical_parameters_xy_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [1, 2],
            'prec': [[10, -5], [-5, 20]],
        }
        canonical_parameters_yx_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['y', 'x'],
            'wmean': [3, 4],
            'prec': [[30, -10], [-10, 40]],
        }

        canonical_quotient_common = vector_normal.divide_canonical_forms(
            canonical_parameters_xy_common, canonical_parameters_yx_common)
        canonical_quotient_diagonal = vector_normal.divide_canonical_forms(
            canonical_parameters_xy_diagonal, canonical_parameters_yx_diagonal)
        canonical_quotient_full = vector_normal.divide_canonical_forms(
            canonical_parameters_xy_full, canonical_parameters_yx_full)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("QUOTIENT - COMMON", 3)
        TEST.LOG(SDict(**canonical_quotient_common), 3)
        TEST.LOG("QUOTIENT - DIAGONAL", 3)
        TEST.LOG(SDict(**canonical_quotient_diagonal), 3)
        TEST.LOG("QUOTIENT - FULL", 3)
        TEST.LOG(SDict(**canonical_quotient_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'y'],
            'wmean': [-3,-1],
            'prec': 0,
        }
        TEST.EQUALS(
            self, canonical_quotient_common, tmp,
            "Error 01")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y'],
            'wmean': [-3,-1],
            'prec': [-30, -10],
        }
        TEST.EQUALS(
            self, canonical_quotient_diagonal, tmp,
            "Error 02")
        tmp = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [-3,-1],
            'prec': [[-30,5],[5,-10]],
        }
        TEST.EQUALS(
            self, canonical_quotient_full, tmp,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_05(self):
        """Tests the following :
        1) Functionality of marginalize_canonical_form
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_CANONICAL_05", 1)
        TEST.LOG(self.test_05.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import vector_normal

        canonical_parameters_xyz_common = {
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x', 'y', 'z'],
            'wmean': [1, 2, 3],
            'prec': 10,
        }
        canonical_parameters_xyz_diagonal = {
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x', 'y', 'z'],
            'wmean': [1, 2, 3],
            'prec': [10,20,30],
        }
        canonical_parameters_xyz_full = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y', 'z'],
            'wmean': [1, 2, 3],
            'prec': [[10,1,0], [1,20,-1],[0,-1,30]],
        }

        canonical_marginal_x_common = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_common, ['x'])
        canonical_marginal_zy_common = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_common, ['z', 'y'])

        canonical_marginal_x_diagonal = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_diagonal, ['x'])
        canonical_marginal_zy_diagonal = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_diagonal, ['z', 'y'])

        canonical_marginal_x_full = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_full, ['x'])
        canonical_marginal_zy_full = vector_normal.marginalize_canonical_form(
            canonical_parameters_xyz_full, ['z', 'y'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MARGINAL X - COMMON", 3)
        TEST.LOG(SDict(**canonical_marginal_x_common), 3)
        TEST.LOG("MARGINAL ZY - COMMON", 3)
        TEST.LOG(SDict(**canonical_marginal_zy_common), 3)
        TEST.LOG("MARGINAL X - DIAGONAL", 3)
        TEST.LOG(SDict(**canonical_marginal_x_diagonal), 3)
        TEST.LOG("MARGINAL ZY - DIAGONAL", 3)
        TEST.LOG(SDict(**canonical_marginal_zy_diagonal), 3)
        TEST.LOG("MARGINAL X - FULL", 3)
        TEST.LOG(SDict(**canonical_marginal_x_full), 3)
        TEST.LOG("MARGINAL ZY - FULL", 3)
        TEST.LOG(SDict(**canonical_marginal_zy_full), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
                'vars': ['x'],
                'wmean': [1],
                'prec': 10,
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
                'vars': ['z','y'],
                'wmean': [3,2],
                'prec': 10,
            }
        ]
        TEST.EQUALS(
            self, [canonical_marginal_x_common, canonical_marginal_zy_common], tmp,
            "Error 01")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x'],
                'wmean': [1],
                'prec': [10],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['z','y'],
                'wmean': [3,2],
                'prec': [30,20],
            }
        ]
        TEST.EQUALS(
            self, [canonical_marginal_x_diagonal, canonical_marginal_zy_diagonal], tmp,
            "Error 02")
        tmp = [
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['x'],
                'wmean': [0.894824],
                'prec': [[9.949916]],
            },
            {
                'cov_form': vector_normal.COVARIANCE_FORM.FULL,
                'vars': ['z','y'],
                'wmean': [3, 1.9],
                'prec': [[30, -1],[-1, 19.9]],
            }
        ]
        TEST.EQUALS(
            self, [canonical_marginal_x_full, canonical_marginal_zy_full], tmp,
            "Error 03")

        TEST.LOG("TEST COMPLETE", 1)

