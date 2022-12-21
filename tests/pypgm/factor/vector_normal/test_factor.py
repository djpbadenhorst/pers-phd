from common import TEST

import unittest


class test_factor(unittest.TestCase):
    """Class containing tests for pypgm/factor/factor.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of Factor.multiply_by for vector_normal
        """

        TEST.LOG("START - PYPGM_FACTOR_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal

        factor_1 = Factor(ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.FULL,
                          vars=['x','y'],
                          mean=[1, 2],
                          cov=[[1, -0.1], [-0.1, 2]])
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.STANDARD,
                          cov_form=vector_normal.COVARIANCE_FORM.FULL,
                          vars=['x', 'z'],
                          mean=[1, 2],
                          cov=[[1, 0.1], [0.1, 4]])
        factor_2.init_parameters()

        factor_1.multiply_by(factor_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PRODUCT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y', 'z'],
            'wmean': [ 2.057908, 1.055276, 0.476190],
            'prec': [[ 2.007531, 0.050251,-0.025062],
                     [ 0.050251, 0.502512, 0.0],
                     [-0.025062, 0.0,      0.250626]],
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of Factor.divide_by for gaussian
        """

        TEST.LOG("START - PYPGM_FACTOR_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal
        
        factor_1 = Factor(ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.CANONICAL,
                          cov_form=vector_normal.COVARIANCE_FORM.FULL,
                          vars=['x', 'y'],
                          wmean=[3, 4],
                          prec=[[3, 0], [0, 4]])
        factor_1.init_parameters()
        
        factor_2 = Factor(ftype=FACTOR_TYPE.VECTOR_NORMAL,
                          par_form=vector_normal.PARAMETER_FORM.CANONICAL,
                          cov_form=vector_normal.COVARIANCE_FORM.FULL,
                          vars=['y', 'x'],
                          wmean=[3, 2],
                          prec=[[2, 0], [0, 1]])
        factor_2.init_parameters()
        
        factor_1.divide_by(factor_2)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("QUOTIENT", 3)
        TEST.LOG(factor_1, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [1, 1],
            'prec': [[2, 0.], [0., 2]],
        }
        TEST.EQUALS(
            self, factor_1.parameters, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_03(self):
        """Tests the following :
        1) Functionality of Factor.marginalize_to for gaussian
        """

        TEST.LOG("START - PYPGM_FACTOR_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)

        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal

        factor = Factor(ftype=FACTOR_TYPE.VECTOR_NORMAL,
                        par_form=vector_normal.PARAMETER_FORM.CANONICAL,
                        cov_form=vector_normal.COVARIANCE_FORM.FULL,
                        vars=['x', 'y'],
                        wmean=[3.0, 4.0],
                        prec=[[9.0, 0.0], [0.0, 16.0]])
        factor.init_parameters()

        factor.marginalize_to(['x'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MARGINAL", 3)
        TEST.LOG(factor, 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x'],
            'wmean': [3],
            'prec': [[9]],
        }
        TEST.EQUALS(
            self, factor.parameters, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
