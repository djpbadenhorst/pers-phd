from common import TEST

import unittest


class test_kl_iowm(unittest.TestCase):
    """Class containing tests for pypgm/factor/dlinear/kl_iowm.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of kl_optimization_iowm
            input: vector_normal (common covariance)
            output: scalar_normal
            weight: vector_normal (common covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOWM_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal, scalar_normal
    
        input_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'wmean': [1, 2],
            'prec': 10,
        }
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3,
            'prec': 10,
        }
        weight_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'wmean': [4, 5],
            'prec': 0.1,
        }
        modval_component = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6,
            'scale': 1./25,
        }
        parameter_set = {
            'input': input_component,
            'output': output_component,
            'weight': weight_component,
            'modval': modval_component,
        }

        estimated_parameter_set = dlinear.kl_optimization_iowm(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MODVAL PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['modval']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6.5,
            'scale': 0.003075,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['modval'], tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_02(self):
        """Tests the following :
        1) Functionality of kl_optimization_iowm
            input: vector_normal (diagonal covariance)
            output: scalar_normal
            weight: vector_normal (diagonal covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOWM_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal, scalar_normal
    
        input_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'wmean': [1, 2],
            'prec': [10, 10],
        }
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3,
            'prec': 10,
        }
        weight_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'wmean': [4, 5],
            'prec': [0.1, 0.1],
        }
        modval_component = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6,
            'scale': 1./25,
        }
    
        parameter_set = {
            'input': input_component,
            'output': output_component,
            'weight': weight_component,
            'modval': modval_component,
        }

        estimated_parameter_set = dlinear.kl_optimization_iowm(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MODVAL PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['modval']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6.5,
            'scale': 0.003075,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['modval'], tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_03(self):
        """Tests the following :
        1) Functionality of kl_optimization_iowm
            input: vector_normal (full covariance)
            output: scalar_normal
            weight: vector_normal (full covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOWM_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal, scalar_normal
    
        input_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [1, 2],
            'prec': [[10, 0.01], [0.01, 10]],
        }
        output_component = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3,
            'prec': 10,
        }
        weight_component = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [4, 5],
            'prec': [[0.1, 0.01], [0.01, 0.1]],
        }
        modval_component = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6,
            'scale': 1./25,
        }
        parameter_set = {
            'input': input_component,
            'output': output_component,
            'weight': weight_component,
            'modval': modval_component,
        }

        estimated_parameter_set = dlinear.kl_optimization_iowm(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MODVAL PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['modval']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.GAMMA,
            'shape': 6.5,
            'scale': 0.003637,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['modval'], tmp,
            "Error 01")
        
        TEST.LOG("TEST COMPLETE", 1)

