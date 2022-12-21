from common import TEST

import unittest


class test_kl_iow(unittest.TestCase):
    """Class containing tests for pypgm/factor/dlinear/kl_iow.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of kl_optimization_iow
            input: vector_normal (common covariance)
            output: scalar_normal
            weight: vector_normal (common covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOW_01", 1)
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

        estimated_parameter_set = dlinear.kl_optimization_iow(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['input']), 3)
        TEST.LOG("OUTPUT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['output']), 3)
        TEST.LOG("WEIGHT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['weight']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'wmean': [-10.850950, 12.616342],
            'prec': 418.061888,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['input'], tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3.093586,
            'prec': 10.199996,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['output'], tmp,
            "Error 02")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'wmean': [4.007166, 5.006892],
            'prec': 0.100636,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['weight'], tmp,
            "Error 03")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_02(self):
        """Tests the following :
        1) Functionality of kl_optimization_iow
            input: vector_normal (diagonal covariance)
            output: scalar_normal
            weight: vector_normal (diagonal covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOW_02", 1)
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

        estimated_parameter_set = dlinear.kl_optimization_iow(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT PARAMETERS", 3)
        TEST.LOG(SDict(**parameter_set['input']), 3)
        TEST.LOG("OUTPUT PARAMETERS", 3)
        TEST.LOG(SDict(**parameter_set['output']), 3)
        TEST.LOG("WEIGHT PARAMETERS", 3)
        TEST.LOG(SDict(**parameter_set['weight']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'wmean': [-8.481693, 15.253172],
            'prec': [328.255561, 507.877441],
        }
        TEST.EQUALS(
            self, estimated_parameter_set['input'], tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3.093592,
            'prec': 10.199998,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['output'], tmp,
            "Error 02")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'wmean': [4.006157, 5.007985],
            'prec': [0.100742, 0.100574],
        }
        TEST.EQUALS(
            self, estimated_parameter_set['weight'], tmp,
            "Error 03")
        
        TEST.LOG("TEST COMPLETE", 1)


    def test_03(self):
        """Tests the following :
        1) Functionality of kl_optimization_iow
            input: vector_normal (full covariance)
            output: scalar_normal
            weight: vector_normal (full covariance)
            modval: gamma
        """
        
        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_KL_IOW_03", 1)
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

        estimated_parameter_set = dlinear.kl_optimization_iow(parameter_set)

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['input']), 3)
        TEST.LOG("OUTPUT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['output']), 3)
        TEST.LOG("WEIGHT PARAMETERS", 3)
        TEST.LOG(SDict(**estimated_parameter_set['weight']), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [3.141768, 4.813097],
            'prec': [[260.635942, 326.805722],
                     [326.805722, 441.148713]],
        }
        TEST.EQUALS(
            self, estimated_parameter_set['input'], tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'wmean': 3.096740,
            'prec': 10.200001,
        }
        TEST.EQUALS(
            self, estimated_parameter_set['output'], tmp,
            "Error 02")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'wmean': [3.998612, 5.001694],
            'prec': [[0.110894, 0.001879],
                     [0.001879, 0.106529]],
        }
        TEST.EQUALS(
            self, estimated_parameter_set['weight'], tmp,
            "Error 03")
        
        TEST.LOG("TEST COMPLETE", 1)

