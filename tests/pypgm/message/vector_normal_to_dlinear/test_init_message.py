from common import TEST

import unittest


class test_init_message(unittest.TestCase):
    """Class containing tests for pypgm/message/vector_normal_to_dlinear/init_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of init_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_VECTOR_NORMAL_TO_DLINEAR_INIT_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal
        
        from pypgm.message import vector_normal_to_dlinear

        vector_normal_from_input = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['x_1','x_2'],
            mean=[1,2],
            cov=[2,3])
        vector_normal_from_input.init_parameters()

        vector_normal_from_weight = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['w_1','w_2'],
            mean=[3,4],
            cov=[4,5])
        vector_normal_from_weight.init_parameters()
        
        dlinear_to = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars = ['x_1','x_2'],
            output_vars = ['y'],
            weight_vars = ['w_1','w_2'],
            modval_vars = ['p'])
        dlinear_to.init_parameters()

        message_parameters_input = vector_normal_to_dlinear.init_message(
            vector_normal_from_input.parameters, dlinear_to.parameters, ['x_1','x_2'])
        message_parameters_weight = vector_normal_to_dlinear.init_message(
            vector_normal_from_weight.parameters, dlinear_to.parameters, ['w_1','w_2'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE INPUT", 3)
        TEST.LOG(SDict(**message_parameters_input), 3)
        TEST.LOG("INIT MESSAGE WEIGHT", 3)
        TEST.LOG(SDict(**message_parameters_weight), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'input_vars': ['x_1','x_2'],
            'input':{
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x_1','x_2'],
                'wmean': [0.5, 2./3],
                'prec': [0.5, 1./3],
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters_input,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'weight_vars': ['w_1','w_2'],
            'weight':{
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['w_1','w_2'],
                'wmean': [0.75, 0.8],
                'prec': [0.25, 0.2],
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters_weight,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
