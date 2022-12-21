from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/vector_normal_to_dlinear/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_VECTOR_NORMAL_TO_DLINEAR_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal
        
        from pypgm.message import vector_normal_to_dlinear

        vector_normal_from = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['w_1','w_2'],
            mean=[1,2],
            cov=[2,3])
        vector_normal_from.init_parameters()
        
        dlinear_to = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars=['x_1','x_2'],
            output_vars=['y'],
            weight_vars=['w_1','w_2'],
            modval_vars=['p'])
        dlinear_to.init_parameters()

        vector_normal_inc = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['w_1', 'w_2'],
            mean=[0.1,0.2],
            cov=[20,30])
        vector_normal_inc.init_parameters()
        
        dlinear_out = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            weight_vars=['w_1', 'w_2'])
        dlinear_out.init_parameters()

        message_parameters = vector_normal_to_dlinear.est_message(
            vector_normal_from.parameters, dlinear_to.parameters,
            vector_normal_inc.parameters, dlinear_out.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'weight_vars': ['w_1', 'w_2'],
            'weight': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['w_1', 'w_2'],
                'prec': [0.45, 0.3],
                'wmean': [0.495, 0.66]
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
