from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/dlinear_to_gamma/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_DLINEAR_TO_GAMMA_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal, vector_normal
        
        from pypgm.message import dlinear_to_gamma

        dlinear_from = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars = ['x_1','x_2'],
            input = {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['x_1','x_2'],
                'wmean': [1,2],
                'prec': [10,20],
            },
            output_vars = ['y'],
            output = {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 1,
                'prec': 10,
            },
            weight_vars = ['w_1','w_2'],
            weight = {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'vars': ['w_1','w_2'],
                'wmean': [0,0],
                'prec': [0.1,0.2],
            },
            modval_vars = ['p'],
            modval = {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 1+1e-3,
                'scale': 1,
            })
        dlinear_from.init_parameters()

        gamma_to = Factor(
            ftype=FACTOR_TYPE.GAMMA,
            vars=['p'],
            shape=0,
            scale=1)
        gamma_to.init_parameters()

        dlinear_inc = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            modval_vars=['p'],
            modval={
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'vars': ['p'],
                'shape': 1,
                'scale': 10,
            })
        dlinear_inc.init_parameters()
        
        gamma_out = Factor(
            ftype=FACTOR_TYPE.GAMMA,
            vars=['p'],
            shape=0,
            scale=1)
        gamma_out.init_parameters()

        message_parameters = dlinear_to_gamma.est_message(
            dlinear_from.parameters, gamma_to.parameters,
            dlinear_inc.parameters, gamma_out.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.GAMMA,
            'vars': ['p'],
            'shape': 1.501,
            'scale': 0.604229,
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
