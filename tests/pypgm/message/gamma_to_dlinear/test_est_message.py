from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/gamma_to_dlinear/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_GAMMA_TO_DLINEAR_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor
        
        from pypgm.message import gamma_to_dlinear

        gamma_from = Factor(
            ftype=FACTOR_TYPE.GAMMA,
            vars=['p'],
            shape=1+1e-3,
            scale=2)
        gamma_from.init_parameters()
        
        dlinear_to = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars=['x_1','x_2'],
            output_vars=['y'],
            weight_vars=['w_1','w_2'],
            modval_vars=['p'])
        dlinear_to.init_parameters()

        gamma_inc = Factor(
            ftype=FACTOR_TYPE.GAMMA,
            vars=['p'],
            shape=0.1,
            scale=20)
        gamma_inc.init_parameters()
        
        dlinear_out = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            modval_vars=['p'])
        dlinear_out.init_parameters()

        message_parameters = gamma_to_dlinear.est_message(
            gamma_from.parameters, dlinear_to.parameters,
            gamma_inc.parameters, dlinear_out.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 1.901,
                'scale': 2.222222,
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
