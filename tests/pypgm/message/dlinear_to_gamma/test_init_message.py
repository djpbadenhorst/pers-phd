from common import TEST

import unittest


class test_init_message(unittest.TestCase):
    """Class containing tests for pypgm/message/dlinear_to_gamma/init_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of init_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_DLINEAR_TO_GAMMA_INIT_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict

        from pypgm.factor import FACTOR_TYPE, Factor
        
        from pypgm.message import dlinear_to_gamma

        dlinear_from = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars = ['x_1','x_2'],
            output_vars = ['y'],
            weight_vars = ['w_1','w_2'],
            modval_vars = ['p'])
        dlinear_from.init_parameters()
        
        gamma_to = Factor(
            ftype=FACTOR_TYPE.GAMMA,
            vars=['p'],
            shape=1,
            scale=2)
        gamma_to.init_parameters()

        message_parameters = dlinear_to_gamma.init_message(
            dlinear_from.parameters, gamma_to.parameters, ['p'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.GAMMA,
            'vars': ['p'],
            'shape': 1,
            'scale': 2,
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
