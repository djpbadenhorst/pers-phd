from common import TEST

import unittest


class test_init_message(unittest.TestCase):
    """Class containing tests for pypgm/message/scalar_normal_to_dlinear/init_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of init_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_SCALAR_NORMAL_TO_DLINEAR_INIT_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal
        
        from pypgm.message import scalar_normal_to_dlinear

        scalar_normal_from = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=1,
            var=2)
        scalar_normal_from.init_parameters()
        
        dlinear_to = Factor(
            ftype=FACTOR_TYPE.DLINEAR,
            input_vars = ['x_1','x_2'],
            output_vars = ['y'],
            weight_vars = ['w_1','w_2'],
            modval_vars = ['p'])
        dlinear_to.init_parameters()

        message_parameters = scalar_normal_to_dlinear.init_message(
            scalar_normal_from.parameters, dlinear_to.parameters, ['y'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'output_vars': ['y'],
            'output':{
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 0.5,
                'prec': 0.5,
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
