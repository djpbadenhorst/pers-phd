from common import TEST

import unittest


class test_init_message(unittest.TestCase):
    """Class containing tests for pypgm/message/scalar_normal_to_scalar_normal/init_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of init_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_SCALAR_NORMAL_TO_SCALAR_NORMAL_INIT_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal
        
        from pypgm.message import scalar_normal_to_scalar_normal

        scalar_normal_from = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=1,
            var=2)
        scalar_normal_from.init_parameters()
        
        scalar_normal_to = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=3,
            var=4)
        scalar_normal_to.init_parameters()

        message_parameters = scalar_normal_to_scalar_normal.init_message(
            scalar_normal_from.parameters, scalar_normal_to.parameters, ['x'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': 0.5,
            'prec': 0.5,
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
