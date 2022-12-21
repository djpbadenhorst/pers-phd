from common import TEST

import unittest


class test_init_message(unittest.TestCase):
    """Class containing tests for pypgm/message/dsigmoid_to_scalar_normal/init_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of init_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_DSIGMOID_TO_SCALAR_NORMAL_INIT_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict

        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal
        
        from pypgm.message import dsigmoid_to_scalar_normal

        dsigmoid_from = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            input_vars = ['y'],
            output_vars = ['z'])
        dsigmoid_from.init_parameters()

        scalar_normal_to_input = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=1,
            var=2)
        scalar_normal_to_input.init_parameters()
        
        scalar_normal_to_output = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['z'],
            mean=3,
            var=4)
        scalar_normal_to_output.init_parameters()

        message_parameters_input = dsigmoid_to_scalar_normal.init_message(
            dsigmoid_from.parameters, scalar_normal_to_input.parameters, ['y'])
        message_parameters_output = dsigmoid_to_scalar_normal.init_message(
            dsigmoid_from.parameters, scalar_normal_to_output.parameters, ['z'])

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE - INPUT", 3)
        TEST.LOG(SDict(**message_parameters_input), 3)
        TEST.LOG("INIT MESSAGE - OUTPUT", 3)
        TEST.LOG(SDict(**message_parameters_output), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 0.5,
            'prec': 0.5,
        }
        TEST.EQUALS(
            self, tmp, message_parameters_input,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['z'],
            'wmean': 0.75,
            'prec': 0.25,
        }
        TEST.EQUALS(
            self, tmp, message_parameters_output,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
