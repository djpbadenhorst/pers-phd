from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/scalar_normal_to_dsigmoid/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_SCALAR_NORMAL_TO_DSIGMOID_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal
        
        from pypgm.message import scalar_normal_to_dsigmoid

        scalar_normal_from_input = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=1,
            var=2)
        scalar_normal_from_input.init_parameters()

        scalar_normal_from_output = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['z'],
            mean=3,
            var=4)
        scalar_normal_from_output.init_parameters()
        
        dsigmoid_to = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            input_vars=['y'],
            output_vars=['z'])
        dsigmoid_to.init_parameters()

        scalar_normal_inc_input = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=0.1,
            var=20)
        scalar_normal_inc_input.init_parameters()

        scalar_normal_inc_output = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['z'],
            mean=0.1,
            var=20)
        scalar_normal_inc_output.init_parameters()
        
        dsigmoid_out_input = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            input_vars=['y'])
        dsigmoid_out_input.init_parameters()

        dsigmoid_out_output = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            output_vars=['z'])
        dsigmoid_out_output.init_parameters()

        message_parameters_input = scalar_normal_to_dsigmoid.est_message(
            scalar_normal_from_input.parameters, dsigmoid_to.parameters,
            scalar_normal_inc_input.parameters, dsigmoid_out_input.parameters)

        message_parameters_output = scalar_normal_to_dsigmoid.est_message(
            scalar_normal_from_output.parameters, dsigmoid_to.parameters,
            scalar_normal_inc_output.parameters, dsigmoid_out_output.parameters)

        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE - INPUT", 3)
        TEST.LOG(SDict(**message_parameters_input), 3)
        TEST.LOG("EST MESSAGE - OUTPUT", 3)
        TEST.LOG(SDict(**message_parameters_output), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DSIGMOID,
            'input_vars': ['y'],
            'input': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 0.495,
                'prec': 0.45,
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters_input,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.DSIGMOID,
            'output_vars': ['z'],
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['z'],
                'wmean': 0.745,
                'prec': 0.2,
            }
        }
        TEST.EQUALS(
            self, tmp, message_parameters_output,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
