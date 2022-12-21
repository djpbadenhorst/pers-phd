from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/dsigmoid_to_scalar_normal/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_DSIGMOID_TO_SCALAR_NORMAL_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        np.random.seed(0)
        
        from copy import deepcopy
        
        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal
        
        from pypgm.message import dsigmoid_to_scalar_normal

        dsigmoid_from = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            input_vars = ['y'],
            input = {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 1,
                'prec': 10,
            },
            output_vars = ['z'],
            output = {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['z'],
                'wmean': 1,
                'prec': 10,
            })
        dsigmoid_from.init_parameters()

        scalar_normal_to_input = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=0,
            var=1)
        scalar_normal_to_input.init_parameters()

        dsigmoid_inc_input = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            input_vars=['y'],
            input={
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 1,
                'prec': 10,
            })
        dsigmoid_inc_input.init_parameters()
        
        scalar_normal_out_input = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['y'],
            mean=0,
            var=1)
        scalar_normal_out_input.init_parameters()

        scalar_normal_to_output = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['z'],
            mean=0,
            var=1)
        scalar_normal_to_output.init_parameters()

        dsigmoid_inc_output = Factor(
            ftype=FACTOR_TYPE.DSIGMOID,
            output_vars=['z'],
            output={
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'vars': ['y'],
                'wmean': 1,
                'prec': 10,
            })
        dsigmoid_inc_output.init_parameters()
        
        scalar_normal_out_output = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['z'],
            mean=0,
            var=1)
        scalar_normal_out_output.init_parameters()

        message_parameters_input = dsigmoid_to_scalar_normal.est_message(
            deepcopy(dsigmoid_from.parameters), scalar_normal_to_input.parameters,
            dsigmoid_inc_input.parameters, scalar_normal_out_input.parameters)
        message_parameters_output = dsigmoid_to_scalar_normal.est_message(
            deepcopy(dsigmoid_from.parameters), scalar_normal_to_output.parameters,
            dsigmoid_inc_output.parameters, scalar_normal_out_output.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE - INPUT", 3)
        TEST.LOG(SDict(**message_parameters_input), 3)
        TEST.LOG("EST MESSAGE - OUTPUT", 3)
        TEST.LOG(SDict(**message_parameters_output), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': -0.153261,
            'prec': 0.0255581,
        }
        TEST.EQUALS(
            self, tmp, message_parameters_input,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['z'],
            'wmean': 89.195899,
            'prec': 168.793695,
        }
        TEST.EQUALS(
            self, tmp, message_parameters_output,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
