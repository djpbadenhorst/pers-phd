from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/vector_normal_to_scalar_normal/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_VECTOR_NORMAL_TO_SCALAR_NORMAL_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal, vector_normal
        
        from pypgm.message import vector_normal_to_scalar_normal

        vector_normal_from = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['x','y'],
            mean=[1,2],
            cov=[2,3])
        vector_normal_from.init_parameters()
        
        scalar_normal_to = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=3,
            var=4)
        scalar_normal_to.init_parameters()

        vector_normal_inc = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.COMMON,
            vars=['x'],
            mean=[4],
            cov=5)
        vector_normal_inc.init_parameters()
        
        scalar_normal_out = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=6,
            var=7)
        scalar_normal_out.init_parameters()

        message_parameters = vector_normal_to_scalar_normal.est_message(
            vector_normal_from.parameters, scalar_normal_to.parameters,
            vector_normal_inc.parameters, scalar_normal_out.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['x'],
            'wmean': -0.3,
            'prec': 0.3,
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
