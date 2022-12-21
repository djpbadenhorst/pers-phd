from common import TEST

import unittest


class test_est_message(unittest.TestCase):
    """Class containing tests for pypgm/message/scalar_normal_to_vector_normal/est_message.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of est_message
        """

        TEST.LOG("START - PYPGM_MESSAGE_SCALAR_NORMAL_TO_VECTOR_NORMAL_EST_MESSAGE_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, Factor, scalar_normal, vector_normal
        
        from pypgm.message import scalar_normal_to_vector_normal

        scalar_normal_from = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=1,
            var=2)
        scalar_normal_from.init_parameters()
        
        vector_normal_to = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['x','y'],
            mean=[3,4],
            cov=[4,5])
        vector_normal_to.init_parameters()

        scalar_normal_inc = Factor(
            ftype=FACTOR_TYPE.SCALAR_NORMAL,
            par_form=scalar_normal.PARAMETER_FORM.STANDARD,
            vars=['x'],
            mean=5,
            var=6)
        scalar_normal_inc.init_parameters()
        
        vector_normal_out = Factor(
            ftype=FACTOR_TYPE.VECTOR_NORMAL,
            par_form=vector_normal.PARAMETER_FORM.STANDARD,
            cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
            vars=['x'],
            mean=7,
            cov=8)
        vector_normal_out.init_parameters()

        message_parameters = scalar_normal_to_vector_normal.est_message(
            scalar_normal_from.parameters, vector_normal_to.parameters,
            scalar_normal_inc.parameters, vector_normal_out.parameters)
        
        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("EST MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.COMMON,
            'vars': ['x'],
            'wmean': [-1./3],
            'prec': 1./3,
        }
        TEST.EQUALS(
            self, tmp, message_parameters,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
