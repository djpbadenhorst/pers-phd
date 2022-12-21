from common import TEST

import unittest


class test_pdf(unittest.TestCase):
    """Class containing tests for pypgm/factor/dlinear/pdf.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of calculate_pdf
            input: vector_normal
            output: scalar_normal
            weight: vector_normal
            modval: gamma
        """

        TEST.LOG("START - PYPGM_FACTOR_DLINEAR_PDF_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        
        from utils import SDict
        
        from pypgm.factor import FACTOR_TYPE, dlinear, vector_normal, scalar_normal

        parameter_set = {
            'input': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'wmean': [1, 2],
                'prec': [10, 20],
            },
            'output': {
                'ftype': FACTOR_TYPE.SCALAR_NORMAL,
                'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
                'wmean': 5,
                'prec': 10,
            },
            'weight': {
                'ftype': FACTOR_TYPE.VECTOR_NORMAL,
                'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
                'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
                'wmean': [0, 0],
                'prec': [0.1, 0.1],
            },
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'shape': 5,
                'scale': 6,
            }
        }

        i_eval_values = np.linspace(-1,1,10)
        o_eval_values = np.linspace(-5,5,10)
        w_eval_values = np.linspace(-5,5,10)
        m_eval_values = np.linspace(0,100,20)

        input_pdf = dlinear.calculate_pdf(
            i_eval_values, o_eval_values, w_eval_values, m_eval_values,
            'input', **parameter_set)[-1]
        output_pdf = dlinear.calculate_pdf(
            i_eval_values, o_eval_values, w_eval_values, m_eval_values,
            'output', **parameter_set)[-1]
        weight_pdf = dlinear.calculate_pdf(
            i_eval_values, o_eval_values, w_eval_values, m_eval_values,
            'weight', **parameter_set)[-1]
        modval_pdf = dlinear.calculate_pdf(
            i_eval_values, o_eval_values, w_eval_values, m_eval_values,
            'modval', **parameter_set)[-1]

        input_mean = [
            np.sum(i_eval_values*np.sum(input_pdf,0)/np.sum(input_pdf)),
            np.sum(i_eval_values*np.sum(input_pdf,1)/np.sum(input_pdf))
        ]
        output_mean = np.sum(o_eval_values*output_pdf/np.sum(output_pdf))
        weight_mean = [
            np.sum(w_eval_values*np.sum(weight_pdf,0)/np.sum(weight_pdf)),
            np.sum(w_eval_values*np.sum(weight_pdf,1)/np.sum(weight_pdf))
        ]
        modval_mean = np.sum(m_eval_values*modval_pdf/np.sum(modval_pdf))

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INPUT MEAN", 3)
        TEST.LOG(input_mean, 3)
        TEST.LOG("OUTPUT MEAN", 3)
        TEST.LOG(output_mean, 3)
        TEST.LOG("WEIGHTS MEAN", 3)
        TEST.LOG(weight_mean, 3)
        TEST.LOG("MODVAL MEAN", 3)
        TEST.LOG(modval_mean, 3)
        
        TEST.LOG("START ASSERTS", 3)
        TEST.EQUALS(
            self, input_mean, [0.079388, 0.069730],
            "Error 01")
        TEST.EQUALS(
            self, output_mean, 0.551592,
            "Error 02")
        TEST.EQUALS(
            self, weight_mean, [0.291865, 0.205697],
            "Error 03")
        TEST.EQUALS(
            self, modval_mean, 30.043351,
            "Error 04")

        TEST.LOG("TEST COMPLETE", 1)
