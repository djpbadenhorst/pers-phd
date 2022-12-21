from common import TEST

import unittest


class test_pdf(unittest.TestCase):
    """Class containing tests for pypgm/factor/vector_normal/pdf.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of calculate_pdf
        """

        TEST.LOG("START - PYPGM_FACTOR_VECTOR_NORMAL_PDF_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        
        from pypgm.factor import vector_normal

        standard_parameters = {
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'mean': [5,5],
            'cov': [[1, -0.1], [-0.1, 2]]
        }
        canonical_parameters = vector_normal.standard_to_canonical(standard_parameters)

        eval_values = np.linspace(0,10,100)
        
        pdf = vector_normal.calculate_pdf(eval_values, **canonical_parameters)

        pdf_mean = [
            np.sum(eval_values*np.sum(pdf,0)/np.sum(pdf)),
            np.sum(eval_values*np.sum(pdf,1)/np.sum(pdf))
        ]

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PDF MEAN", 3)
        TEST.LOG(pdf_mean, 3)
        
        TEST.LOG("START ASSERTS", 3)
        TEST.EQUALS(
            self, pdf_mean, [5,5],
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
