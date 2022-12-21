from common import TEST

import unittest


class test_pdf(unittest.TestCase):
    """Class containing tests for pypgm/factor/gamma/pdf.py"""

    def test_01(self):
        """Tests the following :
        1) Functionality of calculate_pdf
        """

        TEST.LOG("START - PYPGM_FACTOR_GAMMA_PDF_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        import numpy as np
        
        from pypgm.factor import gamma

        eval_values = np.linspace(0,100,1e5)
        
        pdf = gamma.calculate_pdf(eval_values, shape = 3, scale = 1./2)

        pdf_mean = np.sum(eval_values*pdf/np.sum(pdf))

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("PDF MEAN", 3)
        TEST.LOG(pdf_mean, 3)

        TEST.LOG("START ASSERTS", 3)
        TEST.EQUALS(
            self, pdf_mean, 1.5,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)
 
