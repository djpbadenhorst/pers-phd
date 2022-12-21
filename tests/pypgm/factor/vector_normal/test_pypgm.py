from common import TEST

import unittest


class test_pypgm(unittest.TestCase):
    """Class containing tests for pypgm/pypgm.py"""

    def setUp(self):
        """Creates necessary variables"""

        from pypgm import PGM

        self.pgm = PGM('./out/')

        
    def tearDown(self):
        """Destroys necessary variables"""
        
        import shutil
        
        shutil.rmtree('./out/')

        
    def test_01(self):
        """Tests the following :
        1) Functionality of PGM.add_fnode
        2) Functionality of PGM.get_prior
        3) Functionality of PGM.get_belief
        """

        TEST.LOG("START - PYPGM_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils import SDict
        
        from pypgm.factor import FACTOR_TYPE, vector_normal

        self.pgm.add_fnode('f(x,y)',
                           ftype=FACTOR_TYPE.VECTOR_NORMAL,
                           par_form=vector_normal.PARAMETER_FORM.STANDARD,
                           cov_form=vector_normal.COVARIANCE_FORM.FULL,
                           vars=['x', 'y'],
                           mean=[1, 2],
                           cov=[[1, 0], [0, 2]])

        fnode_prior = self.pgm.get_prior_parameters('f(x,y)')
        fnode_belief = self.pgm.get_belief_parameters('f(x,y)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("FNODE PRIOR 1", 3)
        TEST.LOG(SDict(**fnode_prior), 3)
        TEST.LOG("FNODE BELIEF 1", 3)
        TEST.LOG(SDict(**fnode_belief), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.FULL,
            'vars': ['x', 'y'],
            'wmean': [1, 1],
            'prec': [[1, 0], [0, 0.5]],
        }
        TEST.EQUALS(
            self, fnode_prior, tmp,
            "Error 01")
        TEST.EQUALS(
            self, fnode_belief, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)

