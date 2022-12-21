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
        
        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE

        self.pgm.add_fnode('f(y,z)',
                           ftype=FACTOR_TYPE.DSIGMOID)

        fnode_prior = self.pgm.get_prior_parameters('f(y,z)')
        fnode_belief = self.pgm.get_belief_parameters('f(y,z)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("FNODE PRIOR 1", 3)
        TEST.LOG(SDict(**fnode_prior), 3)
        TEST.LOG("FNODE BELIEF 1", 3)
        TEST.LOG(SDict(**fnode_belief), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DSIGMOID,
        }
        TEST.EQUALS(
            self, fnode_prior, tmp,
            "Error 01")
        TEST.EQUALS(
            self, fnode_belief, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)

