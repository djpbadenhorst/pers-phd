from common import TEST

import unittest


class test_pypgm(unittest.TestCase):
    """Class containing tests for pypgm/pypgm.py"""

    def setUp(self):
        """Creates necessary variables"""

        from pypgm import PGM
        from pypgm.factor import FACTOR_TYPE, vector_normal

        self.pgm = PGM('./out/')

        self.pgm.add_fnode('f(x,y)',
                           ftype=FACTOR_TYPE.VECTOR_NORMAL,
                           par_form=vector_normal.PARAMETER_FORM.STANDARD,
                           cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                           vars=['x','y'],
                           mean=[1,2],
                           cov=[2,3])

        self.pgm.add_fnode('f(x,z)',
                           ftype=FACTOR_TYPE.VECTOR_NORMAL,
                           par_form=vector_normal.PARAMETER_FORM.STANDARD,
                           cov_form=vector_normal.COVARIANCE_FORM.DIAGONAL,
                           vars=['x','z'],
                           mean=[3,4],
                           cov=[4,5])

        
    def tearDown(self):
        """Destroys necessary variables"""

        import shutil

        shutil.rmtree('./out/')

        
    def test_01(self):
        """Tests the following :
        1) Functionality of PGM.connect_fnodes
        2) Functionality of PGM.get_message
        """

        TEST.LOG("START - PYPGM_01", 1)
        TEST.LOG(self.test_01.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, vector_normal

        self.pgm.connect_fnodes('f(x,y)', 'f(x,z)', ['x'])

        message_parameters = self.pgm.get_message_parameters(
            'f(x,y)','f(x,z)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x'],
            'wmean': [0.5],
            'prec': [0.5],
        }
        TEST.EQUALS(
            self, message_parameters, tmp,
            "Error 01")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_02(self):
        """Tests the following :
        1) Functionality of PGM.update_belief
        """

        TEST.LOG("START - PYPGM_02", 1)
        TEST.LOG(self.test_02.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, vector_normal

        self.pgm.connect_fnodes('f(x,y)', 'f(x,z)', ['x'])

        belief_before = self.pgm.get_belief_parameters('f(x,z)')
        self.pgm.update_belief('f(x,z)')
        belief_after = self.pgm.get_belief_parameters('f(x,z)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("BELIEF BEFORE", 3)
        TEST.LOG(SDict(**belief_before), 3)
        TEST.LOG("BELIEF AFTER", 3)
        TEST.LOG(SDict(**belief_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x','z'],
            'wmean': [0.75,0.8],
            'prec': [0.25,0.2],
        }
        TEST.EQUALS(
            self, belief_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x','z'],
            'wmean': [1.25,0.8],
            'prec': [0.75,0.2],
        }
        TEST.EQUALS(
            self, belief_after, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)

        
    def test_03(self):
        """Tests the following :
        1) Functionality of PGM.update_message
        """

        TEST.LOG("START - PYPGM_03", 1)
        TEST.LOG(self.test_03.__doc__, 2)

        from utils.sdict import SDict
        
        from pypgm.factor import FACTOR_TYPE, vector_normal

        self.pgm.connect_fnodes('f(x,y)', 'f(x,z)', ['x'])

        message_before = self.pgm.get_message_parameters('f(x,y)','f(x,z)')

        inc_message = self.pgm.get_message_parameters('f(x,y)','f(x,z)')
        inc_message['wmean'] = [0.1]
        inc_message['prec'] = [0.2]
        self.pgm._set_value('f(x,z)->f(x,y)', inc_message)

        self.pgm.update_message('f(x,y)', 'f(x,z)')
        message_after = self.pgm.get_message_parameters('f(x,y)','f(x,z)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MESSAGE BEFORE", 3)
        TEST.LOG(SDict(**message_before), 3)
        TEST.LOG("MESSAGE AFTER", 3)
        TEST.LOG(SDict(**message_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x'],
            'wmean': [0.5],
            'prec': [0.5],
        }
        TEST.EQUALS(
            self, message_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.VECTOR_NORMAL,
            'par_form': vector_normal.PARAMETER_FORM.CANONICAL,
            'cov_form': vector_normal.COVARIANCE_FORM.DIAGONAL,
            'vars': ['x'],
            'wmean': [0.4],
            'prec': [0.3],
        }
        TEST.EQUALS(
            self, message_after, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
