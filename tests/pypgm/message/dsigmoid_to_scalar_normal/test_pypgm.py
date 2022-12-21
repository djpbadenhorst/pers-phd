from common import TEST

import unittest


class test_pypgm(unittest.TestCase):
    """Class containing tests for pypgm/pypgm.py"""

    def setUp(self):
        """Creates necessary variables"""

        from pypgm import PGM
        from pypgm.factor import FACTOR_TYPE, scalar_normal

        self.pgm = PGM('./out/')

        self.pgm.add_fnode('f(y,z)',
                           ftype=FACTOR_TYPE.DSIGMOID,
                           input_vars = ['y'],
                           output_vars = ['z'])

        self.pgm.add_fnode('f(y)',
                           ftype=FACTOR_TYPE.SCALAR_NORMAL,
                           par_form=scalar_normal.PARAMETER_FORM.STANDARD,
                           vars=['y'],
                           mean=0,
                           var=0.1)

        
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
        
        from pypgm.factor import FACTOR_TYPE, scalar_normal

        self.pgm.connect_fnodes('f(y,z)', 'f(y)', ['y'])

        message_parameters = self.pgm.get_message_parameters(
            'f(y,z)', 'f(y)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 0,
            'prec': 10,
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
        
        from pypgm.factor import FACTOR_TYPE, scalar_normal

        self.pgm.connect_fnodes('f(y,z)', 'f(y)', ['y'])

        belief_before = self.pgm.get_belief_parameters('f(y)')
        self.pgm.update_belief('f(y)')
        belief_after = self.pgm.get_belief_parameters('f(y)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("BELIEF BEFORE", 3)
        TEST.LOG(SDict(**belief_before), 3)
        TEST.LOG("BELIEF AFTER", 3)
        TEST.LOG(SDict(**belief_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 0,
            'prec': 10,
        }
        TEST.EQUALS(
            self, belief_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 0,
            'prec': 20,
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
        
        from pypgm.factor import FACTOR_TYPE, vector_normal, scalar_normal

        self.pgm.connect_fnodes('f(y,z)', 'f(y)', ['y'])
        
        message_before = self.pgm.get_message_parameters('f(y,z)','f(y)')

        from_belief = self.pgm.get_belief_parameters('f(y,z)')
        from_belief['input'] = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 2,
            'prec': 10,
        }
        from_belief['output'] = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['z'],
            'wmean': 0.1,
            'prec': 10,
        }
        self.pgm._set_value('belief_f(y,z)', from_belief)

        self.pgm.update_message('f(y,z)', 'f(y)')
        message_after = self.pgm.get_message_parameters('f(y,z)', 'f(y)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MESSAGE BEFORE", 3)
        TEST.LOG(SDict(**message_before), 3)
        TEST.LOG("MESSAGE AFTER", 3)
        TEST.LOG(SDict(**message_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 0,
            'prec': 10,
        }
        TEST.EQUALS(
            self, message_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.SCALAR_NORMAL,
            'par_form': scalar_normal.PARAMETER_FORM.CANONICAL,
            'vars': ['y'],
            'wmean': 1.813084,
            'prec': 0.025197,
        }
        TEST.EQUALS(
            self, message_after, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
