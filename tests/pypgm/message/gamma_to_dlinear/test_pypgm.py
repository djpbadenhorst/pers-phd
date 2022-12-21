from common import TEST

import unittest


class test_pypgm(unittest.TestCase):
    """Class containing tests for pypgm/pypgm.py"""

    def setUp(self):
        """Creates necessary variables"""

        from pypgm import PGM
        from pypgm.factor import FACTOR_TYPE

        self.pgm = PGM('./out/')

        self.pgm.add_fnode('f(p)',
                           ftype=FACTOR_TYPE.GAMMA,
                           vars=['p'],
                           shape=1+1e-3,
                           scale=1)
        
        self.pgm.add_fnode('f(x,y,w,p)',
                           ftype=FACTOR_TYPE.DLINEAR,
                           input_vars = ['x_1','x_2'],
                           output_vars = ['y'],
                           weight_vars = ['w_1','w_2'],
                           modval_vars = ['p'])
        
        
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

        self.pgm.connect_fnodes('f(p)', 'f(x,y,w,p)', ['p'])

        message_parameters = self.pgm.get_message_parameters(
            'f(p)', 'f(x,y,w,p)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("INIT MESSAGE", 3)
        TEST.LOG(SDict(**message_parameters), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 1+1e-3,
                'scale': 1,
            }
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

        self.pgm.connect_fnodes('f(p)', 'f(x,y,w,p)', ['p'])

        belief_before = self.pgm.get_belief_parameters('f(x,y,w,p)')
        self.pgm.update_belief('f(x,y,w,p)')
        belief_after = self.pgm.get_belief_parameters('f(x,y,w,p)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("BELIEF BEFORE", 3)
        TEST.LOG(SDict(**belief_before), 3)
        TEST.LOG("BELIEF AFTER", 3)
        TEST.LOG(SDict(**belief_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'input_vars': ['x_1','x_2'],
            'output_vars': ['y'],
            'weight_vars': ['w_1','w_2'],
            'modval_vars': ['p'],
        }
        TEST.EQUALS(
            self, belief_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'input_vars': ['x_1','x_2'],
            'output_vars': ['y'],
            'weight_vars': ['w_1','w_2'],
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 1+1e-3,
                'scale': 1,
            },
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
        
        from pypgm.factor import FACTOR_TYPE, scalar_normal

        self.pgm.connect_fnodes('f(p)', 'f(x,y,w,p)', ['p'])

        message_before = self.pgm.get_message_parameters('f(p)','f(x,y,w,p)')

        inc_message = self.pgm.get_message_parameters('f(x,y,w,p)','f(p)')
        inc_message['shape'] = 2
        inc_message['scale'] = 2
        self.pgm._set_value('f(x,y,w,p)->f(p)', inc_message)
        
        self.pgm.update_message('f(p)', 'f(x,y,w,p)')
        message_after = self.pgm.get_message_parameters('f(p)', 'f(x,y,w,p)')

        TEST.LOG("INFO LOGGING", 3)
        TEST.LOG("MESSAGE BEFORE", 3)
        TEST.LOG(SDict(**message_before), 3)
        TEST.LOG("MESSAGE AFTER", 3)
        TEST.LOG(SDict(**message_after), 3)

        TEST.LOG("START ASSERTS", 3)
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 1.001,
                'scale': 1,
            }
        }
        TEST.EQUALS(
            self, message_before, tmp,
            "Error 01")
        tmp = {
            'ftype': FACTOR_TYPE.DLINEAR,
            'modval_vars': ['p'],
            'modval': {
                'ftype': FACTOR_TYPE.GAMMA,
                'vars': ['p'],
                'shape': 0.001,
                'scale': 2,
            }
        }
        TEST.EQUALS(
            self, message_after, tmp,
            "Error 02")

        TEST.LOG("TEST COMPLETE", 1)
